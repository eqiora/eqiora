//! Realization-owned polynomial quadrature expands retained Observable dependencies.
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, DynQuantity, Id, RawId, entity::kinds};
use eqiora_meshing::QuadratureRule;
use eqiora_schema::kernel::{
    ExprDag, ExprDagBuilder, ExprId, ExprNode, KernelNode, ObservableReduction, SymbolRef,
};
use eqiora_sem::KernelProgram;

use super::invalid;
use crate::factor_measure::{axes, mapped_sample};
mod candidates;
mod degree;
#[cfg(test)]
mod tests;
pub(super) use candidates::candidates;

type Point = BTreeMap<(RawId, usize), DynQuantity>;
type Degree = BTreeMap<(RawId, usize), u16>;

pub(super) fn expand(kernel: &KernelProgram, expression: &ExprDag) -> Result<ExprDag, Diagnostic> {
    if !expression
        .nodes()
        .iter()
        .any(|node| matches!(node, ExprNode::Symbol(SymbolRef::Observable(_))))
    {
        return Ok(expression.clone());
    }
    let mut context = Context {
        kernel,
        builder: ExprDagBuilder::new(),
        remaining: 65_536,
        degrees: BTreeMap::new(),
    };
    let roots = context.expression(expression, &Point::new(), 0)?;
    context.builder.finish(roots)
}

struct Context<'a> {
    kernel: &'a KernelProgram,
    builder: ExprDagBuilder,
    remaining: usize,
    degrees: BTreeMap<RawId, Degree>,
}

impl Context<'_> {
    fn charge(&mut self, count: usize) -> Result<(), Diagnostic> {
        self.remaining = self.remaining.checked_sub(count).ok_or_else(|| {
            invalid("finite integral expansion exceeds 65536 expression operations")
        })?;
        Ok(())
    }

    fn expression(
        &mut self,
        expression: &ExprDag,
        point: &Point,
        depth: usize,
    ) -> Result<Vec<ExprId>, Diagnostic> {
        if depth > 32 || !expression.properties().is_empty() {
            return Err(invalid(
                "finite integral expansion requires bounded unannotated scalar expressions",
            ));
        }
        self.charge(expression.nodes().len())?;
        let mut values: Vec<ExprId> = Vec::with_capacity(expression.nodes().len());
        for node in expression.nodes() {
            let value = |id: &ExprId| values[id.index() as usize];
            let copied = match node {
                ExprNode::Constant(literal) if literal.real_scalar_value().is_some() => {
                    self.builder.constant(literal.clone())?
                }
                ExprNode::Symbol(SymbolRef::Observable(id)) => {
                    self.observable(*id, point, depth + 1)?
                }
                ExprNode::Symbol(SymbolRef::Coordinate { factor, axis, .. }) => {
                    let coordinate = point.get(&(factor.erase(), *axis)).ok_or_else(|| {
                        invalid("finite integral leaves an unbound output coordinate")
                    })?;
                    self.builder.constant(*coordinate)?
                }
                ExprNode::Symbol(symbol @ (SymbolRef::Field(_) | SymbolRef::Parameter(_))) => {
                    self.builder.symbol(*symbol)?
                }
                ExprNode::Neg(a) => self.builder.neg(value(a))?,
                ExprNode::Add(a, b) => self.builder.add(value(a), value(b))?,
                ExprNode::Sub(a, b) => self.builder.sub(value(a), value(b))?,
                ExprNode::Mul(a, b) => self.builder.mul(value(a), value(b))?,
                ExprNode::Div(a, b) => self.builder.div(value(a), value(b))?,
                ExprNode::PowI(a, power) => self.builder.powi(value(a), *power)?,
                ExprNode::UnaryMath(function, a) => self.builder.unary_math(*function, value(a))?,
                _ => {
                    return Err(invalid(
                        "finite Observable coupling requires real scalar arithmetic",
                    ));
                }
            };
            values.push(copied);
        }
        Ok(expression
            .roots()
            .iter()
            .map(|id| values[id.index() as usize])
            .collect())
    }

    fn observable(
        &mut self,
        id: Id<kinds::Observable>,
        point: &Point,
        depth: usize,
    ) -> Result<ExprId, Diagnostic> {
        if depth > 32 {
            return Err(invalid("finite integral dependency depth exceeds 32"));
        }
        let Some(KernelNode::Observable(definition)) = self.kernel.node(id.erase()) else {
            return Err(invalid("finite expression references a foreign Observable"));
        };
        if definition.reduction().limits().is_some() {
            return Err(invalid(
                "explicit integral limits require admitted numerical endpoint evaluation",
            ));
        }
        if definition.value_type().scalar_domain() != eqiora_core::ScalarDomain::Real
            || !definition.value_type().shape().is_scalar()
            || definition.value_type().array_rank() != 0
        {
            return Err(invalid(
                "finite Observable coupling requires real scalar outputs",
            ));
        }
        let definition = definition.clone();
        match definition.reduction() {
            ObservableReduction::Value => {
                Ok(self.expression(definition.expression(), point, depth)?[0])
            }
            ObservableReduction::SpatialIntegral {
                domain, measure, ..
            } => {
                let selected = axes(self.kernel, domain)?;
                let degree = self.degree_expression(definition.expression(), depth)?;
                let mut maximum = selected
                    .iter()
                    .map(|(axis, _)| degree.get(axis).copied().unwrap_or(0))
                    .max()
                    .ok_or_else(|| invalid("finite integral requires a nonempty measure"))?;
                if measure == eqiora_schema::kernel::ObservableMeasure::SphericalVolume {
                    maximum = maximum
                        .checked_add(2)
                        .ok_or_else(|| invalid("radial polynomial degree overflow"))?;
                }
                let points = (usize::from(maximum) + 2) / 2;
                let count = u32::try_from(selected.len())
                    .ok()
                    .and_then(|axes| points.checked_pow(axes))
                    .filter(|count| *count <= 4096)
                    .ok_or_else(|| invalid("finite integral quadrature exceeds 4096 points"))?;
                self.charge(
                    count
                        .checked_mul(3)
                        .ok_or_else(|| invalid("finite integral work overflow"))?,
                )?;
                let rule = QuadratureRule::tensor_product_gauss_legendre(selected.len(), points)?;
                let mut sum = None;
                for sample in rule.points() {
                    let (coordinates, weight) = mapped_sample(&selected, sample, measure)?;
                    let mut local = point.clone();
                    for ((axis, _), coordinate) in selected.iter().zip(coordinates) {
                        local.insert(*axis, coordinate);
                    }
                    let density = self.expression(definition.expression(), &local, depth)?[0];
                    let weight = self.builder.constant(weight)?;
                    let term = self.builder.mul(weight, density)?;
                    sum = Some(match sum {
                        None => term,
                        Some(previous) => self.builder.add(previous, term)?,
                    });
                }
                sum.ok_or_else(|| invalid("finite integral quadrature is empty"))
            }
        }
    }
}
