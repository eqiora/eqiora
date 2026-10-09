//! Analytic Cartesian polynomial jets use the existing ordered scalar calculus.
//! This evaluates explicit coordinate expressions, never reconstructs unknown Fields.
use super::*;
use eqiora_core::{DimExponents, ValueShape};
use eqiora_schema::kernel::pure_operator::{
    CalculusBuilder, CalculusNode, CalculusNodeId, ExactRational, MAX_FORMALS,
    PureOperatorDefinition, PureValueClass,
};
use eqiora_schema::kernel::typing::{RootContract, SpatialSupport, TypedResidual};
use eqiora_schema::kernel::{BoundarySide, DomainKind, ExprDagBuilder, KernelNode};
use std::collections::{BTreeSet, HashMap};

fn reject(message: impl std::fmt::Display) -> Diagnostic {
    Diagnostic::error(
        codes::NOT_IMPLEMENTED,
        format!("analytic spatial expression: {message}"),
    )
}
fn class(dimension: DimExponents) -> Result<PureValueClass, Diagnostic> {
    PureValueClass::invariant_scalar()
        .with_dimension(dimension)
        .with_scalar_domain(ScalarDomain::Real)
        .map_err(reject)
}

pub(super) fn evaluate(
    program: &KernelProgram,
    owner: RawId,
    expression: &ExprDag,
    root: ExprId,
    point: &EvaluationPoint,
    work: &mut usize,
    resolve: &mut Resolver<'_>,
) -> Result<ValueLiteral, Diagnostic> {
    // An enclosing evaluate(...) has no support. Type the selected spatial
    // subexpression at its exact point while retaining the original node IDs.
    let selected = ExprDagBuilder::from_dag(expression).finish([root])?;
    let typed = program
        .type_derived_residual(
            selected,
            owner,
            Some(point.domain().erase()),
            RootContract::ComponentwiseResidual,
        )
        .map_err(|errors| errors.into_iter().next().expect("failed typing"))?;
    let result = typed
        .node_type(root)
        .ok_or_else(|| reject("untyped spatial root"))?;
    if result.value_type.scalar_domain() != ScalarDomain::Real {
        return Err(reject(
            "the analytic polynomial profile requires real components",
        ));
    }
    let point_support = program
        .spatial_support(point.domain())
        .ok_or_else(|| reject("point has no exact spatial support"))?;
    let parent = point_support.parent().unwrap_or(point_support.domain());
    if result.support.as_ref().is_none_or(|support| {
        support.domain() != point_support.domain() && support.domain() != parent
    }) {
        return Err(reject(
            "spatial expression belongs to a foreign point support",
        ));
    }
    let mut inputs = Vec::new();
    let mut coordinates = HashMap::new();
    for ((factor, axis), value) in point.coordinates() {
        coordinates.insert((factor, axis), inputs.len() as u16);
        inputs.push(ValueLiteral::try_from(value).map_err(reject)?);
    }
    let mut leaves = HashMap::new();
    let mut pending = vec![root];
    let mut seen = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = expression
            .node(id)
            .ok_or_else(|| reject("missing polynomial node"))?;
        let value = match node {
            ExprNode::Constant(value) => Some(value.clone()),
            ExprNode::Symbol(symbol @ SymbolRef::Parameter(_)) => {
                Some(resolve(EvaluationInput::Value(*symbol), Some(point))?)
            }
            ExprNode::Symbol(SymbolRef::Coordinate {
                support,
                factor,
                axis,
            }) => {
                if support.erase() != *parent && support != &point.domain() {
                    return Err(reject("coordinate belongs to a foreign Cartesian support"));
                }
                if !coordinates.contains_key(&(factor.erase(), *axis)) {
                    return Err(reject("point omits an exact coordinate"));
                }
                None
            }
            ExprNode::Neg(a)
            | ExprNode::Gradient(a)
            | ExprNode::Divergence(a)
            | ExprNode::Trace(a)
            | ExprNode::NormalComponent(a)
            | ExprNode::PowI(a, _) => {
                pending.push(*a);
                None
            }
            ExprNode::Add(a, b) | ExprNode::Sub(a, b) | ExprNode::Mul(a, b) => {
                pending.extend([*a, *b]);
                None
            }
            ExprNode::PureOperatorApplication(application) => {
                pending.extend(application.arguments());
                None
            }
            _ => {
                return Err(reject(
                    "only explicit coordinate polynomials are admitted; unknown Field derivatives require a reconstruction",
                ));
            }
        };
        if let Some(value) = value {
            if value.value_type() != &typed.node_type(id).expect("typed leaf").value_type {
                return Err(reject(
                    "bound Parameter differs from its exact declared type",
                ));
            }
            if value.value_type().scalar_domain() != ScalarDomain::Real {
                return Err(reject("polynomial leaves require real components"));
            }
            let count = value
                .value_type()
                .shape()
                .component_count()
                .ok_or_else(|| reject("component count overflows"))?;
            if inputs
                .len()
                .checked_add(count)
                .is_none_or(|count| count > MAX_FORMALS)
            {
                return Err(reject("polynomial inputs exceed the shared calculus bound"));
            }
            for (flat, (value, _)) in value
                .components()
                .ok_or_else(|| reject("nonnumeric polynomial leaf"))?
                .enumerate()
            {
                leaves.insert((id, flat), inputs.len() as u16);
                inputs.push(
                    ValueLiteral::try_from(DynQuantity::new(
                        value,
                        typed.node_type(id).expect("typed leaf").dimension(),
                    ))
                    .map_err(reject)?,
                );
            }
        }
    }
    let formals = inputs
        .iter()
        .map(|value| class(value.value_type().dimension()))
        .collect::<Result<Vec<_>, _>>()?;
    let count = result
        .shape()
        .component_count()
        .ok_or_else(component_budget_error)?;
    check_component_work(*work, count)?;
    *work += count;
    let mut values = Vec::with_capacity(count);
    for flat in 0..count {
        let mut calculus = Projection {
            program,
            typed: &typed,
            point,
            leaves: &leaves,
            coordinates: &coordinates,
            builder: CalculusBuilder::new(formals.clone(), class(result.dimension())?)
                .map_err(reject)?,
            cache: HashMap::new(),
            remaining: 1_000_000,
        };
        let coordinate = unflatten(result.shape(), flat);
        let root = calculus.component(root, &coordinate, 0)?;
        let definition = calculus.builder.finish(root).map_err(reject)?;
        let value =
            super::pure::evaluate(owner, &definition, &inputs.iter().collect::<Vec<_>>(), work)?;
        let scalar = value
            .real_scalar_value()
            .ok_or_else(|| reject("nonreal scalar projection"))?;
        values.push((scalar.value(), 0.));
    }
    ValueLiteral::new(result.value_type.clone(), values).map_err(reject)
}

fn unflatten(shape: &ValueShape, mut flat: usize) -> Vec<u32> {
    let mut result = vec![0; shape.rank()];
    for (axis, extent) in result.iter_mut().zip(shape.extents()).rev() {
        *axis = (flat % extent.get() as usize) as u32;
        flat /= extent.get() as usize;
    }
    result
}

struct Projection<'a> {
    program: &'a KernelProgram,
    typed: &'a TypedResidual<RawId>,
    point: &'a EvaluationPoint,
    leaves: &'a HashMap<(ExprId, usize), u16>,
    coordinates: &'a HashMap<(RawId, usize), u16>,
    builder: CalculusBuilder,
    cache: HashMap<(ExprId, Vec<u32>), CalculusNodeId>,
    remaining: usize,
}

impl Projection<'_> {
    fn push(&mut self, node: CalculusNode) -> Result<CalculusNodeId, Diagnostic> {
        self.builder.push(node).map_err(reject)
    }
    fn formal(&mut self, formal: u16) -> Result<CalculusNodeId, Diagnostic> {
        self.push(CalculusNode::FormalComponent {
            formal,
            axes: Box::new([]),
        })
    }
    fn partial(
        &mut self,
        value: CalculusNodeId,
        source: ExprId,
        axis: usize,
    ) -> Result<CalculusNodeId, Diagnostic> {
        let Some(SpatialSupport::Volume { domain, .. }) = self
            .typed
            .node_type(source)
            .and_then(|ty| ty.support.as_ref())
        else {
            return Err(reject("physical partial requires an exact volume"));
        };
        let formal = self
            .coordinates
            .get(&(*domain, axis))
            .ok_or_else(|| reject("gradient axis is absent from this point"))?;
        self.builder.partial(value, *formal).map_err(reject)
    }
    fn component(
        &mut self,
        id: ExprId,
        indices: &[u32],
        depth: usize,
    ) -> Result<CalculusNodeId, Diagnostic> {
        if depth > 128 {
            return Err(component_budget_error());
        }
        self.remaining = self
            .remaining
            .checked_sub(1)
            .ok_or_else(component_budget_error)?;
        let key = (id, indices.to_vec());
        if let Some(value) = self.cache.get(&key) {
            return Ok(*value);
        }
        let expression = self.typed.expression();
        let node = expression
            .node(id)
            .ok_or_else(|| reject("missing component node"))?;
        let ty = self
            .typed
            .node_type(id)
            .ok_or_else(|| reject("missing component type"))?;
        if ty.shape().rank() != indices.len()
            || ty
                .shape()
                .extents()
                .iter()
                .zip(indices)
                .any(|(n, i)| *i >= n.get())
        {
            return Err(reject("component differs from exact tensor shape"));
        }
        let result = match node {
            ExprNode::Constant(_) | ExprNode::Symbol(SymbolRef::Parameter(_)) => {
                let flat = ty
                    .shape()
                    .extents()
                    .iter()
                    .zip(indices)
                    .fold(0usize, |flat, (extent, index)| {
                        flat * extent.get() as usize + *index as usize
                    });
                self.formal(
                    *self
                        .leaves
                        .get(&(id, flat))
                        .ok_or_else(|| reject("missing leaf binding"))?,
                )?
            }
            ExprNode::Symbol(SymbolRef::Coordinate { factor, axis, .. }) => self.formal(
                *self
                    .coordinates
                    .get(&(factor.erase(), *axis))
                    .ok_or_else(|| reject("missing coordinate binding"))?,
            )?,
            ExprNode::Neg(value) => {
                let value = self.component(*value, indices, depth + 1)?;
                self.push(CalculusNode::Neg(value))?
            }
            ExprNode::Add(a, b) | ExprNode::Sub(a, b) | ExprNode::Mul(a, b) => {
                let operand = |id| {
                    if self
                        .typed
                        .node_type(id)
                        .expect("typed operand")
                        .shape()
                        .is_scalar()
                    {
                        &[][..]
                    } else {
                        indices
                    }
                };
                let (ai, bi) = (operand(*a), operand(*b));
                let left = self.component(*a, ai, depth + 1)?;
                let mut right = self.component(*b, bi, depth + 1)?;
                if matches!(node, ExprNode::Sub(..)) {
                    right = self.push(CalculusNode::Neg(right))?;
                }
                self.push(if matches!(node, ExprNode::Mul(..)) {
                    CalculusNode::Mul(left, right)
                } else {
                    CalculusNode::Add(left, right)
                })?
            }
            ExprNode::PowI(base, exponent) if *exponent >= 0 => {
                if *exponent as usize > self.remaining {
                    return Err(component_budget_error());
                }
                self.remaining -= *exponent as usize;
                let base = self.component(*base, &[], depth + 1)?;
                let mut result = self.push(CalculusNode::Rational {
                    value: ExactRational::integer(1),
                    dimension: DimExponents::DIMENSIONLESS,
                })?;
                for _ in 0..*exponent {
                    result = self.push(CalculusNode::Mul(result, base))?;
                }
                result
            }
            ExprNode::Gradient(value) => {
                let (axis, coordinate) = indices
                    .split_last()
                    .ok_or_else(|| reject("gradient lacks a derivative axis"))?;
                let component = self.component(*value, coordinate, depth + 1)?;
                self.partial(component, *value, *axis as usize)?
            }
            ExprNode::Divergence(value) => {
                let extent = self
                    .typed
                    .node_type(*value)
                    .expect("typed operand")
                    .shape()
                    .extents()
                    .last()
                    .expect("typed divergence")
                    .get();
                let mut sum = None;
                for axis in 0..extent {
                    let mut coordinate = indices.to_vec();
                    coordinate.push(axis);
                    let component = self.component(*value, &coordinate, depth + 1)?;
                    let partial = self.partial(component, *value, axis as usize)?;
                    sum = Some(match sum {
                        None => partial,
                        Some(sum) => self.push(CalculusNode::Add(sum, partial))?,
                    });
                }
                sum.ok_or_else(|| reject("empty divergence"))?
            }
            ExprNode::Trace(value) => self.component(*value, indices, depth + 1)?,
            ExprNode::NormalComponent(value) => {
                let Some(KernelNode::Domain(domain)) =
                    self.program.node(self.point.domain().erase())
                else {
                    return Err(reject("missing boundary"));
                };
                let DomainKind::CartesianBoundary { axis, side } = domain.kind() else {
                    return Err(reject("normal requires an exact Cartesian boundary"));
                };
                let mut coordinate = indices.to_vec();
                coordinate.push(*axis as u32);
                let value = self.component(*value, &coordinate, depth + 1)?;
                if *side == BoundarySide::Lower {
                    self.push(CalculusNode::Neg(value))?
                } else {
                    value
                }
            }
            ExprNode::PureOperatorApplication(application) => {
                let definition = expression
                    .definition(application.definition())
                    .ok_or_else(|| reject("missing pure definition"))?;
                self.pure(definition, application.arguments(), indices, depth + 1)?
            }
            _ => return Err(reject("unsupported analytic polynomial operation")),
        };
        self.cache.insert(key, result);
        Ok(result)
    }
    fn pure(
        &mut self,
        definition: &PureOperatorDefinition,
        arguments: &[ExprId],
        indices: &[u32],
        depth: usize,
    ) -> Result<CalculusNodeId, Diagnostic> {
        self.remaining = self
            .remaining
            .checked_sub(definition.nodes().len())
            .ok_or_else(component_budget_error)?;
        let mut mapped = Vec::new();
        for node in definition.nodes() {
            let get = |id: CalculusNodeId| mapped[id.index() as usize];
            let value = match node {
                CalculusNode::FormalComponent { formal, axes } => {
                    let indices = axes
                        .iter()
                        .map(|axis| axis.resolve(indices))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(reject)?;
                    self.component(arguments[*formal as usize], &indices, depth + 1)?
                }
                CalculusNode::Rational { value, dimension } => {
                    self.push(CalculusNode::Rational {
                        value: *value,
                        dimension: *dimension,
                    })?
                }
                CalculusNode::KroneckerDelta(a, b) => self.push(CalculusNode::Rational {
                    value: ExactRational::integer(i64::from(
                        a.resolve(indices).map_err(reject)?
                            == b.resolve(indices).map_err(reject)?,
                    )),
                    dimension: DimExponents::DIMENSIONLESS,
                })?,
                CalculusNode::BoundInput(value) | CalculusNode::Differentiated { value, .. } => {
                    get(*value)
                }
                CalculusNode::Neg(value) => self.push(CalculusNode::Neg(get(*value)))?,
                CalculusNode::Add(a, b) => self.push(CalculusNode::Add(get(*a), get(*b)))?,
                CalculusNode::Mul(a, b) => self.push(CalculusNode::Mul(get(*a), get(*b)))?,
                _ => {
                    return Err(reject(
                        "pure spatial expression is outside the polynomial profile",
                    ));
                }
            };
            mapped.push(value);
        }
        Ok(mapped[definition.root().index() as usize])
    }
}
