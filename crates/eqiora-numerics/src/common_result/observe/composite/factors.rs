//! Finite coordinate-product quadrature uses the shared scalar Operator IR.
use crate::factor_measure::{axes, mapped_sample};
use eqiora_core::{RawId, ScalarDomain, ValueType};
use eqiora_meshing::ReferenceCell;
use eqiora_schema::kernel::typing::{SpatialSupport, TypedResidual};

use super::*;

// Numerical regularity uses a projection; values retain the canonical point evaluator.
struct FactorExpression {
    observable: Id<kinds::Observable>,
    work: usize,
    inputs: HashMap<SymbolRef, ValueType>,
}

impl FactorExpression {
    fn new(
        observable: Id<kinds::Observable>,
        typed: &TypedResidual<RawId>,
        projected: &ScalarOperatorIr,
    ) -> Result<Self, Diagnostic> {
        let work = typed
            .expression()
            .nodes()
            .len()
            .max(projected.instruction_count());
        Ok(Self {
            observable,
            work,
            inputs: typed
                .expression()
                .nodes()
                .iter()
                .zip(typed.node_types())
                .filter_map(|(node, ty)| match node {
                    ExprNode::Symbol(symbol) => Some((*symbol, ty.value_type.clone())),
                    _ => None,
                })
                .collect(),
        })
    }
}

impl Context<'_> {
    pub(super) fn factor_supported(&self, definition: &ObservableDef) -> bool {
        definition
            .reduction()
            .input_domain()
            .or_else(|| self.output_domain(definition.id()))
            .and_then(|domain| self.program.spatial_support(domain))
            .is_some_and(|support| matches!(support, SpatialSupport::Coordinates { .. }))
    }

    pub(super) fn evaluate_factors(
        &mut self,
        definition: &ObservableDef,
        point: &Point,
        depth: usize,
    ) -> Result<Evaluation, Diagnostic> {
        if self.tangent.is_some() {
            return Err(invalid(
                "coordinate-factor observation State tangents require an admitted field realization",
            ));
        }
        let ty = definition.value_type();
        if ty.scalar_domain() != ScalarDomain::Real
            || !ty.shape().is_scalar()
            || ty.array_rank() != 0
        {
            return Err(invalid(
                "coordinate-factor quadrature requires a real scalar density",
            ));
        }
        let output = self.output_domain(definition.id());
        if let Some(domain) = output {
            for (axis, bounds) in axes(self.program, domain)? {
                let value = point.get(&axis).ok_or_else(|| {
                    invalid("field-valued Observable requires a point on its exact output support")
                })?;
                if !value.is_finite()
                    || *value < bounds.lower().value()
                    || *value > bounds.upper().value()
                {
                    return Err(invalid(
                        "Observable output point is outside its bounded support",
                    ));
                }
            }
        }
        let typed = self
            .program
            .typed_observable(definition.id())
            .map_err(|errors| errors.into_iter().next().expect("failed typing"))?;
        let projected =
            ScalarOperatorIr::lower_affine_map_density(&typed, &mut |symbol| self.resolve(symbol))?;
        let expression = FactorExpression::new(definition.id(), &typed, &projected)?;
        let value = match definition.reduction() {
            ObservableReduction::Value => self.factor_point(&expression, point, depth)?,
            ObservableReduction::SpatialIntegral {
                domain, measure, ..
            } => {
                if measure == eqiora_schema::kernel::ObservableMeasure::Boundary {
                    return Err(invalid(
                        "coordinate product requires a declared factor volume measure",
                    ));
                }
                self.require_regular_density(&projected, depth)?;
                let mut selected = axes(self.program, domain)?;
                let program = self.program;
                let limits = program
                    .evaluate_observable_limits(definition.id(), &mut |symbol| {
                        self.resolve(symbol)
                    })?;
                let mut orientation = 1.0;
                if let Some([lower, upper]) = limits {
                    let [(_, support)] = selected.as_slice() else {
                        return Err(invalid(
                            "explicit limits require one exact coordinate interval",
                        ));
                    };
                    for limit in [lower, upper] {
                        if !limit.value().is_finite()
                            || limit.dim() != support.lower().dim()
                            || limit.value() < support.lower().value()
                            || limit.value() > support.upper().value()
                        {
                            return Err(invalid(
                                "integral limit is outside its exact coordinate support or has wrong units",
                            ));
                        }
                    }
                    if lower.value() == upper.value() {
                        orientation = 0.0;
                    } else {
                        let (lower, upper) = if lower.value() < upper.value() {
                            (lower, upper)
                        } else {
                            orientation = -1.0;
                            (upper, lower)
                        };
                        selected[0].1 = eqiora_schema::kernel::AxisBounds::new(lower, upper)?;
                    }
                }
                let rule = self.quadratures.get(&domain).ok_or_else(|| {
                    invalid(
                        "factor integral requires explicit quadrature for its exact measure Domain",
                    )
                })?;
                if rule.reference_cell() != ReferenceCell::hypercube(selected.len())? {
                    return Err(invalid(
                        "factor quadrature dimension differs from the selected measure",
                    ));
                }
                self.used_rules.insert(domain);
                let mut sum = 0.0;
                let mut correction = 0.0;
                let cells = match (orientation == 0.0, self.result.plan().as_spatial()) {
                    (true, _) => Vec::new(),
                    (false, Some(plan)) => {
                        plan.factor_quadrature_cells(&selected, self.remaining)?
                    }
                    (false, None) => vec![selected.clone()],
                };
                self.remaining = self
                    .remaining
                    .checked_sub(cells.len())
                    .ok_or_else(|| invalid("factor quadrature exceeds its cell work bound"))?;
                for cell in cells {
                    for sample in rule.points() {
                        let mut local = point.clone();
                        let (coordinates, weight) = mapped_sample(&cell, sample, measure)?;
                        for ((axis, _), coordinate) in selected.iter().zip(coordinates) {
                            local.insert(*axis, coordinate.value());
                        }
                        let term =
                            weight.value() * self.factor_point(&expression, &local, depth)?;
                        let corrected = term - correction;
                        let next = sum + corrected;
                        correction = (next - sum) - corrected;
                        sum = next;
                    }
                }
                orientation * sum
            }
        };
        let value = ValueLiteral::from_real(ty.clone(), value)
            .map_err(|_| invalid("factor integral produced a non-finite value"))?;
        let evaluation = (value, None);
        if output.is_none() {
            self.accepted.insert(definition.id(), evaluation.clone());
        }
        Ok(evaluation)
    }

    fn require_regular_density(
        &mut self,
        operator: &ScalarOperatorIr,
        depth: usize,
    ) -> Result<(), Diagnostic> {
        if depth > 256 {
            return Err(invalid(
                "factor density composition exceeds its dependency depth bound",
            ));
        }
        self.remaining = self
            .remaining
            .checked_sub(operator.instruction_count())
            .ok_or_else(|| invalid("factor density admission exceeds its expression work bound"))?;
        operator.require_regular_density(|symbol| match symbol {
            SymbolRef::Coordinate { .. } => true,
            SymbolRef::Observable(id) => self.output_domain(id).is_some(),
            SymbolRef::Field(id) => self.program.edges().iter().any(|edge| {
                edge.from() == id.erase() && edge.kind() == eqiora_graph::EdgeKind::DefinedOn
            }),
            _ => false,
        })?;
        for symbol in operator.symbols() {
            if let SymbolRef::Observable(id) = symbol {
                let typed = self
                    .program
                    .typed_observable(*id)
                    .map_err(|errors| errors.into_iter().next().expect("failed typing"))?;
                let dependency =
                    ScalarOperatorIr::lower_affine_map_density(&typed, &mut |symbol| {
                        self.resolve(symbol)
                    })?;
                self.require_regular_density(&dependency, depth + 1)?;
            }
        }
        Ok(())
    }

    fn factor_point(
        &mut self,
        expression: &FactorExpression,
        point: &Point,
        depth: usize,
    ) -> Result<f64, Diagnostic> {
        self.remaining = self
            .remaining
            .checked_sub(expression.work)
            .ok_or_else(|| invalid("factor quadrature exceeds its expression work bound"))?;
        let program = self.program;
        let Some(KernelNode::Observable(definition)) = program.node(expression.observable.erase())
        else {
            return Err(invalid("factor density Observable is unavailable"));
        };
        let domain = definition
            .reduction()
            .input_domain()
            .or_else(|| self.output_domain(expression.observable))
            .ok_or_else(|| invalid("factor density has no exact input support"))?;
        let selected = self.evaluation_point(domain, point)?;
        let mut resolve = |input, selected: Option<&eqiora_sem::EvaluationPoint>| {
            let selected =
                selected.ok_or_else(|| invalid("factor input has no exact point context"))?;
            let symbol = match input {
                eqiora_sem::EvaluationInput::Value(SymbolRef::Field(field))
                    if self.result.plan().as_spatial().is_some() =>
                {
                    return spatial::sample(self.result, field, selected);
                }
                eqiora_sem::EvaluationInput::Value(symbol) => symbol,
                eqiora_sem::EvaluationInput::CoordinatePartial {
                    field,
                    factor,
                    axis,
                } => {
                    return spatial::sample_partial(self.result, field, factor, axis, selected);
                }
            };
            let point = selected
                .coordinates()
                .map(|(axis, value)| (axis, value.value()))
                .collect();
            self.factor_input(symbol, &expression.inputs[&symbol], &point, depth)
        };
        let value = if definition.reduction() == ObservableReduction::Value {
            program.evaluate_observable_with_points(
                expression.observable,
                Some(&selected),
                &mut resolve,
            )?
        } else {
            program.evaluate_observable_density_with_points(
                expression.observable,
                &selected,
                &mut resolve,
            )?
        };
        value
            .real_scalar_value()
            .map(|value| value.value())
            .ok_or_else(|| invalid("factor density has no real scalar root"))
    }

    fn factor_input(
        &mut self,
        symbol: SymbolRef,
        ty: &ValueType,
        point: &Point,
        depth: usize,
    ) -> Result<ValueLiteral, Diagnostic> {
        let value = match symbol {
            SymbolRef::Coordinate { factor, axis, .. } => *point
                .get(&(factor.erase(), axis))
                .ok_or_else(|| invalid("coordinate factor is unbound at the integration point"))?,
            SymbolRef::Observable(id) => {
                return self
                    .evaluate(id, point, None, depth + 1)
                    .map(|value| value.0);
            }
            _ => {
                return self.resolve(symbol).ok_or_else(|| {
                    invalid("factor density Field or Parameter is unavailable in this Result")
                });
            }
        };
        ValueLiteral::from_real(ty.clone(), value)
            .map_err(|_| invalid("factor input differs from its declared real scalar type"))
    }
}
