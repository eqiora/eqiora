//! Evaluate each live reduced quantity once before combining its value or tangent.
use std::collections::{HashMap, HashSet};

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DynQuantity, Id, ValueLiteral};
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent, ScalarOperatorIr};
use eqiora_meshing::QuadratureRule;
use eqiora_schema::kernel::{ExprNode, KernelNode, ObservableDef, ObservableReduction, SymbolRef};
use eqiora_sem::KernelProgram;

use super::{CommonResult, invalid, spatial, tangent::StateDerivative};

mod factors;
use factors::Point;

type Evaluation = (ValueLiteral, Option<ValueLiteral>);

pub(super) fn evaluate(
    result: &CommonResult,
    program: &KernelProgram,
    observable: Id<kinds::Observable>,
    quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    tangent: Option<&StateDerivative<'_>>,
    coordinates: Option<&[DynQuantity]>,
) -> Result<Evaluation, Diagnostic> {
    let finite_fields = match (result.plan().as_algebraic(), result.finite_values()) {
        (Some(plan), Some(values)) => plan.field_values(values)?.into_iter().collect(),
        _ => HashMap::new(),
    };
    let mut context = Context {
        result,
        program,
        quadratures,
        tangent,
        finite_fields,
        accepted: HashMap::new(),
        used_rules: HashSet::new(),
        remaining: 1_000_000,
    };
    let point = context.output_point(observable, coordinates)?;
    let value = context.evaluate(observable, &point, None, 0)?;
    if context.used_rules.len() != quadratures.len() {
        return Err(invalid(
            "Observable quadrature contains an unused integration Domain",
        ));
    }
    Ok(value)
}

struct Context<'a> {
    result: &'a CommonResult,
    program: &'a KernelProgram,
    quadratures: &'a HashMap<Id<kinds::Domain>, QuadratureRule>,
    tangent: Option<&'a StateDerivative<'a>>,
    finite_fields: HashMap<Id<kinds::Field>, ValueLiteral>,
    accepted: HashMap<Id<kinds::Observable>, Evaluation>,
    used_rules: HashSet<Id<kinds::Domain>>,
    remaining: usize,
}

impl Context<'_> {
    fn evaluate(
        &mut self,
        id: Id<kinds::Observable>,
        point: &Point,
        selected: Option<&eqiora_sem::EvaluationPoint>,
        depth: usize,
    ) -> Result<Evaluation, Diagnostic> {
        if let Some(value) = self.accepted.get(&id) {
            return Ok(value.clone());
        }
        if depth > 256 {
            return Err(invalid(
                "Observable composition exceeds its dependency depth bound",
            ));
        }
        let Some(KernelNode::Observable(definition)) = self.program.node(id.erase()) else {
            return Err(invalid("Observable is outside the exact Result Model"));
        };
        self.remaining = self
            .remaining
            .checked_sub(definition.expression().nodes().len())
            .ok_or_else(|| invalid("Observable composition exceeds its expression work bound"))?;
        if selected.is_none() && self.factor_supported(definition) {
            return self.evaluate_factors(definition, point, depth);
        }
        if definition.reduction() == ObservableReduction::Value {
            let pointwise = selected.is_some()
                || definition
                    .expression()
                    .nodes()
                    .iter()
                    .any(|node| matches!(node, ExprNode::Evaluate { .. }));
            if pointwise && self.tangent.is_some() {
                return Err(invalid(
                    "point observation State tangents require an admitted reconstruction derivative",
                ));
            }
            let program = self.program;
            let value =
                program.evaluate_observable_with_points(id, selected, &mut |input, selected| {
                    let symbol = match input {
                        eqiora_sem::EvaluationInput::Value(symbol) => symbol,
                        eqiora_sem::EvaluationInput::CoordinatePartial {
                            field,
                            factor,
                            axis,
                        } => {
                            let point = selected
                                .ok_or_else(|| invalid("point partial has no exact point"))?;
                            return spatial::sample_partial(
                                self.result,
                                field,
                                factor,
                                axis,
                                point,
                            );
                        }
                    };
                    if let SymbolRef::Observable(dependency) = symbol {
                        let coordinates = selected
                            .map(|point| {
                                point
                                    .coordinates()
                                    .map(|(axis, value)| (axis, value.value()))
                                    .collect()
                            })
                            .unwrap_or_else(|| point.clone());
                        return self
                            .evaluate(dependency, &coordinates, selected, depth + 1)
                            .map(|value| value.0);
                    }
                    if let Some(value) = self.resolve(symbol) {
                        return Ok(value);
                    }
                    if let (SymbolRef::Field(field), Some(point)) = (symbol, selected) {
                        return spatial::sample(self.result, field, point);
                    }
                    Err(invalid(
                        "point observation input is unavailable in the accepted Result",
                    ))
                })?;
            let derivative = self
                .tangent
                .map(|_| self.finite_derivative(definition))
                .transpose()?;
            let evaluation = (value, derivative);
            if self.program.observable_output_support(id)?.is_none() {
                self.accepted.insert(id, evaluation.clone());
            }
            return Ok(evaluation);
        }
        for node in definition.expression().nodes() {
            if let ExprNode::Symbol(SymbolRef::Observable(dependency)) = node {
                self.evaluate(*dependency, point, selected, depth + 1)?;
            }
        }
        let evaluation = match definition.reduction() {
            ObservableReduction::Value => unreachable!("demanded values were evaluated above"),
            ObservableReduction::SpatialIntegral { input, domain, .. } => {
                if input != domain {
                    return Err(invalid(
                        "this Result realization requires a full spatial integral",
                    ));
                }
                let rule = self.quadratures.get(&domain).ok_or_else(|| {
                    invalid("spatial Observable requires an explicit quadrature rule for its exact Domain")
                })?;
                self.used_rules.insert(domain);
                let typed = self.program.typed_observable(id).map_err(|errors| {
                    errors
                        .into_iter()
                        .next()
                        .expect("failed typing has diagnostic")
                })?;
                let value = spatial::integrate(
                    self.result,
                    self.program,
                    definition,
                    &typed,
                    domain,
                    rule,
                    None,
                )?;
                let derivative = self
                    .tangent
                    .map(|tangent| {
                        spatial::integrate(
                            self.result,
                            self.program,
                            definition,
                            &typed,
                            domain,
                            rule,
                            Some(tangent),
                        )
                    })
                    .transpose()?;
                (value, derivative)
            }
        };
        if evaluation.0.value_type() != definition.value_type() {
            return Err(invalid(
                "evaluated Observable type differs from its admitted declaration",
            ));
        }
        self.accepted.insert(id, evaluation.clone());
        Ok(evaluation)
    }

    fn resolve(&self, symbol: SymbolRef) -> Option<ValueLiteral> {
        match symbol {
            SymbolRef::Observable(id) => self.accepted.get(&id).map(|(value, _)| value.clone()),
            SymbolRef::Parameter(id) => self.program.typed_value(id.erase()).cloned(),
            SymbolRef::Field(id) => self.finite_fields.get(&id).cloned(),
            _ => {
                let plan = self.result.plan().as_algebraic()?;
                let values = self.result.finite_values()?;
                let index = plan
                    .symbols()
                    .iter()
                    .position(|candidate| *candidate == symbol)?;
                let ty = eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    plan.dimensions()[index],
                )
                .ok()?;
                ValueLiteral::from_real(ty, values[index]).ok()
            }
        }
    }

    fn finite_derivative(&self, definition: &ObservableDef) -> Result<ValueLiteral, Diagnostic> {
        if matches!(self.tangent, Some(StateDerivative::Second { .. }))
            && definition.expression().nodes().iter().any(|node| {
                !matches!(
                    node,
                    ExprNode::Symbol(SymbolRef::Observable(_))
                        | ExprNode::Add(..)
                        | ExprNode::Sub(..)
                        | ExprNode::Neg(_)
                )
            })
        {
            return Err(invalid(
                "second variation requires sums or differences of fixed spatial Observables",
            ));
        }
        if definition.value_type().scalar_domain() != eqiora_core::ScalarDomain::Real
            || !definition.value_type().shape().is_scalar()
            || definition.value_type().array_rank() != 0
        {
            return Err(invalid(
                "composite Observable State JVP requires a real scalar value",
            ));
        }
        let operator = ScalarOperatorIr::lower(definition.expression())?;
        let mut values = Vec::new();
        let mut directions = Vec::new();
        let mut roles = Vec::new();
        for &symbol in operator.symbols() {
            let value = self
                .resolve(symbol)
                .and_then(|value| value.real_scalar_value())
                .map(|value| value.value())
                .ok_or_else(|| {
                    invalid("composite Observable State JVP requires real scalar inputs")
                })?;
            let direction = match symbol {
                SymbolRef::Parameter(_) => {
                    roles.push(DifferentiationRole::Frozen);
                    values.push(value);
                    continue;
                }
                SymbolRef::Observable(id) => self
                    .accepted
                    .get(&id)
                    .and_then(|(_, derivative)| derivative.as_ref())
                    .and_then(ValueLiteral::real_scalar_value)
                    .map(|value| value.value())
                    .ok_or_else(|| {
                        invalid("composite Observable dependency has no scalar State tangent")
                    })?,
                _ => {
                    return Err(invalid(
                        "composite Observable State JVP requires reduced Observable inputs or fixed Parameters",
                    ));
                }
            };
            roles.push(DifferentiationRole::Unknown);
            values.push(value);
            directions.push(direction);
        }
        let linearization = operator.linearize(&values, &roles)?;
        let mut action = [0.0];
        linearization.jvp(RelationTangent::Unknown(&directions), &mut action)?;
        ValueLiteral::from_real(definition.value_type().clone(), action[0])
            .map_err(|error| invalid(error.to_string()))
    }
}
