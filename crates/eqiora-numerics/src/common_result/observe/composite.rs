//! Evaluate each live reduced quantity once before combining its value or tangent.
use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DynQuantity, Id, ValueLiteral};
use eqiora_ir::{
    ComponentScalarization, DifferentiationRole, LinearizedRelation, RelationTangent,
    ScalarOperatorIr,
};
use eqiora_meshing::QuadratureRule;
use eqiora_schema::kernel::{ExprNode, KernelNode, ObservableDef, ObservableReduction, SymbolRef};
use eqiora_sem::KernelProgram;

use super::{invalid, occurrence::Occurrence, spatial, tangent::StateDerivative};

mod factors;
mod point;
mod probability;
use point::Point;

type Evaluation = (ValueLiteral, Option<ValueLiteral>);

pub(super) fn evaluate(
    occurrence: &Occurrence<'_>,
    program: &KernelProgram,
    observable: Id<kinds::Observable>,
    quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    tangent: Option<&StateDerivative<'_>>,
    coordinates: Option<&[DynQuantity]>,
) -> Result<Evaluation, Diagnostic> {
    let finite_fields = match (occurrence.plan().as_algebraic(), occurrence.finite_values()) {
        (Some(plan), Some(values)) => plan.field_values(values)?.into_iter().collect(),
        _ => HashMap::new(),
    };
    let remaining = Cell::new(1_000_000);
    let mut context = Context {
        occurrence,
        program,
        quadratures,
        tangent,
        finite_fields,
        accepted: HashMap::new(),
        used_rules: HashSet::new(),
        remaining: &remaining,
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
    occurrence: &'a Occurrence<'a>,
    program: &'a KernelProgram,
    quadratures: &'a HashMap<Id<kinds::Domain>, QuadratureRule>,
    tangent: Option<&'a StateDerivative<'a>>,
    finite_fields: HashMap<Id<kinds::Field>, ValueLiteral>,
    accepted: HashMap<Id<kinds::Observable>, Evaluation>,
    used_rules: HashSet<Id<kinds::Domain>>,
    remaining: &'a Cell<usize>,
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
        self.remaining.set(
            self.remaining
                .get()
                .checked_sub(definition.expression().nodes().len())
                .ok_or_else(|| {
                    invalid("Observable composition exceeds its expression work bound")
                })?,
        );
        if matches!(
            definition.reduction(),
            ObservableReduction::RandomVariable { .. }
        ) {
            return Err(invalid(
                "a random variable requires its bound probability-law context",
            ));
        }
        if matches!(
            definition.reduction(),
            ObservableReduction::ProbabilityMoment { .. }
        ) && self.tangent.is_some()
        {
            return Err(invalid(
                "probability moment derivatives require separately admitted differentiation assumptions",
            ));
        }
        if let ObservableReduction::ProbabilityMoment {
            distribution,
            moment,
        } = definition.reduction()
            && let Some(KernelNode::Relation(law)) = self.program.node(distribution.erase())
            && let eqiora_schema::kernel::RelationMeaning::ProbabilityDensity {
                domain,
                measure,
                ..
            } = law.meaning()
        {
            return self
                .evaluate_continuous_moment(definition, *domain, *measure, moment, point, depth);
        }
        if selected.is_none() && self.factor_supported(definition) {
            return self.evaluate_factors(definition, point, depth);
        }
        if matches!(
            definition.reduction(),
            ObservableReduction::Value | ObservableReduction::ProbabilityMoment { .. }
        ) {
            let output_point = if selected.is_none() {
                self.program
                    .observable_output_support(id)?
                    .map(|support| {
                        self.evaluation_point(
                            support.domain().downcast().expect("admitted output Domain"),
                            point,
                        )
                    })
                    .transpose()?
            } else {
                None
            };
            let selected = selected.or(output_point.as_ref());
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
                    self.resolve_at(input, selected, point, depth)
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
            ObservableReduction::Value | ObservableReduction::ProbabilityMoment { .. } => {
                unreachable!("demanded values were evaluated above")
            }
            ObservableReduction::RandomVariable { .. } => {
                unreachable!("unbound random variable was rejected above")
            }
            ObservableReduction::SpatialIntegral { input, domain, .. } => {
                if input != domain || definition.reduction().limits().is_some() {
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
                    self.occurrence,
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
                            self.occurrence,
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

    fn resolve_at(
        &mut self,
        input: eqiora_sem::EvaluationInput,
        selected: Option<&eqiora_sem::EvaluationPoint>,
        point: &Point,
        depth: usize,
    ) -> Result<ValueLiteral, Diagnostic> {
        let symbol = match input {
            eqiora_sem::EvaluationInput::Value(symbol) => symbol,
            eqiora_sem::EvaluationInput::CoordinatePartial {
                field,
                factor,
                axis,
            } => {
                let point = selected.ok_or_else(|| invalid("point partial has no exact point"))?;
                return spatial::sample_partial(self.occurrence, field, factor, axis, point);
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
            return spatial::sample(self.occurrence, field, point);
        }
        Err(invalid(
            "point observation input is unavailable in the accepted Result",
        ))
    }

    fn resolve(&self, symbol: SymbolRef) -> Option<ValueLiteral> {
        match symbol {
            SymbolRef::Observable(id) => self.accepted.get(&id).map(|(value, _)| value.clone()),
            SymbolRef::Parameter(id) => self.program.typed_value(id.erase()).cloned(),
            SymbolRef::Time => ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    eqiora_core::DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0])?,
                )
                .ok()?,
                self.occurrence.time_s()?,
            )
            .ok(),
            SymbolRef::Field(id) => self.finite_fields.get(&id).cloned(),
            _ => {
                let plan = self.occurrence.plan().as_algebraic()?;
                let values = self.occurrence.finite_values()?;
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
        let typed = self
            .program
            .typed_observable(definition.id())
            .map_err(|errors| {
                errors
                    .into_iter()
                    .next()
                    .expect("failed typing has a diagnostic")
            })?;
        let operator = ComponentScalarization::lower(&typed)?;
        let mut bound = HashMap::new();
        for coordinate in operator.rows().iter().flat_map(|row| row.symbols()) {
            if bound.contains_key(coordinate) {
                continue;
            }
            let symbol = coordinate.symbol();
            let value = self
                .resolve(symbol)
                .ok_or_else(|| invalid("composite Observable State input is unavailable"))?;
            let value = super::projection::scalar_coordinate(&value, coordinate)?;
            let (direction, role) = match symbol {
                SymbolRef::Parameter(_) => (0., DifferentiationRole::Frozen),
                SymbolRef::Observable(id) => {
                    let direction = self
                        .accepted
                        .get(&id)
                        .and_then(|(_, direction)| direction.as_ref())
                        .ok_or_else(|| {
                            invalid("composite Observable dependency has no State tangent")
                        })?;
                    (
                        super::projection::scalar_coordinate(direction, coordinate)?,
                        DifferentiationRole::Unknown,
                    )
                }
                _ => {
                    return Err(invalid(
                        "composite Observable State JVP requires reduced Observable inputs or fixed Parameters",
                    ));
                }
            };
            bound.insert(coordinate.clone(), (value, direction, role));
        }
        let linearization = operator.linearize(|coordinate| {
            bound
                .get(coordinate)
                .map(|(value, _, role)| (*value, *role))
        })?;
        let directions = linearization
            .unknown_coordinates()
            .iter()
            .map(|coordinate| bound[coordinate].1)
            .collect::<Vec<_>>();
        let mut action = vec![0.; linearization.residual_dimension()];
        linearization.jvp(RelationTangent::Unknown(&directions), &mut action)?;
        super::projection::from_coordinates(definition.value_type(), &action)
    }
}
