//! Typed functionals consume accepted integration history, never output cadence.

use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, ValueLiteral, ValueType};
use eqiora_ir::ScalarOperatorIr;
use eqiora_schema::kernel::{KernelNode, ObservableReduction, SymbolRef};
use eqiora_sem::KernelProgram;
use eqiora_time::TimeSystem;
use std::collections::HashMap;

use super::{CommonTrajectory, invalid};
use crate::CommonOdePlan;

mod sensitivity;

/// Numerical policy for integrating a retained Observable through time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeFunctionalQuadrature {
    /// Simpson quadrature using the backend's native accepted-step midpoint.
    AcceptedStepSimpson,
}

/// A typed derived quantity bound to one complete accepted Trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonTrajectoryObservation {
    observable: Id<kinds::Observable>,
    trajectory_identity: String,
    value: ValueLiteral,
    quadrature: Option<TimeFunctionalQuadrature>,
    parameter_jvp: bool,
    interval_s: [f64; 2],
}

impl CommonTrajectoryObservation {
    /// Exact Model declaration.
    #[must_use]
    pub const fn observable(&self) -> Id<kinds::Observable> {
        self.observable
    }

    /// Complete accepted Trajectory identity, including its integration history.
    #[must_use]
    pub fn trajectory_identity(&self) -> &str {
        &self.trajectory_identity
    }

    /// Value with the declared physical type and, for an integral, time measure.
    #[must_use]
    pub const fn value(&self) -> &ValueLiteral {
        &self.value
    }

    /// Exact physical interval in coherent SI seconds; terminal scope repeats its time.
    #[must_use]
    pub const fn interval_s(&self) -> [f64; 2] {
        self.interval_s
    }

    /// Fixed endpoint meaning used by this admitted functional profile.
    #[must_use]
    pub const fn endpoint_convention(&self) -> &'static str {
        if self.quadrature.is_some() {
            "fixed-interval-dt"
        } else {
            "terminal-after-events"
        }
    }

    /// Whether this value is a first variation in an admitted Parameter direction.
    #[must_use]
    pub const fn is_parameter_jvp(&self) -> bool {
        self.parameter_jvp
    }

    /// Effective time quadrature; absent for terminal evaluation.
    #[must_use]
    pub const fn quadrature(&self) -> Option<TimeFunctionalQuadrature> {
        self.quadrature
    }
}

impl CommonTrajectory {
    /// Evaluate an Observable at the accepted terminal State.
    ///
    /// The initial profile admits finite scalar ODE Observables. The terminal
    /// state is taken from the accepted integration history, independently of
    /// the requested output schedule.
    pub fn observe_terminal(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
    ) -> Result<CommonTrajectoryObservation, Diagnostic> {
        let evaluator = OdeObservable::new(self, model, observable)?;
        let history = self.ode_history().ok_or_else(|| {
            invalid("terminal Observable requires accepted ODE integration history")
        })?;
        let terminal = history
            .steps()
            .last()
            .ok_or_else(|| invalid("terminal Observable has no accepted integration step"))?;
        let terminal_state = history
            .events()
            .last()
            .filter(|event| event.proposal().time() == terminal.end_time())
            .map_or(terminal.end_state(), |event| event.after_state());
        let value = evaluator.evaluate(terminal.end_time(), terminal_state)?;
        Ok(CommonTrajectoryObservation {
            observable,
            trajectory_identity: self.identity().to_owned(),
            value,
            quadrature: None,
            parameter_jvp: false,
            interval_s: [terminal.end_time(), terminal.end_time()],
        })
    }

    /// Integrate an Observable over the complete accepted Run interval.
    ///
    /// Each accepted step owns its native midpoint. Output States are never
    /// substituted for integration samples. Time measure multiplies the result
    /// dimension by seconds. The selected rule does not change solver cadence.
    pub fn observe_time_integral(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        quadrature: TimeFunctionalQuadrature,
    ) -> Result<CommonTrajectoryObservation, Diagnostic> {
        let evaluator = OdeObservable::new(self, model, observable)?;
        let history = self
            .ode_history()
            .ok_or_else(|| invalid("time Observable requires accepted ODE integration history"))?;
        let mut integral = 0.0;
        for step in history.steps() {
            let start = step.start_time();
            let end = step.end_time();
            let a = evaluator.scalar(start, step.start_state())?;
            let b = evaluator.scalar(start + (end - start) * 0.5, step.midpoint_state())?;
            let c = evaluator.scalar(end, step.end_state())?;
            integral += (end - start) * (a + 4.0 * b + c) / 6.0;
        }
        let seconds = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0])
            .expect("seconds have exact dimension");
        let dimension = evaluator
            .value_type
            .dimension()
            .mul(seconds)
            .ok_or_else(|| invalid("time Observable dimension exceeds its exact representation"))?;
        let value_type = evaluator
            .value_type
            .clone()
            .with_dimension(dimension)
            .map_err(|error| invalid(error.to_string()))?;
        let value = ValueLiteral::from_real(value_type, integral)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(CommonTrajectoryObservation {
            observable,
            trajectory_identity: self.identity().to_owned(),
            value,
            quadrature: Some(quadrature),
            parameter_jvp: false,
            interval_s: [
                history.steps()[0].start_time(),
                history.steps().last().expect("nonempty history").end_time(),
            ],
        })
    }
}

struct OdeObservable<'a> {
    plan: &'a CommonOdePlan,
    program: KernelProgram,
    operator: ScalarOperatorIr,
    input_types: HashMap<SymbolRef, ValueType>,
    root: eqiora_schema::kernel::ExprId,
    value_type: ValueType,
}

impl<'a> OdeObservable<'a> {
    fn new(
        trajectory: &'a CommonTrajectory,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
    ) -> Result<Self, Diagnostic> {
        let CommonTrajectory::Ode { request, .. } = trajectory else {
            return Err(invalid(
                "time functional requires the admitted scalar ODE profile",
            ));
        };
        let plan = request.plan();
        if model != plan.model_artifact() {
            return Err(invalid(
                "time Observable belongs to a foreign or stale Model",
            ));
        }
        let program = model.to_program().map_err(|errors| {
            errors
                .into_iter()
                .next()
                .expect("failed replay has a diagnostic")
        })?;
        let typed = program.typed_observable(observable).map_err(|errors| {
            errors
                .into_iter()
                .next()
                .expect("failed typing has a diagnostic")
        })?;
        let Some(KernelNode::Observable(definition)) = program.node(observable.erase()) else {
            return Err(invalid("time Observable is outside the exact Model"));
        };
        if definition.reduction() != ObservableReduction::Value {
            return Err(invalid(
                "ODE time functional requires an instantaneous finite Observable",
            ));
        }
        let value_type = definition.value_type().clone();
        let operator = ScalarOperatorIr::lower(typed.expression())?;
        let root = typed.expression().roots()[0];
        let input_types = typed
            .expression()
            .nodes()
            .iter()
            .zip(typed.node_types())
            .filter_map(|(node, ty)| {
                if let eqiora_schema::kernel::ExprNode::Symbol(symbol) = node {
                    Some((*symbol, ty.value_type.clone()))
                } else {
                    None
                }
            })
            .collect();
        Ok(Self {
            plan,
            program,
            operator,
            input_types,
            root,
            value_type,
        })
    }

    // The Model supplies derivative meaning; the accepted ODE supplies either
    // a stored lower derivative or its highest rate at this exact point.
    fn coordinate(&self, symbol: SymbolRef) -> Result<(usize, bool), Diagnostic> {
        let (field, order) = match symbol {
            SymbolRef::Field(field) => (field, 0),
            SymbolRef::Derivative(field, order) => (field, order.get()),
            _ => return Err(invalid("Observable input is not a time state coordinate")),
        };
        if let Some(index) = self
            .plan
            .state_coordinates()
            .position(|coordinate| coordinate == (field, order))
        {
            return Ok((index, false));
        }
        if order > 0
            && let Some(index) = self
                .plan
                .state_coordinates()
                .position(|coordinate| coordinate == (field, order - 1))
        {
            return Ok((index, true));
        }
        Err(invalid(
            "Observable derivative order is not supplied by the admitted ODE",
        ))
    }

    fn rates(&self, time: f64, state: &[f64]) -> Result<Option<Vec<f64>>, Diagnostic> {
        let mut required = false;
        for symbol in self.operator.symbols() {
            if matches!(symbol, SymbolRef::Field(_) | SymbolRef::Derivative(..)) {
                required |= self.coordinate(*symbol)?.1;
            }
        }
        if !required {
            return Ok(None);
        }
        let mut rates = vec![0.; state.len()];
        self.plan.system().rhs(time, state, &mut rates)?;
        Ok(Some(rates))
    }

    fn evaluate(&self, time: f64, state: &[f64]) -> Result<ValueLiteral, Diagnostic> {
        let rates = self.rates(time, state)?;
        let mut resolve = |symbol| match symbol {
            SymbolRef::Parameter(id) => self.program.typed_value(id.erase()).cloned(),
            SymbolRef::Field(_) | SymbolRef::Derivative(..) => {
                let (index, rate) = self.coordinate(symbol).ok()?;
                let value = if rate {
                    rates.as_ref()?.get(index)?
                } else {
                    state.get(index)?
                };
                ValueLiteral::from_real(self.input_types.get(&symbol)?.clone(), *value).ok()
            }
            SymbolRef::Time => {
                let dimension = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0])?;
                ValueLiteral::from_real(
                    ValueType::scalar(eqiora_core::ScalarDomain::Real, dimension).ok()?,
                    time,
                )
                .ok()
            }
            _ => None,
        };
        let mut values = self.operator.evaluate_typed(&[self.root], &mut resolve)?;
        let value = values
            .pop()
            .ok_or_else(|| invalid("time Observable has no scalar root"))?;
        if value.value_type() != &self.value_type {
            return Err(invalid(
                "time Observable evaluated outside its declared type",
            ));
        }
        Ok(value)
    }

    fn scalar(&self, time: f64, state: &[f64]) -> Result<f64, Diagnostic> {
        self.evaluate(time, state)?
            .real_scalar_value()
            .map(|value| value.value())
            .ok_or_else(|| invalid("time integration currently requires a real scalar Observable"))
    }
}
