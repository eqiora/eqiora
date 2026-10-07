//! Typed functionals consume accepted integration history, never output cadence.

use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, ValueLiteral, ValueType};
use eqiora_schema::kernel::{KernelNode, ObservableReduction, SymbolRef};
use eqiora_sem::KernelProgram;
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
    /// Finite real and complex ODE Observables retain their full shape. The terminal
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
        let count = evaluator
            .value_type
            .shape()
            .component_count()
            .ok_or_else(|| invalid("time Observable shape overflows"))?;
        let mut integral = vec![(0., 0.); count];
        for step in history.steps() {
            let start = step.start_time();
            let end = step.end_time();
            let a = evaluator.evaluate(start, step.start_state())?;
            let b = evaluator.evaluate(start + (end - start) * 0.5, step.midpoint_state())?;
            let c = evaluator.evaluate(end, step.end_state())?;
            for (index, sum) in integral.iter_mut().enumerate() {
                let a = a
                    .component(index)
                    .ok_or_else(|| invalid("non-numeric time Observable"))?;
                let b = b
                    .component(index)
                    .ok_or_else(|| invalid("non-numeric time Observable"))?;
                let c = c
                    .component(index)
                    .ok_or_else(|| invalid("non-numeric time Observable"))?;
                sum.0 += (end - start) * (a.0 + 4.0 * b.0 + c.0) / 6.;
                sum.1 += (end - start) * (a.1 + 4.0 * b.1 + c.1) / 6.;
            }
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
        let value =
            ValueLiteral::new(value_type, integral).map_err(|error| invalid(error.to_string()))?;
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

pub(super) struct OdeObservable<'a> {
    plan: &'a CommonOdePlan,
    program: KernelProgram,
    observable: Id<kinds::Observable>,
    input_types: HashMap<SymbolRef, ValueType>,
    value_type: ValueType,
}

impl<'a> OdeObservable<'a> {
    pub(super) fn new(
        trajectory: &'a CommonTrajectory,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
    ) -> Result<Self, Diagnostic> {
        let CommonTrajectory::Ode { request, .. } = trajectory else {
            return Err(invalid("time functional requires the admitted ODE profile"));
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
            observable,
            input_types,
            value_type,
        })
    }

    // The Model supplies derivative meaning; the accepted ODE supplies either
    // a stored lower derivative or its highest rate at this exact point.
    fn coordinate(&self, symbol: SymbolRef) -> Result<(usize, bool), Diagnostic> {
        self.component_coordinate(symbol, 0, false)
    }

    fn component_coordinate(
        &self,
        symbol: SymbolRef,
        component: usize,
        imaginary: bool,
    ) -> Result<(usize, bool), Diagnostic> {
        let (field, order) = match symbol {
            SymbolRef::Field(field) => (field, 0),
            SymbolRef::Derivative(field, order) => (field, order.get()),
            _ => return Err(invalid("Observable input is not a time state coordinate")),
        };
        if let Some(index) = self.plan.state_coordinates().position(|coordinate| {
            coordinate == eqiora_core::TimeStateCoordinate::new(field, order, component, imaginary)
        }) {
            return Ok((index, false));
        }
        if order > 0
            && let Some(index) = self.plan.state_coordinates().position(|coordinate| {
                coordinate
                    == eqiora_core::TimeStateCoordinate::new(field, order - 1, component, imaginary)
            })
        {
            return Ok((index, true));
        }
        Err(invalid(
            "Observable derivative order is not supplied by the admitted ODE",
        ))
    }

    fn rates(&self, time: f64, state: &[f64]) -> Result<Option<Vec<f64>>, Diagnostic> {
        let mut required = false;
        for symbol in self.input_types.keys() {
            if matches!(symbol, SymbolRef::Field(_) | SymbolRef::Derivative(..)) {
                required |= self.coordinate(*symbol)?.1;
            }
        }
        if !required {
            return Ok(None);
        }
        let problem = eqiora_time::TimeProblem::new(
            self.plan.system(),
            self.plan.equation_class(),
            eqiora_time::InitialConditionPolicy::Provided,
            state.to_vec(),
        )?;
        Ok(Some(problem.rate(time, state)?))
    }

    pub(super) fn evaluate(&self, time: f64, state: &[f64]) -> Result<ValueLiteral, Diagnostic> {
        let rates = self.rates(time, state)?;
        let mut resolve = |symbol| match symbol {
            SymbolRef::Parameter(id) => self.program.typed_value(id.erase()).cloned(),
            SymbolRef::Field(_) | SymbolRef::Derivative(..) => {
                let ty = self.input_types.get(&symbol)?.clone();
                let complex = ty.scalar_domain() == eqiora_core::ScalarDomain::Complex;
                let mut components = Vec::new();
                for component in 0..ty.shape().component_count()? {
                    let value = |imaginary| {
                        let (index, rate) = self
                            .component_coordinate(symbol, component, imaginary)
                            .ok()?;
                        if rate {
                            rates.as_ref()?.get(index).copied()
                        } else {
                            state.get(index).copied()
                        }
                    };
                    components.push((value(false)?, if complex { value(true)? } else { 0. }));
                }
                ValueLiteral::new(ty, components).ok()
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
        let value = self
            .program
            .evaluate_finite_observable(self.observable, &mut resolve)?;
        Ok(value)
    }

    fn scalar(&self, time: f64, state: &[f64]) -> Result<f64, Diagnostic> {
        self.evaluate(time, state)?
            .real_scalar_value()
            .map(|value| value.value())
            .ok_or_else(|| invalid("time integration currently requires a real scalar Observable"))
    }
}
