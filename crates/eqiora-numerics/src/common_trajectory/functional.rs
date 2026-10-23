//! Typed functionals consume accepted integration history, never output cadence.

use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, ValueLiteral, ValueType};
use eqiora_schema::kernel::{KernelNode, ObservableReduction, SymbolRef};
use eqiora_sem::KernelProgram;
use std::collections::HashMap;

use super::{CommonTrajectory, TrajectoryPayload, invalid};
use crate::CommonTimePlan;

mod evaluation;
mod extrema;
mod integrate;
mod interpolation;
mod interval;
mod sensitivity;
mod spatial;

/// Interpolation of accepted Observable samples, shared by temporal reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeObservationInterpolation {
    /// Quadratic through each accepted step's native endpoint and
    /// midpoint Observable values. Integrating a complete step gives Simpson quadrature;
    /// a clipped window restricts that same quadratic, not the solver State.
    NativeMidpointQuadratic,
    /// Linear interpolation of Observable values at exact accepted-step endpoints.
    EndpointLinear,
}

/// A typed derived quantity bound to one exact accepted Trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonTrajectoryObservation {
    observable: Id<kinds::Observable>,
    trajectory_identity: String,
    value: ValueLiteral,
    interpolation: Option<TimeObservationInterpolation>,
    parameter_jvp: bool,
    extremum: Option<(extrema::Extremum, f64)>,
    interval_s: [f64; 2],
    spatial_quadratures: HashMap<Id<kinds::Domain>, eqiora_meshing::QuadratureRule>,
}

impl CommonTrajectoryObservation {
    /// Exact spatial quadrature used inside the time reduction; empty for finite ODE values.
    #[must_use]
    pub fn spatial_quadratures(
        &self,
    ) -> &HashMap<Id<kinds::Domain>, eqiora_meshing::QuadratureRule> {
        &self.spatial_quadratures
    }

    /// Exact Model declaration.
    #[must_use]
    pub const fn observable(&self) -> Id<kinds::Observable> {
        self.observable
    }

    /// Exact Trajectory identity, including retained States and available integration history.
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
        if self.extremum.is_some() {
            "closed-interval-event-sides-earliest-tie"
        } else if self.interpolation.is_some() {
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

    /// Physical SI time attaining an extremum of the declared interpolant.
    /// This is not an interpolated restart State.
    #[must_use]
    pub fn extremum_time_s(&self) -> Option<f64> {
        self.extremum.map(|(_, time)| time)
    }

    /// Reduction meaning, owned by the native observation rather than adapters.
    #[must_use]
    pub const fn evaluation_kind(&self) -> &'static str {
        match (
            self.extremum,
            self.interpolation.is_some(),
            self.parameter_jvp,
        ) {
            (Some((extrema::Extremum::Minimum, _)), _, _) => "time-minimum",
            (Some((extrema::Extremum::Maximum, _)), _, _) => "time-maximum",
            (None, false, false) => "terminal",
            (None, true, false) => "time-integral",
            (None, false, true) => "terminal-parameter-jvp",
            (None, true, true) => "time-integral-parameter-jvp",
        }
    }

    /// Effective time interpolation; absent for terminal evaluation.
    #[must_use]
    pub const fn interpolation(&self) -> Option<TimeObservationInterpolation> {
        self.interpolation
    }
}

impl CommonTrajectory {
    pub(crate) fn observe_time_output(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        output_index: usize,
    ) -> Result<(ValueLiteral, &crate::CommonTimeState), Diagnostic> {
        let states = self
            .time_states()
            .ok_or_else(|| invalid("Observable requires finite-time outputs"))?;
        let state = states
            .get(output_index)
            .ok_or_else(|| invalid("observation output index is outside the retained schedule"))?;
        let evaluator = TimeObservable::new(self, model, observable)?;
        Ok((evaluator.evaluate(state.time_s(), state.values())?, state))
    }

    /// Evaluate an Observable at the accepted terminal State.
    ///
    /// Finite-time Observables retain their full value shape. The terminal
    /// State comes from the accepted method history, independently of
    /// the requested output schedule.
    pub fn observe_terminal(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
    ) -> Result<CommonTrajectoryObservation, Diagnostic> {
        let evaluator = TimeObservable::new(self, model, observable)?;
        let (time, terminal_state) = match &self.payload {
            TrajectoryPayload::Ode { history, .. } => {
                let terminal = history.steps().last().expect("accepted history");
                let state = history
                    .events()
                    .last()
                    .filter(|event| event.proposal().time() == terminal.end_time())
                    .map_or(terminal.end_state(), |event| event.after_state());
                (terminal.end_time(), state)
            }
            TrajectoryPayload::Process { history, .. } => {
                let state = history.state();
                (state.time_s(), state.values())
            }
            _ => {
                return Err(invalid(
                    "terminal finite-time Observable requires a finite-time trajectory",
                ));
            }
        };
        let value = evaluator.evaluate(time, terminal_state)?;
        Ok(CommonTrajectoryObservation {
            observable,
            trajectory_identity: self.identity().to_owned(),
            value,
            interpolation: None,
            parameter_jvp: false,
            extremum: None,
            interval_s: [time, time],
            spatial_quadratures: HashMap::new(),
        })
    }

    /// Integrate an Observable over a covered interval of accepted history.
    ///
    /// ODE history supplies native endpoint and midpoint values; spatial history
    /// supplies exact accepted endpoint States and admits the trapezoid rule.
    /// Each rule interpolates Observable values, never States. Spatial measures
    /// require explicit quadrature. Time measure multiplies the physical dimension
    /// by seconds; a query does not change solver cadence or output retention.
    /// `None` selects the complete Run; explicit endpoints are fixed SI seconds.
    /// Empty, reversed, non-finite or incompletely covered windows reject.
    pub fn observe_time_integral(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        interpolation: TimeObservationInterpolation,
        interval_s: Option<[f64; 2]>,
        spatial_quadratures: &HashMap<Id<kinds::Domain>, eqiora_meshing::QuadratureRule>,
    ) -> Result<CommonTrajectoryObservation, Diagnostic> {
        let interval = interval::TimeObservationInterval::new(self, interval_s)?;
        let value = match &self.payload {
            TrajectoryPayload::SpatialTime { .. } => {
                return Err(invalid(
                    "continuous spatial trajectory functionals are not yet implemented",
                ));
            }
            TrajectoryPayload::Process { .. } => {
                return Err(invalid(
                    "stochastic trajectory has no admitted smooth time interpolant",
                ));
            }
            TrajectoryPayload::Ode { history, .. } => {
                if !spatial_quadratures.is_empty() {
                    return Err(invalid(
                        "ODE time Observable does not consume spatial quadrature",
                    ));
                }
                let evaluator = TimeObservable::new(self, model, observable)?;
                integrate::integrate(
                    history.steps(),
                    &interval,
                    interpolation,
                    evaluator.value_type.clone(),
                    |step| [step.start_time(), step.end_time()],
                    |step, sample| match sample {
                        0 => evaluator.evaluate(step.start_time(), step.start_state()),
                        1 => evaluator.evaluate(
                            step.start_time() + (step.end_time() - step.start_time()) * 0.5,
                            step.midpoint_state(),
                        ),
                        _ => evaluator.evaluate(step.end_time(), step.end_state()),
                    },
                )?
            }
            TrajectoryPayload::SpatialTransient {
                request, states, ..
            } => spatial::integrate(
                request,
                states,
                model,
                observable,
                interpolation,
                &interval,
                spatial_quadratures,
            )?,
        };
        Ok(CommonTrajectoryObservation {
            observable,
            trajectory_identity: self.identity().to_owned(),
            value,
            interpolation: Some(interpolation),
            parameter_jvp: false,
            extremum: None,
            interval_s: interval.bounds(),
            spatial_quadratures: spatial_quadratures.clone(),
        })
    }
}

pub(super) struct TimeObservable<'a> {
    plan: &'a CommonTimePlan,
    program: KernelProgram,
    observable: Id<kinds::Observable>,
    value_type: ValueType,
}

impl<'a> TimeObservable<'a> {
    pub(super) fn new(
        trajectory: &'a CommonTrajectory,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
    ) -> Result<Self, Diagnostic> {
        let request = trajectory
            .time_request()
            .ok_or_else(|| invalid("finite time Observable requires a finite-time trajectory"))?;
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
        program.typed_observable(observable).map_err(|errors| {
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
        Ok(Self {
            plan,
            program,
            observable,
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
            && self.plan.as_ode().is_some()
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

    pub(super) fn evaluate(&self, time: f64, state: &[f64]) -> Result<ValueLiteral, Diagnostic> {
        evaluation::Context::new(self, time, state, None)
            .evaluate(self.observable, 0)
            .map(|evaluation| evaluation.value)
    }

    fn scalar(&self, time: f64, state: &[f64]) -> Result<f64, Diagnostic> {
        self.evaluate(time, state)?
            .real_scalar_value()
            .map(|value| value.value())
            .ok_or_else(|| invalid("time integration currently requires a real scalar Observable"))
    }
}
