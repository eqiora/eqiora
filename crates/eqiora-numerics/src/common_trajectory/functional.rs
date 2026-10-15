//! Typed functionals consume accepted integration history, never output cadence.

use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, ValueLiteral, ValueType};
use eqiora_schema::kernel::{KernelNode, ObservableReduction, SymbolRef};
use eqiora_sem::KernelProgram;
use std::collections::HashMap;

use super::{CommonTrajectory, invalid};
use crate::CommonOdePlan;

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
    pub(crate) fn observe_ode_output(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        output_index: usize,
    ) -> Result<(ValueLiteral, &crate::CommonOdeState), Diagnostic> {
        let states = self
            .ode_states()
            .ok_or_else(|| invalid("Observable requires ODE outputs"))?;
        let state = states
            .get(output_index)
            .ok_or_else(|| invalid("observation output index is outside the retained schedule"))?;
        let evaluator = OdeObservable::new(self, model, observable)?;
        Ok((evaluator.evaluate(state.time_s(), state.values())?, state))
    }

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
            interpolation: None,
            parameter_jvp: false,
            extremum: None,
            interval_s: [terminal.end_time(), terminal.end_time()],
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
        let value = match self {
            Self::Ode { history, .. } => {
                if !spatial_quadratures.is_empty() {
                    return Err(invalid(
                        "ODE time Observable does not consume spatial quadrature",
                    ));
                }
                let evaluator = OdeObservable::new(self, model, observable)?;
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
            Self::SpatialTransient {
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
