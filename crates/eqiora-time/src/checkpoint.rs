//! Numerical continuation data, separate from a physical State or completed solve.
use crate::diagnostic::invalid_plan;
use crate::{
    AcceptedTimeHistory, InitialConditionPolicy, RootRegistrationId, TimeBackendIdentity,
    TimeEquationClass, TimeExecutionReport, TimeMethod, TimePlan, TimeProblem,
};
use eqiora_core::Diagnostic;

/// Accepted smooth prefix and the controller state needed to continue it exactly.
///
/// The current profile is explicit Tsitouras45 with an I-controller: the next
/// step and endpoint derivative are continuation-critical. Past native midpoint
/// stencils and requested samples are retained, not reconstructed from outputs.
/// A canonical Run must additionally bind its Model/Plan and committed resets.
/// This is neither a completed solution nor a provider-independent restart.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeCheckpoint {
    plan: TimePlan,
    report: TimeExecutionReport,
    registration: Option<RootRegistrationId>,
    initial_state: Vec<f64>,
    next_step: f64,
    derivative: Vec<f64>,
    history: Option<AcceptedTimeHistory>,
    samples: Vec<f64>,
}

impl TimeCheckpoint {
    /// Retain the initialized native method before its first trial.
    ///
    /// `next_step` is the provider's initialized step, which may differ from the
    /// requested guess. `derivative` is its retained endpoint derivative.
    /// # Errors
    /// Rejects unsupported methods/policies or invalid controller data.
    pub fn initial(
        problem: &TimeProblem<'_>,
        plan: &TimePlan,
        backend: TimeBackendIdentity,
        registration: Option<RootRegistrationId>,
        next_step: f64,
        derivative: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        plan.validate_for(problem)?;
        if plan.method() != TimeMethod::Tsitouras45
            || problem.equation_class() != TimeEquationClass::ExplicitOde
            || problem.initial_condition() != InitialConditionPolicy::Provided
        {
            return Err(invalid_plan(
                "exact continuation requires explicit Tsitouras45 with a provided initial state",
            ));
        }
        validate_controller(
            plan.start_time(),
            problem.dimension(),
            next_step,
            &derivative,
        )?;
        Ok(Self {
            plan: plan.clone(),
            report: TimeExecutionReport::new(
                backend,
                plan.method(),
                problem.equation_class(),
                problem.initial_condition(),
            ),
            registration,
            initial_state: problem.initial_state().to_vec(),
            next_step,
            derivative,
            history: None,
            samples: Vec::new(),
        })
    }

    /// Retain the complete smooth prefix at a later accepted native boundary.
    ///
    /// Samples are exactly the requested times at or before that boundary, in
    /// time-major order. A root/reset boundary belongs to the canonical event
    /// owner and starts a new smooth segment after its reset is committed.
    /// # Errors
    /// Rejects missing samples, discontinuities, a foreign origin, or invalid
    /// controller data. A completed horizon must be accepted as a solution.
    pub fn with_history(
        mut self,
        history: AcceptedTimeHistory,
        samples: Vec<f64>,
        next_step: f64,
        derivative: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        let first = &history.steps()[0];
        let last = history
            .steps()
            .last()
            .expect("accepted history is nonempty");
        let dimension = self.initial_state.len();
        let count = self
            .plan
            .output_times()
            .partition_point(|time| *time <= last.end_time());
        if history.dimension() != dimension
            || !history.events().is_empty()
            || first.start_time().to_bits() != self.plan.start_time().to_bits()
            || !same_values(first.start_state(), &self.initial_state)
            || last.end_time() >= *self.plan.output_times().last().expect("validated plan")
            || count.checked_mul(dimension) != Some(samples.len())
            || samples.iter().any(|value| !value.is_finite())
        {
            return Err(invalid_plan(
                "continuation history or samples differ from the exact unfinished smooth request",
            ));
        }
        validate_controller(last.end_time(), dimension, next_step, &derivative)?;
        self.next_step = next_step;
        self.derivative = derivative;
        self.history = Some(history);
        self.samples = samples;
        Ok(self)
    }

    /// Reject a changed numerical request or provider before constructing a solver.
    /// # Errors
    /// Rejects changed controls, outputs, initial state, method, provider, or roots.
    pub fn validate_for(
        &self,
        problem: &TimeProblem<'_>,
        plan: &TimePlan,
        backend: TimeBackendIdentity,
        registration: Option<RootRegistrationId>,
    ) -> Result<(), Diagnostic> {
        if self.plan != *plan
            || self.plan.start_time().to_bits() != plan.start_time().to_bits()
            || !same_values(&self.initial_state, problem.initial_state())
            || self.report
                != TimeExecutionReport::new(
                    backend,
                    plan.method(),
                    problem.equation_class(),
                    problem.initial_condition(),
                )
            || self.registration != registration
        {
            return Err(invalid_plan(
                "checkpoint differs from the exact time request, provider, or root registration",
            ));
        }
        Ok(())
    }

    /// Exact integration and output policy of the original smooth segment.
    #[must_use]
    pub const fn plan(&self) -> &TimePlan {
        &self.plan
    }
    /// Exact provider and admitted numerical profile.
    #[must_use]
    pub const fn report(&self) -> TimeExecutionReport {
        self.report
    }
    /// Root registration, absent for a root-free segment.
    #[must_use]
    pub const fn registration(&self) -> Option<RootRegistrationId> {
        self.registration
    }
    /// Original smooth-segment state, before all retained accepted steps.
    #[must_use]
    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }
    /// Last accepted model time, or the initialized segment origin.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.history
            .as_ref()
            .map_or(self.plan.start_time(), |history| {
                history.steps().last().expect("accepted history").end_time()
            })
    }
    /// Last accepted coordinates; cancelled trial coordinates never belong here.
    #[must_use]
    pub fn state(&self) -> &[f64] {
        self.history
            .as_ref()
            .map_or(&self.initial_state, |history| {
                history
                    .steps()
                    .last()
                    .expect("accepted history")
                    .end_state()
            })
    }
    /// Next native step, including the retained controller decision.
    #[must_use]
    pub const fn next_step(&self) -> f64 {
        self.next_step
    }
    /// Retained derivative at the accepted endpoint, used by the method's next trial.
    #[must_use]
    pub fn derivative(&self) -> &[f64] {
        &self.derivative
    }
    /// Complete smooth accepted history, absent before the first accepted step.
    #[must_use]
    pub const fn history(&self) -> Option<&AcceptedTimeHistory> {
        self.history.as_ref()
    }
    /// Already evaluated requested outputs in time-major order.
    #[must_use]
    pub fn samples(&self) -> &[f64] {
        &self.samples
    }
}

fn validate_controller(
    time: f64,
    dimension: usize,
    next_step: f64,
    derivative: &[f64],
) -> Result<(), Diagnostic> {
    if !next_step.is_finite()
        || next_step <= 0.0
        || time + next_step <= time
        || derivative.len() != dimension
        || derivative.iter().any(|value| !value.is_finite())
    {
        return Err(invalid_plan(
            "checkpoint has incomplete or invalid native controller state",
        ));
    }
    Ok(())
}

fn same_values(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
