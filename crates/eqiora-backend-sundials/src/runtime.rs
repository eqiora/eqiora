use crate::native::Session;
use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_time::{
    AcceptedTimeHistory, ImplicitDaeInitialization, ImplicitDaeProblem, TimeBackendIdentity,
    TimeEquationClass, TimeExecutionReport, TimeMethod, TimePlan, TimeSolution,
};

/// Residual-native adaptive primal integration through SUNDIALS IDA 7.8.0.
///
/// Dense serial real-coordinate algebra; no sensitivity, events or native
/// checkpoint claim. Complex systems use their existing real coordinates.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdaTimeBackend;
impl IdaTimeBackend {
    /// Native release checked before execution.
    pub const IDENTITY: TimeBackendIdentity =
        TimeBackendIdentity::new("eqiora.time.sundials-ida", "7.8.0");
    /// Construct the stateless adapter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
    /// Accept a supplied pair unchanged, or solve the admitted index-one IC problem.
    /// # Errors
    /// Returns native, callback, version or method admission diagnostics.
    pub fn initialize(
        &self,
        problem: &ImplicitDaeProblem<'_>,
        plan: &TimePlan,
    ) -> Result<ImplicitDaeInitialization, Diagnostic> {
        admit(problem, plan)?;
        Session::new(problem, plan)?.initialization()
    }
    /// Integrate the original residual with its analytic paired JVP.
    /// # Errors
    /// Propagates the first callback failure or a native integration diagnostic.
    pub fn solve(
        &self,
        problem: &ImplicitDaeProblem<'_>,
        plan: &TimePlan,
    ) -> Result<TimeSolution, Diagnostic> {
        admit(problem, plan)?;
        let mut session = Session::new(problem, plan)?;
        let mut values = Vec::new();
        let mut steps = Vec::new();
        let mut sample = 0;
        let horizon = *plan
            .output_times()
            .last()
            .expect("validated nonempty outputs");
        while session.time() < horizon {
            let step = session.step(horizon)?;
            while sample < plan.output_times().len()
                && plan.output_times()[sample] <= step.end_time()
            {
                values.extend(step.sample(plan.output_times()[sample])?);
                sample += 1;
            }
            steps.push(step);
        }
        let history = AcceptedTimeHistory::accepted(problem.dimension(), steps, Vec::new())?;
        TimeSolution::accepted_with_history(
            problem.dimension(),
            plan.output_times().to_vec(),
            values,
            TimeExecutionReport::new(
                Self::IDENTITY,
                TimeMethod::Bdf,
                TimeEquationClass::GeneralImplicitDae,
                problem.initial_condition(),
            ),
            history,
        )
    }
}
fn admit(problem: &ImplicitDaeProblem<'_>, plan: &TimePlan) -> Result<(), Diagnostic> {
    plan.validate_for_implicit(problem)?;
    if plan.method() != TimeMethod::Bdf {
        return Err(Diagnostic::error(
            codes::INVALID_EXECUTION_CONFIG,
            "IDA requires the residual-native Bdf method",
        ));
    }
    Ok(())
}
pub(crate) fn failure(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}
