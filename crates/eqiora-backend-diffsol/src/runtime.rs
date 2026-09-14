mod history;
mod sensitivities;

use std::cell::RefCell;
use std::rc::Rc;

use diffsol::{
    NalgebraLU, NalgebraMat, NalgebraVec, OdeBuilder, OdeSolverMethod, OdeSolverStopReason, Vector,
};
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, GraphPath};
use eqiora_time::{
    ForwardSensitivityPlan, ForwardSensitivityProblem, ForwardSensitivitySolution,
    RegisteredRootProblem, RootFunctions, RootProposal, TimeBackendIdentity, TimeEquationClass,
    TimeExecutionReport, TimeMethod, TimePlan, TimeProblem, TimeSolution, TimeSystem,
};

/// Stable adapter identity and exact Diffsol release represented by this build.
pub const DIFFSOL_TIME_BACKEND: TimeBackendIdentity =
    TimeBackendIdentity::new("eqiora.time.diffsol", "0.16.2");

/// Stateless Diffsol adapter for admitted first-order time problems.
#[derive(Debug, Default, Clone, Copy)]
pub struct DiffsolTimeBackend;

impl DiffsolTimeBackend {
    /// Construct the stateless adapter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Check one equation-class/method pair without constructing a problem.
    ///
    /// Tsitouras 5(4) admits only explicit ODEs. BDF admits ODEs and full- or
    /// rank-deficient mass matrices. A general implicit DAE is never rewritten
    /// to fit Diffsol's first-order mass-matrix form.
    ///
    /// # Errors
    /// Returns `EQ0807` when the pair is outside the adapter contract.
    pub fn admit(
        &self,
        equation_class: TimeEquationClass,
        method: TimeMethod,
    ) -> Result<(), Diagnostic> {
        match (equation_class, method) {
            (TimeEquationClass::ExplicitOde, TimeMethod::Tsitouras45 | TimeMethod::Bdf)
            | (TimeEquationClass::MassMatrix { .. }, TimeMethod::Bdf) => Ok(()),
            (_, TimeMethod::ImplicitEuler) => Err(unsupported(
                "ImplicitEuler is the deterministic residual reference method, not a Diffsol method",
            )),
            (TimeEquationClass::MassMatrix { .. }, TimeMethod::Tsitouras45) => Err(unsupported(
                "Tsitouras45 requires an explicit ODE; select BDF for a mass-matrix system",
            )),
            (TimeEquationClass::GeneralImplicitDae, _) => Err(unsupported(
                "Diffsol admits ODE and mass-matrix forms, not a general F(t,y,ydot)=0 residual",
            )),
        }
    }

    /// Solve one validated first-order problem at the requested output times.
    ///
    /// Internal adaptive steps are owned by Diffsol and do not replace Eqiora
    /// model time or event ordering. Returned buffers are copied into an
    /// Eqiora-owned [`TimeSolution`].
    ///
    /// # Errors
    /// Returns stable Eqiora diagnostics for admission, callback, setup, or
    /// integration failures.
    pub fn solve(
        &self,
        problem: &TimeProblem<'_>,
        plan: &TimePlan,
    ) -> Result<TimeSolution, Diagnostic> {
        plan.validate_for(problem)?;
        self.admit(problem.equation_class(), plan.method())?;
        match problem.equation_class() {
            TimeEquationClass::ExplicitOde => solve_ode(problem, plan),
            TimeEquationClass::MassMatrix { .. } => solve_mass_matrix(problem, plan),
            TimeEquationClass::GeneralImplicitDae => unreachable!(
                "TimeProblem rejects the general-implicit class and admission fails closed"
            ),
        }
    }

    /// Integrate continuous forward sensitivities from the same primal/JVP seam.
    ///
    /// Parameter effects enter as `f_p dp` and `y0_p dp` actions supplied by
    /// [`eqiora_time::ParametricTimeSystem`]. Mass-matrix problems are admitted
    /// only after that contract proves `M_p dp = 0`; no dual-number simulator
    /// or Diffsol type enters the public contract.
    ///
    /// # Errors
    /// Returns a stable diagnostic for invalid controls, unsupported
    /// equation/method pairs, callback failures, or integration failures.
    pub fn solve_forward_sensitivities(
        &self,
        problem: &ForwardSensitivityProblem<'_>,
        plan: &TimePlan,
        sensitivity_plan: &ForwardSensitivityPlan,
    ) -> Result<ForwardSensitivitySolution, Diagnostic> {
        plan.validate_for(problem.primal())?;
        sensitivity_plan.validate_for(problem)?;
        self.admit(problem.primal().equation_class(), plan.method())?;
        match problem.primal().equation_class() {
            TimeEquationClass::ExplicitOde => {
                sensitivities::solve(problem, plan, sensitivity_plan, None)?.smooth()
            }
            TimeEquationClass::MassMatrix { .. } => {
                solve_mass_matrix_forward_sensitivities(problem, plan, sensitivity_plan)
            }
            TimeEquationClass::GeneralImplicitDae => unreachable!(
                "ForwardSensitivityProblem rejects general implicit DAE and admission fails closed"
            ),
        }
    }

    /// Integrate continuous fixed-time sensitivities through the first root or horizon.
    ///
    /// The caller supplies the real parameter layout. Native interpolation retains
    /// pre-event sensitivities; event-time/reset derivatives remain canonical-owner work.
    /// # Errors
    /// Rejects non-ODE root problems and invalid controls, callbacks, or integration.
    pub fn solve_until_root_forward_sensitivities(
        &self,
        problem: &ForwardSensitivityProblem<'_>,
        roots: &RegisteredRootProblem<'_>,
        plan: &TimePlan,
        sensitivity_plan: &ForwardSensitivityPlan,
    ) -> Result<eqiora_time::TimeRootSensitivityOutcome, Diagnostic> {
        plan.validate_for(problem.primal())?;
        sensitivity_plan.validate_for(problem)?;
        self.admit(problem.primal().equation_class(), plan.method())?;
        if problem.primal().equation_class() != TimeEquationClass::ExplicitOde
            || roots.proof().root_count() == 0
        {
            return Err(unsupported(
                "root sensitivities require an explicit ODE and at least one registered root",
            ));
        }
        sensitivities::solve(problem, plan, sensitivity_plan, Some(roots))?.root()
    }

    /// Localize the first zero crossing before the plan's final output time.
    ///
    /// The result is only a numerical proposal. Root direction, simultaneous
    /// grouping, priority, and reset commit are owned by Eqiora's hybrid
    /// scheduler. A reset therefore restarts the same [`TimeProblem`] explicitly
    /// rather than invoking Diffsol's automatic-reset semantics.
    ///
    /// The first admitted slice is an explicit ODE. Native smooth history is retained
    /// through the first root or the final requested output.
    ///
    /// # Errors
    /// Returns stable admission, callback, setup, and integration diagnostics.
    pub fn solve_until_root(
        &self,
        problem: &TimeProblem<'_>,
        roots: &RegisteredRootProblem<'_>,
        plan: &TimePlan,
    ) -> Result<eqiora_time::TimeRootOutcome, Diagnostic> {
        plan.validate_for(problem)?;
        self.admit(problem.equation_class(), plan.method())?;
        if problem.equation_class() != TimeEquationClass::ExplicitOde {
            return Err(unsupported(
                "Diffsol root-proposal evidence currently admits explicit ODEs only",
            ));
        }
        if roots.proof().root_count() == 0 {
            return Err(invalid_root_request(
                "root proposal requires at least one root function",
            ));
        }
        propose_ode_root(problem, roots, plan)
    }
}

fn propose_ode_root(
    problem: &TimeProblem<'_>,
    roots: &RegisteredRootProblem<'_>,
    plan: &TimePlan,
) -> Result<eqiora_time::TimeRootOutcome, Diagnostic> {
    let failures = CallbackFailures::default();
    let rhs_failures = failures.clone();
    let jacobian_failures = failures.clone();
    let root_failures = failures.clone();
    let functions = roots.functions();
    let system = problem.system();
    let initial_state = problem.initial_state();
    let ode = OdeBuilder::<NalgebraMat<f64>>::new()
        .t0(plan.start_time())
        .h0(plan.initial_step())
        .rtol(plan.relative_tolerance())
        .atol(plan.absolute_tolerances().iter().copied())
        .use_coloring(false)
        .rhs_implicit(
            move |state, _parameters, time, output| {
                evaluate_rhs(system, &rhs_failures, time, state, output);
            },
            move |state, _parameters, time, direction, output| {
                evaluate_rhs_jvp(system, &jacobian_failures, time, state, direction, output);
            },
        )
        .root(
            move |state, _parameters, time, output| {
                evaluate_roots(functions, &root_failures, time, state, output);
            },
            functions.count(),
        )
        .init(
            move |_parameters, _time, output| copy_initial_state(initial_state, output),
            problem.dimension(),
        )
        .build()
        .map_err(|error| map_failure(&failures, "construct Diffsol root problem", error))?;
    match plan.method() {
        TimeMethod::Tsitouras45 => {
            let mut solver = ode
                .tsit45()
                .map_err(|error| map_failure(&failures, "initialize Diffsol root search", error))?;
            history::capture_until_root(&mut solver, problem, plan, roots, &failures)
        }
        TimeMethod::Bdf => {
            let mut solver = ode.bdf::<NalgebraLU<f64>>().map_err(|error| {
                map_failure(&failures, "initialize Diffsol BDF root search", error)
            })?;
            history::capture_until_root(&mut solver, problem, plan, roots, &failures)
        }
        TimeMethod::ImplicitEuler => {
            unreachable!("Diffsol admission rejects the reference implicit-Euler method")
        }
    }
}

fn solve_mass_matrix_forward_sensitivities(
    problem: &ForwardSensitivityProblem<'_>,
    plan: &TimePlan,
    sensitivity_plan: &ForwardSensitivityPlan,
) -> Result<ForwardSensitivitySolution, Diagnostic> {
    let failures = CallbackFailures::default();
    let rhs_failures = failures.clone();
    let jacobian_failures = failures.clone();
    let parameter_failures = failures.clone();
    let mass_failures = failures.clone();
    let initial_parameter_failures = failures.clone();
    let system = problem.system();
    let initial_state = problem.primal().initial_state();
    let ode = OdeBuilder::<NalgebraMat<f64>>::new()
        .t0(plan.start_time())
        .h0(plan.initial_step())
        .rtol(plan.relative_tolerance())
        .atol(plan.absolute_tolerances().iter().copied())
        .sens_rtol(sensitivity_plan.relative_tolerance())
        .sens_atol(sensitivity_plan.absolute_tolerances().iter().copied())
        .p(problem.parameters().iter().copied())
        .use_coloring(false)
        .rhs_sens_implicit(
            move |state, _parameters, time, output| {
                evaluate_rhs(system, &rhs_failures, time, state, output);
            },
            move |state, _parameters, time, direction, output| {
                evaluate_rhs_jvp(system, &jacobian_failures, time, state, direction, output);
            },
            move |state, _parameters, time, parameter_direction, output| {
                evaluate_rhs_parameter_jvp(
                    system,
                    &parameter_failures,
                    time,
                    state,
                    parameter_direction,
                    output,
                );
            },
        )
        .mass(move |direction, _parameters, time, beta, output| {
            evaluate_mass(system, &mass_failures, time, direction, beta, output);
        })
        .init_sens(
            move |_parameters, _time, output| copy_initial_state(initial_state, output),
            move |_parameters, time, parameter_direction, output| {
                evaluate_initial_parameter_jvp(
                    system,
                    &initial_parameter_failures,
                    time,
                    parameter_direction,
                    output,
                );
            },
            problem.primal().dimension(),
        )
        .build()
        .map_err(|error| {
            map_failure(
                &failures,
                "construct Diffsol mass-matrix forward-sensitivity problem",
                error,
            )
        })?;

    debug_assert_eq!(plan.method(), TimeMethod::Bdf);
    let mut solver = ode.bdf_sens::<NalgebraLU<f64>>().map_err(|error| {
        map_failure(
            &failures,
            "initialize Diffsol mass-matrix BDF sensitivities",
            error,
        )
    })?;
    history::capture(
        &mut solver,
        problem.primal(),
        plan,
        problem.parameter_dimension(),
        &failures,
    )?
    .sensitivities(problem.parameter_dimension())
}

fn solve_ode(problem: &TimeProblem<'_>, plan: &TimePlan) -> Result<TimeSolution, Diagnostic> {
    let failures = CallbackFailures::default();
    let rhs_failures = failures.clone();
    let jacobian_failures = failures.clone();
    let system = problem.system();
    let initial_state = problem.initial_state();
    let ode = OdeBuilder::<NalgebraMat<f64>>::new()
        .t0(plan.start_time())
        .h0(plan.initial_step())
        .rtol(plan.relative_tolerance())
        .atol(plan.absolute_tolerances().iter().copied())
        .use_coloring(false)
        .rhs_implicit(
            move |state, _parameters, time, output| {
                evaluate_rhs(system, &rhs_failures, time, state, output);
            },
            move |state, _parameters, time, direction, output| {
                evaluate_rhs_jvp(system, &jacobian_failures, time, state, direction, output);
            },
        )
        .init(
            move |_parameters, _time, output| copy_initial_state(initial_state, output),
            problem.dimension(),
        )
        .build()
        .map_err(|error| map_failure(&failures, "construct Diffsol ODE problem", error))?;

    match plan.method() {
        TimeMethod::Tsitouras45 => {
            let mut solver = ode
                .tsit45()
                .map_err(|error| map_failure(&failures, "initialize Diffsol Tsitouras45", error))?;
            Ok(history::capture(&mut solver, problem, plan, 0, &failures)?.primal())
        }
        TimeMethod::Bdf => {
            let mut solver = ode
                .bdf::<NalgebraLU<f64>>()
                .map_err(|error| map_failure(&failures, "initialize Diffsol BDF", error))?;
            Ok(history::capture(&mut solver, problem, plan, 0, &failures)?.primal())
        }
        TimeMethod::ImplicitEuler => {
            unreachable!("Diffsol admission rejects the reference implicit-Euler method")
        }
    }
}

fn solve_mass_matrix(
    problem: &TimeProblem<'_>,
    plan: &TimePlan,
) -> Result<TimeSolution, Diagnostic> {
    let failures = CallbackFailures::default();
    let rhs_failures = failures.clone();
    let jacobian_failures = failures.clone();
    let mass_failures = failures.clone();
    let system = problem.system();
    let initial_state = problem.initial_state();
    let ode = OdeBuilder::<NalgebraMat<f64>>::new()
        .t0(plan.start_time())
        .h0(plan.initial_step())
        .rtol(plan.relative_tolerance())
        .atol(plan.absolute_tolerances().iter().copied())
        .use_coloring(false)
        .rhs_implicit(
            move |state, _parameters, time, output| {
                evaluate_rhs(system, &rhs_failures, time, state, output);
            },
            move |state, _parameters, time, direction, output| {
                evaluate_rhs_jvp(system, &jacobian_failures, time, state, direction, output);
            },
        )
        .mass(move |direction, _parameters, time, beta, output| {
            evaluate_mass(system, &mass_failures, time, direction, beta, output);
        })
        .init(
            move |_parameters, _time, output| copy_initial_state(initial_state, output),
            problem.dimension(),
        )
        .build()
        .map_err(|error| map_failure(&failures, "construct Diffsol mass-matrix problem", error))?;

    let mut solver = ode.bdf::<NalgebraLU<f64>>().map_err(|error| {
        map_failure(&failures, "initialize consistent Diffsol BDF state", error)
    })?;
    Ok(history::capture(&mut solver, problem, plan, 0, &failures)?.primal())
}

#[derive(Clone, Default)]
struct CallbackFailures(Rc<RefCell<Option<Diagnostic>>>);

impl CallbackFailures {
    fn record(&self, diagnostic: Diagnostic) {
        let mut failure = self.0.borrow_mut();
        if failure.is_none() {
            *failure = Some(diagnostic);
        }
    }

    fn take(&self) -> Option<Diagnostic> {
        self.0.borrow_mut().take()
    }
}

fn evaluate_rhs(
    system: &dyn TimeSystem,
    failures: &CallbackFailures,
    time: f64,
    state: &NalgebraVec<f64>,
    output: &mut NalgebraVec<f64>,
) {
    let state = collect_vector(state);
    let mut result = vec![0.0; system.dimension()];
    complete_callback(
        system.rhs(time, &state, &mut result),
        result,
        failures,
        output,
    );
}

fn evaluate_rhs_jvp(
    system: &dyn TimeSystem,
    failures: &CallbackFailures,
    time: f64,
    state: &NalgebraVec<f64>,
    direction: &NalgebraVec<f64>,
    output: &mut NalgebraVec<f64>,
) {
    let state = collect_vector(state);
    let direction = collect_vector(direction);
    let mut result = vec![0.0; system.dimension()];
    complete_callback(
        system.rhs_jvp(time, &state, &direction, &mut result),
        result,
        failures,
        output,
    );
}

fn evaluate_rhs_parameter_jvp(
    system: &dyn eqiora_time::ParametricTimeSystem,
    failures: &CallbackFailures,
    time: f64,
    state: &NalgebraVec<f64>,
    parameter_direction: &NalgebraVec<f64>,
    output: &mut NalgebraVec<f64>,
) {
    let state = collect_vector(state);
    let parameter_direction = collect_vector(parameter_direction);
    let mut result = vec![0.0; system.dimension()];
    complete_callback(
        system.rhs_parameter_jvp(time, &state, &parameter_direction, &mut result),
        result,
        failures,
        output,
    );
}

fn evaluate_initial_parameter_jvp(
    system: &dyn eqiora_time::ParametricTimeSystem,
    failures: &CallbackFailures,
    time: f64,
    parameter_direction: &NalgebraVec<f64>,
    output: &mut NalgebraVec<f64>,
) {
    let parameter_direction = collect_vector(parameter_direction);
    let mut result = vec![0.0; system.dimension()];
    complete_callback(
        system.initial_parameter_jvp(time, &parameter_direction, &mut result),
        result,
        failures,
        output,
    );
}

fn evaluate_roots(
    roots: &dyn RootFunctions,
    failures: &CallbackFailures,
    time: f64,
    state: &NalgebraVec<f64>,
    output: &mut NalgebraVec<f64>,
) {
    let state = collect_vector(state);
    let mut result = vec![0.0; roots.count()];
    complete_callback(
        roots.evaluate(time, &state, &mut result),
        result,
        failures,
        output,
    );
}

fn evaluate_mass(
    system: &dyn TimeSystem,
    failures: &CallbackFailures,
    time: f64,
    direction: &NalgebraVec<f64>,
    beta: f64,
    output: &mut NalgebraVec<f64>,
) {
    let direction = collect_vector(direction);
    let mut result = vec![0.0; system.dimension()];
    match system.mass_action(time, &direction, &mut result) {
        Ok(()) if result.iter().all(|value| value.is_finite()) => {
            for (index, value) in result.into_iter().enumerate() {
                output[index] = value + beta * output[index];
            }
        }
        Ok(()) => poison_callback(
            solve_failed("time mass action produced a non-finite value"),
            failures,
            output,
        ),
        Err(diagnostic) => poison_callback(diagnostic, failures, output),
    }
}

fn complete_callback(
    evaluation: Result<(), Diagnostic>,
    result: Vec<f64>,
    failures: &CallbackFailures,
    output: &mut NalgebraVec<f64>,
) {
    match evaluation {
        Ok(()) if result.iter().all(|value| value.is_finite()) => {
            for (index, value) in result.into_iter().enumerate() {
                output[index] = value;
            }
        }
        Ok(()) => poison_callback(
            solve_failed("time operator callback produced a non-finite value"),
            failures,
            output,
        ),
        Err(diagnostic) => poison_callback(diagnostic, failures, output),
    }
}

fn poison_callback(
    diagnostic: Diagnostic,
    failures: &CallbackFailures,
    output: &mut NalgebraVec<f64>,
) {
    failures.record(diagnostic);
    for index in 0..output.len() {
        output[index] = f64::NAN;
    }
}

fn collect_vector(vector: &NalgebraVec<f64>) -> Vec<f64> {
    (0..vector.len()).map(|index| vector[index]).collect()
}

fn copy_initial_state(initial_state: &[f64], output: &mut NalgebraVec<f64>) {
    for (index, value) in initial_state.iter().copied().enumerate() {
        output[index] = value;
    }
}

fn map_failure(
    failures: &CallbackFailures,
    phase: &str,
    error: impl std::fmt::Display,
) -> Diagnostic {
    failures
        .take()
        .unwrap_or_else(|| solve_failed(format!("failed to {phase}: {error}")))
}

fn unsupported(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message).with_graph_path(GraphPath::new([
        "time",
        "backend",
        "diffsol",
        "admission",
    ]))
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
        .with_graph_path(GraphPath::new(["time", "backend", "diffsol"]))
}

fn invalid_root_request(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXECUTION_CONFIG, message)
        .with_graph_path(GraphPath::new(["time", "backend", "diffsol", "root"]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_time::MassMatrixRank;

    #[test]
    fn equation_admission_is_exact() {
        let backend = DiffsolTimeBackend::new();
        assert!(
            backend
                .admit(TimeEquationClass::ExplicitOde, TimeMethod::Tsitouras45)
                .is_ok()
        );
        assert!(
            backend
                .admit(
                    TimeEquationClass::MassMatrix {
                        rank: MassMatrixRank::RankDeficient,
                    },
                    TimeMethod::Bdf,
                )
                .is_ok()
        );
        assert_eq!(
            backend
                .admit(
                    TimeEquationClass::MassMatrix {
                        rank: MassMatrixRank::Full,
                    },
                    TimeMethod::Tsitouras45,
                )
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
        assert_eq!(
            backend
                .admit(TimeEquationClass::GeneralImplicitDae, TimeMethod::Bdf)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }
}
