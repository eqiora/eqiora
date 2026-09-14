//! Shared native forward builder and method dispatch for smooth and root-prefix solves.
use super::*;
use diffsol::{OdeEquationsImplicitSens, OdeSolverProblem};
use eqiora_time::TimeRootSensitivityOutcome;

pub(super) enum Outcome {
    Smooth(ForwardSensitivitySolution),
    Root(TimeRootSensitivityOutcome),
}
impl Outcome {
    pub(super) fn smooth(self) -> Result<ForwardSensitivitySolution, Diagnostic> {
        match self {
            Self::Smooth(solution) => Ok(solution),
            Self::Root(_) => Err(solve_failed("unexpected rooted sensitivity outcome")),
        }
    }
    pub(super) fn root(self) -> Result<TimeRootSensitivityOutcome, Diagnostic> {
        match self {
            Self::Root(solution) => Ok(solution),
            Self::Smooth(_) => Err(solve_failed("missing rooted sensitivity outcome")),
        }
    }
}

pub(super) fn solve(
    problem: &ForwardSensitivityProblem<'_>,
    plan: &TimePlan,
    sensitivity_plan: &ForwardSensitivityPlan,
    roots: Option<&RegisteredRootProblem<'_>>,
) -> Result<Outcome, Diagnostic> {
    let failures = CallbackFailures::default();
    let rhs_failures = failures.clone();
    let jacobian_failures = failures.clone();
    let parameter_failures = failures.clone();
    let initial_parameter_failures = failures.clone();
    let system = problem.system();
    let initial_state = problem.primal().initial_state();
    let builder = OdeBuilder::<NalgebraMat<f64>>::new()
        .t0(plan.start_time())
        .h0(plan.initial_step())
        .rtol(plan.relative_tolerance())
        .atol(plan.absolute_tolerances().iter().copied())
        .sens_rtol(sensitivity_plan.relative_tolerance())
        .sens_atol(
            sensitivity_plan.absolute_tolerances()[..problem.primal().dimension()]
                .iter()
                .copied(),
        )
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
        );

    if let Some(roots) = roots {
        let root_failures = failures.clone();
        let jacobian_failures = failures.clone();
        let parameter_failures = failures.clone();
        let ode = builder
            .root_sens_implicit(
                move |state, _parameters, time, output| {
                    evaluate_roots(roots.functions(), &root_failures, time, state, output)
                },
                move |_state, _parameters, _time, _direction, output| {
                    reject_root_derivative(&jacobian_failures, output)
                },
                move |_state, _parameters, _time, _direction, output| {
                    reject_root_derivative(&parameter_failures, output)
                },
                roots.functions().count(),
            )
            .build()
            .map_err(|error| {
                map_failure(&failures, "construct Diffsol rooted forward problem", error)
            })?;
        integrate(ode, problem, plan, sensitivity_plan, Some(roots), &failures)
    } else {
        let ode = builder
            .build()
            .map_err(|error| map_failure(&failures, "construct Diffsol forward problem", error))?;
        integrate(ode, problem, plan, sensitivity_plan, None, &failures)
    }
}

fn integrate<E>(
    mut ode: OdeSolverProblem<E>,
    problem: &ForwardSensitivityProblem<'_>,
    plan: &TimePlan,
    sensitivity_plan: &ForwardSensitivityPlan,
    roots: Option<&RegisteredRootProblem<'_>>,
    failures: &CallbackFailures,
) -> Result<Outcome, Diagnostic>
where
    E: OdeEquationsImplicitSens<
            T = f64,
            V = NalgebraVec<f64>,
            M = NalgebraMat<f64>,
            C = diffsol::NalgebraContext,
        >,
{
    // The builder accepts only a shared state column; the native problem owns
    // independent per-parameter vectors used by both adaptive methods.
    ode.sens_atol = Some(
        sensitivity_plan
            .absolute_tolerances()
            .chunks_exact(problem.primal().dimension())
            .map(|column| {
                let mut tolerance = ode.atol.clone();
                copy_initial_state(column, &mut tolerance);
                tolerance
            })
            .collect(),
    );
    match plan.method() {
        TimeMethod::Tsitouras45 => {
            let mut solver = ode.tsit45_sens().map_err(|error| {
                map_failure(
                    failures,
                    "initialize Diffsol Tsitouras45 sensitivities",
                    error,
                )
            })?;
            capture(&mut solver, problem, plan, roots, failures)
        }
        TimeMethod::Bdf => {
            let mut solver = ode.bdf_sens::<NalgebraLU<f64>>().map_err(|error| {
                map_failure(failures, "initialize Diffsol BDF sensitivities", error)
            })?;
            capture(&mut solver, problem, plan, roots, failures)
        }
        TimeMethod::ImplicitEuler => {
            unreachable!("Diffsol admission rejects reference implicit Euler")
        }
    }
}

fn capture<'a, E, S>(
    solver: &mut S,
    problem: &ForwardSensitivityProblem<'_>,
    plan: &TimePlan,
    roots: Option<&RegisteredRootProblem<'_>>,
    failures: &CallbackFailures,
) -> Result<Outcome, Diagnostic>
where
    E: diffsol::OdeEquations<T = f64, V = NalgebraVec<f64>, M = NalgebraMat<f64>> + 'a,
    S: OdeSolverMethod<'a, E>,
{
    if let Some(roots) = roots {
        history::capture_until_root_sensitivities(solver, problem, plan, roots, failures)
            .map(Outcome::Root)
    } else {
        history::capture(
            solver,
            problem.primal(),
            plan,
            problem.parameter_dimension(),
            failures,
        )?
        .sensitivities(problem.parameter_dimension())
        .map(Outcome::Smooth)
    }
}

// Diffsol's sensitivity constructor requires these root traits even when no
// reset is configured. Our accepted-prefix loop must never invoke them.
fn reject_root_derivative(failures: &CallbackFailures, output: &mut NalgebraVec<f64>) {
    poison_callback(
        unsupported(
            "root derivatives belong to the canonical event owner; native root-prefix integration must not request them",
        ),
        failures,
        output,
    );
}
