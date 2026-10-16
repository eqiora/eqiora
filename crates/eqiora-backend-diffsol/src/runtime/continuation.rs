//! One explicit builder for ordinary, rooted, and interrupted smooth segments.
use super::*;
use diffsol::{OdeEquationsImplicit, OdeSolverProblem, OdeSolverState};
use eqiora_time::{TimeCheckpoint, TimeHistoryStep, TimeRootOutcome};
use std::ops::ControlFlow;
use std::sync::{Arc, atomic::AtomicBool, mpsc::SyncSender};

pub(super) struct Control<'a> {
    pub(super) checkpoint: Option<&'a TimeCheckpoint>,
    pub(super) cancellation: &'a Arc<AtomicBool>,
    pub(super) progress: Option<&'a SyncSender<TimeHistoryStep>>,
}

impl DiffsolTimeBackend {
    /// Continue an explicit adaptive segment, retaining its last accepted boundary.
    ///
    /// Cancellation is checked between model operator evaluations and before a
    /// trial is committed. Progress uses nonblocking sends to a bounded channel;
    /// a full or disconnected observer cannot change numerical execution.
    /// `Break` is an unfinished checkpoint, never an accepted solution. The
    /// current exact profile is Tsitouras45 with Diffsol's default I-controller.
    /// Canonical Run admission must bind Model identity and committed reset history.
    /// # Errors
    /// Rejects incompatible checkpoints, unsupported profiles, and model failures.
    pub fn advance(
        &self,
        problem: &TimeProblem<'_>,
        plan: &TimePlan,
        roots: Option<&RegisteredRootProblem<'_>>,
        checkpoint: Option<&TimeCheckpoint>,
        cancellation: &Arc<AtomicBool>,
        progress: Option<&SyncSender<TimeHistoryStep>>,
    ) -> Result<ControlFlow<TimeCheckpoint, TimeRootOutcome>, Diagnostic> {
        plan.validate_for(problem)?;
        if roots.is_some_and(|roots| roots.proof().root_count() == 0) {
            return Err(invalid_root_request(
                "continuation requires nonempty registered roots",
            ));
        }
        if plan.method() != TimeMethod::Tsitouras45
            || problem.equation_class() != TimeEquationClass::ExplicitOde
            || problem.initial_condition() != eqiora_time::InitialConditionPolicy::Provided
        {
            return Err(unsupported(
                "exact continuation currently requires explicit Tsitouras45 with provided initial data",
            ));
        }
        if let Some(checkpoint) = checkpoint {
            checkpoint.validate_for(
                problem,
                plan,
                Self::IDENTITY,
                roots.map(|roots| roots.registration()),
            )?;
        }
        solve(
            problem,
            plan,
            roots,
            Some(Control {
                checkpoint,
                cancellation,
                progress,
            }),
        )
    }
}

pub(super) fn solve(
    problem: &TimeProblem<'_>,
    plan: &TimePlan,
    roots: Option<&RegisteredRootProblem<'_>>,
    control: Option<Control<'_>>,
) -> Result<ControlFlow<TimeCheckpoint, TimeRootOutcome>, Diagnostic> {
    let failures = CallbackFailures::default();
    let rhs_failures = failures.clone();
    let jacobian_failures = failures.clone();
    let system = problem.system();
    let initial_state = problem.initial_state();
    let builder = OdeBuilder::<NalgebraMat<f64>>::new()
        .t0(plan.start_time())
        .h0(plan.initial_step())
        .rtol(plan.relative_tolerance())
        .atol(plan.absolute_tolerances().iter().copied())
        .use_coloring(false)
        .rhs_implicit(
            move |state, _parameters, time, output| {
                evaluate_rhs(system, &rhs_failures, time, state, output)
            },
            move |state, _parameters, time, direction, output| {
                evaluate_rhs_jvp(system, &jacobian_failures, time, state, direction, output)
            },
        )
        .init(
            move |_parameters, _time, output| copy_initial_state(initial_state, output),
            problem.dimension(),
        );
    if let Some(roots) = roots {
        let root_failures = failures.clone();
        let ode = builder
            .root(
                move |state, _parameters, time, output| {
                    evaluate_roots(roots.functions(), &root_failures, time, state, output)
                },
                roots.functions().count(),
            )
            .build()
            .map_err(|error| map_failure(&failures, "construct Diffsol root problem", error))?;
        integrate(ode, problem, plan, Some(roots), control.as_ref(), &failures)
    } else {
        let ode = builder
            .build()
            .map_err(|error| map_failure(&failures, "construct Diffsol ODE problem", error))?;
        integrate(ode, problem, plan, None, control.as_ref(), &failures)
    }
}

fn integrate<E>(
    ode: OdeSolverProblem<E>,
    problem: &TimeProblem<'_>,
    plan: &TimePlan,
    roots: Option<&RegisteredRootProblem<'_>>,
    control: Option<&Control<'_>>,
    failures: &CallbackFailures,
) -> Result<ControlFlow<TimeCheckpoint, TimeRootOutcome>, Diagnostic>
where
    E: OdeEquationsImplicit<
            T = f64,
            V = NalgebraVec<f64>,
            M = NalgebraMat<f64>,
            C = diffsol::NalgebraContext,
        >,
{
    match plan.method() {
        TimeMethod::Tsitouras45 => {
            let mut solver = ode
                .tsit45()
                .map_err(|error| map_failure(failures, "initialize Diffsol Tsitouras45", error))?;
            failures.check()?;
            let controlled = if let Some(control) = control {
                // This profile has no previous-error controller term. Fail closed
                // if provider defaults change rather than silently losing history.
                if ode.ode_options.pi_control_proportional != 0.0 {
                    return Err(unsupported(
                        "Diffsol PI-controller history is not exported for exact continuation",
                    ));
                }
                let checkpoint = if let Some(saved) = control.checkpoint {
                    let mut native = solver.checkpoint();
                    let state = native.as_mut();
                    *state.t = saved.time();
                    *state.h = saved.next_step();
                    copy_initial_state(saved.state(), state.y);
                    copy_initial_state(saved.derivative(), state.dy);
                    solver.set_state(native);
                    saved.clone()
                } else {
                    TimeCheckpoint::initial(
                        problem,
                        plan,
                        DiffsolTimeBackend::IDENTITY,
                        roots.map(|roots| roots.registration()),
                        solver.state().h,
                        collect_vector(solver.state().dy),
                    )?
                };
                failures.enable_cancellation(Arc::clone(control.cancellation));
                Some((checkpoint, control))
            } else {
                None
            };
            history::capture_controlled(&mut solver, problem, plan, roots, failures, controlled)
        }
        TimeMethod::Bdf => {
            let mut solver = ode
                .bdf::<NalgebraLU<f64>>()
                .map_err(|error| map_failure(failures, "initialize Diffsol BDF", error))?;
            history::capture_controlled(&mut solver, problem, plan, roots, failures, None)
        }
        TimeMethod::ImplicitEuler | TimeMethod::ImplicitMidpoint => {
            unreachable!("Diffsol admission rejects fixed-step methods")
        }
    }
}
