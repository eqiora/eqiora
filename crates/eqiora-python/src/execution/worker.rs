use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Instant;

use eqiora::Diagnostic;
use eqiora::backends::diffsol::DiffsolTimeBackend;
use eqiora::backends::faer::{FAER_SOLVER_PROVIDER, FaerLinearSolver};
use eqiora::backends::sundials::IdaTimeBackend;
use eqiora::diagnostic::codes;
use eqiora::solver::{
    LinearSolverBackend, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, SolverProvider,
};
use eqiora_numerics::{
    CommonResult, CommonSpatialPlan, CommonSteadyStokesPlan, CommonTimeRunRequest,
    CommonTrajectory, CommonTransientRunRequest,
};

use super::evidence::PyRunProgress;
use super::{NativeRunCancellation, NativeRunOutput, RunFailure, RunShared, RunTerminal};
use crate::error::catch_native_panic;

#[derive(Debug)]
pub(super) enum NativeRunJob {
    Eigen(Box<eqiora_numerics::CommonEigenPlan>),
    Algebraic(
        Box<eqiora_numerics::CommonAlgebraicPlan>,
        eqiora_numerics::CommonAlgebraicState,
    ),
    Spatial(Box<CommonSpatialPlan>, Option<eqiora_numerics::CommonState>),
    SteadyStokes(Box<CommonSteadyStokesPlan>),
    Transient(Box<CommonTransientRunRequest>),
    Time(
        Box<CommonTimeRunRequest>,
        Option<Box<eqiora_numerics::CommonTimeCheckpoint>>,
    ),
}

impl NativeRunJob {
    fn family(&self) -> &'static str {
        match self {
            Self::Eigen(_) => "eigen",
            Self::Algebraic(..) => "algebraic",
            Self::Spatial(..) => "spatial",
            Self::SteadyStokes(..) => "steady_stokes",
            Self::Transient(..) => "transient_flow",
            Self::Time(..) => "time",
        }
    }
}

fn run_phase(family: &str) -> tracing::Span {
    tracing::span!(target: crate::profile::TELEMETRY_TARGET, tracing::Level::INFO, "eqiora_phase", phase = "run", family)
}

fn setup_phase(role: &'static str) -> tracing::Span {
    tracing::span!(target: crate::profile::TELEMETRY_TARGET, tracing::Level::INFO, "eqiora_phase", phase = "setup", role)
}

fn solve_phase(index: usize) -> tracing::Span {
    tracing::span!(target: crate::profile::TELEMETRY_TARGET, tracing::Level::INFO, "eqiora_phase", phase = "solve", solve = index)
}

fn postprocess_phase() -> tracing::Span {
    tracing::span!(target: crate::profile::TELEMETRY_TARGET, tracing::Level::INFO, "eqiora_phase", phase = "postprocess")
}

enum NativeWorkerOutcome {
    Completed(NativeRunOutput),
    Cancelled(NativeRunCancellation),
}

pub(crate) fn resolved_linear_backend(
    provider: SolverProvider,
) -> Result<&'static dyn LinearSolverBackend, Vec<Diagnostic>> {
    let _setup = setup_phase("backend_resolution").entered();
    if provider == FAER_SOLVER_PROVIDER {
        Ok(&FaerLinearSolver)
    } else if provider == REFERENCE_SOLVER_PROVIDER {
        Ok(&REFERENCE_LINEAR_SOLVER)
    } else {
        Err(vec![Diagnostic::error(
            codes::INVALID_REALIZATION,
            "execution rejected a solver provider outside the resolved common Plan",
        )])
    }
}

fn execute_job(
    job: NativeRunJob,
    shared: &Arc<RunShared>,
) -> Result<NativeWorkerOutcome, Vec<Diagnostic>> {
    let family = job.family();
    let _run = run_phase(family).entered();
    match job {
        NativeRunJob::Eigen(plan) => {
            let backend = resolved_linear_backend(plan.solver_provider())?;
            let result = plan.run_result(backend).map_err(|d| vec![d])?;
            Ok(NativeWorkerOutcome::Completed(NativeRunOutput::Result(
                Box::new(result),
                None,
            )))
        }
        NativeRunJob::Algebraic(plan, state) => {
            let started = Instant::now();
            let backend = resolved_linear_backend(plan.solver_provider())?;
            let _solve = solve_phase(1).entered();
            let result = plan
                .run_result(&state, backend)
                .and_then(|result| result.with_elapsed_seconds(started.elapsed().as_secs_f64()))
                .map_err(|d| vec![d])?;
            Ok(NativeWorkerOutcome::Completed(NativeRunOutput::Result(
                Box::new(result),
                None,
            )))
        }
        NativeRunJob::Spatial(plan, initial) => {
            let started = Instant::now();
            let backend = resolved_linear_backend(plan.solver_provider())?;
            let _solve = solve_phase(1).entered();
            let result = plan
                .run_result(initial.as_ref(), backend)
                .and_then(|result| result.with_elapsed_seconds(started.elapsed().as_secs_f64()))
                .map_err(|diagnostic| vec![diagnostic])?;
            Ok(NativeWorkerOutcome::Completed(NativeRunOutput::Result(
                Box::new(result),
                None,
            )))
        }
        NativeRunJob::SteadyStokes(plan) => {
            let started = Instant::now();
            let backend = resolved_linear_backend(plan.solver_provider())?;
            let _solve = solve_phase(1).entered();
            let result = plan
                .run_result(backend)
                .map_err(|diagnostic| vec![diagnostic])?;
            let result = result
                .with_elapsed_seconds(started.elapsed().as_secs_f64())
                .map_err(|diagnostic| vec![diagnostic])?;
            Ok(NativeWorkerOutcome::Completed(NativeRunOutput::Result(
                Box::new(result),
                None,
            )))
        }
        NativeRunJob::Transient(request) => {
            let started = Instant::now();
            let backend = resolved_linear_backend(
                request
                    .plan()
                    .linear_solver_provider()
                    .expect("transient linear solver"),
            )?;
            let _solve = solve_phase(1).entered();
            if request.plan().ode_temporal().is_some() {
                let trajectory = request
                    .run(
                        backend,
                        |problem, plan| DiffsolTimeBackend::new().solve(problem, plan),
                        |problem, plan| IdaTimeBackend::new().solve(problem, plan),
                    )
                    .map_err(|diagnostic| vec![diagnostic])?;
                shared.publish_progress(PyRunProgress {
                    accepted_steps: trajectory
                        .spatial_time_history()
                        .expect("controlled spatial history")
                        .steps()
                        .len(),
                    maximum_steps: None,
                    model_time_bits: request.until_s().to_bits(),
                });
                let _postprocess = postprocess_phase().entered();
                let result =
                    CommonResult::accept_trajectory(started.elapsed().as_secs_f64(), trajectory)
                        .map_err(|diagnostic| vec![diagnostic])?;
                return Ok(NativeWorkerOutcome::Completed(NativeRunOutput::Result(
                    Box::new(result),
                    None,
                )));
            }
            let maximum_steps = request
                .accepted_steps()
                .map_err(|diagnostic| vec![diagnostic])?
                .get();
            let outcome = request
                .advance_accepted_actions(backend, |accepted_steps, state| {
                    if accepted_steps > 0 {
                        shared.publish_progress(PyRunProgress {
                            accepted_steps,
                            maximum_steps: Some(maximum_steps),
                            model_time_bits: state
                                .time_s()
                                .expect("admitted transient Run")
                                .to_bits(),
                        });
                    }
                    shared.cancellation_requested()
                })
                .map_err(|diagnostic| vec![diagnostic])?;
            match outcome {
                ControlFlow::Break((accepted_steps, state)) => Ok(NativeWorkerOutcome::Cancelled(
                    NativeRunCancellation::CommonTransient {
                        accepted_steps,
                        maximum_steps,
                        model_time_s: state.time_s().expect("admitted transient Run"),
                        request_identity: request.identity().to_owned(),
                        state,
                    },
                )),
                ControlFlow::Continue(states) => {
                    let _postprocess = postprocess_phase().entered();
                    let trajectory = CommonTrajectory::accept_spatial_transient(*request, states)
                        .map_err(|diagnostic| vec![diagnostic])?;
                    let result = CommonResult::accept_trajectory(
                        started.elapsed().as_secs_f64(),
                        trajectory,
                    )
                    .map_err(|diagnostic| vec![diagnostic])?;
                    Ok(NativeWorkerOutcome::Completed(NativeRunOutput::Result(
                        Box::new(result),
                        None,
                    )))
                }
            }
        }
        NativeRunJob::Time(request, checkpoint) => {
            let started = Instant::now();
            let sensitivity_plan = {
                let _setup = setup_phase("sensitivity_preparation").entered();
                request
                    .plan()
                    .as_ode()
                    .and_then(|plan| plan.forward_sensitivity_plan())
                    .cloned()
            };
            let (trajectory, sensitivity) = {
                let _solve = solve_phase(1).entered();
                if request.plan().as_ode().is_none() {
                    match request
                        .advance_process(checkpoint.as_deref(), |state, accepted_steps| {
                            shared.publish_progress(PyRunProgress {
                                accepted_steps,
                                maximum_steps: None,
                                model_time_bits: state.time_s().to_bits(),
                            });
                            if shared.cancellation_requested() {
                                ControlFlow::Break(())
                            } else {
                                ControlFlow::Continue(())
                            }
                        })
                        .map_err(|d| vec![d])?
                    {
                        ControlFlow::Break(checkpoint) => {
                            return Ok(NativeWorkerOutcome::Cancelled(
                                NativeRunCancellation::Time(Box::new(checkpoint)),
                            ));
                        }
                        ControlFlow::Continue(trajectory) => Ok((trajectory, None)),
                    }
                } else if let Some(sensitivity_plan) = sensitivity_plan {
                    if request
                        .plan()
                        .as_ode()
                        .is_some_and(|plan| plan.event_policy().is_some())
                    {
                        request.run_with_event_forward_sensitivities(|problem, roots, plan| {
                            DiffsolTimeBackend::new().solve_until_root_forward_sensitivities(
                                problem,
                                roots,
                                plan,
                                &sensitivity_plan,
                            )
                        })
                    } else {
                        request
                            .forward_sensitivity_problem()
                            .and_then(|problem| {
                                DiffsolTimeBackend::new().solve_forward_sensitivities(
                                    &problem,
                                    request.time_plan()?,
                                    &sensitivity_plan,
                                )
                            })
                            .and_then(|solution| {
                                CommonTrajectory::accept_ode_forward_sensitivities(
                                    *request, solution,
                                )
                            })
                    }
                    .map(|(trajectory, sensitivity)| (trajectory, Some(sensitivity)))
                } else if request.require_continuation().is_ok() {
                    let outcome = request
                        .advance(
                            checkpoint.as_deref(),
                            |problem, roots, plan, saved, prefix_steps| {
                                // Readers consume bounded native observations directly; the
                                // solver never waits for a Python reader or invokes user callbacks.
                                let sender = shared.observe_progress(prefix_steps);
                                DiffsolTimeBackend::new().advance(
                                    problem,
                                    plan,
                                    roots,
                                    saved,
                                    &shared.cancellation_requested,
                                    Some(&sender),
                                )
                            },
                        )
                        .map_err(|d| vec![d])?;
                    match outcome {
                        ControlFlow::Break(checkpoint) => {
                            let state = checkpoint.state().map_err(|d| vec![d])?;
                            shared.publish_progress(PyRunProgress {
                                accepted_steps: checkpoint.accepted_steps(),
                                maximum_steps: None,
                                model_time_bits: state.time_s().to_bits(),
                            });
                            return Ok(NativeWorkerOutcome::Cancelled(
                                NativeRunCancellation::Time(Box::new(checkpoint)),
                            ));
                        }
                        ControlFlow::Continue(trajectory) => {
                            shared.publish_progress(PyRunProgress {
                                accepted_steps: trajectory
                                    .ode_history()
                                    .expect("ODE history")
                                    .steps()
                                    .len(),
                                maximum_steps: None,
                                model_time_bits: request.until_s().to_bits(),
                            });
                            Ok((trajectory, None))
                        }
                    }
                } else {
                    if request
                        .plan()
                        .as_ode()
                        .is_some_and(|plan| plan.event_policy().is_some())
                    {
                        request.run_with_events(|problem, roots, plan| {
                            DiffsolTimeBackend::new().solve_until_root(problem, roots, plan)
                        })
                    } else {
                        let backend = request
                            .plan()
                            .linear_solver_provider()
                            .map(resolved_linear_backend)
                            .transpose()?
                            .unwrap_or(&FaerLinearSolver);
                        request.run(backend, |problem, plan| {
                            DiffsolTimeBackend::new().solve(problem, plan)
                        })
                    }
                    .map(|trajectory| (trajectory, None))
                }
                .map_err(|diagnostic| vec![diagnostic])?
            };
            let result = {
                let _postprocess = postprocess_phase().entered();
                let elapsed = started.elapsed().as_secs_f64();
                match sensitivity {
                    Some(sensitivity) => {
                        CommonResult::accept_trajectory_with_parameter_sensitivity(
                            elapsed,
                            trajectory,
                            sensitivity,
                        )
                    }
                    None => CommonResult::accept_trajectory(elapsed, trajectory),
                }
                .map_err(|diagnostic| vec![diagnostic])?
            };
            Ok(NativeWorkerOutcome::Completed(NativeRunOutput::Result(
                Box::new(result),
                None,
            )))
        }
    }
}

/// A map occurrence uses the ordinary worker synchronously: its bounded outer
/// scheduler owns the thread, while this worker owns all solver state.
pub(crate) fn run_mapped_time(
    request: &CommonTimeRunRequest,
    checkpoint: Option<&eqiora_numerics::CommonTimeCheckpoint>,
) -> Result<ControlFlow<eqiora_numerics::CommonTimeCheckpoint, CommonResult>, Vec<Diagnostic>> {
    let shared = Arc::new(RunShared::new());
    match execute_job(
        NativeRunJob::Time(Box::new(request.clone()), checkpoint.cloned().map(Box::new)),
        &shared,
    )? {
        NativeWorkerOutcome::Completed(NativeRunOutput::Result(result, _)) => {
            Ok(ControlFlow::Continue(*result))
        }
        NativeWorkerOutcome::Cancelled(NativeRunCancellation::Time(checkpoint)) => {
            Ok(ControlFlow::Break(*checkpoint))
        }
        NativeWorkerOutcome::Cancelled(_) => Err(vec![Diagnostic::error(
            codes::INTERNAL_FAILURE,
            "time worker returned a foreign cancellation family",
        )]),
    }
}

pub(super) fn run_worker(job: NativeRunJob, shared: Arc<RunShared>, profile: bool) {
    shared.mark_running();
    let outcome = if profile {
        let collector = crate::profile::ProfileCollector::default();
        let mut outcome = collector.capture(|| catch_native_panic(|| execute_job(job, &shared)));
        if let Ok(Ok(NativeWorkerOutcome::Completed(output))) = &mut outcome {
            output.attach_profile(collector.finish());
        }
        outcome
    } else {
        catch_native_panic(|| execute_job(job, &shared))
    };
    match outcome {
        Ok(Ok(NativeWorkerOutcome::Completed(result))) => {
            shared.finish(RunTerminal::Completed(Some(result)));
        }
        Ok(Ok(NativeWorkerOutcome::Cancelled(cancellation))) => {
            shared.finish(RunTerminal::Cancelled(cancellation));
        }
        Ok(Err(diagnostics)) => {
            shared.finish(RunTerminal::Failed(RunFailure::Execution(diagnostics)));
        }
        Err(diagnostic) => {
            shared.finish(RunTerminal::Failed(RunFailure::Internal(vec![diagnostic])));
        }
    }
}
