use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Instant;

use eqiora::Diagnostic;
use eqiora::backends::diffsol::DiffsolTimeBackend;
use eqiora::backends::faer::{FAER_SOLVER_PROVIDER, FaerLinearSolver};
use eqiora::diagnostic::codes;
use eqiora::solver::{
    LinearSolverBackend, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, SolverProvider,
};
use eqiora_numerics::{
    CommonElasticityPlan, CommonFsiRunRequest, CommonOdeRunRequest, CommonResult, CommonScalarPlan,
    CommonSteadyStokesPlan, CommonTrajectory, CommonTransientRunRequest,
};

use super::evidence::PyCommonTransientRunProgress;
use super::{
    NativeRunCancellation, NativeRunOutput, NativeRunProgress, RunFailure, RunShared, RunTerminal,
};
use crate::error::catch_native_panic;

#[derive(Debug)]
pub(super) enum NativeRunJob {
    Algebraic(
        Box<eqiora_numerics::CommonAlgebraicPlan>,
        eqiora_numerics::CommonAlgebraicState,
    ),
    Scalar(Box<CommonScalarPlan>),
    Elasticity(Box<CommonElasticityPlan>),
    SteadyStokes(Box<CommonSteadyStokesPlan>),
    Transient(Box<CommonTransientRunRequest>),
    Fsi(Box<CommonFsiRunRequest>),
    Ode(Box<CommonOdeRunRequest>),
}

impl NativeRunJob {
    fn family(&self) -> &'static str {
        match self {
            Self::Algebraic(..) => "algebraic",
            Self::Scalar(..) => "scalar",
            Self::Elasticity(..) => "elasticity",
            Self::SteadyStokes(..) => "steady_stokes",
            Self::Transient(..) => "transient_flow",
            Self::Fsi(..) => "fsi",
            Self::Ode(..) => "ode",
        }
    }
}

fn run_phase(family: &str) -> tracing::Span {
    tracing::span!(target: crate::profile::TELEMETRY_TARGET, tracing::Level::INFO, "eqiora_phase", phase = "run", family)
}

fn setup_phase() -> tracing::Span {
    tracing::span!(target: crate::profile::TELEMETRY_TARGET, tracing::Level::INFO, "eqiora_phase", phase = "setup")
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

fn resolved_linear_backend(
    provider: SolverProvider,
) -> Result<&'static dyn LinearSolverBackend, Vec<Diagnostic>> {
    let _setup = setup_phase().entered();
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
        NativeRunJob::Scalar(plan) => {
            let started = Instant::now();
            let backend = resolved_linear_backend(plan.solver_provider())?;
            let _solve = solve_phase(1).entered();
            let result = plan
                .run_result(backend)
                .and_then(|result| result.with_elapsed_seconds(started.elapsed().as_secs_f64()))
                .map_err(|diagnostic| vec![diagnostic])?;
            Ok(NativeWorkerOutcome::Completed(NativeRunOutput::Result(
                Box::new(result),
                None,
            )))
        }
        NativeRunJob::Elasticity(plan) => {
            let started = Instant::now();
            let backend = resolved_linear_backend(plan.solver_provider())?;
            let _solve = solve_phase(1).entered();
            let result = plan
                .run_result(backend)
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
            let maximum_steps = request.accepted_steps().get();
            let backend = resolved_linear_backend(request.plan().solver_provider())?;
            let outcome = request
                .advance_accepted_actions(backend, |accepted_steps, state| {
                    if accepted_steps > 0 {
                        shared.publish_progress(NativeRunProgress::CommonTransient(
                            PyCommonTransientRunProgress {
                                accepted_steps,
                                maximum_steps,
                                model_time_bits: state.time_s().to_bits(),
                            },
                        ));
                    }
                    shared.cancellation_requested()
                })
                .map_err(|diagnostic| vec![diagnostic])?;
            match outcome {
                ControlFlow::Break((accepted_steps, state)) => Ok(NativeWorkerOutcome::Cancelled(
                    NativeRunCancellation::CommonTransient {
                        accepted_steps,
                        maximum_steps,
                        model_time_s: state.time_s(),
                        request_identity: request.identity().to_owned(),
                        state,
                    },
                )),
                ControlFlow::Continue(states) => {
                    let _postprocess = postprocess_phase().entered();
                    let trajectory = CommonTrajectory::accept_transient_flow(*request, states)
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
        NativeRunJob::Fsi(request) => {
            let started = Instant::now();
            let backend = resolved_linear_backend(request.plan().solver_provider())?;
            let maximum_steps = request.accepted_steps().get();
            let outcome = request
                .advance_accepted_actions(backend, |accepted_steps, state| {
                    if accepted_steps > 0 {
                        shared.publish_progress(NativeRunProgress::CommonTransient(
                            PyCommonTransientRunProgress {
                                accepted_steps,
                                maximum_steps,
                                model_time_bits: state.time_s().to_bits(),
                            },
                        ));
                    }
                    shared.cancellation_requested()
                })
                .map_err(|diagnostic| vec![diagnostic])?;
            match outcome {
                ControlFlow::Break((accepted_steps, state)) => Ok(NativeWorkerOutcome::Cancelled(
                    NativeRunCancellation::CommonTransient {
                        accepted_steps,
                        maximum_steps,
                        model_time_s: state.time_s(),
                        request_identity: request.identity().to_owned(),
                        state,
                    },
                )),
                ControlFlow::Continue(states) => {
                    let _postprocess = postprocess_phase().entered();
                    let trajectory = CommonTrajectory::accept_fsi(*request, states)
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
        NativeRunJob::Ode(request) => {
            let started = Instant::now();
            let sensitivity_plan = {
                let _setup = setup_phase().entered();
                request.plan().forward_sensitivity_plan().cloned()
            };
            let (trajectory, sensitivity) = {
                let _solve = solve_phase(1).entered();
                if let Some(sensitivity_plan) = sensitivity_plan {
                    if request.plan().event_policy().is_some() {
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
                                    request.time_plan(),
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
                } else {
                    if request.plan().event_policy().is_some() {
                        request.run_with_events(|problem, roots, plan| {
                            DiffsolTimeBackend::new().solve_until_root(problem, roots, plan)
                        })
                    } else {
                        request
                            .problem()
                            .and_then(|problem| {
                                DiffsolTimeBackend::new().solve(&problem, request.time_plan())
                            })
                            .and_then(|solution| CommonTrajectory::accept_ode(*request, solution))
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
