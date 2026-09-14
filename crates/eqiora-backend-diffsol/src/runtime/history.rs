//! Copy accepted native stencils while their dense output remains available.
use super::*;
use diffsol::OdeEquations;
use eqiora_time::{AcceptedTimeHistory, TimeHistoryStep};

pub(super) struct CapturedSolution {
    primal: TimeSolution,
    sensitivities: Vec<f64>,
    sensitivity_history: Option<AcceptedTimeHistory>,
}

impl CapturedSolution {
    pub(super) fn primal(self) -> TimeSolution {
        self.primal
    }
    pub(super) fn sensitivities(
        self,
        count: usize,
    ) -> Result<ForwardSensitivitySolution, Diagnostic> {
        ForwardSensitivitySolution::accepted_with_history(
            self.primal,
            count,
            self.sensitivities,
            self.sensitivity_history
                .ok_or_else(|| solve_failed("missing native sensitivity history"))?,
        )
    }
}

struct CapturedSegment {
    history: AcceptedTimeHistory,
    times: Vec<f64>,
    values: Vec<f64>,
    sensitivities: Vec<f64>,
    sensitivity_history: Option<AcceptedTimeHistory>,
    proposal: Option<RootProposal>,
}

pub(super) fn capture<'a, E, S>(
    solver: &mut S,
    problem: &TimeProblem<'_>,
    plan: &TimePlan,
    parameters: usize,
    failures: &CallbackFailures,
) -> Result<CapturedSolution, Diagnostic>
where
    E: OdeEquations<T = f64, V = NalgebraVec<f64>, M = NalgebraMat<f64>> + 'a,
    S: OdeSolverMethod<'a, E>,
{
    let captured = capture_segment(solver, problem, plan, parameters, None, failures)?;
    let primal = TimeSolution::accepted_with_history(
        problem.dimension(),
        captured.times,
        captured.values,
        report(problem, plan),
        captured.history,
    )?;
    Ok(CapturedSolution {
        primal,
        sensitivities: captured.sensitivities,
        sensitivity_history: captured.sensitivity_history,
    })
}

pub(super) fn capture_until_root<'a, E, S>(
    solver: &mut S,
    problem: &TimeProblem<'_>,
    plan: &TimePlan,
    roots: &RegisteredRootProblem<'_>,
    failures: &CallbackFailures,
) -> Result<eqiora_time::TimeRootOutcome, Diagnostic>
where
    E: OdeEquations<T = f64, V = NalgebraVec<f64>, M = NalgebraMat<f64>> + 'a,
    S: OdeSolverMethod<'a, E>,
{
    let captured = capture_segment(solver, problem, plan, 0, Some(roots), failures)?;
    accept_root(captured, problem, plan)
}

pub(super) fn capture_until_root_sensitivities<'a, E, S>(
    solver: &mut S,
    problem: &ForwardSensitivityProblem<'_>,
    plan: &TimePlan,
    roots: &RegisteredRootProblem<'_>,
    failures: &CallbackFailures,
) -> Result<eqiora_time::TimeRootSensitivityOutcome, Diagnostic>
where
    E: OdeEquations<T = f64, V = NalgebraVec<f64>, M = NalgebraMat<f64>> + 'a,
    S: OdeSolverMethod<'a, E>,
{
    let mut captured = capture_segment(
        solver,
        problem.primal(),
        plan,
        problem.parameter_dimension(),
        Some(roots),
        failures,
    )?;
    let sensitivity = captured
        .sensitivity_history
        .take()
        .ok_or_else(|| solve_failed("missing native root sensitivity history"))?;
    let primal = accept_root(captured, problem.primal(), plan)?;
    eqiora_time::TimeRootSensitivityOutcome::accepted(
        primal,
        problem.parameter_dimension(),
        sensitivity,
    )
}

fn accept_root(
    captured: CapturedSegment,
    problem: &TimeProblem<'_>,
    plan: &TimePlan,
) -> Result<eqiora_time::TimeRootOutcome, Diagnostic> {
    if let Some(proposal) = captured.proposal {
        let samples = if captured.times.is_empty() {
            None
        } else {
            Some(TimeSolution::accepted(
                problem.dimension(),
                captured.times,
                captured.values,
                report(problem, plan),
            )?)
        };
        eqiora_time::TimeRootOutcome::localized(proposal, captured.history, samples)
    } else {
        eqiora_time::TimeRootOutcome::horizon(TimeSolution::accepted_with_history(
            problem.dimension(),
            captured.times,
            captured.values,
            report(problem, plan),
            captured.history,
        )?)
    }
}

fn capture_segment<'a, E, S>(
    solver: &mut S,
    problem: &TimeProblem<'_>,
    plan: &TimePlan,
    parameters: usize,
    roots: Option<&RegisteredRootProblem<'_>>,
    failures: &CallbackFailures,
) -> Result<CapturedSegment, Diagnostic>
where
    E: OdeEquations<T = f64, V = NalgebraVec<f64>, M = NalgebraMat<f64>> + 'a,
    S: OdeSolverMethod<'a, E>,
{
    let dimension = problem.dimension();
    let times = plan.output_times();
    let final_time = *times.last().expect("validated output times");
    solver
        .set_stop_time(final_time)
        .map_err(|error| map_failure(failures, "set Diffsol history horizon", error))?;
    if problem.initial_condition() == eqiora_time::InitialConditionPolicy::Provided
        && collect_vector(solver.state().y) != problem.initial_state()
    {
        return Err(solve_failed(
            "native initialization changed a provided initial state",
        ));
    }
    let mut steps = Vec::new();
    let mut sensitivity_steps = Vec::new();
    let mut values = Vec::with_capacity(dimension * times.len());
    let mut sensitivities = vec![0.0; parameters * dimension * times.len()];
    let mut sample = 0;
    let mut proposal = None;
    loop {
        let start_time = solver.state().t;
        let start_state = collect_vector(solver.state().y);
        let start_sensitivity = flatten(solver.state().s);
        let stop = solver
            .step()
            .map_err(|error| map_failure(failures, "advance Diffsol accepted history", error))?;
        let end_time = if let OdeSolverStopReason::RootFound(time, index) = stop {
            let roots =
                roots.ok_or_else(|| solve_failed("ordinary history cannot accept a root"))?;
            if time <= start_time {
                return Err(solve_failed(
                    "root at the current step start has no positive smooth prefix",
                ));
            }
            let state = solver
                .interpolate(time)
                .map_err(|error| map_failure(failures, "capture native root state", error))?;
            proposal = Some(RootProposal::accepted(
                roots.registration(),
                time,
                index,
                roots.functions().count(),
                collect_vector(&state),
                dimension,
                report(problem, plan),
            )?);
            time
        } else {
            solver.state().t
        };
        let midpoint = start_time + (end_time - start_time) * 0.5;
        let middle = solver
            .interpolate(midpoint)
            .map_err(|error| map_failure(failures, "capture native Diffsol midpoint", error))?;
        steps.push(TimeHistoryStep::accepted(
            start_time,
            end_time,
            start_state,
            collect_vector(&middle),
            proposal.as_ref().map_or_else(
                || collect_vector(solver.state().y),
                |proposal| proposal.state().to_vec(),
            ),
        )?);
        if parameters > 0 {
            let middle = solver.interpolate_sens(midpoint).map_err(|error| {
                map_failure(
                    failures,
                    "capture native Diffsol sensitivity midpoint",
                    error,
                )
            })?;
            let end_sensitivity = if proposal.is_some() {
                solver.interpolate_sens(end_time).map_err(|error| {
                    map_failure(failures, "capture native root sensitivities", error)
                })?
            } else {
                solver.state().s.to_vec()
            };
            sensitivity_steps.push(TimeHistoryStep::accepted(
                start_time,
                end_time,
                start_sensitivity,
                flatten(&middle),
                flatten(&end_sensitivity),
            )?);
        }
        while sample < times.len()
            && (times[sample] < end_time || (proposal.is_none() && times[sample] == end_time))
        {
            let step = steps.last().expect("captured native step");
            if let Some(state) = stencil_state(step, times[sample]) {
                values.extend_from_slice(state);
            } else {
                let state = solver.interpolate(times[sample]).map_err(|error| {
                    map_failure(failures, "sample native Diffsol history", error)
                })?;
                values.extend(collect_vector(&state));
            }
            if parameters > 0 {
                let step = sensitivity_steps.last().expect("captured sensitivity step");
                let interpolated;
                let state = if let Some(state) = stencil_state(step, times[sample]) {
                    state
                } else {
                    let native = solver.interpolate_sens(times[sample]).map_err(|error| {
                        map_failure(failures, "sample native Diffsol sensitivities", error)
                    })?;
                    interpolated = flatten(&native);
                    &interpolated
                };
                if state.len() != parameters * dimension {
                    return Err(solve_failed("unexpected native sensitivity count"));
                }
                for (parameter, state) in state.chunks_exact(dimension).enumerate() {
                    let start = (parameter * times.len() + sample) * dimension;
                    sensitivities[start..start + dimension].copy_from_slice(state);
                }
            }
            sample += 1;
        }
        if proposal.is_some() || stop == OdeSolverStopReason::TstopReached {
            break;
        }
    }
    if (proposal.is_none() && sample != times.len()) || steps[0].start_time() != plan.start_time() {
        return Err(solve_failed(
            "native accepted history does not span the requested solve",
        ));
    }
    if let Some(failure) = failures.take() {
        return Err(failure);
    }
    let history = AcceptedTimeHistory::accepted(dimension, steps, Vec::new())?;
    let sensitivity_history = if parameters == 0 {
        None
    } else {
        Some(AcceptedTimeHistory::accepted(
            parameters * dimension,
            sensitivity_steps,
            Vec::new(),
        )?)
    };
    Ok(CapturedSegment {
        history,
        times: times[..sample].to_vec(),
        values,
        sensitivities,
        sensitivity_history,
        proposal,
    })
}

fn flatten(states: &[NalgebraVec<f64>]) -> Vec<f64> {
    states.iter().flat_map(collect_vector).collect()
}

fn report(problem: &TimeProblem<'_>, plan: &TimePlan) -> TimeExecutionReport {
    TimeExecutionReport::new(
        DIFFSOL_TIME_BACKEND,
        plan.method(),
        problem.equation_class(),
        problem.initial_condition(),
    )
}

// Native endpoint state and dense-output evaluation can differ by a rounding bit.
// A requested retained timestamp has one authoritative value in both projections.
fn stencil_state(step: &TimeHistoryStep, time: f64) -> Option<&[f64]> {
    if time == step.start_time() {
        Some(step.start_state())
    } else if time == step.end_time() {
        Some(step.end_state())
    } else if time == step.start_time() + (step.end_time() - step.start_time()) * 0.5 {
        Some(step.midpoint_state())
    } else {
        None
    }
}
