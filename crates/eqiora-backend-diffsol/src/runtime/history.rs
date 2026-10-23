//! Copy accepted native stencils while their dense output remains available.
use super::continuation::Control;
use super::*;
use diffsol::{MatrixCommon, OdeEquations, Tableau};
use eqiora_time::{AcceptedTimeHistory, TimeCheckpoint, TimeDenseOutput, TimeHistoryStep};
use std::ops::ControlFlow;
use std::sync::atomic::Ordering;

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
    let captured = capture_segment(solver, problem, plan, parameters, None, failures, None)?
        .continue_value()
        .ok_or_else(|| solve_failed("uncontrolled capture interrupted"))?;
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

pub(super) fn capture_controlled<'a, E, S>(
    solver: &mut S,
    problem: &TimeProblem<'_>,
    plan: &TimePlan,
    roots: Option<&RegisteredRootProblem<'_>>,
    failures: &CallbackFailures,
    control: Option<(TimeCheckpoint, &Control<'_>)>,
) -> Result<ControlFlow<TimeCheckpoint, eqiora_time::TimeRootOutcome>, Diagnostic>
where
    E: OdeEquations<T = f64, V = NalgebraVec<f64>, M = NalgebraMat<f64>> + 'a,
    S: OdeSolverMethod<'a, E>,
{
    match capture_segment(solver, problem, plan, 0, roots, failures, control)? {
        ControlFlow::Break(checkpoint) => Ok(ControlFlow::Break(checkpoint)),
        ControlFlow::Continue(captured) => {
            accept_root(captured, problem, plan).map(ControlFlow::Continue)
        }
    }
}

pub(super) fn capture_until_root_sensitivities<'a, E, S>(
    solver: &mut S,
    problem: &ForwardSensitivityProblem<'_>,
    plan: &TimePlan,
    roots: &RegisteredRootProblem<'_>,
    failures: &CallbackFailures,
) -> Result<eqiora_time::TimeRootOutcome, Diagnostic>
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
        None,
    )?
    .continue_value()
    .ok_or_else(|| solve_failed("uncontrolled sensitivity capture interrupted"))?;
    let sensitivity = captured
        .sensitivity_history
        .take()
        .ok_or_else(|| solve_failed("missing native root sensitivity history"))?;
    let samples = std::mem::take(&mut captured.sensitivities);
    let primal = accept_root(captured, problem.primal(), plan)?;
    primal.with_sensitivities(problem.parameter_dimension(), sensitivity, samples)
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
    control: Option<(TimeCheckpoint, &Control<'_>)>,
) -> Result<ControlFlow<TimeCheckpoint, CapturedSegment>, Diagnostic>
where
    E: OdeEquations<T = f64, V = NalgebraVec<f64>, M = NalgebraMat<f64>> + 'a,
    S: OdeSolverMethod<'a, E>,
{
    failures.check()?;
    let dimension = problem.dimension();
    let times = plan.output_times();
    let final_time = *times.last().expect("validated output times");
    solver
        .set_stop_time(final_time)
        .map_err(|error| map_failure(failures, "set Diffsol history horizon", error))?;
    if problem.initial_condition() == eqiora_time::InitialConditionPolicy::Provided
        && control
            .as_ref()
            .is_none_or(|(saved, _)| saved.history().is_none())
        && collect_vector(solver.state().y) != problem.initial_state()
    {
        return Err(solve_failed(
            "native initialization changed a provided initial state",
        ));
    }
    let mut steps = control
        .as_ref()
        .and_then(|(saved, _)| saved.history())
        .map_or_else(Vec::new, |history| history.steps().to_vec());
    let mut sensitivity_steps = Vec::new();
    let mut values = control.as_ref().map_or_else(
        || Vec::with_capacity(dimension * times.len()),
        |(saved, _)| saved.samples().to_vec(),
    );
    let mut sensitivities = vec![0.0; parameters * dimension * times.len()];
    let mut sample = values.len() / dimension;
    let mut next_step = solver.state().h;
    let mut derivative = control.as_ref().map(|_| collect_vector(solver.state().dy));
    let mut proposal = None;
    let mut cancelled = false;
    loop {
        if control
            .as_ref()
            .is_some_and(|(_, control)| control.cancellation.load(Ordering::Acquire))
        {
            cancelled = true;
            break;
        }
        let start_time = solver.state().t;
        let start_state = collect_vector(solver.state().y);
        let start_sensitivity = flatten(solver.state().s);
        let stop = solver.step();
        if control
            .as_ref()
            .is_some_and(|(_, control)| control.cancellation.load(Ordering::Acquire))
        {
            cancelled = true;
            break;
        }
        let stop =
            stop.map_err(|error| map_failure(failures, "advance Diffsol accepted history", error))?;
        // A native successful return does not establish callback success. In
        // particular, a poisoned RK error norm may still admit a native step.
        // Reject the original failure before retaining any state or root.
        failures.check()?;
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
        let degree = match plan.method() {
            TimeMethod::Tsitouras45 => Tableau::<NalgebraMat<f64>>::tsit45(Default::default())
                .beta()
                .expect("native Tsitouras continuous extension")
                .ncols(),
            TimeMethod::Bdf => solver.order(),
            _ => return Err(solve_failed("unsupported native dense-output method")),
        };
        let dense = capture_polynomial(start_time, end_time, degree, |time| {
            solver
                .interpolate(time)
                .map(|state| collect_vector(&state))
                .map_err(|error| map_failure(failures, "capture native dense output", error))
        })?;
        steps.push(TimeHistoryStep::accepted(
            start_time,
            end_time,
            start_state,
            collect_vector(&middle),
            proposal.as_ref().map_or_else(
                || collect_vector(solver.state().y),
                |proposal| proposal.state().to_vec(),
            ),
            dense,
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
            let dense = capture_polynomial(start_time, end_time, degree, |time| {
                solver
                    .interpolate_sens(time)
                    .map(|states| flatten(&states))
                    .map_err(|error| {
                        map_failure(failures, "capture native sensitivity dense output", error)
                    })
            })?;
            sensitivity_steps.push(TimeHistoryStep::accepted(
                start_time,
                end_time,
                start_sensitivity,
                flatten(&middle),
                flatten(&end_sensitivity),
                dense,
            )?);
        }
        while sample < times.len()
            && (times[sample] < end_time || (proposal.is_none() && times[sample] == end_time))
        {
            let step = steps.last().expect("captured native step");
            values.extend(step.sample(times[sample])?);
            if parameters > 0 {
                let step = sensitivity_steps.last().expect("captured sensitivity step");
                let state = step.sample(times[sample])?;
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
        next_step = solver.state().h;
        if let Some(derivative) = &mut derivative {
            *derivative = collect_vector(solver.state().dy);
        }
        if let Some((_, control)) = &control
            && let Some(progress) = control.progress
        {
            // Observation is bounded and never blocks numerical acceptance.
            let _ = progress.try_send((steps.len(), end_time));
        }
        if proposal.is_some() || stop == OdeSolverStopReason::TstopReached {
            break;
        }
    }
    if let Some((checkpoint, _)) = control
        && cancelled
    {
        let saved = if steps.is_empty() {
            checkpoint
        } else {
            checkpoint.with_history(
                AcceptedTimeHistory::accepted(dimension, steps, Vec::new())?,
                values,
                next_step,
                derivative.expect("controlled native derivative"),
            )?
        };
        return Ok(ControlFlow::Break(saved));
    }
    if (proposal.is_none() && sample != times.len()) || steps[0].start_time() != plan.start_time() {
        return Err(solve_failed(
            "native accepted history does not span the requested solve",
        ));
    }
    failures.check()?;
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
    if sample < times.len() {
        for parameter in 0..parameters {
            let start = parameter * times.len() * dimension;
            sensitivities.copy_within(
                start..start + sample * dimension,
                parameter * sample * dimension,
            );
        }
        sensitivities.truncate(parameters * sample * dimension);
    }
    Ok(ControlFlow::Continue(CapturedSegment {
        history,
        times: times[..sample].to_vec(),
        values,
        sensitivities,
        sensitivity_history,
        proposal,
    }))
}

fn flatten(states: &[NalgebraVec<f64>]) -> Vec<f64> {
    states.iter().flat_map(collect_vector).collect()
}

fn report(problem: &TimeProblem<'_>, plan: &TimePlan) -> TimeExecutionReport {
    TimeExecutionReport::new(
        DiffsolTimeBackend::IDENTITY,
        plan.method(),
        problem.equation_class(),
        problem.initial_condition(),
    )
}

fn capture_polynomial(
    start: f64,
    end: f64,
    degree: usize,
    mut sample: impl FnMut(f64) -> Result<Vec<f64>, Diagnostic>,
) -> Result<TimeDenseOutput, Diagnostic> {
    if degree == 0 {
        return Err(solve_failed(
            "native dense output omitted its polynomial degree",
        ));
    }
    let times = (0..=degree)
        .map(|index| {
            if index == degree {
                end
            } else {
                start + (end - start) * (index as f64 / degree as f64)
            }
        })
        .collect::<Vec<_>>();
    let states = times
        .iter()
        .map(|&time| sample(time))
        .collect::<Result<Vec<_>, _>>()?;
    TimeDenseOutput::from_samples(times, states)
}
