//! Immutable process paths, disposable method trials, one accepted boundary.
use super::{exact, invalid, program::ProcessProgram, realization::RealizedNoise};
use crate::{CommonTimeRunRequest, CommonTimeState};
use eqiora_core::{Diagnostic, Id, entity::kinds};
use eqiora_schema::kernel::{NoiseProcess, StochasticCalculus};
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use std::{collections::HashMap, sync::Arc};

mod advance;
mod checkpoint;
mod jump;
mod time;
mod wiener;
use jump::JumpEvent;

/// Numerical admission is separate from the authored process law.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "method", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum ProcessMethod {
    EulerMaruyama { depth: u8 },
    PoissonEvents { max_events: usize },
}

impl ProcessMethod {
    pub(crate) fn admit(
        self,
        program: &ProcessProgram,
    ) -> Result<(NoiseProcess, u8, usize), Diagnostic> {
        let admission = match self {
            ProcessMethod::EulerMaruyama { depth } => {
                if depth > 52
                    || program.laws.iter().any(|law| {
                        law.calculus != Some(StochasticCalculus::Ito) || !law.jumps.is_empty()
                    })
                {
                    return Err(invalid(
                        "Euler-Maruyama requires Ito diffusion laws without counting jumps and depth at most 52",
                    ));
                }
                (NoiseProcess::Wiener, depth, 0)
            }
            ProcessMethod::PoissonEvents { max_events } => {
                if max_events > 1_000_000
                    || program.laws.iter().any(|law| {
                        law.calculus.is_some() || law.drift.is_some() || !law.diffusions.is_empty()
                    })
                {
                    return Err(invalid(
                        "Poisson event execution requires pure jump laws and at most 1000000 events",
                    ));
                }
                (NoiseProcess::Poisson, 0, max_events)
            }
        };
        for noise in program
            .noises
            .iter()
            .filter(|noise| noise.process == admission.0)
        {
            noise.require_constant_parameter()?;
        }
        Ok(admission)
    }
}

/// Numerical method state; the common Run owner supplies persistence and control.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProcessRun {
    request: CommonTimeRunRequest,
    paths: HashMap<Id<kinds::Relation>, RealizedNoise>,
    events: Vec<JumpEvent>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProcessCursor {
    run: Arc<ProcessRun>,
    elapsed: BigRational,
    state: CommonTimeState,
    accepted_positions: Vec<BigRational>,
    outputs: Vec<CommonTimeState>,
}

#[derive(Debug)]
pub(super) struct ProcessTrial {
    from_elapsed: BigRational,
    from_identity: String,
    run: Arc<ProcessRun>,
    elapsed: BigRational,
    state: CommonTimeState,
}

impl ProcessRun {
    pub(crate) fn new(request: CommonTimeRunRequest) -> Result<Arc<Self>, Diagnostic> {
        let (program, method) = request
            .plan()
            .process()
            .ok_or_else(|| invalid("process execution requires a stochastic time Plan"))?;
        let (master, identity) = request
            .sampling()
            .ok_or_else(|| invalid("process execution requires sample identity"))?;
        program.validate_state(request.state())?;
        let duration = request.until_s() - request.state().time_s();
        let (process, depth, max_events) = method.admit(program)?;
        let paths = program
            .noises
            .iter()
            .filter(|noise| noise.process == process)
            .map(|noise| {
                Ok((
                    noise.id,
                    noise.realize(master, identity, duration, depth, max_events)?,
                ))
            })
            .collect::<Result<_, Diagnostic>>()?;
        let events = match method {
            ProcessMethod::PoissonEvents { max_events } => jump::schedule(&paths, max_events)?,
            ProcessMethod::EulerMaruyama { .. } => Vec::new(),
        };
        Ok(Arc::new(Self {
            request,
            paths,
            events,
        }))
    }

    fn program(&self) -> &ProcessProgram {
        self.request
            .plan()
            .process()
            .expect("admitted process request")
            .0
    }
    fn method(&self) -> ProcessMethod {
        self.request
            .plan()
            .process()
            .expect("admitted process request")
            .1
    }
    fn duration(&self) -> f64 {
        self.request.until_s() - self.request.state().time_s()
    }

    fn time(&self, elapsed: &BigRational) -> Result<f64, Diagnostic> {
        if elapsed < &BigRational::zero() || elapsed > &exact(self.duration()) {
            return Err(invalid("process Run position exceeds its original horizon"));
        }
        finite(exact(self.request.state().time_s()) + elapsed)
    }

    pub(crate) fn start(self: &Arc<Self>) -> ProcessCursor {
        ProcessCursor {
            run: self.clone(),
            elapsed: BigRational::zero(),
            state: self.request.state().clone(),
            accepted_positions: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

impl ProcessCursor {
    pub(crate) fn state(&self) -> &CommonTimeState {
        &self.state
    }

    pub(crate) fn request(&self) -> &CommonTimeRunRequest {
        &self.run.request
    }
    pub(crate) fn accepted_steps(&self) -> usize {
        self.accepted_positions.len()
    }
    pub(crate) fn accepted_events(&self) -> usize {
        self.run.accepted_events(&self.elapsed)
    }

    /// Discarding this candidate consumes no path, pending event or physical state.
    pub(super) fn trial(&self, to: &BigRational) -> Result<ProcessTrial, Diagnostic> {
        if to <= &self.elapsed {
            return Err(invalid("process trial must advance its accepted position"));
        }
        let end = self.run.time(to)?;
        if end <= self.state.time_s() {
            return Err(invalid(
                "process trial interval is not representable in physical time",
            ));
        }
        let values = match self.run.method() {
            ProcessMethod::EulerMaruyama { depth } => self.run.wiener_values(self, to, depth)?,
            ProcessMethod::PoissonEvents { .. } => self.run.jump_values(self, to)?,
        };
        let state = self.run.program().accept_state(end, values, "result")?;
        Ok(ProcessTrial {
            from_elapsed: self.elapsed.clone(),
            from_identity: self.state.identity().to_owned(),
            run: self.run.clone(),
            elapsed: to.clone(),
            state,
        })
    }

    /// Atomically publish physical state and path position for this exact Run.
    pub(super) fn accept(&mut self, trial: ProcessTrial) -> Result<(), Diagnostic> {
        if trial.from_elapsed != self.elapsed
            || trial.from_identity != self.state.identity()
            || trial.run != self.run
        {
            return Err(invalid(
                "process trial is stale or belongs to a different Run",
            ));
        }
        let next_output = self.next_output();
        if next_output
            .as_ref()
            .is_some_and(|position| position < &trial.elapsed)
        {
            return Err(invalid(
                "accepted process step skips a requested output boundary",
            ));
        }
        if next_output.as_ref() == Some(&trial.elapsed) {
            self.outputs.push(trial.state.clone());
        }
        // No fallible numerical work remains after admission. A discarded
        // trial never copies or extends the accepted partition journal.
        self.accepted_positions.push(trial.elapsed.clone());
        (self.elapsed, self.state) = (trial.elapsed, trial.state);
        Ok(())
    }
}

fn finite(value: BigRational) -> Result<f64, Diagnostic> {
    value
        .to_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| invalid("process Run value exceeds binary64 range"))
}
