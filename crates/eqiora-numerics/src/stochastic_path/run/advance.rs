//! Ordinary finite-time Run execution over immutable retained paths.
use super::*;
use crate::{CommonTimeCheckpoint, CommonTrajectory};
use std::ops::ControlFlow;

impl ProcessCursor {
    pub(super) fn next_output(&self) -> Option<BigRational> {
        self.request()
            .output_times_s()
            .get(self.outputs.len())
            .map(|time| exact(*time) - exact(self.request().state().time_s()))
    }

    fn next_position(&self) -> Result<BigRational, Diagnostic> {
        let horizon = exact(self.run.duration());
        match self.run.method() {
            ProcessMethod::EulerMaruyama { depth } => {
                let tick = super::time::tick(&self.elapsed, self.run.duration(), depth)?;
                Ok(horizon * BigRational::new((tick + 1).into(), (1_u64 << depth).into()))
            }
            ProcessMethod::PoissonEvents { .. } => {
                let mut next = self.next_output().unwrap_or(horizon);
                if let Some(arrival) = self.run.next_arrival(&self.elapsed) {
                    next = next.min(arrival.clone());
                }
                Ok(next)
            }
        }
    }
}

impl CommonTimeRunRequest {
    /// Execute or resume the same serial stochastic Run. Ito steps use the Plan's
    /// finest admitted dyadic grid; pure Poisson jumps use exact retained arrivals.
    /// The callback runs before the next trial at each accepted boundary. Break
    /// saves that boundary atomically; it neither consumes nor redraws a path.
    /// The terminal boundary is also reported; completion takes precedence over a
    /// terminal Break and returns the accepted Trajectory, including sparse outputs.
    pub fn advance_process<F>(
        &self,
        checkpoint: Option<&CommonTimeCheckpoint>,
        mut control: F,
    ) -> Result<ControlFlow<CommonTimeCheckpoint, CommonTrajectory>, Diagnostic>
    where
        F: FnMut(&CommonTimeState, usize) -> ControlFlow<()>,
    {
        let mut cursor = if let Some(checkpoint) = checkpoint {
            if checkpoint.request().identity() != self.identity() {
                return Err(invalid(
                    "checkpoint belongs to a different exact time Run request",
                ));
            }
            checkpoint.process()?.clone()
        } else {
            ProcessRun::new(self.clone())?.start()
        };
        loop {
            let action = control(cursor.state(), cursor.accepted_steps());
            if cursor.elapsed == exact(cursor.run.duration()) {
                return CommonTrajectory::accept_process(history::ProcessHistory::from_cursor(
                    cursor,
                )?)
                .map(ControlFlow::Continue);
            }
            if action.is_break() {
                return CommonTimeCheckpoint::from_process(cursor).map(ControlFlow::Break);
            }
            cursor.accept(cursor.trial(&cursor.next_position()?)?)?;
        }
    }
}
