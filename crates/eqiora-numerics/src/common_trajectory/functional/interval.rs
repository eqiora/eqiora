//! Restrict an accepted-step interpolation without inventing a State interpolant.
use super::{TimeObservationInterpolation, invalid};
use eqiora_core::Diagnostic;

pub(super) struct TimeObservationInterval([f64; 2]);

impl TimeObservationInterval {
    pub(super) fn new(
        trajectory: &super::CommonTrajectory,
        requested: Option<[f64; 2]>,
    ) -> Result<Self, Diagnostic> {
        match &trajectory.payload {
            super::TrajectoryPayload::Process { .. } => Err(invalid(
                "stochastic trajectory has no admitted smooth time interpolant",
            )),
            super::TrajectoryPayload::Ode { history, .. }
            | super::TrajectoryPayload::SpatialTime { history, .. } => Self::covered(
                [
                    history.steps()[0].start_time(),
                    history.steps().last().expect("accepted history").end_time(),
                ],
                requested,
                history
                    .steps()
                    .iter()
                    .map(|step| [step.start_time(), step.end_time()]),
            ),
            super::TrajectoryPayload::SpatialTransient {
                request, states, ..
            } => {
                let dt = request
                    .plan()
                    .backward_euler()
                    .ok_or_else(|| invalid("time observation requires accepted step coordinates"))?
                    .step()
                    .value();
                let start = request.state().require_time_s()?;
                Self::covered(
                    [start, start + request.accepted_steps()?.get() as f64 * dt],
                    requested,
                    states.iter().map(|(_, step)| step.time_interval()),
                )
            }
        }
    }

    fn covered(
        run: [f64; 2],
        requested: Option<[f64; 2]>,
        steps: impl IntoIterator<Item = [f64; 2]>,
    ) -> Result<Self, Diagnostic> {
        let [start, end] = requested.unwrap_or(run);
        if !start.is_finite() || !end.is_finite() || start >= end || !(end - start).is_finite() {
            return Err(invalid(
                "time observation requires a finite increasing interval",
            ));
        }
        if start < run[0] || end > run[1] {
            return Err(invalid(
                "time observation interval is outside complete accepted history",
            ));
        }
        let mut covered = start;
        for [left, right] in steps {
            if right <= covered {
                continue;
            }
            if left > covered {
                break;
            }
            covered = right.min(end);
            if covered == end {
                return Ok(Self([start, end]));
            }
        }
        Err(invalid(
            "time observation interval is not covered by retained accepted steps",
        ))
    }

    pub(super) fn bounds(&self) -> [f64; 2] {
        self.0
    }

    pub(super) fn weights(
        &self,
        step: [f64; 2],
        rule: TimeObservationInterpolation,
    ) -> Option<[f64; 3]> {
        let [start, end] = self.clip(step)?;
        (start < end).then(|| rule.integral_weights(step, [start, end]))
    }

    pub(super) fn clip(&self, step: [f64; 2]) -> Option<[f64; 2]> {
        let start = self.0[0].max(step[0]);
        let end = self.0[1].min(step[1]);
        (start <= end).then_some([start, end])
    }
}
