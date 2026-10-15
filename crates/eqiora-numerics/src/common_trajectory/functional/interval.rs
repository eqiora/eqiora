//! Restrict an accepted-step quadrature without inventing a State interpolant.
use super::{TimeFunctionalQuadrature, invalid};
use eqiora_core::Diagnostic;

pub(super) struct TimeIntegralInterval([f64; 2]);

impl TimeIntegralInterval {
    pub(super) fn new(
        trajectory: &super::CommonTrajectory,
        requested: Option<[f64; 2]>,
    ) -> Result<Self, Diagnostic> {
        match trajectory {
            super::CommonTrajectory::Ode { history, .. } => Self::covered(
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
            super::CommonTrajectory::SpatialTransient {
                request, states, ..
            } => {
                let dt = request
                    .plan()
                    .backward_euler()
                    .ok_or_else(|| invalid("time integral requires accepted step coordinates"))?
                    .step()
                    .value();
                let start = request.state().time_s();
                Self::covered(
                    [start, start + request.accepted_steps().get() as f64 * dt],
                    requested,
                    states
                        .iter()
                        .map(|(_, step)| [step.previous_state().time_s(), step.state().time_s()]),
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
                "time integral requires a finite increasing interval",
            ));
        }
        if start < run[0] || end > run[1] {
            return Err(invalid(
                "time integral interval is outside complete accepted history",
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
            "time integral interval is not covered by retained accepted steps",
        ))
    }

    pub(super) fn bounds(&self) -> [f64; 2] {
        self.0
    }

    pub(super) fn weights(
        &self,
        step: [f64; 2],
        rule: TimeFunctionalQuadrature,
    ) -> Option<[f64; 3]> {
        let start = self.0[0].max(step[0]);
        let end = self.0[1].min(step[1]);
        if start >= end {
            return None;
        }
        let width = end - start;
        match rule {
            TimeFunctionalQuadrature::AcceptedStepTrapezoid => {
                let width_step = step[1] - step[0];
                let mean = ((start - step[0]) / width_step + (end - step[0]) / width_step) * 0.5;
                Some([width * (1. - mean), 0., width * mean])
            }
            TimeFunctionalQuadrature::AcceptedStepSimpson => {
                if start == step[0] && end == step[1] {
                    return Some([width / 6., width * (2. / 3.), width / 6.]);
                }
                let step_width = step[1] - step[0];
                let a = (start - step[0]) / step_width;
                let b = (end - step[0]) / step_width;
                // Integrals of the Lagrange basis at 0, 1/2, 1, using the
                // first two mean moments on [a,b] to avoid primitive subtraction.
                let first = (a + b) / 2.;
                let second = (a * a + a * b + b * b) / 3.;
                Some([
                    width * (2. * second - 3. * first + 1.),
                    width * (4. * first - 4. * second),
                    width * (2. * second - first),
                ])
            }
        }
    }
}
