//! Restrict an accepted-step quadrature without inventing a State interpolant.
use super::{TimeFunctionalQuadrature, invalid};
use eqiora_core::Diagnostic;
use eqiora_time::{AcceptedTimeHistory, TimeHistoryStep};

pub(super) struct TimeIntegralInterval([f64; 2]);

impl TimeIntegralInterval {
    pub(super) fn new(
        history: &AcceptedTimeHistory,
        requested: Option<[f64; 2]>,
    ) -> Result<Self, Diagnostic> {
        let coverage = [
            history
                .steps()
                .first()
                .expect("accepted history")
                .start_time(),
            history.steps().last().expect("accepted history").end_time(),
        ];
        let [start, end] = requested.unwrap_or(coverage);
        if !start.is_finite() || !end.is_finite() || start >= end {
            return Err(invalid(
                "time integral requires a finite increasing interval",
            ));
        }
        if start < coverage[0] || end > coverage[1] {
            return Err(invalid(
                "time integral interval is outside complete accepted history",
            ));
        }
        Ok(Self([start, end]))
    }

    pub(super) fn bounds(&self) -> [f64; 2] {
        self.0
    }

    pub(super) fn weights(
        &self,
        step: &TimeHistoryStep,
        rule: TimeFunctionalQuadrature,
    ) -> Option<[f64; 3]> {
        let start = self.0[0].max(step.start_time());
        let end = self.0[1].min(step.end_time());
        if start >= end {
            return None;
        }
        let width = end - start;
        match rule {
            TimeFunctionalQuadrature::AcceptedStepSimpson => {
                if start == step.start_time() && end == step.end_time() {
                    return Some([width / 6., width * (2. / 3.), width / 6.]);
                }
                let step_width = step.end_time() - step.start_time();
                let a = (start - step.start_time()) / step_width;
                let b = (end - step.start_time()) / step_width;
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
