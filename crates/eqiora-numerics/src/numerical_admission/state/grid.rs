//! One fixed-step clock retained across spatial checkpoint boundaries.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TimeGrid {
    pub(in crate::numerical_admission) origin_s: f64,
    pub(in crate::numerical_admission) index: usize,
}

impl TimeGrid {
    pub(crate) fn time(self, step_s: f64) -> Result<f64, Diagnostic> {
        if !self.origin_s.is_finite()
            || self.origin_s < 0.
            || self.origin_s.to_bits() == (-0_f64).to_bits()
            || !step_s.is_finite()
            || step_s <= 0.
        {
            return Err(invalid(
                "Time grid requires a finite nonnegative origin and positive step",
            ));
        }
        let time = self.origin_s + self.index as f64 * step_s;
        if !time.is_finite()
            || (self.index > 0 && time <= self.origin_s + (self.index - 1) as f64 * step_s)
        {
            return Err(invalid(
                "Time grid must advance to a distinct finite timestamp",
            ));
        }
        Ok(time)
    }

    pub(crate) fn advance(self, count: usize) -> Result<Self, Diagnostic> {
        let index = self
            .index
            .checked_add(count)
            .ok_or_else(|| invalid("Time grid step count overflows usize"))?;
        Ok(Self { index, ..self })
    }

    pub(in crate::numerical_admission) fn previous(self) -> Result<Self, Diagnostic> {
        let index = self
            .index
            .checked_sub(1)
            .ok_or_else(|| invalid("discrete State requires a preceding Time grid point"))?;
        Ok(Self { index, ..self })
    }

    pub(super) fn steps_until(
        self,
        target_s: f64,
        step_s: f64,
        label: &str,
    ) -> Result<usize, Diagnostic> {
        if !target_s.is_finite() || target_s <= self.time(step_s)? {
            return Err(invalid(format!(
                "{label} values must be finite and later than State.time_s"
            )));
        }
        // Division only proposes an integer. The exact forward clock expression
        // authenticates it, including decimal steps whose quotient is rounded.
        let candidate = ((target_s - self.origin_s) / step_s).round();
        if !candidate.is_finite() || candidate < 0. || candidate >= usize::MAX as f64 {
            return Err(invalid(format!(
                "{label} values must align exactly to the Plan Backward-Euler grid"
            )));
        }
        let index = candidate as usize;
        let selected = Self { index, ..self };
        if index <= self.index || selected.time(step_s)?.to_bits() != target_s.to_bits() {
            return Err(invalid(format!(
                "{label} values must align exactly to the Plan Backward-Euler grid"
            )));
        }
        Ok(index - self.index)
    }
}

impl CommonState {
    pub(crate) fn time_grid(&self) -> Result<TimeGrid, Diagnostic> {
        match &self.kind {
            CommonStateKind::Spatial {
                previous: Some(previous),
                ..
            } => Ok(previous.grid),
            _ => Ok(TimeGrid {
                origin_s: self.require_time_s()?,
                index: 0,
            }),
        }
    }
}
