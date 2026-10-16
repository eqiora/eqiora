//! Exact physical-time admission for the serial process profile.
use super::*;

impl ProcessMethod {
    pub(crate) fn validate_times(
        self,
        start: f64,
        until: f64,
        outputs: &[f64],
    ) -> Result<(), Diagnostic> {
        let duration = until - start;
        if !duration.is_finite() || duration <= 0. || exact(duration) != exact(until) - exact(start)
        {
            return Err(invalid(
                "serial stochastic horizon requires an exactly representable binary64 duration",
            ));
        }
        if let Self::EulerMaruyama { depth } = self {
            for &time in outputs {
                tick(&(exact(time) - exact(start)), duration, depth)?;
            }
        }
        Ok(())
    }
}

pub(super) fn tick(elapsed: &BigRational, duration: f64, depth: u8) -> Result<u64, Diagnostic> {
    let tick = elapsed * BigRational::from_integer((1_u64 << depth).into()) / exact(duration);
    if !tick.is_integer() {
        return Err(invalid(
            "Wiener trial boundary is outside its admitted dyadic grid",
        ));
    }
    tick.to_integer()
        .to_u64()
        .ok_or_else(|| invalid("invalid Wiener trial tick"))
}
