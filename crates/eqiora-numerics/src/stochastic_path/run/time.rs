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

/// The same accepted output boundary is used by live cursors and stored histories.
pub(super) fn accept_output(
    request: &CommonTimeRunRequest,
    outputs: &mut Vec<CommonTimeState>,
    elapsed: &BigRational,
    state: &CommonTimeState,
) -> Result<(), Diagnostic> {
    if let Some(time) = request.output_times_s().get(outputs.len()) {
        let position = exact(*time) - exact(request.state().time_s());
        if elapsed > &position {
            return Err(invalid(
                "accepted process step skips a requested output boundary",
            ));
        }
        if elapsed == &position {
            outputs.push(state.clone());
        }
    }
    Ok(())
}

pub(super) fn physical_time(
    request: &CommonTimeRunRequest,
    elapsed: &BigRational,
) -> Result<f64, Diagnostic> {
    if elapsed < &BigRational::zero()
        || elapsed > &exact(request.until_s() - request.state().time_s())
    {
        return Err(invalid("process Run position exceeds its original horizon"));
    }
    finite(exact(request.state().time_s()) + elapsed)
}

pub(super) fn advance_time(
    request: &CommonTimeRunRequest,
    from: &BigRational,
    state: &CommonTimeState,
    to: &BigRational,
) -> Result<f64, Diagnostic> {
    if to <= from {
        return Err(invalid("process trial must advance its accepted position"));
    }
    let end = physical_time(request, to)?;
    if end <= state.time_s() {
        return Err(invalid(
            "process trial interval is not representable in physical time",
        ));
    }
    Ok(end)
}
