//! Left-endpoint Ito coefficients on the admitted exact dyadic grid.
use super::*;

impl ProcessRun {
    pub(super) fn wiener_values(
        &self,
        cursor: &ProcessCursor,
        to: &BigRational,
        depth: u8,
    ) -> Result<Vec<f64>, Diagnostic> {
        let from_tick = super::time::tick(&cursor.elapsed, self.duration(), depth)?;
        let to_tick = super::time::tick(to, self.duration(), depth)?;
        let dt = finite(to - &cursor.elapsed)?;
        let mut values = cursor.state.values().to_vec();
        for law in &self.program().laws {
            let drift = law
                .drift
                .as_ref()
                .expect("Ito law has drift")
                .evaluate(cursor.state.values(), cursor.state.time_s())?;
            let output = &mut values[law.state.clone()];
            if drift.len() != output.len() {
                return Err(invalid("drift coordinates differ from the owned State"));
            }
            for (value, rate) in output.iter_mut().zip(drift) {
                *value += dt * rate;
            }
            for (group, coefficient) in &law.diffusions {
                let path = self
                    .paths
                    .get(group)
                    .ok_or_else(|| invalid("Wiener trial omits a declared noise group"))?;
                let increments = path.wiener_increment(from_tick, to_tick)?;
                let coefficients =
                    coefficient.evaluate(cursor.state.values(), cursor.state.time_s())?;
                if output.len().checked_mul(increments.len()) != Some(coefficients.len()) {
                    return Err(invalid(
                        "diffusion channel coordinates differ from the owned State",
                    ));
                }
                for (channel, increment) in coefficients.chunks_exact(output.len()).zip(increments)
                {
                    let increment = finite(increment)?;
                    for (value, coefficient) in output.iter_mut().zip(channel) {
                        *value += coefficient * increment;
                    }
                }
            }
        }
        Ok(values)
    }
}
