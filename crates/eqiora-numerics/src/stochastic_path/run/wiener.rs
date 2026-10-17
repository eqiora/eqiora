//! Left-endpoint Ito coefficients on the admitted exact dyadic grid.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct WienerIncrement {
    pub(super) group: Id<kinds::Relation>,
    pub(super) values: Vec<BigRational>,
}

impl ProcessRun {
    pub(super) fn wiener_inputs(
        &self,
        from: &BigRational,
        to: &BigRational,
        depth: u8,
    ) -> Result<Vec<WienerIncrement>, Diagnostic> {
        let from_tick = super::time::tick(from, self.duration(), depth)?;
        let to_tick = super::time::tick(to, self.duration(), depth)?;
        self.program()
            .noises
            .iter()
            .filter(|noise| noise.process == NoiseProcess::Wiener)
            .map(|noise| {
                Ok(WienerIncrement {
                    group: noise.id,
                    values: self
                        .paths
                        .get(&noise.id)
                        .ok_or_else(|| invalid("Wiener trial omits a declared noise group"))?
                        .independent_wiener_increment(from_tick, to_tick)?,
                })
            })
            .collect()
    }
}

impl ProcessProgram {
    /// One original-equation action, shared by execution and stored-stencil checks.
    /// Inputs are already realized; this owner has no path or generator handle.
    pub(super) fn wiener_values(
        &self,
        state: &CommonTimeState,
        dt: f64,
        increments: &[WienerIncrement],
    ) -> Result<Vec<f64>, Diagnostic> {
        let mut noises = self
            .noises
            .iter()
            .filter(|noise| noise.process == NoiseProcess::Wiener);
        let mut correlated = Vec::with_capacity(increments.len());
        for increment in increments {
            let noise = noises
                .next()
                .ok_or_else(|| invalid("foreign Wiener increment group"))?;
            let factor = noise.wiener_factor()?;
            if increment.group != noise.id || increment.values.len() != factor.len().isqrt() {
                return Err(invalid(
                    "Wiener increment differs from its exact noise channels",
                ));
            }
            correlated.push(WienerIncrement {
                group: increment.group,
                values: super::super::realization::mix_wiener_increments(
                    &factor,
                    &increment.values,
                ),
            });
        }
        if noises.next().is_some() {
            return Err(invalid("Wiener stencil omits a declared noise group"));
        }
        let mut values = state.values().to_vec();
        for law in &self.laws {
            let drift = law
                .drift
                .as_ref()
                .expect("Ito law has drift")
                .evaluate(state.values(), state.time_s())?;
            let output = &mut values[law.state.clone()];
            if drift.len() != output.len() {
                return Err(invalid("drift coordinates differ from the owned State"));
            }
            for (value, rate) in output.iter_mut().zip(drift) {
                *value += dt * rate;
            }
            for (group, coefficient) in &law.diffusions {
                let increments = &correlated
                    .iter()
                    .find(|input| input.group == *group)
                    .ok_or_else(|| invalid("Wiener stencil omits a declared noise group"))?
                    .values;
                let coefficients = coefficient.evaluate(state.values(), state.time_s())?;
                if output.len().checked_mul(increments.len()) != Some(coefficients.len()) {
                    return Err(invalid(
                        "diffusion channel coordinates differ from the owned State",
                    ));
                }
                for (channel, increment) in coefficients.chunks_exact(output.len()).zip(increments)
                {
                    let increment = finite(increment.clone())?;
                    for (value, coefficient) in output.iter_mut().zip(channel) {
                        *value += coefficient * increment;
                    }
                }
            }
        }
        Ok(values)
    }
}
