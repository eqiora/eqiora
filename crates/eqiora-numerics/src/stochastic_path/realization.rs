//! Bind immutable process laws to sample/channel paths without worker ordinals.
use super::{SamplingIdentity, WienerPath, exact, invalid, poisson::PoissonPath, program::Noise};
use eqiora_core::{Diagnostic, ValueLiteral};
use eqiora_schema::kernel::NoiseProcess;
use num_rational::BigRational;
use num_traits::Zero;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(
    tag = "process",
    content = "path",
    rename_all = "kebab-case",
    deny_unknown_fields
)]
pub(super) enum RealizedNoise {
    Wiener {
        independent: Vec<WienerPath>,
        #[serde(with = "super::retained::values")]
        factor: Vec<BigRational>,
    },
    Poisson(Vec<PoissonPath>),
}

impl Noise {
    pub(super) fn require_constant_parameter(&self) -> Result<(), Diagnostic> {
        if !self.parameter.constant() {
            return Err(invalid(
                "this path realization requires Model-constant covariance or Poisson intensity",
            ));
        }
        Ok(())
    }
    pub(super) fn wiener_factor(&self) -> Result<Vec<BigRational>, Diagnostic> {
        self.require_constant_parameter()?;
        let covariance = ValueLiteral::new(
            self.parameter.value_type.clone(),
            self.parameter
                .evaluate_constant()?
                .into_iter()
                .map(|value| (value, 0.)),
        )
        .map_err(|_| invalid("invalid realized covariance value"))?;
        Ok(NoiseProcess::wiener_covariance_factor(&covariance)?
            .into_iter()
            .map(exact)
            .collect())
    }

    pub(super) fn realize(
        &self,
        master: &[u8; 32],
        identity: &SamplingIdentity,
        duration: f64,
        depth: u8,
        max_events: usize,
    ) -> Result<RealizedNoise, Diagnostic> {
        self.require_constant_parameter()?;
        let parameters = self.parameter.evaluate_constant()?;
        let channel = |component: usize| {
            let mut hash = Sha256::new();
            hash.update(b"eqiora.process-noise.channel.v1\0");
            hash.update((identity.channel().len() as u16).to_be_bytes());
            hash.update(identity.channel());
            hash.update(self.id.ulid().to_bytes());
            hash.update((component as u64).to_be_bytes());
            SamplingIdentity::new(
                identity.sample_id(),
                &hash.finalize(),
                identity.lineage(),
                identity.coupling_id(),
            )
        };
        match self.process {
            NoiseProcess::Wiener => {
                let factor = self.wiener_factor()?;
                let n = factor.len().isqrt();
                let independent = (0..n)
                    .map(|index| WienerPath::new(master, &channel(index)?, 1., duration, depth))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(RealizedNoise::Wiener {
                    independent,
                    factor,
                })
            }
            NoiseProcess::Poisson => Ok(RealizedNoise::Poisson(
                parameters
                    .into_iter()
                    .enumerate()
                    .map(|(index, rate)| {
                        PoissonPath::new(master, &channel(index)?, rate, duration, max_events)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )),
        }
    }
}

impl RealizedNoise {
    #[cfg(test)]
    pub(super) fn wiener_increment(
        &self,
        from: u64,
        to: u64,
    ) -> Result<Vec<BigRational>, Diagnostic> {
        let Self::Wiener {
            independent,
            factor,
        } = self
        else {
            return Err(invalid("Wiener increments require a Wiener process law"));
        };
        let increments = independent
            .iter()
            .map(|path| path.increment(from, to))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(mix_wiener_increments(factor, &increments))
    }

    pub(super) fn independent_wiener_increment(
        &self,
        from: u64,
        to: u64,
    ) -> Result<Vec<BigRational>, Diagnostic> {
        let Self::Wiener { independent, .. } = self else {
            return Err(invalid("Wiener increments require a Wiener process law"));
        };
        independent
            .iter()
            .map(|path| path.increment(from, to))
            .collect()
    }

    #[cfg(test)]
    pub(super) fn poisson_increment(
        &self,
        from: &BigRational,
        to: &BigRational,
    ) -> Result<Vec<i64>, Diagnostic> {
        let Self::Poisson(paths) = self else {
            return Err(invalid("counting increments require a Poisson process law"));
        };
        paths.iter().map(|path| path.increment(from, to)).collect()
    }
}

pub(super) fn mix_wiener_increments(
    factor: &[BigRational],
    increments: &[BigRational],
) -> Vec<BigRational> {
    factor
        .chunks_exact(increments.len())
        .map(|row| {
            row.iter()
                .zip(increments)
                .fold(BigRational::zero(), |sum, (coefficient, increment)| {
                    sum + coefficient * increment
                })
        })
        .collect()
}
