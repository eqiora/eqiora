//! Identity-addressed paths; queries never advance an ambient random generator.
//!
//! The dyadic bridge retains exact sums of its binary64 normal draws. Conversion
//! to a solver's storage precision belongs at the increment consumption boundary.
//! In particular, separately rounded binary64 children need not sum bit-for-bit;
//! the retained increments do. Transcendentals remain environment-dependent.

use eqiora_core::{Diagnostic, diagnostic::codes};
use num_rational::BigRational;
use num_traits::Zero;
use sha2::{Digest, Sha256};

use crate::SamplingIdentity;

mod poisson;
pub(crate) mod program;
mod realization;
mod retained;
pub(crate) mod run;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WienerPath {
    stream: [u8; 32],
    variance_duration: f64,
    depth: u8,
}

impl WienerPath {
    pub(crate) const GENERATOR: &str = "eqiora.wiener-bridge.box-muller.sha256.v1";

    /// Admit a finite time horizon and a bounded dyadic query grid. The grid is
    /// numerical policy, not a restriction on the authored process law.
    pub(crate) fn new(
        master: &[u8; 32],
        identity: &SamplingIdentity,
        variance: f64,
        duration: f64,
        depth: u8,
    ) -> Result<Self, Diagnostic> {
        if !variance.is_finite()
            || variance < 0.
            || !duration.is_finite()
            || duration <= 0.
            || depth > 52
        {
            return Err(invalid(
                "Wiener path requires finite nonnegative variance, positive duration and depth at most 52",
            ));
        }
        let variance_duration = variance * duration;
        let leaf_variance = variance_duration / (1_u64 << depth) as f64;
        if !variance_duration.is_finite() || (variance > 0. && leaf_variance == 0.) {
            return Err(invalid(
                "Wiener bridge variance exceeds the admitted binary64 range",
            ));
        }
        Ok(Self {
            stream: identity.stream_digest(master, Self::GENERATOR),
            variance_duration,
            depth,
        })
    }

    pub(crate) fn ticks(&self) -> u64 {
        1_u64 << self.depth
    }

    /// Exact retained increment, including signed queries and empty intervals.
    pub(crate) fn increment(&self, from: u64, to: u64) -> Result<BigRational, Diagnostic> {
        Ok(self.value(to)? - self.value(from)?)
    }

    fn value(&self, tick: u64) -> Result<BigRational, Diagnostic> {
        if tick > self.ticks() {
            return Err(invalid("Wiener query is outside the admitted horizon"));
        }
        let mut left = BigRational::zero();
        if tick == 0 || self.variance_duration == 0. {
            return Ok(left);
        }
        let mut right = exact(self.variance_duration.sqrt()) * exact(self.normal(0));
        let mut lo = 0;
        let mut hi = self.ticks();
        let mut address = 1;
        let mut variance = self.variance_duration;
        loop {
            if tick == lo {
                return Ok(left);
            }
            if tick == hi {
                return Ok(right);
            }
            let middle = lo + (hi - lo) / 2;
            // Conditional mean (L+R)/2 and variance C*(b-a)/4. Exact
            // arithmetic makes every recursively retained endpoint identical
            // under query reordering and every interval split telescope.
            let mean = (&left + &right) / BigRational::from_integer(2.into());
            let value = mean + exact(variance.sqrt() * 0.5) * exact(self.normal(address));
            if tick <= middle {
                right = value;
                hi = middle;
                address *= 2;
            } else {
                left = value;
                lo = middle;
                address = address * 2 + 1;
            }
            variance *= 0.5;
        }
    }

    fn normal(&self, address: u64) -> f64 {
        let digest = addressed_digest(&self.stream, b"eqiora.wiener-bridge.node.v1\0", address);
        let radial = open_unit(u64::from_be_bytes(
            digest[..8].try_into().expect("eight bytes"),
        ));
        let angle = open_unit(u64::from_be_bytes(
            digest[8..16].try_into().expect("eight bytes"),
        ));
        (-2. * radial.ln()).sqrt() * (std::f64::consts::TAU * angle).cos()
    }
}

fn addressed_digest(stream: &[u8; 32], domain: &[u8], address: u64) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(stream);
    hash.update(address.to_be_bytes());
    hash.finalize().into()
}

/// Exactly represented midpoints of 2^52 cells: both logarithm endpoints reject
/// by construction, including the maximum input word (which cannot round to 1).
fn open_unit(word: u64) -> f64 {
    ((word >> 12) as f64 + 0.5) * (1. / 4_503_599_627_370_496.)
}

fn exact(value: f64) -> BigRational {
    BigRational::from_float(value).expect("admitted finite draw and scale")
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXECUTION_CONFIG, message)
}

#[cfg(test)]
mod tests;
