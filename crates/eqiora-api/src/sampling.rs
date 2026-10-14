//! Frozen, identity-addressed real Parameter inputs, drawn before evaluation.

use std::sync::Arc;

use eqiora_core::{Diagnostic, diagnostic::codes};
use sha2::{Digest, Sha256};

use crate::{DifferentiableProgram, DifferentiableProgramIdentity, DifferentiableValues};

/// Admitted deterministic generator and conversion contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingGenerator {
    /// SHA-256 identity/counter derivation and binary64 uniform conversion V1.
    Sha256UniformV1,
}

/// Whether distinct sample requests deliberately share random numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplingCoupling {
    /// Derive from the complete sample ID. Distinct IDs request distinct streams.
    IndependentReplicate,
    /// Derive from this explicit coupling ID instead of the sample ID.
    /// The channel must also agree to share a stream.
    CommonRandomNumbers(Vec<u8>),
}

/// Full client identity, retained independently of numerical Parameter values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplingIdentity {
    sample: Vec<u8>,
    channel: Vec<u8>,
    lineage: Vec<u64>,
    coupling: SamplingCoupling,
}

impl SamplingIdentity {
    /// Admit nonempty IDs of at most 256 bytes and at most 32 lineage entries.
    /// Lineage records request association only and never changes random draws.
    ///
    /// # Errors
    /// Rejects empty/oversized IDs or oversized lineage before copying them.
    pub fn new(
        sample: &[u8],
        channel: &[u8],
        lineage: &[u64],
        coupling: SamplingCoupling,
    ) -> Result<Self, Diagnostic> {
        for id in [sample, channel] {
            validate_id(id)?;
        }
        if let SamplingCoupling::CommonRandomNumbers(id) = &coupling {
            validate_id(id)?;
        }
        if lineage.len() > 32 {
            return Err(invalid("sampling occurrence lineage exceeds 32 entries"));
        }
        Ok(Self {
            sample: sample.to_vec(),
            channel: channel.to_vec(),
            lineage: lineage.to_vec(),
            coupling,
        })
    }

    /// Complete client sample ID, including under explicit coupling.
    #[must_use]
    pub fn sample_id(&self) -> &[u8] {
        &self.sample
    }

    /// Complete stream/noise-channel role.
    #[must_use]
    pub fn channel(&self) -> &[u8] {
        &self.channel
    }

    /// Ordered occurrence association metadata, excluded from stream derivation.
    #[must_use]
    pub fn lineage(&self) -> &[u64] {
        &self.lineage
    }

    /// Explicit replicate/coupling request.
    #[must_use]
    pub const fn coupling(&self) -> &SamplingCoupling {
        &self.coupling
    }
}

/// Bounded sampler in one admitted Program's exact ordered Parameter coordinates.
///
/// Bounds use coherent SI values in that existing typed input order. This is a
/// deterministic finite-grid sampler, not evidence of statistical independence
/// or a time-indexed stochastic process. No draw happens inside evaluation.
#[derive(Debug, Clone)]
pub struct ParameterSampler {
    program: Arc<DifferentiableProgram>,
    generator: SamplingGenerator,
    master: [u8; 32],
    bounds: Vec<[f64; 2]>,
}

impl ParameterSampler {
    /// Admit 1..=256 finite increasing intervals with finite widths.
    ///
    /// # Errors
    /// Rejects shape/type-input disagreement and invalid intervals before sampling.
    pub fn new(
        program: Arc<DifferentiableProgram>,
        generator: SamplingGenerator,
        master: [u8; 32],
        bounds: &[[f64; 2]],
    ) -> Result<Self, Diagnostic> {
        if bounds.is_empty() || bounds.len() > 256 {
            return Err(invalid(
                "sampling requires 1..=256 real Parameter intervals",
            ));
        }
        for &[lower, upper] in bounds {
            if !lower.is_finite()
                || !upper.is_finite()
                || lower >= upper
                || !(upper - lower).is_finite()
            {
                return Err(invalid(
                    "sampling intervals require finite increasing endpoints and finite width",
                ));
            }
        }
        if bounds.len() != program.identity().input_dimension() {
            return Err(invalid(
                "sampling intervals differ from the Program input dimension",
            ));
        }
        Ok(Self {
            program,
            generator,
            master,
            bounds: bounds.to_vec(),
        })
    }

    /// Exact generator/conversion version.
    #[must_use]
    pub const fn generator(&self) -> SamplingGenerator {
        self.generator
    }

    /// Full master stream identity; no ambient seed is consulted.
    #[must_use]
    pub const fn master_stream(&self) -> &[u8; 32] {
        &self.master
    }

    /// Ordered coherent-SI half-open sampling intervals.
    #[must_use]
    pub fn bounds(&self) -> &[[f64; 2]] {
        &self.bounds
    }

    /// Freeze one complete typed point. Repeated identities deliberately replay.
    ///
    /// Call in any order, partition, worker or retry schedule. The generated
    /// values are passed explicitly to `DifferentiableProgram::evaluate` or
    /// `EvaluationMapPlan::new`; those owners retain numerical admission.
    ///
    /// # Errors
    /// Returns the existing typed input admission diagnostic if conversion fails.
    pub fn sample(&self, identity: &SamplingIdentity) -> Result<SampledParameterPoint, Diagnostic> {
        let stream = stream_digest(&self.master, identity);
        let values = self
            .bounds
            .iter()
            .enumerate()
            .map(|(index, &[lower, upper])| {
                let unit = uniform(&stream, index as u64);
                convert(lower, upper, unit)
            })
            .collect::<Vec<_>>();
        Ok(SampledParameterPoint {
            generator: self.generator,
            master: self.master,
            identity: identity.clone(),
            stream,
            program: self.program.identity().clone(),
            point: self.program.point_from_coordinates(&values)?,
        })
    }
}

/// A frozen generated input and its complete sampling and Program association.
#[derive(Debug, Clone, PartialEq)]
pub struct SampledParameterPoint {
    generator: SamplingGenerator,
    master: [u8; 32],
    identity: SamplingIdentity,
    stream: [u8; 32],
    program: DifferentiableProgramIdentity,
    point: DifferentiableValues,
}

impl SampledParameterPoint {
    /// Exact generator version.
    #[must_use]
    pub const fn generator(&self) -> SamplingGenerator {
        self.generator
    }
    /// Full master stream identity.
    #[must_use]
    pub const fn master_stream(&self) -> &[u8; 32] {
        &self.master
    }
    /// Full request identity, not inferred from the generated point.
    #[must_use]
    pub const fn identity(&self) -> &SamplingIdentity {
        &self.identity
    }
    /// All 256 derived identity bits, not the truncated floating-point draw.
    #[must_use]
    pub const fn stream_digest(&self) -> &[u8; 32] {
        &self.stream
    }
    /// Exact existing Program identity; sample IDs never alter it.
    #[must_use]
    pub const fn program_identity(&self) -> &DifferentiableProgramIdentity {
        &self.program
    }
    /// Complete typed frozen input for explicit client passing and replay.
    #[must_use]
    pub const fn point(&self) -> &DifferentiableValues {
        &self.point
    }
}

fn validate_id(id: &[u8]) -> Result<(), Diagnostic> {
    if id.is_empty() || id.len() > 256 {
        return Err(invalid("sampling IDs require 1..=256 bytes"));
    }
    Ok(())
}

fn stream_digest(master: &[u8; 32], identity: &SamplingIdentity) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"eqiora.parameter-uniform.sha256.v1\0");
    hash.update(master);
    append_id(&mut hash, &identity.channel);
    match &identity.coupling {
        SamplingCoupling::IndependentReplicate => {
            hash.update([0]);
            append_id(&mut hash, &identity.sample);
        }
        SamplingCoupling::CommonRandomNumbers(id) => {
            hash.update([1]);
            append_id(&mut hash, id);
        }
    }
    hash.finalize().into()
}

fn append_id(hash: &mut Sha256, id: &[u8]) {
    hash.update((id.len() as u16).to_be_bytes());
    hash.update(id);
}

fn uniform(stream: &[u8; 32], index: u64) -> f64 {
    let mut hash = Sha256::new();
    hash.update(b"eqiora.parameter-uniform.draw.v1\0");
    hash.update(stream);
    hash.update(index.to_be_bytes());
    let digest = hash.finalize();
    let word = u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 has eight bytes"));
    (word >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

fn convert(lower: f64, upper: f64, unit: f64) -> f64 {
    let offset = (upper - lower) * unit;
    let value = lower + offset;
    if value >= upper {
        upper.next_down()
    } else {
        value
    }
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXECUTION_CONFIG, message)
}

#[cfg(test)]
mod tests;
