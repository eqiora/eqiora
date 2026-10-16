//! Frozen, identity-addressed real Parameter inputs, drawn before evaluation.

use std::sync::Arc;

use eqiora_core::{Diagnostic, diagnostic::codes};
use sha2::{Digest, Sha256};

use crate::{DifferentiableProgram, DifferentiableProgramIdentity, DifferentiableValues};

use eqiora_numerics::SamplingIdentity;

/// Bounded sampler in one admitted Program's exact ordered Parameter coordinates.
///
/// Bounds use coherent SI values in that existing typed input order. This is a
/// deterministic finite-grid sampler, not evidence of statistical independence
/// or a time-indexed stochastic process. No draw happens inside evaluation.
#[derive(Debug, Clone)]
pub struct ParameterSampler {
    program: Arc<DifferentiableProgram>,
    master: [u8; 32],
    bounds: Vec<[f64; 2]>,
    fixed: Arc<DifferentiableValues>,
    fixed_coordinates: Vec<usize>,
    sampled_coordinates: Vec<usize>,
}

impl ParameterSampler {
    /// Exact supported SHA-256 stream derivation and binary64 conversion contract.
    pub const GENERATOR: &'static str = "eqiora.parameter-uniform.sha256.v1";

    /// Freeze selected typed inputs and admit 1..=256 finite sampling intervals.
    ///
    /// Bounds cover only the remaining Program coordinates, in their original
    /// order. A draw retains its absolute Program coordinate index, so fixing
    /// another input does not shift its stream. An empty fixed product samples
    /// every input. Changing fixed values makes a new explicit sampling request;
    /// execution and differentiation never resample.
    ///
    /// # Errors
    /// Rejects shape/type-input disagreement and invalid intervals before sampling.
    pub fn new(
        program: Arc<DifferentiableProgram>,
        master: [u8; 32],
        fixed: &DifferentiableValues,
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
        let selected =
            crate::evaluation_map::partition::selected_parameters(program.identity(), fixed)?;
        let (fixed_coordinates, sampled_coordinates) =
            crate::evaluation_map::partition::coordinates(program.identity(), &selected)?;
        if bounds.len() != sampled_coordinates.len() {
            return Err(invalid(
                "sampling intervals differ from the remaining Program input coordinates",
            ));
        }
        Ok(Self {
            program,
            master,
            bounds: bounds.to_vec(),
            fixed: Arc::new(fixed.clone()),
            fixed_coordinates,
            sampled_coordinates,
        })
    }

    /// Full master stream identity; no ambient seed is consulted.
    #[must_use]
    pub const fn master_stream(&self) -> &[u8; 32] {
        &self.master
    }

    /// Frozen typed values excluded from random draws, in their selected leaf order.
    #[must_use]
    pub fn fixed_values(&self) -> &DifferentiableValues {
        &self.fixed
    }

    /// Ordered coherent-SI half-open intervals for the remaining input coordinates.
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
        let stream = identity.stream_digest(&self.master, Self::GENERATOR);
        let draws = self
            .bounds
            .iter()
            .zip(&self.sampled_coordinates)
            .map(|(&[lower, upper], &index)| {
                let unit = uniform(&stream, index as u64);
                convert(lower, upper, unit)
            })
            .collect::<Vec<_>>();
        let values = crate::evaluation_map::partition::assemble_coordinates(
            self.program.identity().input_dimension(),
            &self.fixed_coordinates,
            &self.sampled_coordinates,
            self.fixed.coordinates(),
            &draws,
        );
        Ok(SampledParameterPoint {
            fixed: Arc::clone(&self.fixed),
            generator: Self::GENERATOR,
            bounds: self.bounds.clone(),
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
    fixed: Arc<DifferentiableValues>,
    generator: &'static str,
    bounds: Vec<[f64; 2]>,
    master: [u8; 32],
    identity: SamplingIdentity,
    stream: [u8; 32],
    program: DifferentiableProgramIdentity,
    point: DifferentiableValues,
}

impl SampledParameterPoint {
    /// Fixed values of this request, distinct from generated Parameter coordinates.
    #[must_use]
    pub fn fixed_values(&self) -> &DifferentiableValues {
        &self.fixed
    }

    /// Frozen intervals in remaining Program coordinate order, excluding fixed leaves.
    #[must_use]
    pub fn bounds(&self) -> &[[f64; 2]] {
        &self.bounds
    }

    /// Exact generator version.
    #[must_use]
    pub const fn generator(&self) -> &'static str {
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
