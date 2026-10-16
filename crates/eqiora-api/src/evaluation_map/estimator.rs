//! Estimator meaning binds accepted occurrences; it does not create another evaluator.
use super::{CompleteEvaluationMap, invalid};
use crate::ModelEntityRef;
use eqiora_core::{Diagnostic, ValueType};
use eqiora_schema::kernel::ProbabilityMoment;
use std::collections::BTreeSet;

/// Interpretation of a finite sample moment, separate from an exact population law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MomentEstimatorConvention {
    /// Normalize explicit nonnegative weights. Duplicate occurrences retain their
    /// multiplicity; this makes no independence, importance-sampling or bias claim.
    Empirical,
    /// The caller explicitly assumes identically distributed independent outputs
    /// with the required finite moments. Use equal positive weights and, for
    /// variance/covariance, the n-1 divisor instead of n. Stream metadata checks
    /// necessary consistency conditions; a deterministic generator cannot prove IID.
    UnbiasedIid,
}

/// An admitted moment-estimator specification over a complete accepted map.
///
/// The borrowed map owns exact Program/Plan identity, ordered sample identities,
/// frozen inputs and acceptance evidence. Borrowed weights cannot change while
/// this binding exists. All occurrences participate, including equal values;
/// neither incomplete membership nor implicit success-only filtering is allowed.
/// This binds the typed convention. Numerical reduction, gradients and sampling
/// error reporting are separate consumers, not an implicit side effect of binding.
#[derive(Debug)]
pub struct EvaluationMapEstimator<'a> {
    map: &'a CompleteEvaluationMap,
    moment: ProbabilityMoment,
    outputs: Vec<usize>,
    weights: &'a [f64],
    convention: MomentEstimatorConvention,
    value_type: ValueType,
}

impl<'a> EvaluationMapEstimator<'a> {
    /// Bind the moment's ordered scalar outputs and one weight per occurrence.
    /// No population-expectation or error-bar convention is accepted here.
    /// # Errors
    /// Rejects empty/invalid membership, foreign or unsupported outputs, bad
    /// weights, or sample records contradicting the explicit IID convention.
    pub fn new(
        map: &'a CompleteEvaluationMap,
        moment: ProbabilityMoment,
        outputs: &[ModelEntityRef],
        weights: &'a [f64],
        convention: MomentEstimatorConvention,
    ) -> Result<Self, Diagnostic> {
        map.validate()?;
        if map.is_empty() || map.len() != weights.len() || outputs.len() != moment.arity() {
            return Err(invalid(
                "estimator requires nonempty complete membership, one weight per occurrence and the moment's output arity",
            ));
        }
        ProbabilityMoment::validate_weights(weights.iter().copied())?;
        let identity = map.plan().program_identity();
        let mut selected = Vec::with_capacity(outputs.len());
        let mut types = Vec::with_capacity(outputs.len());
        for output in outputs {
            let (index, signature) = identity
                .outputs()
                .iter()
                .enumerate()
                .find(|(_, value)| {
                    output.model() == identity.model() && output.id() == value.entity()
                })
                .ok_or_else(|| invalid("estimator output is outside the exact accepted Program"))?;
            if signature.mathematical_type().support.is_some() || !signature.shape().is_empty() {
                return Err(invalid(
                    "moment estimator requires a finite scalar output, not spatial coefficients",
                ));
            }
            selected.push(index);
            types.push(&signature.mathematical_type().value_type);
        }
        let value_type = moment.output_type(&types)?;
        if convention == MomentEstimatorConvention::UnbiasedIid {
            validate_iid(map, moment, weights)?;
        }
        Ok(Self {
            map,
            moment,
            outputs: selected,
            weights,
            convention,
            value_type,
        })
    }

    /// Exact complete membership, sample metadata and numerical acceptance evidence.
    #[must_use]
    pub const fn map(&self) -> &'a CompleteEvaluationMap {
        self.map
    }
    /// Mathematical moment whose finite-sample convention is bound here.
    #[must_use]
    pub const fn moment(&self) -> ProbabilityMoment {
        self.moment
    }
    /// Selected output positions in the exact Program, in paired moment order.
    #[must_use]
    pub fn output_indices(&self) -> &[usize] {
        &self.outputs
    }
    /// Proportional weights in accepted occurrence order, including zeros.
    #[must_use]
    pub const fn weights(&self) -> &'a [f64] {
        self.weights
    }
    /// Explicit empirical or caller-assumed IID interpretation.
    #[must_use]
    pub const fn convention(&self) -> MomentEstimatorConvention {
        self.convention
    }
    /// Typed result units, independently derived from the selected outputs.
    #[must_use]
    pub const fn value_type(&self) -> &ValueType {
        &self.value_type
    }
}

fn validate_iid(
    map: &CompleteEvaluationMap,
    moment: ProbabilityMoment,
    weights: &[f64],
) -> Result<(), Diagnostic> {
    if (moment != ProbabilityMoment::Expectation && map.len() < 2)
        || weights[0] <= 0.0
        || weights.iter().any(|weight| *weight != weights[0])
    {
        return Err(invalid(
            "unbiased IID moments require equal positive weights and at least two occurrences for variance/covariance",
        ));
    }
    let samples = map
        .plan()
        .samples()
        .ok_or_else(|| invalid("IID estimator requires explicit frozen sample-design metadata"))?;
    let first = &samples[0];
    let mut ids = BTreeSet::new();
    let mut streams = BTreeSet::new();
    for sample in samples {
        if sample.identity().coupling_id().is_some()
            || !ids.insert(sample.identity().sample_id())
            || !streams.insert(sample.stream_digest())
            || sample.generator() != first.generator()
            || sample.master_stream() != first.master_stream()
            || sample.identity().channel() != first.identity().channel()
            || sample.bounds() != first.bounds()
        {
            return Err(invalid(
                "IID estimator requires distinct uncoupled sample identities and streams from one declared sampling design",
            ));
        }
    }
    Ok(())
}
