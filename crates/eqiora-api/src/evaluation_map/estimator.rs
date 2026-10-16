//! Finite statistics retain their complete accepted map and exact convention.
use super::{CompleteEvaluationMap, EvaluationMapProducts, EvaluationMapResult, invalid};
use crate::{DifferentiableCotangent, DifferentiableEvaluation, ModelEntityRef};
use eqiora_core::{Diagnostic, ValueLiteral, ValueType};
use eqiora_schema::kernel::{ProbabilityMoment, WeightedMoments};
use std::{borrow::Cow, collections::BTreeSet};

mod pathwise;

/// Mathematical interpretation retained with a statistic and its first derivative.
/// Exact Model identity, membership, weights and reduction remain in the estimator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientEstimator {
    /// Differentiate the exact finite objective with frozen samples and weights.
    FixedSamples,
    /// Pathwise location-scale expectation estimate under explicitly assumed local
    /// domination and finite moments; structural admission binds the exact law.
    PathwiseLocationScale {
        /// Fixed base probability law in the owning exact Model.
        distribution: eqiora_core::Id<eqiora_core::entity::kinds::Relation>,
        /// Declared random variable `location + scale * base_noise`.
        variable: eqiora_core::Id<eqiora_core::entity::kinds::Observable>,
        /// Active location Parameter.
        location: eqiora_core::Id<eqiora_core::entity::kinds::Parameter>,
        /// Active positive scale Parameter.
        scale: eqiora_core::Id<eqiora_core::entity::kinds::Parameter>,
    },
}

/// Finite collection operation and its explicit statistical interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleReduction {
    /// Sum the weighted values without normalization or any IID claim.
    Sum,
    /// Normalize explicit nonnegative weights; duplicate occurrences retain
    /// multiplicity. No independence, importance-sampling or unbiasedness claim.
    Empirical(ProbabilityMoment),
    /// The caller assumes identically distributed independent outputs and the
    /// required finite moments. Equal positive weights and the n-1 divisor are
    /// required for variance/covariance. Stream metadata cannot prove IID.
    UnbiasedIid(ProbabilityMoment),
}

impl EnsembleReduction {
    fn moment(self) -> ProbabilityMoment {
        match self {
            Self::Sum => ProbabilityMoment::Expectation,
            Self::Empirical(moment) | Self::UnbiasedIid(moment) => moment,
        }
    }
    fn correction(self) -> usize {
        usize::from(matches!(
            self,
            Self::UnbiasedIid(ProbabilityMoment::Variance | ProbabilityMoment::Covariance)
        ))
    }
}

/// A computed finite statistic bound to its complete accepted occurrence axis.
///
/// Every occurrence participates, including duplicates and zero-weight members;
/// there is no success-only filtering. Selected outputs are scalar quantities,
/// not additional sample axes. The exact Program/Plan, sample metadata, weights
/// and reduction remain inspectable alongside the computed typed value.
/// Arithmetic streams through one accepted member at a time. A Recompute map
/// reaccepts each frozen point against its original receipt without resampling.
/// First-order results retain the statistic but cannot be differentiated again:
///
/// ```compile_fail
/// use eqiora_api::{EvaluationMapEstimator, EvaluationMapResult, DifferentiableCotangent};
/// fn second(first: EvaluationMapEstimator<'_, EvaluationMapResult<DifferentiableCotangent>>) {
///     first.vjp(1024);
/// }
/// ```
#[derive(Debug)]
pub struct EvaluationMapEstimator<'a, T = ()> {
    map: &'a CompleteEvaluationMap,
    reduction: EnsembleReduction,
    outputs: Vec<usize>,
    weights: &'a [f64],
    moments: std::sync::Arc<WeightedMoments>,
    value: ValueLiteral,
    gradient_estimator: GradientEstimator,
    derivative: T,
}

impl<'a> EvaluationMapEstimator<'a> {
    /// Reduce every accepted occurrence with one fixed nonnegative weight each.
    ///
    /// Sum is unnormalized; an empirical expectation is the weighted arithmetic
    /// mean. Values use bounded exact rational accumulation and round once. No
    /// population-expectation or confidence-interval claim follows from execution.
    /// # Errors
    /// Rejects incomplete/empty membership, foreign or unsupported outputs, invalid
    /// weights/design, failed recomputation, overflow or exhausted arithmetic work.
    pub fn evaluate(
        map: &'a CompleteEvaluationMap,
        reduction: EnsembleReduction,
        outputs: &[ModelEntityRef],
        weights: &'a [f64],
    ) -> Result<Self, Diagnostic> {
        map.validate()?;
        let moment = reduction.moment();
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
            types.push(signature.mathematical_type().value_type.clone());
        }
        let mut moments = WeightedMoments::new(moment, &types)?;
        if matches!(reduction, EnsembleReduction::UnbiasedIid(_)) {
            validate_iid(map, moment, weights)?;
        }
        for (index, &weight) in weights.iter().enumerate() {
            let member = map.accepted_member(index)?;
            moments.push(&row(&member, &selected)?, weight)?;
        }
        let value = if reduction == EnsembleReduction::Sum {
            moments.sum()?
        } else {
            moments.value(reduction.correction())?
        };
        Ok(Self {
            map,
            reduction,
            outputs: selected,
            weights,
            moments: std::sync::Arc::new(moments),
            value,
            gradient_estimator: GradientEstimator::FixedSamples,
            derivative: (),
        })
    }
}

impl<'a, T> EvaluationMapEstimator<'a, T> {
    /// Explicit finite-objective or admitted expectation-gradient interpretation.
    #[must_use]
    pub const fn gradient_estimator(&self) -> &GradientEstimator {
        &self.gradient_estimator
    }
    /// Computed first-order action, or `()` before differentiation.
    #[must_use]
    pub const fn derivative(&self) -> &T {
        &self.derivative
    }
    /// Complete membership, exact sampling design and original acceptance evidence.
    #[must_use]
    pub const fn map(&self) -> &'a CompleteEvaluationMap {
        self.map
    }
    /// Exact collection operation and declared statistical convention.
    #[must_use]
    pub const fn reduction(&self) -> EnsembleReduction {
        self.reduction
    }
    /// Ordered output positions, preserving covariance-pair order and repetition.
    #[must_use]
    pub fn output_indices(&self) -> &[usize] {
        &self.outputs
    }
    /// Fixed weights in accepted occurrence order, including zeros.
    #[must_use]
    pub const fn weights(&self) -> &'a [f64] {
        self.weights
    }
    /// Computed finite statistic, distinct from an exact population moment.
    #[must_use]
    pub const fn value(&self) -> &ValueLiteral {
        &self.value
    }
    /// Units derived from the selected output quantities and reduction.
    #[must_use]
    pub fn value_type(&self) -> &ValueType {
        self.value.value_type()
    }

    /// Estimated standard error of an explicitly declared equal-weight IID mean.
    ///
    /// Requires the caller's finite population second-moment assumption. Returns
    /// sqrt(sample variance / n), retaining the mean's units. This estimates
    /// uncertainty, not a confidence interval or a bound on population error.
    /// # Errors
    /// Rejects empirical/weighted interpretations, non-mean reductions, n < 2
    /// and unrepresentable results. Stream metadata alone never establishes IID.
    pub fn standard_error(&self) -> Result<ValueLiteral, Diagnostic> {
        if self.reduction != EnsembleReduction::UnbiasedIid(ProbabilityMoment::Expectation) {
            return Err(invalid(
                "standard error requires an explicit equal-weight IID mean",
            ));
        }
        self.moments.mean_standard_error()
    }
}

impl<'a> EvaluationMapEstimator<'a> {
    /// Differentiate the statistic and retain its explicit estimator convention.
    ///
    /// Weights and sample design stay fixed. Existing mapped VJP execution owns
    /// exact point acceptance, shared-input sums and mapped covectors. Returned
    /// coordinates have statistic-unit/input-unit; base signatures retain input
    /// spaces. The default is finite-objective AD; a pathwise interpretation
    /// requires separate structural admission and the caller's analytic assumptions.
    /// The result retains the complete statistic context alongside its action. Row seeds
    /// are rounded once and shared gradients follow the existing ordered finite
    /// accumulation/overflow policy, not arbitrary-order bit reproducibility.
    ///
    /// The numerical limit has the existing mapped-product scope, plus one
    /// output seed buffer; it excludes the already computed statistic, metadata
    /// and transient solver scratch. Exact sufficient statistics are shared.
    /// # Errors
    /// Rejects insufficient storage, failed member acceptance/derivatives and
    /// nonfinite seeds or shared sums. No partial gradient is returned.
    pub fn vjp(
        &self,
        numerical_bytes_limit: usize,
    ) -> Result<EvaluationMapEstimator<'a, EvaluationMapResult<DifferentiableCotangent>>, Diagnostic>
    {
        let identity = self.map.plan().program_identity();
        let seed_bytes = identity
            .output_dimension()
            .checked_mul(size_of::<f64>())
            .ok_or_else(|| invalid("statistic seed storage overflow"))?;
        let limit = numerical_bytes_limit
            .checked_sub(seed_bytes)
            .ok_or_else(|| invalid("statistic seed exceeds numerical byte limit"))?;
        let products = EvaluationMapProducts::new(self.map, &[self.map.len()], &[], &[0], limit)?;
        let offsets = crate::differentiation::signature::layout(identity.outputs())
            .map(|(_, range)| range.start)
            .collect::<Vec<_>>();
        let signatures = identity.outputs().to_vec().into();
        let derivative = products.vjp_with(|index, member| {
            let weight = self.weights[index];
            let coefficients = if self.reduction == EnsembleReduction::Sum {
                vec![weight]
            } else {
                self.moments.derivative(
                    &row(member, &self.outputs)?,
                    weight,
                    self.reduction.correction(),
                )?
            };
            let mut seed = vec![0.0; identity.output_dimension()];
            for (&output, coefficient) in self.outputs.iter().zip(coefficients) {
                seed[offsets[output]] += coefficient;
            }
            DifferentiableCotangent::from_coordinates(std::sync::Arc::clone(&signatures), seed)
                .map(Cow::Owned)
        })?;
        Ok(EvaluationMapEstimator {
            map: self.map,
            reduction: self.reduction,
            outputs: self.outputs.clone(),
            weights: self.weights,
            moments: self.moments.clone(),
            value: self.value.clone(),
            gradient_estimator: self.gradient_estimator.clone(),
            derivative,
        })
    }
}

fn row(
    member: &DifferentiableEvaluation,
    outputs: &[usize],
) -> Result<Vec<ValueLiteral>, Diagnostic> {
    let primal = member.primal();
    outputs
        .iter()
        .map(|&output| {
            primal
                .output()
                .leaf(output)
                .ok_or_else(|| invalid("statistic output missing from accepted member"))?
                .coefficient(0)
        })
        .collect()
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
            || sample.fixed_values() != first.fixed_values()
        {
            return Err(invalid(
                "IID estimator requires distinct uncoupled sample identities and streams from one declared sampling design",
            ));
        }
    }
    Ok(())
}
