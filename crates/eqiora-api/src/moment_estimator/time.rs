//! Typed statistics over explicitly supplied accepted time Results.
use super::*;
use eqiora_artifact::{CanonicalModelArtifact, ModelArtifactReference};
use eqiora_core::{Id, entity::kinds};
use eqiora_numerics::{CommonResult, CommonTimeRunRequest};
use std::{collections::HashMap, sync::Arc};

type Output = ModelEntityRef<Id<kinds::Observable>>;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub(super) struct TimeSamples {
    pub(super) model: ModelArtifactReference,
    results: Arc<[CommonResult]>,
    time_s: f64,
    outputs: Vec<Output>,
}

impl MomentEstimator<'static> {
    /// Reduce the explicitly supplied accepted Results at one exact retained time.
    ///
    /// Every occurrence participates, including duplicates and zero weights. This
    /// operation neither executes members nor filters failed Runs: callers supply
    /// their complete collection of successful Results. Every Result retains its
    /// original request, physical initial State, sample identity and accepted history.
    /// All members must use one exact finite-time Plan and retain the requested
    /// observation time. Observations use the ordinary Result owner, including its
    /// post-event convention; there is no interpolation or retiming here.
    ///
    /// UnbiasedIid additionally checks equal positive weights, distinct uncoupled
    /// samples, one master seed/channel, and identical initial State and horizon.
    /// The caller still assumes IID and finite required moments. Metadata cannot
    /// prove independence. No trajectory derivative is admitted by this operation.
    ///
    /// # Errors
    /// Rejects empty or mismatched membership, population interpretations, foreign
    /// Plans/outputs, unavailable times, invalid sampling designs and arithmetic
    /// failures. The existing weighted-moment owner controls units and rounding.
    pub fn from_time_results(
        results: Arc<[CommonResult]>,
        time_s: f64,
        reduction: MomentReduction,
        outputs: &[ModelEntityRef<Id<kinds::Observable>>],
        weights: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        let moment = reduction.moment();
        if results.is_empty()
            || results.len() != weights.len()
            || outputs.len() != moment.arity()
            || !time_s.is_finite()
            || matches!(reduction, MomentReduction::Population(_))
        {
            return Err(invalid(
                "time statistic requires nonempty complete membership, matching weights, finite time and a sample reduction with the selected output arity",
            ));
        }
        ProbabilityMoment::validate_weights(weights.iter().copied())?;
        let first = request(&results[0])?;
        let model = first.plan().model_artifact().artifact_reference()?;
        if outputs.iter().any(|output| output.model() != &model) {
            return Err(invalid("time statistic output belongs to a foreign Model"));
        }
        if matches!(reduction, MomentReduction::UnbiasedIid(_)) {
            validate_iid_weights(moment, &weights)?;
            validate_design(results.iter().map(request).collect::<Result<Vec<_>, _>>()?)?;
        }
        let mut moments = None;
        for (result, &weight) in results.iter().zip(&weights) {
            let current = request(result)?;
            if current.plan().identity() != first.plan().identity() {
                return Err(invalid(
                    "time statistic requires one exact finite-time Plan",
                ));
            }
            let output_index = current
                .output_times_s()
                .iter()
                .position(|time| *time == time_s)
                .ok_or_else(|| {
                    invalid("statistic time is not a retained output of every member")
                })?;
            let row = outputs
                .iter()
                .map(|output| {
                    result
                        .observe(
                            current.plan().model_artifact(),
                            output.id(),
                            &HashMap::new(),
                            Some(output_index),
                        )
                        .map(|observation| observation.value().clone())
                })
                .collect::<Result<Vec<_>, _>>()?;
            if moments.is_none() {
                moments = Some(WeightedMoments::new(
                    moment,
                    &row.iter()
                        .map(|value| value.value_type().clone())
                        .collect::<Vec<_>>(),
                )?);
            }
            moments
                .as_mut()
                .expect("nonempty accepted row")
                .push(&row, weight)?;
        }
        let moments = Arc::new(moments.expect("nonempty complete collection"));
        let value = reduction.reduce(&moments)?;
        Ok(Self {
            source: Source::Time(TimeSamples {
                model,
                results,
                time_s,
                outputs: outputs.to_vec(),
            }),
            reduction,
            outputs: Vec::new(),
            weights: Cow::Owned(weights),
            moments,
            value,
            gradient_estimator: GradientEstimator::FixedSamples,
            derivative: (),
        })
    }
}

impl<T> MomentEstimator<'_, T> {
    /// Complete accepted time membership in supplied occurrence order.
    #[must_use]
    pub fn time_results(&self) -> Option<&[CommonResult]> {
        match &self.source {
            Source::Time(samples) => Some(&samples.results),
            _ => None,
        }
    }

    /// Exact retained physical time and ordered Observable selection.
    #[must_use]
    pub fn time_observation(&self) -> Option<(f64, &[ModelEntityRef<Id<kinds::Observable>>])> {
        match &self.source {
            Source::Time(samples) => Some((samples.time_s, &samples.outputs)),
            _ => None,
        }
    }
}

fn request(result: &CommonResult) -> Result<&CommonTimeRunRequest, Diagnostic> {
    result
        .trajectory()
        .and_then(|trajectory| trajectory.time_request())
        .ok_or_else(|| invalid("time statistics require accepted finite-time trajectories"))
}

pub(super) fn validate_design<'a>(
    requests: impl IntoIterator<Item = &'a CommonTimeRunRequest>,
) -> Result<(), Diagnostic> {
    let mut requests = requests.into_iter();
    let first = requests
        .next()
        .ok_or_else(|| invalid("IID time statistic requires nonempty requests"))?;
    let sampling = first
        .sampling()
        .ok_or_else(|| invalid("IID time statistic requires explicit sample identities"))?;
    let mut identities = BTreeSet::new();
    for current in std::iter::once(first).chain(requests) {
        let (seed, identity) = current
            .sampling()
            .ok_or_else(|| invalid("IID time statistic requires explicit sample identities"))?;
        if seed != &sampling.0
            || identity.channel() != sampling.1.channel()
            || identity.coupling_id().is_some()
            || !identities.insert(identity.sample_id())
            || current.state() != first.state()
            || current.until_s() != first.until_s()
        {
            return Err(invalid(
                "IID time statistic requires distinct uncoupled samples from one initial State, horizon and master seed/channel",
            ));
        }
    }
    Ok(())
}
