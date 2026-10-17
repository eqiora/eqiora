//! Immutable computed statistics attributed to one complete execution group.
use super::*;
use crate::{EvaluationCollectionGroup, EvaluationMapOccurrenceId, RecordedEvaluationOutcome};
use eqiora_artifact::{ArtifactDigest, ReplayableCanonicalModelArtifact};
use eqiora_core::{Id, entity::kinds};

pub(crate) mod artifact;

/// A previously computed typed statistic, with its exact complete group membership.
///
/// Reading validates attribution, units and the estimator's structural assumptions,
/// without recomputing the statistic or authenticating a historical producer. This
/// is read-only provenance, not an executable estimator or a derivative product.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedMomentEstimator {
    identity: String,
    request: String,
    occurrences: Vec<EvaluationMapOccurrenceId>,
    references: Vec<String>,
    selection: Selection,
    reduction: MomentReduction,
    weights: Vec<f64>,
    value: ValueLiteral,
    gradient: GradientEstimator,
}

#[derive(Debug, Clone, PartialEq)]
enum Selection {
    Program(Vec<usize>),
    Time {
        time_s: f64,
        outputs: Vec<Id<kinds::Observable>>,
    },
}

impl RecordedMomentEstimator {
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn occurrences(&self) -> &[EvaluationMapOccurrenceId] {
        &self.occurrences
    }
    pub fn references(&self) -> &[String] {
        &self.references
    }
    pub fn reduction(&self) -> MomentReduction {
        self.reduction
    }
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
    pub fn value(&self) -> &ValueLiteral {
        &self.value
    }
    pub fn gradient_estimator(&self) -> &GradientEstimator {
        &self.gradient
    }
    pub fn output_indices(&self) -> Option<&[usize]> {
        match &self.selection {
            Selection::Program(outputs) => Some(outputs),
            _ => None,
        }
    }
    pub fn time_observation(&self) -> Option<(f64, &[Id<kinds::Observable>])> {
        match &self.selection {
            Selection::Time { time_s, outputs } => Some((*time_s, outputs)),
            _ => None,
        }
    }

    pub(crate) fn capture<T>(
        group: &EvaluationCollectionGroup,
        estimator: &MomentEstimator<'_, T>,
    ) -> Result<Self, Diagnostic> {
        let references = references(group)?;
        let selection = if let Some(map) = estimator.map() {
            if map.execution_id() != group.execution_id() {
                return Err(invalid("estimator belongs to a different execution group"));
            }
            Selection::Program(estimator.outputs.clone())
        } else if let Some(results) = estimator.time_results() {
            if results.len() != references.len()
                || results
                    .iter()
                    .zip(&references)
                    .any(|(result, reference)| result.identity() != reference)
            {
                return Err(invalid(
                    "estimator Results differ from the group's exact ordered membership",
                ));
            }
            let (time_s, outputs) = estimator.time_observation().expect("time estimator");
            Selection::Time {
                time_s,
                outputs: outputs.iter().map(|output| output.id()).collect(),
            }
        } else {
            return Err(invalid(
                "a declared-law estimator has no evaluation collection group",
            ));
        };
        let mut value = Self {
            identity: String::new(),
            request: group.request_identity()?,
            occurrences: (0..group.len())
                .map(|i| group.occurrence(i).expect("group inventory").0)
                .collect(),
            references,
            selection,
            reduction: estimator.reduction,
            weights: estimator.weights.to_vec(),
            value: estimator.value.clone(),
            gradient: estimator.gradient_estimator.clone(),
        };
        value.validate(group)?;
        value.identity = value.wire()?.content_identity()?;
        Ok(value)
    }

    pub(crate) fn validate(&self, group: &EvaluationCollectionGroup) -> Result<(), Diagnostic> {
        let references = references(group)?;
        if self.request != group.request_identity()?
            || self.references != references
            || self.occurrences.len() != group.len()
            || self
                .occurrences
                .iter()
                .enumerate()
                .any(|(i, id)| Some(*id) != group.occurrence(i).map(|v| v.0))
            || self.weights.len() != group.len()
        {
            return Err(invalid(
                "recorded estimator attribution differs from its complete execution group",
            ));
        }
        if matches!(self.reduction, MomentReduction::Population(_)) {
            return Err(invalid(
                "a collection statistic cannot claim exhaustive population membership",
            ));
        }
        ProbabilityMoment::validate_weights(self.weights.iter().copied())?;
        let moment = self.reduction.moment();
        if matches!(self.reduction, MomentReduction::UnbiasedIid(_)) {
            validate_iid_weights(moment, &self.weights)?;
        }
        let types = match &self.selection {
            Selection::Program(outputs) => {
                let identity = group
                    .program_identity()
                    .ok_or_else(|| invalid("statistic requires its exact Program"))?;
                if matches!(self.reduction, MomentReduction::UnbiasedIid(_)) {
                    validate_iid_samples(
                        group
                            .samples()
                            .ok_or_else(|| invalid("IID estimator requires frozen samples"))?,
                    )?;
                }
                let types = outputs
                    .iter()
                    .map(|&i| {
                        let output = identity
                            .outputs()
                            .get(i)
                            .ok_or_else(|| invalid("statistic output is outside its Program"))?;
                        if output.mathematical_type().support.is_some()
                            || !output.shape().is_empty()
                        {
                            return Err(invalid("statistic requires finite scalar outputs"));
                        }
                        Ok(output.mathematical_type().value_type.clone())
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                match &self.gradient {
                    GradientEstimator::FixedSamples => {}
                    GradientEstimator::PathwiseLocationScale {
                        variable,
                        location,
                        scale,
                        ..
                    } => {
                        let model = group.plan().model_artifact().replay_model()?;
                        let actual = pathwise::validate(
                            pathwise::SampleDesign {
                                identity,
                                kernel: model.program(),
                                shared: group.shared_inputs().expect("Program group"),
                                samples: group.samples(),
                            },
                            self.reduction,
                            outputs,
                            &self.weights,
                            *variable,
                            *location,
                            *scale,
                        )?;
                        if actual != self.gradient {
                            return Err(invalid(
                                "recorded gradient interpretation differs from its declared law",
                            ));
                        }
                    }
                    GradientEstimator::EnumeratedScore { .. } => {
                        return Err(invalid(
                            "sample collections cannot claim an exhaustive score estimator",
                        ));
                    }
                }
                types
            }
            Selection::Time { time_s, outputs } => {
                let requests = group
                    .time_requests()
                    .ok_or_else(|| invalid("time statistic requires a time execution group"))?;
                if !time_s.is_finite()
                    || requests
                        .iter()
                        .any(|request| !request.output_times_s().contains(time_s))
                    || self.gradient != GradientEstimator::FixedSamples
                {
                    return Err(invalid(
                        "time statistic must select a retained time without a trajectory derivative claim",
                    ));
                }
                if matches!(self.reduction, MomentReduction::UnbiasedIid(_)) {
                    time::validate_design(requests)?;
                }
                let model = group.plan().model_artifact().replay_model()?;
                outputs
                    .iter()
                    .map(|id| match model.program().node(id.erase()) {
                        Some(eqiora_schema::kernel::KernelNode::Observable(value)) => {
                            let typed = model
                                .program()
                                .symbol_type(eqiora_schema::kernel::SymbolRef::Observable(*id))
                                .ok_or_else(|| {
                                    invalid("time statistic Observable has no exact type")
                                })?;
                            if typed.support.is_some() {
                                return Err(invalid(
                                    "time statistic requires a finite scalar Observable",
                                ));
                            }
                            Ok(value.value_type().clone())
                        }
                        _ => Err(invalid(
                            "time statistic Observable is outside its exact Model",
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        if moment.output_type(&types.iter().collect::<Vec<_>>())? != *self.value.value_type() {
            return Err(invalid(
                "recorded statistic units or type differ from its selected outputs",
            ));
        }
        Ok(())
    }
}

fn references(group: &EvaluationCollectionGroup) -> Result<Vec<String>, Diagnostic> {
    if group.is_empty() || !group.is_complete() {
        return Err(invalid(
            "a recorded statistic requires a nonempty complete execution group",
        ));
    }
    (0..group.len())
        .map(|i| match group.occurrence(i).expect("group inventory").1 {
            RecordedEvaluationOutcome::Accepted(value) => Ok(value.identity().to_owned()),
            RecordedEvaluationOutcome::TimeAccepted(value) => Ok(value.identity().to_owned()),
            _ => Err(invalid("statistic cannot filter an incomplete execution")),
        })
        .collect()
}
