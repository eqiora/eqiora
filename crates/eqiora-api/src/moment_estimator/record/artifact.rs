//! Collection-local statistic content; the shared typed literal codec owns its value.
use super::*;
use eqiora_artifact::{WireValueLiteral, encode_execution_artifact};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireEstimator {
    identity: String,
    content: Content,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Content {
    request: String,
    occurrences: Vec<String>,
    references: Vec<String>,
    selection: WireSelection,
    reduction: WireReduction,
    weights: Vec<f64>,
    value: WireValueLiteral,
    gradient: WireGradient,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case", deny_unknown_fields)]
enum WireSelection {
    Program { outputs: Vec<usize> },
    Time { time_s: f64, outputs: Vec<String> },
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireReduction {
    Sum,
    Empirical { moment: WireMoment },
    UnbiasedIid { moment: WireMoment },
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireMoment {
    Expectation,
    Variance,
    Covariance,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireGradient {
    FixedSamples,
    PathwiseLocationScale {
        distribution: String,
        variable: String,
        location: String,
        scale: String,
    },
}

impl RecordedMomentEstimator {
    pub(crate) fn wire(&self) -> Result<WireEstimator, Diagnostic> {
        let reduction = match self.reduction {
            MomentReduction::Sum => WireReduction::Sum,
            MomentReduction::Empirical(moment) => WireReduction::Empirical {
                moment: moment.into(),
            },
            MomentReduction::UnbiasedIid(moment) => WireReduction::UnbiasedIid {
                moment: moment.into(),
            },
            MomentReduction::Population(_) => {
                return Err(invalid("population statistic has no sampled collection"));
            }
        };
        let gradient = match &self.gradient {
            GradientEstimator::FixedSamples => WireGradient::FixedSamples,
            GradientEstimator::PathwiseLocationScale {
                distribution,
                variable,
                location,
                scale,
            } => WireGradient::PathwiseLocationScale {
                distribution: distribution.ulid().to_string(),
                variable: variable.ulid().to_string(),
                location: location.ulid().to_string(),
                scale: scale.ulid().to_string(),
            },
            GradientEstimator::EnumeratedScore { .. } => {
                return Err(invalid("finite score has no sampled collection"));
            }
        };
        Ok(WireEstimator {
            identity: self.identity.clone(),
            content: Content {
                request: self.request.clone(),
                occurrences: self.occurrences.iter().map(ToString::to_string).collect(),
                references: self.references.clone(),
                selection: match &self.selection {
                    Selection::Program(outputs) => WireSelection::Program {
                        outputs: outputs.clone(),
                    },
                    Selection::Time { time_s, outputs } => WireSelection::Time {
                        time_s: *time_s,
                        outputs: outputs.iter().map(|id| id.ulid().to_string()).collect(),
                    },
                },
                reduction,
                weights: self.weights.clone(),
                value: WireValueLiteral::encode(&self.value)?,
                gradient,
            },
        })
    }
}
impl WireEstimator {
    pub(crate) fn content_identity(&self) -> Result<String, Diagnostic> {
        let bytes = encode_execution_artifact(&self.content)
            .map_err(|_| invalid("cannot encode recorded statistic"))?;
        Ok(
            ArtifactDigest::compute(b"eqiora.recorded-moment-estimator/v1", &bytes)
                .as_str()
                .to_owned(),
        )
    }
    pub(crate) fn replay(
        self,
        group: &EvaluationCollectionGroup,
    ) -> Result<RecordedMomentEstimator, Diagnostic> {
        if self.identity != self.content_identity()? {
            return Err(invalid(
                "recorded statistic identity differs from its content",
            ));
        }
        let content = self.content;
        let result = RecordedMomentEstimator {
            identity: self.identity,
            request: content.request,
            occurrences: content
                .occurrences
                .iter()
                .map(|value| value.parse())
                .collect::<Result<_, _>>()?,
            references: content.references,
            selection: match content.selection {
                WireSelection::Program { outputs } => Selection::Program(outputs),
                WireSelection::Time { time_s, outputs } => Selection::Time {
                    time_s,
                    outputs: outputs
                        .iter()
                        .map(|value| id(value))
                        .collect::<Result<_, _>>()?,
                },
            },
            reduction: match content.reduction {
                WireReduction::Sum => MomentReduction::Sum,
                WireReduction::Empirical { moment } => MomentReduction::Empirical(moment.into()),
                WireReduction::UnbiasedIid { moment } => {
                    MomentReduction::UnbiasedIid(moment.into())
                }
            },
            weights: content.weights,
            value: content.value.decode()?,
            gradient: match content.gradient {
                WireGradient::FixedSamples => GradientEstimator::FixedSamples,
                WireGradient::PathwiseLocationScale {
                    distribution,
                    variable,
                    location,
                    scale,
                } => GradientEstimator::PathwiseLocationScale {
                    distribution: id(&distribution)?,
                    variable: id(&variable)?,
                    location: id(&location)?,
                    scale: id(&scale)?,
                },
            },
        };
        result.validate(group)?;
        Ok(result)
    }
}
fn id<E: eqiora_core::entity::Entity>(value: &str) -> Result<Id<E>, Diagnostic> {
    let ulid: ulid::Ulid = value
        .parse()
        .map_err(|_| invalid("invalid recorded statistic entity identity"))?;
    if ulid.to_string() != value {
        return Err(invalid("noncanonical recorded statistic entity identity"));
    }
    Ok(Id::from_ulid(ulid))
}
impl From<ProbabilityMoment> for WireMoment {
    fn from(value: ProbabilityMoment) -> Self {
        match value {
            ProbabilityMoment::Expectation => Self::Expectation,
            ProbabilityMoment::Variance => Self::Variance,
            ProbabilityMoment::Covariance => Self::Covariance,
        }
    }
}
impl From<WireMoment> for ProbabilityMoment {
    fn from(value: WireMoment) -> Self {
        match value {
            WireMoment::Expectation => Self::Expectation,
            WireMoment::Variance => Self::Variance,
            WireMoment::Covariance => Self::Covariance,
        }
    }
}
