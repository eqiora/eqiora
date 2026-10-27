//! Exact request codecs, never reconstruction of an executable map.
use super::artifact::{decode, one};
use super::*;
use crate::differentiation::context::WireContext;
use base64::{Engine, engine::general_purpose::STANDARD};
use eqiora_numerics::{CommonTimeRunRequest, RecordedTimeCheckpoint, ResolvedCommonPlan};
use eqiora_solver::LinearSolverBackend;
use eqiora_time::TimeBackendCapabilities;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireRequest {
    Evaluations {
        context: WireContext,
        program: String,
        points: Vec<Vec<f64>>,
        shared: Vec<String>,
        samples: Option<Vec<String>>,
    },
    Parameters {
        template: String,
        parameters: Vec<String>,
        points: Vec<Vec<eqiora_artifact::WireValueLiteral>>,
        policy: (usize, usize, bool, usize, usize),
    },
    Time {
        plan: String,
        requests: Vec<String>,
        checkpoints: Vec<Option<String>>,
    },
}

impl RecordedRequest {
    pub(super) fn wire(&self) -> Result<WireRequest, Diagnostic> {
        Ok(match self {
            Self::Evaluations {
                context,
                program,
                points,
                shared,
                samples,
            } => WireRequest::Evaluations {
                context: context.wire()?,
                program: STANDARD.encode(program.to_bytes()?),
                points: points
                    .iter()
                    .map(|point| point.coordinates().to_vec())
                    .collect(),
                shared: shared.iter().map(|id| id.erase().to_string()).collect(),
                samples: samples
                    .as_ref()
                    .map(|samples| {
                        samples
                            .iter()
                            .map(|sample| sample.to_bytes().map(|bytes| STANDARD.encode(bytes)))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?,
            },
            Self::Parameters {
                template,
                parameters,
                points,
                policy,
                member_bytes_limit,
            } => WireRequest::Parameters {
                template: STANDARD.encode(template.to_bytes()?),
                parameters: parameters.iter().map(|id| id.ulid().to_string()).collect(),
                points: points
                    .iter()
                    .map(|point| {
                        point
                            .iter()
                            .map(eqiora_artifact::WireValueLiteral::encode)
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<_, _>>()?,
                policy: (
                    policy.chunk_size(),
                    policy.workers(),
                    policy.retention() == EvaluationMapRetention::Retain,
                    policy.storage_bytes_limit(),
                    *member_bytes_limit,
                ),
            },
            Self::Time {
                plan,
                requests,
                checkpoints,
            } => WireRequest::Time {
                plan: STANDARD.encode(plan.to_bytes()?),
                requests: requests
                    .iter()
                    .map(|request| request.to_bytes().map(|bytes| STANDARD.encode(bytes)))
                    .collect::<Result<_, _>>()?,
                checkpoints: checkpoints
                    .iter()
                    .map(|checkpoint| {
                        checkpoint
                            .as_ref()
                            .map(|value| STANDARD.encode(value.as_bytes()))
                    })
                    .collect(),
            },
        })
    }
}

impl WireRequest {
    pub(super) fn identity(&self) -> Result<String, Diagnostic> {
        let bytes = eqiora_artifact::JsonLimits::EXECUTION
            .encode(self)
            .map_err(|_| invalid("cannot encode collection request"))?;
        Ok(
            ArtifactDigest::compute(b"eqiora.evaluation-collection-request/v2", &bytes)
                .as_str()
                .to_owned(),
        )
    }

    pub(super) fn replay(
        self,
        backend: &dyn LinearSolverBackend,
        time_backend: &[TimeBackendCapabilities],
    ) -> Result<RecordedRequest, Vec<Diagnostic>> {
        let request = match self {
            Self::Evaluations {
                program,
                context,
                points,
                shared,
                samples,
                ..
            } => {
                let context = context.replay(backend, time_backend).map_err(one)?;
                let program = DifferentiableProgramIdentity::from_context_bytes(
                    &decode(&program).map_err(one)?,
                    &context,
                )?;
                let points = points
                    .into_iter()
                    .map(|values| {
                        DifferentiableValues::from_coordinates(
                            program.inputs().to_vec().into(),
                            values,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(one)?;
                let shared = shared
                    .iter()
                    .map(|id| {
                        program
                            .inputs()
                            .iter()
                            .find(|input| input.entity().to_string() == *id)
                            .map(|input| input.parameter())
                            .ok_or_else(|| {
                                invalid("collection shared Parameter is outside its exact Program")
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(one)?;
                let samples = samples
                    .map(|samples| {
                        samples
                            .into_iter()
                            .map(|bytes| {
                                SampledParameterPoint::from_bytes(&decode(&bytes)?, &program)
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()
                    .map_err(one)?;
                RecordedRequest::Evaluations {
                    context: Box::new(context),
                    program: Box::new(program),
                    points,
                    shared,
                    samples,
                }
            }
            Self::Parameters {
                template,
                parameters: selected,
                points,
                policy,
            } => {
                let template = ResolvedCommonPlan::from_bytes(
                    &decode(&template).map_err(one)?,
                    backend,
                    time_backend,
                )
                .map_err(one)?;
                let parameters = selected
                    .iter()
                    .map(|id| {
                        id.parse::<ulid::Ulid>()
                            .map(Parameter::from_ulid)
                            .map_err(|_| one(invalid("invalid stationary map Parameter identity")))
                    })
                    .collect::<Result<_, _>>()?;
                let points = points
                    .iter()
                    .map(|point| {
                        point
                            .iter()
                            .map(eqiora_artifact::WireValueLiteral::decode)
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<_, _>>()
                    .map_err(one)?;
                let (chunk, workers, retain, limit, member_bytes_limit) = policy;
                let policy = EvaluationMapExecutionPolicy::new(
                    chunk,
                    eqiora_realization::Target::HostCpu {
                        threads: std::num::NonZeroUsize::new(workers)
                            .ok_or_else(|| one(invalid("invalid stationary map worker count")))?,
                    },
                    if retain {
                        EvaluationMapRetention::Retain
                    } else {
                        EvaluationMapRetention::Recompute
                    },
                    limit,
                )
                .map_err(one)?;
                RecordedRequest::Parameters {
                    template,
                    parameters,
                    points,
                    policy,
                    member_bytes_limit,
                }
            }
            Self::Time {
                plan,
                requests,
                checkpoints,
            } => {
                let plan = ResolvedCommonPlan::from_bytes(
                    &decode(&plan).map_err(one)?,
                    backend,
                    time_backend,
                )
                .map_err(one)?;
                let time_plan = plan
                    .as_time()
                    .ok_or_else(|| one(invalid("collection time requests require a time Plan")))?;
                let requests = requests
                    .into_iter()
                    .map(|value| CommonTimeRunRequest::from_bytes(&decode(&value)?, time_plan))
                    .collect::<Result<_, _>>()
                    .map_err(one)?;
                let checkpoints = checkpoints
                    .into_iter()
                    .map(|value| {
                        value
                            .map(|value| {
                                RecordedTimeCheckpoint::from_bytes(&decode(&value)?, time_plan)
                            })
                            .transpose()
                    })
                    .collect::<Result<_, _>>()
                    .map_err(one)?;
                RecordedRequest::Time {
                    plan,
                    requests,
                    checkpoints,
                }
            }
        };
        request.validate().map_err(one)?;
        Ok(request)
    }
}
