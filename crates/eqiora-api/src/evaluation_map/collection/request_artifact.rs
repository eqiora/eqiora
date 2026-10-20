//! Exact request codecs, never reconstruction of an executable map.
use super::artifact::{decode, one};
use super::*;
use base64::{Engine, engine::general_purpose::STANDARD};
use eqiora_numerics::{
    CommonInitialState, CommonTimeRunRequest, RecordedTimeCheckpoint, ResolvedCommonPlan,
};
use eqiora_solver::LinearSolverBackend;
use eqiora_time::TimeBackendCapabilities;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireRequest {
    Evaluations {
        plan: String,
        program: String,
        initial: Option<String>,
        points: Vec<Vec<f64>>,
        shared: Vec<String>,
        samples: Option<Vec<String>>,
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
                plan,
                program,
                initial,
                points,
                shared,
                samples,
            } => WireRequest::Evaluations {
                plan: STANDARD.encode(plan.to_bytes()?),
                program: STANDARD.encode(program.to_bytes()?),
                initial: initial
                    .as_ref()
                    .map(|state| state.to_bytes().map(|bytes| STANDARD.encode(bytes)))
                    .transpose()?,
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
            ArtifactDigest::compute(b"eqiora.evaluation-collection-request/v1", &bytes)
                .as_str()
                .to_owned(),
        )
    }

    pub(super) fn replay(
        self,
        backend: &dyn LinearSolverBackend,
        time_backend: TimeBackendCapabilities,
    ) -> Result<RecordedRequest, Vec<Diagnostic>> {
        let plan_bytes = match &self {
            Self::Evaluations { plan, .. } | Self::Time { plan, .. } => plan,
        };
        let plan = ResolvedCommonPlan::from_bytes(
            &decode(plan_bytes).map_err(one)?,
            backend,
            time_backend,
        )
        .map_err(one)?;
        let request = match self {
            Self::Evaluations {
                program,
                initial,
                points,
                shared,
                samples,
                ..
            } => {
                let initial = initial
                    .map(|value| CommonInitialState::from_bytes(&decode(&value)?, &plan))
                    .transpose()
                    .map_err(one)?;
                let program = DifferentiableProgramIdentity::from_bytes(
                    &decode(&program).map_err(one)?,
                    &plan,
                    initial.as_ref(),
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
                    plan,
                    program: Box::new(program),
                    initial,
                    points,
                    shared,
                    samples,
                }
            }
            Self::Time {
                requests,
                checkpoints,
                ..
            } => {
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
