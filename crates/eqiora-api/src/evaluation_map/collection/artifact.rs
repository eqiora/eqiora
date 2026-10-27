//! Bounded root projection referencing the existing typed member artifacts.
use super::request_artifact::WireRequest;
use super::*;
use base64::{Engine, engine::general_purpose::STANDARD};
use eqiora_solver::LinearSolverBackend;
use eqiora_time::TimeBackendCapabilities;
use serde::{Deserialize, Serialize};

pub(super) const SCHEMA: &str = "eqiora.evaluation-collection/v4";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCollection {
    schema: String,
    complete: bool,
    groups: Vec<WireGroup>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGroup {
    execution: String,
    request_identity: String,
    request: WireRequest,
    members: Vec<WireMember>,
    records: BTreeMap<String, String>,
    estimators: Vec<crate::moment_estimator::record::artifact::WireEstimator>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMember {
    occurrence: String,
    outcome: WireOutcome,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "kebab-case", deny_unknown_fields)]
enum WireOutcome {
    Accepted {
        reference: String,
        plan: Option<String>,
    },
    Failed {
        diagnostics: Vec<DiagnosticRecord>,
    },
    Cancelled {
        checkpoint: Option<String>,
    },
    NotStarted,
}

impl EvaluationCollection {
    /// Encode exact grouped requests, occurrence attribution and referenced payloads.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        let groups = self
            .groups
            .iter()
            .map(WireGroup::capture)
            .collect::<Result<_, _>>()?;
        eqiora_artifact::JsonLimits::EXECUTION
            .encode(&WireCollection {
                schema: SCHEMA.to_owned(),
                complete: self.is_complete(),
                groups,
            })
            .map_err(|error| invalid(format!("cannot encode evaluation collection: {error}")))
    }

    /// Reopen exact Plans and member artifacts through their owning decoders.
    /// No solve, sampler draw or statistical reduction is performed. Checkpoints
    /// remain immutable references; actual restart admission is explicit.
    /// Supplied provider capabilities must match each Plan; no provider is selected here.
    pub fn from_bytes(
        bytes: &[u8],
        backend: &dyn LinearSolverBackend,
        time_backend: &[TimeBackendCapabilities],
    ) -> Result<Self, Vec<Diagnostic>> {
        eqiora_artifact::JsonLimits::EXECUTION
            .check(bytes)
            .map_err(one)?;
        let wire: WireCollection = serde_json::from_slice(bytes).map_err(|error| {
            one(invalid(format!(
                "invalid evaluation collection JSON: {error}"
            )))
        })?;
        if wire.schema != SCHEMA {
            return Err(one(invalid("unknown evaluation collection schema")));
        }
        let groups = wire
            .groups
            .into_iter()
            .map(|group| group.replay(backend, time_backend))
            .collect::<Result<_, _>>()?;
        let collection = Self::new(groups).map_err(one)?;
        if wire.complete != collection.is_complete() {
            return Err(one(invalid(
                "collection completion label differs from its full requested inventory",
            )));
        }
        if collection.to_bytes().map_err(one)? != bytes {
            return Err(one(invalid(
                "evaluation collection bytes are not canonical",
            )));
        }
        Ok(collection)
    }
}

impl EvaluationCollectionGroup {
    /// Ordered request content identity, independent of this execution and its outcomes.
    pub fn request_identity(&self) -> Result<String, Diagnostic> {
        self.request.wire()?.identity()
    }
}

impl WireGroup {
    fn capture(group: &EvaluationCollectionGroup) -> Result<Self, Diagnostic> {
        let mut records = BTreeMap::new();
        let mut members = Vec::with_capacity(group.len());
        for (index, value) in group.outcomes.iter().enumerate() {
            let outcome = match value {
                RecordedEvaluationOutcome::Accepted(record) => {
                    if !records.contains_key(record.identity()) {
                        records.insert(
                            record.identity().to_owned(),
                            STANDARD.encode(record.to_bytes()?),
                        );
                    }
                    WireOutcome::Accepted {
                        reference: record.identity().to_owned(),
                        plan: None,
                    }
                }
                RecordedEvaluationOutcome::RunAccepted(result) => {
                    if !records.contains_key(result.identity()) {
                        records.insert(
                            result.identity().to_owned(),
                            STANDARD.encode(result.to_bytes()?),
                        );
                    }
                    WireOutcome::Accepted {
                        reference: result.identity().to_owned(),
                        plan: matches!(&group.request, RecordedRequest::Parameters { .. })
                            .then(|| result.plan().to_bytes().map(|bytes| STANDARD.encode(bytes)))
                            .transpose()?,
                    }
                }
                RecordedEvaluationOutcome::Failed(diagnostics) => WireOutcome::Failed {
                    diagnostics: diagnostics.clone(),
                },
                RecordedEvaluationOutcome::Cancelled(checkpoint) => WireOutcome::Cancelled {
                    checkpoint: checkpoint
                        .as_ref()
                        .map(|value| STANDARD.encode(value.as_bytes())),
                },
                RecordedEvaluationOutcome::NotStarted => WireOutcome::NotStarted,
            };
            members.push(WireMember {
                occurrence: EvaluationMapOccurrenceId::new(group.execution, index).to_string(),
                outcome,
            });
        }
        let request = group.request.wire()?;
        Ok(Self {
            execution: group.execution.to_string(),
            request_identity: request.identity()?,
            request,
            members,
            records,
            estimators: group
                .estimators
                .iter()
                .map(|value| value.wire())
                .collect::<Result<_, _>>()?,
        })
    }

    fn replay(
        self,
        backend: &dyn LinearSolverBackend,
        time_backend: &[TimeBackendCapabilities],
    ) -> Result<EvaluationCollectionGroup, Vec<Diagnostic>> {
        let execution: ulid::Ulid = self
            .execution
            .parse()
            .map_err(|_| one(invalid("invalid collection execution identity")))?;
        if execution.to_string() != self.execution
            || self.request.identity().map_err(one)? != self.request_identity
        {
            return Err(one(invalid(
                "collection execution or request identity differs from canonical content",
            )));
        }
        let request = self.request.replay(backend, time_backend)?;
        if self.members.len() != request.len() {
            return Err(one(invalid("collection is missing planned occurrences")));
        }
        let mut outcomes = Vec::with_capacity(self.members.len());
        let mut used = BTreeSet::new();
        let mut accepted = BTreeMap::<String, RecordedEvaluationOutcome>::new();
        for (index, member) in self.members.into_iter().enumerate() {
            let id: EvaluationMapOccurrenceId = member.occurrence.parse().map_err(one)?;
            if id != EvaluationMapOccurrenceId::new(execution, index) {
                return Err(one(invalid(
                    "collection occurrence attribution is reordered, duplicated or foreign",
                )));
            }
            let outcome = match member.outcome {
                WireOutcome::Accepted {
                    reference,
                    plan: member_plan,
                } => {
                    let encoded = self.records.get(&reference).ok_or_else(|| {
                        one(invalid("collection accepted member reference is missing"))
                    })?;
                    used.insert(reference.clone());
                    if let Some(outcome) = accepted.get(&reference) {
                        let expected_plan = match (&request, outcome) {
                            (
                                RecordedRequest::Parameters { .. },
                                RecordedEvaluationOutcome::RunAccepted(result),
                            ) => Some(STANDARD.encode(result.plan().to_bytes().map_err(one)?)),
                            _ => None,
                        };
                        if member_plan != expected_plan {
                            return Err(one(invalid(
                                "collection duplicate member has a foreign Plan association",
                            )));
                        }
                    }

                    if let Some(outcome) = accepted.get(&reference) {
                        outcome.clone()
                    } else {
                        let bytes = decode(encoded).map_err(one)?;
                        let outcome = match &request {
                            RecordedRequest::Evaluations { context, .. } => {
                                if member_plan.is_some() {
                                    return Err(one(invalid(
                                        "differentiable collection member cannot carry an ordinary Run Plan",
                                    )));
                                }
                                let record = RecordedDifferentiableEvaluation::from_context_bytes(
                                    &bytes, context,
                                )?;
                                if record.identity() != reference {
                                    return Err(one(invalid(
                                        "collection member reference differs from its content",
                                    )));
                                }
                                RecordedEvaluationOutcome::Accepted(Arc::new(record))
                            }
                            RecordedRequest::Parameters {
                                template,
                                parameters: selected,
                                points,
                                ..
                            } => {
                                let encoded_plan = member_plan.as_ref().ok_or_else(|| {
                                    one(invalid("stationary member requires its exact Plan"))
                                })?;
                                let member_plan = eqiora_numerics::ResolvedCommonPlan::from_bytes(
                                    &decode(encoded_plan).map_err(one)?,
                                    backend,
                                    time_backend,
                                )
                                .map_err(one)?;
                                let original = parameters::document(template)?;
                                let model = parameters::bind(&original, selected, &points[index])?;
                                let expected = template
                                    .resolve_parameter_model(&model, backend)
                                    .map_err(one)?;
                                if member_plan != expected {
                                    return Err(one(invalid(
                                        "stationary member Plan differs from its exact request policies or Parameter point",
                                    )));
                                }
                                let result =
                                    CommonResult::from_bytes(&bytes, &member_plan).map_err(one)?;
                                if result.identity() != reference {
                                    return Err(one(invalid(
                                        "stationary Result reference differs from its content",
                                    )));
                                }
                                RecordedEvaluationOutcome::RunAccepted(Arc::new(result))
                            }
                            RecordedRequest::Time { plan, .. } => {
                                if member_plan.is_some() {
                                    return Err(one(invalid(
                                        "time collection members use their exact request Plan without a member Plan payload",
                                    )));
                                }
                                let result = CommonResult::from_bytes(&bytes, plan).map_err(one)?;
                                if result.identity() != reference {
                                    return Err(one(invalid(
                                        "collection time Result reference differs from its content",
                                    )));
                                }
                                RecordedEvaluationOutcome::RunAccepted(Arc::new(result))
                            }
                        };
                        accepted.insert(reference, outcome.clone());
                        outcome
                    }
                }
                WireOutcome::Failed { diagnostics } => {
                    RecordedEvaluationOutcome::Failed(diagnostics)
                }
                WireOutcome::Cancelled { checkpoint } => RecordedEvaluationOutcome::Cancelled(
                    checkpoint
                        .map(|encoded| {
                            let plan = request.plan().as_time().ok_or_else(|| {
                                invalid("collection checkpoint requires its time Plan")
                            })?;
                            RecordedTimeCheckpoint::from_bytes(&decode(&encoded)?, plan)
                                .map(Box::new)
                        })
                        .transpose()
                        .map_err(one)?,
                ),
                WireOutcome::NotStarted => RecordedEvaluationOutcome::NotStarted,
            };
            request.validate_outcome(index, &outcome).map_err(one)?;
            outcomes.push(outcome);
        }
        if used.len() != self.records.len() {
            return Err(one(invalid(
                "collection contains unreferenced member artifacts",
            )));
        }
        let mut group = EvaluationCollectionGroup {
            execution,
            request,
            outcomes,
            estimators: Vec::new(),
        };
        group.estimators = self
            .estimators
            .into_iter()
            .map(|value| value.replay(&group))
            .collect::<Result<_, _>>()
            .map_err(one)?;
        Ok(group)
    }
}

pub(super) fn outcome_bytes(value: &RecordedEvaluationOutcome) -> Result<usize, Diagnostic> {
    match value {
        RecordedEvaluationOutcome::Accepted(value) => value.to_bytes().map(|v| v.len()),
        RecordedEvaluationOutcome::RunAccepted(value) => value.to_bytes().map(|v| v.len()),
        _ => Err(invalid("only accepted delivery has a record payload")),
    }
}
pub(super) fn decode(value: &str) -> Result<Vec<u8>, Diagnostic> {
    STANDARD
        .decode(value)
        .map_err(|error| invalid(format!("invalid collection artifact encoding: {error}")))
}
pub(super) fn one(error: Diagnostic) -> Vec<Diagnostic> {
    vec![error]
}
