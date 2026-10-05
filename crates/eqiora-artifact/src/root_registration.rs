use std::collections::HashSet;
use std::str::FromStr;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_ir::ScalarOperatorIr;
use eqiora_schema::kernel::{ActivationKind, EventDirection, ExprDag, KernelNode, SymbolRef};
use eqiora_sem::KernelProgram;
use eqiora_time::{
    RootActivationGroup, RootRegistrationId, RootRegistrationProof, TimeEquationClass,
};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::time::TimeLoweringEnvelopeV2;
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, ModelEnvelope, TimeDecoderLimits, check_json_limits,
    invalid_artifact,
};

const ROOT_REGISTRATION_SCHEMA: &str = "eqiora.root-registration-envelope/v1";

/// Content-addressed order and atomic grouping of every canonical Event root.
///
/// A backend-local root index has no meaning without this artifact. The wire
/// links its complete Activation partition to one immutable model and one
/// explicit-ODE lowering; callbacks are later rebuilt from those canonical
/// definitions in this exact order.
#[derive(Debug, Clone, PartialEq)]
pub struct RootRegistrationEnvelopeV1 {
    wire: WireRootRegistrationEnvelopeV1,
}

impl RootRegistrationEnvelopeV1 {
    /// Discover and bind the complete Event Activation partition.
    ///
    /// Structurally identical guard expressions with the same direction form
    /// one atomic group. Guards in this first slice may read only lowering
    /// state Fields, finite Parameters, and model Time.
    ///
    /// # Errors
    /// Returns `EQ0901` for linkage drift, unsupported guard symbols, a
    /// non-scalar guard, no Event Activation, or a non-explicit lowering.
    pub fn new(
        model: &ModelEnvelope,
        program: &KernelProgram,
        lowering: &TimeLoweringEnvelopeV2,
    ) -> Result<Self, Diagnostic> {
        validate_model_program(model, program)?;
        lowering.validate_against(model, program)?;
        let lowering_proof = lowering.proof()?;
        if lowering_proof.equation_class() != TimeEquationClass::ExplicitOde {
            return Err(invalid_artifact(
                "root registration v1 currently requires an explicit-ODE time lowering",
            ));
        }
        let proof = discover_root_registration(program, lowering_proof.state_coordinates())?;
        let wire = WireRootRegistrationEnvelopeV1 {
            schema: ROOT_REGISTRATION_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            model_sha256: model.digest()?.0,
            model_ulid: program.model().ulid().to_string(),
            semantic_revision: program.revision().0,
            lowering_sha256: lowering.digest()?.0,
            groups: proof
                .groups()
                .iter()
                .map(WireRootActivationGroup::encode)
                .collect(),
        };
        let envelope = Self { wire };
        envelope.validate_local()?;
        envelope.validate_against(model, program, lowering)?;
        Ok(envelope)
    }

    /// Decode and locally validate one root registration.
    ///
    /// Complete model/lowering linkage and Activation partitioning are
    /// rechecked by [`Self::validate_against`] after referenced artifacts load.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version,
    /// non-canonical, duplicate, overlapping, or invalid-ID data.
    pub fn from_json(bytes: &[u8], limits: TimeDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire: WireRootRegistrationEnvelopeV1 =
            serde_json::from_slice(bytes).map_err(|error| {
                invalid_artifact(format!("invalid root registration envelope JSON: {error}"))
            })?;
        if wire.groups.len() > limits.max_root_functions {
            return Err(invalid_artifact(format!(
                "root function count exceeds decoder limit {}",
                limits.max_root_functions
            )));
        }
        let activation_count = wire.groups.iter().try_fold(0_usize, |count, group| {
            count.checked_add(group.activation_ulids.len())
        });
        if activation_count.is_none_or(|count| count > limits.max_root_activation_references) {
            return Err(invalid_artifact(format!(
                "root Activation references exceed decoder limit {}",
                limits.max_root_activation_references
            )));
        }
        let envelope = Self { wire };
        envelope.validate_local()?;
        Ok(envelope)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize root registration envelope: {error}"
            ))
        })
    }

    /// Domain-separated SHA-256 identity of the complete registration.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            ROOT_REGISTRATION_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Opaque L2 execution identity derived from the complete artifact.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn registration_id(&self) -> Result<RootRegistrationId, Diagnostic> {
        Ok(RootRegistrationId::from_sha256(
            self.digest()?.sha256_bytes(),
        ))
    }

    /// Referenced canonical model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.model_sha256.clone())
    }

    /// Referenced time-lowering artifact.
    #[must_use]
    pub fn lowering_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.lowering_sha256.clone())
    }

    /// Semantic graph revision captured by registration.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.semantic_revision
    }

    /// Decode the canonical callback order and atomic Activation groups.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn proof(&self) -> Result<RootRegistrationProof, Diagnostic> {
        let groups = self
            .wire
            .groups
            .iter()
            .map(WireRootActivationGroup::decode)
            .collect::<Result<Vec<_>, _>>()?;
        RootRegistrationProof::new(groups).map_err(|error| invalid_artifact(error.message()))
    }

    /// Rebuild and compare the complete partition against independently loaded
    /// model and lowering artifacts.
    ///
    /// # Errors
    /// Returns `EQ0901` for any digest, identity, revision, equation-class, or
    /// canonical Event partition drift.
    pub fn validate_against(
        &self,
        model: &ModelEnvelope,
        program: &KernelProgram,
        lowering: &TimeLoweringEnvelopeV2,
    ) -> Result<(), Diagnostic> {
        validate_model_program(model, program)?;
        lowering.validate_against(model, program)?;
        let lowering_proof = lowering.proof()?;
        if self.model_artifact() != model.digest()?
            || self.lowering_artifact() != lowering.digest()?
            || self.wire.model_ulid != program.model().ulid().to_string()
            || self.semantic_revision() != program.revision().0
            || lowering_proof.equation_class() != TimeEquationClass::ExplicitOde
            || self.proof()?
                != discover_root_registration(program, lowering_proof.state_coordinates())?
        {
            return Err(invalid_artifact(
                "root registration model/lowering linkage or Event partition does not match",
            ));
        }
        Ok(())
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != ROOT_REGISTRATION_SCHEMA || self.wire.encoding != CANONICAL_ENCODING
        {
            return Err(invalid_artifact(
                "unsupported root-registration-envelope schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        ArtifactDigest::from_hex(self.wire.lowering_sha256.clone())?;
        parse_ulid(&self.wire.model_ulid)?;
        let proof = self.proof()?;
        let canonical_groups = proof
            .groups()
            .iter()
            .map(WireRootActivationGroup::encode)
            .collect::<Vec<_>>();
        if canonical_groups != self.wire.groups {
            return Err(invalid_artifact(
                "root registration groups must be canonically ordered and unique",
            ));
        }
        Ok(())
    }
}

fn discover_root_registration(
    program: &KernelProgram,
    state_coordinates: &[(Id<kinds::Field>, u32)],
) -> Result<RootRegistrationProof, Diagnostic> {
    let state_coordinates = state_coordinates.iter().copied().collect::<HashSet<_>>();
    let mut structural_groups: Vec<(ExprDag, EventDirection, Vec<Id<kinds::Activation>>)> =
        Vec::new();

    for node in program.nodes() {
        let KernelNode::Activation(activation) = node else {
            continue;
        };
        let ActivationKind::Event {
            guard,
            direction,
            priority,
        } = activation.kind()
        else {
            continue;
        };
        if *priority != 0 {
            return Err(invalid_artifact(
                "root registration does not support event priority",
            ));
        }
        let operator =
            ScalarOperatorIr::lower(guard).map_err(|error| invalid_artifact(error.message()))?;
        if operator.residual_count() != 1 {
            return Err(invalid_artifact(
                "root registration requires one scalar residual per Event guard",
            ));
        }
        for symbol in operator.symbols() {
            match *symbol {
                SymbolRef::Field(field) if state_coordinates.contains(&(field, 0)) => {}
                SymbolRef::Parameter(parameter) => {
                    let value = match program.node(parameter.erase()) {
                        Some(KernelNode::Parameter(definition)) => {
                            let initial = definition.real_scalar_value().ok_or_else(|| {
                                invalid_artifact("Event guard requires real scalar Parameters")
                            })?;
                            program.value(parameter.erase()).unwrap_or(initial)
                        }
                        _ => {
                            return Err(invalid_artifact(
                                "Event guard references a Parameter absent from the model",
                            ));
                        }
                    };
                    if !value.value().is_finite() {
                        return Err(invalid_artifact(
                            "Event guard Parameter value must be finite",
                        ));
                    }
                }
                SymbolRef::Time => {}
                _ => {
                    return Err(invalid_artifact(
                        "Event guard may read only lowering state Fields, Parameters, and Time",
                    ));
                }
            }
        }

        if let Some((_, _, activations)) =
            structural_groups
                .iter_mut()
                .find(|(candidate, candidate_direction, _)| {
                    candidate == guard && candidate_direction == direction
                })
        {
            activations.push(activation.id());
        } else {
            structural_groups.push((guard.clone(), *direction, vec![activation.id()]));
        }
    }

    let groups = structural_groups
        .into_iter()
        .map(|(_, _, activations)| {
            RootActivationGroup::new(activations).map_err(|error| invalid_artifact(error.message()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    RootRegistrationProof::new(groups).map_err(|error| invalid_artifact(error.message()))
}

fn validate_model_program(
    model: &ModelEnvelope,
    program: &KernelProgram,
) -> Result<(), Diagnostic> {
    if model.model()? != program.model() || model.source_revision() != program.revision().0 {
        Err(invalid_artifact(
            "model envelope and KernelProgram identity or revision differs",
        ))
    } else {
        Ok(())
    }
}

fn parse_ulid(value: &str) -> Result<Ulid, Diagnostic> {
    Ulid::from_str(value).map_err(|_| invalid_artifact("root registration ULID is malformed"))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRootRegistrationEnvelopeV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    model_ulid: String,
    semantic_revision: u64,
    lowering_sha256: String,
    groups: Vec<WireRootActivationGroup>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRootActivationGroup {
    activation_ulids: Vec<String>,
}

impl WireRootActivationGroup {
    fn encode(value: &RootActivationGroup) -> Self {
        Self {
            activation_ulids: value
                .activations()
                .iter()
                .map(|activation| activation.ulid().to_string())
                .collect(),
        }
    }

    fn decode(&self) -> Result<RootActivationGroup, Diagnostic> {
        let activations = self
            .activation_ulids
            .iter()
            .map(|value| parse_ulid(value).map(Id::<kinds::Activation>::from_ulid))
            .collect::<Result<Vec<_>, _>>()?;
        RootActivationGroup::new(activations).map_err(|error| invalid_artifact(error.message()))
    }
}
