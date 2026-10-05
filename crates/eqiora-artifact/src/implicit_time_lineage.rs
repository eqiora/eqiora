use std::collections::HashMap;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_schema::kernel::SymbolRef;
use eqiora_sem::KernelProgram;
use serde::{Deserialize, Serialize};

use crate::implicit_time::validate_general_proof;
use crate::time::canonical_time_operator;
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, GeneralImplicitTimeLoweringEnvelopeV3,
    ImplicitTimeInitialDataEnvelopeV1, ImplicitTimeRunManifestV1, TimeDecoderLimits,
    check_json_limits, invalid_artifact,
};

const IMPLICIT_CHECKPOINT_SCHEMA: &str = "eqiora.implicit-time-checkpoint-envelope/v1";
const IMPLICIT_RESTART_SCHEMA: &str = "eqiora.implicit-time-restart-manifest/v1";

/// Content-addressed accepted `(time, state, derivative)` point for a
/// residual-native lowering.
///
/// A checkpoint deliberately does not reference a run manifest. The run may
/// list the checkpoint as an output without creating a content-digest cycle;
/// a separate [`ImplicitTimeRestartManifestV1`] later links parent and child
/// runs. The stored residual norm is independently replayed from canonical
/// Operator IR rather than trusted from a backend report.
#[derive(Debug, Clone, PartialEq)]
pub struct ImplicitTimeCheckpointEnvelopeV1 {
    wire: WireImplicitTimeCheckpointEnvelopeV1,
}

impl ImplicitTimeCheckpointEnvelopeV1 {
    /// Capture one accepted residual-native point after canonical replay.
    ///
    /// # Errors
    /// Returns `EQ0901` for model/lowering drift, invalid point data, or a
    /// canonical residual infinity norm above `residual_tolerance`.
    pub fn from_accepted_pair(
        lowering: &GeneralImplicitTimeLoweringEnvelopeV3,
        program: &KernelProgram,
        time: f64,
        mut state: Vec<f64>,
        mut derivative: Vec<f64>,
        residual_tolerance: f64,
    ) -> Result<Self, Diagnostic> {
        if !residual_tolerance.is_finite() || residual_tolerance < 0.0 {
            return Err(invalid_artifact(
                "implicit time checkpoint residual tolerance must be finite and nonnegative",
            ));
        }
        normalize_zeros(&mut state);
        normalize_zeros(&mut derivative);
        let time = normalize_zero(time);
        let residual_tolerance = normalize_zero(residual_tolerance);
        let residual_infinity_norm = normalize_zero(replay_residual_norm(
            lowering,
            program,
            time,
            &state,
            &derivative,
        )?);
        let checkpoint = Self {
            wire: WireImplicitTimeCheckpointEnvelopeV1 {
                schema: IMPLICIT_CHECKPOINT_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: lowering.model_artifact().0,
                semantic_revision: lowering.semantic_revision(),
                lowering_sha256: lowering.digest()?.0,
                time,
                state,
                derivative,
                residual_infinity_norm,
                residual_tolerance,
            },
        };
        checkpoint.validate_local()?;
        checkpoint.validate_against(lowering, program)?;
        Ok(checkpoint)
    }

    /// Decode and locally validate checkpoint data.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version,
    /// shape-mismatched, non-finite, or non-canonical data.
    pub fn from_json(bytes: &[u8], limits: TimeDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid implicit time checkpoint JSON: {error}"))
        })?;
        let checkpoint = Self { wire };
        if checkpoint.wire.state.len() > limits.max_time_state_dimension
            || checkpoint.wire.derivative.len() > limits.max_time_state_dimension
        {
            return Err(invalid_artifact(format!(
                "implicit time checkpoint dimension exceeds decoder limit {}",
                limits.max_time_state_dimension
            )));
        }
        checkpoint.validate_local()?;
        Ok(checkpoint)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize implicit time checkpoint: {error}"
            ))
        })
    }

    /// Domain-separated SHA-256 identity of this accepted point and proof.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            IMPLICIT_CHECKPOINT_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Referenced canonical model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.model_sha256.clone())
    }

    /// Referenced residual-native lowering witness.
    #[must_use]
    pub fn lowering(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.lowering_sha256.clone())
    }

    /// Semantic graph revision shared with the lowering.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.semantic_revision
    }

    /// Accepted model time.
    #[must_use]
    pub const fn time(&self) -> f64 {
        self.wire.time
    }

    /// Accepted state in canonical lowering order.
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.wire.state
    }

    /// Accepted derivative in canonical lowering order.
    #[must_use]
    pub fn derivative(&self) -> &[f64] {
        &self.wire.derivative
    }

    /// Canonically replayed residual infinity norm.
    #[must_use]
    pub const fn residual_infinity_norm(&self) -> f64 {
        self.wire.residual_infinity_norm
    }

    /// Acceptance tolerance applied to the replayed residual.
    #[must_use]
    pub const fn residual_tolerance(&self) -> f64 {
        self.wire.residual_tolerance
    }

    /// Revalidate lowering linkage and replay the residual from canonical
    /// Operator IR.
    ///
    /// # Errors
    /// Returns `EQ0901` for model/revision/digest, state-order, value, norm,
    /// or acceptance drift.
    pub fn validate_against(
        &self,
        lowering: &GeneralImplicitTimeLoweringEnvelopeV3,
        program: &KernelProgram,
    ) -> Result<(), Diagnostic> {
        if self.model_artifact() != lowering.model_artifact()
            || self.semantic_revision() != lowering.semantic_revision()
            || self.lowering() != lowering.digest()?
        {
            return Err(invalid_artifact(
                "implicit time checkpoint does not match its lowering witness",
            ));
        }
        let residual_infinity_norm = normalize_zero(replay_residual_norm(
            lowering,
            program,
            self.time(),
            self.state(),
            self.derivative(),
        )?);
        if residual_infinity_norm != self.residual_infinity_norm()
            || residual_infinity_norm > self.residual_tolerance()
        {
            return Err(invalid_artifact(
                "implicit time checkpoint residual replay differs or exceeds its tolerance",
            ));
        }
        Ok(())
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != IMPLICIT_CHECKPOINT_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
        {
            return Err(invalid_artifact(
                "unsupported implicit-time-checkpoint schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        ArtifactDigest::from_hex(self.wire.lowering_sha256.clone())?;
        if self.wire.state.is_empty()
            || self.wire.state.len() != self.wire.derivative.len()
            || !self.wire.time.is_finite()
            || !self.wire.residual_infinity_norm.is_finite()
            || self.wire.residual_infinity_norm < 0.0
            || !self.wire.residual_tolerance.is_finite()
            || self.wire.residual_tolerance < 0.0
            || self.wire.residual_infinity_norm > self.wire.residual_tolerance
            || is_negative_zero(self.wire.time)
            || is_negative_zero(self.wire.residual_infinity_norm)
            || is_negative_zero(self.wire.residual_tolerance)
            || self
                .wire
                .state
                .iter()
                .chain(&self.wire.derivative)
                .any(|value| !value.is_finite() || is_negative_zero(*value))
        {
            return Err(invalid_artifact(
                "implicit time checkpoint requires a finite canonical point and accepted residual norm",
            ));
        }
        Ok(())
    }
}

/// Content-addressed semantic restart edge between residual-native runs.
///
/// This manifest proves restart from an accepted state/derivative pair. It
/// does not claim bitwise continuation of adaptive-controller or backend
/// solver history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplicitTimeRestartManifestV1 {
    wire: WireImplicitTimeRestartManifestV1,
}

impl ImplicitTimeRestartManifestV1 {
    /// Link a parent run output through one checkpoint to provided child-run
    /// initial data and the resulting child run.
    ///
    /// # Errors
    /// Returns `EQ0901` for any canonical replay, digest, point, time, or run
    /// linkage contradiction.
    pub fn new(
        lowering: &GeneralImplicitTimeLoweringEnvelopeV3,
        program: &KernelProgram,
        parent_run: &ImplicitTimeRunManifestV1,
        checkpoint: &ImplicitTimeCheckpointEnvelopeV1,
        child_initial: &ImplicitTimeInitialDataEnvelopeV1,
        child_run: &ImplicitTimeRunManifestV1,
    ) -> Result<Self, Diagnostic> {
        let manifest = Self {
            wire: WireImplicitTimeRestartManifestV1 {
                schema: IMPLICIT_RESTART_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: lowering.model_artifact().0,
                semantic_revision: lowering.semantic_revision(),
                lowering_sha256: lowering.digest()?.0,
                parent_run_sha256: parent_run.digest()?.0,
                checkpoint_sha256: checkpoint.digest()?.0,
                child_initial_data_sha256: child_initial.digest()?.0,
                child_run_sha256: child_run.digest()?.0,
            },
        };
        manifest.validate_local()?;
        manifest.validate_against(
            lowering,
            program,
            parent_run,
            checkpoint,
            child_initial,
            child_run,
        )?;
        Ok(manifest)
    }

    /// Decode and locally validate one semantic restart edge.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version, or invalid
    /// digest data.
    pub fn from_json(bytes: &[u8], limits: TimeDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!(
                "invalid implicit time restart manifest JSON: {error}"
            ))
        })?;
        let manifest = Self { wire };
        manifest.validate_local()?;
        Ok(manifest)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize implicit time restart manifest: {error}"
            ))
        })
    }

    /// Domain-separated SHA-256 identity of this complete restart edge.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            IMPLICIT_RESTART_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Referenced canonical model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.model_sha256.clone())
    }

    /// Referenced residual-native lowering witness.
    #[must_use]
    pub fn lowering(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.lowering_sha256.clone())
    }

    /// Semantic graph revision shared with both runs.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.semantic_revision
    }

    /// Parent run that emitted the checkpoint.
    #[must_use]
    pub fn parent_run(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.parent_run_sha256.clone())
    }

    /// Accepted checkpoint selected for restart.
    #[must_use]
    pub fn checkpoint(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.checkpoint_sha256.clone())
    }

    /// Provided initial-data artifact derived from the checkpoint.
    #[must_use]
    pub fn child_initial_data(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.child_initial_data_sha256.clone())
    }

    /// Child run started from the accepted pair.
    #[must_use]
    pub fn child_run(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.child_run_sha256.clone())
    }

    /// Revalidate canonical checkpoint content and every external digest/time
    /// edge.
    ///
    /// # Errors
    /// Returns `EQ0901` for any foreign, missing, cyclic, or contradictory
    /// linkage.
    pub fn validate_against(
        &self,
        lowering: &GeneralImplicitTimeLoweringEnvelopeV3,
        program: &KernelProgram,
        parent_run: &ImplicitTimeRunManifestV1,
        checkpoint: &ImplicitTimeCheckpointEnvelopeV1,
        child_initial: &ImplicitTimeInitialDataEnvelopeV1,
        child_run: &ImplicitTimeRunManifestV1,
    ) -> Result<(), Diagnostic> {
        checkpoint.validate_against(lowering, program)?;
        child_initial.validate_against(lowering)?;
        child_run.validate_against(lowering, child_initial, child_initial)?;
        let lowering_digest = lowering.digest()?;
        let checkpoint_digest = checkpoint.digest()?;
        let child_initial_digest = child_initial.digest()?;
        let parent_plan = parent_run.plan()?;
        let child_plan = child_run.plan()?;
        if ArtifactDigest(self.wire.model_sha256.clone()) != lowering.model_artifact()
            || self.wire.semantic_revision != lowering.semantic_revision()
            || ArtifactDigest(self.wire.lowering_sha256.clone()) != lowering_digest
            || parent_run.model() != lowering.model_artifact()
            || child_run.model() != lowering.model_artifact()
            || parent_run.semantic_revision() != lowering.semantic_revision()
            || child_run.semantic_revision() != lowering.semantic_revision()
            || parent_run.lowering() != lowering_digest
            || child_run.lowering() != lowering_digest
            || self.parent_run() != parent_run.digest()?
            || self.checkpoint() != checkpoint_digest
            || self.child_initial_data() != child_initial_digest
            || self.child_run() != child_run.digest()?
            || self.parent_run() == self.child_run()
            || !parent_run.outputs().contains(&checkpoint_digest)
            || child_initial.initial_condition() != eqiora_time::InitialConditionPolicy::Provided
            || child_initial.state() != checkpoint.state()
            || child_initial.derivative() != checkpoint.derivative()
            || child_run.input_initial_data() != child_initial_digest
            || child_run.accepted_initial_data() != child_initial_digest
            || parent_plan.start_time() >= checkpoint.time()
            || parent_plan
                .output_times()
                .last()
                .is_none_or(|last| *last < checkpoint.time())
            || child_plan.start_time() != checkpoint.time()
        {
            return Err(invalid_artifact(
                "implicit time restart lineage does not match parent checkpoint and child run",
            ));
        }
        Ok(())
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != IMPLICIT_RESTART_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported implicit-time-restart schema or canonical encoding",
            ));
        }
        for digest in [
            &self.wire.model_sha256,
            &self.wire.lowering_sha256,
            &self.wire.parent_run_sha256,
            &self.wire.checkpoint_sha256,
            &self.wire.child_initial_data_sha256,
            &self.wire.child_run_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        if self.wire.parent_run_sha256 == self.wire.child_run_sha256 {
            return Err(invalid_artifact(
                "implicit time restart parent and child runs must be distinct",
            ));
        }
        Ok(())
    }
}

fn replay_residual_norm(
    lowering: &GeneralImplicitTimeLoweringEnvelopeV3,
    program: &KernelProgram,
    time: f64,
    state: &[f64],
    derivative: &[f64],
) -> Result<f64, Diagnostic> {
    if lowering.model()? != program.model() || lowering.semantic_revision() != program.revision().0
    {
        return Err(invalid_artifact(
            "implicit time checkpoint model identity or revision differs from the program",
        ));
    }
    let proof = lowering.proof()?;
    validate_general_proof(&proof, program)?;
    let (operator, state_coordinates) =
        canonical_time_operator(program, proof.relation(), proof.state_coordinates().len())?;
    if state.len() != state_coordinates.len()
        || derivative.len() != state_coordinates.len()
        || !time.is_finite()
        || state
            .iter()
            .chain(derivative)
            .any(|value| !value.is_finite())
    {
        return Err(invalid_artifact(
            "implicit time checkpoint point does not match canonical state shape",
        ));
    }
    // These coordinates come from the independently checked scalar canonical operator.
    let state_coordinates = state_coordinates
        .iter()
        .map(|coordinate| (coordinate.field(), coordinate.derivative_order()))
        .collect::<Vec<_>>();
    let coordinates = state_coordinates
        .iter()
        .copied()
        .enumerate()
        .map(|(coordinate, field)| (field, coordinate))
        .collect::<HashMap<(Id<kinds::Field>, u32), usize>>();
    let inputs = operator
        .symbols()
        .iter()
        .map(|symbol| match *symbol {
            SymbolRef::Field(field) => coordinates
                .get(&(field, 0))
                .map(|coordinate| state[*coordinate])
                .ok_or_else(|| invalid_artifact("checkpoint Field is outside canonical state")),
            SymbolRef::Derivative(field, order) => coordinates
                .get(&(field, order.get()))
                .map(|coordinate| state[*coordinate])
                .or_else(|| {
                    coordinates
                        .get(&(field, order.get() - 1))
                        .map(|coordinate| derivative[*coordinate])
                })
                .ok_or_else(|| {
                    invalid_artifact("checkpoint Derivative is outside canonical state")
                }),
            SymbolRef::Parameter(parameter) => program
                .value(parameter.erase())
                .map(|value| value.value())
                .ok_or_else(|| invalid_artifact("checkpoint Parameter has no canonical value")),
            SymbolRef::Time => Ok(time),
            SymbolRef::Pre(_) | SymbolRef::Next(_) | SymbolRef::Port(_) => Err(invalid_artifact(
                "checkpoint replay admits only state, derivative, Parameter, and time symbols",
            )),
            _ => Err(invalid_artifact(
                "checkpoint replay encountered a newer unsupported symbol kind",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let residual = operator
        .evaluate(&inputs)
        .map_err(|error| invalid_artifact(format!("checkpoint residual replay failed: {error}")))?;
    let mut norm = residual.into_iter().map(f64::abs).fold(0.0_f64, f64::max);
    for (index, &(field, order)) in state_coordinates.iter().enumerate() {
        if let Some(next) = coordinates.get(&(field, order + 1)) {
            let residual = derivative[index] - state[*next];
            if !residual.is_finite() {
                return Err(invalid_artifact(
                    "checkpoint companion residual is not finite",
                ));
            }
            norm = norm.max(residual.abs());
        }
    }
    Ok(norm)
}

fn normalize_zeros(values: &mut [f64]) {
    for value in values {
        *value = normalize_zero(*value);
    }
}

fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn is_negative_zero(value: f64) -> bool {
    value == 0.0 && value.is_sign_negative()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireImplicitTimeCheckpointEnvelopeV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    semantic_revision: u64,
    lowering_sha256: String,
    time: f64,
    state: Vec<f64>,
    derivative: Vec<f64>,
    residual_infinity_norm: f64,
    residual_tolerance: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireImplicitTimeRestartManifestV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    semantic_revision: u64,
    lowering_sha256: String,
    parent_run_sha256: String,
    checkpoint_sha256: String,
    child_initial_data_sha256: String,
    child_run_sha256: String,
}
