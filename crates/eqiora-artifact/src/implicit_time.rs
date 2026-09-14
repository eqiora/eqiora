use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, OntologyId};
use eqiora_ir::{ScalarOperatorIr, SymbolicLinearityFailure};
use eqiora_schema::Model;
use eqiora_schema::kernel::SymbolRef;
use eqiora_sem::KernelProgram;
use eqiora_time::{
    DaeVariableKind, GeneralImplicitLoweringProof, GeneralImplicitReason,
    ImplicitDaeInitialization, ImplicitDaeProblem, InitialConditionPolicy, TimeEquationClass,
    TimeExecutionReport, TimeMethod, TimePlan,
};
use serde::{Deserialize, Serialize};

use crate::time::{canonical_time_operator, parse_ulid, validate_model_program};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, ModelEnvelope, TimeDecoderLimits, check_json_limits,
    invalid_artifact, validate_text,
};

const GENERAL_IMPLICIT_LOWERING_SCHEMA: &str = "eqiora.general-implicit-time-lowering-envelope/v1";
const IMPLICIT_INITIAL_DATA_SCHEMA: &str = "eqiora.implicit-time-initial-data-envelope/v1";
const IMPLICIT_RUN_SCHEMA: &str = "eqiora.implicit-time-run-manifest/v1";

/// Content-addressed witness for canonical Relation → residual-native time
/// lowering.
///
/// The envelope is deliberately separate from [`crate::TimeLoweringEnvelopeV1`].
/// It records the structural obstruction to a constant first-order projection
/// and the effective differential/algebraic partition, not a fabricated mass
/// matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralImplicitTimeLoweringEnvelopeV1 {
    wire: WireGeneralImplicitTimeLoweringEnvelopeV1,
}

impl GeneralImplicitTimeLoweringEnvelopeV1 {
    /// Bind a runtime-produced residual-native witness to one immutable model.
    ///
    /// # Errors
    /// Returns `EQ0901` for model/program drift, an invalid continuous
    /// Relation, state-order/partition drift, or a Relation that has a valid
    /// constant first-order projection.
    pub fn from_proof(
        model: &ModelEnvelope,
        program: &KernelProgram,
        proof: &GeneralImplicitLoweringProof,
    ) -> Result<Self, Diagnostic> {
        validate_model_program(model, program)?;
        validate_general_proof(proof, program)?;
        let wire = WireGeneralImplicitTimeLoweringEnvelopeV1 {
            schema: GENERAL_IMPLICIT_LOWERING_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            model_sha256: model.digest()?.0,
            model_ulid: program.model().ulid().to_string(),
            semantic_revision: program.revision().0,
            relation_ulid: proof.relation().ulid().to_string(),
            state_field_ulids: proof
                .state_fields()
                .iter()
                .map(|field| field.ulid().to_string())
                .collect(),
            variable_kinds: proof
                .variable_kinds()
                .iter()
                .copied()
                .map(WireDaeVariableKind::encode)
                .collect(),
            reason: WireGeneralImplicitReason::encode(proof.reason()),
        };
        let envelope = Self { wire };
        envelope.validate_local()?;
        Ok(envelope)
    }

    /// Decode and locally validate a residual-native lowering envelope.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version, invalid-ID,
    /// or internally contradictory witness data.
    pub fn from_json(bytes: &[u8], limits: TimeDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!(
                "invalid general implicit time lowering envelope JSON: {error}"
            ))
        })?;
        let envelope = Self { wire };
        if envelope.wire.state_field_ulids.len() > limits.max_time_state_dimension {
            return Err(invalid_artifact(format!(
                "general implicit time state dimension exceeds decoder limit {}",
                limits.max_time_state_dimension
            )));
        }
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
                "cannot serialize general implicit time lowering envelope: {error}"
            ))
        })
    }

    /// Domain-separated SHA-256 identity of the complete lowering witness.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            GENERAL_IMPLICIT_LOWERING_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Referenced canonical model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.model_sha256.clone())
    }

    /// Typed Semantic Model identity.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn model(&self) -> Result<OntologyId<Model>, Diagnostic> {
        parse_ulid(&self.wire.model_ulid).map(OntologyId::from_ulid)
    }

    /// Semantic graph revision captured by the lowering.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.semantic_revision
    }

    /// Decode the typed residual-native witness.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn proof(&self) -> Result<GeneralImplicitLoweringProof, Diagnostic> {
        let relation = Id::<kinds::Relation>::from_ulid(parse_ulid(&self.wire.relation_ulid)?);
        let state_fields = self
            .wire
            .state_field_ulids
            .iter()
            .map(|value| parse_ulid(value).map(Id::<kinds::Field>::from_ulid))
            .collect::<Result<Vec<_>, _>>()?;
        let variable_kinds = self
            .wire
            .variable_kinds
            .iter()
            .copied()
            .map(WireDaeVariableKind::decode)
            .collect();
        GeneralImplicitLoweringProof::new(
            relation,
            state_fields,
            variable_kinds,
            self.wire.reason.decode(),
        )
        .map_err(|error| invalid_artifact(error.message()))
    }

    /// Revalidate model linkage and all structural facts against Operator IR.
    ///
    /// # Errors
    /// Returns `EQ0901` for digest/revision, state-order, partition, or reason
    /// drift.
    pub fn validate_against(
        &self,
        model: &ModelEnvelope,
        program: &KernelProgram,
    ) -> Result<(), Diagnostic> {
        validate_model_program(model, program)?;
        if self.model_artifact() != model.digest()?
            || self.model()? != program.model()
            || self.semantic_revision() != program.revision().0
        {
            return Err(invalid_artifact(
                "general implicit time lowering model digest, identity, or revision does not match",
            ));
        }
        validate_general_proof(&self.proof()?, program)
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != GENERAL_IMPLICIT_LOWERING_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
        {
            return Err(invalid_artifact(
                "unsupported general-implicit-time-lowering schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        parse_ulid(&self.wire.model_ulid)?;
        parse_ulid(&self.wire.relation_ulid)?;
        for field in &self.wire.state_field_ulids {
            parse_ulid(field)?;
        }
        self.proof()?;
        Ok(())
    }
}

/// Versioned initial state/derivative data linked to one residual-native
/// lowering.
///
/// A consistency-solve input and the accepted consistent pair are separate
/// artifacts. This keeps backend output from silently replacing the supplied
/// guess in run provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct ImplicitTimeInitialDataEnvelopeV1 {
    wire: WireImplicitTimeInitialDataEnvelopeV1,
}

impl ImplicitTimeInitialDataEnvelopeV1 {
    /// Capture the initial pair or consistency-solve guess from a validated
    /// residual-native problem.
    ///
    /// # Errors
    /// Returns `EQ0901` if dimension or variable partition differs from the
    /// linked lowering.
    pub fn from_problem(
        lowering: &GeneralImplicitTimeLoweringEnvelopeV1,
        problem: &ImplicitDaeProblem<'_>,
    ) -> Result<Self, Diagnostic> {
        let proof = lowering.proof()?;
        if problem.variable_kinds() != proof.variable_kinds() {
            return Err(invalid_artifact(
                "implicit initial problem partition differs from its lowering witness",
            ));
        }
        Self::new(
            lowering,
            problem.initial_condition(),
            problem.initial_state().to_vec(),
            problem.initial_derivative().to_vec(),
        )
    }

    /// Capture one backend-accepted consistent initial pair.
    ///
    /// # Errors
    /// Returns `EQ0901` if the accepted pair dimension differs from the linked
    /// lowering.
    pub fn from_initialization(
        lowering: &GeneralImplicitTimeLoweringEnvelopeV1,
        initialization: &ImplicitDaeInitialization,
    ) -> Result<Self, Diagnostic> {
        Self::new(
            lowering,
            InitialConditionPolicy::Provided,
            initialization.state().to_vec(),
            initialization.derivative().to_vec(),
        )
    }

    /// Convert one independently replayed accepted checkpoint into provided
    /// restart data.
    ///
    /// # Errors
    /// Returns `EQ0901` if checkpoint content or canonical Operator-IR linkage
    /// does not match the lowering/program pair.
    pub fn from_checkpoint(
        lowering: &GeneralImplicitTimeLoweringEnvelopeV1,
        checkpoint: &crate::ImplicitTimeCheckpointEnvelopeV1,
        program: &KernelProgram,
    ) -> Result<Self, Diagnostic> {
        checkpoint.validate_against(lowering, program)?;
        Self::new(
            lowering,
            InitialConditionPolicy::Provided,
            checkpoint.state().to_vec(),
            checkpoint.derivative().to_vec(),
        )
    }

    fn new(
        lowering: &GeneralImplicitTimeLoweringEnvelopeV1,
        initial_condition: InitialConditionPolicy,
        mut state: Vec<f64>,
        mut derivative: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        normalize_zeros(&mut state);
        normalize_zeros(&mut derivative);
        let wire = WireImplicitTimeInitialDataEnvelopeV1 {
            schema: IMPLICIT_INITIAL_DATA_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            model_sha256: lowering.model_artifact().0,
            semantic_revision: lowering.semantic_revision(),
            lowering_sha256: lowering.digest()?.0,
            initial_condition: WireInitialCondition::encode(initial_condition),
            state,
            derivative,
        };
        let envelope = Self { wire };
        envelope.validate_local()?;
        envelope.validate_against(lowering)?;
        Ok(envelope)
    }

    /// Decode and locally validate initial data.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version,
    /// shape-mismatched, or non-finite data.
    pub fn from_json(bytes: &[u8], limits: TimeDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid implicit time initial data JSON: {error}"))
        })?;
        let envelope = Self { wire };
        if envelope.wire.state.len() > limits.max_time_state_dimension
            || envelope.wire.derivative.len() > limits.max_time_state_dimension
        {
            return Err(invalid_artifact(format!(
                "implicit time initial-data dimension exceeds decoder limit {}",
                limits.max_time_state_dimension
            )));
        }
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
                "cannot serialize implicit time initial data: {error}"
            ))
        })
    }

    /// Domain-separated SHA-256 identity of this complete initial data.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            IMPLICIT_INITIAL_DATA_SCHEMA.as_bytes(),
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

    /// Semantic graph revision shared with the linked lowering.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.semantic_revision
    }

    /// Initial-condition treatment attached to these values.
    #[must_use]
    pub const fn initial_condition(&self) -> InitialConditionPolicy {
        self.wire.initial_condition.decode()
    }

    /// Initial state or consistency-solve guess.
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.wire.state
    }

    /// Initial derivative or consistency-solve guess.
    #[must_use]
    pub fn derivative(&self) -> &[f64] {
        &self.wire.derivative
    }

    /// Revalidate lowering/model/revision linkage and state dimension.
    ///
    /// # Errors
    /// Returns `EQ0901` for any linkage or shape drift.
    pub fn validate_against(
        &self,
        lowering: &GeneralImplicitTimeLoweringEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        let proof = lowering.proof()?;
        if self.model_artifact() != lowering.model_artifact()
            || self.semantic_revision() != lowering.semantic_revision()
            || self.lowering() != lowering.digest()?
            || self.wire.state.len() != proof.state_fields().len()
            || self.wire.derivative.len() != proof.state_fields().len()
        {
            return Err(invalid_artifact(
                "implicit time initial data does not match its lowering witness",
            ));
        }
        Ok(())
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != IMPLICIT_INITIAL_DATA_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
        {
            return Err(invalid_artifact(
                "unsupported implicit-time-initial-data schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        ArtifactDigest::from_hex(self.wire.lowering_sha256.clone())?;
        if self.wire.state.is_empty()
            || self.wire.state.len() != self.wire.derivative.len()
            || self
                .wire
                .state
                .iter()
                .chain(&self.wire.derivative)
                .any(|value| !value.is_finite() || is_negative_zero(*value))
        {
            return Err(invalid_artifact(
                "implicit time initial state/derivative must be non-empty, equally shaped, finite, and canonically zeroed",
            ));
        }
        Ok(())
    }
}

/// Reproducible residual-native plan, initial-pair lineage, backend evidence,
/// and content-addressed outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct ImplicitTimeRunManifestV1 {
    wire: WireImplicitTimeRunManifestV1,
}

impl ImplicitTimeRunManifestV1 {
    /// Link one residual-native execution to its lowering, supplied initial
    /// data, backend-accepted consistent pair, plan, and report.
    ///
    /// # Errors
    /// Returns `EQ0901` for any linkage, dimension, method, equation-class,
    /// initial-condition, or adapter-supplied backend-version contradiction.
    pub fn new(
        lowering: &GeneralImplicitTimeLoweringEnvelopeV1,
        input: &ImplicitTimeInitialDataEnvelopeV1,
        accepted: &ImplicitTimeInitialDataEnvelopeV1,
        plan: &TimePlan,
        report: TimeExecutionReport,
    ) -> Result<Self, Diagnostic> {
        input.validate_against(lowering)?;
        accepted.validate_against(lowering)?;
        let backend_version = report.backend_version().to_owned();
        validate_text("implicit time backend version", &backend_version)?;
        let proof = lowering.proof()?;
        if accepted.initial_condition() != InitialConditionPolicy::Provided
            || report.method() != plan.method()
            || report.equation_class() != TimeEquationClass::GeneralImplicitDae
            || report.initial_condition() != input.initial_condition()
            || plan.absolute_tolerances().len() != proof.state_fields().len()
        {
            return Err(invalid_artifact(
                "implicit time run plan/report/initial data contradict their lowering witness",
            ));
        }
        let manifest = Self {
            wire: WireImplicitTimeRunManifestV1 {
                schema: IMPLICIT_RUN_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: lowering.model_artifact().0,
                semantic_revision: lowering.semantic_revision(),
                lowering_sha256: lowering.digest()?.0,
                input_initial_data_sha256: input.digest()?.0,
                accepted_initial_data_sha256: accepted.digest()?.0,
                plan: WireImplicitTimePlan::encode(plan)?,
                execution: WireImplicitTimeExecution {
                    backend: report.backend().to_owned(),
                    backend_version,
                    method: WireImplicitTimeMethod::encode(report.method())?,
                    equation_class: WireImplicitEquationClass::GeneralImplicitDae,
                    initial_condition: WireInitialCondition::encode(report.initial_condition()),
                },
                output_sha256: Vec::new(),
            },
        };
        manifest.validate_local()?;
        manifest.validate_against(lowering, input, accepted)?;
        Ok(manifest)
    }

    /// Add a content-addressed trajectory or evidence output.
    #[must_use]
    pub fn with_output(mut self, output: ArtifactDigest) -> Self {
        self.wire.output_sha256.push(output.0);
        self.wire.output_sha256.sort();
        self.wire.output_sha256.dedup();
        self
    }

    /// Decode and locally validate a residual-native run manifest.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version, duplicate,
    /// non-finite, or internally contradictory data.
    pub fn from_json(bytes: &[u8], limits: TimeDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid implicit time run manifest JSON: {error}"))
        })?;
        let manifest = Self { wire };
        if manifest.wire.plan.absolute_tolerances.len() > limits.max_time_state_dimension {
            return Err(invalid_artifact(format!(
                "implicit time run state dimension exceeds decoder limit {}",
                limits.max_time_state_dimension
            )));
        }
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
                "cannot serialize implicit time run manifest: {error}"
            ))
        })
    }

    /// Domain-separated SHA-256 identity of the complete run manifest.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            IMPLICIT_RUN_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Referenced canonical model artifact.
    #[must_use]
    pub fn model(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.model_sha256.clone())
    }

    /// Referenced residual-native lowering witness.
    #[must_use]
    pub fn lowering(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.lowering_sha256.clone())
    }

    /// Supplied initial pair or consistency-solve guess.
    #[must_use]
    pub fn input_initial_data(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.input_initial_data_sha256.clone())
    }

    /// Backend-accepted consistent initial pair.
    #[must_use]
    pub fn accepted_initial_data(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.accepted_initial_data_sha256.clone())
    }

    /// Semantic graph revision shared by all linked inputs.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.semantic_revision
    }

    /// Reconstructed backend-neutral time plan.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn plan(&self) -> Result<TimePlan, Diagnostic> {
        self.wire.plan.decode()
    }

    /// Stable adapter identity.
    #[must_use]
    pub fn backend(&self) -> &str {
        &self.wire.execution.backend
    }

    /// Exact adapter/library release recorded at execution.
    #[must_use]
    pub fn backend_version(&self) -> &str {
        &self.wire.execution.backend_version
    }

    /// Sorted content-addressed outputs.
    #[must_use]
    pub fn outputs(&self) -> Vec<ArtifactDigest> {
        self.wire
            .output_sha256
            .iter()
            .cloned()
            .map(ArtifactDigest)
            .collect()
    }

    /// Revalidate all content linkage and execution admission against
    /// separately loaded artifacts.
    ///
    /// # Errors
    /// Returns `EQ0901` for any model/revision/digest, plan, method,
    /// equation-class, or initial-condition drift.
    pub fn validate_against(
        &self,
        lowering: &GeneralImplicitTimeLoweringEnvelopeV1,
        input: &ImplicitTimeInitialDataEnvelopeV1,
        accepted: &ImplicitTimeInitialDataEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        input.validate_against(lowering)?;
        accepted.validate_against(lowering)?;
        let proof = lowering.proof()?;
        let plan = self.plan()?;
        if self.model() != lowering.model_artifact()
            || self.semantic_revision() != lowering.semantic_revision()
            || self.lowering() != lowering.digest()?
            || self.input_initial_data() != input.digest()?
            || self.accepted_initial_data() != accepted.digest()?
            || accepted.initial_condition() != InitialConditionPolicy::Provided
            || self.wire.execution.equation_class.decode() != TimeEquationClass::GeneralImplicitDae
            || self.wire.execution.initial_condition.decode() != input.initial_condition()
            || self.wire.execution.method.decode() != plan.method()
            || plan.absolute_tolerances().len() != proof.state_fields().len()
        {
            return Err(invalid_artifact(
                "implicit time run linkage does not match its lowering and initial-data artifacts",
            ));
        }
        Ok(())
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != IMPLICIT_RUN_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported implicit-time-run schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        ArtifactDigest::from_hex(self.wire.lowering_sha256.clone())?;
        ArtifactDigest::from_hex(self.wire.input_initial_data_sha256.clone())?;
        ArtifactDigest::from_hex(self.wire.accepted_initial_data_sha256.clone())?;
        validate_text("implicit time backend", &self.wire.execution.backend)?;
        validate_text(
            "implicit time backend version",
            &self.wire.execution.backend_version,
        )?;
        let plan = self.wire.plan.decode()?;
        if self.wire.execution.method.decode() != plan.method()
            || self.wire.execution.equation_class.decode() != TimeEquationClass::GeneralImplicitDae
        {
            return Err(invalid_artifact(
                "implicit time execution method or equation class contradicts its plan",
            ));
        }
        let mut outputs = self.wire.output_sha256.clone();
        for output in &outputs {
            ArtifactDigest::from_hex(output.clone())?;
        }
        outputs.sort();
        outputs.dedup();
        if outputs != self.wire.output_sha256 {
            return Err(invalid_artifact(
                "implicit time run outputs must be sorted and unique",
            ));
        }
        Ok(())
    }
}

pub(crate) fn validate_general_proof(
    proof: &GeneralImplicitLoweringProof,
    program: &KernelProgram,
) -> Result<(), Diagnostic> {
    let (operator, expected_fields) = canonical_time_operator(program, proof.relation())?;
    if expected_fields != proof.state_fields() || operator.residual_count() != expected_fields.len()
    {
        return Err(invalid_artifact(
            "general implicit time state order or residual shape differs from canonical Operator IR",
        ));
    }
    let derivatives = expected_fields
        .iter()
        .copied()
        .map(SymbolRef::Derivative)
        .collect::<Vec<_>>();
    let reason = match operator.constant_symbol_jacobian(&derivatives) {
        Err(SymbolicLinearityFailure::VariableCoefficient { .. }) => {
            GeneralImplicitReason::NonconstantDerivativeJacobian
        }
        Err(SymbolicLinearityFailure::Nonlinear { .. }) => {
            GeneralImplicitReason::NonlinearDerivativeDependence
        }
        Err(failure) => {
            return Err(invalid_artifact(format!(
                "general implicit derivative analysis failed: {failure:?}"
            )));
        }
        Ok(_) => {
            return Err(invalid_artifact(
                "Relation has a constant derivative Jacobian and cannot enter the general implicit artifact",
            ));
        }
    };
    let variable_kinds = expected_fields
        .iter()
        .copied()
        .map(|field| effective_derivative_kind(&operator, field))
        .collect::<Result<Vec<_>, _>>()?;
    if reason != proof.reason() || variable_kinds != proof.variable_kinds() {
        return Err(invalid_artifact(
            "general implicit reason or variable partition differs from Operator IR",
        ));
    }
    Ok(())
}

fn effective_derivative_kind(
    operator: &ScalarOperatorIr,
    field: Id<kinds::Field>,
) -> Result<DaeVariableKind, Diagnostic> {
    match operator.constant_symbol_jacobian(&[SymbolRef::Derivative(field)]) {
        Ok(jacobian) => Ok(
            if jacobian
                .coefficients()
                .iter()
                .any(|coefficient| *coefficient != 0.0)
            {
                DaeVariableKind::Differential
            } else {
                DaeVariableKind::Algebraic
            },
        ),
        Err(
            SymbolicLinearityFailure::VariableCoefficient { .. }
            | SymbolicLinearityFailure::Nonlinear { .. },
        ) => Ok(DaeVariableKind::Differential),
        Err(failure) => Err(invalid_artifact(format!(
            "general implicit variable partition failed: {failure:?}"
        ))),
    }
}

fn normalize_zeros(values: &mut [f64]) {
    for value in values {
        if *value == 0.0 {
            *value = 0.0;
        }
    }
}

fn is_negative_zero(value: f64) -> bool {
    value == 0.0 && value.is_sign_negative()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeneralImplicitTimeLoweringEnvelopeV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    model_ulid: String,
    semantic_revision: u64,
    relation_ulid: String,
    state_field_ulids: Vec<String>,
    variable_kinds: Vec<WireDaeVariableKind>,
    reason: WireGeneralImplicitReason,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireImplicitTimeInitialDataEnvelopeV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    semantic_revision: u64,
    lowering_sha256: String,
    initial_condition: WireInitialCondition,
    state: Vec<f64>,
    derivative: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireImplicitTimeRunManifestV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    semantic_revision: u64,
    lowering_sha256: String,
    input_initial_data_sha256: String,
    accepted_initial_data_sha256: String,
    plan: WireImplicitTimePlan,
    execution: WireImplicitTimeExecution,
    output_sha256: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireImplicitTimePlan {
    method: WireImplicitTimeMethod,
    start_time: f64,
    initial_step: f64,
    relative_tolerance: f64,
    absolute_tolerances: Vec<f64>,
    output_times: Vec<f64>,
}

impl WireImplicitTimePlan {
    fn encode(value: &TimePlan) -> Result<Self, Diagnostic> {
        let wire = Self {
            method: WireImplicitTimeMethod::encode(value.method())?,
            start_time: value.start_time(),
            initial_step: value.initial_step(),
            relative_tolerance: value.relative_tolerance(),
            absolute_tolerances: value.absolute_tolerances().to_vec(),
            output_times: value.output_times().to_vec(),
        };
        wire.decode()?;
        Ok(wire)
    }

    fn decode(&self) -> Result<TimePlan, Diagnostic> {
        TimePlan::new(
            self.method.decode(),
            self.start_time,
            self.initial_step,
            self.relative_tolerance,
            self.absolute_tolerances.clone(),
            self.output_times.clone(),
        )
        .map_err(|error| invalid_artifact(error.message()))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireImplicitTimeExecution {
    backend: String,
    backend_version: String,
    method: WireImplicitTimeMethod,
    equation_class: WireImplicitEquationClass,
    initial_condition: WireInitialCondition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireDaeVariableKind {
    Differential,
    Algebraic,
}

impl WireDaeVariableKind {
    const fn encode(value: DaeVariableKind) -> Self {
        match value {
            DaeVariableKind::Differential => Self::Differential,
            DaeVariableKind::Algebraic => Self::Algebraic,
        }
    }

    const fn decode(self) -> DaeVariableKind {
        match self {
            Self::Differential => DaeVariableKind::Differential,
            Self::Algebraic => DaeVariableKind::Algebraic,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireGeneralImplicitReason {
    NonconstantDerivativeJacobian,
    NonlinearDerivativeDependence,
}

impl WireGeneralImplicitReason {
    const fn encode(value: GeneralImplicitReason) -> Self {
        match value {
            GeneralImplicitReason::NonconstantDerivativeJacobian => {
                Self::NonconstantDerivativeJacobian
            }
            GeneralImplicitReason::NonlinearDerivativeDependence => {
                Self::NonlinearDerivativeDependence
            }
        }
    }

    const fn decode(self) -> GeneralImplicitReason {
        match self {
            Self::NonconstantDerivativeJacobian => {
                GeneralImplicitReason::NonconstantDerivativeJacobian
            }
            Self::NonlinearDerivativeDependence => {
                GeneralImplicitReason::NonlinearDerivativeDependence
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireInitialCondition {
    Provided,
    SolveConsistent,
}

impl WireInitialCondition {
    const fn encode(value: InitialConditionPolicy) -> Self {
        match value {
            InitialConditionPolicy::Provided => Self::Provided,
            InitialConditionPolicy::SolveConsistent => Self::SolveConsistent,
        }
    }

    const fn decode(self) -> InitialConditionPolicy {
        match self {
            Self::Provided => InitialConditionPolicy::Provided,
            Self::SolveConsistent => InitialConditionPolicy::SolveConsistent,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireImplicitTimeMethod {
    ImplicitEuler,
    Bdf,
}

impl WireImplicitTimeMethod {
    fn encode(value: TimeMethod) -> Result<Self, Diagnostic> {
        match value {
            TimeMethod::ImplicitEuler => Ok(Self::ImplicitEuler),
            TimeMethod::Bdf => Ok(Self::Bdf),
            TimeMethod::Tsitouras45 => Err(invalid_artifact(
                "Tsitouras45 cannot enter a residual-native time artifact",
            )),
        }
    }

    const fn decode(self) -> TimeMethod {
        match self {
            Self::ImplicitEuler => TimeMethod::ImplicitEuler,
            Self::Bdf => TimeMethod::Bdf,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireImplicitEquationClass {
    GeneralImplicitDae,
}

impl WireImplicitEquationClass {
    const fn decode(self) -> TimeEquationClass {
        match self {
            Self::GeneralImplicitDae => TimeEquationClass::GeneralImplicitDae,
        }
    }
}
