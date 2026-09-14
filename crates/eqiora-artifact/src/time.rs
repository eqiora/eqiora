use std::collections::HashSet;
use std::str::FromStr;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, OntologyId};
use eqiora_graph::EdgeKind;
use eqiora_ir::ScalarOperatorIr;
use eqiora_schema::Model;
use eqiora_schema::kernel::{ActivationKind, KernelNode, SymbolRef};
use eqiora_sem::KernelProgram;
use eqiora_time::{
    ConstantDerivativeMatrixProof, InitialConditionPolicy, MassMatrixRank, TimeEquationClass,
    TimeExecutionReport, TimeLoweringProof, TimeMethod, TimePlan,
};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, JsonDecoderLimits, ModelEnvelope, check_json_limits,
    invalid_artifact, validate_text,
};

const TIME_LOWERING_SCHEMA: &str = "eqiora.time-lowering-envelope/v1";
const TIME_RUN_SCHEMA: &str = "eqiora.time-run-manifest/v1";

/// Semantic work budgets shared by residual-native time artifact generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeDecoderLimits {
    /// Common JSON syntax admission.
    pub json: JsonDecoderLimits,
    /// Maximum state dimension for exact rational rank replay.
    pub max_exact_rank_dimension: usize,
    /// Maximum state dimension in a residual-native time artifact.
    pub max_time_state_dimension: usize,
    /// Maximum scalar root callbacks in one root registration envelope.
    pub max_root_functions: usize,
    /// Maximum Activation references summed across one root registration.
    pub max_root_activation_references: usize,
}

impl Default for TimeDecoderLimits {
    fn default() -> Self {
        Self {
            json: JsonDecoderLimits::default(),
            max_exact_rank_dimension: 128,
            max_time_state_dimension: 128,
            max_root_functions: 4_096,
            max_root_activation_references: 100_000,
        }
    }
}

/// Content-addressed witness linking one canonical Relation to its admitted
/// first-order equation class.
///
/// The wire records the complete residual-ordered constant derivative matrix
/// and its exact binary-rational rank, not only a class label. Construction
/// and external linkage validation independently lower the referenced Relation
/// to scalar Operator IR, compare every coefficient, and replay rank proof.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeLoweringEnvelopeV1 {
    wire: WireTimeLoweringEnvelopeV1,
}

impl TimeLoweringEnvelopeV1 {
    /// Bind a runtime-produced witness to one immutable model artifact and
    /// independently verify it against the canonical Relation.
    ///
    /// # Errors
    /// Returns `EQ0901` when model/program identity differs, referenced nodes
    /// are absent or incorrectly typed, activation is not continuous, state
    /// order differs, or the exact Operator-IR derivative Jacobian does not
    /// match the witness.
    pub fn from_proof(
        model: &ModelEnvelope,
        program: &KernelProgram,
        proof: &TimeLoweringProof,
    ) -> Result<Self, Diagnostic> {
        validate_model_program(model, program)?;
        validate_proof_program(proof, program)?;
        let wire = WireTimeLoweringEnvelopeV1 {
            schema: TIME_LOWERING_SCHEMA.to_owned(),
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
            equation_class: WireTimeEquationClass::encode(proof.equation_class())?,
            derivative_matrix: WireConstantDerivativeMatrixProof::encode(
                proof.derivative_matrix(),
            )?,
        };
        let envelope = Self { wire };
        envelope.validate_local()?;
        Ok(envelope)
    }

    /// Decode and locally validate a lowering envelope.
    ///
    /// Exact canonical linkage is rechecked with [`Self::validate_against`]
    /// after the referenced model artifact is loaded.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version, invalid-ID,
    /// or internally contradictory witness data.
    pub fn from_json(bytes: &[u8], limits: TimeDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid time lowering envelope JSON: {error}"))
        })?;
        let envelope = Self { wire };
        if envelope.wire.state_field_ulids.len() > limits.max_exact_rank_dimension {
            return Err(invalid_artifact(format!(
                "time lowering state dimension exceeds decoder limit {}",
                limits.max_exact_rank_dimension
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
            invalid_artifact(format!("cannot serialize time lowering envelope: {error}"))
        })
    }

    /// Domain-separated SHA-256 identity of the complete lowering witness.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            TIME_LOWERING_SCHEMA.as_bytes(),
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

    /// Decode the typed witness.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn proof(&self) -> Result<TimeLoweringProof, Diagnostic> {
        let relation = Id::<kinds::Relation>::from_ulid(parse_ulid(&self.wire.relation_ulid)?);
        let state_fields = self
            .wire
            .state_field_ulids
            .iter()
            .map(|value| parse_ulid(value).map(Id::<kinds::Field>::from_ulid))
            .collect::<Result<Vec<_>, _>>()?;
        let derivative_matrix = self.wire.derivative_matrix.decode(state_fields.len())?;
        TimeLoweringProof::new(relation, state_fields, derivative_matrix)
            .map_err(|error| invalid_artifact(error.message()))
    }

    /// Revalidate digest/revision linkage and the complete derivative matrix
    /// against separately loaded model and program artifacts.
    ///
    /// # Errors
    /// Returns `EQ0901` for any linkage or structural-proof drift.
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
                "time lowering model digest, identity, or revision does not match the supplied model",
            ));
        }
        validate_proof_program(&self.proof()?, program)
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != TIME_LOWERING_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported time-lowering-envelope schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        parse_ulid(&self.wire.model_ulid)?;
        parse_ulid(&self.wire.relation_ulid)?;
        for field in &self.wire.state_field_ulids {
            parse_ulid(field)?;
        }
        let proof = self.proof()?;
        if WireTimeEquationClass::encode(proof.equation_class())? != self.wire.equation_class {
            return Err(invalid_artifact(
                "time lowering equation class contradicts its exact derivative-matrix witness",
            ));
        }
        Ok(())
    }
}

/// Reproducible plan, adapter evidence, and content linkage for one production
/// first-order time run.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeRunManifestV1 {
    wire: WireTimeRunManifestV1,
}

impl TimeRunManifestV1 {
    /// Bind one validated time plan and accepted backend report to the exact
    /// lowering witness that admitted execution.
    ///
    /// # Errors
    /// Returns `EQ0901` if method, equation class, or initial-condition policy
    /// differs across plan/report/lowering, the plan selects a residual-native
    /// reference method, or the adapter-supplied backend version is invalid.
    pub fn new(
        lowering: &TimeLoweringEnvelopeV1,
        plan: &TimePlan,
        report: TimeExecutionReport,
    ) -> Result<Self, Diagnostic> {
        let backend_version = report.backend_version().to_owned();
        validate_text("time backend version", &backend_version)?;
        let proof = lowering.proof()?;
        if report.method() != plan.method()
            || report.equation_class() != proof.equation_class()
            || report.initial_condition() != proof.initial_condition_policy()
            || plan.absolute_tolerances().len() != proof.state_fields().len()
        {
            return Err(invalid_artifact(
                "time run plan/report contradicts the linked lowering witness",
            ));
        }
        let manifest = Self {
            wire: WireTimeRunManifestV1 {
                schema: TIME_RUN_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: lowering.model_artifact().0,
                semantic_revision: lowering.semantic_revision(),
                lowering_sha256: lowering.digest()?.0,
                plan: WireTimePlan::encode(plan)?,
                execution: WireTimeExecution {
                    backend: report.backend().to_owned(),
                    backend_version,
                    method: WireTimeMethod::encode(report.method())?,
                    equation_class: WireTimeEquationClass::encode(report.equation_class())?,
                    initial_condition: WireInitialCondition::encode(report.initial_condition()),
                },
                output_sha256: Vec::new(),
            },
        };
        manifest.validate_local()?;
        manifest.validate_against(lowering)?;
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

    /// Decode and locally validate a time run manifest.
    ///
    /// External linkage is rechecked with [`Self::validate_against`] after the
    /// referenced lowering artifact is loaded.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version, duplicate,
    /// non-finite, or internally contradictory data.
    pub fn from_json(bytes: &[u8], limits: TimeDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid time run JSON: {error}")))?;
        let manifest = Self { wire };
        manifest.validate_local()?;
        Ok(manifest)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire)
            .map_err(|error| invalid_artifact(format!("cannot serialize time run: {error}")))
    }

    /// Domain-separated SHA-256 identity of the complete run manifest.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            TIME_RUN_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Referenced canonical model artifact.
    #[must_use]
    pub fn model(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.model_sha256.clone())
    }

    /// Referenced lowering witness artifact.
    #[must_use]
    pub fn lowering(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.lowering_sha256.clone())
    }

    /// Semantic revision shared with the lowering witness.
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

    /// Revalidate model/revision/digest linkage and execution admission against
    /// a separately loaded lowering witness.
    ///
    /// # Errors
    /// Returns `EQ0901` for any linkage, method, equation-class, or
    /// initial-condition drift.
    pub fn validate_against(&self, lowering: &TimeLoweringEnvelopeV1) -> Result<(), Diagnostic> {
        let proof = lowering.proof()?;
        let plan = self.plan()?;
        if self.model() != lowering.model_artifact()
            || self.semantic_revision() != lowering.semantic_revision()
            || self.lowering() != lowering.digest()?
            || self.wire.execution.equation_class.decode() != proof.equation_class()
            || self.wire.execution.initial_condition.decode() != proof.initial_condition_policy()
            || self.wire.execution.method.decode() != plan.method()
            || plan.absolute_tolerances().len() != proof.state_fields().len()
        {
            return Err(invalid_artifact(
                "time run model/lowering/plan/execution linkage does not match",
            ));
        }
        Ok(())
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != TIME_RUN_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported time-run-manifest schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        ArtifactDigest::from_hex(self.wire.lowering_sha256.clone())?;
        validate_text("time backend", &self.wire.execution.backend)?;
        validate_text("time backend version", &self.wire.execution.backend_version)?;
        let plan = self.wire.plan.decode()?;
        if self.wire.execution.method.decode() != plan.method()
            || !self
                .wire
                .execution
                .equation_class
                .admits(self.wire.execution.initial_condition.decode())
        {
            return Err(invalid_artifact(
                "time run execution method or equation class contradicts its plan",
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
                "time run outputs must be sorted and unique",
            ));
        }
        Ok(())
    }
}

pub(crate) fn validate_model_program(
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

fn validate_proof_program(
    proof: &TimeLoweringProof,
    program: &KernelProgram,
) -> Result<(), Diagnostic> {
    let (operator, expected_fields) = canonical_time_operator(program, proof.relation())?;
    if expected_fields != proof.state_fields() {
        return Err(invalid_artifact(
            "time lowering proof state order differs from canonical symbol order",
        ));
    }
    let derivatives = proof
        .state_fields()
        .iter()
        .copied()
        .map(SymbolRef::Derivative)
        .collect::<Vec<_>>();
    let jacobian = operator
        .constant_symbol_jacobian(&derivatives)
        .map_err(|failure| {
            invalid_artifact(format!(
                "time lowering proof has no constant derivative Jacobian: {failure:?}"
            ))
        })?;
    let witness = proof.derivative_matrix();
    if jacobian.row_count() != witness.dimension() || jacobian.column_count() != witness.dimension()
    {
        return Err(invalid_artifact(
            "time lowering derivative matrix shape differs from Operator IR",
        ));
    }
    if jacobian.coefficients() != witness.coefficients() {
        return Err(invalid_artifact(
            "time lowering derivative matrix differs from Operator IR",
        ));
    }
    Ok(())
}

pub(crate) fn canonical_time_operator(
    program: &KernelProgram,
    relation_id: Id<kinds::Relation>,
) -> Result<(ScalarOperatorIr, Vec<Id<kinds::Field>>), Diagnostic> {
    let relation = match program.node(relation_id.erase()) {
        Some(KernelNode::Relation(relation)) => relation,
        _ => {
            return Err(invalid_artifact(
                "time lowering proof Relation is absent from the model",
            ));
        }
    };
    let activation = program
        .edges()
        .iter()
        .find(|edge| edge.kind() == EdgeKind::Activates && edge.to() == relation_id.erase())
        .map(|edge| edge.from())
        .ok_or_else(|| invalid_artifact("time lowering proof Relation has no Activation"))?;
    if !matches!(
        program.node(activation),
        Some(KernelNode::Activation(activation))
            if matches!(activation.kind(), ActivationKind::Continuous)
    ) {
        return Err(invalid_artifact(
            "time lowering proof requires a continuously activated Relation",
        ));
    }

    let typed = program
        .typed_relation_residual(relation.id())
        .map_err(|errors| invalid_artifact(errors[0].message()))?;
    let operator = ScalarOperatorIr::lower_typed_scalar(&typed)
        .map_err(|error| invalid_artifact(error.message()))?;
    let mut expected_fields = Vec::new();
    let mut seen = HashSet::new();
    for symbol in operator.symbols() {
        let field = match *symbol {
            SymbolRef::Field(field) | SymbolRef::Derivative(field) => Some(field),
            _ => None,
        };
        if let Some(field) = field.filter(|field| seen.insert(*field)) {
            expected_fields.push(field);
        }
    }
    Ok((operator, expected_fields))
}

pub(crate) fn parse_ulid(value: &str) -> Result<Ulid, Diagnostic> {
    Ulid::from_str(value).map_err(|_| invalid_artifact("time lowering ULID is malformed"))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTimeLoweringEnvelopeV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    model_ulid: String,
    semantic_revision: u64,
    relation_ulid: String,
    state_field_ulids: Vec<String>,
    equation_class: WireTimeEquationClass,
    derivative_matrix: WireConstantDerivativeMatrixProof,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTimeRunManifestV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    semantic_revision: u64,
    lowering_sha256: String,
    plan: WireTimePlan,
    execution: WireTimeExecution,
    output_sha256: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTimePlan {
    method: WireTimeMethod,
    start_time: f64,
    initial_step: f64,
    relative_tolerance: f64,
    absolute_tolerances: Vec<f64>,
    output_times: Vec<f64>,
}

impl WireTimePlan {
    fn encode(value: &TimePlan) -> Result<Self, Diagnostic> {
        let wire = Self {
            method: WireTimeMethod::encode(value.method())?,
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
struct WireTimeExecution {
    backend: String,
    backend_version: String,
    method: WireTimeMethod,
    equation_class: WireTimeEquationClass,
    initial_condition: WireInitialCondition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireTimeMethod {
    Tsitouras45,
    Bdf,
}

impl WireTimeMethod {
    fn encode(value: TimeMethod) -> Result<Self, Diagnostic> {
        match value {
            TimeMethod::ImplicitEuler => Err(invalid_artifact(
                "reference ImplicitEuler runs require a residual-native run artifact",
            )),
            TimeMethod::Tsitouras45 => Ok(Self::Tsitouras45),
            TimeMethod::Bdf => Ok(Self::Bdf),
        }
    }

    const fn decode(self) -> TimeMethod {
        match self {
            Self::Tsitouras45 => TimeMethod::Tsitouras45,
            Self::Bdf => TimeMethod::Bdf,
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
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireTimeEquationClass {
    ExplicitOde,
    MassMatrix { rank: WireMassMatrixRank },
}

impl WireTimeEquationClass {
    fn encode(value: TimeEquationClass) -> Result<Self, Diagnostic> {
        match value {
            TimeEquationClass::ExplicitOde => Ok(Self::ExplicitOde),
            TimeEquationClass::MassMatrix { rank } => Ok(Self::MassMatrix {
                rank: WireMassMatrixRank::encode(rank),
            }),
            TimeEquationClass::GeneralImplicitDae => Err(invalid_artifact(
                "general implicit DAE cannot enter a first-order lowering envelope",
            )),
        }
    }

    const fn decode(self) -> TimeEquationClass {
        match self {
            Self::ExplicitOde => TimeEquationClass::ExplicitOde,
            Self::MassMatrix { rank } => TimeEquationClass::MassMatrix {
                rank: rank.decode(),
            },
        }
    }

    const fn admits(self, initial_condition: InitialConditionPolicy) -> bool {
        matches!(
            (self.decode(), initial_condition),
            (
                TimeEquationClass::ExplicitOde,
                InitialConditionPolicy::Provided
            ) | (
                TimeEquationClass::MassMatrix {
                    rank: MassMatrixRank::Full
                },
                InitialConditionPolicy::Provided
            ) | (
                TimeEquationClass::MassMatrix {
                    rank: MassMatrixRank::RankDeficient
                },
                InitialConditionPolicy::SolveConsistent
            )
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireMassMatrixRank {
    Full,
    RankDeficient,
}

impl WireMassMatrixRank {
    const fn encode(value: MassMatrixRank) -> Self {
        match value {
            MassMatrixRank::Full => Self::Full,
            MassMatrixRank::RankDeficient => Self::RankDeficient,
        }
    }

    const fn decode(self) -> MassMatrixRank {
        match self {
            Self::Full => MassMatrixRank::Full,
            Self::RankDeficient => MassMatrixRank::RankDeficient,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireConstantDerivativeMatrixProof {
    coefficients: Vec<f64>,
    exact_rank: u64,
}

impl WireConstantDerivativeMatrixProof {
    fn encode(value: &ConstantDerivativeMatrixProof) -> Result<Self, Diagnostic> {
        Ok(Self {
            coefficients: value.coefficients().to_vec(),
            exact_rank: u64::try_from(value.exact_rank())
                .map_err(|_| invalid_artifact("exact derivative rank exceeds wire u64"))?,
        })
    }

    fn decode(&self, dimension: usize) -> Result<ConstantDerivativeMatrixProof, Diagnostic> {
        let expected_rank = usize::try_from(self.exact_rank)
            .map_err(|_| invalid_artifact("exact derivative rank exceeds local usize"))?;
        let proof = ConstantDerivativeMatrixProof::new(dimension, self.coefficients.clone())
            .map_err(|error| invalid_artifact(error.message()))?;
        if proof.exact_rank() != expected_rank {
            return Err(invalid_artifact(
                "time lowering exact rank contradicts its derivative matrix",
            ));
        }
        Ok(proof)
    }
}
