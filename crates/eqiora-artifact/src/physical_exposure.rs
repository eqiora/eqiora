//! Durable identity for physical Ports eliminated by hierarchy normalization.
//!
//! The canonical Semantic Model retains one maximal conserving Connection,
//! not transparent public aliases. This artifact preserves the exact cut
//! through that Connection which gives an eliminated exposure its observable
//! meaning. Numerical values remain in separate typed result artifacts.

use std::collections::BTreeSet;
use std::str::FromStr;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_graph::EdgeKind;
use eqiora_schema::kernel::{ConnectionSemantics, KernelNode};
use eqiora_sem::KernelProgram;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{ArtifactDigest, CANONICAL_ENCODING, check_json_limits, invalid_artifact};

mod observation;

pub use observation::{PhysicalExposureObservationBindingV1, PhysicalExposureQuantityV1};

const CATALOG_SCHEMA: &str = "eqiora.physical-exposure-catalog/v1";
const PROJECTION_SCHEMA: &str = "eqiora.physical-exposure-projection/v1";
const SOURCE_PATH_LIMIT: usize = 4_096;

/// Semantic work budgets for physical-exposure projection artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalExposureDecoderLimits {
    /// Common JSON syntax admission.
    pub json: crate::JsonDecoderLimits,
    /// Maximum eliminated physical exposures in one projection catalog.
    pub max_physical_exposure_projections: usize,
    /// Maximum retained Port identities summed across exposure cuts.
    pub max_physical_exposure_cut_members: usize,
    /// Maximum complete source origins in one exposure catalog.
    pub max_physical_exposure_origins: usize,
    /// Maximum source-path bytes summed across one exposure catalog.
    pub max_physical_exposure_source_path_bytes: usize,
}

impl Default for PhysicalExposureDecoderLimits {
    fn default() -> Self {
        Self {
            json: crate::JsonDecoderLimits::default(),
            max_physical_exposure_projections: 100_000,
            max_physical_exposure_cut_members: 1_000_000,
            max_physical_exposure_origins: 1_000_000,
            max_physical_exposure_source_path_bytes: 64 * 1_024 * 1_024,
        }
    }
}

/// One workspace-relative source span preserved as artifact provenance.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalExposureSourceSpanV1 {
    file: String,
    start: u32,
    end: u32,
}

impl PhysicalExposureSourceSpanV1 {
    /// Construct one half-open UTF-8 byte span.
    ///
    /// # Errors
    /// Returns `EQ0901` for an empty, oversized, control-containing, or
    /// reversed source location.
    pub fn new(file: impl Into<String>, start: u32, end: u32) -> Result<Self, Diagnostic> {
        let span = Self {
            file: file.into(),
            start,
            end,
        };
        span.validate()?;
        Ok(span)
    }

    /// Workspace-relative source identity.
    #[must_use]
    pub fn file(&self) -> &str {
        &self.file
    }

    /// Inclusive start byte.
    #[must_use]
    pub const fn start(&self) -> u32 {
        self.start
    }

    /// Exclusive end byte.
    #[must_use]
    pub const fn end(&self) -> u32 {
        self.end
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        if self.file.is_empty()
            || self.file.len() > SOURCE_PATH_LIMIT
            || self.file.chars().any(char::is_control)
        {
            return Err(invalid_artifact(
                "physical exposure source path must be bounded, nonempty UTF-8 without controls",
            ));
        }
        if self.start > self.end {
            return Err(invalid_artifact(
                "physical exposure source span must not be reversed",
            ));
        }
        Ok(())
    }
}

/// One indivisible definition/instance/binding origin.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalExposureSourceOriginV1 {
    definition: PhysicalExposureSourceSpanV1,
    instance: PhysicalExposureSourceSpanV1,
    bindings: Vec<PhysicalExposureSourceSpanV1>,
}

impl PhysicalExposureSourceOriginV1 {
    /// Construct one complete source origin with canonical binding order.
    ///
    /// # Errors
    /// Returns `EQ0901` if a source span is invalid.
    pub fn new(
        definition: PhysicalExposureSourceSpanV1,
        instance: PhysicalExposureSourceSpanV1,
        mut bindings: Vec<PhysicalExposureSourceSpanV1>,
    ) -> Result<Self, Diagnostic> {
        definition.validate()?;
        instance.validate()?;
        for binding in &bindings {
            binding.validate()?;
        }
        bindings.sort();
        bindings.dedup();
        Ok(Self {
            definition,
            instance,
            bindings,
        })
    }

    /// Definition source span.
    #[must_use]
    pub const fn definition(&self) -> &PhysicalExposureSourceSpanV1 {
        &self.definition
    }

    /// Instance occurrence source span.
    #[must_use]
    pub const fn instance(&self) -> &PhysicalExposureSourceSpanV1 {
        &self.instance
    }

    /// Canonically ordered binding source spans.
    #[must_use]
    pub fn bindings(&self) -> &[PhysicalExposureSourceSpanV1] {
        &self.bindings
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        self.definition.validate()?;
        self.instance.validate()?;
        for binding in &self.bindings {
            binding.validate()?;
        }
        if !strictly_sorted_unique(&self.bindings) && self.bindings.len() > 1 {
            return Err(invalid_artifact(
                "physical exposure binding spans must be sorted and unique",
            ));
        }
        Ok(())
    }
}

/// Closed nominal connector/support contract of one eliminated exposure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalExposureContractV1 {
    /// Scalar acausal across/through connector.
    ScalarPhysical {
        /// Full identity of the nominal scalar-physical Domain.
        connector_sha256: [u8; 32],
    },
    /// Field-valued trace/outward-flux connector on one exact boundary.
    FieldBoundary {
        /// Full identity of the nominal boundary-physical connector Domain.
        connector_sha256: [u8; 32],
        /// Full identity of the exact boundary Domain.
        boundary_sha256: [u8; 32],
    },
}

/// One eliminated exposure and the exact retained cut that gives it meaning.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalExposureProjectionV1 {
    projection_sha256: String,
    selector: String,
    exposure_sha256: String,
    connection_sha256: String,
    interior_port_sha256: Vec<String>,
    contract: WirePhysicalExposureContractV1,
    origins: Vec<PhysicalExposureSourceOriginV1>,
}

impl PhysicalExposureProjectionV1 {
    /// Construct one scalar-physical exposure projection.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed identity, cut, selector, or provenance
    /// data. Model membership is checked when the catalog is constructed.
    pub fn scalar(
        selector: impl Into<String>,
        exposure_sha256: [u8; 32],
        connection_sha256: [u8; 32],
        interior_port_sha256: Vec<[u8; 32]>,
        connector_sha256: [u8; 32],
        origins: Vec<PhysicalExposureSourceOriginV1>,
    ) -> Result<Self, Diagnostic> {
        Self::new(
            selector.into(),
            exposure_sha256,
            connection_sha256,
            interior_port_sha256,
            WirePhysicalExposureContractV1::ScalarPhysical {
                connector_sha256: encode_sha256(connector_sha256),
            },
            origins,
        )
    }

    /// Construct one field-valued boundary exposure projection.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed identity, cut, selector, or provenance
    /// data. Model membership is checked when the catalog is constructed.
    pub fn field_boundary(
        selector: impl Into<String>,
        exposure_sha256: [u8; 32],
        connection_sha256: [u8; 32],
        interior_port_sha256: Vec<[u8; 32]>,
        connector_sha256: [u8; 32],
        boundary_sha256: [u8; 32],
        origins: Vec<PhysicalExposureSourceOriginV1>,
    ) -> Result<Self, Diagnostic> {
        Self::new(
            selector.into(),
            exposure_sha256,
            connection_sha256,
            interior_port_sha256,
            WirePhysicalExposureContractV1::FieldBoundary {
                connector_sha256: encode_sha256(connector_sha256),
                boundary_sha256: encode_sha256(boundary_sha256),
            },
            origins,
        )
    }

    fn new(
        selector: String,
        exposure_sha256: [u8; 32],
        connection_sha256: [u8; 32],
        mut interior_port_sha256: Vec<[u8; 32]>,
        contract: WirePhysicalExposureContractV1,
        mut origins: Vec<PhysicalExposureSourceOriginV1>,
    ) -> Result<Self, Diagnostic> {
        interior_port_sha256.sort_unstable();
        interior_port_sha256.dedup();
        origins.sort();
        origins.dedup();
        let mut value = Self {
            projection_sha256: String::new(),
            selector,
            exposure_sha256: encode_sha256(exposure_sha256),
            connection_sha256: encode_sha256(connection_sha256),
            interior_port_sha256: interior_port_sha256
                .into_iter()
                .map(encode_sha256)
                .collect(),
            contract,
            origins,
        };
        value.projection_sha256 = value.compute_projection_id()?.to_string();
        value.validate_local()?;
        Ok(value)
    }

    /// Presentation selector; never used as durable identity.
    #[must_use]
    pub fn selector(&self) -> &str {
        &self.selector
    }

    /// Meaning identity derived without selector, source spans, or Model data.
    #[must_use]
    pub fn projection_id(&self) -> ArtifactDigest {
        ArtifactDigest(self.projection_sha256.clone())
    }

    /// Full identity of the eliminated public Port occurrence.
    #[must_use]
    pub fn exposure_sha256(&self) -> [u8; 32] {
        decode_sha256(&self.exposure_sha256).expect("validated projection identity")
    }

    /// Full identity of the final maximal conserving Connection.
    #[must_use]
    pub fn connection_sha256(&self) -> [u8; 32] {
        decode_sha256(&self.connection_sha256).expect("validated projection identity")
    }

    /// Full identities of the retained Ports inside the occurrence cut.
    #[must_use]
    pub fn interior_port_sha256(&self) -> Vec<[u8; 32]> {
        self.interior_port_sha256
            .iter()
            .map(|identity| decode_sha256(identity).expect("validated projection identity"))
            .collect()
    }

    /// Exact nominal connector/support contract.
    #[must_use]
    pub fn contract(&self) -> PhysicalExposureContractV1 {
        self.contract
            .decode()
            .expect("validated projection contract")
    }

    /// Complete source origins in canonical tuple order.
    #[must_use]
    pub fn origins(&self) -> &[PhysicalExposureSourceOriginV1] {
        &self.origins
    }

    fn compute_projection_id(&self) -> Result<ArtifactDigest, Diagnostic> {
        let identity = WireProjectionIdentityV1 {
            exposure_sha256: &self.exposure_sha256,
            connection_sha256: &self.connection_sha256,
            interior_port_sha256: &self.interior_port_sha256,
            contract: &self.contract,
        };
        let bytes = serde_json::to_vec(&identity).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize physical projection identity: {error}"
            ))
        })?;
        Ok(ArtifactDigest::compute(
            PROJECTION_SCHEMA.as_bytes(),
            &bytes,
        ))
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        validate_selector(&self.selector)?;
        ArtifactDigest::from_hex(self.projection_sha256.clone())?;
        decode_sha256(&self.exposure_sha256)?;
        decode_sha256(&self.connection_sha256)?;
        if self.interior_port_sha256.is_empty() {
            return Err(invalid_artifact(
                "physical exposure projection requires a nonempty interior cut",
            ));
        }
        for identity in &self.interior_port_sha256 {
            decode_sha256(identity)?;
        }
        if !strictly_sorted_unique(&self.interior_port_sha256) {
            return Err(invalid_artifact(
                "physical exposure interior identities must be sorted and unique",
            ));
        }
        self.contract.validate()?;
        if self.origins.is_empty() {
            return Err(invalid_artifact(
                "physical exposure projection requires complete source provenance",
            ));
        }
        for origin in &self.origins {
            origin.validate()?;
        }
        if !strictly_sorted_unique(&self.origins) && self.origins.len() > 1 {
            return Err(invalid_artifact(
                "physical exposure origins must be sorted and unique",
            ));
        }
        if self.compute_projection_id()?.as_str() != self.projection_sha256 {
            return Err(invalid_artifact(
                "physical exposure projection digest does not match its semantic cut",
            ));
        }
        Ok(())
    }

    fn validate_against(&self, program: &KernelProgram) -> Result<(), Diagnostic> {
        let exposure = projected_id::<kinds::Port>(&self.exposure_sha256)?;
        if program.node(exposure.erase()).is_some() {
            return Err(invalid_artifact(
                "an eliminated physical exposure must not be a retained Kernel Port",
            ));
        }
        let connection = projected_id::<kinds::Connection>(&self.connection_sha256)?;
        match program.node(connection.erase()) {
            Some(KernelNode::Connection(definition))
                if definition.semantics() == ConnectionSemantics::Conserving => {}
            _ => {
                return Err(invalid_artifact(
                    "physical exposure projection references no conserving Connection",
                ));
            }
        }
        let members = program
            .edges()
            .iter()
            .filter(|edge| edge.kind() == EdgeKind::Connects && edge.from() == connection.erase())
            .map(|edge| edge.to())
            .collect::<BTreeSet<_>>();
        if self.interior_port_sha256.len() >= members.len() {
            return Err(invalid_artifact(
                "physical exposure interior cut must be a proper subset of its Connection",
            ));
        }
        let contract = self.contract.decode()?;
        validate_domain(program, contract.connector_sha256())?;
        if let Some(boundary) = contract.boundary_sha256() {
            validate_domain(program, boundary)?;
        }
        for full_identity in &self.interior_port_sha256 {
            let port = projected_id::<kinds::Port>(full_identity)?;
            if !members.contains(&port.erase()) {
                return Err(invalid_artifact(
                    "physical exposure cut contains a Port outside its Connection",
                ));
            }
            let Some(KernelNode::Port(definition)) = program.node(port.erase()) else {
                return Err(invalid_artifact(
                    "physical exposure cut references no retained Port",
                ));
            };
            let matches = match contract {
                PhysicalExposureContractV1::ScalarPhysical { connector_sha256 } => {
                    definition.physical_domain()
                        == Some(projected_id::<kinds::Domain>(&encode_sha256(
                            connector_sha256,
                        ))?)
                }
                PhysicalExposureContractV1::FieldBoundary {
                    connector_sha256,
                    boundary_sha256,
                } => {
                    definition.boundary_physical_contract()
                        == Some((
                            projected_id::<kinds::Domain>(&encode_sha256(connector_sha256))?,
                            projected_id::<kinds::Domain>(&encode_sha256(boundary_sha256))?,
                        ))
                }
            };
            if !matches {
                return Err(invalid_artifact(
                    "physical exposure cut Port contradicts its connector/support contract",
                ));
            }
        }
        Ok(())
    }
}

impl PhysicalExposureContractV1 {
    const fn connector_sha256(self) -> [u8; 32] {
        match self {
            Self::ScalarPhysical { connector_sha256 }
            | Self::FieldBoundary {
                connector_sha256, ..
            } => connector_sha256,
        }
    }

    const fn boundary_sha256(self) -> Option<[u8; 32]> {
        match self {
            Self::ScalarPhysical { .. } => None,
            Self::FieldBoundary {
                boundary_sha256, ..
            } => Some(boundary_sha256),
        }
    }
}

/// Versioned catalog of all eliminated physical exposures in one compilation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalExposureCatalogEnvelopeV1 {
    wire: WirePhysicalExposureCatalogV1,
}

impl PhysicalExposureCatalogEnvelopeV1 {
    /// Seal a structurally valid physical exposure catalog for one admitted
    /// Model and package compilation.
    ///
    /// This L3 check proves flat-graph membership and connector/support
    /// consistency. A flat Model cannot authenticate occurrence topology,
    /// full-identity suffixes, or source spans. Exact package derivation proof
    /// requires `PackagedModelDocument::validate_physical_exposure_catalog`,
    /// which compares this artifact with compiler-owned sidecars.
    ///
    /// # Errors
    /// Returns `EQ0901` when identities, source lineage, ordering, graph
    /// membership, connector/support, or resource bounds are invalid.
    pub fn new(
        model_artifact: ArtifactDigest,
        program: &KernelProgram,
        package_compilation: ArtifactDigest,
        mut projections: Vec<PhysicalExposureProjectionV1>,
    ) -> Result<Self, Diagnostic> {
        projections.sort_by(|left, right| {
            left.exposure_sha256
                .cmp(&right.exposure_sha256)
                .then_with(|| left.selector.cmp(&right.selector))
        });
        let envelope = Self {
            wire: WirePhysicalExposureCatalogV1 {
                schema: CATALOG_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: model_artifact.to_string(),
                model_ulid: program.model().ulid().to_string(),
                semantic_revision: program.revision().0,
                source_lineage: WireSourceLineageV1::PackageCompilationV1 {
                    sha256: package_compilation.to_string(),
                },
                projections,
            },
        };
        envelope.validate_local(PhysicalExposureDecoderLimits::default())?;
        envelope.validate_against(model_artifact, program, package_compilation)?;
        Ok(envelope)
    }

    /// Decode and locally validate one bounded catalog.
    ///
    /// Complete Model and package-compilation replay is performed by
    /// [`Self::validate_against`].
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, noncanonical, inconsistent, or
    /// oversized artifact data.
    pub fn from_json(
        bytes: &[u8],
        limits: PhysicalExposureDecoderLimits,
    ) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid physical exposure catalog JSON: {error}"))
        })?;
        let envelope = Self { wire };
        envelope.validate_local(limits)?;
        Ok(envelope)
    }

    /// Canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize physical exposure catalog: {error}"
            ))
        })
    }

    /// Domain-separated content identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            CATALOG_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact canonical Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.model_sha256.clone())
    }

    /// Semantic graph revision captured by the catalog.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.semantic_revision
    }

    /// Exact package-compilation source lineage.
    #[must_use]
    pub fn package_compilation(&self) -> ArtifactDigest {
        match &self.wire.source_lineage {
            WireSourceLineageV1::PackageCompilationV1 { sha256 } => ArtifactDigest(sha256.clone()),
        }
    }

    /// Canonically ordered exposure projections.
    #[must_use]
    pub fn projections(&self) -> &[PhysicalExposureProjectionV1] {
        &self.wire.projections
    }

    /// Resolve one durable projection identity exactly.
    #[must_use]
    pub fn projection(&self, projection: &ArtifactDigest) -> Option<&PhysicalExposureProjectionV1> {
        self.wire
            .projections
            .iter()
            .find(|entry| entry.projection_sha256 == projection.as_str())
    }

    /// Replay Model identity, declared package lineage, and flat-graph cut
    /// structure.
    ///
    /// This deliberately does not prove that the cut or provenance was
    /// compiler-derived. The exact package façade must reseal and compare the
    /// complete catalog for that stronger claim.
    ///
    /// # Errors
    /// Returns `EQ0901` for stale/wrong linkage, missing entities, cut drift,
    /// or connector/support mismatch.
    pub fn validate_against(
        &self,
        model_artifact: ArtifactDigest,
        program: &KernelProgram,
        package_compilation: ArtifactDigest,
    ) -> Result<(), Diagnostic> {
        self.validate_local(PhysicalExposureDecoderLimits::default())?;
        if self.model_artifact() != model_artifact
            || self.wire.model_ulid != program.model().ulid().to_string()
            || self.semantic_revision() != program.revision().0
            || self.package_compilation() != package_compilation
        {
            return Err(invalid_artifact(
                "physical exposure catalog Model, revision, or package compilation differs",
            ));
        }
        for projection in &self.wire.projections {
            projection.validate_against(program)?;
        }
        Ok(())
    }

    fn validate_local(&self, limits: PhysicalExposureDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != CATALOG_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported physical-exposure-catalog schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        Ulid::from_str(&self.wire.model_ulid)
            .map_err(|_| invalid_artifact("physical exposure catalog Model ULID is malformed"))?;
        self.wire.source_lineage.validate()?;
        if self.wire.projections.len() > limits.max_physical_exposure_projections {
            return Err(invalid_artifact(
                "physical exposure projection count exceeds decoder limit",
            ));
        }
        if self.wire.projections.is_empty() {
            return Err(invalid_artifact(
                "physical exposure catalog requires at least one projection",
            ));
        }
        let mut cut_members = 0_usize;
        let mut origins = 0_usize;
        let mut path_bytes = 0_usize;
        let mut selectors = BTreeSet::new();
        let mut projection_ids = BTreeSet::new();
        for projection in &self.wire.projections {
            projection.validate_local()?;
            if !selectors.insert(projection.selector.as_str()) {
                return Err(invalid_artifact(
                    "physical exposure selectors must be unique",
                ));
            }
            if !projection_ids.insert(projection.projection_sha256.as_str()) {
                return Err(invalid_artifact(
                    "physical exposure projection identities must be unique",
                ));
            }
            cut_members = cut_members
                .checked_add(projection.interior_port_sha256.len())
                .ok_or_else(|| invalid_artifact("physical exposure cut count overflow"))?;
            origins = origins
                .checked_add(projection.origins.len())
                .ok_or_else(|| invalid_artifact("physical exposure origin count overflow"))?;
            for origin in &projection.origins {
                for span in std::iter::once(origin.definition())
                    .chain(std::iter::once(origin.instance()))
                    .chain(origin.bindings())
                {
                    path_bytes = path_bytes
                        .checked_add(span.file().len())
                        .ok_or_else(|| invalid_artifact("physical exposure path size overflow"))?;
                }
            }
        }
        if !self
            .wire
            .projections
            .windows(2)
            .all(|pair| pair[0].exposure_sha256 < pair[1].exposure_sha256)
        {
            return Err(invalid_artifact(
                "physical exposure catalog must be ordered by unique full exposure identity",
            ));
        }
        if cut_members > limits.max_physical_exposure_cut_members
            || origins > limits.max_physical_exposure_origins
            || path_bytes > limits.max_physical_exposure_source_path_bytes
        {
            return Err(invalid_artifact(
                "physical exposure catalog exceeds decoder resource limits",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePhysicalExposureCatalogV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    model_ulid: String,
    semantic_revision: u64,
    source_lineage: WireSourceLineageV1,
    projections: Vec<PhysicalExposureProjectionV1>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireSourceLineageV1 {
    PackageCompilationV1 { sha256: String },
}

impl WireSourceLineageV1 {
    fn validate(&self) -> Result<(), Diagnostic> {
        match self {
            Self::PackageCompilationV1 { sha256 } => {
                ArtifactDigest::from_hex(sha256.clone()).map(|_| ())
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WirePhysicalExposureContractV1 {
    ScalarPhysical {
        connector_sha256: String,
    },
    FieldBoundary {
        connector_sha256: String,
        boundary_sha256: String,
    },
}

impl WirePhysicalExposureContractV1 {
    fn validate(&self) -> Result<(), Diagnostic> {
        self.decode().map(|_| ())
    }

    fn decode(&self) -> Result<PhysicalExposureContractV1, Diagnostic> {
        match self {
            Self::ScalarPhysical { connector_sha256 } => {
                Ok(PhysicalExposureContractV1::ScalarPhysical {
                    connector_sha256: decode_sha256(connector_sha256)?,
                })
            }
            Self::FieldBoundary {
                connector_sha256,
                boundary_sha256,
            } => Ok(PhysicalExposureContractV1::FieldBoundary {
                connector_sha256: decode_sha256(connector_sha256)?,
                boundary_sha256: decode_sha256(boundary_sha256)?,
            }),
        }
    }
}

#[derive(Serialize)]
struct WireProjectionIdentityV1<'a> {
    exposure_sha256: &'a str,
    connection_sha256: &'a str,
    interior_port_sha256: &'a [String],
    contract: &'a WirePhysicalExposureContractV1,
}

fn validate_selector(selector: &str) -> Result<(), Diagnostic> {
    if selector.is_empty()
        || selector.len() > SOURCE_PATH_LIMIT
        || selector.chars().any(char::is_control)
    {
        Err(invalid_artifact(
            "physical exposure selector must be bounded, nonempty UTF-8 without controls",
        ))
    } else {
        Ok(())
    }
}

fn validate_domain(program: &KernelProgram, full_identity: [u8; 32]) -> Result<(), Diagnostic> {
    let id = projected_id::<kinds::Domain>(&encode_sha256(full_identity))?;
    if matches!(program.node(id.erase()), Some(KernelNode::Domain(_))) {
        Ok(())
    } else {
        Err(invalid_artifact(
            "physical exposure contract references no retained Domain",
        ))
    }
}

fn projected_id<E: eqiora_core::entity::Entity>(identity: &str) -> Result<Id<E>, Diagnostic> {
    let full = decode_sha256(identity)?;
    let mut short = [0_u8; 16];
    short.copy_from_slice(&full[..16]);
    Ok(Id::from_ulid(Ulid::from_bytes(short)))
}

fn encode_sha256(bytes: [u8; 32]) -> String {
    ArtifactDigest::from_sha256(bytes).to_string()
}

fn decode_sha256(value: &str) -> Result<[u8; 32], Diagnostic> {
    ArtifactDigest::from_hex(value.to_owned()).map(|digest| digest.sha256_bytes())
}

fn strictly_sorted_unique<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}
