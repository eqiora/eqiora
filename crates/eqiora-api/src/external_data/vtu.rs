//! Application-owned artifact composition for bounded VTU imports.
//!
//! The L3 adapter owns XML and VTK grammar. This L4 seam owns only the
//! composition of accepted mesh/field values with the shared provenance and
//! artifact contracts, followed by exact persisted replay.

use eqiora_artifact::{
    ArtifactDigest, DiscreteFieldEnvelopeV1, ExternalAdapterIdentityV1, ExternalImportManifestV1,
    ExternalImportObservationV1, ExternalImportSelectionV1, ExternalImportSourceV1,
    ResolvedArrayV1, ResolvedImportArrayV1, SelectedSourceEntityV1, SimplicialMeshEnvelopeV1,
    StructuralSelectorV1,
};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_io_xml::{VTU_ADAPTER_ID, VTU_ADAPTER_VERSION, VtuImportPlan, VtuImportedField};
use eqiora_meshing::MeshQualityGate;

/// Freshly derived VTU artifacts ready for independent persistence.
///
/// This value is not a replay proof. Persist its constituents and reload them
/// independently before calling [`verify_vtu_import_v1`].
#[derive(Debug)]
pub struct VtuImportArtifactsV1 {
    manifest: ExternalImportManifestV1,
    manifest_digest: ArtifactDigest,
    mesh: SimplicialMeshEnvelopeV1,
    fields: Vec<DiscreteFieldEnvelopeV1>,
}

impl VtuImportArtifactsV1 {
    /// Canonical source-to-array-to-artifact lineage assertion.
    #[must_use]
    pub const fn manifest(&self) -> &ExternalImportManifestV1 {
        &self.manifest
    }

    /// Exact manifest identity computed during derivation.
    #[must_use]
    pub const fn manifest_digest(&self) -> &ArtifactDigest {
        &self.manifest_digest
    }

    /// Accepted affine-simplex mesh artifact.
    #[must_use]
    pub const fn mesh(&self) -> &SimplicialMeshEnvelopeV1 {
        &self.mesh
    }

    /// Accepted mesh-bound fields in explicit VTU selection order.
    #[must_use]
    pub fn fields(&self) -> &[DiscreteFieldEnvelopeV1] {
        &self.fields
    }
}

/// Opaque proof that persisted VTU artifacts equal one fresh accepted replay.
///
/// The handle is deliberately nonserializable. A caller obtains it only by
/// replaying the immutable format plan against independently loaded expected
/// artifacts.
#[derive(Debug)]
pub struct VerifiedVtuImportV1 {
    manifest: ExternalImportManifestV1,
    manifest_digest: ArtifactDigest,
    mesh: SimplicialMeshEnvelopeV1,
    fields: Vec<DiscreteFieldEnvelopeV1>,
}

impl VerifiedVtuImportV1 {
    /// Canonical source-to-array-to-artifact lineage assertion.
    #[must_use]
    pub const fn manifest(&self) -> &ExternalImportManifestV1 {
        &self.manifest
    }

    /// Exact manifest identity computed during replay.
    #[must_use]
    pub const fn manifest_digest(&self) -> &ArtifactDigest {
        &self.manifest_digest
    }

    /// Accepted affine-simplex mesh artifact.
    #[must_use]
    pub const fn mesh(&self) -> &SimplicialMeshEnvelopeV1 {
        &self.mesh
    }

    /// Accepted mesh-bound fields in explicit VTU selection order.
    #[must_use]
    pub fn fields(&self) -> &[DiscreteFieldEnvelopeV1] {
        &self.fields
    }
}

/// Derive artifacts from one pure, immutable VTU import plan.
///
/// # Errors
/// Returns a structured diagnostic when the shared mesh, field, provenance,
/// or artifact contracts reject the accepted adapter output.
pub fn import_vtu_v1(
    plan: &VtuImportPlan,
    quality_gate: MeshQualityGate,
) -> Result<VtuImportArtifactsV1, Diagnostic> {
    let candidate = derive_candidate(plan, quality_gate)?;
    Ok(VtuImportArtifactsV1 {
        manifest: candidate.manifest,
        manifest_digest: candidate.manifest_digest,
        mesh: candidate.mesh,
        fields: candidate.fields,
    })
}

/// Replay independently persisted VTU artifacts and issue an opaque handle.
///
/// # Errors
/// Returns a structured import or artifact diagnostic when fresh derivation
/// fails or any expected manifest, mesh, field, order, source, or accepted-
/// array identity differs from it.
pub fn verify_vtu_import_v1(
    expected_manifest: &ExternalImportManifestV1,
    expected_mesh: &SimplicialMeshEnvelopeV1,
    expected_fields: &[DiscreteFieldEnvelopeV1],
    plan: &VtuImportPlan,
    quality_gate: MeshQualityGate,
) -> Result<VerifiedVtuImportV1, Diagnostic> {
    let candidate = derive_candidate(plan, quality_gate)?;
    expected_manifest.validate_references(
        &candidate.observation,
        expected_mesh,
        expected_fields,
    )?;
    let expected_manifest_digest = expected_manifest.digest()?;

    if expected_mesh != &candidate.mesh
        || expected_fields != candidate.fields
        || expected_manifest != &candidate.manifest
        || expected_manifest_digest != candidate.manifest_digest
    {
        return Err(invalid_external_import(
            "persisted VTU manifest or accepted artifacts differ from fresh derivation",
        ));
    }

    Ok(VerifiedVtuImportV1 {
        manifest: candidate.manifest,
        manifest_digest: candidate.manifest_digest,
        mesh: candidate.mesh,
        fields: candidate.fields,
    })
}

#[derive(Debug)]
struct DerivedCandidate {
    observation: ExternalImportObservationV1,
    manifest: ExternalImportManifestV1,
    manifest_digest: ArtifactDigest,
    mesh: SimplicialMeshEnvelopeV1,
    fields: Vec<DiscreteFieldEnvelopeV1>,
}

fn derive_candidate(
    plan: &VtuImportPlan,
    quality_gate: MeshQualityGate,
) -> Result<DerivedCandidate, Diagnostic> {
    let accepted = plan.accept(quality_gate)?;
    let mesh = SimplicialMeshEnvelopeV1::from_mesh(accepted.mesh())?;
    let mut fields = Vec::new();
    fields
        .try_reserve_exact(accepted.fields().len())
        .map_err(|_| invalid_external_import("VTU field artifact allocation failed"))?;
    for field in accepted.fields() {
        fields.push(DiscreteFieldEnvelopeV1::from_payload(
            &mesh,
            field.payload(),
        )?);
    }
    let observation = observation_from(plan, accepted.fields())?;
    let selection = selection_from(plan, accepted.fields())?;
    let manifest = ExternalImportManifestV1::from_observation(
        ExternalAdapterIdentityV1::new(VTU_ADAPTER_ID, VTU_ADAPTER_VERSION)?,
        Vec::new(),
        selection,
        &observation,
        &mesh,
        &fields,
    )?;
    let manifest_digest = manifest.digest()?;
    Ok(DerivedCandidate {
        observation,
        manifest,
        manifest_digest,
        mesh,
        fields,
    })
}

fn observation_from(
    plan: &VtuImportPlan,
    fields: &[VtuImportedField],
) -> Result<ExternalImportObservationV1, Diagnostic> {
    let metadata = ExternalImportSourceV1::metadata_document(
        copy_slice(plan.source_bytes(), "VTU source observation")?,
        None,
    )?;
    let geometry = ResolvedImportArrayV1::new(
        0,
        StructuralSelectorV1::new(copy_slice(
            plan.geometry_selector(),
            "VTU geometry selector",
        )?),
        None,
        ResolvedArrayV1::from_f64(
            copy_slice(plan.geometry_shape(), "VTU geometry shape")?,
            copy_slice(plan.normalized_geometry(), "VTU geometry values")?,
        )?,
    )?;
    let topology = ResolvedImportArrayV1::new(
        0,
        StructuralSelectorV1::new(copy_slice(
            plan.topology_selector(),
            "VTU topology selector",
        )?),
        None,
        ResolvedArrayV1::from_u64(
            copy_slice(plan.topology_shape(), "VTU topology shape")?,
            copy_slice(plan.normalized_topology(), "VTU topology values")?,
        )?,
    )?;
    let mut resolved_fields = Vec::new();
    resolved_fields
        .try_reserve_exact(fields.len())
        .map_err(|_| invalid_external_import("VTU field observation allocation failed"))?;
    for field in fields {
        resolved_fields.push(ResolvedImportArrayV1::new(
            0,
            StructuralSelectorV1::new(copy_slice(field.selector(), "VTU field selector")?),
            None,
            ResolvedArrayV1::from_f64(
                copy_slice(field.raw_shape(), "VTU field shape")?,
                copy_slice(field.payload().values(), "VTU field values")?,
            )?,
        )?);
    }
    ExternalImportObservationV1::new(metadata, Vec::new(), geometry, topology, resolved_fields)
}

fn selection_from(
    plan: &VtuImportPlan,
    fields: &[VtuImportedField],
) -> Result<ExternalImportSelectionV1, Diagnostic> {
    if fields.len() != plan.selection().fields().len() {
        return Err(invalid_external_import(
            "accepted VTU field count differs from the explicit selection",
        ));
    }
    let piece = SelectedSourceEntityV1::new(
        StructuralSelectorV1::new(copy_slice(plan.selection().piece(), "VTU Piece selector")?),
        None,
    )?;
    let fields = plan
        .selection()
        .fields()
        .iter()
        .zip(fields)
        .map(|(selector, field)| {
            if field.selector() != selector {
                return Err(invalid_external_import(
                    "accepted VTU field origin differs from the explicit selection",
                ));
            }
            SelectedSourceEntityV1::new(
                StructuralSelectorV1::new(copy_slice(selector, "VTU selected field selector")?),
                field.name().map(str::to_owned),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    ExternalImportSelectionV1::new(piece, fields)
}

fn invalid_external_import(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXTERNAL_DATA_IMPORT, message)
}

fn copy_slice<T: Copy>(values: &[T], label: &str) -> Result<Vec<T>, Diagnostic> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(values.len())
        .map_err(|_| invalid_external_import(format!("{label} allocation failed")))?;
    copy.extend_from_slice(values);
    Ok(copy)
}
