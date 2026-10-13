//! Application-owned replay of bounded XDMF external-data plans.
//!
//! Syntax adapters remain L3 and have no artifact authority. This module is
//! the narrow L4 seam that replays an accepted XDMF plan through the shared
//! mesh, field, source-observation, and artifact contracts before issuing an
//! opaque lineage handle.

use eqiora_artifact::{
    ArtifactDigest, DiscreteFieldEnvelopeV1, ExternalAdapterIdentityV1, ExternalImportManifestV1,
    ExternalImportObservationV1, ExternalImportSelectionV1, ExternalImportSourceV1,
    ExternalRuntimeComponentV1, ResolvedArrayV1, ResolvedImportArrayV1, SelectedSourceEntityV1,
    SimplicialMeshEnvelopeV1, StructuralSelectorV1,
};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_io_xml::{
    XDMF_ADAPTER_ID, XDMF_ADAPTER_VERSION, XdmfArrayResponse, XdmfArrayValues, XdmfImportPlan,
    XdmfImportedField,
};
use eqiora_meshing::MeshQualityGate;

#[cfg(feature = "hdf5")]
mod hdf5;
#[cfg(feature = "hdf5")]
pub use hdf5::{import_xdmf_hdf5_v1, verify_xdmf_hdf5_import_v1};
/// Freshly derived XDMF artifacts ready for persistence.
///
/// This value is not a replay proof. Persist its versioned constituents and
/// independently reload them before calling [`verify_xdmf_import_v1`].
#[derive(Debug)]
pub struct XdmfImportArtifactsV1 {
    manifest: ExternalImportManifestV1,
    manifest_digest: ArtifactDigest,
    mesh: SimplicialMeshEnvelopeV1,
    fields: Vec<DiscreteFieldEnvelopeV1>,
}

impl XdmfImportArtifactsV1 {
    /// Canonical source-to-array-to-artifact lineage assertion.
    #[must_use]
    pub const fn manifest(&self) -> &ExternalImportManifestV1 {
        &self.manifest
    }

    /// Exact manifest identity computed during fresh derivation.
    #[must_use]
    pub const fn manifest_digest(&self) -> &ArtifactDigest {
        &self.manifest_digest
    }

    /// Accepted affine-simplex mesh artifact.
    #[must_use]
    pub const fn mesh(&self) -> &SimplicialMeshEnvelopeV1 {
        &self.mesh
    }

    /// Accepted mesh-bound fields in explicit XDMF selection order.
    #[must_use]
    pub fn fields(&self) -> &[DiscreteFieldEnvelopeV1] {
        &self.fields
    }
}

/// Opaque proof that persisted XDMF artifacts equal one fresh derivation from
/// the exact current plan and resolver responses.
///
/// The handle is intentionally nonserializable. Persist its versioned
/// artifacts, then obtain a new handle by replaying source bytes and resolved
/// arrays rather than deserializing an assertion of successful replay.
#[derive(Debug)]
pub struct VerifiedXdmfImportV1 {
    manifest: ExternalImportManifestV1,
    manifest_digest: ArtifactDigest,
    mesh: SimplicialMeshEnvelopeV1,
    fields: Vec<DiscreteFieldEnvelopeV1>,
}

impl VerifiedXdmfImportV1 {
    /// Canonical source-to-array-to-artifact lineage assertion.
    #[must_use]
    pub const fn manifest(&self) -> &ExternalImportManifestV1 {
        &self.manifest
    }

    /// Exact manifest identity computed during fresh replay.
    #[must_use]
    pub const fn manifest_digest(&self) -> &ArtifactDigest {
        &self.manifest_digest
    }

    /// Accepted affine-simplex mesh artifact.
    #[must_use]
    pub const fn mesh(&self) -> &SimplicialMeshEnvelopeV1 {
        &self.mesh
    }

    /// Accepted mesh-bound fields in explicit XDMF selection order.
    #[must_use]
    pub fn fields(&self) -> &[DiscreteFieldEnvelopeV1] {
        &self.fields
    }
}

/// Derive artifacts from one pure XDMF metadata plan.
///
/// Every response is a distinct external-source occurrence even when several
/// requests name the same locator or contain identical complete source bytes.
/// No caller-supplied artifact, manifest, digest, path resolution, or network
/// authority enters this function.
///
/// # Errors
/// Returns a structured diagnostic when the response set or a shared mesh,
/// field, or artifact invariant is invalid.
pub fn import_xdmf_v1(
    plan: &XdmfImportPlan,
    responses: &[XdmfArrayResponse],
    quality_gate: MeshQualityGate,
) -> Result<XdmfImportArtifactsV1, Diagnostic> {
    let candidate = derive_candidate(plan, responses, quality_gate)?;
    let DerivedCandidate {
        observation: _,
        manifest,
        manifest_digest,
        mesh,
        fields,
    } = candidate;
    Ok(XdmfImportArtifactsV1 {
        manifest,
        manifest_digest,
        mesh,
        fields,
    })
}

/// Replay exact persisted XDMF artifacts and issue an opaque verified-lineage
/// handle.
///
/// The expected manifest, mesh, and fields are caller-owned, independently
/// loaded artifacts. This function freshly derives the complete import and
/// requires exact manifest, accepted-content, order, and reference equality.
/// Resolver honesty remains outside this contract: the caller supplies both
/// complete source observations and normalized typed values.
///
/// # Errors
/// Returns `EQ0810` when the persisted artifacts differ from fresh derivation.
/// Shared mesh, field, and artifact contracts retain their own diagnostics.
pub fn verify_xdmf_import_v1(
    expected_manifest: &ExternalImportManifestV1,
    expected_mesh: &SimplicialMeshEnvelopeV1,
    expected_fields: &[DiscreteFieldEnvelopeV1],
    plan: &XdmfImportPlan,
    responses: &[XdmfArrayResponse],
    quality_gate: MeshQualityGate,
) -> Result<VerifiedXdmfImportV1, Diagnostic> {
    let candidate = derive_candidate(plan, responses, quality_gate)?;
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
            "persisted XDMF manifest or accepted artifacts differ from fresh derivation",
        ));
    }

    Ok(VerifiedXdmfImportV1 {
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
    plan: &XdmfImportPlan,
    responses: &[XdmfArrayResponse],
    quality_gate: MeshQualityGate,
) -> Result<DerivedCandidate, Diagnostic> {
    derive_candidate_with_context(
        plan,
        responses,
        quality_gate,
        ExternalAdapterIdentityV1::new(XDMF_ADAPTER_ID, XDMF_ADAPTER_VERSION)?,
        Vec::new(),
    )
}

fn derive_candidate_with_context(
    plan: &XdmfImportPlan,
    responses: &[XdmfArrayResponse],
    quality_gate: MeshQualityGate,
    adapter: ExternalAdapterIdentityV1,
    runtime_stack: Vec<ExternalRuntimeComponentV1>,
) -> Result<DerivedCandidate, Diagnostic> {
    let accepted = plan.accept(responses, quality_gate)?;
    let mesh = SimplicialMeshEnvelopeV1::from_mesh(accepted.mesh())?;
    let fields = accepted
        .fields()
        .iter()
        .map(|field| DiscreteFieldEnvelopeV1::from_payload(&mesh, field.payload()))
        .collect::<Result<Vec<_>, _>>()?;
    let observation = observation_from(plan, responses)?;
    let selection = selection_from(plan, accepted.fields())?;
    let manifest = ExternalImportManifestV1::from_observation(
        adapter,
        runtime_stack,
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
    plan: &XdmfImportPlan,
    responses: &[XdmfArrayResponse],
) -> Result<ExternalImportObservationV1, Diagnostic> {
    if responses.len() != plan.requests().len() {
        return Err(invalid_external_import(
            "XDMF response count differs from its immutable request plan",
        ));
    }
    let metadata = ExternalImportSourceV1::metadata_document(plan.metadata_bytes().to_vec(), None)?;
    let mut external_sources = Vec::with_capacity(responses.len());
    let mut resolved = Vec::with_capacity(responses.len());
    for (index, response) in responses.iter().enumerate() {
        let request = response.request();
        let source_ordinal = u32::try_from(index.checked_add(1).ok_or_else(|| {
            invalid_external_import("XDMF source occurrence ordinal overflows usize")
        })?)
        .map_err(|_| invalid_external_import("XDMF source occurrence ordinal exceeds u32"))?;
        let selector = StructuralSelectorV1::new(request.origin_selector().to_vec());
        external_sources.push(ExternalImportSourceV1::external_array_source(
            selector.clone(),
            response.source_bytes().to_vec(),
            Some(request.source_locator().to_owned()),
        )?);
        let shape = request.shape().to_vec();
        let array = match response.values() {
            XdmfArrayValues::U64(values) => ResolvedArrayV1::from_u64(shape, values.clone())?,
            XdmfArrayValues::F64(values) => ResolvedArrayV1::from_f64(shape, values.clone())?,
        };
        resolved.push(ResolvedImportArrayV1::new(
            source_ordinal,
            selector,
            Some(request.dataset_path().to_owned()),
            array,
        )?);
    }

    let mut arrays = resolved.into_iter();
    let geometry = arrays
        .next()
        .ok_or_else(|| invalid_external_import("XDMF plan has no Geometry array"))?;
    let topology = arrays
        .next()
        .ok_or_else(|| invalid_external_import("XDMF plan has no Topology array"))?;
    ExternalImportObservationV1::new(
        metadata,
        external_sources,
        geometry,
        topology,
        arrays.collect(),
    )
}

fn selection_from(
    plan: &XdmfImportPlan,
    fields: &[XdmfImportedField],
) -> Result<ExternalImportSelectionV1, Diagnostic> {
    let selected = plan.selection();
    if fields.len() != selected.attributes().len() {
        return Err(invalid_external_import(
            "accepted XDMF field count differs from explicit attribute selection",
        ));
    }
    let grid = SelectedSourceEntityV1::new(
        StructuralSelectorV1::new(selected.grid().to_vec()),
        plan.grid_name().map(str::to_owned),
    )?;
    let attributes = selected
        .attributes()
        .iter()
        .zip(fields)
        .map(|(path, field)| {
            if field.origin_selector() != path {
                return Err(invalid_external_import(
                    "accepted XDMF field origin differs from explicit attribute selection",
                ));
            }
            SelectedSourceEntityV1::new(
                StructuralSelectorV1::new(path.clone()),
                field.name().map(str::to_owned),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    ExternalImportSelectionV1::new(grid, attributes)
}

fn invalid_external_import(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXTERNAL_DATA_IMPORT, message)
}
