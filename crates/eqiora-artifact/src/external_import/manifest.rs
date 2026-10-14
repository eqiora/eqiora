//! Canonical assertion linking external source occurrences, normalized arrays,
//! and independently accepted artifacts.

use std::collections::BTreeSet;

use eqiora_core::Diagnostic;
use serde::{Deserialize, Serialize};

use super::types::ExternalSourceKind;
use super::wire::{WireAdapter, WireRuntimeComponent};
use super::{
    ExternalAdapterIdentityV1, ExternalImportObservationV1, ExternalImportSelectionV1,
    ExternalRuntimeComponentV1, RawSourceSha256, SelectedSourceEntityV1, StructuralSelectorV1,
};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, DiscreteFieldEnvelopeV1, ExternalImportDecoderLimits,
    ResolvedArrayScalarV1, SimplicialMeshEnvelopeV1, check_json_limits, invalid_artifact,
    validate_text,
};

const EXTERNAL_IMPORT_SCHEMA: &str = "eqiora.external-import-manifest/v1";

/// Canonical lineage assertion for one external import attempt.
///
/// This manifest keeps three independently content-addressed facts in one
/// ordered record: complete source occurrences, normalized arrays presented to
/// Eqiora, and artifacts accepted by Eqiora. Construction and
/// [`Self::validate_references`] establish exact identity linkage only. They do
/// not prove that a source produced an array or that an array produced an
/// artifact; only a separately named deterministic format replay may make that
/// bounded derivation claim.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalImportManifestV1 {
    wire: WireExternalImportManifestV1,
    adapter: ExternalAdapterIdentityV1,
    runtime_stack: Vec<ExternalRuntimeComponentV1>,
    selection: ExternalImportSelectionV1,
}

impl ExternalImportManifestV1 {
    /// Capture exact source, normalized-array, and accepted-artifact identities.
    ///
    /// Digests are always computed from the supplied complete objects. No
    /// caller-provided source/array/artifact digest pair is admitted.
    ///
    /// # Errors
    /// Returns `EQ0901` for duplicate runtime identities, inconsistent
    /// selection/array/artifact cardinality, invalid field-to-mesh linkage, or
    /// a value that cannot be represented by the portable v1 wire.
    pub fn from_observation(
        adapter: ExternalAdapterIdentityV1,
        runtime_stack: Vec<ExternalRuntimeComponentV1>,
        selection: ExternalImportSelectionV1,
        observation: &ExternalImportObservationV1,
        mesh: &SimplicialMeshEnvelopeV1,
        fields: &[DiscreteFieldEnvelopeV1],
    ) -> Result<Self, Diagnostic> {
        validate_runtime_stack(&runtime_stack)?;
        validate_cardinality_and_selection(&selection, observation, fields.len())?;
        for field in fields {
            field.validate_mesh_artifact(mesh)?;
        }

        let sources = std::iter::once(observation.metadata())
            .chain(observation.external_sources())
            .enumerate()
            .map(|(index, source)| {
                Ok(WireSource {
                    ordinal: portable_ordinal(index, "external import source")?,
                    role: match source.kind() {
                        ExternalSourceKind::MetadataDocument => WireSourceRole::MetadataDocument,
                        ExternalSourceKind::ExternalArraySource => {
                            WireSourceRole::ExternalArraySource
                        }
                    },
                    origin_selector: source.origin_selector().into(),
                    display_locator: source.display_locator().map(str::to_owned),
                    source_sha256: source.digest().to_string(),
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;

        let resolved_arrays =
            std::iter::once((WireResolvedRole::MeshGeometry, observation.mesh_geometry()))
                .chain(std::iter::once((
                    WireResolvedRole::MeshTopology,
                    observation.mesh_topology(),
                )))
                .chain(
                    observation
                        .fields()
                        .iter()
                        .map(|field| (WireResolvedRole::Field, field)),
                )
                .enumerate()
                .map(|(index, (role, resolved))| {
                    Ok(WireResolvedArray {
                        ordinal: portable_ordinal(index, "resolved import array")?,
                        role,
                        source_ordinal: resolved.source_ordinal(),
                        origin_selector: resolved.origin_selector().into(),
                        storage_display_selector: resolved
                            .storage_display_selector()
                            .map(str::to_owned),
                        scalar: resolved.array().scalar().into(),
                        shape: resolved.array().shape().to_vec(),
                        resolved_sha256: resolved.array().digest()?.to_string(),
                    })
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;

        let accepted_artifacts =
            std::iter::once(mesh.digest().map(|digest| (WireAcceptedRole::Mesh, digest)))
                .chain(fields.iter().map(|field| {
                    field
                        .digest()
                        .map(|digest| (WireAcceptedRole::Field, digest))
                }))
                .enumerate()
                .map(|(index, artifact)| {
                    let (role, digest) = artifact?;
                    Ok(WireAcceptedArtifact {
                        ordinal: portable_ordinal(index, "accepted import artifact")?,
                        role,
                        artifact_sha256: digest.to_string(),
                    })
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;

        Ok(Self {
            wire: WireExternalImportManifestV1 {
                schema: EXTERNAL_IMPORT_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                adapter: (&adapter).into(),
                runtime_stack: runtime_stack.iter().map(Into::into).collect(),
                selection: (&selection).into(),
                sources,
                resolved_arrays,
                accepted_artifacts,
            },
            adapter,
            runtime_stack,
            selection,
        })
    }

    /// Decode the exact closed DTO under byte, nesting, text, list, rank, and
    /// shape-product limits. Decoding performs no I/O and does not replay an
    /// adapter.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed or unknown data, non-canonical ordering,
    /// invalid cross-references, or any resource-limit excess.
    pub fn from_json(
        bytes: &[u8],
        limits: ExternalImportDecoderLimits,
    ) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire: WireExternalImportManifestV1 =
            serde_json::from_slice(bytes).map_err(|error| {
                invalid_artifact(format!("invalid external import manifest JSON: {error}"))
            })?;
        let (adapter, runtime_stack, selection) = validate_wire(&wire, limits)?;
        Ok(Self {
            wire,
            adapter,
            runtime_stack,
            selection,
        })
    }

    /// Deterministic ordered DTO bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize external import manifest: {error}"
            ))
        })
    }

    /// Domain-separated SHA-256 identity of the complete lineage assertion.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            EXTERNAL_IMPORT_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact adapter identity named by this assertion.
    #[must_use]
    pub const fn adapter(&self) -> &ExternalAdapterIdentityV1 {
        &self.adapter
    }

    /// Exact native runtime stack in outer-to-inner call order.
    #[must_use]
    pub fn runtime_stack(&self) -> &[ExternalRuntimeComponentV1] {
        &self.runtime_stack
    }

    /// Explicit grid and caller-ordered attribute selection.
    #[must_use]
    pub const fn selection(&self) -> &ExternalImportSelectionV1 {
        &self.selection
    }

    /// Exact accepted mesh artifact identity.
    #[must_use]
    pub fn accepted_mesh_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.accepted_artifacts[0].artifact_sha256.clone())
    }

    /// Accepted field artifact identities in selection order. Equal identities
    /// remain repeated because occurrence order is provenance.
    #[must_use]
    pub fn accepted_field_artifacts(&self) -> Vec<ArtifactDigest> {
        self.wire.accepted_artifacts[1..]
            .iter()
            .map(|artifact| ArtifactDigest(artifact.artifact_sha256.clone()))
            .collect()
    }

    /// Recompute all recorded source, normalized-array, and accepted-artifact
    /// identities from independently loaded values and require the complete
    /// canonical assertion to match. Adapter identity, runtime stack, and
    /// selection remain the manifest's asserted import plan and are reused
    /// unchanged for this comparison.
    ///
    /// This is reference validation, not derivation proof. A deliberately
    /// cross-wired but internally self-consistent source/array/artifact triple
    /// can pass this method; adapter-specific deterministic replay must reject
    /// such a triple before issuing a verified-lineage handle.
    ///
    /// # Errors
    /// Returns `EQ0901` if any independently supplied source byte, normalized
    /// array, mesh, field, order, or associated display provenance differs.
    pub fn validate_references(
        &self,
        observation: &ExternalImportObservationV1,
        mesh: &SimplicialMeshEnvelopeV1,
        fields: &[DiscreteFieldEnvelopeV1],
    ) -> Result<(), Diagnostic> {
        let candidate = Self::from_observation(
            self.adapter.clone(),
            self.runtime_stack.clone(),
            self.selection.clone(),
            observation,
            mesh,
            fields,
        )?;
        if candidate.wire != self.wire {
            return Err(invalid_artifact(
                "external import manifest references differ from independently supplied values",
            ));
        }
        Ok(())
    }
}

fn validate_wire(
    wire: &WireExternalImportManifestV1,
    limits: ExternalImportDecoderLimits,
) -> Result<
    (
        ExternalAdapterIdentityV1,
        Vec<ExternalRuntimeComponentV1>,
        ExternalImportSelectionV1,
    ),
    Diagnostic,
> {
    if wire.schema != EXTERNAL_IMPORT_SCHEMA || wire.encoding != CANONICAL_ENCODING {
        return Err(invalid_artifact(
            "unsupported external-import schema or canonical encoding",
        ));
    }
    require_count(
        "external import runtime entries",
        wire.runtime_stack.len(),
        limits.max_import_runtime_entries,
    )?;
    require_count(
        "external import selected attributes",
        wire.selection.attributes.len(),
        limits.max_import_selection_attributes,
    )?;
    require_count(
        "external import sources",
        wire.sources.len(),
        limits.max_import_sources,
    )?;
    require_count(
        "external import resolved arrays",
        wire.resolved_arrays.len(),
        limits.max_import_resolved_arrays,
    )?;
    require_count(
        "external import accepted artifacts",
        wire.accepted_artifacts.len(),
        limits.max_import_accepted_artifacts,
    )?;
    validate_text_budget(wire, limits.max_import_manifest_text_bytes)?;

    let adapter = wire.adapter.decode()?;
    let runtime_stack = wire
        .runtime_stack
        .iter()
        .map(WireRuntimeComponent::decode)
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    validate_runtime_stack(&runtime_stack)?;
    let selection = selection_from_wire(&wire.selection)?;

    validate_sources(&wire.sources)?;
    validate_resolved_arrays(&wire.resolved_arrays, &wire.sources, &selection, limits)?;
    validate_accepted_artifacts(&wire.accepted_artifacts, selection.attributes().len())?;
    Ok((adapter, runtime_stack, selection))
}

fn validate_runtime_stack(stack: &[ExternalRuntimeComponentV1]) -> Result<(), Diagnostic> {
    let mut identities = BTreeSet::new();
    for entry in stack {
        if !identities.insert((entry.role(), entry.implementation())) {
            return Err(invalid_artifact(
                "external import runtime role/implementation pairs must be unique",
            ));
        }
    }
    Ok(())
}

fn validate_cardinality_and_selection(
    selection: &ExternalImportSelectionV1,
    observation: &ExternalImportObservationV1,
    field_count: usize,
) -> Result<(), Diagnostic> {
    if observation.fields().len() != selection.attributes().len()
        || field_count != selection.attributes().len()
    {
        return Err(invalid_artifact(
            "selected attributes, resolved fields, and accepted fields must have equal counts",
        ));
    }
    for (attribute, resolved) in selection.attributes().iter().zip(observation.fields()) {
        if attribute.selector() != resolved.origin_selector() {
            return Err(invalid_artifact(
                "resolved field origin selectors must follow explicit attribute selection order",
            ));
        }
    }
    Ok(())
}

fn validate_sources(sources: &[WireSource]) -> Result<(), Diagnostic> {
    if sources.is_empty() {
        return Err(invalid_artifact(
            "external import requires one metadata source at ordinal zero",
        ));
    }
    let mut external_selectors = BTreeSet::new();
    for (index, source) in sources.iter().enumerate() {
        require_ordinal(source.ordinal, index, "external import source")?;
        RawSourceSha256::from_hex(source.source_sha256.clone())?;
        validate_optional_wire_text(
            "external source display locator",
            source.display_locator.as_deref(),
        )?;
        match (
            index,
            source.role,
            source.origin_selector.element_path.is_empty(),
        ) {
            (0, WireSourceRole::MetadataDocument, true) => {}
            (0, _, _) => {
                return Err(invalid_artifact(
                    "source ordinal zero must be the root metadata document",
                ));
            }
            (_, WireSourceRole::ExternalArraySource, false) => {
                if !external_selectors.insert(source.origin_selector.clone()) {
                    return Err(invalid_artifact(
                        "external source occurrence selectors must be unique",
                    ));
                }
            }
            _ => {
                return Err(invalid_artifact(
                    "nonzero sources must be non-root external-array occurrences",
                ));
            }
        }
    }
    Ok(())
}

fn validate_resolved_arrays(
    arrays: &[WireResolvedArray],
    sources: &[WireSource],
    selection: &ExternalImportSelectionV1,
    limits: ExternalImportDecoderLimits,
) -> Result<(), Diagnostic> {
    if arrays.len() != selection.attributes().len().saturating_add(2) {
        return Err(invalid_artifact(
            "resolved arrays must contain geometry, topology, then one entry per selected field",
        ));
    }
    let mut selectors = BTreeSet::new();
    let mut seen_sources = vec![false; sources.len()];
    let mut next_external = 1_usize;
    for (index, array) in arrays.iter().enumerate() {
        require_ordinal(array.ordinal, index, "resolved import array")?;
        let expected_role = match index {
            0 => WireResolvedRole::MeshGeometry,
            1 => WireResolvedRole::MeshTopology,
            _ => WireResolvedRole::Field,
        };
        if array.role != expected_role {
            return Err(invalid_artifact(
                "resolved array roles must be geometry, topology, then fields",
            ));
        }
        if array.origin_selector.element_path.is_empty()
            || !selectors.insert(array.origin_selector.clone())
        {
            return Err(invalid_artifact(
                "resolved array origin selectors must be non-root and unique",
            ));
        }
        validate_optional_wire_text(
            "resolved array storage display selector",
            array.storage_display_selector.as_deref(),
        )?;
        if index >= 2
            && array.origin_selector.element_path
                != selection.attributes()[index - 2].selector().element_path()
        {
            return Err(invalid_artifact(
                "resolved field origins must follow explicit attribute selection order",
            ));
        }
        let source = usize::try_from(array.source_ordinal)
            .map_err(|_| invalid_artifact("resolved source ordinal exceeds local usize"))?;
        if source >= sources.len() {
            return Err(invalid_artifact(
                "resolved import array references a missing source occurrence",
            ));
        }
        if source > 0 && !seen_sources[source] {
            if source != next_external {
                return Err(invalid_artifact(
                    "external source ordinals must follow resolved-array first-use order",
                ));
            }
            if sources[source].origin_selector != array.origin_selector {
                return Err(invalid_artifact(
                    "external source occurrence must be declared by its first resolved array origin",
                ));
            }
            seen_sources[source] = true;
            next_external += 1;
        }
        validate_shape(&array.shape, limits)?;
        ArtifactDigest::from_hex(array.resolved_sha256.clone())?;
    }
    if arrays[0].scalar != WireScalar::F64 || arrays[1].scalar != WireScalar::U64 {
        return Err(invalid_artifact(
            "mesh geometry requires f64 and mesh topology requires u64",
        ));
    }
    if next_external != sources.len() {
        return Err(invalid_artifact(
            "every external source occurrence must be used by a resolved array",
        ));
    }
    Ok(())
}

fn validate_accepted_artifacts(
    artifacts: &[WireAcceptedArtifact],
    field_count: usize,
) -> Result<(), Diagnostic> {
    if artifacts.len() != field_count.saturating_add(1) {
        return Err(invalid_artifact(
            "accepted artifacts must contain one mesh then one entry per selected field",
        ));
    }
    for (index, artifact) in artifacts.iter().enumerate() {
        require_ordinal(artifact.ordinal, index, "accepted import artifact")?;
        let expected_role = if index == 0 {
            WireAcceptedRole::Mesh
        } else {
            WireAcceptedRole::Field
        };
        if artifact.role != expected_role {
            return Err(invalid_artifact(
                "accepted artifact roles must be mesh then fields",
            ));
        }
        ArtifactDigest::from_hex(artifact.artifact_sha256.clone())?;
    }
    Ok(())
}

fn validate_shape(shape: &[u64], limits: ExternalImportDecoderLimits) -> Result<(), Diagnostic> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(invalid_artifact(
            "resolved import array shapes require positive dimensions",
        ));
    }
    require_count(
        "resolved import array rank",
        shape.len(),
        limits.resolved_array.max_rank,
    )?;
    let product = shape.iter().try_fold(1_usize, |product, &dimension| {
        let dimension = usize::try_from(dimension)
            .map_err(|_| invalid_artifact("resolved array dimension exceeds local usize"))?;
        product
            .checked_mul(dimension)
            .ok_or_else(|| invalid_artifact("resolved array shape product overflows usize"))
    })?;
    require_count(
        "resolved import array scalar values",
        product,
        limits.resolved_array.max_values,
    )
}

fn validate_text_budget(
    wire: &WireExternalImportManifestV1,
    limit: usize,
) -> Result<(), Diagnostic> {
    let mut total = 0_usize;
    let mut add = |value: &str| -> Result<(), Diagnostic> {
        total = total.checked_add(value.len()).ok_or_else(|| {
            invalid_artifact("external import manifest text count overflows usize")
        })?;
        if total > limit {
            return Err(invalid_artifact(format!(
                "external import manifest text bytes {total} exceed decoder limit {limit}",
            )));
        }
        Ok(())
    };
    add(&wire.adapter.id)?;
    add(&wire.adapter.version)?;
    for entry in &wire.runtime_stack {
        add(&entry.implementation)?;
        add(&entry.version)?;
    }
    add_optional(&mut add, wire.selection.grid.display_name.as_deref())?;
    for attribute in &wire.selection.attributes {
        add_optional(&mut add, attribute.display_name.as_deref())?;
    }
    for source in &wire.sources {
        add_optional(&mut add, source.display_locator.as_deref())?;
    }
    for array in &wire.resolved_arrays {
        add_optional(&mut add, array.storage_display_selector.as_deref())?;
    }
    Ok(())
}

fn add_optional(
    add: &mut impl FnMut(&str) -> Result<(), Diagnostic>,
    value: Option<&str>,
) -> Result<(), Diagnostic> {
    if let Some(value) = value {
        add(value)?;
    }
    Ok(())
}

fn validate_optional_wire_text(label: &str, value: Option<&str>) -> Result<(), Diagnostic> {
    if let Some(value) = value {
        validate_text(label, value)?;
    }
    Ok(())
}

fn selection_from_wire(wire: &WireSelection) -> Result<ExternalImportSelectionV1, Diagnostic> {
    let grid = selected_from_wire(&wire.grid)?;
    let attributes = wire
        .attributes
        .iter()
        .map(selected_from_wire)
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    ExternalImportSelectionV1::new(grid, attributes)
}

fn selected_from_wire(wire: &WireSelectedEntity) -> Result<SelectedSourceEntityV1, Diagnostic> {
    SelectedSourceEntityV1::new(
        StructuralSelectorV1::new(wire.selector.element_path.clone()),
        wire.display_name.clone(),
    )
}

fn portable_ordinal(index: usize, label: &str) -> Result<u32, Diagnostic> {
    u32::try_from(index).map_err(|_| invalid_artifact(format!("{label} ordinal exceeds u32")))
}

fn require_ordinal(ordinal: u32, index: usize, label: &str) -> Result<(), Diagnostic> {
    if portable_ordinal(index, label)? != ordinal {
        return Err(invalid_artifact(format!(
            "{label} ordinals must be contiguous from zero",
        )));
    }
    Ok(())
}

fn require_count(label: &str, actual: usize, limit: usize) -> Result<(), Diagnostic> {
    if actual > limit {
        Err(invalid_artifact(format!(
            "{label} count {actual} exceeds decoder limit {limit}",
        )))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireExternalImportManifestV1 {
    schema: String,
    encoding: String,
    adapter: WireAdapter,
    runtime_stack: Vec<WireRuntimeComponent>,
    selection: WireSelection,
    sources: Vec<WireSource>,
    resolved_arrays: Vec<WireResolvedArray>,
    accepted_artifacts: Vec<WireAcceptedArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSelection {
    grid: WireSelectedEntity,
    attributes: Vec<WireSelectedEntity>,
}

impl From<&ExternalImportSelectionV1> for WireSelection {
    fn from(value: &ExternalImportSelectionV1) -> Self {
        Self {
            grid: value.grid().into(),
            attributes: value.attributes().iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSelectedEntity {
    selector: WireStructuralSelector,
    display_name: Option<String>,
}

impl From<&SelectedSourceEntityV1> for WireSelectedEntity {
    fn from(value: &SelectedSourceEntityV1) -> Self {
        Self {
            selector: value.selector().into(),
            display_name: value.display_name().map(str::to_owned),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireStructuralSelector {
    element_path: Vec<u32>,
}

impl From<&StructuralSelectorV1> for WireStructuralSelector {
    fn from(value: &StructuralSelectorV1) -> Self {
        Self {
            element_path: value.element_path().to_vec(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSource {
    ordinal: u32,
    role: WireSourceRole,
    origin_selector: WireStructuralSelector,
    display_locator: Option<String>,
    source_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireSourceRole {
    MetadataDocument,
    ExternalArraySource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireResolvedArray {
    ordinal: u32,
    role: WireResolvedRole,
    source_ordinal: u32,
    origin_selector: WireStructuralSelector,
    storage_display_selector: Option<String>,
    scalar: WireScalar,
    shape: Vec<u64>,
    resolved_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireResolvedRole {
    MeshGeometry,
    MeshTopology,
    Field,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireScalar {
    U64,
    F64,
}

impl From<ResolvedArrayScalarV1> for WireScalar {
    fn from(value: ResolvedArrayScalarV1) -> Self {
        match value {
            ResolvedArrayScalarV1::U64 => Self::U64,
            ResolvedArrayScalarV1::F64 => Self::F64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAcceptedArtifact {
    ordinal: u32,
    role: WireAcceptedRole,
    artifact_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireAcceptedRole {
    Mesh,
    Field,
}

#[cfg(test)]
mod tests {
    use eqiora_meshing::{
        DiscreteFieldAssociation, DiscreteFieldPayload, DiscreteFieldShape, MeshQualityGate,
        SimplicialMesh,
    };
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::{
        ExternalImportSourceV1, ExternalRuntimeRoleV1, ResolvedArrayV1, ResolvedImportArrayV1,
    };

    const MANIFEST_GOLDEN_DIGEST: &str =
        "1ab2c05efc0ad12b9ed285539c10fe6ae3b17a9bedb49d1db7f77eb258b271d8";
    // This complete literal deliberately shares no serialization helpers with
    // the writer. Any field, tag, value, or RFC-mandated order drift fails
    // before the separate domain-hash oracle below runs.
    const MANIFEST_GOLDEN_JSON: &[u8] = br#"{"schema":"eqiora.external-import-manifest/v1","encoding":"eqiora.canonical-json/v1","adapter":{"id":"eqiora.test-import","version":"1.0.0"},"runtime_stack":[{"role":"rust-binding","implementation":"test-binding","version":"0.4.0"}],"selection":{"grid":{"selector":{"element_path":[0]},"display_name":"grid-alpha"},"attributes":[{"selector":{"element_path":[0,2]},"display_name":"temperature-alpha"}]},"sources":[{"ordinal":0,"role":"metadata-document","origin_selector":{"element_path":[]},"display_locator":"alpha.xdmf","source_sha256":"3f0e8f43b363d6d876d891329b646c47769d5d388d471baa2c5bc4b47dc20c31"},{"ordinal":1,"role":"external-array-source","origin_selector":{"element_path":[0,0]},"display_locator":"alpha.h5","source_sha256":"7c5f241d6e450299ff87419bbf02ff33c40f7a5cf34b2db715b107687d650353"},{"ordinal":2,"role":"external-array-source","origin_selector":{"element_path":[0,1]},"display_locator":"alpha.h5","source_sha256":"1fdbca61e6f94b71eb6d75710445c14cb59e9f5e8150a0082f1c18359df1d3ec"},{"ordinal":3,"role":"external-array-source","origin_selector":{"element_path":[0,2]},"display_locator":"alpha.h5","source_sha256":"785c4f7bdb5f5858ece21e76dd8a63d786927064b4e40242910e3f7f82ac018e"}],"resolved_arrays":[{"ordinal":0,"role":"mesh-geometry","source_ordinal":1,"origin_selector":{"element_path":[0,0]},"storage_display_selector":"/alpha/geometry","scalar":"f64","shape":[4,2],"resolved_sha256":"71208c4e9e6d84af387ae35e4fd3ccf5eba44b758a926f7f53d5ef564e993b45"},{"ordinal":1,"role":"mesh-topology","source_ordinal":2,"origin_selector":{"element_path":[0,1]},"storage_display_selector":"/alpha/topology","scalar":"u64","shape":[2,3],"resolved_sha256":"7eb50929210246d8d2a7e2eb9770d01455d1dff45c7e398e2ddbdb3c583b1592"},{"ordinal":2,"role":"field","source_ordinal":3,"origin_selector":{"element_path":[0,2]},"storage_display_selector":"/alpha/temperature","scalar":"f64","shape":[4],"resolved_sha256":"636883a7cd7a4a121d0e49a96f29594e17003259bef08a00f6a8138d27c96e38"}],"accepted_artifacts":[{"ordinal":0,"role":"mesh","artifact_sha256":"0ac4d9506d6f36c45cf9da67e1ca61825e2acca59d23ac3cba4bba32c6ed8e2a"},{"ordinal":1,"role":"field","artifact_sha256":"3ecf87012feb5e85ea5bfc21a85feb602edb208a6f0eedf571e9374b99522979"}]}"#;

    struct Fixture {
        adapter: ExternalAdapterIdentityV1,
        runtime_stack: Vec<ExternalRuntimeComponentV1>,
        selection: ExternalImportSelectionV1,
        observation: ExternalImportObservationV1,
        mesh: SimplicialMeshEnvelopeV1,
        fields: Vec<DiscreteFieldEnvelopeV1>,
    }

    impl Fixture {
        fn manifest(&self) -> ExternalImportManifestV1 {
            ExternalImportManifestV1::from_observation(
                self.adapter.clone(),
                self.runtime_stack.clone(),
                self.selection.clone(),
                &self.observation,
                &self.mesh,
                &self.fields,
            )
            .unwrap()
        }
    }

    fn selector(path: &[u32]) -> StructuralSelectorV1 {
        StructuralSelectorV1::new(path.to_vec())
    }

    fn resolved(
        source: u32,
        path: &[u32],
        storage: Option<String>,
        array: ResolvedArrayV1,
    ) -> ResolvedImportArrayV1 {
        ResolvedImportArrayV1::new(source, selector(path), storage, array).unwrap()
    }

    fn fixture(label: &str) -> Fixture {
        let mesh = SimplicialMeshEnvelopeV1::from_mesh(
            &SimplicialMesh::new(
                2,
                vec![
                    vec![0.0, 0.0],
                    vec![1.0, 0.0],
                    vec![1.0, 1.0],
                    vec![0.0, 1.0],
                ],
                vec![vec![0, 1, 2], vec![0, 2, 3]],
                MeshQualityGate::new(0.2).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let payload = DiscreteFieldPayload::new(
            mesh.mesh(),
            DiscreteFieldAssociation::Vertex,
            DiscreteFieldShape::Scalar,
            vec![10.0, 20.0, 30.0, 40.0],
        )
        .unwrap();
        let field = DiscreteFieldEnvelopeV1::from_payload(&mesh, &payload).unwrap();
        let metadata = ExternalImportSourceV1::metadata_document(
            format!("<metadata label='{label}'/>").into_bytes(),
            Some(format!("{label}.xdmf")),
        )
        .unwrap();
        let external_sources = vec![
            ExternalImportSourceV1::external_array_source(
                selector(&[0, 0]),
                format!("geometry-source-{label}").into_bytes(),
                Some(format!("{label}.h5")),
            )
            .unwrap(),
            ExternalImportSourceV1::external_array_source(
                selector(&[0, 1]),
                format!("topology-source-{label}").into_bytes(),
                Some(format!("{label}.h5")),
            )
            .unwrap(),
            ExternalImportSourceV1::external_array_source(
                selector(&[0, 2]),
                format!("field-source-{label}").into_bytes(),
                Some(format!("{label}.h5")),
            )
            .unwrap(),
        ];
        let observation = ExternalImportObservationV1::new(
            metadata,
            external_sources,
            resolved(
                1,
                &[0, 0],
                Some(format!("/{label}/geometry")),
                ResolvedArrayV1::from_f64(vec![4, 2], vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
                    .unwrap(),
            ),
            resolved(
                2,
                &[0, 1],
                Some(format!("/{label}/topology")),
                ResolvedArrayV1::from_u64(vec![2, 3], vec![0, 1, 2, 0, 2, 3]).unwrap(),
            ),
            vec![resolved(
                3,
                &[0, 2],
                Some(format!("/{label}/temperature")),
                ResolvedArrayV1::from_f64(vec![4], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
            )],
        )
        .unwrap();
        let grid =
            SelectedSourceEntityV1::new(selector(&[0]), Some(format!("grid-{label}"))).unwrap();
        let attribute =
            SelectedSourceEntityV1::new(selector(&[0, 2]), Some(format!("temperature-{label}")))
                .unwrap();
        Fixture {
            adapter: ExternalAdapterIdentityV1::new("eqiora.test-import", "1.0.0").unwrap(),
            runtime_stack: vec![
                ExternalRuntimeComponentV1::new(
                    ExternalRuntimeRoleV1::RustBinding,
                    "test-binding",
                    "0.4.0",
                )
                .unwrap(),
            ],
            selection: ExternalImportSelectionV1::new(grid, vec![attribute]).unwrap(),
            observation,
            mesh,
            fields: vec![field],
        }
    }

    fn encode_wire(wire: &WireExternalImportManifestV1) -> Vec<u8> {
        serde_json::to_vec(wire).unwrap()
    }

    #[test]
    fn canonical_round_trip_digest_and_independent_references_are_exact() {
        let fixture = fixture("alpha");
        let manifest = fixture.manifest();
        let bytes = manifest.canonical_json().unwrap();
        assert_eq!(bytes, MANIFEST_GOLDEN_JSON);
        assert_eq!(manifest.digest().unwrap().as_str(), MANIFEST_GOLDEN_DIGEST);
        let decoded =
            ExternalImportManifestV1::from_json(&bytes, ExternalImportDecoderLimits::default())
                .unwrap();
        assert_eq!(decoded, manifest);
        assert_eq!(decoded.canonical_json().unwrap(), bytes);
        decoded
            .validate_references(&fixture.observation, &fixture.mesh, &fixture.fields)
            .unwrap();
        assert_eq!(
            decoded.accepted_mesh_artifact(),
            fixture.mesh.digest().unwrap()
        );
        assert_eq!(
            decoded.accepted_field_artifacts(),
            vec![fixture.fields[0].digest().unwrap()]
        );

        let mut oracle = Sha256::new();
        oracle.update(EXTERNAL_IMPORT_SCHEMA.as_bytes());
        oracle.update([0]);
        oracle.update(&bytes);
        assert_eq!(
            manifest.digest().unwrap().sha256_bytes(),
            <[u8; 32]>::from(oracle.finalize())
        );
    }

    #[test]
    fn display_and_source_provenance_change_only_manifest_identity() {
        let alpha = fixture("alpha");
        let beta = fixture("beta");
        let alpha_manifest = alpha.manifest();
        let beta_manifest = beta.manifest();

        assert_eq!(alpha.mesh.digest().unwrap(), beta.mesh.digest().unwrap());
        assert_eq!(
            alpha.fields[0].digest().unwrap(),
            beta.fields[0].digest().unwrap()
        );
        assert_ne!(
            alpha_manifest.canonical_json().unwrap(),
            beta_manifest.canonical_json().unwrap()
        );
        assert_ne!(
            alpha_manifest.digest().unwrap(),
            beta_manifest.digest().unwrap()
        );
    }

    #[test]
    fn independent_linkage_is_deliberately_not_derivation_proof() {
        let fixture = fixture("cross-wired");
        let manifest = fixture.manifest();

        // These arbitrary source bytes do not encode the normalized geometry.
        // The manifest truthfully records all three identities but cannot issue
        // a verified replay handle by itself.
        assert_ne!(
            fixture.observation.external_sources()[0].bytes(),
            fixture
                .observation
                .mesh_geometry()
                .array()
                .canonical_json()
                .unwrap()
        );
        manifest
            .validate_references(&fixture.observation, &fixture.mesh, &fixture.fields)
            .unwrap();
    }

    #[test]
    fn any_independent_reference_mutation_fails_closed() {
        let alpha = fixture("alpha");
        let manifest = alpha.manifest();

        let changed_source = ExternalImportObservationV1::new(
            ExternalImportSourceV1::metadata_document(
                b"different metadata bytes".to_vec(),
                alpha
                    .observation
                    .metadata()
                    .display_locator()
                    .map(str::to_owned),
            )
            .unwrap(),
            alpha.observation.external_sources().to_vec(),
            alpha.observation.mesh_geometry().clone(),
            alpha.observation.mesh_topology().clone(),
            alpha.observation.fields().to_vec(),
        )
        .unwrap();
        assert!(
            manifest
                .validate_references(&changed_source, &alpha.mesh, &alpha.fields)
                .is_err()
        );

        let mut selector_sources = alpha.observation.external_sources().to_vec();
        selector_sources[0] = ExternalImportSourceV1::external_array_source(
            selector(&[9]),
            selector_sources[0].bytes().to_vec(),
            selector_sources[0].display_locator().map(str::to_owned),
        )
        .unwrap();
        let selector_geometry = resolved(
            alpha.observation.mesh_geometry().source_ordinal(),
            &[9],
            alpha
                .observation
                .mesh_geometry()
                .storage_display_selector()
                .map(str::to_owned),
            alpha.observation.mesh_geometry().array().clone(),
        );
        let changed_selector = ExternalImportObservationV1::new(
            alpha.observation.metadata().clone(),
            selector_sources,
            selector_geometry,
            alpha.observation.mesh_topology().clone(),
            alpha.observation.fields().to_vec(),
        )
        .unwrap();
        assert!(
            manifest
                .validate_references(&changed_selector, &alpha.mesh, &alpha.fields)
                .is_err()
        );

        let changed_array = ExternalImportObservationV1::new(
            alpha.observation.metadata().clone(),
            alpha.observation.external_sources().to_vec(),
            resolved(
                alpha.observation.mesh_geometry().source_ordinal(),
                alpha
                    .observation
                    .mesh_geometry()
                    .origin_selector()
                    .element_path(),
                alpha
                    .observation
                    .mesh_geometry()
                    .storage_display_selector()
                    .map(str::to_owned),
                ResolvedArrayV1::from_f64(vec![4, 2], vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.5])
                    .unwrap(),
            ),
            alpha.observation.mesh_topology().clone(),
            alpha.observation.fields().to_vec(),
        )
        .unwrap();
        assert!(
            manifest
                .validate_references(&changed_array, &alpha.mesh, &alpha.fields)
                .is_err()
        );

        let changed_payload = DiscreteFieldPayload::new(
            alpha.mesh.mesh(),
            DiscreteFieldAssociation::Vertex,
            DiscreteFieldShape::Scalar,
            vec![10.0, 20.0, 30.0, 41.0],
        )
        .unwrap();
        let changed_field =
            DiscreteFieldEnvelopeV1::from_payload(&alpha.mesh, &changed_payload).unwrap();
        assert!(
            manifest
                .validate_references(&alpha.observation, &alpha.mesh, &[changed_field])
                .is_err()
        );

        let different_mesh = SimplicialMeshEnvelopeV1::from_mesh(
            &SimplicialMesh::new(
                2,
                vec![
                    vec![0.0, 0.0],
                    vec![2.0, 0.0],
                    vec![2.0, 2.0],
                    vec![0.0, 2.0],
                ],
                vec![vec![0, 1, 2], vec![0, 2, 3]],
                MeshQualityGate::new(0.2).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(
            ExternalImportManifestV1::from_observation(
                alpha.adapter.clone(),
                alpha.runtime_stack.clone(),
                alpha.selection.clone(),
                &alpha.observation,
                &different_mesh,
                &alpha.fields,
            )
            .is_err()
        );
    }

    #[test]
    fn runtime_selection_source_order_and_closed_wire_are_enforced() {
        let fixture = fixture("alpha");
        let manifest = fixture.manifest();

        let duplicate = vec![
            ExternalRuntimeComponentV1::new(ExternalRuntimeRoleV1::RustBinding, "same", "1")
                .unwrap(),
            ExternalRuntimeComponentV1::new(ExternalRuntimeRoleV1::RustBinding, "same", "2")
                .unwrap(),
        ];
        assert!(
            ExternalImportManifestV1::from_observation(
                fixture.adapter.clone(),
                duplicate,
                fixture.selection.clone(),
                &fixture.observation,
                &fixture.mesh,
                &fixture.fields,
            )
            .is_err()
        );

        let mut noncontiguous = manifest.wire.clone();
        noncontiguous.sources[1].ordinal = 9;
        assert!(
            ExternalImportManifestV1::from_json(
                &encode_wire(&noncontiguous),
                ExternalImportDecoderLimits::default()
            )
            .is_err()
        );

        let mut dangling = manifest.wire.clone();
        dangling.resolved_arrays[0].source_ordinal = 99;
        assert!(
            ExternalImportManifestV1::from_json(
                &encode_wire(&dangling),
                ExternalImportDecoderLimits::default()
            )
            .is_err()
        );

        let mut mismatched_origin = manifest.wire.clone();
        mismatched_origin.sources[1].origin_selector.element_path = vec![9];
        assert!(
            ExternalImportManifestV1::from_json(
                &encode_wire(&mismatched_origin),
                ExternalImportDecoderLimits::default()
            )
            .is_err()
        );

        let mut unknown: serde_json::Value =
            serde_json::from_slice(&manifest.canonical_json().unwrap()).unwrap();
        unknown
            .as_object_mut()
            .unwrap()
            .insert("extension".to_owned(), serde_json::Value::Bool(true));
        assert!(
            ExternalImportManifestV1::from_json(
                &serde_json::to_vec(&unknown).unwrap(),
                ExternalImportDecoderLimits::default()
            )
            .is_err()
        );
    }

    #[test]
    fn manifest_specific_decoder_limits_are_independent_and_fail_closed() {
        let manifest = fixture("alpha").manifest();
        let bytes = manifest.canonical_json().unwrap();
        for limits in [
            ExternalImportDecoderLimits {
                max_import_manifest_text_bytes: 1,
                ..ExternalImportDecoderLimits::default()
            },
            ExternalImportDecoderLimits {
                max_import_runtime_entries: 0,
                ..ExternalImportDecoderLimits::default()
            },
            ExternalImportDecoderLimits {
                max_import_selection_attributes: 0,
                ..ExternalImportDecoderLimits::default()
            },
            ExternalImportDecoderLimits {
                max_import_sources: 3,
                ..ExternalImportDecoderLimits::default()
            },
            ExternalImportDecoderLimits {
                max_import_resolved_arrays: 2,
                ..ExternalImportDecoderLimits::default()
            },
            ExternalImportDecoderLimits {
                max_import_accepted_artifacts: 1,
                ..ExternalImportDecoderLimits::default()
            },
            ExternalImportDecoderLimits {
                resolved_array: crate::ResolvedArrayLimits {
                    max_rank: 1,
                    ..crate::ResolvedArrayLimits::default()
                },
                ..ExternalImportDecoderLimits::default()
            },
            ExternalImportDecoderLimits {
                resolved_array: crate::ResolvedArrayLimits {
                    max_values: 7,
                    ..crate::ResolvedArrayLimits::default()
                },
                ..ExternalImportDecoderLimits::default()
            },
        ] {
            assert!(ExternalImportManifestV1::from_json(&bytes, limits).is_err());
        }
    }

    #[test]
    fn equal_field_artifacts_remain_distinct_ordered_occurrences() {
        let mut fixture = fixture("equal-fields");
        let second_attribute =
            SelectedSourceEntityV1::new(selector(&[0, 3]), Some("copy".to_owned())).unwrap();
        fixture.selection = ExternalImportSelectionV1::new(
            fixture.selection.grid().clone(),
            vec![fixture.selection.attributes()[0].clone(), second_attribute],
        )
        .unwrap();
        let second_source = ExternalImportSourceV1::external_array_source(
            selector(&[0, 3]),
            b"second-field-occurrence".to_vec(),
            Some("same.h5".to_owned()),
        )
        .unwrap();
        let second_resolved = resolved(
            4,
            &[0, 3],
            Some("/copy".to_owned()),
            ResolvedArrayV1::from_f64(vec![4], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
        );
        fixture.observation = ExternalImportObservationV1::new(
            fixture.observation.metadata().clone(),
            fixture
                .observation
                .external_sources()
                .iter()
                .cloned()
                .chain(std::iter::once(second_source))
                .collect(),
            fixture.observation.mesh_geometry().clone(),
            fixture.observation.mesh_topology().clone(),
            vec![fixture.observation.fields()[0].clone(), second_resolved],
        )
        .unwrap();
        fixture.fields.push(fixture.fields[0].clone());
        let manifest = fixture.manifest();
        let accepted = manifest.accepted_field_artifacts();
        assert_eq!(accepted.len(), 2);
        assert_eq!(accepted[0], accepted[1]);
    }
}
