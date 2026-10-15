//! Native file-image resolution for the XDMF composition.

use eqiora_artifact::{
    DiscreteFieldEnvelopeV1, ExternalAdapterIdentityV1, ExternalImportManifestV1,
    ExternalRuntimeComponentV1, ExternalRuntimeRoleV1, SimplicialMeshEnvelopeV1,
};
use eqiora_core::Diagnostic;
use eqiora_io_hdf5::{
    Hdf5DatasetRequest, Hdf5FileImage, Hdf5ResolveLimits, Hdf5ResolvedValues, Hdf5RuntimeIdentity,
    Hdf5ScalarType, resolve_hdf5_file_image,
};
use eqiora_io_xml::{XdmfArrayResponse, XdmfArrayValues, XdmfImportPlan, XdmfScalarType};
use eqiora_meshing::MeshQualityGate;

use super::{
    VerifiedXdmfImportV1, XdmfImportArtifactsV1, derive_candidate_with_context,
    invalid_external_import,
};

const XDMF_HDF5_ADAPTER_ID: &str = "eqiora.xdmf-hdf5.file-image";
const XDMF_HDF5_ADAPTER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Derive XDMF mesh, field, and lineage artifacts through the native HDF5
/// file-image resolver.
///
/// The first native slice intentionally binds every HDF `DataItem` to one
/// complete source image and requires one common display locator. The locator
/// is checked for plan coherence but is never opened. HDF5 audits and
/// preflights the whole request batch before its first value read.
///
/// # Errors
/// Returns `EQ0810` for a multi-source plan, HDF5 rejection, resolver mismatch,
/// resource-limit excess, or a downstream mesh/field/artifact invariant.
pub fn import_xdmf_hdf5_v1(
    plan: &XdmfImportPlan,
    source: Hdf5FileImage<'_>,
    hdf5_limits: Hdf5ResolveLimits,
    quality_gate: MeshQualityGate,
) -> Result<XdmfImportArtifactsV1, Diagnostic> {
    let (responses, runtime_stack) = native_responses(plan, source, hdf5_limits)?;
    let candidate = derive_candidate_with_context(
        plan,
        &responses,
        quality_gate,
        native_adapter_identity()?,
        runtime_stack,
    )?;
    Ok(XdmfImportArtifactsV1 {
        manifest: candidate.manifest,
        manifest_digest: candidate.manifest_digest,
        mesh: candidate.mesh,
        fields: candidate.fields,
    })
}

/// Replay exact persisted XDMF/HDF5 artifacts through the current native
/// binding and issue an opaque verified-lineage handle.
///
/// The expected artifacts are caller-owned and independently loaded. Runtime
/// binding and native-library releases are part of exact manifest identity.
///
/// # Errors
/// Returns `EQ0810` when native resolution or the exact persisted replay
/// differs. Shared artifact diagnostics retain their own codes.
pub fn verify_xdmf_hdf5_import_v1(
    expected_manifest: &ExternalImportManifestV1,
    expected_mesh: &SimplicialMeshEnvelopeV1,
    expected_fields: &[DiscreteFieldEnvelopeV1],
    plan: &XdmfImportPlan,
    source: Hdf5FileImage<'_>,
    hdf5_limits: Hdf5ResolveLimits,
    quality_gate: MeshQualityGate,
) -> Result<VerifiedXdmfImportV1, Diagnostic> {
    let (responses, runtime_stack) = native_responses(plan, source, hdf5_limits)?;
    let candidate = derive_candidate_with_context(
        plan,
        &responses,
        quality_gate,
        native_adapter_identity()?,
        runtime_stack,
    )?;
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
            "persisted XDMF/HDF5 manifest or accepted artifacts differ from fresh native replay",
        ));
    }

    Ok(VerifiedXdmfImportV1 {
        manifest: candidate.manifest,
        manifest_digest: candidate.manifest_digest,
        mesh: candidate.mesh,
        fields: candidate.fields,
    })
}

fn native_responses(
    plan: &XdmfImportPlan,
    source: Hdf5FileImage<'_>,
    hdf5_limits: Hdf5ResolveLimits,
) -> Result<(Vec<XdmfArrayResponse>, Vec<ExternalRuntimeComponentV1>), Diagnostic> {
    let requests = plan.requests();
    let first_locator = requests
        .first()
        .ok_or_else(|| invalid_external_import("XDMF native HDF5 plan has no array requests"))?
        .source_locator();
    if requests
        .iter()
        .any(|request| request.source_locator() != first_locator)
    {
        return Err(invalid_external_import(
            "XDMF native HDF5 v1 requires one common source locator and one complete file image",
        ));
    }
    preflight_source_occurrences(plan, source.bytes().len())?;

    let mut native_requests = Vec::new();
    native_requests
        .try_reserve(requests.len())
        .map_err(|error| {
            invalid_external_import(format!(
                "cannot reserve bounded XDMF/HDF5 request translation: {error}",
            ))
        })?;
    for request in requests {
        native_requests.push(Hdf5DatasetRequest::new(
            request.dataset_path(),
            match request.scalar() {
                XdmfScalarType::U64 => Hdf5ScalarType::U64,
                XdmfScalarType::F64 => Hdf5ScalarType::F64,
            },
            copy_shape(request.shape())?,
        )?);
    }
    let resolution = resolve_hdf5_file_image(source, &native_requests, hdf5_limits)?;
    if resolution.values().len() != requests.len() {
        return Err(invalid_external_import(
            "native HDF5 response count differs from the immutable XDMF plan",
        ));
    }

    let mut responses = Vec::new();
    responses.try_reserve(requests.len()).map_err(|error| {
        invalid_external_import(format!(
            "cannot reserve bounded XDMF/HDF5 response translation: {error}",
        ))
    })?;
    let runtime_stack = runtime_stack(resolution.runtime())?;
    for (request, values) in requests.iter().zip(resolution.into_values()) {
        responses.push(XdmfArrayResponse::new(
            request,
            copy_source(source.bytes())?,
            match values {
                Hdf5ResolvedValues::U64(values) => XdmfArrayValues::U64(values),
                Hdf5ResolvedValues::F64(values) => XdmfArrayValues::F64(values),
            },
        ));
    }
    Ok((responses, runtime_stack))
}

fn preflight_source_occurrences(
    plan: &XdmfImportPlan,
    source_bytes: usize,
) -> Result<(), Diagnostic> {
    let limits = plan.limits();
    if source_bytes > limits.max_source_bytes {
        return Err(invalid_external_import(
            "native HDF5 source exceeds the XDMF per-occurrence source-byte limit",
        ));
    }
    let total = source_bytes
        .checked_mul(plan.requests().len())
        .ok_or_else(|| {
            invalid_external_import("aggregate XDMF/HDF5 source bytes overflow usize")
        })?;
    if total > limits.max_total_source_bytes {
        return Err(invalid_external_import(
            "native HDF5 source occurrences exceed the XDMF aggregate source-byte limit",
        ));
    }
    Ok(())
}

fn copy_source(source: &[u8]) -> Result<Vec<u8>, Diagnostic> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(source.len()).map_err(|error| {
        invalid_external_import(format!(
            "cannot reserve complete HDF5 source occurrence: {error}",
        ))
    })?;
    copy.extend_from_slice(source);
    Ok(copy)
}

fn copy_shape(shape: &[u64]) -> Result<Vec<u64>, Diagnostic> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(shape.len()).map_err(|error| {
        invalid_external_import(format!(
            "cannot reserve bounded XDMF/HDF5 request shape: {error}",
        ))
    })?;
    copy.extend_from_slice(shape);
    Ok(copy)
}

fn runtime_stack(
    runtime: &Hdf5RuntimeIdentity,
) -> Result<Vec<ExternalRuntimeComponentV1>, Diagnostic> {
    Ok(vec![
        ExternalRuntimeComponentV1::new(
            ExternalRuntimeRoleV1::RustBinding,
            runtime.binding_id(),
            runtime.binding_version(),
        )?,
        ExternalRuntimeComponentV1::new(
            ExternalRuntimeRoleV1::NativeStorageLibrary,
            runtime.native_library_id(),
            runtime.native_library_version(),
        )?,
    ])
}

fn native_adapter_identity() -> Result<ExternalAdapterIdentityV1, Diagnostic> {
    ExternalAdapterIdentityV1::new(XDMF_HDF5_ADAPTER_ID, XDMF_HDF5_ADAPTER_VERSION)
}
