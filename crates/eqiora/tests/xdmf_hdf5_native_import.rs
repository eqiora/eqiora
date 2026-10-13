use eqiora::api::{import_xdmf_hdf5_v1, import_xdmf_v1, verify_xdmf_hdf5_import_v1};
use eqiora::artifact::{
    DiscreteFieldEnvelopeV1, ExternalImportManifestV1, ExternalRuntimeRoleV1,
    SimplicialMeshEnvelopeV1,
};
use eqiora::diagnostic::codes;
use eqiora::io::hdf5::{
    HDF5_BINDING_ID, HDF5_BINDING_VERSION, HDF5_NATIVE_LIBRARY_ID, Hdf5DatasetRequest,
    Hdf5FileImage, Hdf5ResolveLimits, Hdf5ScalarType, resolve_hdf5_file_image,
};
use eqiora::io::xml::{
    XdmfArrayResponse, XdmfArrayValues, XdmfImportLimits, XdmfImportPlan, XdmfSelection,
};
use eqiora::meshing::MeshQualityGate;

const METADATA: &[u8] =
    include_bytes!("../../../verify/artifacts/xdmf-hdf5-native-import/fixtures/unit-square.xdmf");
const HDF5: &[u8] =
    include_bytes!("../../../verify/artifacts/xdmf-hdf5-native-import/fixtures/unit-square.h5");

const HOSTILE_FILE_IMAGES: &[(&str, &[u8])] = &[
    (
        "attribute",
        include_bytes!(
            "../../../verify/artifacts/xdmf-hdf5-native-import/fixtures/reject-attribute.h5"
        ),
    ),
    (
        "compound datatype",
        include_bytes!(
            "../../../verify/artifacts/xdmf-hdf5-native-import/fixtures/reject-compound-datatype.h5"
        ),
    ),
    (
        "external link",
        include_bytes!(
            "../../../verify/artifacts/xdmf-hdf5-native-import/fixtures/reject-external-link.h5"
        ),
    ),
    (
        "external raw storage",
        include_bytes!(
            "../../../verify/artifacts/xdmf-hdf5-native-import/fixtures/reject-external-storage.h5"
        ),
    ),
    (
        "filter pipeline",
        include_bytes!(
            "../../../verify/artifacts/xdmf-hdf5-native-import/fixtures/reject-filter.h5"
        ),
    ),
    (
        "hard-link alias",
        include_bytes!(
            "../../../verify/artifacts/xdmf-hdf5-native-import/fixtures/reject-hard-link-alias.h5"
        ),
    ),
    (
        "hard-link cycle",
        include_bytes!(
            "../../../verify/artifacts/xdmf-hdf5-native-import/fixtures/reject-hard-link-cycle.h5"
        ),
    ),
    (
        "soft link",
        include_bytes!(
            "../../../verify/artifacts/xdmf-hdf5-native-import/fixtures/reject-soft-link.h5"
        ),
    ),
    (
        "unlinked committed datatype",
        include_bytes!(
            "../../../verify/artifacts/xdmf-hdf5-native-import/fixtures/reject-unlinked-committed-datatype.h5"
        ),
    ),
    (
        "virtual dataset",
        include_bytes!(
            "../../../verify/artifacts/xdmf-hdf5-native-import/fixtures/reject-virtual-dataset.h5"
        ),
    ),
];

fn plan() -> XdmfImportPlan {
    XdmfImportPlan::parse(
        METADATA,
        XdmfSelection::new(vec![0, 0], vec![vec![0, 0, 2], vec![0, 0, 3]]).unwrap(),
        XdmfImportLimits::default(),
    )
    .unwrap()
}

fn hdf5_requests() -> Vec<Hdf5DatasetRequest> {
    vec![
        Hdf5DatasetRequest::new("/mesh/geometry", Hdf5ScalarType::F64, vec![4, 2]).unwrap(),
        Hdf5DatasetRequest::new("/mesh/topology", Hdf5ScalarType::U64, vec![2, 3]).unwrap(),
        Hdf5DatasetRequest::new("/fields/temperature", Hdf5ScalarType::F64, vec![4]).unwrap(),
        Hdf5DatasetRequest::new("/fields/flux", Hdf5ScalarType::F64, vec![2, 2]).unwrap(),
    ]
}

fn caller_responses(plan: &XdmfImportPlan) -> Vec<XdmfArrayResponse> {
    plan.requests()
        .iter()
        .map(|request| {
            let values = match request.dataset_path() {
                "/mesh/geometry" => {
                    XdmfArrayValues::F64(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
                }
                "/mesh/topology" => XdmfArrayValues::U64(vec![0, 1, 2, 0, 2, 3]),
                "/fields/temperature" => XdmfArrayValues::F64(vec![10.0, 20.0, 30.0, 40.0]),
                "/fields/flux" => XdmfArrayValues::F64(vec![1.0, 0.0, 0.0, 1.0]),
                path => panic!("unexpected fixture dataset {path}"),
            };
            XdmfArrayResponse::new(request, HDF5.to_vec(), values)
        })
        .collect()
}

#[test]
fn native_file_image_resolution_replays_exact_persisted_artifacts() {
    let plan = plan();
    let quality = MeshQualityGate::new(0.5).unwrap();
    let derived = import_xdmf_hdf5_v1(
        &plan,
        Hdf5FileImage::new(HDF5),
        Hdf5ResolveLimits::default(),
        quality,
    )
    .unwrap();
    let persisted_manifest = ExternalImportManifestV1::from_json(
        &derived.manifest().canonical_json().unwrap(),
        Default::default(),
    )
    .unwrap();
    let persisted_mesh = SimplicialMeshEnvelopeV1::from_json(
        &derived.mesh().canonical_json().unwrap(),
        Default::default(),
    )
    .unwrap();
    let persisted_fields = derived
        .fields()
        .iter()
        .map(|field| {
            DiscreteFieldEnvelopeV1::from_json(&field.canonical_json().unwrap(), Default::default())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let verified = verify_xdmf_hdf5_import_v1(
        &persisted_manifest,
        &persisted_mesh,
        &persisted_fields,
        &plan,
        Hdf5FileImage::new(HDF5),
        Hdf5ResolveLimits::default(),
        quality,
    )
    .unwrap();

    assert_eq!(verified.mesh(), derived.mesh());
    assert_eq!(verified.fields(), derived.fields());
    assert_eq!(verified.manifest(), derived.manifest());
    assert_eq!(verified.manifest_digest(), derived.manifest_digest());
    assert_eq!(
        verified.manifest().adapter().id(),
        "eqiora.xdmf-hdf5.file-image"
    );
    assert_eq!(
        verified.manifest().adapter().version(),
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(verified.manifest().runtime_stack().len(), 2);
    assert_eq!(
        verified.manifest().runtime_stack()[0].role(),
        ExternalRuntimeRoleV1::RustBinding
    );
    assert_eq!(
        verified.manifest().runtime_stack()[0].implementation(),
        HDF5_BINDING_ID
    );
    assert_eq!(
        verified.manifest().runtime_stack()[0].version(),
        HDF5_BINDING_VERSION
    );
    assert_eq!(
        verified.manifest().runtime_stack()[1].role(),
        ExternalRuntimeRoleV1::NativeStorageLibrary
    );
    assert_eq!(
        verified.manifest().runtime_stack()[1].implementation(),
        HDF5_NATIVE_LIBRARY_ID
    );
    assert_eq!(verified.manifest().runtime_stack()[1].version(), "2.1.0");

    let caller_resolved = import_xdmf_v1(&plan, &caller_responses(&plan), quality).unwrap();
    assert_eq!(caller_resolved.mesh(), derived.mesh());
    assert_eq!(caller_resolved.fields(), derived.fields());
    assert_ne!(caller_resolved.manifest(), derived.manifest());
    assert_ne!(caller_resolved.manifest_digest(), derived.manifest_digest());
    assert_eq!(
        verify_xdmf_hdf5_import_v1(
            caller_resolved.manifest(),
            caller_resolved.mesh(),
            caller_resolved.fields(),
            &plan,
            Hdf5FileImage::new(HDF5),
            Hdf5ResolveLimits::default(),
            quality,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT,
    );
}

#[test]
fn native_source_and_plan_changes_fail_before_verified_lineage() {
    let plan = plan();
    let quality = MeshQualityGate::new(0.5).unwrap();
    let baseline = import_xdmf_hdf5_v1(
        &plan,
        Hdf5FileImage::new(HDF5),
        Hdf5ResolveLimits::default(),
        quality,
    )
    .unwrap();

    let mut same_values_different_source = HDF5.to_vec();
    same_values_different_source.push(0);
    let rebound = import_xdmf_hdf5_v1(
        &plan,
        Hdf5FileImage::new(&same_values_different_source),
        Hdf5ResolveLimits::default(),
        quality,
    )
    .unwrap();
    assert_eq!(rebound.mesh(), baseline.mesh());
    assert_eq!(rebound.fields(), baseline.fields());
    assert_ne!(rebound.manifest(), baseline.manifest());
    assert_ne!(rebound.manifest_digest(), baseline.manifest_digest());
    assert_eq!(
        verify_xdmf_hdf5_import_v1(
            baseline.manifest(),
            baseline.mesh(),
            baseline.fields(),
            &plan,
            Hdf5FileImage::new(&same_values_different_source),
            Hdf5ResolveLimits::default(),
            quality,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_ARTIFACT,
    );

    let limits = Hdf5ResolveLimits {
        max_source_bytes: HDF5.len() - 1,
        ..Hdf5ResolveLimits::default()
    };
    assert_eq!(
        import_xdmf_hdf5_v1(&plan, Hdf5FileImage::new(HDF5), limits, quality)
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT,
    );

    let work_limits = Hdf5ResolveLimits {
        max_audit_work: 1,
        ..Hdf5ResolveLimits::default()
    };
    assert_eq!(
        import_xdmf_hdf5_v1(&plan, Hdf5FileImage::new(HDF5), work_limits, quality,)
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT,
    );
}

#[test]
fn immutable_native_batch_is_fully_preflighted_before_values_are_returned() {
    let mut wrong_shape = hdf5_requests();
    wrong_shape[3] = Hdf5DatasetRequest::new("/fields/flux", Hdf5ScalarType::F64, vec![4]).unwrap();
    assert_eq!(
        resolve_hdf5_file_image(
            Hdf5FileImage::new(HDF5),
            &wrong_shape,
            Hdf5ResolveLimits::default(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT,
    );

    let mut wrong_scalar = hdf5_requests();
    wrong_scalar[0] =
        Hdf5DatasetRequest::new("/mesh/geometry", Hdf5ScalarType::U64, vec![4, 2]).unwrap();
    assert_eq!(
        resolve_hdf5_file_image(
            Hdf5FileImage::new(HDF5),
            &wrong_scalar,
            Hdf5ResolveLimits::default(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT,
    );
}

#[test]
fn forbidden_hdf5_constructs_fail_before_artifact_derivation() {
    let plan = plan();
    let quality = MeshQualityGate::new(0.5).unwrap();

    for (construct, source) in HOSTILE_FILE_IMAGES {
        let error = import_xdmf_hdf5_v1(
            &plan,
            Hdf5FileImage::new(source),
            Hdf5ResolveLimits::default(),
            quality,
        )
        .unwrap_err();
        assert_eq!(
            error.code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT,
            "hostile HDF5 construct was not rejected: {construct}",
        );
    }
}
