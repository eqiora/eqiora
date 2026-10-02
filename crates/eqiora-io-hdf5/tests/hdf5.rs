use eqiora_core::diagnostic::codes;
use eqiora_io_hdf5::{
    HDF5_BINDING_ID, HDF5_BINDING_VERSION, HDF5_NATIVE_LIBRARY_ID, Hdf5DatasetRequest,
    Hdf5DatasetWrite, Hdf5FileImage, Hdf5ResolveLimits, Hdf5ResolvedValues, Hdf5ScalarType,
    Hdf5WriteLimits, resolve_hdf5_file_image, write_hdf5_file_image,
};

const SOURCE: &[u8] =
    include_bytes!("../../../verify/artifacts/xdmf-uniform-grid-import/fixtures/unit-square.h5");

#[test]
fn runtime_binding_provenance_matches_the_resolved_workspace_dependency() {
    let manifest: toml::Value = toml::from_str(include_str!("../../../Cargo.toml")).unwrap();
    let lock: toml::Value = toml::from_str(include_str!("../../../Cargo.lock")).unwrap();
    let binding = &manifest["workspace"]["dependencies"][HDF5_BINDING_ID];
    let pinned = binding["version"]
        .as_str()
        .unwrap()
        .strip_prefix('=')
        .unwrap();
    let resolved: Vec<_> = lock["package"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|package| package["name"].as_str() == Some(HDF5_BINDING_ID))
        .collect();
    assert_eq!(
        resolved.len(),
        1,
        "runtime provenance needs one binding release"
    );
    assert_eq!(resolved[0]["version"].as_str(), Some(pinned));
    let runtime = resolve_hdf5_file_image(
        Hdf5FileImage::new(SOURCE),
        &requests(),
        Hdf5ResolveLimits::default(),
    )
    .unwrap();
    assert_eq!(runtime.runtime().binding_version(), pinned);
}

fn requests() -> Vec<Hdf5DatasetRequest> {
    vec![
        Hdf5DatasetRequest::new("/mesh/geometry", Hdf5ScalarType::F64, vec![4, 2]).unwrap(),
        Hdf5DatasetRequest::new("/mesh/topology", Hdf5ScalarType::U64, vec![2, 3]).unwrap(),
        Hdf5DatasetRequest::new("/fields/temperature", Hdf5ScalarType::F64, vec![4]).unwrap(),
        Hdf5DatasetRequest::new("/fields/flux", Hdf5ScalarType::F64, vec![2, 2]).unwrap(),
    ]
}

#[test]
fn resolves_one_audited_file_image_batch_in_request_order() {
    let resolved = resolve_hdf5_file_image(
        Hdf5FileImage::new(SOURCE),
        &requests(),
        Hdf5ResolveLimits::default(),
    )
    .unwrap();

    assert_eq!(resolved.runtime().binding_id(), HDF5_BINDING_ID);
    assert_eq!(resolved.runtime().binding_version(), HDF5_BINDING_VERSION);
    assert_eq!(
        resolved.runtime().native_library_id(),
        HDF5_NATIVE_LIBRARY_ID
    );
    assert_ne!(resolved.runtime().native_library_version(), "0.0.0");
    assert_eq!(
        resolved.values(),
        &[
            Hdf5ResolvedValues::F64(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
            Hdf5ResolvedValues::U64(vec![0, 1, 2, 0, 2, 3]),
            Hdf5ResolvedValues::F64(vec![10.0, 20.0, 30.0, 40.0]),
            Hdf5ResolvedValues::F64(vec![1.0, 0.0, 0.0, 1.0]),
        ]
    );
}

#[test]
fn preflight_rejects_any_request_mismatch_before_values_are_returned() {
    let mut wrong_shape = requests();
    wrong_shape[3] = Hdf5DatasetRequest::new("/fields/flux", Hdf5ScalarType::F64, vec![4]).unwrap();
    assert_eq!(
        resolve_hdf5_file_image(
            Hdf5FileImage::new(SOURCE),
            &wrong_shape,
            Hdf5ResolveLimits::default(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT,
    );

    let mut wrong_scalar = requests();
    wrong_scalar[0] =
        Hdf5DatasetRequest::new("/mesh/geometry", Hdf5ScalarType::U64, vec![4, 2]).unwrap();
    assert_eq!(
        resolve_hdf5_file_image(
            Hdf5FileImage::new(SOURCE),
            &wrong_scalar,
            Hdf5ResolveLimits::default(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT,
    );
}

#[test]
fn source_request_and_work_budgets_fail_closed() {
    let limits = Hdf5ResolveLimits {
        max_source_bytes: SOURCE.len() - 1,
        ..Hdf5ResolveLimits::default()
    };
    assert!(resolve_hdf5_file_image(Hdf5FileImage::new(SOURCE), &requests(), limits).is_err());

    let limits = Hdf5ResolveLimits {
        max_requests: 3,
        ..Hdf5ResolveLimits::default()
    };
    assert!(resolve_hdf5_file_image(Hdf5FileImage::new(SOURCE), &requests(), limits).is_err());

    let limits = Hdf5ResolveLimits {
        max_audit_work: 1,
        ..Hdf5ResolveLimits::default()
    };
    assert!(resolve_hdf5_file_image(Hdf5FileImage::new(SOURCE), &requests(), limits).is_err());
}

#[test]
fn malformed_images_and_noncanonical_paths_are_rejected() {
    assert!(Hdf5DatasetRequest::new("mesh/geometry", Hdf5ScalarType::F64, vec![4, 2]).is_err());
    assert!(Hdf5DatasetRequest::new("/mesh//geometry", Hdf5ScalarType::F64, vec![4, 2]).is_err());
    assert!(Hdf5DatasetRequest::new("/mesh/geometry", Hdf5ScalarType::F64, vec![]).is_err());
    assert!(
        resolve_hdf5_file_image(
            Hdf5FileImage::new(b"not-hdf5"),
            &requests(),
            Hdf5ResolveLimits::default(),
        )
        .is_err()
    );
}

#[test]
fn deterministic_writer_round_trips_through_the_complete_audit() {
    let geometry = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let topology = [0, 1, 2];
    let forward = vec![
        Hdf5DatasetWrite::f64("/mesh/geometry", vec![3, 2], &geometry).unwrap(),
        Hdf5DatasetWrite::u64("/mesh/topology", vec![1, 3], &topology).unwrap(),
    ];
    let reverse = vec![forward[1].clone(), forward[0].clone()];

    let first = write_hdf5_file_image(&forward, Hdf5WriteLimits::default()).unwrap();
    let second = write_hdf5_file_image(&reverse, Hdf5WriteLimits::default()).unwrap();
    assert_eq!(first.bytes(), second.bytes());
    assert_eq!(first.runtime(), second.runtime());

    let requests = [
        Hdf5DatasetRequest::new("/mesh/topology", Hdf5ScalarType::U64, vec![1, 3]).unwrap(),
        Hdf5DatasetRequest::new("/mesh/geometry", Hdf5ScalarType::F64, vec![3, 2]).unwrap(),
    ];
    let resolved = resolve_hdf5_file_image(
        Hdf5FileImage::new(first.bytes()),
        &requests,
        Hdf5ResolveLimits::default(),
    )
    .unwrap();
    assert_eq!(
        resolved.values(),
        &[
            Hdf5ResolvedValues::U64(topology.to_vec()),
            Hdf5ResolvedValues::F64(geometry.to_vec()),
        ]
    );
}

#[test]
fn writer_rejects_ambiguous_values_and_every_resource_excess() {
    let values = [1_u64, 2];
    let dataset = Hdf5DatasetWrite::u64("/values", vec![2], &values).unwrap();
    let duplicate = [dataset.clone(), dataset.clone()];
    assert_eq!(
        write_hdf5_file_image(&duplicate, Hdf5WriteLimits::default())
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_EXPORT,
    );
    assert!(Hdf5DatasetWrite::f64("/bad", vec![1], &[-0.0]).is_err());
    assert!(Hdf5DatasetWrite::u64("relative", vec![2], &values).is_err());

    let too_few_datasets = Hdf5WriteLimits {
        max_datasets: 1,
        ..Hdf5WriteLimits::default()
    };
    assert!(write_hdf5_file_image(&duplicate, too_few_datasets).is_err());

    let written =
        write_hdf5_file_image(std::slice::from_ref(&dataset), Hdf5WriteLimits::default()).unwrap();
    let too_few_bytes = Hdf5WriteLimits {
        max_output_bytes: written.bytes().len() - 1,
        ..Hdf5WriteLimits::default()
    };
    assert!(write_hdf5_file_image(&[dataset], too_few_bytes).is_err());
}
