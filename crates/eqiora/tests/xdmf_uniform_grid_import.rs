use eqiora::api::{import_xdmf_v1, verify_xdmf_import_v1};
use eqiora::artifact::{
    DiscreteFieldEnvelopeV1, ExternalImportManifestV1, ExternalImportObservationV1,
    ExternalImportSourceV1, RawSourceSha256, ResolvedArrayV1, ResolvedImportArrayV1,
    SimplicialMeshEnvelopeV1, StructuralSelectorV1,
};
use eqiora::diagnostic::codes;
use eqiora::io::xml::{
    XDMF_ADAPTER_ID, XDMF_ADAPTER_VERSION, XdmfArrayResponse, XdmfArrayValues, XdmfImportLimits,
    XdmfImportPlan, XdmfSelection,
};
use eqiora::meshing::{DiscreteFieldAssociation, DiscreteFieldShape, MeshQualityGate};

const METADATA: &[u8] =
    include_bytes!("../../../verify/artifacts/xdmf-uniform-grid-import/fixtures/unit-square.xdmf");
const HDF5: &[u8] =
    include_bytes!("../../../verify/artifacts/xdmf-uniform-grid-import/fixtures/unit-square.h5");

fn selection(attributes: Vec<Vec<u32>>) -> XdmfSelection {
    XdmfSelection::new(vec![0, 0], attributes).unwrap()
}

fn plan(attributes: Vec<Vec<u32>>) -> XdmfImportPlan {
    XdmfImportPlan::parse(METADATA, selection(attributes), XdmfImportLimits::default()).unwrap()
}

fn responses(plan: &XdmfImportPlan, source: &[u8]) -> Vec<XdmfArrayResponse> {
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
            XdmfArrayResponse::new(request, source.to_vec(), values)
        })
        .collect()
}

fn observation(responses: &[XdmfArrayResponse]) -> ExternalImportObservationV1 {
    let metadata = ExternalImportSourceV1::metadata_document(METADATA.to_vec(), None).unwrap();
    let mut sources = Vec::new();
    let mut arrays = Vec::new();
    for (index, response) in responses.iter().enumerate() {
        let request = response.request();
        let selector = StructuralSelectorV1::new(request.origin_selector().to_vec());
        sources.push(
            ExternalImportSourceV1::external_array_source(
                selector.clone(),
                response.source_bytes().to_vec(),
                Some(request.source_locator().to_owned()),
            )
            .unwrap(),
        );
        let array = match response.values() {
            XdmfArrayValues::U64(values) => {
                ResolvedArrayV1::from_u64(request.shape().to_vec(), values.clone()).unwrap()
            }
            XdmfArrayValues::F64(values) => {
                ResolvedArrayV1::from_f64(request.shape().to_vec(), values.clone()).unwrap()
            }
        };
        arrays.push(
            ResolvedImportArrayV1::new(
                u32::try_from(index + 1).unwrap(),
                selector,
                Some(request.dataset_path().to_owned()),
                array,
            )
            .unwrap(),
        );
    }
    let mut arrays = arrays.into_iter();
    ExternalImportObservationV1::new(
        metadata,
        sources,
        arrays.next().unwrap(),
        arrays.next().unwrap(),
        arrays.collect(),
    )
    .unwrap()
}

#[test]
fn uniform_grid_replay_derives_exact_mesh_fields_and_lineage() {
    assert_eq!(
        RawSourceSha256::from_source_bytes(METADATA).as_str(),
        "e0d23f535e5fcf2c1b982650eefb41c3836fda6a791394190c6062b986aab2ba"
    );
    assert_eq!(
        RawSourceSha256::from_source_bytes(HDF5).as_str(),
        "297ee959a81bd5fee1d77dbc9ee8d11fa4c4eacf0802558860a2e7f60124e5cb"
    );

    let plan = plan(vec![vec![0, 0, 2], vec![0, 0, 3]]);
    let responses = responses(&plan, HDF5);
    let quality = MeshQualityGate::new(0.5).unwrap();
    let derived = import_xdmf_v1(&plan, &responses, quality).unwrap();
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
    let first = verify_xdmf_import_v1(
        &persisted_manifest,
        &persisted_mesh,
        &persisted_fields,
        &plan,
        &responses,
        quality,
    )
    .unwrap();
    let second = verify_xdmf_import_v1(
        &persisted_manifest,
        &persisted_mesh,
        &persisted_fields,
        &plan,
        &responses,
        quality,
    )
    .unwrap();

    assert_eq!(first.mesh(), second.mesh());
    assert_eq!(first.fields(), second.fields());
    assert_eq!(first.manifest(), second.manifest());
    assert_eq!(first.manifest_digest(), second.manifest_digest());
    assert_eq!(first.mesh().mesh().vertices().len(), 4);
    assert_eq!(first.mesh().mesh().cells(), &[vec![0, 1, 2], vec![0, 2, 3]]);
    assert_eq!(first.fields().len(), 2);
    assert_eq!(
        first.fields()[0].association(),
        DiscreteFieldAssociation::Vertex
    );
    assert_eq!(
        first.fields()[0].component_shape(),
        DiscreteFieldShape::Scalar
    );
    assert_eq!(first.fields()[0].values(), &[10.0, 20.0, 30.0, 40.0]);
    assert_eq!(
        first.fields()[1].association(),
        DiscreteFieldAssociation::Cell
    );
    assert_eq!(
        first.fields()[1].component_shape(),
        DiscreteFieldShape::Vector {
            components: std::num::NonZeroU32::new(2).unwrap(),
        }
    );
    assert_eq!(first.fields()[1].values(), &[1.0, 0.0, 0.0, 1.0]);

    let manifest = first.manifest();
    assert_eq!(manifest.adapter().id(), XDMF_ADAPTER_ID);
    assert_eq!(manifest.adapter().version(), XDMF_ADAPTER_VERSION);
    assert!(manifest.runtime_stack().is_empty());
    assert_eq!(
        manifest.selection().grid().selector().element_path(),
        &[0, 0]
    );
    assert_eq!(
        manifest.selection().grid().display_name(),
        Some("unit-square")
    );
    assert_eq!(
        manifest
            .selection()
            .attributes()
            .iter()
            .map(|attribute| attribute.selector().element_path())
            .collect::<Vec<_>>(),
        vec![&[0, 0, 2][..], &[0, 0, 3][..]]
    );
    assert_eq!(
        manifest
            .selection()
            .attributes()
            .iter()
            .map(|attribute| attribute.display_name())
            .collect::<Vec<_>>(),
        vec![Some("temperature"), Some("flux")]
    );
    assert_eq!(
        manifest.accepted_mesh_artifact(),
        first.mesh().digest().unwrap()
    );
    assert_eq!(
        manifest.accepted_field_artifacts(),
        first
            .fields()
            .iter()
            .map(|field| field.digest().unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(&manifest.digest().unwrap(), first.manifest_digest());

    let canonical = manifest.canonical_json().unwrap();
    let decoded = ExternalImportManifestV1::from_json(&canonical, Default::default()).unwrap();
    assert_eq!(&decoded, manifest);
    decoded
        .validate_references(&observation(&responses), first.mesh(), first.fields())
        .unwrap();
    let mut crossed_fields = first.fields().to_vec();
    crossed_fields.swap(0, 1);
    assert!(
        decoded
            .validate_references(&observation(&responses), first.mesh(), &crossed_fields)
            .is_err()
    );
    assert_eq!(
        verify_xdmf_import_v1(
            &decoded,
            first.mesh(),
            &crossed_fields,
            &plan,
            &responses,
            quality,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_ARTIFACT
    );
}

#[test]
fn response_identity_and_order_fail_closed_but_source_provenance_is_independent() {
    let plan = plan(vec![vec![0, 0, 2], vec![0, 0, 3]]);
    let quality = MeshQualityGate::new(0.5).unwrap();
    let baseline_responses = responses(&plan, HDF5);
    let baseline = import_xdmf_v1(&plan, &baseline_responses, quality).unwrap();

    let mut reordered = baseline_responses.clone();
    reordered.swap(0, 1);
    assert_eq!(
        import_xdmf_v1(&plan, &reordered, quality)
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );

    let mut cross_wired = baseline_responses.clone();
    cross_wired.swap(2, 3);
    assert_eq!(
        import_xdmf_v1(&plan, &cross_wired, quality)
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );

    let mut changed_source = HDF5.to_vec();
    changed_source[0] ^= 0x01;
    let changed_responses = responses(&plan, &changed_source);
    let changed = import_xdmf_v1(&plan, &changed_responses, quality).unwrap();
    assert_eq!(baseline.mesh(), changed.mesh());
    assert_eq!(baseline.fields(), changed.fields());
    assert_ne!(baseline.manifest_digest(), changed.manifest_digest());
    assert!(
        verify_xdmf_import_v1(
            baseline.manifest(),
            baseline.mesh(),
            baseline.fields(),
            &plan,
            &changed_responses,
            quality,
        )
        .is_err()
    );

    let mut rebound_values = baseline_responses.clone();
    let temperature = rebound_values[2].values().clone();
    let flux = rebound_values[3].values().clone();
    rebound_values[2] = XdmfArrayResponse::new(&plan.requests()[2], HDF5.to_vec(), flux);
    rebound_values[3] = XdmfArrayResponse::new(&plan.requests()[3], HDF5.to_vec(), temperature);
    assert!(import_xdmf_v1(&plan, &rebound_values, quality).is_ok());
    assert!(
        verify_xdmf_import_v1(
            baseline.manifest(),
            baseline.mesh(),
            baseline.fields(),
            &plan,
            &rebound_values,
            quality,
        )
        .is_err()
    );
}

#[test]
fn explicit_attribute_order_is_preserved_in_requests_fields_and_manifest() {
    let plan = plan(vec![vec![0, 0, 3], vec![0, 0, 2]]);
    assert_eq!(plan.requests()[2].dataset_path(), "/fields/flux");
    assert_eq!(plan.requests()[3].dataset_path(), "/fields/temperature");
    let responses = responses(&plan, HDF5);
    let derived = import_xdmf_v1(&plan, &responses, MeshQualityGate::new(0.5).unwrap()).unwrap();
    let verified = verify_xdmf_import_v1(
        derived.manifest(),
        derived.mesh(),
        derived.fields(),
        &plan,
        &responses,
        MeshQualityGate::new(0.5).unwrap(),
    )
    .unwrap();

    assert_eq!(
        verified.fields()[0].association(),
        DiscreteFieldAssociation::Cell
    );
    assert_eq!(verified.fields()[0].values(), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(
        verified.fields()[1].association(),
        DiscreteFieldAssociation::Vertex
    );
    assert_eq!(verified.fields()[1].values(), &[10.0, 20.0, 30.0, 40.0]);
    assert_eq!(
        verified
            .manifest()
            .selection()
            .attributes()
            .iter()
            .map(|attribute| attribute.selector().element_path())
            .collect::<Vec<_>>(),
        vec![&[0, 0, 3][..], &[0, 0, 2][..]]
    );
}
