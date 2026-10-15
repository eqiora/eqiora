#![cfg(feature = "vtu")]

use std::num::NonZeroU32;

use eqiora::api::{import_vtu_v1, verify_vtu_import_v1};
use eqiora::artifact::{
    DiscreteFieldEnvelopeV1, ExternalImportManifestV1, RawSourceSha256, SimplicialMeshEnvelopeV1,
};
use eqiora::diagnostic::codes;
use eqiora::io::xml::{
    VTU_ADAPTER_ID, VTU_ADAPTER_VERSION, VtuCellKind, VtuImportLimits, VtuImportPlan, VtuSelection,
};
use eqiora::meshing::{
    DiscreteFieldAssociation, DiscreteFieldPayload, DiscreteFieldShape, MeshQualityGate,
    SimplicialMesh,
};

const VTU: &[u8] = include_bytes!(
    "../../../verify/artifacts/vtu-unstructured-grid-import/fixtures/unit-square-tri3-ascii.vtu"
);
const EXPECTED_SOURCE_DIGEST: &str =
    include_str!("../../../verify/artifacts/vtu-unstructured-grid-import/expected/source.sha256");

fn selection() -> VtuSelection {
    VtuSelection::new(vec![0, 0], vec![vec![0, 0, 0, 0], vec![0, 0, 1, 0]]).unwrap()
}

fn plan(source: &[u8]) -> VtuImportPlan {
    VtuImportPlan::parse(source, selection(), VtuImportLimits::default()).unwrap()
}

fn quality_gate() -> MeshQualityGate {
    MeshQualityGate::new(0.5).unwrap()
}

#[test]
fn official_ascii_vtu_replays_exact_shared_mesh_fields_and_lineage() {
    let plan = plan(VTU);
    assert_eq!(plan.cell_kind(), VtuCellKind::Triangle);
    assert_eq!(plan.selection().piece(), &[0, 0]);
    assert_eq!(plan.geometry_selector(), &[0, 0, 2, 0]);
    assert_eq!(plan.topology_selector(), &[0, 0, 3]);
    assert_eq!(plan.geometry_shape(), &[4, 2]);
    assert_eq!(plan.topology_shape(), &[2, 3]);

    let derived = import_vtu_v1(&plan, quality_gate()).unwrap();
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
    let verified = verify_vtu_import_v1(
        &persisted_manifest,
        &persisted_mesh,
        &persisted_fields,
        &plan,
        quality_gate(),
    )
    .unwrap();

    let oracle_mesh = SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ],
        vec![vec![0, 1, 2], vec![0, 2, 3]],
        quality_gate(),
    )
    .unwrap();
    let oracle_mesh = SimplicialMeshEnvelopeV1::from_mesh(&oracle_mesh).unwrap();
    let temperature = DiscreteFieldEnvelopeV1::from_payload(
        &oracle_mesh,
        &DiscreteFieldPayload::new(
            oracle_mesh.mesh(),
            DiscreteFieldAssociation::Vertex,
            DiscreteFieldShape::Scalar,
            vec![300.0, 310.0, 320.0, 330.0],
        )
        .unwrap(),
    )
    .unwrap();
    let flux = DiscreteFieldEnvelopeV1::from_payload(
        &oracle_mesh,
        &DiscreteFieldPayload::new(
            oracle_mesh.mesh(),
            DiscreteFieldAssociation::Cell,
            DiscreteFieldShape::Vector {
                components: NonZeroU32::new(2).unwrap(),
            },
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(verified.mesh(), &oracle_mesh);
    assert_eq!(verified.fields(), &[temperature, flux]);

    let manifest = verified.manifest();
    assert_eq!(manifest.adapter().id(), VTU_ADAPTER_ID);
    assert_eq!(manifest.adapter().version(), VTU_ADAPTER_VERSION);
    assert!(manifest.runtime_stack().is_empty());
    assert_eq!(
        manifest.selection().grid().selector().element_path(),
        &[0, 0]
    );
    assert_eq!(manifest.selection().grid().display_name(), None);
    assert_eq!(
        manifest
            .selection()
            .attributes()
            .iter()
            .map(|field| field.selector().element_path())
            .collect::<Vec<_>>(),
        vec![&[0, 0, 0, 0][..], &[0, 0, 1, 0][..]],
    );
    assert_eq!(
        manifest
            .selection()
            .attributes()
            .iter()
            .map(|field| field.display_name())
            .collect::<Vec<_>>(),
        vec![Some("temperature"), Some("flux")],
    );
    assert_eq!(
        manifest.accepted_mesh_artifact(),
        verified.mesh().digest().unwrap()
    );
    assert_eq!(
        manifest.accepted_field_artifacts(),
        verified
            .fields()
            .iter()
            .map(|field| field.digest().unwrap())
            .collect::<Vec<_>>(),
    );
    assert_eq!(verified.manifest_digest(), &manifest.digest().unwrap());

    let expected_source_digest = EXPECTED_SOURCE_DIGEST
        .split_whitespace()
        .next()
        .expect("source digest fixture must contain one checksum");
    assert_eq!(
        RawSourceSha256::from_source_bytes(VTU).as_str(),
        expected_source_digest,
    );
}

#[test]
fn source_provenance_changes_without_changing_accepted_content() {
    let baseline = import_vtu_v1(&plan(VTU), quality_gate()).unwrap();
    let changed_source = std::str::from_utf8(VTU).unwrap().replacen(
        "  <UnstructuredGrid>",
        "   <UnstructuredGrid>",
        1,
    );
    let changed = import_vtu_v1(&plan(changed_source.as_bytes()), quality_gate()).unwrap();

    assert_eq!(changed.mesh(), baseline.mesh());
    assert_eq!(changed.fields(), baseline.fields());
    assert_ne!(changed.manifest(), baseline.manifest());
    assert_ne!(changed.manifest_digest(), baseline.manifest_digest());
    assert_eq!(
        verify_vtu_import_v1(
            baseline.manifest(),
            baseline.mesh(),
            baseline.fields(),
            &plan(changed_source.as_bytes()),
            quality_gate(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_ARTIFACT,
    );

    let mut crossed = baseline.fields().to_vec();
    crossed.swap(0, 1);
    assert_eq!(
        verify_vtu_import_v1(
            baseline.manifest(),
            baseline.mesh(),
            &crossed,
            &plan(VTU),
            quality_gate(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_ARTIFACT,
    );
}

#[test]
fn field_name_is_display_provenance_not_selection_identity() {
    let baseline_plan = plan(VTU);
    let baseline = import_vtu_v1(&baseline_plan, quality_gate()).unwrap();
    let renamed_source = std::str::from_utf8(VTU)
        .unwrap()
        .replace("temperature", "thermal_state");
    let renamed_plan = plan(renamed_source.as_bytes());
    let renamed = import_vtu_v1(&renamed_plan, quality_gate()).unwrap();

    assert_eq!(renamed_plan.selection(), baseline_plan.selection());
    assert_eq!(renamed.mesh(), baseline.mesh());
    assert_eq!(renamed.fields(), baseline.fields());
    assert_eq!(
        renamed.manifest().accepted_mesh_artifact(),
        baseline.manifest().accepted_mesh_artifact(),
    );
    assert_eq!(
        renamed.manifest().accepted_field_artifacts(),
        baseline.manifest().accepted_field_artifacts(),
    );
    assert_ne!(
        RawSourceSha256::from_source_bytes(renamed_source.as_bytes()),
        RawSourceSha256::from_source_bytes(VTU),
    );
    assert_ne!(renamed.manifest(), baseline.manifest());
    assert_ne!(renamed.manifest_digest(), baseline.manifest_digest());
    assert_eq!(
        field_selectors(renamed.manifest()),
        field_selectors(baseline.manifest()),
    );
    assert_eq!(
        field_display_names(baseline.manifest()),
        ["temperature", "flux"]
    );
    assert_eq!(
        field_display_names(renamed.manifest()),
        ["thermal_state", "flux"],
    );
}

#[test]
fn every_registered_rejection_falsifier_fails_closed() {
    let rejected_sources = [
        (
            "multiple Piece",
            replace_once(
                VTU,
                "  </UnstructuredGrid>",
                "    <Piece NumberOfPoints=\"4\" NumberOfCells=\"2\"></Piece>\n  </UnstructuredGrid>",
            ),
            "requires exactly one Piece child",
        ),
        (
            "compressor",
            replace_once(
                VTU,
                "header_type=\"UInt32\">",
                "header_type=\"UInt32\" compressor=\"vtkZLibDataCompressor\">",
            ),
            "compressor is outside the admitted ASCII VTU subset",
        ),
        (
            "non-finite geometry",
            replace_once(VTU, "          0 0 0 1 0 0\n", "          NaN 0 0 1 0 0\n"),
            "values must all be finite",
        ),
        (
            "wrong offsets",
            replace_once(VTU, "          3 6\n", "          3 5\n"),
            "offsets do not describe fixed-width homogeneous simplices",
        ),
        (
            "wrong cell types",
            replace_once(VTU, "          5 5\n", "          5 10\n"),
            "cell types must be homogeneous",
        ),
        (
            "binary DataArray",
            replace_once(VTU, "format=\"ascii\"", "format=\"binary\""),
            "DataArray format must be ascii",
        ),
    ];
    for (falsifier, invalid, expected_message) in rejected_sources {
        assert_rejected(&invalid, selection(), falsifier, expected_message);
    }

    assert_rejected(
        VTU,
        VtuSelection::new(vec![0, 0], vec![vec![0, 0, 7, 0]]).unwrap(),
        "missing structural selector",
        "selection references a missing PointData/CellData DataArray",
    );

    let limits = VtuImportLimits {
        max_source_bytes: VTU.len() - 1,
        ..VtuImportLimits::default()
    };
    let error = VtuImportPlan::parse(VTU, selection(), limits).unwrap_err();
    assert_eq!(error.code(), codes::INVALID_EXTERNAL_DATA_IMPORT);
    assert!(
        error
            .message()
            .contains("source exceeds the configured byte limit"),
        "resource excess reached the wrong rejection gate: {error}",
    );
}

fn assert_rejected(
    source: &[u8],
    selection: VtuSelection,
    falsifier: &str,
    expected_message: &str,
) {
    let error = VtuImportPlan::parse(source, selection, VtuImportLimits::default()).unwrap_err();
    assert_eq!(error.code(), codes::INVALID_EXTERNAL_DATA_IMPORT);
    assert!(
        error.message().contains(expected_message),
        "{falsifier} reached the wrong rejection gate: {error}",
    );
}

fn field_display_names(manifest: &ExternalImportManifestV1) -> Vec<&str> {
    manifest
        .selection()
        .attributes()
        .iter()
        .map(|field| field.display_name().unwrap())
        .collect()
}

fn field_selectors(manifest: &ExternalImportManifestV1) -> Vec<&[u32]> {
    manifest
        .selection()
        .attributes()
        .iter()
        .map(|field| field.selector().element_path())
        .collect()
}

fn replace_once(source: &[u8], from: &str, to: &str) -> Vec<u8> {
    std::str::from_utf8(source)
        .unwrap()
        .replacen(from, to, 1)
        .into_bytes()
}
