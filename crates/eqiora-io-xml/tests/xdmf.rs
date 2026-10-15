use std::num::NonZeroU32;

use eqiora_core::diagnostic::codes;
use eqiora_io_xml::{
    XdmfArrayResponse, XdmfArrayRole, XdmfArrayValues, XdmfImportLimits, XdmfImportPlan,
    XdmfSelection, XdmfTemporalExportLimits, XdmfTemporalExportPlan, XdmfTemporalField,
    XdmfTemporalFrame,
};
use eqiora_meshing::{DiscreteFieldAssociation, DiscreteFieldShape, MeshQualityGate};

const XML: &[u8] =
    include_bytes!("../../../verify/artifacts/xdmf-uniform-grid-import/fixtures/unit-square.xdmf");
const HDF: &[u8] =
    include_bytes!("../../../verify/artifacts/xdmf-uniform-grid-import/fixtures/unit-square.h5");

fn selection(attributes: Vec<Vec<u32>>) -> XdmfSelection {
    XdmfSelection::new(vec![0, 0], attributes).unwrap()
}

fn plan(attributes: Vec<Vec<u32>>) -> XdmfImportPlan {
    XdmfImportPlan::parse(XML, selection(attributes), XdmfImportLimits::default()).unwrap()
}

fn responses(plan: &XdmfImportPlan) -> Vec<XdmfArrayResponse> {
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
                other => panic!("unexpected request {other}"),
            };
            XdmfArrayResponse::new(request, HDF.to_vec(), values)
        })
        .collect()
}

fn temporal_field(name: &str, dataset_path: &str) -> XdmfTemporalField {
    XdmfTemporalField::new(
        name,
        DiscreteFieldAssociation::Vertex,
        DiscreteFieldShape::Scalar,
        dataset_path,
    )
    .unwrap()
}

fn temporal_frame(
    sequence: u64,
    time_s: f64,
    vertex_count: usize,
    cell_count: usize,
    suffix: &str,
    fields: Vec<XdmfTemporalField>,
) -> XdmfTemporalFrame {
    XdmfTemporalFrame::new(
        sequence,
        time_s,
        2,
        vertex_count,
        cell_count,
        format!("/geometry/{suffix}/coordinates"),
        format!("/meshes/{suffix}/topology"),
        fields,
    )
    .unwrap()
}

#[test]
fn plan_is_structural_and_requests_canonical_order() {
    let plan = plan(vec![vec![0, 0, 3], vec![0, 0, 2]]);
    assert_eq!(plan.metadata_bytes(), XML);
    assert_eq!(plan.selection().grid(), [0, 0]);
    assert_eq!(
        plan.requests()
            .iter()
            .map(|request| request.role())
            .collect::<Vec<_>>(),
        [
            XdmfArrayRole::Geometry,
            XdmfArrayRole::Topology,
            XdmfArrayRole::Attribute,
            XdmfArrayRole::Attribute,
        ]
    );
    assert_eq!(plan.requests()[0].origin_selector(), [0, 0, 1]);
    assert_eq!(plan.requests()[1].origin_selector(), [0, 0, 0]);
    assert_eq!(plan.requests()[2].origin_selector(), [0, 0, 3]);
    assert_eq!(plan.requests()[3].origin_selector(), [0, 0, 2]);
    assert_eq!(plan.requests()[0].dataset_path(), "/mesh/geometry");
    assert_eq!(plan.requests()[0].source_locator(), "unit-square.h5");
}

#[test]
fn caller_responses_reconstruct_shared_mesh_and_field_contracts() {
    let plan = plan(vec![vec![0, 0, 2], vec![0, 0, 3]]);
    let accepted = plan
        .accept(&responses(&plan), MeshQualityGate::new(0.01).unwrap())
        .unwrap();
    assert_eq!(accepted.mesh().vertices().len(), 4);
    assert_eq!(accepted.mesh().cells(), [vec![0, 1, 2], vec![0, 2, 3]]);
    assert_eq!(accepted.fields().len(), 2);
    assert_eq!(accepted.fields()[0].name(), Some("temperature"));
    assert_eq!(
        accepted.fields()[0].payload().association(),
        DiscreteFieldAssociation::Vertex
    );
    assert_eq!(
        accepted.fields()[0].payload().component_shape(),
        DiscreteFieldShape::Scalar
    );
    assert_eq!(accepted.fields()[1].name(), Some("flux"));
    assert_eq!(
        accepted.fields()[1].payload().association(),
        DiscreteFieldAssociation::Cell
    );
}

#[test]
fn response_identity_type_shape_and_resource_mismatches_fail_closed() {
    let plan = plan(vec![vec![0, 0, 2], vec![0, 0, 3]]);
    let mut resolved = responses(&plan);
    resolved.swap(0, 1);
    assert_eq!(
        plan.accept(&resolved, MeshQualityGate::new(0.01).unwrap())
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );

    let mut resolved = responses(&plan);
    resolved[0] = XdmfArrayResponse::new(
        &plan.requests()[0],
        HDF.to_vec(),
        XdmfArrayValues::U64(vec![0; 8]),
    );
    assert!(
        plan.accept(&resolved, MeshQualityGate::new(0.01).unwrap())
            .is_err()
    );

    let mut resolved = responses(&plan);
    resolved[2] = XdmfArrayResponse::new(
        &plan.requests()[2],
        HDF.to_vec(),
        XdmfArrayValues::F64(vec![1.0]),
    );
    assert!(
        plan.accept(&resolved, MeshQualityGate::new(0.01).unwrap())
            .is_err()
    );

    let mut resolved = responses(&plan);
    resolved[0] = XdmfArrayResponse::new(
        &plan.requests()[0],
        Vec::new(),
        XdmfArrayValues::F64(vec![0.0; 8]),
    );
    assert!(
        plan.accept(&resolved, MeshQualityGate::new(0.01).unwrap())
            .is_err()
    );
}

#[test]
fn hostile_xml_subset_and_every_truncated_prefix_are_rejected() {
    let meaningful_bytes = XML
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .unwrap()
        + 1;
    for prefix in 0..meaningful_bytes {
        assert!(
            XdmfImportPlan::parse(
                &XML[..prefix],
                selection(vec![vec![0, 0, 2]]),
                XdmfImportLimits::default(),
            )
            .is_err(),
            "accepted truncated prefix {prefix}"
        );
    }
    for hostile in [
        String::from("<!DOCTYPE Xdmf [<!ENTITY x 'y'>]><Xdmf Version=\"3.0\"></Xdmf>"),
        String::from("<?bad instruction?><Xdmf Version=\"3.0\"></Xdmf>"),
        String::from("<Xdmf Version=\"3.0\"><xi:include href=\"x\"/></Xdmf>"),
        String::from_utf8(XML.to_vec())
            .unwrap()
            .replace("<Domain>", "<!-- bad -- comment --><Domain>"),
        String::from_utf8(XML.to_vec())
            .unwrap()
            .replace("Name=\"unit-square\"", "Name=\"unit<square\""),
        String::from_utf8(XML.to_vec())
            .unwrap()
            .replace("unit-square.h5:/mesh/geometry", "]]>"),
        String::from_utf8(XML.to_vec())
            .unwrap()
            .replace("Name=\"temperature\"", "Name=\"temp\0erature\""),
        String::from_utf8(XML.to_vec())
            .unwrap()
            .replace("encoding=\"UTF-8\"", "encoding=\"UTF-8\" bad=\"1\""),
        String::from_utf8(XML.to_vec())
            .unwrap()
            .replace("encoding=\"UTF-8\"", "version=\"1.0\" encoding=\"UTF-8\""),
        String::from_utf8(XML.to_vec()).unwrap().replace(
            "version=\"1.0\" encoding=\"UTF-8\"",
            "encoding=\"UTF-8\" version=\"1.0\"",
        ),
        format!(" \n{}", String::from_utf8(XML.to_vec()).unwrap()),
        format!("\u{00A0}{}", String::from_utf8(XML.to_vec()).unwrap()),
        format!("{}\u{2003}", String::from_utf8(XML.to_vec()).unwrap()),
        String::from_utf8(XML.to_vec())
            .unwrap()
            .replace("<Domain>", "<Domain>\u{00A0}"),
        String::from_utf8(XML.to_vec()).unwrap().replace(
            "<Geometry GeometryType=\"XY\">",
            "<Geometry GeometryType=\"XY\" Bad=\"1\">",
        ),
        String::from_utf8(XML.to_vec())
            .unwrap()
            .replace("</Grid>", "<Time Value=\"0\"></Time></Grid>"),
    ] {
        assert!(
            XdmfImportPlan::parse(
                hostile.as_bytes(),
                selection(vec![vec![0, 0, 2]]),
                XdmfImportLimits::default(),
            )
            .is_err()
        );
    }
}

#[test]
fn hdf_reference_trims_only_xml_space_and_preserves_unicode_identity() {
    let unicode_dataset = String::from_utf8(XML.to_vec())
        .unwrap()
        .replace("/mesh/geometry", "/mesh/geometry\u{00A0}");
    let plan = XdmfImportPlan::parse(
        unicode_dataset.as_bytes(),
        selection(Vec::new()),
        XdmfImportLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.requests()[0].dataset_path(), "/mesh/geometry\u{00A0}");
}

#[test]
fn every_attribute_has_entity_and_topological_vector_shape() {
    let base = String::from_utf8(XML.to_vec()).unwrap();
    let vector3 = base.replace("Dimensions=\"2 2\"", "Dimensions=\"2 3\"");
    assert!(
        XdmfImportPlan::parse(
            vector3.as_bytes(),
            selection(vec![vec![0, 0, 3]]),
            XdmfImportLimits::default(),
        )
        .is_err()
    );

    let wrong_unselected_entities = base.replace(
        "Dimensions=\"4\">unit-square.h5:/fields/temperature",
        "Dimensions=\"3\">unit-square.h5:/fields/temperature",
    );
    assert!(
        XdmfImportPlan::parse(
            wrong_unselected_entities.as_bytes(),
            selection(Vec::new()),
            XdmfImportLimits::default(),
        )
        .is_err()
    );

    let tetrahedron_with_vector = br#"<Xdmf Version="3.0"><Domain><Grid GridType="Uniform"><Geometry GeometryType="XYZ"><DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="4 3">a.h5:/g</DataItem></Geometry><Topology TopologyType="Tetrahedron" NumberOfElements="1"><DataItem Format="HDF" DataType="UInt" Precision="8" Dimensions="1 4">a.h5:/t</DataItem></Topology><Attribute AttributeType="Vector" Center="Node"><DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="4 2">a.h5:/v</DataItem></Attribute></Grid></Domain></Xdmf>"#;
    assert!(
        XdmfImportPlan::parse(
            tetrahedron_with_vector,
            selection(vec![vec![0, 0, 2]]),
            XdmfImportLimits::default(),
        )
        .is_err()
    );
    let vector3 = std::str::from_utf8(tetrahedron_with_vector)
        .unwrap()
        .replace("Dimensions=\"4 2\">a.h5:/v", "Dimensions=\"4 3\">a.h5:/v");
    assert!(
        XdmfImportPlan::parse(
            vector3.as_bytes(),
            selection(vec![vec![0, 0, 2]]),
            XdmfImportLimits::default(),
        )
        .is_ok()
    );
}

#[test]
fn selectors_and_parser_limits_are_fail_closed() {
    assert!(
        XdmfImportPlan::parse(
            XML,
            selection(vec![vec![0, 0, 9]]),
            XdmfImportLimits::default(),
        )
        .is_err()
    );
    assert!(
        XdmfImportPlan::parse(
            XML,
            XdmfSelection::new(vec![9], Vec::new()).unwrap(),
            XdmfImportLimits::default(),
        )
        .is_err()
    );
    assert!(
        XdmfImportPlan::parse(
            XML,
            selection(Vec::new()),
            XdmfImportLimits {
                max_metadata_bytes: XML.len() - 1,
                ..XdmfImportLimits::default()
            },
        )
        .is_err()
    );
    assert!(
        XdmfImportPlan::parse(
            XML,
            selection(Vec::new()),
            XdmfImportLimits {
                max_depth: 2,
                ..XdmfImportLimits::default()
            },
        )
        .is_err()
    );
    assert!(
        XdmfImportPlan::parse(
            XML,
            selection(Vec::new()),
            XdmfImportLimits {
                max_data_items: 3,
                ..XdmfImportLimits::default()
            },
        )
        .is_err()
    );
    assert!(
        XdmfImportPlan::parse(
            XML,
            selection(Vec::new()),
            XdmfImportLimits {
                max_parser_work: 1,
                ..XdmfImportLimits::default()
            },
        )
        .is_err()
    );

    let resolution_limited = XdmfImportPlan::parse(
        XML,
        selection(Vec::new()),
        XdmfImportLimits {
            max_resolution_work: 1,
            ..XdmfImportLimits::default()
        },
    )
    .unwrap();
    assert!(
        resolution_limited
            .accept(
                &responses(&resolution_limited),
                MeshQualityGate::new(0.01).unwrap(),
            )
            .is_err()
    );
}

#[test]
fn tetrahedron_and_zero_z_xyz_are_admitted_but_nonzero_z_is_not() {
    let tet = br#"<Xdmf Version="3.0"><Domain><Grid GridType="Uniform"><Geometry GeometryType="XYZ"><DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="4 3">a.h5:/g</DataItem></Geometry><Topology TopologyType="Tetrahedron" NumberOfElements="1"><DataItem Format="HDF" DataType="UInt" Precision="8" Dimensions="1 4">a.h5:/t</DataItem></Topology></Grid></Domain></Xdmf>"#;
    let tet_plan =
        XdmfImportPlan::parse(tet, selection(Vec::new()), XdmfImportLimits::default()).unwrap();
    let tet_responses = vec![
        XdmfArrayResponse::new(
            &tet_plan.requests()[0],
            vec![1],
            XdmfArrayValues::F64(vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ]),
        ),
        XdmfArrayResponse::new(
            &tet_plan.requests()[1],
            vec![1],
            XdmfArrayValues::U64(vec![0, 1, 2, 3]),
        ),
    ];
    assert!(
        tet_plan
            .accept(&tet_responses, MeshQualityGate::new(0.01).unwrap())
            .is_ok()
    );

    let xyz = String::from_utf8(XML.to_vec())
        .unwrap()
        .replace("GeometryType=\"XY\"", "GeometryType=\"XYZ\"")
        .replace("Dimensions=\"4 2\"", "Dimensions=\"4 3\"");
    let xyz_plan = XdmfImportPlan::parse(
        xyz.as_bytes(),
        selection(Vec::new()),
        XdmfImportLimits::default(),
    )
    .unwrap();
    let geometry = &xyz_plan.requests()[0];
    let topology = &xyz_plan.requests()[1];
    let accepted = vec![
        XdmfArrayResponse::new(
            geometry,
            vec![1],
            XdmfArrayValues::F64(vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ]),
        ),
        XdmfArrayResponse::new(
            topology,
            vec![1],
            XdmfArrayValues::U64(vec![0, 1, 2, 0, 2, 3]),
        ),
    ];
    assert!(
        xyz_plan
            .accept(&accepted, MeshQualityGate::new(0.01).unwrap())
            .is_ok()
    );
    let mut rejected = accepted;
    rejected[0] = XdmfArrayResponse::new(
        geometry,
        vec![1],
        XdmfArrayValues::F64(vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ]),
    );
    assert!(
        xyz_plan
            .accept(&rejected, MeshQualityGate::new(0.01).unwrap())
            .is_err()
    );
}

#[test]
fn temporal_collection_is_canonical_across_declaration_order_and_remeshing() {
    let first_fields = vec![
        temporal_field("velocity", "/fields/velocity-a/values"),
        temporal_field("pressure", "/fields/pressure-a/values"),
    ];
    let second_fields = vec![
        temporal_field("pressure", "/fields/pressure-b/values"),
        temporal_field("velocity", "/fields/velocity-b/values"),
    ];
    let first = temporal_frame(0, 0.0, 3, 1, "mesh-a", first_fields);
    let second = temporal_frame(1, 0.25, 4, 2, "mesh-b", second_fields);

    let forward = XdmfTemporalExportPlan::new(
        "trajectory&fields.h5",
        vec![first.clone(), second.clone()],
        XdmfTemporalExportLimits::default(),
    )
    .unwrap();
    let reverse = XdmfTemporalExportPlan::new(
        "trajectory&fields.h5",
        vec![second, first],
        XdmfTemporalExportLimits::default(),
    )
    .unwrap();

    assert_eq!(forward.metadata_bytes(), reverse.metadata_bytes());
    assert_eq!(forward.frames()[0].sequence(), 0);
    assert_eq!(forward.frames()[1].vertex_count(), 4);
    let metadata = std::str::from_utf8(forward.metadata_bytes()).unwrap();
    assert!(metadata.contains("GridType=\"Collection\" CollectionType=\"Temporal\""));
    assert!(metadata.contains("<Time Value=\"0\"/>"));
    assert!(metadata.contains("<Time Value=\"0.25\"/>"));
    assert!(metadata.contains("Dimensions=\"2 3\""));
    assert!(metadata.contains("Dimensions=\"4\""));
    assert!(metadata.contains("trajectory&amp;fields.h5:/meshes/mesh-b/topology"));
    assert!(
        metadata.find("Name=\"pressure\"").unwrap() < metadata.find("Name=\"velocity\"").unwrap()
    );
}

#[test]
fn temporal_collection_allows_content_addressed_array_reuse_between_frames() {
    let fields = || vec![temporal_field("temperature", "/fields/stable/values")];
    let first = temporal_frame(0, 0.0, 3, 1, "stable", fields());
    let second = temporal_frame(1, 1.0, 3, 1, "stable", fields());
    let plan = XdmfTemporalExportPlan::new(
        "stable.h5",
        vec![first, second],
        XdmfTemporalExportLimits::default(),
    )
    .unwrap();
    let metadata = std::str::from_utf8(plan.metadata_bytes()).unwrap();

    assert_eq!(
        metadata
            .matches("stable.h5:/meshes/stable/topology")
            .count(),
        2
    );
    assert_eq!(
        metadata.matches("stable.h5:/fields/stable/values").count(),
        2
    );
}

#[test]
fn distinct_semantic_fields_may_share_one_exact_content_array() {
    let shared = "/fields/shared/values";
    let fields = vec![
        temporal_field("solid-displacement", shared),
        temporal_field("solid-velocity", shared),
    ];
    let plan = XdmfTemporalExportPlan::new(
        "stable.h5",
        vec![
            temporal_frame(0, 0.0, 3, 1, "stable", fields.clone()),
            temporal_frame(1, 1.0, 3, 1, "stable", fields),
        ],
        XdmfTemporalExportLimits::default(),
    )
    .unwrap();
    let metadata = std::str::from_utf8(plan.metadata_bytes()).unwrap();

    assert_eq!(
        metadata.matches("stable.h5:/fields/shared/values").count(),
        4
    );
}

#[test]
fn temporal_frame_and_collection_semantics_fail_closed() {
    let duplicate_path = XdmfTemporalField::new(
        "temperature",
        DiscreteFieldAssociation::Vertex,
        DiscreteFieldShape::Scalar,
        "/geometry/a/coordinates",
    )
    .unwrap();
    assert!(
        XdmfTemporalFrame::new(
            0,
            0.0,
            2,
            3,
            1,
            "/geometry/a/coordinates",
            "/meshes/a/topology",
            vec![duplicate_path],
        )
        .is_err()
    );

    let wrong_vector = XdmfTemporalField::new(
        "velocity",
        DiscreteFieldAssociation::Vertex,
        DiscreteFieldShape::Vector {
            components: NonZeroU32::new(3).unwrap(),
        },
        "/fields/velocity/values",
    )
    .unwrap();
    assert!(
        XdmfTemporalFrame::new(
            0,
            0.0,
            2,
            3,
            1,
            "/geometry/a/coordinates",
            "/meshes/a/topology",
            vec![wrong_vector],
        )
        .is_err()
    );

    let field = || vec![temporal_field("temperature", "/fields/t/values")];
    let first = temporal_frame(0, 0.0, 3, 1, "a", field());
    let gap = temporal_frame(2, 1.0, 3, 1, "a", field());
    assert!(
        XdmfTemporalExportPlan::new(
            "a.h5",
            vec![first.clone(), gap],
            XdmfTemporalExportLimits::default(),
        )
        .is_err()
    );

    let equal_time = temporal_frame(1, 0.0, 3, 1, "a", field());
    assert!(
        XdmfTemporalExportPlan::new(
            "a.h5",
            vec![first.clone(), equal_time],
            XdmfTemporalExportLimits::default(),
        )
        .is_err()
    );

    let changed_inventory = temporal_frame(
        1,
        1.0,
        3,
        1,
        "a",
        vec![temporal_field("pressure", "/fields/p/values")],
    );
    assert!(
        XdmfTemporalExportPlan::new(
            "a.h5",
            vec![first, changed_inventory],
            XdmfTemporalExportLimits::default(),
        )
        .is_err()
    );
}

#[test]
fn every_temporal_export_budget_is_enforced() {
    let fields = vec![
        temporal_field("pressure", "/fields/p/values"),
        temporal_field("velocity", "/fields/v/values"),
    ];
    let frames = vec![
        temporal_frame(0, 0.0, 3, 1, "a", fields.clone()),
        temporal_frame(1, 1.0, 3, 1, "a", fields),
    ];
    let complete =
        XdmfTemporalExportPlan::new("a.h5", frames.clone(), XdmfTemporalExportLimits::default())
            .unwrap();

    for limits in [
        XdmfTemporalExportLimits {
            max_frames: 1,
            ..XdmfTemporalExportLimits::default()
        },
        XdmfTemporalExportLimits {
            max_fields_per_frame: 1,
            ..XdmfTemporalExportLimits::default()
        },
        XdmfTemporalExportLimits {
            max_text_bytes: 4,
            ..XdmfTemporalExportLimits::default()
        },
        XdmfTemporalExportLimits {
            max_metadata_bytes: complete.metadata_bytes().len() - 1,
            ..XdmfTemporalExportLimits::default()
        },
    ] {
        assert!(XdmfTemporalExportPlan::new("a.h5", frames.clone(), limits).is_err());
    }
}

#[test]
fn parser_work_includes_metadata_bytes_before_event_allocation() {
    assert!(XdmfImportPlan::parse(XML, selection(Vec::new()), XdmfImportLimits::default()).is_ok());
    assert!(
        XdmfImportPlan::parse(
            XML,
            selection(Vec::new()),
            XdmfImportLimits {
                max_parser_work: XML.len() - 1,
                ..XdmfImportLimits::default()
            }
        )
        .is_err()
    );
}
