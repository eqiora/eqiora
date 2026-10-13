use eqiora_core::diagnostic::codes;
use eqiora_meshing::{DiscreteFieldAssociation, DiscreteFieldShape, MeshQualityGate, MeshTopology};

use super::super::plan::{PORTABLE_MAX_SELECTED_FIELDS, PORTABLE_MAX_SELECTOR_DEPTH};

use super::*;

const PIECE: &[u32] = &[0, 0];
const POINT_FIELD: &[u32] = &[0, 0, 2, 0];
const CELL_FIELD: &[u32] = &[0, 0, 3, 0];
const OFFICIAL_VTK: &str = include_str!(
    "../../../../verify/artifacts/vtu-unstructured-grid-import/fixtures/unit-square-tri3-ascii.vtu"
);

fn source(offset: &str, cell_type: &str, point_values: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <UnstructuredGrid>
    <Piece NumberOfPoints="3" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">0 0 0  1 0 0  0 1 0</DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii">0 1 2</DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii">{offset}</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">{cell_type}</DataArray>
      </Cells>
      <PointData>
        <DataArray type="Float64" Name="temperature" NumberOfComponents="1" format="ascii">{point_values}</DataArray>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="flux" NumberOfComponents="2" format="ascii">4 5</DataArray>
      </CellData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#
    )
}

fn tetrahedron_source(connectivity: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 0  1 0 0  0 1 0  0 0 1
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">{connectivity}</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">10</DataArray>
      </Cells>
      <PointData>
        <DataArray type="Float64" Name="temperature" format="ascii">1 2 3 4</DataArray>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="flux" NumberOfComponents="2" format="ascii">5 6</DataArray>
      </CellData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#
    )
}

fn selection() -> VtuSelection {
    VtuSelection::new(
        PIECE.to_vec(),
        vec![POINT_FIELD.to_vec(), CELL_FIELD.to_vec()],
    )
    .unwrap()
}

#[test]
fn selection_constructor_enforces_portable_metadata_bounds() {
    assert_eq!(
        VtuSelection::new(vec![0; PORTABLE_MAX_SELECTOR_DEPTH + 1], Vec::new())
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );
    assert_eq!(
        VtuSelection::new(vec![0], vec![vec![0]; PORTABLE_MAX_SELECTED_FIELDS + 1],)
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );

    let fields = (0..PORTABLE_MAX_SELECTED_FIELDS)
        .map(|index| {
            let mut selector = vec![0; PORTABLE_MAX_SELECTOR_DEPTH];
            selector[0] = u32::try_from(index).unwrap();
            selector
        })
        .collect();
    assert_eq!(
        VtuSelection::new(vec![u32::MAX; PORTABLE_MAX_SELECTOR_DEPTH], fields)
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );
}

#[test]
fn one_ascii_piece_reconstructs_shared_mesh_and_selected_fields() {
    let source = source("3", "5", "1 2 3");
    let plan =
        VtuImportPlan::parse(source.as_bytes(), selection(), VtuImportLimits::default()).unwrap();
    assert_eq!(plan.source_bytes(), source.as_bytes());
    assert_eq!(plan.selection().piece(), PIECE);
    assert_eq!(plan.cell_kind(), VtuCellKind::Triangle);
    assert_eq!(plan.geometry_selector(), &[0, 0, 0, 0]);
    assert_eq!(plan.topology_selector(), &[0, 0, 1]);
    assert_eq!(plan.geometry_shape(), &[3, 2]);
    assert_eq!(plan.topology_shape(), &[1, 3]);
    assert_eq!(plan.normalized_geometry(), &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    assert_eq!(plan.normalized_topology(), &[0, 1, 2]);

    let imported = plan.accept(MeshQualityGate::new(0.5).unwrap()).unwrap();
    assert_eq!(imported.mesh().topological_dimension(), 2);
    assert_eq!(imported.mesh().entity_count(0), Some(3));
    assert_eq!(imported.mesh().entity_count(2), Some(1));
    assert_eq!(imported.fields().len(), 2);
    assert_eq!(imported.fields()[0].selector(), POINT_FIELD);
    assert_eq!(imported.fields()[0].name(), Some("temperature"));
    assert_eq!(imported.fields()[0].raw_shape(), &[3]);
    assert_eq!(
        imported.fields()[0].payload().association(),
        DiscreteFieldAssociation::Vertex
    );
    assert_eq!(
        imported.fields()[0].payload().component_shape(),
        DiscreteFieldShape::Scalar
    );
    assert_eq!(imported.fields()[0].payload().values(), &[1.0, 2.0, 3.0]);
    assert_eq!(imported.fields()[1].selector(), CELL_FIELD);
    assert_eq!(imported.fields()[1].raw_shape(), &[1, 2]);
    assert_eq!(
        imported.fields()[1].payload().association(),
        DiscreteFieldAssociation::Cell
    );
}

#[test]
fn official_vtk_metadata_is_bounded_and_nonsemantic() {
    let selection =
        VtuSelection::new(PIECE.to_vec(), vec![vec![0, 0, 0, 0], vec![0, 0, 1, 0]]).unwrap();
    let plan = VtuImportPlan::parse(
        OFFICIAL_VTK.as_bytes(),
        selection,
        VtuImportLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.geometry_selector(), &[0, 0, 2, 0]);
    assert_eq!(plan.topology_selector(), &[0, 0, 3]);
    assert_eq!(plan.geometry_shape(), &[4, 2]);
    assert_eq!(plan.topology_shape(), &[2, 3]);
    let imported = plan.accept(MeshQualityGate::new(0.5).unwrap()).unwrap();
    assert_eq!(imported.mesh().entity_count(0), Some(4));
    assert_eq!(imported.mesh().entity_count(2), Some(2));
    assert_eq!(imported.fields()[0].name(), Some("temperature"));
    assert_eq!(imported.fields()[1].name(), Some("flux"));
}

#[test]
fn wrong_offsets_fail_closed() {
    let error = VtuImportPlan::parse(
        source("2", "5", "1 2 3").as_bytes(),
        selection(),
        VtuImportLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_EXTERNAL_DATA_IMPORT);
}

#[test]
fn positive_tetrahedron_is_admitted_and_negative_orientation_is_rejected() {
    let positive = VtuImportPlan::parse(
        tetrahedron_source("0 1 2 3").as_bytes(),
        selection(),
        VtuImportLimits::default(),
    )
    .unwrap();
    assert_eq!(positive.cell_kind(), VtuCellKind::Tetrahedron);
    assert_eq!(positive.geometry_shape(), &[4, 3]);
    assert_eq!(positive.topology_shape(), &[1, 4]);
    assert_eq!(positive.normalized_topology(), &[0, 1, 2, 3]);
    assert_eq!(
        positive
            .accept(MeshQualityGate::new(1.0e-12).unwrap())
            .unwrap()
            .mesh()
            .topological_dimension(),
        3
    );

    let negative = VtuImportPlan::parse(
        tetrahedron_source("0 2 1 3").as_bytes(),
        selection(),
        VtuImportLimits::default(),
    )
    .unwrap();
    assert_eq!(
        negative
            .accept(MeshQualityGate::new(1.0e-12).unwrap())
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );
}

#[test]
fn unselected_primitive_fields_are_preflighted_without_retention() {
    let source = source("3", "5", "1 2 3")
        .replace("<PointData>", "<PointData Normals=\"temperature\">")
        .replace("<CellData>", "<CellData Tensors=\"flux\">")
        .replace(
            "type=\"Float64\" Name=\"temperature\"",
            "type=\"Int16\" Name=\"temperature\"",
        )
        .replace(
            "type=\"Float64\" Name=\"flux\"",
            "type=\"Float32\" Name=\"flux\"",
        );
    let unselected = VtuSelection::new(PIECE.to_vec(), Vec::new()).unwrap();
    let plan = VtuImportPlan::parse(
        source.as_bytes(),
        unselected.clone(),
        VtuImportLimits::default(),
    )
    .unwrap();
    assert!(
        plan.accept(MeshQualityGate::new(0.5).unwrap())
            .unwrap()
            .fields()
            .is_empty()
    );

    let invalid = source.replace(">1 2 3</DataArray>", ">1 invalid 3</DataArray>");
    assert_eq!(
        VtuImportPlan::parse(invalid.as_bytes(), unselected, VtuImportLimits::default(),)
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );
}

#[test]
fn field_shape_count_and_required_containers_fail_closed() {
    let wrong_count = source("3", "5", "1 2");
    assert_eq!(
        VtuImportPlan::parse(
            wrong_count.as_bytes(),
            selection(),
            VtuImportLimits::default(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );

    let missing_cell_data = source("3", "5", "1 2 3").replace(
            "      <CellData>\n        <DataArray type=\"Float64\" Name=\"flux\" NumberOfComponents=\"2\" format=\"ascii\">4 5</DataArray>\n      </CellData>\n",
            "",
        );
    assert_eq!(
        VtuImportPlan::parse(
            missing_cell_data.as_bytes(),
            selection(),
            VtuImportLimits::default(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );
}

#[test]
fn unsupported_cell_type_fails_closed() {
    let error = VtuImportPlan::parse(
        source("3", "9", "1 2 3").as_bytes(),
        selection(),
        VtuImportLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_EXTERNAL_DATA_IMPORT);
}

#[test]
fn nonfinite_field_fails_closed() {
    let error = VtuImportPlan::parse(
        source("3", "5", "1 NaN 3").as_bytes(),
        selection(),
        VtuImportLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_EXTERNAL_DATA_IMPORT);
}

#[test]
fn point_and_work_limits_fail_closed() {
    let source = source("3", "5", "1 2 3");
    let limits = VtuImportLimits {
        max_points: 2,
        ..VtuImportLimits::default()
    };
    assert_eq!(
        VtuImportPlan::parse(source.as_bytes(), selection(), limits)
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );

    let limits = VtuImportLimits {
        max_parser_work: 8,
        ..VtuImportLimits::default()
    };
    let error = VtuImportPlan::parse(source.as_bytes(), selection(), limits).unwrap_err();
    assert_eq!(error.code(), codes::INVALID_EXTERNAL_DATA_IMPORT);
    assert!(
        error
            .message()
            .contains("parser work exceeds the configured limit"),
        "source scan reached the wrong rejection gate: {error}",
    );

    let limits = VtuImportLimits {
        max_selected_fields: 1,
        ..VtuImportLimits::default()
    };
    assert_eq!(
        VtuImportPlan::parse(source.as_bytes(), selection(), limits)
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );

    let limits = VtuImportLimits {
        max_selector_depth: 3,
        ..VtuImportLimits::default()
    };
    assert_eq!(
        VtuImportPlan::parse(source.as_bytes(), selection(), limits)
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );

    let limits = VtuImportLimits {
        max_selector_values: 9,
        ..VtuImportLimits::default()
    };
    assert_eq!(
        VtuImportPlan::parse(source.as_bytes(), selection(), limits)
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );

    let limits = VtuImportLimits {
        max_resolved_bytes: 8,
        ..VtuImportLimits::default()
    };
    assert_eq!(
        VtuImportPlan::parse(source.as_bytes(), selection(), limits)
            .unwrap_err()
            .code(),
        codes::INVALID_EXTERNAL_DATA_IMPORT
    );
}
