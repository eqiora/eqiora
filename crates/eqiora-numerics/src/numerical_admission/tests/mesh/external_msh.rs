//! Expectations authored before the import implementation: two square triangles
//! and one unit tetrahedron, with analytic boundaries and unchanged input cells.
use super::*;
use eqiora_artifact::MshImportPolicyV1;
use eqiora_geometry::{PlanarFace, PlanarRegion};

const SQUARE: &[u8] = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n$Elements\n1 2 1 2\n2 1 2 2\n1 1 2 3\n2 1 3 4\n$EndElements\n";
const TETRAHEDRON: &[u8] = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 4 1 4\n3 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n$Elements\n1 1 1 1\n3 1 4 1\n1 1 2 3 4\n$EndElements\n";

fn square(width: f64) -> CanonicalGeometryV1 {
    CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            vec![[0., 0.], [width, 0.], [width, 1.], [0., 1.]],
            vec![PlanarFace::new(vec![0, 1, 2, 3], vec![])],
            vec![
                NamedEntitySet::new("body", 2, vec![0]),
                NamedEntitySet::new("outer", 1, vec![0, 1, 2, 3]),
            ],
            1e-12,
        )
        .unwrap(),
    )
    .unwrap()
}

fn tetrahedron() -> CanonicalGeometryV1 {
    CanonicalGeometryV1::from_convex_polyhedra(
        vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        vec![vec![
            vec![0, 2, 1],
            vec![0, 1, 3],
            vec![0, 3, 2],
            vec![1, 2, 3],
        ]],
        vec![
            NamedEntitySet::new("body", 3, vec![0]),
            NamedEntitySet::new("outer", 2, vec![0, 1, 2, 3]),
        ],
        1e-12,
    )
    .unwrap()
}

fn policy(facets: usize, bytes: usize) -> MshImportPolicyV1 {
    MshImportPolicyV1::metres(0.01, facets, bytes).unwrap()
}

#[test]
fn external_msh_retains_two_triangles_or_one_tetrahedron_and_replays_honest_origin() {
    for (geometry, source, dimension, cells) in [
        (square(1.), SQUARE, 2, 2),
        (tetrahedron(), TETRAHEDRON, 3, 1),
    ] {
        let owner = AuthenticatedCommonMesh::import_msh41(
            geometry.clone(),
            policy(4, 4096),
            source.to_vec(),
        )
        .unwrap();
        assert_eq!(owner.geometry(), Some(&geometry));
        let mesh = owner.simplicial_mesh().unwrap();
        assert_eq!(mesh.dimension(), dimension);
        assert_eq!(mesh.mesh().entity_count(0), Some(4));
        assert_eq!(mesh.mesh().entity_count(dimension), Some(cells));
        let expected_cells = if dimension == 2 {
            vec![vec![0, 1, 2], vec![0, 2, 3]]
        } else {
            vec![vec![0, 1, 2, 3]]
        };
        assert_eq!(mesh.mesh().cells(), expected_cells);
        assert_eq!(
            (0..mesh.mesh().entity_count(dimension - 1).unwrap())
                .filter(|&i| mesh
                    .mesh()
                    .is_boundary_entity(MeshEntity::new(dimension - 1, i))
                    == Some(true))
                .count(),
            4
        );
        assert_eq!(
            owner.production().unwrap().provider_identity(),
            "eqiora.external-msh"
        );
        assert_eq!(owner.production().unwrap().provider_version(), "4.1");
        assert!(owner.production().unwrap().gmsh_mesh_policy().is_none());
        assert!(owner.gmsh_provider_output().is_none());
        let bytes = owner.to_bytes().unwrap();
        let replay = AuthenticatedCommonMesh::from_bytes(&bytes).unwrap();
        assert_eq!(owner, replay);
        assert_eq!(owner.digest().unwrap(), replay.digest().unwrap());
        let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(wire["resources"]["mesh"]["kind"], "external-msh41");
        let text = String::from_utf8(bytes).unwrap();
        let wrong_origin = text.replace("\"kind\":\"external-msh41\"", "\"kind\":\"gmsh4152\"");
        assert_ne!(wrong_origin, text);
        assert!(AuthenticatedCommonMesh::from_bytes(wrong_origin.as_bytes()).is_err());
    }
}

#[test]
fn external_msh_rejects_foreign_geometry_and_both_dimension_work_limits() {
    assert!(
        AuthenticatedCommonMesh::import_msh41(square(2.), policy(4, 4096), SQUARE.to_vec())
            .is_err()
    );
    for (geometry, source) in [(square(1.), SQUARE), (tetrahedron(), TETRAHEDRON)] {
        assert!(
            AuthenticatedCommonMesh::import_msh41(
                geometry.clone(),
                policy(3, 4096),
                source.to_vec()
            )
            .is_err()
        );
        assert!(
            AuthenticatedCommonMesh::import_msh41(
                geometry,
                policy(4, source.len() - 1),
                source.to_vec()
            )
            .is_err()
        );
    }
}

#[test]
fn external_msh_replay_rejects_inconsistent_source_policy_and_resource_substitutions() {
    let owner = AuthenticatedCommonMesh::import_msh41(square(1.), policy(4, 4096), SQUARE.to_vec())
        .unwrap();
    let bytes = String::from_utf8(owner.to_bytes().unwrap()).unwrap();
    let wire: serde_json::Value = serde_json::from_str(&bytes).unwrap();
    let production = wire["resources"]["mesh"]["production_base64"]
        .as_str()
        .unwrap();
    let lineage = String::from_utf8(BASE64_STANDARD.decode(production).unwrap()).unwrap();
    for replacement in [
        lineage.replace("\"coordinate_unit\":\"m\"", "\"coordinate_unit\":\"mm\""),
        lineage.replace(
            "\"maximum_source_bytes\":4096",
            "\"maximum_source_bytes\":1",
        ),
    ] {
        assert_ne!(replacement, lineage);
        let changed = bytes.replace(production, &BASE64_STANDARD.encode(replacement));
        assert!(AuthenticatedCommonMesh::from_bytes(changed.as_bytes()).is_err());
    }
    let source = wire["resources"]["mesh"]["provider_output_base64"]
        .as_str()
        .unwrap();
    let changed_source = String::from_utf8(SQUARE.to_vec())
        .unwrap()
        .replace("1 1 2 3", "1 1 3 2");
    assert_ne!(changed_source.as_bytes(), SQUARE);
    assert!(
        AuthenticatedCommonMesh::from_bytes(
            bytes
                .replace(source, &BASE64_STANDARD.encode(changed_source))
                .as_bytes()
        )
        .is_err()
    );
    // Changing quality alone leaves an inconsistent retained Mesh quality gate.
    let new_policy = lineage.replace("\"minimum_mean_ratio\":0.01", "\"minimum_mean_ratio\":0.02");
    assert_ne!(new_policy, lineage);
    assert!(
        AuthenticatedCommonMesh::from_bytes(
            bytes
                .replace(production, &BASE64_STANDARD.encode(new_policy))
                .as_bytes()
        )
        .is_err()
    );
    // A coherent newly imported policy/resource occurrence is valid and distinct.
    let new_occurrence = AuthenticatedCommonMesh::import_msh41(
        square(1.),
        MshImportPolicyV1::metres(0.02, 4, 4096).unwrap(),
        SQUARE.to_vec(),
    )
    .unwrap();
    let new_bytes = new_occurrence.to_bytes().unwrap();
    assert_eq!(
        AuthenticatedCommonMesh::from_bytes(&new_bytes).unwrap(),
        new_occurrence
    );
    assert_ne!(new_occurrence.digest().unwrap(), owner.digest().unwrap());
    assert_eq!(
        new_occurrence.simplicial_mesh().unwrap().mesh().cells(),
        owner.simplicial_mesh().unwrap().mesh().cells()
    );
    let new_wire: serde_json::Value = serde_json::from_slice(&new_bytes).unwrap();
    let old_mesh = wire["resources"]["mesh"]["mesh_base64"].as_str().unwrap();
    let new_mesh = new_wire["resources"]["mesh"]["mesh_base64"]
        .as_str()
        .unwrap();
    assert_ne!(old_mesh, new_mesh);
    assert!(
        AuthenticatedCommonMesh::from_bytes(bytes.replace(old_mesh, new_mesh).as_bytes()).is_err()
    );
}

#[test]
fn external_msh_policy_deserialization_cannot_bypass_constructor_invariants() {
    for (quality, facets, bytes) in [(0., 4, 4096), (2., 4, 4096), (0.01, 0, 4096), (0.01, 4, 0)] {
        let raw = format!(
            r#"{{"coordinate_unit":"m","minimum_mean_ratio":{quality},"maximum_boundary_facets":{facets},"maximum_source_bytes":{bytes}}}"#
        );
        assert!(serde_json::from_str::<MshImportPolicyV1>(&raw).is_err());
    }
}

#[test]
fn external_msh_skew_preserves_original_cell_orientation_and_analytic_area() {
    use eqiora_meshing::{GeometryMap, MeshGeometry};
    // (x,y) -> (2x+y,3y), determinant 6; two unit-reference triangles
    // each therefore have physical area 3. No Eqiora-derived oracle is stored.
    let vertices = vec![[0., 0.], [2., 0.], [3., 3.], [1., 3.]];
    let geometry = CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            vertices.clone(),
            vec![PlanarFace::new(vec![0, 1, 2, 3], vec![])],
            vec![
                NamedEntitySet::new("body", 2, vec![0]),
                NamedEntitySet::new("outer", 1, vec![0, 1, 2, 3]),
            ],
            1e-12,
        )
        .unwrap(),
    )
    .unwrap();
    let source = String::from_utf8(SQUARE.to_vec())
        .unwrap()
        .replace("1 0 0\n1 1 0\n0 1 0", "2 0 0\n3 3 0\n1 3 0")
        .replace("1 1 2 3", "1 2 3 1");
    let owner =
        AuthenticatedCommonMesh::import_msh41(geometry, policy(4, 4096), source.into_bytes())
            .unwrap();
    let mesh = owner.simplicial_mesh().unwrap().mesh();
    assert_eq!(
        mesh.vertices(),
        vertices.iter().map(|v| v.to_vec()).collect::<Vec<_>>()
    );
    assert_eq!(mesh.cells(), [vec![1, 2, 0], vec![0, 2, 3]]);
    let mut area = 0.;
    for cell in 0..2 {
        let map = mesh.geometry_map(MeshEntity::new(2, cell)).unwrap();
        let mut jacobian = [0.; 4];
        map.jacobian_at(&[0.25, 0.25], &mut jacobian).unwrap();
        let determinant = jacobian[0] * jacobian[3] - jacobian[1] * jacobian[2];
        assert_eq!(determinant, 6.);
        area += determinant / 2.;
    }
    assert_eq!(area, 6.);
    assert_eq!(
        AuthenticatedCommonMesh::from_bytes(&owner.to_bytes().unwrap()).unwrap(),
        owner
    );
}

#[test]
fn archived_pre_import_v1_v4_encodings_and_planar_gmsh_acceptance_stay_unchanged() {
    let lineage = include_bytes!("fixtures/lineage-v1-gmsh4152.json");
    let production = MeshProductionLineageEnvelopeV1::from_json(lineage).unwrap();
    assert_eq!(production.canonical_json().unwrap(), lineage);
    assert_eq!(production.provider_identity(), "eqiora.gmsh-cli");
    assert!(production.msh_import_policy().is_none());
    for bytes in [
        include_bytes!("fixtures/commonmesh-v4-gmsh4152.json").as_slice(),
        include_bytes!("fixtures/commonmesh-v4-planar-over-facet-policy.json").as_slice(),
    ] {
        let owner = AuthenticatedCommonMesh::from_bytes(bytes).unwrap();
        assert_eq!(owner.to_bytes().unwrap(), bytes);
        assert_eq!(
            owner.production().unwrap().provider_identity(),
            "eqiora.gmsh-cli"
        );
        assert!(owner.msh_source_bytes().is_none());
    }
    let legacy = AuthenticatedCommonMesh::from_bytes(include_bytes!(
        "fixtures/commonmesh-v4-planar-over-facet-policy.json"
    ))
    .unwrap();
    assert_eq!(
        legacy
            .production()
            .unwrap()
            .gmsh_mesh_policy()
            .unwrap()
            .maximum_boundary_facets(),
        8
    );
    assert!(
        AuthenticatedCommonMesh::import_msh41(
            legacy.geometry().unwrap().clone(),
            policy(8, 4096),
            legacy.gmsh_provider_output().unwrap().to_vec()
        )
        .is_err()
    );
}

fn exact_rectangle() -> CanonicalGeometryV1 {
    exact_rectangle_width(1.)
}

fn exact_rectangle_width(width: f64) -> CanonicalGeometryV1 {
    let graph = eqiora_geometry::GeometryGraph::new();
    let face = graph.rectangle([0., width], [0., 1.]).unwrap();
    let sides = face.boundaries();
    graph
        .build(
            &face,
            &BTreeMap::from([
                ("body".to_owned(), vec![face.region().into()]),
                ("left".to_owned(), vec![sides[0].into()]),
                ("right".to_owned(), vec![sides[1].into()]),
                ("bottom".to_owned(), vec![sides[2].into()]),
                ("top".to_owned(), vec![sides[3].into()]),
            ]),
        )
        .unwrap()
}

#[test]
fn external_msh_existing_rectangle_keeps_identity_and_all_four_sides() {
    let geometry = exact_rectangle();
    assert!(geometry.region().is_none());
    let owner =
        AuthenticatedCommonMesh::import_msh41(geometry.clone(), policy(4, 4096), SQUARE.to_vec())
            .unwrap();
    assert_eq!(owner.geometry(), Some(&geometry));
    assert_eq!(
        owner.simplicial_mesh().unwrap().mesh().cells(),
        [vec![0, 1, 2], vec![0, 2, 3]]
    );
    let correspondence = owner.correspondence().unwrap();
    assert_eq!(
        correspondence
            .planar_entity_set_entities(&geometry, "body")
            .unwrap()
            .len(),
        2
    );
    for side in ["left", "right", "bottom", "top"] {
        let entities = correspondence
            .planar_entity_set_entities(&geometry, side)
            .unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].dimension(), 1);
    }
    let bytes = owner.to_bytes().unwrap();
    assert_eq!(
        AuthenticatedCommonMesh::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let encoded = wire["resources"]["mesh"]["correspondence_base64"]
        .as_str()
        .unwrap();
    let correspondence_json = String::from_utf8(BASE64_STANDARD.decode(encoded).unwrap()).unwrap();
    let wrong_source = correspondence_json.replace(
        "\"source\":\"planar-rectangle-v2-simplicial-import-v1\"",
        "\"source\":\"planar-rectangle-v2-affine-triangle-v1\"",
    );
    assert_ne!(wrong_source, correspondence_json);
    GeometryMeshCorrespondenceEnvelopeV1::from_json(
        wrong_source.as_bytes(),
        eqiora_artifact::GeometryDecoderLimits::default(),
    )
    .unwrap();
    let substituted = String::from_utf8(bytes.clone())
        .unwrap()
        .replace(encoded, &BASE64_STANDARD.encode(wrong_source));
    assert!(AuthenticatedCommonMesh::from_bytes(substituted.as_bytes()).is_err());

    assert!(
        AuthenticatedCommonMesh::gmsh_4152(
            geometry.clone(),
            eqiora_artifact::GmshMeshPolicyV1::explicit(1e-12, 0.01, 8, 5.).unwrap(),
            SQUARE.to_vec()
        )
        .is_err()
    );
    let legacy_policy = eqiora_artifact::AffineTriangleMeshPolicy::new([1, 1]).unwrap();
    let (mesh, legacy_correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_affine_triangles(&geometry, &legacy_policy)
            .unwrap();
    let wire: serde_json::Value =
        serde_json::from_slice(&legacy_correspondence.canonical_json().unwrap()).unwrap();
    assert_eq!(wire["source"], "planar-rectangle-v2-affine-triangle-v1");
    assert_eq!(mesh.mesh().cells(), [vec![0, 2, 3], vec![0, 3, 1]]);
}

#[test]
fn external_msh_exact_rectangle_rejects_near_boundary_outside_and_source_substitution() {
    let geometry = exact_rectangle();
    let owner =
        AuthenticatedCommonMesh::import_msh41(geometry.clone(), policy(4, 4096), SQUARE.to_vec())
            .unwrap();
    let original = String::from_utf8(SQUARE.to_vec()).unwrap();
    for changed in [
        original.replace("0 0 0\n1 0 0", "0.00000000000001 0 0\n1 0 0"),
        original.replace("0 0 0\n1 0 0", "-0.00000000000001 0 0\n1 0 0"),
    ] {
        assert_ne!(changed, original);
        assert!(
            AuthenticatedCommonMesh::import_msh41(
                geometry.clone(),
                policy(4, 4096),
                changed.clone().into_bytes()
            )
            .is_err()
        );
        let bytes = String::from_utf8(owner.to_bytes().unwrap()).unwrap();
        let changed_wire = bytes.replace(
            &BASE64_STANDARD.encode(SQUARE),
            &BASE64_STANDARD.encode(changed),
        );
        assert_ne!(changed_wire, bytes);
        assert!(AuthenticatedCommonMesh::from_bytes(changed_wire.as_bytes()).is_err());
    }
    let partial = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 3 1 3\n2 1 0 3\n1\n2\n3\n0 0 0\n1 0 0\n0 1 0\n$EndNodes\n$Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\n$EndElements\n";
    assert!(
        AuthenticatedCommonMesh::import_msh41(geometry, policy(4, 4096), partial.to_vec()).is_err()
    );
}

#[test]
fn external_msh_exact_rectangle_rejects_duplicate_geometric_coverage() {
    // Two disconnected copies have valid positive/manifold simplex topology,
    // but duplicate every rectangle-side interval. They cannot cover it once.
    let duplicate = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 8 1 8\n2 1 0 8\n1\n2\n3\n4\n5\n6\n7\n8\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n$Elements\n1 4 1 4\n2 1 2 4\n1 1 2 3\n2 1 3 4\n3 5 6 7\n4 5 7 8\n$EndElements\n";
    let parsed = eqiora_io_gmsh::import_msh41(
        duplicate,
        eqiora_io_gmsh::Msh41Policy::mesh(2, eqiora_meshing::MeshQualityGate::new(0.01).unwrap())
            .unwrap(),
        |_, _, _| {},
    )
    .unwrap();
    assert_eq!(parsed.cells().len(), 4);
    let failure = AuthenticatedCommonMesh::import_msh41(
        exact_rectangle(),
        policy(8, 4096),
        duplicate.to_vec(),
    )
    .unwrap_err();
    assert!(failure.message().contains("cover"));
}

#[test]
fn external_msh_existing_rectangle_executes_existing_affine_p1_field() {
    let geometry = exact_rectangle();
    let body = geometry.entity_set("body").unwrap();
    let mut supports = vec![("body", body, None)];
    supports.extend(["left", "right", "bottom", "top"].map(|side| {
        (
            side,
            geometry.entity_set(side).unwrap(),
            Some(("body", body)),
        )
    }));
    let model = compile_model(
        "rectangle-import-p1.eqi",
        r#"
public component Affine(support body:volume(ambient_dimension=2), support left:boundary(parent=body), support right:boundary(parent=body), support bottom:boundary(parent=body), support top:boundary(parent=body)) {
    variable q:m on body in smooth;
    relation balance on body { -div(grad(q))=0[1/m]; }
    relation l on left { trace(q)=coordinate(0)+2*coordinate(1); }
    relation r on right { trace(q)=coordinate(0)+2*coordinate(1); }
    relation b on bottom { trace(q)=coordinate(0)+2*coordinate(1); }
    relation t on top { trace(q)=coordinate(0)+2*coordinate(1); }
}
"#,
        &geometry,
        "Affine",
        &supports,
        &[],
    );
    let source = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 5 1 5\n2 1 0 5\n1\n2\n3\n4\n5\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0.5 0.5 0\n$EndNodes\n$Elements\n1 4 1 4\n2 1 2 4\n1 1 2 5\n2 2 3 5\n3 3 4 5\n4 4 1 5\n$EndElements\n";
    let owner =
        AuthenticatedCommonMesh::import_msh41(geometry, policy(4, 4096), source.to_vec()).unwrap();
    let foreign_source = String::from_utf8(source.to_vec())
        .unwrap()
        .replace("1 0 0\n1 1 0", "2 0 0\n2 1 0")
        .replace("0.5 0.5 0", "1 0.5 0");
    let foreign = AuthenticatedCommonMesh::import_msh41(
        exact_rectangle_width(2.),
        policy(4, 4096),
        foreign_source.into_bytes(),
    )
    .unwrap();
    let rejected = ResolvedCommonPlan::resolve(
        &model,
        foreign,
        CommonSpatialPolicy::P1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-11,
            1e-13,
            NonZeroUsize::new(1000).unwrap(),
        )),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
    )
    .unwrap_err();
    assert!(rejected.message().contains("foreign Geometry"));
    for mesh in [
        owner.clone(),
        AuthenticatedCommonMesh::from_bytes(&owner.to_bytes().unwrap()).unwrap(),
    ] {
        let plan = ResolvedCommonPlan::resolve(
            &model,
            mesh,
            CommonSpatialPolicy::P1,
            CommonSolvePolicy::Linear(exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-11,
                1e-13,
                NonZeroUsize::new(1000).unwrap(),
            )),
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
            None,
        )
        .unwrap();
        let result = plan
            .as_spatial()
            .unwrap()
            .run_result(None, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let (association, values, shape) = result.field_block(0, 0).unwrap();
        assert_eq!(association, "vertex");
        assert_eq!(shape, &[5]);
        for (value, point) in values
            .iter()
            .zip(owner.simplicial_mesh().unwrap().mesh().vertices())
        {
            assert!((value - (point[0] + 2. * point[1])).abs() < 1e-10);
        }
    }
}

fn hidden_interior_double_cover_source() -> &'static [u8] {
    b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 8 1 8\n2 1 0 8\n1\n2\n3\n4\n5\n6\n7\n8\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0.25 0.25 0\n0.75 0.25 0\n0.75 0.75 0\n0.25 0.75 0\n$EndNodes\n$Elements\n1 6 1 6\n2 1 2 6\n1 1 2 3\n2 1 3 4\n3 5 6 7\n4 5 7 8\n5 5 6 8\n6 6 7 8\n$EndElements\n"
}

#[test]
fn external_msh_rectangle_rejects_hidden_positive_interior_double_cover() {
    // The external square boundary is correct. A disconnected interior K4
    // has four positive cells and two parents on every edge, but its induced
    // orientations do not cancel. Boundary coverage alone cannot detect it.
    let source = hidden_interior_double_cover_source();
    let parsed = eqiora_io_gmsh::import_msh41(
        source,
        eqiora_io_gmsh::Msh41Policy::mesh(2, eqiora_meshing::MeshQualityGate::new(0.01).unwrap())
            .unwrap(),
        |_, _, _| {},
    )
    .unwrap();
    assert_eq!(parsed.cells().len(), 6);
    assert_eq!(
        (0..parsed.entity_count(1).unwrap())
            .filter(|&i| parsed.is_boundary_entity(MeshEntity::new(1, i)) == Some(true))
            .count(),
        4
    );
    let failure =
        AuthenticatedCommonMesh::import_msh41(exact_rectangle(), policy(4, 4096), source.to_vec())
            .unwrap_err();
    assert!(failure.message().contains("cancel"));
}

#[test]
fn external_msh_generic_planar_rejects_hidden_interior_double_cover() {
    let source = hidden_interior_double_cover_source();
    let parsed = eqiora_io_gmsh::import_msh41(
        source,
        eqiora_io_gmsh::Msh41Policy::mesh(2, eqiora_meshing::MeshQualityGate::new(0.01).unwrap())
            .unwrap(),
        |_, _, _| {},
    )
    .unwrap();
    assert_eq!(parsed.cells().len(), 6);
    let failure =
        AuthenticatedCommonMesh::import_msh41(square(1.), policy(4, 4096), source.to_vec())
            .unwrap_err();
    assert!(failure.message().contains("cancel"));
}
