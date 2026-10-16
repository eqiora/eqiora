use std::collections::BTreeMap;

use eqiora_geometry::{GeometryGraph, PlanarTopologyHandle};

use super::*;
use crate::MeshProductionLineageEnvelopeV1;

fn geometry() -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let left = graph.rectangle([0.0, 1.0], [0.0, 1.0]).unwrap();
    let right = graph.rectangle([1.0, 2.0], [0.0, 1.0]).unwrap();
    let left_edges = left.boundaries();
    let right_edges = right.boundaries();
    let partition = graph
        .partition(&left, &right, [left_edges[1], right_edges[0]])
        .unwrap();
    graph
        .build(
            &partition,
            &BTreeMap::from([
                (
                    "fluid".to_owned(),
                    vec![PlanarTopologyHandle::from(left.region())],
                ),
                ("solid".to_owned(), vec![right.region().into()]),
                (
                    "interface".to_owned(),
                    vec![left_edges[1].into(), right_edges[0].into()],
                ),
                ("inlet".to_owned(), vec![left_edges[0].into()]),
                ("outlet".to_owned(), vec![right_edges[1].into()]),
                (
                    "walls".to_owned(),
                    vec![
                        left_edges[2].into(),
                        left_edges[3].into(),
                        right_edges[2].into(),
                        right_edges[3].into(),
                    ],
                ),
            ]),
        )
        .unwrap()
}

#[test]
fn refinement_preserves_both_parent_relative_sides_of_an_internal_interface() {
    let geometry = geometry();
    let mut policy = AffineTriangleMeshPolicy::new([2, 2]).unwrap();
    let (mut mesh, _) =
        GeometryMeshCorrespondenceEnvelopeV1::from_affine_triangles(&geometry, &policy).unwrap();
    for _ in 0..4 {
        policy = policy
            .with_refinement((0..mesh.mesh().cells().len()).collect(), 1024)
            .unwrap();
        (mesh, _) = GeometryMeshCorrespondenceEnvelopeV1::from_affine_triangles(&geometry, &policy)
            .unwrap();
    }
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_affine_triangles(&geometry, &policy).unwrap();
    let WireCorrespondenceV1::AuthoredRegion(wire) = &correspondence.wire else {
        panic!("region correspondence")
    };
    let left = &wire.frontiers[1];
    let right = &wire.frontiers[7];
    assert!(left.facet_indices.len() > 2);
    assert_eq!(left.facet_indices, right.facet_indices);
    assert!(
        left.parent_outward
            .iter()
            .zip(&right.parent_outward)
            .all(|(left, right)| left != right)
    );
    for face in &wire.faces {
        for &cell in &face.cell_indices {
            for &vertex in &mesh.mesh().cells()[cell as usize] {
                let x = mesh.mesh().vertices()[vertex][0];
                assert!(if face.geometry_face == 0 {
                    x <= 1.0
                } else {
                    x >= 1.0
                });
            }
        }
    }
    // Independent coordinate/length check: each side covers x=1, 0<=y<=1 exactly.
    let mut length = 0.0;
    for (&facet, &outward) in left.facet_indices.iter().zip(&left.parent_outward) {
        let vertices = mesh
            .mesh()
            .entity_vertices(MeshEntity::new(1, facet as usize))
            .unwrap();
        let a = &mesh.mesh().vertices()[vertices[0].index()];
        let b = &mesh.mesh().vertices()[vertices[1].index()];
        assert_eq!((a[0], b[0]), (1.0, 1.0));
        assert_eq!(
            outward,
            if b[1] > a[1] {
                WireFacetOutward::RightOfCanonicalFacet
            } else {
                WireFacetOutward::LeftOfCanonicalFacet
            }
        );
        length += (b[1] - a[1]).abs();
    }
    assert_eq!(length, 1.0);
    let production = MeshProductionLineageEnvelopeV1::from_affine_triangle_resources(
        &policy,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    production
        .validate_against_affine_triangle_resources(&policy, &geometry, &mesh, &correspondence)
        .unwrap();
    let mut changed = correspondence.clone();
    let WireCorrespondenceV1::AuthoredRegion(wire) = &mut changed.wire else {
        unreachable!()
    };
    wire.frontiers[1].parent_outward[0] = wire.frontiers[7].parent_outward[0];
    assert!(
        changed
            .validate_affine_triangle_policy(&geometry, &mesh, &policy)
            .is_err()
    );
}

#[test]
fn exact_partition_replays_mesh_correspondence_and_parent_relative_interface() {
    let geometry = geometry();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_adjacent_rectangle_partition_affine_triangles(
            &geometry,
            [2, 2],
        )
        .unwrap();
    assert_eq!(
        mesh.mesh().vertices(),
        &[
            vec![0.0, 0.0],
            vec![0.0, 0.5],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 0.5],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![2.0, 0.5],
            vec![2.0, 1.0],
        ]
    );
    assert_eq!(
        mesh.mesh().cells(),
        &[
            vec![0, 3, 4],
            vec![0, 4, 1],
            vec![1, 4, 5],
            vec![1, 5, 2],
            vec![3, 6, 7],
            vec![3, 7, 4],
            vec![4, 7, 8],
            vec![4, 8, 5],
        ]
    );
    assert_eq!(
        correspondence
            .adjacent_rectangle_partition_entity_set_entities(&geometry, "fluid")
            .unwrap()
            .len(),
        4
    );
    let wire: serde_json::Value =
        serde_json::from_slice(&correspondence.canonical_json().unwrap()).unwrap();
    assert_eq!(wire["source"], ADJACENT_PARTITION_SOURCE);
    assert_eq!(
        wire["frontiers"][1]["facet_indices"],
        serde_json::json!([7, 10])
    );
    assert_eq!(
        wire["frontiers"][7]["facet_indices"],
        serde_json::json!([7, 10])
    );
    assert_eq!(
        wire["frontiers"][1]["parent_outward"],
        serde_json::json!(["right-of-canonical-facet", "right-of-canonical-facet"])
    );
    assert_eq!(
        wire["frontiers"][7]["parent_outward"],
        serde_json::json!(["left-of-canonical-facet", "left-of-canonical-facet"])
    );
    assert_eq!(
        correspondence
            .adjacent_rectangle_partition_entity_set_entities(&geometry, "solid")
            .unwrap()
            .len(),
        4
    );
    assert_eq!(
        correspondence
            .adjacent_rectangle_partition_entity_set_entities(&geometry, "interface")
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        correspondence
            .adjacent_rectangle_partition_entity_set_entities(&geometry, "walls")
            .unwrap()
            .len(),
        4
    );
    correspondence
        .validate_affine_triangle_policy(
            &geometry,
            &mesh,
            &AffineTriangleMeshPolicy::new([2, 2]).unwrap(),
        )
        .unwrap();
    let policy = AffineTriangleMeshPolicy::new([2, 2]).unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_affine_triangle_resources(
        &policy,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    production
        .validate_against_affine_triangle_resources(&policy, &geometry, &mesh, &correspondence)
        .unwrap();
    assert!(
        GeometryMeshCorrespondenceEnvelopeV1::from_adjacent_rectangle_partition_affine_triangles(
            &geometry,
            [1, 2]
        )
        .is_err()
    );

    let mut changed = mesh.canonical_json().unwrap();
    let position = changed
        .windows(3)
        .position(|window| window == b"0.5")
        .unwrap();
    changed.splice(position..position + 3, b"0.6".iter().copied());
    let changed = SimplicialMeshEnvelopeV1::from_json(&changed, Default::default());
    assert!(changed.is_err());

    let mut changed_correspondence = correspondence.clone();
    let WireCorrespondenceV1::AuthoredRegion(wire) = &mut changed_correspondence.wire else {
        panic!("partition correspondence must use the authored-region wire")
    };
    wire.mesh_sha256 = "00".repeat(32);
    assert!(
        changed_correspondence
            .validate_affine_triangle_policy(
                &geometry,
                &mesh,
                &AffineTriangleMeshPolicy::new([2, 2]).unwrap()
            )
            .is_err()
    );
    assert!(
        production
            .validate_against_affine_triangle_resources(
                &policy,
                &geometry,
                &mesh,
                &changed_correspondence,
            )
            .is_err()
    );
}
