use std::collections::BTreeMap;

use eqiora_geometry::{GeometryGraph, PlanarTopologyHandle};
use eqiora_meshing::EntityIncidence;

use super::*;
use crate::{CartesianMeshCellsV2, MeshProductionLineageEnvelopeV1};

fn rectangle() -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 2.0], [-1.0, 2.0]).unwrap();
    let edges = rectangle.boundaries();
    graph
        .build(
            &rectangle,
            &BTreeMap::from([
                (
                    "region".to_owned(),
                    vec![PlanarTopologyHandle::from(rectangle.region())],
                ),
                ("left".to_owned(), vec![edges[0].into()]),
                ("right".to_owned(), vec![edges[1].into()]),
                ("bottom".to_owned(), vec![edges[2].into()]),
                ("top".to_owned(), vec![edges[3].into()]),
            ]),
        )
        .unwrap()
}

fn entity_indices(entities: Vec<MeshEntity>) -> Vec<usize> {
    entities.into_iter().map(MeshEntity::index).collect()
}

fn non_rectangle() -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 2.0], [-1.0, 2.0]).unwrap();
    let circle = graph.circle([1.0, 0.5], 0.25).unwrap();
    let cut = graph.subtract(&rectangle, &circle).unwrap();
    let outer = rectangle.boundaries();
    let hole = circle.boundaries();
    graph
        .build(
            &cut,
            &BTreeMap::from([
                ("region".to_owned(), vec![cut.region().into()]),
                ("left".to_owned(), vec![outer[0].into()]),
                ("right".to_owned(), vec![outer[1].into()]),
                ("walls".to_owned(), vec![outer[2].into(), outer[3].into()]),
                ("hole".to_owned(), vec![hole[0].into()]),
            ]),
        )
        .unwrap()
}

#[derive(Clone, Copy)]
enum TopologyMutation {
    Connectivity,
    LocalOrdinal,
    Orientation,
}

struct MutatedTopology<'a> {
    native: &'a CartesianMesh,
    mutation: TopologyMutation,
}

impl RectangleCartesianTopology for MutatedTopology<'_> {
    fn entity_count(&self, dimension: usize) -> Option<usize> {
        MeshTopology::entity_count(self.native, dimension)
    }

    fn axis_cell_count(&self, axis: usize) -> Option<usize> {
        self.native.axis_cell_count(axis)
    }

    fn incidence(
        &self,
        entity: MeshEntity,
        target_dimension: usize,
    ) -> Option<Vec<EntityIncidence>> {
        let mut incidence = MeshTopology::incidence(self.native, entity, target_dimension)?;
        if entity == MeshEntity::new(1, 8) && target_dimension == 2 {
            match self.mutation {
                TopologyMutation::LocalOrdinal => incidence[0].local_ordinal = 0,
                TopologyMutation::Orientation => incidence[0].orientation = OrientationCode::new(1),
                TopologyMutation::Connectivity => {}
            }
        }
        Some(incidence)
    }

    fn entity_vertices(&self, entity: MeshEntity) -> Option<Vec<MeshEntity>> {
        let source = if matches!(self.mutation, TopologyMutation::Connectivity)
            && entity == MeshEntity::new(1, 8)
        {
            MeshEntity::new(1, 11)
        } else {
            entity
        };
        self.native.entity_vertices(source)
    }

    fn vertex_multi_index(&self, vertex: MeshEntity) -> Option<&[usize]> {
        self.native.vertex_multi_index(vertex)
    }
}

fn mutate_wire(
    correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    mutation: impl FnOnce(&mut WirePlanarRectangleV2CorrespondenceV1),
) -> GeometryMeshCorrespondenceEnvelopeV1 {
    let mut mutated = correspondence.clone();
    let WireCorrespondenceV1::PlanarRectangleV2(wire) = &mut mutated.wire else {
        panic!("expected planar rectangle correspondence")
    };
    mutation(wire);
    mutated
}

#[test]
fn rectangle_cartesian_resources_have_analytic_counts_and_direct_membership() {
    let geometry = rectangle();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(&geometry, [2, 3])
            .unwrap();
    let native = mesh.mesh();
    assert_eq!(native.axis_coordinates(0), Some(&[0.0, 1.0, 2.0][..]));
    assert_eq!(native.axis_coordinates(1), Some(&[-1.0, 0.0, 1.0, 2.0][..]));
    let expected_coordinates = [
        [0.0, -1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, -1.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [2.0, 2.0],
    ];
    for (vertex, expected) in expected_coordinates.into_iter().enumerate() {
        assert_eq!(
            native.vertex_coordinates(MeshEntity::new(0, vertex)),
            Some(expected.to_vec())
        );
    }
    let expected_cells = [
        [0, 4, 1, 5],
        [1, 5, 2, 6],
        [2, 6, 3, 7],
        [4, 8, 5, 9],
        [5, 9, 6, 10],
        [6, 10, 7, 11],
    ];
    for (cell, expected) in expected_cells.into_iter().enumerate() {
        assert_eq!(
            entity_indices(native.entity_vertices(MeshEntity::new(2, cell)).unwrap()),
            expected
        );
    }
    let expected_facets = [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 8],
        [5, 9],
        [6, 10],
        [7, 11],
        [0, 1],
        [1, 2],
        [2, 3],
        [4, 5],
        [5, 6],
        [6, 7],
        [8, 9],
        [9, 10],
        [10, 11],
    ];
    for (facet, expected) in expected_facets.into_iter().enumerate() {
        assert_eq!(
            entity_indices(native.entity_vertices(MeshEntity::new(1, facet)).unwrap()),
            expected
        );
    }
    assert_eq!(
        ["left", "right", "bottom", "top", "region"].map(|name| entity_indices(
            correspondence
                .planar_rectangle_v2_entity_set_entities(&geometry, name)
                .unwrap()
        )),
        [
            vec![8, 9, 10],
            vec![14, 15, 16],
            vec![0, 4],
            vec![3, 7],
            vec![0, 1, 2, 3, 4, 5],
        ]
    );
    let expected_boundary_incidence = [
        (0, 0, 0),
        (4, 3, 0),
        (3, 2, 1),
        (7, 5, 1),
        (8, 0, 2),
        (9, 1, 2),
        (10, 2, 2),
        (14, 3, 3),
        (15, 4, 3),
        (16, 5, 3),
    ];
    for (facet, cell, local_ordinal) in expected_boundary_incidence {
        assert_eq!(
            MeshTopology::incidence(native, MeshEntity::new(1, facet), 2),
            Some(vec![EntityIncidence {
                entity: MeshEntity::new(2, cell),
                local_ordinal,
                orientation: OrientationCode::identity(),
            }])
        );
    }
    let expected_boundary = BTreeSet::from([0, 3, 4, 7, 8, 9, 10, 14, 15, 16]);
    let expected_interior = BTreeSet::from([1, 2, 5, 6, 11, 12, 13]);
    let actual_boundary = (0..17)
        .filter(|&facet| {
            MeshTopology::incidence(native, MeshEntity::new(1, facet), 2)
                .is_some_and(|parents| parents.len() == 1)
        })
        .collect::<BTreeSet<_>>();
    let actual_interior = (0..17)
        .filter(|&facet| {
            MeshTopology::incidence(native, MeshEntity::new(1, facet), 2)
                .is_some_and(|parents| parents.len() == 2)
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(actual_boundary, expected_boundary);
    assert_eq!(actual_interior, expected_interior);
    assert!(actual_boundary.is_disjoint(&actual_interior));
    assert_eq!(actual_boundary.len() + actual_interior.len(), 17);

    let WireCorrespondenceV1::PlanarRectangleV2(wire) = &correspondence.wire else {
        panic!("expected planar rectangle correspondence")
    };
    assert_eq!(wire.face.cell_indices, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(
        wire.frontiers
            .iter()
            .map(|frontier| frontier.facet_indices.clone())
            .collect::<Vec<_>>(),
        [vec![8, 9, 10], vec![14, 15, 16], vec![0, 4], vec![3, 7]]
    );
    correspondence
        .validate_against_planar_rectangle_v2_cartesian(&geometry, &mesh, [2, 3])
        .unwrap();
    assert!(
        correspondence
            .validate_against_planar_rectangle_v2_cartesian(&geometry, &mesh, [3, 2])
            .is_err()
    );
    let (replayed_mesh, replayed_correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(&geometry, [2, 3])
            .unwrap();
    assert_eq!(replayed_mesh, mesh);
    assert_eq!(replayed_correspondence, correspondence);
}

#[test]
fn rectangle_correspondence_rejects_wire_and_resource_mutations() {
    let geometry = rectangle();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(&geometry, [2, 3])
            .unwrap();
    let bytes = correspondence.canonical_json().unwrap();
    assert_eq!(
        GeometryMeshCorrespondenceEnvelopeV1::from_json(&bytes, Default::default()).unwrap(),
        correspondence
    );
    for mutation in [
        |value: &mut serde_json::Value| value["source"] = "other".into(),
        |value: &mut serde_json::Value| value["dimension"] = 3.into(),
        |value: &mut serde_json::Value| {
            value["frontiers"][0]["facet_indices"] = serde_json::json!([])
        },
        |value: &mut serde_json::Value| value["frontiers"][1]["geometry_edge"] = 0.into(),
    ] {
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        mutation(&mut value);
        assert!(
            GeometryMeshCorrespondenceEnvelopeV1::from_json(
                &serde_json::to_vec(&value).unwrap(),
                Default::default(),
            )
            .is_err()
        );
    }
    for swapped in [
        mutate_wire(&correspondence, |wire| {
            let left = wire.frontiers[0].facet_indices.clone();
            wire.frontiers[0].facet_indices = wire.frontiers[1].facet_indices.clone();
            wire.frontiers[1].facet_indices = left;
        }),
        mutate_wire(&correspondence, |wire| {
            let bottom = wire.frontiers[2].facet_indices.clone();
            wire.frontiers[2].facet_indices = wire.frontiers[3].facet_indices.clone();
            wire.frontiers[3].facet_indices = bottom;
        }),
        mutate_wire(&correspondence, |wire| {
            wire.frontiers[0].facet_indices = vec![8, 9, 11];
        }),
    ] {
        swapped.validate_local(Default::default()).unwrap();
        assert!(
            swapped
                .validate_against_planar_rectangle_v2_cartesian(&geometry, &mesh, [2, 3])
                .is_err()
        );
    }
    for mutation in [
        TopologyMutation::Connectivity,
        TopologyMutation::LocalOrdinal,
        TopologyMutation::Orientation,
    ] {
        assert!(
            topology_assignments(&MutatedTopology {
                native: mesh.mesh(),
                mutation,
            })
            .is_err()
        );
    }
    let alternate_geometry = {
        let graph = GeometryGraph::new();
        let rectangle = graph.rectangle([0.0, 3.0], [-1.0, 2.0]).unwrap();
        let edges = rectangle.boundaries();
        graph
            .build(
                &rectangle,
                &BTreeMap::from([
                    ("region".to_owned(), vec![rectangle.region().into()]),
                    ("left".to_owned(), vec![edges[0].into()]),
                    ("right".to_owned(), vec![edges[1].into()]),
                    ("bottom".to_owned(), vec![edges[2].into()]),
                    ("top".to_owned(), vec![edges[3].into()]),
                ]),
            )
            .unwrap()
    };
    assert!(
        correspondence
            .validate_against_planar_rectangle_v2_cartesian(&alternate_geometry, &mesh, [2, 3])
            .is_err()
    );
    assert!(
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(
            &non_rectangle(),
            [2, 3],
        )
        .is_err()
    );
    for invalid_cells in [[0, 3], [2, 0], [usize::MAX, 3], [2, usize::MAX]] {
        assert!(
            GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(
                &geometry,
                invalid_cells,
            )
            .is_err()
        );
    }
    let (foreign_mesh, foreign_correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(&geometry, [3, 2])
            .unwrap();
    assert!(
        correspondence
            .validate_against_planar_rectangle_v2_cartesian(&geometry, &foreign_mesh, [2, 3])
            .is_err()
    );
    assert!(
        foreign_correspondence
            .validate_against_planar_rectangle_v2_cartesian(&geometry, &mesh, [2, 3])
            .is_err()
    );
    let cells = CartesianMeshCellsV2::new([2, 3]).unwrap();
    let lineage = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &cells,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let foreign_cells = CartesianMeshCellsV2::new([3, 2]).unwrap();
    let foreign_lineage = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &foreign_cells,
        &geometry,
        &foreign_mesh,
        &foreign_correspondence,
    )
    .unwrap();
    lineage
        .validate_against_structured_cartesian_v2_resources(
            &cells,
            &geometry,
            &mesh,
            &correspondence,
        )
        .unwrap();
    assert!(
        foreign_lineage
            .validate_against_structured_cartesian_v2_resources(
                &cells,
                &geometry,
                &mesh,
                &correspondence,
            )
            .is_err()
    );
}

#[test]
fn registered_rectangle_cartesian_common_mesh_evidence() {
    rectangle_cartesian_resources_have_analytic_counts_and_direct_membership();
    rectangle_correspondence_rejects_wire_and_resource_mutations();
}

#[test]
fn rectangle_affine_triangle_resources_are_exact_and_fail_closed() {
    let geometry = rectangle();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_affine_triangles(
            &geometry,
            [2, 3],
        )
        .unwrap();
    let native = mesh.mesh();
    assert_eq!(native.vertices().len(), 12);
    assert_eq!(native.cells().len(), 12);
    assert_eq!(native.entity_count(1), Some(23));
    assert_eq!(native.vertices()[0], [0.0, -1.0]);
    assert_eq!(native.vertices()[11], [2.0, 2.0]);
    assert_eq!(
        native.cells(),
        &[
            vec![0, 4, 5],
            vec![0, 5, 1],
            vec![1, 5, 6],
            vec![1, 6, 2],
            vec![2, 6, 7],
            vec![2, 7, 3],
            vec![4, 8, 9],
            vec![4, 9, 5],
            vec![5, 9, 10],
            vec![5, 10, 6],
            vec![6, 10, 11],
            vec![6, 11, 7],
        ]
    );
    assert_eq!(
        ["left", "right", "bottom", "top", "region"].map(|name| {
            correspondence
                .planar_rectangle_v2_entity_set_entities(&geometry, name)
                .unwrap()
                .len()
        }),
        [3, 3, 2, 2, 12]
    );
    let boundary = (0..native.entity_count(1).unwrap())
        .filter(|&index| {
            native
                .is_boundary_entity(MeshEntity::new(1, index))
                .unwrap()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(boundary.len(), 10);

    let mesh_bytes = mesh.canonical_json().unwrap();
    assert_eq!(
        SimplicialMeshEnvelopeV1::from_json(&mesh_bytes, MeshDecoderLimits::default()).unwrap(),
        mesh
    );
    let correspondence_bytes = correspondence.canonical_json().unwrap();
    assert_eq!(
        GeometryMeshCorrespondenceEnvelopeV1::from_json(
            &correspondence_bytes,
            GeometryDecoderLimits::default(),
        )
        .unwrap(),
        correspondence
    );
    correspondence
        .validate_affine_triangle_policy(
            &geometry,
            &mesh,
            &AffineTriangleMeshPolicy::new([2, 3]).unwrap(),
        )
        .unwrap();
    let (replayed_mesh, replayed_correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_affine_triangles(
            &geometry,
            [2, 3],
        )
        .unwrap();
    assert_eq!(replayed_mesh, mesh);
    assert_eq!(replayed_correspondence, correspondence);

    let policy = AffineTriangleMeshPolicy::new([2, 3]).unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_affine_triangle_resources(
        &policy,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let production_json: serde_json::Value =
        serde_json::from_slice(&production.canonical_json().unwrap()).unwrap();
    assert_eq!(
        production_json["provider"],
        serde_json::json!({
            "identity": "eqiora.affine-triangle",
            "version": "2"
        })
    );
    assert_eq!(
        production_json["effective_policy"],
        serde_json::json!({
            "kind": "affine-triangle",
            "cells": [2, 3],
            "diagonal": "lower-left-to-upper-right",
            "refinements": []
        })
    );
    production
        .validate_against_affine_triangle_resources(&policy, &geometry, &mesh, &correspondence)
        .unwrap();

    let (foreign_mesh, foreign_correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_affine_triangles(
            &geometry,
            [3, 2],
        )
        .unwrap();
    assert!(
        correspondence
            .validate_affine_triangle_policy(
                &geometry,
                &foreign_mesh,
                &AffineTriangleMeshPolicy::new([2, 3]).unwrap()
            )
            .is_err()
    );
    assert!(
        foreign_correspondence
            .validate_affine_triangle_policy(
                &geometry,
                &mesh,
                &AffineTriangleMeshPolicy::new([2, 3]).unwrap()
            )
            .is_err()
    );
    assert!(
        correspondence
            .validate_affine_triangle_policy(
                &rectangle_with_xmax(3.0),
                &mesh,
                &AffineTriangleMeshPolicy::new([2, 3]).unwrap()
            )
            .is_err()
    );
    assert!(
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_affine_triangles(
            &non_rectangle(),
            [2, 3],
        )
        .is_err()
    );
    for invalid_cells in [[0, 3], [2, 0], [usize::MAX, 3], [2, usize::MAX]] {
        assert!(
            GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_affine_triangles(
                &geometry,
                invalid_cells,
            )
            .is_err()
        );
    }

    let mut changed_vertices = native.vertices().to_vec();
    changed_vertices[5][0] += 0.125;
    let changed_coordinates = SimplicialMesh::new(
        2,
        changed_vertices,
        native.cells().to_vec(),
        native.quality_gate(),
    )
    .and_then(|mesh| SimplicialMeshEnvelopeV1::from_mesh(&mesh))
    .unwrap();
    assert!(
        correspondence
            .validate_affine_triangle_policy(
                &geometry,
                &changed_coordinates,
                &AffineTriangleMeshPolicy::new([2, 3]).unwrap()
            )
            .is_err()
    );

    let mut reordered_cells = native.cells().to_vec();
    reordered_cells.swap(0, 1);
    let reordered = SimplicialMesh::new(
        2,
        native.vertices().to_vec(),
        reordered_cells,
        native.quality_gate(),
    )
    .and_then(|mesh| SimplicialMeshEnvelopeV1::from_mesh(&mesh))
    .unwrap();
    assert!(
        correspondence
            .validate_affine_triangle_policy(
                &geometry,
                &reordered,
                &AffineTriangleMeshPolicy::new([2, 3]).unwrap()
            )
            .is_err()
    );

    let mut mutated: serde_json::Value = serde_json::from_slice(&correspondence_bytes).unwrap();
    mutated["frontiers"].as_array_mut().unwrap().swap(0, 1);
    let mutated = GeometryMeshCorrespondenceEnvelopeV1::from_json(
        &serde_json::to_vec(&mutated).unwrap(),
        GeometryDecoderLimits::default(),
    );
    assert!(mutated.is_err());
}

fn rectangle_with_xmax(xmax: f64) -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, xmax], [-1.0, 2.0]).unwrap();
    let edges = rectangle.boundaries();
    graph
        .build(
            &rectangle,
            &BTreeMap::from([
                ("region".to_owned(), vec![rectangle.region().into()]),
                ("left".to_owned(), vec![edges[0].into()]),
                ("right".to_owned(), vec![edges[1].into()]),
                ("bottom".to_owned(), vec![edges[2].into()]),
                ("top".to_owned(), vec![edges[3].into()]),
            ]),
        )
        .unwrap()
}

#[test]
fn local_refinement_history_replays_exact_boundary_membership_and_rejects_crosswires() {
    // Square cells: first split the lower-left diagonal, then child 1's bottom edge.
    // Cyclic longest-edge labeling puts the right-edge child first.
    // Eight seed triangles become ten and then eleven, with two new midpoints.
    let geometry = rectangle_with_xmax(3.0);
    let seed = AffineTriangleMeshPolicy::new([2, 2]).unwrap();
    let policy = seed
        .clone()
        .with_refinement(vec![0, 0], 10)
        .unwrap()
        .with_refinement(vec![1], 11)
        .unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_affine_triangles(&geometry, &policy).unwrap();
    assert_eq!(mesh.mesh().cells().len(), 11);
    assert_eq!(
        &mesh.mesh().vertices()[9..],
        &[vec![0.75, -0.25], vec![0.75, -1.0]]
    );
    for (name, count) in [
        ("region", 11),
        ("bottom", 3),
        ("top", 2),
        ("left", 2),
        ("right", 2),
    ] {
        assert_eq!(
            correspondence
                .planar_rectangle_v2_entity_set_entities(&geometry, name)
                .unwrap()
                .len(),
            count
        );
    }
    let production = MeshProductionLineageEnvelopeV1::from_affine_triangle_resources(
        &policy,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let reopened =
        MeshProductionLineageEnvelopeV1::from_json(&production.canonical_json().unwrap()).unwrap();
    assert_eq!(reopened.affine_triangle_policy().unwrap(), policy);
    reopened
        .validate_against_affine_triangle_resources(&policy, &geometry, &mesh, &correspondence)
        .unwrap();
    let (seed_mesh, seed_correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_affine_triangles(&geometry, &seed).unwrap();
    assert!(
        reopened
            .validate_against_affine_triangle_resources(
                &policy,
                &geometry,
                &seed_mesh,
                &seed_correspondence
            )
            .is_err()
    );
    assert!(
        reopened
            .validate_against_affine_triangle_resources(&seed, &geometry, &mesh, &correspondence)
            .is_err()
    );
    for invalid in [
        seed.clone().with_refinement(vec![8], 20).unwrap(),
        seed.with_refinement(vec![0], 9).unwrap(),
    ] {
        assert!(
            GeometryMeshCorrespondenceEnvelopeV1::from_affine_triangles(&geometry, &invalid)
                .is_err()
        );
    }
    let mut changed: serde_json::Value =
        serde_json::from_slice(&correspondence.canonical_json().unwrap()).unwrap();
    // A missing child still decodes locally, but exact replay must reject it.
    changed["face"]["cell_indices"]
        .as_array_mut()
        .unwrap()
        .pop();
    let changed = GeometryMeshCorrespondenceEnvelopeV1::from_json(
        &serde_json::to_vec(&changed).unwrap(),
        GeometryDecoderLimits::default(),
    )
    .unwrap();
    // Even a new lineage binding the forged bytes cannot authorize them.
    let forged = MeshProductionLineageEnvelopeV1::from_affine_triangle_resources(
        &policy, &geometry, &mesh, &changed,
    )
    .unwrap();
    assert!(
        forged
            .validate_against_affine_triangle_resources(&policy, &geometry, &mesh, &changed)
            .is_err()
    );
}
