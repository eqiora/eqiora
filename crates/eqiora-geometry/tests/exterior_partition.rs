//! Entity-incidence fixtures, independent of meshes and Cartesian side labels.

use eqiora_geometry::{
    CanonicalGeometryLimits, CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion,
};

fn set(name: &str, dimension: usize, members: &[usize]) -> NamedEntitySet {
    NamedEntitySet::new(name, dimension, members.to_vec())
}

fn cylinder() -> CanonicalGeometryV1 {
    CanonicalGeometryV1::from_circular_hole(
        [[0.0, 2.2], [0.0, 0.41]],
        [0.2, 0.2],
        0.05,
        vec![
            set("fluid", 2, &[0]),
            set("inlet", 1, &[0]),
            set("outlet", 1, &[1]),
            set("walls", 1, &[2, 3]),
            set("hole", 1, &[4]),
        ],
    )
    .unwrap()
}

fn check(
    geometry: &CanonicalGeometryV1,
    parent: &str,
    parts: &[&str],
) -> Result<(), eqiora_core::Diagnostic> {
    geometry.validate_complete_exterior_partition(
        geometry.entity_set(parent).unwrap(),
        &parts
            .iter()
            .map(|name| geometry.entity_set(name).unwrap())
            .collect::<Vec<_>>(),
        CanonicalGeometryLimits::default(),
    )
}

fn triangles() -> CanonicalGeometryV1 {
    // Canonical face loops (0,2,1) and (1,2,3) have reversed shared edges
    // with distinct entity IDs 1 and 3. Exterior IDs are independently 0,2,4,5.
    CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            vec![[0., 0.], [0., 1.], [1., 0.], [1., 1.]],
            vec![
                PlanarFace::new(vec![0, 2, 1], vec![]),
                PlanarFace::new(vec![1, 2, 3], vec![]),
            ],
            vec![
                set("both", 2, &[0, 1]),
                set("first", 2, &[0]),
                set("second", 2, &[1]),
                set("a", 1, &[0, 2]),
                set("b", 1, &[4, 5]),
                set("all", 1, &[0, 2, 4, 5]),
                set("alias", 1, &[0]),
                set("inside-a", 1, &[1]),
                set("inside-b", 1, &[3]),
            ],
            1e-12,
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn circular_hole_frontier_includes_the_hole_and_grouped_walls() {
    let geometry = cylinder();
    let before = geometry.canonical_bytes().to_vec();
    assert!(check(&geometry, "fluid", &["inlet", "outlet", "walls", "hole"]).is_ok());
    assert!(check(&geometry, "fluid", &["hole", "walls", "outlet", "inlet"]).is_ok());
    assert!(check(&geometry, "fluid", &["inlet", "outlet", "walls"]).is_err());
    assert!(
        check(
            &geometry,
            "fluid",
            &["inlet", "inlet", "outlet", "walls", "hole"]
        )
        .is_err()
    );
    assert!(check(&geometry, "fluid", &[]).is_err());
    assert_eq!(geometry.canonical_bytes(), before);
}

#[test]
fn exact_object_catalog_ownership_rejects_replayed_and_stale_handles() {
    let geometry = cylinder();
    let same_identity = CanonicalGeometryV1::replay_canonical(
        geometry.canonical_bytes(),
        CanonicalGeometryLimits::default(),
    )
    .unwrap();
    assert_eq!(geometry.digest_bytes(), same_identity.digest_bytes());
    let parent = geometry.entity_set("fluid").unwrap();
    assert!(
        geometry
            .validate_complete_exterior_partition(
                parent,
                &[same_identity.entity_set("walls").unwrap()],
                CanonicalGeometryLimits::default()
            )
            .is_err()
    );
    assert!(
        geometry
            .validate_complete_exterior_partition(
                same_identity.entity_set("fluid").unwrap(),
                &[geometry.entity_set("walls").unwrap()],
                CanonicalGeometryLimits::default()
            )
            .is_err()
    );
    let cloned_set = geometry.entity_set("hole").unwrap().clone();
    assert!(
        geometry
            .validate_complete_exterior_partition(
                parent,
                &[&cloned_set],
                CanonicalGeometryLimits::default()
            )
            .is_err()
    );
    let stale = triangles();
    assert!(
        geometry
            .validate_complete_exterior_partition(
                parent,
                &[stale.entity_set("a").unwrap()],
                CanonicalGeometryLimits::default()
            )
            .is_err()
    );
    assert!(check(&geometry, "walls", &["inlet"]).is_err());
    assert!(check(&geometry, "fluid", &["fluid"]).is_err());
}

#[test]
fn groups_are_disjoint_entity_partitions_and_interior_edges_are_not_exterior() {
    let geometry = triangles();
    assert!(check(&geometry, "both", &["a", "b"]).is_ok());
    assert!(check(&geometry, "both", &["all"]).is_ok());
    assert!(check(&geometry, "both", &["b", "a"]).is_ok());
    assert!(check(&geometry, "both", &["a", "alias", "b"]).is_err());
    assert!(check(&geometry, "first", &["all"]).is_err());
    assert!(check(&geometry, "both", &["a", "b", "inside-a"]).is_err());
    assert!(check(&geometry, "both", &["a", "b", "inside-b"]).is_err());
    // Preserve the singular query's existing selected-face membership meaning.
    assert!(geometry.selection_is_boundary_of(
        geometry.entity_set("inside-a").unwrap(),
        geometry.entity_set("both").unwrap()
    ));
}

#[test]
fn nonrectangular_polygon_requires_unnamed_frontier_entities_too() {
    // A triangle has exactly three edge entities; no rectangle-side assumptions.
    let geometry = CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            vec![[0., 0.], [0., 2.], [3., 0.]],
            vec![PlanarFace::new(vec![0, 2, 1], vec![])],
            vec![
                set("body", 2, &[0]),
                set("two", 1, &[0, 1]),
                set("all", 1, &[0, 1, 2]),
            ],
            1e-12,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(check(&geometry, "body", &["all"]).is_ok());
    assert!(check(&geometry, "body", &["two"]).is_err());
    // No selection names the last edge in this separate admitted geometry.
    let unnamed = CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            vec![[0., 0.], [0., 2.], [3., 0.]],
            vec![PlanarFace::new(vec![0, 2, 1], vec![])],
            vec![set("body", 2, &[0]), set("two", 1, &[0, 1])],
            1e-12,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(check(&unnamed, "body", &["two"]).is_err());
}

#[test]
fn ambiguous_planar_incidence_rejects_instead_of_cancelling() {
    for vertices in [
        vec![[0., 0.], [0., 1.], [1., 0.], [1., 1.]],
        vec![[0., 0.], [0., 1.], [1., 0.], [1., 1.], [2., 1.]],
    ] {
        // These distinct CCW triangles share directed 0->2, rather than opposite
        // incidences. The third face makes that edge nonmanifold as well.
        let mut faces = vec![
            PlanarFace::new(vec![0, 2, 1], vec![]),
            PlanarFace::new(vec![0, 2, 3], vec![]),
        ];
        if vertices.len() == 5 {
            faces.push(PlanarFace::new(vec![0, 2, 4], vec![]));
        }
        let count = faces.len();
        let geometry = CanonicalGeometryV1::from_region(
            &PlanarRegion::new(
                vertices,
                faces,
                vec![
                    set("body", 2, &(0..count).collect::<Vec<_>>()),
                    set("edges", 1, &(0..3 * count).collect::<Vec<_>>()),
                ],
                1e-12,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(check(&geometry, "body", &["edges"]).is_err());
    }
}

#[test]
fn shared_polyhedral_facets_cancel_from_a_multi_volume_frontier() {
    // Two tetrahedra share global canonical facet 3. Seven facets total, with
    // exterior IDs 0,1,2,4,5,6; the shared facet has opposite parent incidence.
    let geometry = CanonicalGeometryV1::from_convex_polyhedra(
        vec![
            [-1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [1., 0., 0.],
        ],
        vec![
            vec![vec![1, 3, 2], vec![0, 3, 1], vec![0, 2, 3], vec![0, 1, 2]],
            vec![vec![1, 2, 3], vec![4, 1, 3], vec![4, 3, 2], vec![4, 2, 1]],
        ],
        vec![
            set("both", 3, &[0, 1]),
            set("first", 3, &[0]),
            set("outer", 2, &[0, 1, 2, 4, 5, 6]),
            set("inside", 2, &[3]),
            set("all", 2, &[0, 1, 2, 3, 4, 5, 6]),
        ],
        1e-12,
    )
    .unwrap();
    assert!(check(&geometry, "both", &["outer"]).is_ok());
    assert!(check(&geometry, "both", &["all"]).is_err());
    assert!(check(&geometry, "both", &["outer", "inside"]).is_err());
    assert!(check(&geometry, "first", &["outer"]).is_err());
}

#[test]
fn membership_and_topology_budgets_have_exact_positive_boundaries() {
    let geometry = cylinder();
    let parent = geometry.entity_set("fluid").unwrap();
    let parts = ["inlet", "outlet", "walls", "hole"].map(|name| geometry.entity_set(name).unwrap());
    let exact = CanonicalGeometryLimits {
        max_faces: 1,
        max_loop_indices: 5,
        max_entity_sets: 4,
        max_entity_set_members: 6,
        ..CanonicalGeometryLimits::default()
    };
    assert!(
        geometry
            .validate_complete_exterior_partition(parent, &parts, exact)
            .is_ok()
    );
    for limits in [
        CanonicalGeometryLimits {
            max_faces: 0,
            ..exact
        },
        CanonicalGeometryLimits {
            max_loop_indices: 4,
            ..exact
        },
        CanonicalGeometryLimits {
            max_entity_sets: 3,
            ..exact
        },
        CanonicalGeometryLimits {
            max_entity_set_members: 5,
            ..exact
        },
    ] {
        let error = geometry
            .validate_complete_exterior_partition(parent, &parts, limits)
            .unwrap_err();
        assert!(error.message().contains("limit"));
    }
    let polygon = triangles();
    assert!(
        polygon
            .validate_complete_exterior_partition(
                polygon.entity_set("both").unwrap(),
                &[polygon.entity_set("all").unwrap()],
                CanonicalGeometryLimits {
                    max_loop_indices: 5,
                    ..CanonicalGeometryLimits::default()
                }
            )
            .is_err()
    );
}
