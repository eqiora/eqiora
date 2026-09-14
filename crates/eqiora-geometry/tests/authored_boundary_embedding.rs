use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion};
use eqiora_schema::kernel::BoundarySide::{Lower, Upper};

fn authored(vertices: Vec<[f64; 2]>, faces: Vec<PlanarFace>, prefix: &str) -> CanonicalGeometryV1 {
    let region = PlanarRegion::new(vertices, faces, vec![], 1e-9).unwrap();
    let mut sets = (0..region.faces().len())
        .map(|i| NamedEntitySet::new(format!("{prefix}-parent-{i}"), 2, vec![i]))
        .chain(
            (0..region.edge_count())
                .map(|i| NamedEntitySet::new(format!("{prefix}-edge-{i}"), 1, vec![i])),
        )
        .collect::<Vec<_>>();
    sets.push(NamedEntitySet::new("grouped-sides", 1, vec![0, 1]));
    if region.faces().len() > 1 {
        sets.push(NamedEntitySet::new("grouped-parents", 2, vec![0, 1]));
    }
    CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            region.vertices().to_vec(),
            region.faces().to_vec(),
            sets,
            1e-9,
        )
        .unwrap(),
    )
    .unwrap()
}

fn rectangles(prefix: &str, permuted: bool) -> CanonicalGeometryV1 {
    let mut vertices = vec![
        [2.0, -3.0],
        [5.0, -3.0],
        [7.0, -3.0],
        [2.0, 7.0],
        [5.0, 7.0],
        [7.0, 7.0],
    ];
    let loops = if permuted {
        vertices.reverse();
        vec![vec![0, 1, 4, 3], vec![1, 2, 5, 4]]
    } else {
        vec![vec![0, 1, 4, 3], vec![1, 2, 5, 4]]
    };
    authored(
        vertices,
        loops
            .into_iter()
            .map(|outer| PlanarFace::new(outer, vec![]))
            .collect(),
        prefix,
    )
}

#[test]
fn exact_authored_rectangle_embedding_ignores_names_and_author_order() {
    for (prefix, permuted) in [("scalar", false), ("unrelated", true)] {
        let geometry = rectangles(prefix, permuted);
        for (face, [xmin, xmax]) in [[2.0, 5.0], [5.0, 7.0]].into_iter().enumerate() {
            let parent = geometry
                .entity_set(&format!("{prefix}-parent-{face}"))
                .unwrap();
            // Independent rectangle sides in the canonical counterclockwise loop.
            for (local, axis, side, coordinate, interval, normal) in [
                (0, 1, Lower, -3.0, (xmin, xmax), [0.0, -1.0]),
                (1, 0, Upper, xmax, (-3.0, 7.0), [1.0, 0.0]),
                (2, 1, Upper, 7.0, (xmin, xmax), [0.0, 1.0]),
                (3, 0, Lower, xmin, (-3.0, 7.0), [-1.0, 0.0]),
            ] {
                let name = format!("{prefix}-edge-{}", 4 * face + local);
                let edge = geometry.entity_set(&name).unwrap();
                let embedding = geometry.cartesian_boundary_embedding(edge, parent).unwrap();
                assert_eq!(embedding.ambient_dimension(), 2);
                assert_eq!(embedding.normal_axis(), axis);
                assert_eq!(embedding.side(), side);
                assert_eq!(embedding.coordinate(), coordinate);
                assert_eq!(embedding.tangential_intervals(), &[interval]);
                assert_eq!(geometry.constant_parent_outward_normal(&name), Some(normal));
            }
        }
        // The two occurrences of x=5 have distinct parents and opposite normals.
        assert_eq!(
            geometry.constant_parent_outward_normal(&format!("{prefix}-edge-1")),
            Some([1.0, 0.0])
        );
        assert_eq!(
            geometry.constant_parent_outward_normal(&format!("{prefix}-edge-7")),
            Some([-1.0, 0.0])
        );
    }
}

#[test]
fn exact_embedding_rejects_grouped_wrong_parent_and_foreign_selections() {
    let geometry = rectangles("a", false);
    let replay = rectangles("a", true);
    assert_eq!(geometry, replay);
    let edge = geometry.entity_set("a-edge-1").unwrap();
    let parent = geometry.entity_set("a-parent-0").unwrap();
    for (boundary, parent) in [
        (edge, geometry.entity_set("a-parent-1").unwrap()),
        (edge, geometry.entity_set("grouped-parents").unwrap()),
        (geometry.entity_set("grouped-sides").unwrap(), parent),
        (replay.entity_set("a-edge-1").unwrap(), parent),
        (edge, replay.entity_set("a-parent-0").unwrap()),
        (parent, parent),
        (edge, edge),
    ] {
        assert!(
            geometry
                .cartesian_boundary_embedding(boundary, parent)
                .is_none()
        );
    }
    assert_eq!(
        geometry.constant_parent_outward_normal("grouped-sides"),
        None
    );
    assert_eq!(geometry.constant_parent_outward_normal("a-parent-0"), None);
}

#[test]
fn diagonal_nonrectangular_and_holey_faces_have_no_rectangular_embedding() {
    for (vertices, outer, holes) in [
        (
            vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            vec![0, 1, 2],
            vec![],
        ),
        // Rotated square: a rectangle is insufficient without exact axis alignment.
        (
            vec![[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]],
            vec![0, 1, 2, 3],
            vec![],
        ),
        // Every bounding corner occurs, but the notch is not a box-side segment.
        (
            vec![
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [1.5, 2.0],
                [1.0, 1.0],
                [0.5, 2.0],
                [0.0, 2.0],
            ],
            vec![0, 1, 2, 3, 4, 5, 6],
            vec![],
        ),
        (
            vec![
                [0.0, 0.0],
                [3.0, 0.0],
                [3.0, 3.0],
                [0.0, 3.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [1.0, 2.0],
            ],
            vec![0, 1, 2, 3],
            vec![vec![4, 5, 6, 7]],
        ),
        // A near-axis edge must not be snapped using geometry classification tolerance.
        (
            vec![[0.0, 0.0], [2.0, 1e-12], [2.0, 2.0], [0.0, 2.0]],
            vec![0, 1, 2, 3],
            vec![],
        ),
    ] {
        let geometry = authored(vertices, vec![PlanarFace::new(outer, holes)], "x");
        let parent = geometry.entity_set("x-parent-0").unwrap();
        for edge in geometry
            .entity_sets()
            .iter()
            .filter(|set| set.dimension() == 1)
        {
            assert!(
                geometry
                    .cartesian_boundary_embedding(edge, parent)
                    .is_none()
            );
            assert_eq!(geometry.constant_parent_outward_normal(edge.name()), None);
        }
    }
}

#[test]
fn proper_side_segments_reject_while_other_complete_sides_remain_exact() {
    let geometry = authored(
        vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        vec![PlanarFace::new(vec![0, 1, 2, 3, 4], vec![])],
        "x",
    );
    let parent = geometry.entity_set("x-parent-0").unwrap();
    for index in [0, 1] {
        let name = format!("x-edge-{index}");
        assert!(
            geometry
                .cartesian_boundary_embedding(geometry.entity_set(&name).unwrap(), parent)
                .is_none()
        );
        assert_eq!(geometry.constant_parent_outward_normal(&name), None);
    }
    let right = geometry
        .cartesian_boundary_embedding(geometry.entity_set("x-edge-2").unwrap(), parent)
        .unwrap();
    assert_eq!(right.normal_axis(), 0);
    assert_eq!(right.side(), Upper);
    assert_eq!(right.coordinate(), 2.0);
    assert_eq!(right.tangential_intervals(), &[(0.0, 2.0)]);
}

#[test]
fn primitive_partition_retains_every_exact_parent_side_through_the_same_embedding() {
    use eqiora_geometry::GeometryGraph;
    use std::collections::BTreeMap;

    let graph = GeometryGraph::new();
    let first = graph.rectangle([2.0, 5.0], [-3.0, 7.0]).unwrap();
    let second = graph.rectangle([5.0, 7.0], [-3.0, 7.0]).unwrap();
    let partition = graph
        .partition(
            &first,
            &second,
            [first.boundaries()[1], second.boundaries()[0]],
        )
        .unwrap();
    let mut names = BTreeMap::new();
    for (name, rectangle) in [("omega", &first), ("alpha", &second)] {
        names.insert(name.to_owned(), vec![rectangle.region().into()]);
        for (side, edge) in rectangle.boundaries().iter().enumerate() {
            names.insert(format!("{name}-arbitrary-{side}"), vec![(*edge).into()]);
        }
    }
    let geometry = graph.build(&partition, &names).unwrap();
    let replay = CanonicalGeometryV1::decode_planar_adjacent_rectangle_partition_v1_canonical(
        geometry.canonical_bytes(),
        Default::default(),
    )
    .unwrap();
    for (name, xmin, xmax) in [("omega", 2.0, 5.0), ("alpha", 5.0, 7.0)] {
        let parent = geometry.entity_set(name).unwrap();
        for (index, axis, side, coordinate, interval, normal) in [
            (0, 0, Lower, xmin, (-3.0, 7.0), [-1.0, 0.0]),
            (1, 0, Upper, xmax, (-3.0, 7.0), [1.0, 0.0]),
            (2, 1, Lower, -3.0, (xmin, xmax), [0.0, -1.0]),
            (3, 1, Upper, 7.0, (xmin, xmax), [0.0, 1.0]),
        ] {
            let key = format!("{name}-arbitrary-{index}");
            let edge = geometry.entity_set(&key).unwrap();
            let embedding = geometry.cartesian_boundary_embedding(edge, parent).unwrap();
            assert_eq!(embedding.normal_axis(), axis);
            assert_eq!(embedding.side(), side);
            assert_eq!(embedding.coordinate(), coordinate);
            assert_eq!(embedding.tangential_intervals(), &[interval]);
            assert_eq!(geometry.constant_parent_outward_normal(&key), Some(normal));
            assert!(
                geometry
                    .cartesian_boundary_embedding(replay.entity_set(&key).unwrap(), parent)
                    .is_none()
            );
            assert!(
                geometry
                    .cartesian_boundary_embedding(edge, replay.entity_set(name).unwrap())
                    .is_none()
            );
            let other = if name == "omega" { "alpha" } else { "omega" };
            assert!(
                geometry
                    .cartesian_boundary_embedding(edge, geometry.entity_set(other).unwrap())
                    .is_none()
            );
        }
    }
    assert_eq!(
        geometry.constant_parent_outward_normal("omega-arbitrary-1"),
        Some([1.0, 0.0])
    );
    assert_eq!(
        geometry.constant_parent_outward_normal("alpha-arbitrary-0"),
        Some([-1.0, 0.0])
    );
}
