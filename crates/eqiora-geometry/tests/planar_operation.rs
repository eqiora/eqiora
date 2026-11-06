use std::collections::BTreeMap;

use eqiora_geometry::{
    CanonicalGeometryV1, EDGE_DIMENSION, FACE_DIMENSION, GeometryGraph, GeometryTopologyHandle,
};

fn named(
    entries: impl IntoIterator<Item = (&'static str, Vec<GeometryTopologyHandle>)>,
) -> BTreeMap<String, Vec<GeometryTopologyHandle>> {
    entries
        .into_iter()
        .map(|(name, handles)| (name.to_owned(), handles))
        .collect()
}

#[test]
fn rectangle_publishes_direct_source_owned_topology() {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 2.2], [0.0, 0.41]).unwrap();
    let boundaries = rectangle.boundaries();
    assert_eq!(boundaries.len(), 4);

    let geometry = graph
        .build(
            &rectangle,
            &named([
                ("domain", vec![rectangle.region().into()]),
                ("left", vec![boundaries[0].into()]),
                ("right", vec![boundaries[1].into()]),
                ("bottom", vec![boundaries[2].into()]),
                ("top", vec![boundaries[3].into()]),
            ]),
        )
        .unwrap();

    assert_eq!(
        geometry.planar_rectangle_bounds(),
        Some(&[[0.0, 2.2], [0.0, 0.41]])
    );
    assert_eq!(geometry.classification_tolerance_m(), None);
    assert_eq!(
        geometry.entity_set_dimension("domain"),
        Some(FACE_DIMENSION)
    );
    assert_eq!(geometry.entity_set_dimension("left"), Some(EDGE_DIMENSION));
    let reference = &geometry;
    assert_eq!(
        ["left", "right", "bottom", "top"].map(|name| {
            reference
                .constant_parent_outward_normal(name)
                .expect("direct boundary handle names one canonical rectangle side")
        }),
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
    );
    assert_eq!(reference.constant_parent_outward_normal("domain"), None);
    assert_eq!(
        CanonicalGeometryV1::decode_planar_rectangle_v2_canonical(
            geometry.canonical_bytes(),
            Default::default(),
        )
        .unwrap(),
        geometry
    );

    let canonical = std::str::from_utf8(geometry.canonical_bytes()).unwrap();
    let out_of_range = canonical.replacen(
        r#""name":"left","dimension":1,"members":[0]"#,
        r#""name":"left","dimension":1,"members":[4]"#,
        1,
    );
    assert_ne!(out_of_range, canonical);
    assert!(
        CanonicalGeometryV1::decode_planar_rectangle_v2_canonical(
            out_of_range.as_bytes(),
            Default::default(),
        )
        .is_err()
    );
}

#[test]
fn subtract_projects_predecessor_handles_without_names_or_coordinates() {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 2.2], [0.0, 0.41]).unwrap();
    let circle = graph.circle([0.2, 0.2], 0.05).unwrap();
    let result = graph.subtract(&rectangle, &circle).unwrap();
    let outer = rectangle.boundaries();
    let cut = circle.boundaries();

    let geometry = graph
        .build(
            &result,
            &named([
                ("fluid", vec![result.region().into()]),
                ("inlet", vec![outer[0].into()]),
                ("outlet", vec![outer[1].into()]),
                ("walls", vec![outer[2].into(), outer[3].into()]),
                ("cylinder", vec![cut[0].into()]),
            ]),
        )
        .unwrap();
    assert_eq!(geometry.circular_hole_center(), Some([0.2, 0.2]));
    assert_eq!(geometry.circular_hole_radius_m(), Some(0.05));

    let result_boundaries = result.boundaries();
    let direct = graph
        .build(
            &result,
            &named([
                ("fluid", vec![result.region().into()]),
                ("inlet", vec![result_boundaries[0].into()]),
                ("outlet", vec![result_boundaries[1].into()]),
                (
                    "walls",
                    vec![result_boundaries[2].into(), result_boundaries[3].into()],
                ),
                ("cylinder", vec![result_boundaries[4].into()]),
            ]),
        )
        .unwrap();
    assert_eq!(direct, geometry);
}

#[test]
fn foreign_deleted_stale_incomplete_and_mixed_handles_fail_closed() {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 2.2], [0.0, 0.41]).unwrap();
    let circle = graph.circle([0.2, 0.2], 0.05).unwrap();
    let result = graph.subtract(&rectangle, &circle).unwrap();
    let outer = rectangle.boundaries();
    let cut = circle.boundaries();

    let foreign_graph = GeometryGraph::new();
    let foreign = foreign_graph.rectangle([0.0, 2.2], [0.0, 0.41]).unwrap();
    let stale = graph.rectangle([0.0, 1.0], [0.0, 1.0]).unwrap();

    for invalid_mapping in [
        named([
            ("fluid", vec![result.region().into()]),
            ("inlet", vec![foreign.boundaries()[0].into()]),
            ("outlet", vec![outer[1].into()]),
            ("walls", vec![outer[2].into(), outer[3].into()]),
            ("cylinder", vec![cut[0].into()]),
        ]),
        named([
            ("fluid", vec![rectangle.region().into()]),
            ("inlet", vec![outer[0].into()]),
            ("outlet", vec![outer[1].into()]),
            ("walls", vec![outer[2].into(), outer[3].into()]),
            ("cylinder", vec![cut[0].into()]),
        ]),
        named([
            ("fluid", vec![result.region().into()]),
            ("inlet", vec![stale.boundaries()[0].into()]),
            ("outlet", vec![outer[1].into()]),
            ("walls", vec![outer[2].into(), outer[3].into()]),
            ("cylinder", vec![cut[0].into()]),
        ]),
        named([
            ("fluid", vec![result.region().into()]),
            ("outer", vec![outer[0].into(), outer[1].into()]),
            ("cylinder", vec![cut[0].into()]),
        ]),
        named([
            ("mixed", vec![result.region().into(), outer[0].into()]),
            ("outlet", vec![outer[1].into()]),
            ("walls", vec![outer[2].into(), outer[3].into()]),
            ("cylinder", vec![cut[0].into()]),
        ]),
    ] {
        assert!(graph.build(&result, &invalid_mapping).is_err());
    }

    assert!(graph.circle([f64::NAN, 0.0], 1.0).is_err());
    let tangent = graph.circle([0.05, 0.2], 0.05).unwrap();
    assert!(graph.subtract(&rectangle, &tangent).is_err());
}

#[test]
fn explicit_adjacent_partition_owns_both_parents_and_complete_frontiers() {
    let graph = GeometryGraph::new();
    let left = graph.rectangle([0.0, 1.0], [0.0, 1.0]).unwrap();
    let right = graph.rectangle([1.0, 2.0], [0.0, 1.0]).unwrap();
    let left_edges = left.boundaries();
    let right_edges = right.boundaries();
    let partition = graph
        .partition(&left, &right, [left_edges[1], right_edges[0]])
        .unwrap();
    let geometry = graph
        .build(
            &partition,
            &named([
                ("fluid", vec![left.region().into()]),
                ("solid", vec![right.region().into()]),
                (
                    "interface",
                    vec![left_edges[1].into(), right_edges[0].into()],
                ),
                ("inlet", vec![left_edges[0].into()]),
                ("outlet", vec![right_edges[1].into()]),
                (
                    "walls",
                    vec![
                        left_edges[2].into(),
                        left_edges[3].into(),
                        right_edges[2].into(),
                        right_edges[3].into(),
                    ],
                ),
            ]),
        )
        .unwrap();
    let region = geometry.region().unwrap();
    assert_eq!(
        region.vertices(),
        &[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0]
        ]
    );
    assert_eq!(region.faces()[0].outer(), [0, 2, 3, 1]);
    assert_eq!(region.faces()[1].outer(), [2, 4, 5, 3]);
    assert_eq!(geometry.entity_set("interface").unwrap().members(), [1, 7]);
    assert_eq!(geometry.classification_tolerance_m(), None);
    assert_eq!(
        CanonicalGeometryV1::decode_planar_adjacent_rectangle_partition_v1_canonical(
            geometry.canonical_bytes(),
            Default::default(),
        )
        .unwrap(),
        geometry,
    );
    let changed_orientation = std::str::from_utf8(geometry.canonical_bytes())
        .unwrap()
        .replace("opposite-parent-outward", "same-direction");
    assert!(
        CanonicalGeometryV1::decode_planar_adjacent_rectangle_partition_v1_canonical(
            changed_orientation.as_bytes(),
            Default::default(),
        )
        .is_err()
    );

    assert!(
        graph
            .partition(&left, &right, [right_edges[0], left_edges[1]])
            .is_err()
    );
    let gap = graph.rectangle([1.25, 2.0], [0.0, 1.0]).unwrap();
    assert!(
        graph
            .partition(&left, &gap, [left_edges[1], gap.boundaries()[0]])
            .is_err()
    );
    let overlap = graph.rectangle([0.75, 2.0], [0.0, 1.0]).unwrap();
    assert!(
        graph
            .partition(&left, &overlap, [left_edges[1], overlap.boundaries()[0]])
            .is_err()
    );

    let incomplete = named([
        ("fluid", vec![left.region().into()]),
        ("solid", vec![right.region().into()]),
        (
            "interface",
            vec![left_edges[1].into(), right_edges[0].into()],
        ),
    ]);
    assert!(graph.build(&partition, &incomplete).is_err());
}

#[test]
fn primitive_boundary_embeddings_keep_topology_orientation_and_exact_parent() {
    use eqiora_schema::kernel::BoundarySide::{Lower, Upper};
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([2.0, 5.0], [-3.0, 7.0]).unwrap();
    let geometry = graph
        .build(
            &rectangle,
            &named([
                ("body", vec![rectangle.region().into()]),
                ("a", vec![rectangle.boundaries()[0].into()]),
                ("b", vec![rectangle.boundaries()[1].into()]),
                ("c", vec![rectangle.boundaries()[2].into()]),
                ("d", vec![rectangle.boundaries()[3].into()]),
            ]),
        )
        .unwrap();
    let parent = geometry.entity_set("body").unwrap();
    for (name, axis, side, coordinate, tangent) in [
        ("a", 0, Lower, 2.0, (-3.0, 7.0)),
        ("b", 0, Upper, 5.0, (-3.0, 7.0)),
        ("c", 1, Lower, -3.0, (2.0, 5.0)),
        ("d", 1, Upper, 7.0, (2.0, 5.0)),
    ] {
        let embedding = geometry
            .cartesian_boundary_embedding(geometry.entity_set(name).unwrap(), parent)
            .unwrap();
        assert_eq!(embedding.ambient_dimension(), 2);
        assert_eq!(embedding.normal_axis(), axis);
        assert_eq!(embedding.side(), side);
        assert_eq!(embedding.coordinate(), coordinate);
        assert_eq!(embedding.tangential_intervals(), &[tangent]);
    }
    let boundary = geometry.entity_set("a").unwrap();
    let replay = CanonicalGeometryV1::decode_planar_rectangle_v2_canonical(
        geometry.canonical_bytes(),
        Default::default(),
    )
    .unwrap();
    assert!(
        geometry
            .cartesian_boundary_embedding(replay.entity_set("a").unwrap(), parent)
            .is_none()
    );
    assert!(
        geometry
            .cartesian_boundary_embedding(boundary, replay.entity_set("body").unwrap())
            .is_none()
    );
    assert!(
        geometry
            .cartesian_boundary_embedding(boundary, boundary)
            .is_none()
    );
    assert!(
        geometry
            .cartesian_boundary_embedding(parent, parent)
            .is_none()
    );
}

#[test]
fn primitive_embedding_does_not_turn_a_group_or_curved_region_into_a_box_side() {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 3.0], [0.0, 2.0]).unwrap();
    let grouped = graph
        .build(
            &rectangle,
            &named([
                ("body", vec![rectangle.region().into()]),
                ("left", vec![rectangle.boundaries()[0].into()]),
                ("right", vec![rectangle.boundaries()[1].into()]),
                (
                    "walls",
                    vec![
                        rectangle.boundaries()[2].into(),
                        rectangle.boundaries()[3].into(),
                    ],
                ),
            ]),
        )
        .unwrap();
    assert!(
        grouped
            .cartesian_boundary_embedding(
                grouped.entity_set("walls").unwrap(),
                grouped.entity_set("body").unwrap()
            )
            .is_none()
    );
    let circle = graph.circle([1.0, 1.0], 0.25).unwrap();
    let cut = graph.subtract(&rectangle, &circle).unwrap();
    let curved = graph
        .build(
            &cut,
            &named([
                ("body", vec![cut.region().into()]),
                ("left", vec![cut.boundaries()[0].into()]),
                ("right", vec![cut.boundaries()[1].into()]),
                (
                    "walls",
                    vec![cut.boundaries()[2].into(), cut.boundaries()[3].into()],
                ),
                ("circle", vec![cut.boundaries()[4].into()]),
            ]),
        )
        .unwrap();
    for name in ["left", "right", "walls", "circle"] {
        assert!(
            curved
                .cartesian_boundary_embedding(
                    curved.entity_set(name).unwrap(),
                    curved.entity_set("body").unwrap()
                )
                .is_none()
        );
    }
}
