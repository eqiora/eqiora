use super::intersection::ConstructedPoint2d;
use super::intersection::{convex_hull, next_up, proper_segment_intersection};
use super::*;
use crate::{MeshQualityGate, MeshTopology};

fn quality() -> MeshQualityGate {
    MeshQualityGate::new(1.0e-8).unwrap()
}

fn source_square() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ],
        vec![vec![0, 1, 2], vec![0, 2, 3]],
        quality(),
    )
    .unwrap()
}

fn crossed_square() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ],
        vec![vec![0, 1, 3], vec![1, 2, 3]],
        quality(),
    )
    .unwrap()
}

fn split_boundary_square() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![0.5, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
        ],
        vec![
            vec![0, 1, 5],
            vec![1, 2, 5],
            vec![2, 3, 5],
            vec![3, 4, 5],
            vec![4, 0, 5],
        ],
        quality(),
    )
    .unwrap()
}

fn unit_triangle() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![vec![0, 1, 2]],
        quality(),
    )
    .unwrap()
}

fn equal_measure_centroid_shear() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![
            vec![-1.0 / 12.0, 0.0],
            vec![11.0 / 12.0, 0.0],
            vec![1.0 / 6.0, 1.0],
        ],
        vec![vec![0, 1, 2]],
        quality(),
    )
    .unwrap()
}

fn all_cells(mesh: &SimplicialMesh) -> Vec<CellId> {
    (0..mesh.entity_count(2).unwrap())
        .map(CellId::new)
        .collect()
}

fn facet_with_vertices(mesh: &SimplicialMesh, vertices: [usize; 2]) -> FacetId {
    let expected = vertices
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    (0..mesh.entity_count(1).unwrap())
        .map(FacetId::new)
        .find(|facet| {
            mesh.entity_vertices(MeshEntity::new(1, facet.index()))
                .unwrap()
                .into_iter()
                .map(MeshEntity::index)
                .collect::<std::collections::BTreeSet<_>>()
                == expected
        })
        .unwrap()
}

#[test]
fn crossed_diagonals_produce_many_to_many_fragments_without_index_inference() {
    let source = source_square();
    let target = crossed_square();
    let overlap = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        &source,
        &all_cells(&source),
        &target,
        &all_cells(&target),
    )
    .unwrap();

    assert_eq!(overlap.cell_fragments().len(), 4);
    let pairs = overlap
        .cell_fragments()
        .iter()
        .map(|fragment| (fragment.source_cell(), fragment.target_cell()))
        .collect::<Vec<_>>();
    assert_eq!(
        pairs,
        vec![
            (CellId::new(0), CellId::new(0)),
            (CellId::new(0), CellId::new(1)),
            (CellId::new(1), CellId::new(0)),
            (CellId::new(1), CellId::new(1)),
        ]
    );
    assert!(overlap.cell_fragments().iter().all(|fragment| {
        (fragment.area() - 0.25).abs() <= 16.0 * f64::EPSILON
            && orientation(
                fragment.triangle()[0],
                fragment.triangle()[1],
                fragment.triangle()[2],
            ) > 0.0
    }));
}

#[test]
fn retained_boundary_split_has_complete_orientation_aware_coverage() {
    let source = source_square();
    let target = split_boundary_square();
    let source_bottom =
        RetainedFacetSide2d::new(facet_with_vertices(&source, [0, 1]), CellId::new(0));
    let target_left =
        RetainedFacetSide2d::new(facet_with_vertices(&target, [0, 1]), CellId::new(0));
    let target_right =
        RetainedFacetSide2d::new(facet_with_vertices(&target, [1, 2]), CellId::new(1));
    let overlap = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::CurrentSpatial,
        &source,
        &all_cells(&source),
        &target,
        &all_cells(&target),
    )
    .unwrap()
    .with_retained_facets(
        &source,
        &[source_bottom],
        &target,
        &[target_right, target_left],
    )
    .unwrap();

    assert_eq!(overlap.retained_facet_fragments().len(), 2);
    assert_eq!(
        overlap.target_retained_facets(),
        &[target_left, target_right]
    );
    for fragment in overlap.retained_facet_fragments() {
        assert!((fragment.length() - 0.5).abs() <= 8.0 * f64::EPSILON);
        assert_eq!(fragment.source_outward_normal(), [0.0, -1.0]);
        assert_eq!(fragment.target_outward_normal(), [0.0, -1.0]);
        assert_eq!(
            fragment.source_outward_normal()[0].to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            fragment.target_outward_normal()[0].to_bits(),
            0.0_f64.to_bits()
        );
    }
}

#[test]
fn selected_region_requires_bidirectional_coverage() {
    let source = source_square();
    let target = crossed_square();
    let error = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        &source,
        &all_cells(&source),
        &target,
        &[CellId::new(0)],
    )
    .unwrap_err();
    assert!(error.message().contains("incomplete or multiply covered"));
}

#[test]
fn robust_predicate_rejects_a_degenerate_triangle() {
    let degenerate = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let valid = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let error = intersect_triangles(&degenerate, &valid).unwrap_err();
    assert!(error.message().contains("robustly degenerate"));
}

#[test]
fn exact_positive_fragment_that_collapses_in_binary64_fails_closed() {
    let source = [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]];
    let target = [[0.0, -1.0], [2.0, -1.0], [1.0, f64::from_bits(1)]];
    let error = intersect_triangles(&source, &target).unwrap_err();
    assert!(
        error
            .message()
            .contains("collapses under canonical nearest-binary64")
    );
}

#[test]
fn exact_intersection_construction_is_canonical_near_an_axis() {
    let source = [[0.5, 0.0], [0.500_375_000_000_000_8, 0.500_124_791_077_715]];
    let target = [
        [0.500_198_529_411_765_1, 0.250_066_065_864_672_67],
        [0.0, 0.25],
    ];
    let expected = proper_segment_intersection(source, target)
        .unwrap()
        .expect("fixture is one robust proper crossing");
    for (source, target) in [
        ([source[1], source[0]], target),
        (source, [target[1], target[0]]),
        (target, source),
        ([target[1], target[0]], [source[1], source[0]]),
    ] {
        let actual = proper_segment_intersection(source, target)
            .unwrap()
            .expect("orientation changes preserve the proper crossing");
        assert_eq!(
            actual.rounded.map(f64::to_bits),
            expected.rounded.map(f64::to_bits)
        );
        assert_eq!(actual.exact, expected.exact);
    }
}

#[test]
fn rounding_cells_admit_one_trace_split_but_reject_larger_drift() {
    let source = [[1.0, 0.0], [1.001_500_000_000_003, 0.500_499_164_310_860_2]];
    let rounded_midpoint = [1.000_750_000_000_001_6, 0.250_249_582_155_430_1];
    let target = [source[0], rounded_midpoint];
    let forward = collinear_segment_overlap(&source, &target)
        .unwrap()
        .expect("one-step-rounded trace split is certified");
    let reverse = collinear_segment_overlap(&target, &source)
        .unwrap()
        .expect("source-target exchange preserves the certificate");
    assert_eq!(
        forward.map(|point| point.map(f64::to_bits)),
        reverse.map(|point| point.map(f64::to_bits))
    );

    let mut drifted = target;
    drifted[1][0] = next_up(next_up(rounded_midpoint[0]));
    assert!(
        collinear_segment_overlap(&source, &drifted)
            .unwrap()
            .is_none()
    );
}

#[test]
fn unrelated_adjacent_binary64_crossings_remain_distinct() {
    let adjacent = next_up(1.0);
    let first = proper_segment_intersection([[1.0, -1.0], [1.0, 2.0]], [[0.0, 1.0], [2.0, 1.0]])
        .unwrap()
        .unwrap();
    let second = proper_segment_intersection(
        [[adjacent, -1.0], [adjacent, 2.0]],
        [[0.0, adjacent], [2.0, adjacent]],
    )
    .unwrap()
    .unwrap();
    assert_ne!(first.exact, second.exact);
    assert_ne!(first.rounded, second.rounded);

    let third = ConstructedPoint2d::from_binary64([0.0, 2.0]).unwrap();
    let hull = convex_hull(vec![first, second, third]);
    assert_eq!(hull.len(), 3);
    let skinny = [hull[0].rounded, hull[1].rounded, hull[2].rounded];
    assert!(triangle_measure(&skinny).unwrap().0 > 0.0);
}

#[test]
fn input_order_is_normalized_but_duplicates_fail() {
    let source = source_square();
    let target = crossed_square();
    let forward = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        &source,
        &[CellId::new(0), CellId::new(1)],
        &target,
        &[CellId::new(0), CellId::new(1)],
    )
    .unwrap();
    let reordered = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        &source,
        &[CellId::new(1), CellId::new(0)],
        &target,
        &[CellId::new(1), CellId::new(0)],
    )
    .unwrap();
    assert_eq!(forward, reordered);

    let duplicate = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        &source,
        &[CellId::new(0), CellId::new(0)],
        &target,
        &all_cells(&target),
    )
    .unwrap_err();
    assert!(duplicate.message().contains("duplicate"));
}

#[test]
fn retained_facets_reject_same_measure_same_centroid_stale_geometry() {
    let source = unit_triangle();
    let target = unit_triangle();
    let stale_source = equal_measure_centroid_shear();
    let overlap = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        &source,
        &all_cells(&source),
        &target,
        &all_cells(&target),
    )
    .unwrap();
    let stale_bottom =
        RetainedFacetSide2d::new(facet_with_vertices(&stale_source, [0, 1]), CellId::new(0));
    let target_bottom =
        RetainedFacetSide2d::new(facet_with_vertices(&target, [0, 1]), CellId::new(0));

    let error = overlap
        .with_retained_facets(&stale_source, &[stale_bottom], &target, &[target_bottom])
        .unwrap_err();
    assert!(error.message().contains("exactly match"));
}

#[test]
fn retained_sides_must_be_true_frontiers_and_cover_both_revisions() {
    let source = source_square();
    let target = split_boundary_square();
    let interior = RetainedFacetSide2d::new(facet_with_vertices(&source, [0, 2]), CellId::new(0));
    let target_left =
        RetainedFacetSide2d::new(facet_with_vertices(&target, [0, 1]), CellId::new(0));
    let overlap = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        &source,
        &all_cells(&source),
        &target,
        &all_cells(&target),
    )
    .unwrap();
    let error = overlap
        .with_retained_facets(&source, &[interior], &target, &[target_left])
        .unwrap_err();
    assert!(error.message().contains("true frontier"));

    let source_bottom =
        RetainedFacetSide2d::new(facet_with_vertices(&source, [0, 1]), CellId::new(0));
    let overlap = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        &source,
        &all_cells(&source),
        &target,
        &all_cells(&target),
    )
    .unwrap();
    let error = overlap
        .with_retained_facets(&source, &[source_bottom], &target, &[target_left])
        .unwrap_err();
    assert!(error.message().contains("incomplete or multiply covered"));

    let overlap = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        &source,
        &all_cells(&source),
        &target,
        &all_cells(&target),
    )
    .unwrap();
    let error = overlap
        .with_retained_facets(
            &source,
            &[source_bottom, source_bottom],
            &target,
            &[target_left],
        )
        .unwrap_err();
    assert!(error.message().contains("duplicate facet"));
}
