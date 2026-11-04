use std::collections::BTreeSet;

use eqiora::artifact::{
    GeometryDefinitionV1, GeometryMeshCorrespondenceEnvelopeV1, SimplicialMeshEnvelopeV1,
};
use eqiora::diagnostic::codes;
use eqiora::geometry::{CadAuthoredFaceMesh, GeometryGraph, GeometrySolidOperation};
use eqiora::meshing::{MeshEntity, MeshQualityGate, MeshTopology};

const GEOMETRY_TOLERANCE_M: f64 = 5.0e-10;

fn graph() -> GeometrySolidOperation {
    rectangle_graph((-2.0, 3.0), (-1.0, 2.0), 0.5, 4.0, 1.0e-9)
}

fn rectangle_graph(
    x_bounds: (f64, f64),
    y_bounds: (f64, f64),
    plane_z: f64,
    depth: f64,
    tolerance: f64,
) -> GeometrySolidOperation {
    GeometryGraph::new()
        .rectangle_extrusion(x_bounds, y_bounds, plane_z, depth, tolerance)
        .unwrap()
}

fn quality(minimum: f64) -> MeshQualityGate {
    MeshQualityGate::new(minimum).unwrap()
}

fn realize(
    graph: &GeometrySolidOperation,
    provenance: &str,
    tolerance_m: f64,
    target_m: f64,
    maximum_triangles: usize,
    minimum_quality: f64,
) -> CadAuthoredFaceMesh {
    let face = graph.face_handle(provenance).unwrap();
    CadAuthoredFaceMesh::from_face(
        graph,
        &face,
        tolerance_m,
        target_m,
        maximum_triangles,
        quality(minimum_quality),
    )
    .unwrap()
}

fn maximum_realized_axis_gap(length: f64, divisions: usize) -> f64 {
    let spacing = length / divisions as f64;
    let mut previous = 0.0;
    let mut maximum = 0.0_f64;
    for index in 1..=divisions {
        let coordinate = if index == divisions {
            length
        } else {
            index as f64 * spacing
        };
        maximum = maximum.max(coordinate - previous);
        previous = coordinate;
    }
    maximum
}

fn maximum_mesh_edge(realization: &CadAuthoredFaceMesh) -> f64 {
    realization
        .mesh()
        .cells()
        .iter()
        .flat_map(|cell| [[cell[0], cell[1]], [cell[1], cell[2]], [cell[2], cell[0]]])
        .map(|edge| {
            let left = &realization.mesh().vertices()[edge[0]];
            let right = &realization.mesh().vertices()[edge[1]];
            (right[0] - left[0]).hypot(right[1] - left[1])
        })
        .fold(0.0_f64, f64::max)
}

#[test]
fn dual_oracle_end_cap_freezes_frame_topology_geometry_and_quality() {
    let graph = graph();
    let realization = realize(&graph, "end-cap", GEOMETRY_TOLERANCE_M, 2.0, 24, 0.95);

    assert_eq!(
        realization.source_graph_digest_bytes(),
        graph.digest_bytes()
    );
    assert_eq!(realization.source_face().provenance_key(), "end-cap");
    assert_eq!(
        realization.geometry_classification_tolerance_m(),
        GEOMETRY_TOLERANCE_M
    );
    assert_eq!(realization.target_maximum_edge_length_m(), 2.0);
    assert_eq!(realization.maximum_triangles(), 24);
    assert_eq!(realization.origin_m(), [-2.0, -1.0, 4.5]);
    assert_eq!(realization.u_hat(), [1.0, 0.0, 0.0]);
    assert_eq!(realization.v_hat(), [0.0, 1.0, 0.0]);
    assert_eq!(realization.parent_outward_normal(), [0.0, 0.0, 1.0]);
    assert_eq!(
        (realization.u_length_m(), realization.v_length_m()),
        (5.0, 3.0)
    );
    assert_eq!(
        (realization.u_divisions(), realization.v_divisions()),
        (4, 3)
    );
    assert_eq!(
        (
            realization.u_maximum_realized_gap_m(),
            realization.v_maximum_realized_gap_m()
        ),
        (1.25, 1.0)
    );
    assert_eq!(
        realization.lift_intrinsic_point_m([0.0, 0.0]),
        [-2.0, -1.0, 4.5]
    );
    assert_eq!(
        realization.lift_intrinsic_point_m([5.0, 3.0]),
        [3.0, 2.0, 4.5]
    );

    let region = realization.region();
    assert_eq!(region.vertices().len(), 4);
    assert_eq!(region.faces().len(), 1);
    assert_eq!(region.edge_count(), 4);
    assert_eq!(region.tolerance_m(), GEOMETRY_TOLERANCE_M);

    let mesh = realization.mesh();
    assert_eq!(mesh.vertices().len(), 20);
    assert_eq!(mesh.cells().len(), 24);
    let edge_count = mesh.entity_count(1).unwrap();
    assert_eq!(edge_count, 43);
    let boundary_edges = (0..edge_count)
        .filter(|&index| mesh.is_boundary_entity(MeshEntity::new(1, index)) == Some(true))
        .count();
    assert_eq!(boundary_edges, 14);
    assert_eq!(edge_count - boundary_edges, 29);
    assert_eq!(
        mesh.vertices().len() as i64 - edge_count as i64 + mesh.cells().len() as i64,
        1
    );
    assert_eq!(mesh.vertices()[0], [0.0, 0.0]);
    assert_eq!(mesh.vertices()[4], [5.0, 0.0]);
    assert_eq!(mesh.vertices()[19], [5.0, 3.0]);
    assert_eq!(mesh.vertices()[15], [0.0, 3.0]);
    assert_eq!(mesh.cells()[0], [1, 6, 0]);
    assert_eq!(mesh.cells()[1], [5, 0, 6]);
    assert_eq!(mesh.cells()[22], [14, 19, 13]);
    assert_eq!(mesh.cells()[23], [18, 13, 19]);
    assert_eq!(mesh.quality_report().minimum_mean_ratio(), 40.0 / 41.0);
    assert_eq!(
        mesh.quality_report().minimum_signed_measure_scale(),
        5.0 / 4.0
    );

    let mut area_m2 = 0.0;
    let mut maximum_edge_m = 0.0_f64;
    let mut unique_edges = BTreeSet::new();
    for cell in mesh.cells() {
        let [a, b, c] = [
            &mesh.vertices()[cell[0]],
            &mesh.vertices()[cell[1]],
            &mesh.vertices()[cell[2]],
        ];
        let signed_double_area =
            (b[0] - a[0]).mul_add(c[1] - a[1], -((c[0] - a[0]) * (b[1] - a[1])));
        assert_eq!(signed_double_area, 5.0 / 4.0);
        area_m2 += signed_double_area / 2.0;
        for edge in [[cell[0], cell[1]], [cell[1], cell[2]], [cell[2], cell[0]]] {
            let mut edge = edge;
            edge.sort_unstable();
            unique_edges.insert(edge);
            let left = &mesh.vertices()[edge[0]];
            let right = &mesh.vertices()[edge[1]];
            maximum_edge_m = maximum_edge_m.max((right[0] - left[0]).hypot(right[1] - left[1]));
        }
    }
    assert_eq!(area_m2, 15.0);
    assert_eq!(unique_edges.len(), 43);
    assert_eq!(maximum_edge_m, 41.0_f64.sqrt() / 4.0);

    let boundary_perimeter_m = unique_edges
        .iter()
        .filter(|edge| {
            let entity = (0..mesh.entity_count(1).unwrap())
                .map(|index| MeshEntity::new(1, index))
                .find(|&candidate| {
                    mesh.entity_vertices(candidate)
                        .unwrap()
                        .iter()
                        .map(|entity| entity.index())
                        .collect::<BTreeSet<_>>()
                        == edge.iter().copied().collect::<BTreeSet<_>>()
                })
                .unwrap();
            mesh.is_boundary_entity(entity) == Some(true)
        })
        .map(|edge| {
            let left = &mesh.vertices()[edge[0]];
            let right = &mesh.vertices()[edge[1]];
            (right[0] - left[0]).hypot(right[1] - left[1])
        })
        .sum::<f64>();
    assert_eq!(boundary_perimeter_m, 16.0);

    let geometry = GeometryDefinitionV1::from_region(realization.region());
    let mesh = SimplicialMeshEnvelopeV1::from_mesh(realization.mesh()).unwrap();
    let correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::from_region(&geometry, &mesh).unwrap();
    correspondence
        .validate_against_region(&geometry, &mesh)
        .unwrap();
}

#[test]
fn start_cap_orientation_is_congruent_but_not_reconstructed_from_global_axes() {
    let graph = graph();
    let end = realize(&graph, "end-cap", GEOMETRY_TOLERANCE_M, 2.0, 24, 0.95);
    let start = realize(&graph, "start-cap", GEOMETRY_TOLERANCE_M, 2.0, 24, 0.95);

    assert_eq!(start.origin_m(), [-2.0, -1.0, 0.5]);
    assert_eq!(start.u_hat(), [0.0, 1.0, 0.0]);
    assert_eq!(start.v_hat(), [1.0, 0.0, 0.0]);
    assert_eq!(start.parent_outward_normal(), [0.0, 0.0, -1.0]);
    assert_eq!((start.u_divisions(), start.v_divisions()), (3, 4));
    assert_eq!(start.mesh().vertices()[1], [1.0, 0.0]);
    assert_eq!(start.mesh().vertices()[3], [3.0, 0.0]);
    assert_eq!(start.mesh().vertices()[19], [3.0, 5.0]);
    assert_eq!(start.mesh().vertices()[16], [0.0, 5.0]);
    assert_eq!(start.mesh().cells()[0], [1, 5, 0]);
    assert_eq!(start.mesh().cells()[1], [4, 0, 5]);
    assert_eq!(start.mesh().cells()[22], [15, 19, 14]);
    assert_eq!(start.mesh().cells()[23], [18, 14, 19]);
    assert_ne!(start.source_face(), end.source_face());
    assert_ne!(start.mesh().vertices(), end.mesh().vertices());
    assert_ne!(start.mesh().cells(), end.mesh().cells());
    assert_eq!(start.lift_intrinsic_point_m([3.0, 5.0]), [3.0, 2.0, 0.5]);
}

#[test]
fn binary64_minimality_boundaries_and_estimate_correction_are_exact() {
    let graph = graph();
    let at = realize(
        &graph,
        "end-cap",
        GEOMETRY_TOLERANCE_M,
        f64::from_bits(0x4002_db2e_aabf_5c80),
        16,
        0.5,
    );
    let below = realize(
        &graph,
        "end-cap",
        GEOMETRY_TOLERANCE_M,
        f64::from_bits(0x4002_db2e_aabf_5c7f),
        16,
        0.5,
    );
    assert_eq!((at.u_divisions(), at.v_divisions()), (3, 2));
    assert_eq!(
        (at.mesh().vertices().len(), at.mesh().cells().len()),
        (12, 12)
    );
    assert_eq!((below.u_divisions(), below.v_divisions()), (4, 2));
    assert_eq!(
        (below.mesh().vertices().len(), below.mesh().cells().len()),
        (15, 16)
    );

    let replacement_length = 4.875;
    let replacement_target = f64::from_bits(0x3fef_844a_57e8_134b);
    assert_eq!(
        (std::f64::consts::SQRT_2 * replacement_length / replacement_target).ceil(),
        7.0
    );
    assert_eq!(
        (replacement_length / (replacement_target / std::f64::consts::SQRT_2)).ceil(),
        7.0
    );
    assert_eq!((replacement_length / 7.0).to_bits(), 0x3fe6_4924_9249_2492);
    let n7_gap = maximum_realized_axis_gap(replacement_length, 7);
    assert_eq!(n7_gap.to_bits(), 0x3fe6_4924_9249_2498);
    assert_eq!(n7_gap.hypot(n7_gap).to_bits(), 0x3fef_844a_57e8_1353);
    assert!(n7_gap.hypot(n7_gap) > replacement_target);

    let replacement_graph = rectangle_graph(
        (0.0, replacement_length),
        (0.0, replacement_length),
        0.0,
        1.0,
        1.0e-9,
    );
    let replacement = realize(
        &replacement_graph,
        "end-cap",
        GEOMETRY_TOLERANCE_M,
        replacement_target,
        128,
        0.5,
    );
    assert_eq!(
        (replacement.u_divisions(), replacement.v_divisions()),
        (8, 8)
    );
    assert_eq!(
        (
            replacement.mesh().vertices().len(),
            replacement.mesh().cells().len()
        ),
        (81, 128)
    );
    assert_eq!(replacement.mesh().entity_count(1), Some(208));
    assert_eq!(
        maximum_mesh_edge(&replacement).to_bits(),
        0x3feb_93c1_0ceb_10e1
    );
    assert!(maximum_mesh_edge(&replacement) <= replacement_target);
}

#[test]
fn retained_regression_witness_corrects_nominal_seven_to_eight() {
    let length = 8.375_f64;
    let target = f64::from_bits(0x3ffb_1274_5f33_a78c);
    assert!((length / 6.0_f64).hypot(length / 6.0_f64) > target);
    assert!((length / 7.0_f64).hypot(length / 7.0_f64) <= target);
    let n7_gap = maximum_realized_axis_gap(length, 7);
    assert_eq!(n7_gap.to_bits(), 0x3ff3_2492_4924_924c);
    assert!(n7_gap.hypot(n7_gap) > target);
    let n8_gap = maximum_realized_axis_gap(length, 8);
    assert!(n8_gap.hypot(n8_gap) <= target);

    let graph = rectangle_graph((0.0, length), (0.0, 1.0), 0.0, 1.0, 1.0e-9);
    let corrected = realize(&graph, "end-cap", GEOMETRY_TOLERANCE_M, target, 16, 0.5);
    assert_eq!((corrected.u_divisions(), corrected.v_divisions()), (8, 1));
}

#[test]
fn both_axes_of_a_non_square_face_snap_independently_before_acceptance() {
    let u_length = 17.5_f64;
    let v_length = 10.5_f64;
    let target = f64::from_bits(0x3fef_adaa_8f7e_ed51);
    assert_eq!((u_length / 25.0).to_bits(), 0x3fe6_6666_6666_6666);
    assert_eq!((v_length / 15.0).to_bits(), 0x3fe6_6666_6666_6666);
    let nominal = u_length / 25.0;
    assert_eq!(nominal.hypot(nominal), target);

    let u_gap = maximum_realized_axis_gap(u_length, 25);
    let v_gap = maximum_realized_axis_gap(v_length, 15);
    assert_eq!(u_gap.to_bits(), 0x3fe6_6666_6666_6680);
    assert_eq!(v_gap.to_bits(), 0x3fe6_6666_6666_6670);
    assert_ne!(u_gap, v_gap);
    assert_eq!(u_gap.hypot(u_gap).to_bits(), 0x3fef_adaa_8f7e_ed76);
    assert_eq!(v_gap.hypot(v_gap).to_bits(), 0x3fef_adaa_8f7e_ed5f);
    assert!(u_gap.hypot(u_gap) > target);
    assert!(v_gap.hypot(v_gap) > target);

    let u_accepted_gap = maximum_realized_axis_gap(u_length, 26);
    let v_accepted_gap = maximum_realized_axis_gap(v_length, 16);
    assert_eq!(u_accepted_gap.to_bits(), 0x3fe5_89d8_9d89_d8a0);
    assert_eq!(v_accepted_gap.to_bits(), 0x3fe5_0000_0000_0000);
    assert_ne!(u_accepted_gap, v_accepted_gap);
    assert!(u_accepted_gap.hypot(u_accepted_gap) <= target);
    assert!(v_accepted_gap.hypot(v_accepted_gap) <= target);

    let graph = rectangle_graph((0.0, u_length), (0.0, v_length), 0.0, 1.0, 1.0e-9);
    let realization = realize(&graph, "end-cap", GEOMETRY_TOLERANCE_M, target, 1000, 0.5);
    assert_eq!(
        (realization.u_divisions(), realization.v_divisions()),
        (26, 16)
    );
    assert_eq!(
        (
            realization.mesh().vertices().len(),
            realization.mesh().cells().len()
        ),
        (459, 832)
    );
    let edge_count = realization.mesh().entity_count(1).unwrap();
    assert_eq!(edge_count, 1290);
    let boundary_edges = (0..edge_count)
        .filter(|&index| {
            realization
                .mesh()
                .is_boundary_entity(MeshEntity::new(1, index))
                == Some(true)
        })
        .count();
    assert_eq!(boundary_edges, 84);
    assert_eq!(edge_count - boundary_edges, 1206);
    // Check the realized geometry without requiring a particular libm hypot rounding.
    let mesh = realization.mesh();
    let mut widest_diagonal_exists = false;
    for edge in mesh
        .cells()
        .iter()
        .flat_map(|cell| [[cell[0], cell[1]], [cell[1], cell[2]], [cell[2], cell[0]]])
    {
        let left = &mesh.vertices()[edge[0]];
        let right = &mesh.vertices()[edge[1]];
        let du = (right[0] - left[0]).abs();
        let dv = (right[1] - left[1]).abs();
        assert!(du <= u_accepted_gap);
        assert!(dv <= v_accepted_gap);
        widest_diagonal_exists |= du == u_accepted_gap && dv == v_accepted_gap;
    }
    assert!(widest_diagonal_exists);
    assert!(maximum_mesh_edge(&realization) <= target);
}

#[test]
fn endpoint_snapping_is_measured_before_a_target_is_accepted() {
    let length = 3.0_f64;
    let target = f64::from_bits(0x3fe3_651a_0eb6_3341);
    let nominal = length / 7.0;
    assert_eq!(nominal.to_bits(), 0x3fdb_6db6_db6d_b6db);
    assert_eq!(nominal.hypot(nominal), target);
    let realized_gap = maximum_realized_axis_gap(length, 7);
    assert_eq!(realized_gap.to_bits(), 0x3fdb_6db6_db6d_b6e0);
    assert_eq!(
        realized_gap.hypot(realized_gap).to_bits(),
        0x3fe3_651a_0eb6_3345
    );
    assert!(realized_gap.hypot(realized_gap) > target);

    let graph = rectangle_graph((0.0, length), (0.0, length), 0.0, 1.0, 1.0e-9);
    let realization = realize(&graph, "end-cap", GEOMETRY_TOLERANCE_M, target, 128, 0.5);
    assert_eq!(
        (realization.u_divisions(), realization.v_divisions()),
        (8, 8)
    );
    assert_eq!(
        (
            realization.mesh().vertices().len(),
            realization.mesh().cells().len()
        ),
        (81, 128)
    );
    let edge_count = realization.mesh().entity_count(1).unwrap();
    assert_eq!(edge_count, 208);
    let boundary_edges = (0..edge_count)
        .filter(|&index| {
            realization
                .mesh()
                .is_boundary_entity(MeshEntity::new(1, index))
                == Some(true)
        })
        .count();
    assert_eq!(boundary_edges, 32);
    assert_eq!(edge_count - boundary_edges, 176);
    assert_eq!(
        maximum_mesh_edge(&realization).to_bits(),
        0x3fe0_f876_ccdf_6cd9
    );
    assert!(maximum_mesh_edge(&realization) <= target);
}

#[test]
fn source_policy_budget_and_quality_falsifiers_fail_closed() {
    let owner = GeometryGraph::new();
    let graph = owner
        .rectangle_extrusion((-2.0, 3.0), (-1.0, 2.0), 0.5, 4.0, 1.0e-9)
        .unwrap();
    let face = graph.face_handle("end-cap").unwrap();
    let accepted =
        CadAuthoredFaceMesh::from_face(&graph, &face, GEOMETRY_TOLERANCE_M, 2.0, 24, quality(0.95))
            .unwrap();
    assert_eq!(accepted.mesh().cells().len(), 24);

    let over_budget =
        CadAuthoredFaceMesh::from_face(&graph, &face, GEOMETRY_TOLERANCE_M, 2.0, 23, quality(0.95))
            .unwrap_err();
    assert_eq!(over_budget.code(), codes::INVALID_ARTIFACT);
    assert!(over_budget.message().contains("24 triangles"));

    let too_strict =
        CadAuthoredFaceMesh::from_face(&graph, &face, GEOMETRY_TOLERANCE_M, 2.0, 24, quality(0.98))
            .unwrap_err();
    assert_eq!(too_strict.code(), codes::INVALID_MESH);

    for target in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            CadAuthoredFaceMesh::from_face(
                &graph,
                &face,
                GEOMETRY_TOLERANCE_M,
                target,
                24,
                quality(0.5),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_ARTIFACT
        );
    }
    for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            CadAuthoredFaceMesh::from_face(&graph, &face, tolerance, 2.0, 24, quality(0.5),)
                .unwrap_err()
                .code(),
            codes::INVALID_ARTIFACT
        );
    }
    for maximum in [0, 1, 100_001] {
        assert_eq!(
            CadAuthoredFaceMesh::from_face(
                &graph,
                &face,
                GEOMETRY_TOLERANCE_M,
                2.0,
                maximum,
                quality(0.5),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_ARTIFACT
        );
    }

    let changed_identity = rectangle_graph((-2.0, 3.0), (-1.0, 2.0), 0.5, 4.0, 2.0e-9);
    let stale = CadAuthoredFaceMesh::from_face(
        &changed_identity,
        &face,
        GEOMETRY_TOLERANCE_M,
        2.0,
        24,
        quality(0.5),
    )
    .unwrap_err();
    assert_eq!(stale.code(), codes::INVALID_ARTIFACT);

    for candidate in graph.face_handles().unwrap() {
        CadAuthoredFaceMesh::from_face(
            &graph,
            &candidate,
            GEOMETRY_TOLERANCE_M,
            100.0,
            2,
            quality(0.05),
        )
        .unwrap();
    }

    let cut = owner
        .circular_through_cut(&graph, [0.0, 0.0], 0.25, 1.0e-9)
        .unwrap();
    assert_eq!(
        CadAuthoredFaceMesh::from_face(&cut, &face, GEOMETRY_TOLERANCE_M, 2.0, 24, quality(0.5),)
            .unwrap_err()
            .code(),
        codes::INVALID_ARTIFACT
    );
    for unsupported in ["start-cap", "end-cap", "cut-wall"] {
        let unsupported = cut.face_handle(unsupported).unwrap();
        assert_eq!(
            CadAuthoredFaceMesh::from_face(
                &cut,
                &unsupported,
                GEOMETRY_TOLERANCE_M,
                100.0,
                2,
                quality(0.05),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_ARTIFACT
        );
    }
    for lateral in [
        "profile-x-lower",
        "profile-x-upper",
        "profile-y-lower",
        "profile-y-upper",
    ] {
        let face = cut.face_handle(lateral).unwrap();
        CadAuthoredFaceMesh::from_face(&cut, &face, GEOMETRY_TOLERANCE_M, 100.0, 2, quality(0.05))
            .unwrap();
    }
}

#[test]
fn fine_target_exceeds_the_axis_division_bound_before_allocation() {
    let graph = graph();
    let face = graph.face_handle("end-cap").unwrap();
    let error =
        CadAuthoredFaceMesh::from_face(&graph, &face, GEOMETRY_TOLERANCE_M, 0.01, 24, quality(0.5))
            .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_ARTIFACT);
    assert_eq!(error.code().0, "EQ0901");
    assert!(error.message().contains("more than 12 divisions"));
}

#[test]
fn geometry_tolerance_and_sizing_requests_retain_distinct_ownership() {
    let graph = graph();
    let baseline = realize(&graph, "end-cap", 5.0e-10, 2.0, 24, 0.95);
    let changed_target = realize(&graph, "end-cap", 5.0e-10, 2.1, 24, 0.95);
    let changed_tolerance = realize(&graph, "end-cap", 6.0e-10, 2.0, 24, 0.95);

    assert_eq!(baseline.source_face(), changed_target.source_face());
    assert_eq!(baseline.source_face(), changed_tolerance.source_face());
    assert_eq!(baseline.mesh(), changed_target.mesh());
    assert_eq!(baseline.mesh(), changed_tolerance.mesh());
    assert_ne!(
        baseline.target_maximum_edge_length_m(),
        changed_target.target_maximum_edge_length_m()
    );

    let baseline_geometry = GeometryDefinitionV1::from_region(baseline.region());
    let target_geometry = GeometryDefinitionV1::from_region(changed_target.region());
    let tolerance_geometry = GeometryDefinitionV1::from_region(changed_tolerance.region());
    assert_eq!(
        baseline_geometry.digest().unwrap(),
        target_geometry.digest().unwrap()
    );
    assert_ne!(
        baseline_geometry.digest().unwrap(),
        tolerance_geometry.digest().unwrap()
    );

    let baseline_mesh = SimplicialMeshEnvelopeV1::from_mesh(baseline.mesh()).unwrap();
    let tolerance_mesh = SimplicialMeshEnvelopeV1::from_mesh(changed_tolerance.mesh()).unwrap();
    assert_eq!(
        baseline_mesh.digest().unwrap(),
        tolerance_mesh.digest().unwrap()
    );
    let baseline_correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::from_region(&baseline_geometry, &baseline_mesh)
            .unwrap();
    let tolerance_correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::from_region(&tolerance_geometry, &tolerance_mesh)
            .unwrap();
    assert_ne!(
        baseline_correspondence.canonical_json().unwrap(),
        tolerance_correspondence.canonical_json().unwrap()
    );
}
