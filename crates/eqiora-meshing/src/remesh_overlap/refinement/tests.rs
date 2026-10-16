use super::*;
use crate::{MeshGeometry, MeshQualityGate};

struct TestRefinement {
    mesh: SimplicialMesh,
    parent_cells: Vec<usize>,
    bisected_edges: Vec<[usize; 2]>,
}

fn refined(
    source: &SimplicialMesh,
    marked: &[usize],
    limit: usize,
) -> Result<TestRefinement, Diagnostic> {
    let (mesh, overlap) = source.bisect_marked(
        OverlapCoordinateChart2d::Material,
        &marked.iter().copied().map(CellId::new).collect::<Vec<_>>(),
        limit,
    )?;
    assert_eq!(overlap.cell_fragments().len(), mesh.cells().len());
    let mut parent_cells = vec![usize::MAX; mesh.cells().len()];
    for fragment in overlap.cell_fragments() {
        assert_eq!(parent_cells[fragment.target_cell().index()], usize::MAX);
        parent_cells[fragment.target_cell().index()] = fragment.source_cell().index();
    }
    // Independent midpoint reconstruction is only a fixture oracle; production
    // retains native overlap fragments, not this coordinate lookup.
    let edges = (0..source.entity_count(1).unwrap())
        .map(|i| {
            let v = source.entity_vertices(MeshEntity::new(1, i)).unwrap();
            edge(v[0].index(), v[1].index())
        })
        .collect::<Vec<_>>();
    let bisected_edges = mesh.vertices()[source.vertices().len()..]
        .iter()
        .map(|point| {
            *edges
                .iter()
                .find(|&&[a, b]| {
                    (0..2).all(|axis| {
                        point[axis]
                            == source.vertices()[a][axis].midpoint(source.vertices()[b][axis])
                    })
                })
                .expect("fixture midpoint belongs to one source edge")
        })
        .collect();
    Ok(TestRefinement {
        mesh,
        parent_cells,
        bisected_edges,
    })
}

fn grid(n: usize) -> SimplicialMesh {
    let vertices = (0..=n)
        .flat_map(|x| (0..=n).map(move |y| vec![x as f64 / n as f64, y as f64 / n as f64]))
        .collect();
    let v = |x, y| x * (n + 1) + y;
    let mut cells = vec![];
    for x in 0..n {
        for y in 0..n {
            cells.push(vec![v(x, y), v(x + 1, y), v(x + 1, y + 1)]);
            cells.push(vec![v(x, y), v(x + 1, y + 1), v(x, y + 1)]);
        }
    }
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.01).unwrap()).unwrap()
}
fn area(mesh: &SimplicialMesh, cell: usize) -> f64 {
    mesh.geometry_map(MeshEntity::new(2, cell))
        .unwrap()
        .measure_scale()
        / 2.
}
fn assert_coverage(source: &SimplicialMesh, r: &TestRefinement) {
    let mut area_sums = vec![0.; source.cells().len()];
    for (child, &parent) in r.parent_cells.iter().enumerate() {
        area_sums[parent] += area(&r.mesh, child);
    }
    for (parent, sum) in area_sums.iter().enumerate() {
        assert!((sum - area(source, parent)).abs() < 1e-14);
    }
    // Boundary ancestry uses the old edge identity and its exact midpoint,
    // never a floating-point coordinate classification.
    let split = r
        .bisected_edges
        .iter()
        .enumerate()
        .map(|(i, &e)| (e, source.vertices().len() + i))
        .collect::<BTreeMap<_, _>>();
    let mut expected_boundary = std::collections::BTreeSet::new();
    for index in 0..source.entity_count(1).unwrap() {
        let e = MeshEntity::new(1, index);
        if source.is_boundary_entity(e) != Some(true) {
            continue;
        }
        let v = source.entity_vertices(e).unwrap();
        let [a, b] = edge(v[0].index(), v[1].index());
        if let Some(&m) = split.get(&[a, b]) {
            expected_boundary.insert(edge(a, m));
            expected_boundary.insert(edge(m, b));
        } else {
            expected_boundary.insert([a, b]);
        }
    }
    let actual_boundary = (0..r.mesh.entity_count(1).unwrap())
        .filter_map(|index| {
            let e = MeshEntity::new(1, index);
            if r.mesh.is_boundary_entity(e) != Some(true) {
                return None;
            }
            let v = r.mesh.entity_vertices(e).unwrap();
            Some(edge(v[0].index(), v[1].index()))
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(expected_boundary, actual_boundary);
    for (offset, &[a, b]) in r.bisected_edges.iter().enumerate() {
        let mid = source.vertices().len() + offset;
        // An edge is split on every incident cell; no old uncut segment remains.
        assert!(
            !r.mesh
                .cells()
                .iter()
                .any(|c| c.contains(&a) && c.contains(&b))
        );
        assert!(
            r.mesh
                .cells()
                .iter()
                .any(|c| c.contains(&mid) && c.contains(&a))
        );
        assert!(
            r.mesh
                .cells()
                .iter()
                .any(|c| c.contains(&mid) && c.contains(&b))
        );
    }
}
#[test]
fn one_triangle_mark_has_exact_conforming_diamond_closure() {
    let source = grid(2);
    let r = refined(&source, &[0], 100).unwrap();
    assert_eq!(r.mesh.vertices().len(), 10);
    assert_eq!(r.mesh.cells().len(), 10);
    assert_eq!(r.bisected_edges, vec![[0, 4]]);
    assert_eq!(r.parent_cells, vec![0, 0, 1, 1, 2, 3, 4, 5, 6, 7]);
    assert_coverage(&source, &r);
    assert_eq!(refined(&source, &[0, 0], 100).unwrap().mesh, r.mesh);
    assert!(refined(&source, &[0], 9).is_err());
    assert!(refined(&source, &[8], 100).is_err());
    assert_eq!(refined(&source, &[], 100).unwrap().mesh, source);
}
#[test]
fn completion_propagates_to_a_different_neighbor_reference_edge() {
    let source = SimplicialMesh::new(
        2,
        vec![vec![0., 0.], vec![1., -0.1], vec![2., 0.], vec![0., 3.]],
        vec![vec![0, 1, 2], vec![0, 2, 3]],
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap();
    // Cell0 requests diagonal02. Its neighbor's longest edge is23,
    // so conformity requires two bisections of that neighbor.
    let r = refined(&source, &[0], 5).unwrap();
    assert_eq!(r.bisected_edges, vec![[0, 2], [2, 3]]);
    assert_eq!(r.parent_cells, vec![0, 0, 1, 1, 1]);
    assert_eq!(r.mesh.vertices()[4], vec![1., 0.]);
    assert_eq!(r.mesh.vertices()[5], vec![1., 1.5]);
    assert_coverage(&source, &r);
    assert!(refined(&source, &[0], 4).is_err());
    // Exact P1 prolongation follows source-edge averages, including
    // non-affine nodal data. Each child has the same source gradient.
    let old = [2., -1., 7., 4.];
    let mut transferred = old.to_vec();
    transferred.extend(
        r.bisected_edges
            .iter()
            .map(|&[a, b]| (old[a] + old[b]) / 2.),
    );
    assert_eq!(transferred, vec![2., -1., 7., 4., 4.5, 5.5]);
    for (child, &parent) in r.parent_cells.iter().enumerate() {
        let coarse = &source.cells()[parent];
        let a = &source.vertices()[coarse[0]];
        let b = &source.vertices()[coarse[1]];
        let c = &source.vertices()[coarse[2]];
        let det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        for &v in &r.mesh.cells()[child] {
            let p = &r.mesh.vertices()[v];
            let l1 = ((p[0] - a[0]) * (c[1] - a[1]) - (p[1] - a[1]) * (c[0] - a[0])) / det;
            let l2 = ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) / det;
            assert!(
                (transferred[v]
                    - ((1. - l1 - l2) * old[coarse[0]]
                        + l1 * old[coarse[1]]
                        + l2 * old[coarse[2]]))
                    .abs()
                    < 1e-13
            );
        }
    }
}
#[test]
fn all_marked_subsets_and_repeated_local_refinement_preserve_coverage() {
    let source = grid(2);
    for mask in 0..256 {
        let marked = (0..8).filter(|i| mask & (1 << i) != 0).collect::<Vec<_>>();
        assert_coverage(&source, &refined(&source, &marked, 32).unwrap());
    }
    let mut source = grid(4);
    for _ in 0..8 {
        let marked = source
            .cells()
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.iter().map(|&i| source.vertices()[i][0]).sum::<f64>() / 3. < 0.2
                    && c.iter().map(|&i| source.vertices()[i][1]).sum::<f64>() / 3. < 0.2
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let r = refined(&source, &marked, 10000).unwrap();
        assert_coverage(&source, &r);
        // All descendants of these right-isosceles triangles are similar.
        // 2*abs(det J)/||J||_F^2 is 1 at the right-angle origin and
        // 2/3 at either acute-angle origin.
        assert!(r.mesh.quality_report().minimum_mean_ratio() >= 2. / 3. - 1e-13);
        source = r.mesh;
    }
}

fn exterior_sides(mesh: &SimplicialMesh) -> Vec<RetainedFacetSide2d> {
    (0..mesh.entity_count(1).unwrap())
        .filter_map(|index| {
            let facet = MeshEntity::new(1, index);
            if mesh.is_boundary_entity(facet) != Some(true) {
                return None;
            }
            let parents = mesh.incidence(facet, 2).unwrap();
            assert_eq!(parents.len(), 1);
            Some(RetainedFacetSide2d::new(
                FacetId::new(index),
                CellId::new(parents[0].entity.index()),
            ))
        })
        .collect()
}

fn region_sides(mesh: &SimplicialMesh, cells: &[CellId]) -> Vec<RetainedFacetSide2d> {
    (0..mesh.entity_count(1).unwrap())
        .filter_map(|index| {
            let parents = mesh
                .incidence(MeshEntity::new(1, index), 2)
                .unwrap()
                .iter()
                .map(|entry| CellId::new(entry.entity.index()))
                .filter(|cell| cells.contains(cell))
                .collect::<Vec<_>>();
            match parents.as_slice() {
                &[parent] => Some(RetainedFacetSide2d::new(FacetId::new(index), parent)),
                _ => None,
            }
        })
        .collect()
}

#[test]
fn selected_refinement_preserves_internal_frontiers_and_rejects_missing_children() {
    let source = grid(2);
    let (target, overlap) = source
        .bisect_marked(OverlapCoordinateChart2d::Material, &[CellId::new(0)], 32)
        .unwrap();
    let source_cells = [CellId::new(0)];
    let target_cells = [CellId::new(0), CellId::new(1)];
    let selected = overlap.select_cells(&source_cells, &target_cells).unwrap();
    let intersection = SimplicialRevisionOverlap2d::new(
        overlap.chart(),
        &source,
        &source_cells,
        &target,
        &target_cells,
    )
    .unwrap();
    assert_eq!(selected, intersection);
    assert_eq!(
        selected,
        overlap
            .select_cells(&source_cells, &[target_cells[1], target_cells[0]])
            .unwrap()
    );
    let source_sides = region_sides(&source, &source_cells);
    let target_sides = region_sides(&target, &target_cells);
    // This region includes internal mesh facets. The whole-mesh witness cannot
    // authenticate them as frontiers until its exact parent region is selected.
    assert!(
        overlap
            .clone()
            .with_retained_facets(&source, &source_sides, &target, &target_sides,)
            .is_err()
    );
    let attached = selected
        .with_retained_facets(&source, &source_sides, &target, &target_sides)
        .unwrap();
    assert_eq!(
        attached,
        intersection
            .with_retained_facets(&source, &source_sides, &target, &target_sides,)
            .unwrap()
    );
    assert!(
        attached
            .select_cells(&source_cells, &target_cells)
            .unwrap()
            .retained_facet_fragments()
            .is_empty()
    );
    for invalid in [
        vec![],
        vec![CellId::new(0)],
        vec![CellId::new(0), CellId::new(2)],
        vec![CellId::new(0), CellId::new(1), CellId::new(2)],
        vec![CellId::new(0), CellId::new(0)],
        vec![CellId::new(99)],
    ] {
        assert!(overlap.select_cells(&source_cells, &invalid).is_err());
    }
    assert!(
        attached
            .select_cells(&[CellId::new(1)], &target_cells)
            .is_err()
    );
}

#[test]
fn bisection_reuses_the_same_overlap_and_boundary_witness_as_geometric_intersection() {
    let source = grid(2);
    let (target, overlap) = source
        .bisect_marked(OverlapCoordinateChart2d::Material, &[CellId::new(0)], 32)
        .unwrap();
    let source_sides = exterior_sides(&source);
    let target_sides = exterior_sides(&target);
    let intersection = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        &source,
        &(0..source.cells().len())
            .map(CellId::new)
            .collect::<Vec<_>>(),
        &target,
        &(0..target.cells().len())
            .map(CellId::new)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(overlap, intersection);
    let attached = overlap
        .clone()
        .with_retained_facets(&source, &source_sides, &target, &target_sides)
        .unwrap();
    assert_eq!(
        attached,
        intersection
            .with_retained_facets(&source, &source_sides, &target, &target_sides)
            .unwrap()
    );
    assert!(
        overlap
            .clone()
            .with_retained_facets(&source, &source_sides, &target, &target_sides[1..])
            .is_err()
    );
    let mut shifted = source.vertices().to_vec();
    for p in &mut shifted {
        p[0] += 0.125;
    }
    let stale =
        SimplicialMesh::new(2, shifted, source.cells().to_vec(), source.quality_gate()).unwrap();
    assert!(
        overlap
            .with_retained_facets(&stale, &source_sides, &target, &target_sides)
            .is_err()
    );
}

#[test]
fn bisection_rejects_a_midpoint_collapsed_by_coordinate_precision() {
    // With one ULP between coordinates near 1, the diagonal midpoint rounds
    // back to the even endpoint. No distinct source-edge midpoint exists.
    let next = 1.0 + f64::EPSILON;
    let source = SimplicialMesh::new(
        2,
        vec![vec![1., 1.], vec![next, 1.], vec![next, next]],
        vec![vec![0, 1, 2]],
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap();
    let error = source
        .bisect_marked(OverlapCoordinateChart2d::Material, &[CellId::new(0)], 2)
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("midpoint cannot be represented distinctly")
    );
}
