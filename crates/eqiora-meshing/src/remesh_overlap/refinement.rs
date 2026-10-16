//! Conforming triangle bisection with an existing revision-overlap witness.
//!
//! Longest-edge labeling followed by edge completion and newest-vertex children:
//! <https://webapps.math.uci.edu/~chenlong/ifemdoc/bisectdoc.html>

use super::*;
use std::collections::{BTreeMap, VecDeque};

#[cfg(test)]
mod tests;

#[derive(Debug)]
struct TriangleRefinement {
    mesh: SimplicialMesh,
    // Target cells retain exact source-cell ancestry.
    parent_cells: Vec<usize>,
}

fn edge(a: usize, b: usize) -> [usize; 2] {
    [a.min(b), a.max(b)]
}

fn refine(
    source: &SimplicialMesh,
    marked: &[usize],
    maximum_cells: usize,
) -> Result<TriangleRefinement, Diagnostic> {
    if source.topological_dimension() != 2 {
        return Err(invalid_overlap("requires planar triangles"));
    }
    if marked.iter().any(|&cell| cell >= source.cells().len()) {
        return Err(invalid_overlap("marked cell is outside source mesh"));
    }
    if source.cells().len() > maximum_cells {
        return Err(invalid_overlap("source exceeds cell limit"));
    }
    if marked.is_empty() {
        return Ok(TriangleRefinement {
            mesh: source.clone(),
            parent_cells: (0..source.cells().len()).collect(),
        });
    }
    let edge_count = source.entity_count(1).expect("triangle edges");
    let edges = (0..edge_count)
        .map(|index| {
            let vertices = source
                .entity_vertices(MeshEntity::new(1, index))
                .expect("edge vertices");
            edge(vertices[0].index(), vertices[1].index())
        })
        .collect::<Vec<_>>();
    let mut edge_cells = vec![Vec::new(); edge_count];
    let mut reference_edges = Vec::with_capacity(source.cells().len());
    let mut labeled = Vec::with_capacity(source.cells().len());
    for (cell_index, cell) in source.cells().iter().enumerate() {
        let incidence = source
            .incidence(MeshEntity::new(2, cell_index), 1)
            .expect("cell edges");
        let mut longest = None;
        for incidence in incidence {
            let id = incidence.entity.index();
            edge_cells[id].push(cell_index);
            let [a, b] = edges[id];
            let p = &source.vertices()[a];
            let q = &source.vertices()[b];
            let length = (p[0] - q[0]).hypot(p[1] - q[1]);
            if longest
                .is_none_or(|(best, best_id)| length > best || (length == best && id < best_id))
            {
                longest = Some((length, id));
            }
        }
        let id = longest.expect("triangle has edges").1;
        reference_edges.push(id);
        let peak = cell
            .iter()
            .position(|v| !edges[id].contains(v))
            .expect("opposite vertex");
        labeled.push([cell[peak], cell[(peak + 1) % 3], cell[(peak + 2) % 3]]);
    }
    let mut cut = vec![false; edge_count];
    let mut pending = VecDeque::new();
    for &cell in marked {
        pending.push_back(reference_edges[cell]);
    }
    while let Some(id) = pending.pop_front() {
        if cut[id] {
            continue;
        }
        cut[id] = true;
        for &cell in &edge_cells[id] {
            pending.push_back(reference_edges[cell]);
        }
    }
    let target_cells = cut.iter().enumerate().filter(|(_, c)| **c).try_fold(
        source.cells().len(),
        |count, (i, _)| {
            count
                .checked_add(edge_cells[i].len())
                .ok_or_else(|| invalid_overlap("refined cell count overflows"))
        },
    )?;
    if target_cells > maximum_cells {
        return Err(invalid_overlap("conforming completion exceeds cell limit"));
    }
    let mut vertices = source.vertices().to_vec();
    let mut midpoint = BTreeMap::new();
    for (id, &[a, b]) in edges.iter().enumerate() {
        if !cut[id] {
            continue;
        }
        let p = &source.vertices()[a];
        let q = &source.vertices()[b];
        let point = certified_line_point(
            &[[p[0], p[1]], [q[0], q[1]]],
            [p[0].midpoint(q[0]), p[1].midpoint(q[1])],
        )?
        .to_vec();
        if point == *p || point == *q {
            return Err(invalid_overlap(
                "edge midpoint cannot be represented distinctly",
            ));
        }
        midpoint.insert(edge(a, b), vertices.len());
        vertices.push(point);
    }
    let mut cells = Vec::with_capacity(target_cells);
    let mut parent_cells = Vec::with_capacity(target_cells);
    for (parent, triangle) in labeled.into_iter().enumerate() {
        if !cut[reference_edges[parent]] {
            cells.push(source.cells()[parent].clone());
            parent_cells.push(parent);
            continue;
        }
        let mut pending = vec![triangle];
        while let Some([a, b, c]) = pending.pop() {
            if let Some(&m) = midpoint.get(&edge(b, c)) {
                // Positive children with the inserted midpoint as their newest vertex.
                pending.push([m, c, a]);
                pending.push([m, a, b]);
            } else {
                cells.push(vec![a, b, c]);
                parent_cells.push(parent);
            }
        }
    }
    if cells.len() != target_cells {
        return Err(invalid_overlap(
            "refinement did not consume the exact edge closure",
        ));
    }
    let mesh = SimplicialMesh::new(2, vertices, cells, source.quality_gate())?;
    Ok(TriangleRefinement { mesh, parent_cells })
}

impl SimplicialMesh {
    /// Bisect selected triangles and close neighboring edges to preserve conformity.
    ///
    /// Each stage labels the longest edge of each source triangle, breaking ties
    /// by canonical edge identity. Completion marks incident triangles' reference
    /// edges; children use their new midpoint as the newest vertex. Existing
    /// vertices remain a prefix, and unchanged cells retain their orientation.
    /// Mark order and repeated marks do not change the result.
    ///
    /// Returns the accepted target Mesh and the ordinary common-refinement witness.
    /// Its cell fragments retain each child's exact source cell. Use
    /// `SimplicialRevisionOverlap2d::with_retained_facets` to verify a requested
    /// semantic boundary relation. Coordinates use binary64 midpoint construction
    /// with the same rounding-cell certificates as other revision overlaps.
    ///
    /// # Errors
    /// Rejects nonplanar meshes, out-of-range cells, an insufficient completion budget,
    /// unrepresentable midpoints, a quality-gate failure or incomplete coverage.
    pub fn bisect_marked(
        &self,
        chart: OverlapCoordinateChart2d,
        marked: &[CellId],
        maximum_cells: usize,
    ) -> Result<(Self, SimplicialRevisionOverlap2d), Diagnostic> {
        let marked = marked.iter().map(|cell| cell.index()).collect::<Vec<_>>();
        let refinement = refine(self, &marked, maximum_cells.min(MAX_CELL_FRAGMENTS))?;
        let source_cells = (0..self.cells().len()).map(CellId::new).collect::<Vec<_>>();
        let target_cells = (0..refinement.mesh.cells().len())
            .map(CellId::new)
            .collect::<Vec<_>>();
        let source_triangles = selected_triangles(self, &source_cells, "source")?;
        let target_triangles = selected_triangles(&refinement.mesh, &target_cells, "target")?;
        let mut fragments = refinement
            .parent_cells
            .iter()
            .enumerate()
            .map(|(child, &parent)| {
                let mut triangle = target_triangles[child];
                // Fragment order is canonical geometry, independent of the
                // target cell's local affine-map origin.
                let first = (0..3)
                    .min_by(|&a, &b| compare_points(&triangle[a], &triangle[b]))
                    .expect("triangle vertices");
                triangle.rotate_left(first);
                let (area, first_moment) = triangle_measure(&triangle)?;
                Ok(RevisionCellFragment2d {
                    source_cell: CellId::new(parent),
                    target_cell: CellId::new(child),
                    triangle,
                    area,
                    first_moment,
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        fragments.sort_by(compare_cell_fragments);
        require_unique_cell_fragments(&fragments)?;
        validate_cell_coverage(
            &source_cells,
            &source_triangles,
            &target_cells,
            &target_triangles,
            &fragments,
        )?;
        let overlap = SimplicialRevisionOverlap2d {
            chart,
            source_cells,
            target_cells,
            source_triangles,
            target_triangles,
            cell_fragments: fragments,
            source_retained_facets: Vec::new(),
            target_retained_facets: Vec::new(),
            retained_facet_fragments: Vec::new(),
        };
        Ok((refinement.mesh, overlap))
    }
}
