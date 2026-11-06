use eqiora_meshing::{CartesianMesh, MeshEntity, MeshTopology};

/// Candidate discovery for one assembly only. Region containment, ownership and
/// boundary evaluation stay with the caller, in the original facet order.
pub(super) struct CartesianFacetIndex<'a> {
    mesh: &'a CartesianMesh,
    planes: Vec<Vec<Vec<MeshEntity>>>,
}

impl<'a> CartesianFacetIndex<'a> {
    pub(super) fn new(mesh: &'a CartesianMesh) -> Self {
        let dimension = mesh.topological_dimension();
        let mut planes = (0..dimension)
            .map(|axis| {
                vec![Vec::new(); mesh.axis_coordinates(axis).expect("Cartesian axis").len()]
            })
            .collect::<Vec<_>>();
        // CartesianMesh construction guarantees finite, strictly increasing axes,
        // nonempty facet vertices and exactly one fixed axis per facet. These
        // accesses add no fallible validation ahead of the caller's diagnostics.
        for index in 0..mesh.entity_count(dimension - 1).expect("Cartesian facets") {
            let facet = MeshEntity::new(dimension - 1, index);
            let free = mesh.entity_free_axes(facet).expect("Cartesian facet axes");
            let axis = (0..dimension)
                .find(|axis| !free.contains(axis))
                .expect("Cartesian facet fixed axis");
            let vertices = mesh
                .entity_vertices(facet)
                .expect("Cartesian facet vertices");
            let point = mesh
                .vertex_coordinates(vertices[0])
                .expect("Cartesian vertex");
            let coordinate = mesh
                .axis_coordinates(axis)
                .expect("Cartesian axis")
                .binary_search_by(|value| {
                    value
                        .partial_cmp(&point[axis])
                        .expect("finite Cartesian coordinate")
                })
                .expect("Cartesian vertex lies on its axis");
            planes[axis][coordinate].push(facet);
        }
        Self { mesh, planes }
    }

    pub(super) fn on_plane(&self, axis: usize, coordinate: f64) -> &[MeshEntity] {
        let Some(values) = self.mesh.axis_coordinates(axis) else {
            return &[];
        };
        if !coordinate.is_finite() {
            return &[];
        }
        // partial_cmp preserves the old equality predicate for +0.0 and -0.0.
        let Ok(index) = values.binary_search_by(|value| {
            value
                .partial_cmp(&coordinate)
                .expect("finite Cartesian coordinate")
        }) else {
            return &[];
        };
        &self.planes[axis][index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selected(
        mesh: &CartesianMesh,
        facets: impl IntoIterator<Item = MeshEntity>,
        axis: usize,
        coordinate: f64,
        bounds: &[[f64; 2]],
    ) -> Vec<MeshEntity> {
        facets
            .into_iter()
            .filter(|facet| {
                mesh.entity_vertices(*facet).unwrap().iter().all(|vertex| {
                    let point = mesh.vertex_coordinates(*vertex).unwrap();
                    point[axis] == coordinate
                        && bounds
                            .iter()
                            .enumerate()
                            .all(|(a, b)| point[a] >= b[0] && point[a] <= b[1])
                })
            })
            .collect()
    }

    #[test]
    fn plane_candidates_preserve_exhaustive_selection_and_order_in_one_to_three_dimensions() {
        for dimension in 1..=3 {
            let axes = vec![vec![-2.0, -0.0, 0.25, 3.0]; dimension];
            let mesh = CartesianMesh::from_axes(axes).unwrap();
            let index = CartesianFacetIndex::new(&mesh);
            let full = vec![[-2.0, 3.0]; dimension];
            let partial = vec![[0.0, 0.25]; dimension];
            for axis in 0..dimension {
                for coordinate in [-2.0, -0.0, 0.0, 0.25, 3.0, 0.5] {
                    for bounds in [&full, &partial] {
                        let actual = selected(
                            &mesh,
                            index.on_plane(axis, coordinate).iter().copied(),
                            axis,
                            coordinate,
                            bounds,
                        );
                        // The former exhaustive caller predicate is a reference
                        // selection, rather than an expected numerical solution.
                        let exhaustive = selected(
                            &mesh,
                            (0..mesh.entity_count(dimension - 1).unwrap())
                                .map(|i| MeshEntity::new(dimension - 1, i)),
                            axis,
                            coordinate,
                            bounds,
                        );
                        assert_eq!(actual, exhaustive);
                        assert!(
                            actual
                                .windows(2)
                                .all(|pair| pair[0].index() < pair[1].index())
                        );
                    }
                    // A plane through a 3-cell-per-axis grid has 3^(d-1)
                    // facets, independently of the mesh's entity numbering.
                    let expected = if coordinate == 0.5 {
                        0
                    } else {
                        3_usize.pow(dimension as u32 - 1)
                    };
                    assert_eq!(index.on_plane(axis, coordinate).len(), expected);
                }
                assert_eq!(index.on_plane(axis, -0.0), index.on_plane(axis, 0.0));
                assert!(index.on_plane(axis, f64::NAN).is_empty());
                assert!(index.on_plane(axis, f64::INFINITY).is_empty());
            }
        }
    }

    #[test]
    fn interior_planes_keep_both_parents_for_the_existing_region_ownership_check() {
        for dimension in 1..=3 {
            let mesh = CartesianMesh::from_axes(vec![vec![-1.0, -0.0, 2.0]; dimension]).unwrap();
            let index = CartesianFacetIndex::new(&mesh);
            assert_eq!(
                index.on_plane(0, 0.0).len(),
                2_usize.pow(dimension as u32 - 1)
            );
            for &facet in index.on_plane(0, 0.0) {
                let parents = mesh.incidence(facet, dimension).unwrap();
                assert_eq!(parents.len(), 2);
                for lower in [true, false] {
                    let owned = parents
                        .iter()
                        .filter(|parent| {
                            let center = mesh.entity_center(parent.entity).unwrap();
                            (center[0] < 0.0) == lower
                        })
                        .count();
                    assert_eq!(owned, 1);
                }
            }
        }
    }
}
