use super::{MeshEntity, MeshTopology, SimplicialMesh, invalid_mesh};
use eqiora_core::Diagnostic;

impl SimplicialMesh {
    /// Check that top-cell orientations cancel on every shared codimension-one facet.
    ///
    /// This explicit check does not change constructor admission or establish
    /// boundary coverage. Geometry-bound import combines it with complete source
    /// correspondence. Signs come from the existing exact boundary operator.
    ///
    /// # Errors
    /// Returns `EQ0803` for missing incidence, non-cancelling shared facets or an
    /// exposed facet whose signed multiplicity is not one.
    pub fn validate_signed_boundary_cancellation(&self) -> Result<(), Diagnostic> {
        let dimension = self.topological_dimension();
        let facet_dimension = dimension
            .checked_sub(1)
            .ok_or_else(|| invalid_mesh("signed cell boundary requires positive dimension"))?;
        let facets = self
            .entity_count(facet_dimension)
            .ok_or_else(|| invalid_mesh("mesh omitted its facet stratum"))?;
        let cells = self
            .entity_count(dimension)
            .ok_or_else(|| invalid_mesh("mesh omitted its cell stratum"))?;
        let mut signed = vec![0_i8; facets];
        for cell in 0..cells {
            let boundary = self
                .signed_boundary(MeshEntity::new(dimension, cell))
                .ok_or_else(|| invalid_mesh("cell omitted signed boundary"))?;
            for (facet, coefficient) in boundary {
                let sum = signed
                    .get_mut(facet.index())
                    .ok_or_else(|| invalid_mesh("signed boundary references an absent facet"))?;
                *sum = sum
                    .checked_add(coefficient)
                    .ok_or_else(|| invalid_mesh("signed facet incidence overflows"))?;
            }
        }
        for (facet, &sum) in signed.iter().enumerate() {
            let parents = self
                .incidence(MeshEntity::new(facet_dimension, facet), dimension)
                .ok_or_else(|| invalid_mesh("facet omitted parent incidence"))?;
            match parents.len() {
                1 if matches!(sum, -1 | 1) => {}
                2 if sum == 0 => {}
                _ => {
                    return Err(invalid_mesh(
                        "interior cell boundary orientations must cancel and exposed facets must occur once",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Signed codimension-one boundary in this exact mesh's entity ordering.
    ///
    /// A top-dimensional cell retains its supplied vertex orientation; lower
    /// entities use the mesh's canonical vertex ordering. Coefficients are exact
    /// integers. Consecutive boundary operators compose to zero, independently
    /// of physical coordinates and floating-point differentiation. Transposes
    /// give the oriented vertex/edge/face cochain incidence maps.
    ///
    /// Returns `None` for an absent entity. A vertex has an empty boundary.
    #[must_use]
    pub fn signed_boundary(&self, entity: MeshEntity) -> Option<Vec<(MeshEntity, i8)>> {
        let vertices = self.entity_vertex_indices(entity)?;
        if entity.dimension() == 0 {
            return Some(Vec::new());
        }
        self.lower_incidence(entity, entity.dimension() - 1)?
            .into_iter()
            .map(|entry| {
                let face_vertices = self.entity_vertex_indices(entry.entity)?;
                let omitted = vertices
                    .iter()
                    .position(|vertex| !face_vertices.contains(vertex))?;
                let boundary_sign = if omitted % 2 == 0 { 1 } else { -1 };
                let permutation =
                    self.orientation_permutation(entry.orientation, face_vertices.len())?;
                Some((entry.entity, boundary_sign * permutation.sign()))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeshQualityGate, MeshTopology};
    use std::collections::BTreeMap;

    fn mesh(cells: Vec<Vec<usize>>) -> SimplicialMesh {
        SimplicialMesh::new(
            3,
            vec![
                vec![0.0, 0.0, 0.0],
                vec![2.0, 0.0, 0.0],
                vec![0.0, 3.0, 0.0],
                vec![0.0, 0.0, 4.0],
                vec![2.0, 3.0, 4.0],
            ],
            cells,
            MeshQualityGate::new(0.01).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn exact_curl_grad_and_div_curl_closure_survive_cell_permutations() {
        for cells in [
            vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4]],
            vec![vec![1, 0, 3, 2], vec![2, 1, 4, 3]],
        ] {
            let mesh = mesh(cells);
            assert_eq!(
                (0..=3)
                    .map(|d| mesh.entity_count(d).unwrap())
                    .collect::<Vec<_>>(),
                [5, 9, 7, 2]
            );
            for dimension in 2..=3 {
                for index in 0..mesh.entity_count(dimension).unwrap() {
                    let mut composition = BTreeMap::<MeshEntity, i32>::new();
                    for (face, sign) in mesh
                        .signed_boundary(MeshEntity::new(dimension, index))
                        .unwrap()
                    {
                        for (edge, inner) in mesh.signed_boundary(face).unwrap() {
                            *composition.entry(edge).or_default() +=
                                i32::from(sign) * i32::from(inner);
                        }
                    }
                    assert!(composition.values().all(|coefficient| *coefficient == 0));
                }
            }
            let shared = (0..7)
                .map(|index| MeshEntity::new(2, index))
                .find(|&face| {
                    mesh.entity_vertices(face)
                        .unwrap()
                        .iter()
                        .map(|v| v.index())
                        .collect::<Vec<_>>()
                        == [1, 2, 3]
                })
                .unwrap();
            // Positive cells lie on opposite sides of this face. Their outward
            // orientations are opposite, separately from any flux value.
            for (cell, expected) in [(0, 1), (1, -1)] {
                assert_eq!(
                    mesh.signed_boundary(MeshEntity::new(3, cell))
                        .unwrap()
                        .iter()
                        .find(|(face, _)| *face == shared)
                        .unwrap()
                        .1,
                    expected
                );
            }
        }
    }

    #[test]
    fn boundary_rows_have_independently_known_simplex_signs() {
        let mesh = mesh(vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4]]);
        let rows = mesh.signed_boundary(MeshEntity::new(3, 0)).unwrap();
        let closures = rows
            .iter()
            .map(|(face, sign)| {
                (
                    mesh.entity_vertices(*face)
                        .unwrap()
                        .iter()
                        .map(|v| v.index())
                        .collect::<Vec<_>>(),
                    *sign,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            closures,
            [
                (vec![0, 1, 2], -1),
                (vec![0, 1, 3], 1),
                (vec![0, 2, 3], -1),
                (vec![1, 2, 3], 1)
            ]
        );
        let edge = MeshEntity::new(1, 0);
        assert_eq!(
            mesh.signed_boundary(edge).unwrap(),
            [(MeshEntity::new(0, 0), -1), (MeshEntity::new(0, 1), 1)]
        );
        assert_eq!(mesh.signed_boundary(MeshEntity::new(0, 0)), Some(vec![]));
        assert!(mesh.signed_boundary(MeshEntity::new(3, 2)).is_none());
        assert!(mesh.signed_boundary(MeshEntity::new(4, 0)).is_none());
    }
    #[test]
    fn explicit_signed_cancellation_accepts_valid_shared_interfaces_in_each_dimension() {
        use crate::MeshQualityGate;
        for (dimension, vertices, cells) in [
            (
                1,
                vec![vec![0.], vec![1.], vec![2.]],
                vec![vec![0, 1], vec![1, 2]],
            ),
            (
                2,
                vec![vec![0., 0.], vec![1., 0.], vec![1., 1.], vec![0., 1.]],
                vec![vec![0, 1, 2], vec![0, 2, 3]],
            ),
        ] {
            SimplicialMesh::new(
                dimension,
                vertices,
                cells,
                MeshQualityGate::new(0.01).unwrap(),
            )
            .unwrap()
            .validate_signed_boundary_cancellation()
            .unwrap();
        }
        for cells in [
            vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4]],
            vec![vec![1, 0, 3, 2], vec![2, 1, 4, 3]],
        ] {
            mesh(cells).validate_signed_boundary_cancellation().unwrap();
        }
    }

    #[test]
    fn explicit_signed_cancellation_rejects_positive_closed_overlapping_complex() {
        use crate::MeshQualityGate;
        let overlapping = SimplicialMesh::new(
            2,
            vec![
                vec![0., 0.],
                vec![1., 0.],
                vec![1., 1.],
                vec![0., 1.],
                vec![0.25, 0.25],
                vec![0.75, 0.25],
                vec![0.75, 0.75],
                vec![0.25, 0.75],
            ],
            vec![
                vec![0, 1, 2],
                vec![0, 2, 3],
                vec![4, 5, 6],
                vec![4, 6, 7],
                vec![4, 5, 7],
                vec![5, 6, 7],
            ],
            MeshQualityGate::new(0.01).unwrap(),
        )
        .unwrap();
        assert_eq!(overlapping.cells().len(), 6);
        assert!(overlapping.validate_signed_boundary_cancellation().is_err());
    }
}
