//! Exact compatible cochain projections of an already admitted Field.
use super::*;
use eqiora_core::{Id, entity::kinds};
use eqiora_meshing::{MeshEntity, SimplicialMesh};

impl CommonLinearPlan {
    /// Independent gradient modes in one tetrahedral edge Field's coefficient space.
    ///
    /// Each key is a retained potential vertex; its sparse column contains exact
    /// signed edge integrals of that vertex's piecewise-affine nodal gradient.
    /// The smallest vertex in each connected component is omitted, removing only
    /// redundant constant potentials. Potential coordinates have the same units
    /// as the edge coefficients (Field units times length).
    ///
    /// These columns span the discrete gradient space and are annihilated by curl.
    /// They do not assert that the complete Model operator annihilates gradients,
    /// or that every curl-free mode is a gradient on a multiply connected support.
    /// A gauge or spectral consumer must still choose and verify its constraint.
    /// The entities and orientation belong to this Plan's retained Mesh; no
    /// metric normalization, numerical rank threshold or zero-mode filter is used.
    pub fn field_gradient_modes(
        &self,
        field: Id<kinds::Field>,
    ) -> Result<BTreeMap<MeshEntity, Vec<(MeshEntity, i8)>>, Diagnostic> {
        if self.spatial() != CommonSpatialPolicy::TetrahedralEdge {
            return Err(invalid(
                "gradient modes require an admitted tetrahedral edge Field",
            ));
        }
        let (mesh, entities, _) = self.compatible_topology(field)?;
        gradient_modes(mesh, &entities)
    }

    /// Exact integrated exterior derivative on one admitted edge or face Field.
    ///
    /// Map keys are oriented faces (edge Field) or cells (face Field). Each row
    /// pairs coefficient entities with integer signs from the retained Mesh.
    /// Applied to edge circulations it yields face-integrated curl; applied to
    /// face fluxes it yields cell-integrated divergence. No area/volume division
    /// is implicit: these are integrals, not point values or normalized averages.
    /// A zero row action expresses the corresponding curl/divergence constraint.
    ///
    /// Only the exact Field support participates. This projection is independent
    /// of real/complex coefficient storage and derives from the same bound
    /// topology as assembly; it adds no persisted or user-supplied placement data.
    pub fn field_exterior_derivative(
        &self,
        field: Id<kinds::Field>,
    ) -> Result<BTreeMap<MeshEntity, Vec<(MeshEntity, i8)>>, Diagnostic> {
        let (mesh, entities, cells) = self.compatible_topology(field)?;
        let columns = entities.into_iter().collect::<BTreeSet<_>>();
        let dimension = columns
            .first()
            .ok_or_else(|| invalid("empty moment support"))?
            .dimension()
            + 1;
        let mut rows = BTreeSet::new();
        for cell in cells {
            if dimension == 3 {
                rows.insert(cell);
            } else {
                rows.extend(
                    mesh.incidence(cell, dimension)
                        .ok_or_else(|| invalid("missing compatible cell closure"))?
                        .into_iter()
                        .map(|entry| entry.entity),
                );
            }
        }
        rows.into_iter()
            .map(|row| {
                let mut boundary = mesh
                    .signed_boundary(row)
                    .ok_or_else(|| invalid("missing signed compatible boundary"))?;
                if boundary.iter().any(|(entity, _)| !columns.contains(entity)) {
                    return Err(invalid(
                        "exterior derivative crosses exact Field coefficient support",
                    ));
                }
                boundary.sort_by_key(|(entity, _)| *entity);
                Ok((row, boundary))
            })
            .collect()
    }

    fn compatible_topology(
        &self,
        field: Id<kinds::Field>,
    ) -> Result<(&SimplicialMesh, Vec<MeshEntity>, Vec<MeshEntity>), Diagnostic> {
        self.reauthenticate_portable_realization()?;
        let (
            NativeSpatialPolicy::LinearFiniteElement(space),
            NativeMeshResources::GmshSimplicial { mesh, .. },
        ) = (self.admission.spatial, self.admission.resources())
        else {
            return Err(invalid(
                "compatible incidence requires an admitted tetrahedral moment Field",
            ));
        };
        if !matches!(
            space.family(),
            SpaceFamily::TetrahedralEdge | SpaceFamily::TetrahedralFace
        ) {
            return Err(invalid(
                "compatible incidence requires edge or face moments",
            ));
        }
        let (entities, cells) = match self.admission.recognized_model() {
            RecognizedNativeModel::Linear(equations) => {
                support::simplicial_topology(equations, mesh, field.erase(), space)?
            }
            RecognizedNativeModel::ComplexLinear(equations) => {
                support::simplicial_topology(equations, mesh, field.erase(), space)?
            }
            _ => return Err(invalid("missing compatible Field equations")),
        };
        Ok((mesh.mesh(), entities, cells))
    }
}

fn gradient_modes(
    mesh: &SimplicialMesh,
    edges: &[MeshEntity],
) -> Result<BTreeMap<MeshEntity, Vec<(MeshEntity, i8)>>, Diagnostic> {
    let mut columns = BTreeMap::<_, Vec<_>>::new();
    let mut neighbours = BTreeMap::<_, Vec<_>>::new();
    for &edge in edges {
        if edge.dimension() != 1 {
            return Err(invalid("gradient coordinates must be exact edges"));
        }
        let boundary = mesh
            .signed_boundary(edge)
            .ok_or_else(|| invalid("gradient edge is absent from the exact Mesh"))?;
        let [(a, sa), (b, sb)] = boundary.as_slice() else {
            return Err(invalid("gradient edge must have two oriented endpoints"));
        };
        columns.entry(*a).or_default().push((edge, *sa));
        columns.entry(*b).or_default().push((edge, *sb));
        neighbours.entry(*a).or_default().push(*b);
        neighbours.entry(*b).or_default().push(*a);
    }
    let mut visited = BTreeSet::new();
    for &root in neighbours.keys() {
        if !visited.insert(root) {
            continue;
        }
        columns.remove(&root);
        let mut pending = vec![root];
        while let Some(vertex) = pending.pop() {
            for &next in &neighbours[&vertex] {
                if visited.insert(next) {
                    pending.push(next);
                }
            }
        }
    }
    Ok(columns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_basis_removes_one_constant_per_exact_connected_support() {
        // Two disjoint tetrahedra: each has four vertices and three independent
        // nodal gradients.
        let vertices = vec![
            vec![0., 0., 0.],
            vec![2., 0., 0.],
            vec![0., 3., 0.],
            vec![0., 0., 4.],
            vec![10., 0., 0.],
            vec![12., 0., 0.],
            vec![10., 3., 0.],
            vec![10., 0., 4.],
        ];
        let mesh = SimplicialMesh::new(
            3,
            vertices,
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            eqiora_meshing::MeshQualityGate::new(0.01).unwrap(),
        )
        .unwrap();
        let edges = (0..mesh.entity_count(1).unwrap())
            .map(|i| MeshEntity::new(1, i))
            .collect::<Vec<_>>();
        let modes = gradient_modes(&mesh, &edges).unwrap();
        assert_eq!(
            modes
                .keys()
                .map(|entity| entity.index())
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 5, 6, 7]
        );
        // Restrict to the second component; no first-component coordinate leaks in.
        let selected = mesh
            .incidence(MeshEntity::new(3, 1), 1)
            .unwrap()
            .into_iter()
            .map(|entry| entry.entity)
            .collect::<Vec<_>>();
        let local = gradient_modes(&mesh, &selected).unwrap();
        assert_eq!(
            local
                .keys()
                .map(|entity| entity.index())
                .collect::<Vec<_>>(),
            vec![5, 6, 7]
        );
        assert!(
            local
                .values()
                .flatten()
                .all(|(edge, _)| selected.contains(edge))
        );
        for (vertex, mode) in &modes {
            for face in 0..mesh.entity_count(2).unwrap() {
                let curl = mesh
                    .signed_boundary(MeshEntity::new(2, face))
                    .unwrap()
                    .iter()
                    .map(|(edge, sign)| {
                        i32::from(*sign)
                            * mode
                                .iter()
                                .find(|(column, _)| column == edge)
                                .map_or(0, |(_, v)| i32::from(*v))
                    })
                    .sum::<i32>();
                assert_eq!(curl, 0, "vertex {vertex:?}, face {face}");
            }
        }
    }
}
