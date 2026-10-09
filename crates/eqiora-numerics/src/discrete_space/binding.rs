use eqiora_meshing::{MeshEntity, MeshTopology};

use super::*;

impl DiscreteSpace {
    /// Bind local basis functionals to exact mesh entities. Vector moment signs
    /// are independently reconstructed from vertex closure and checked against
    /// the mesh's orientation code, so a stale code cannot silently reorder DOFs.
    pub(crate) fn bind_cell(
        &self,
        mesh: &dyn MeshTopology,
        cell: MeshEntity,
    ) -> Result<Vec<(MeshEntity, i8)>, Diagnostic> {
        let dimension = self.cell.dimension();
        if cell.dimension() != dimension
            || mesh.topological_dimension() != dimension
            || mesh
                .entity_count(dimension)
                .is_none_or(|count| cell.index() >= count)
        {
            return Err(invalid_space(
                "local basis requires a cell in its exact mesh stratum",
            ));
        }
        let reference = ReferenceTopology::new(self.cell)?;
        let vertices = mesh
            .incidence(cell, 0)
            .ok_or_else(|| invalid_space("cell has no vertex closure"))?;
        if vertices.len() != reference_vertex_count(self.cell)?
            || vertices.iter().enumerate().any(|(local, entry)| {
                entry.entity.dimension() != 0
                    || entry.local_ordinal != local
                    || mesh
                        .entity_count(0)
                        .is_none_or(|count| entry.entity.index() >= count)
                    || vertices[..local]
                        .iter()
                        .any(|old| old.entity == entry.entity)
            })
        {
            return Err(invalid_space(
                "cell vertex closure differs from the selected reference",
            ));
        }
        self.local_dofs
            .iter()
            .map(|dof| {
                if dof.entity_dimension == dimension {
                    return if dof.entity_ordinal == 0 {
                        Ok((cell, 1))
                    } else {
                        Err(invalid_space("invalid cell-local basis support"))
                    };
                }
                let closure = mesh
                    .incidence(cell, dof.entity_dimension)
                    .ok_or_else(|| invalid_space("basis support is absent from cell closure"))?;
                let entry = closure.get(dof.entity_ordinal).ok_or_else(|| {
                    invalid_space("basis entity ordinal is absent from cell closure")
                })?;
                if entry.local_ordinal != dof.entity_ordinal
                    || entry.entity.dimension() != dof.entity_dimension
                    || mesh
                        .entity_count(dof.entity_dimension)
                        .is_none_or(|count| entry.entity.index() >= count)
                {
                    return Err(invalid_space(
                        "basis incidence has stale local or global identity",
                    ));
                }
                if !self.vector_element() {
                    return Ok((entry.entity, 1));
                }
                let local = reference
                    .entity(dof.entity_dimension, dof.entity_ordinal)
                    .expect("validated reference entity")
                    .vertex_ordinals();
                let canonical = mesh
                    .incidence(entry.entity, 0)
                    .ok_or_else(|| invalid_space("moment entity has no vertex closure"))?;
                if canonical.len() != local.len()
                    || canonical.iter().enumerate().any(|(ordinal, vertex)| {
                        vertex.local_ordinal != ordinal || vertex.entity.dimension() != 0
                    })
                {
                    return Err(invalid_space("moment entity has an invalid vertex closure"));
                }
                let images = local
                    .iter()
                    .map(|&vertex| {
                        canonical
                            .iter()
                            .position(|candidate| candidate.entity == vertices[vertex].entity)
                            .ok_or_else(|| {
                                invalid_space(
                                    "moment support does not match its exact cell vertices",
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let permutation = VertexPermutation::new(images)?;
                // Mesh codes map canonical ordinals to induced local ordinals;
                // the closure lookup above maps local ordinals to canonical ones.
                if mesh
                    .orientation_permutation(entry.orientation, local.len())
                    .as_ref()
                    != Some(&permutation.inverse())
                {
                    return Err(invalid_space(
                        "moment orientation code differs from exact vertex incidence",
                    ));
                }
                Ok((entry.entity, permutation.sign()))
            })
            .collect()
    }
}
