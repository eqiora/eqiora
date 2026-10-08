//! Exact structural dependencies of the prepared ALE FSI Jacobian.

use super::*;

pub(in crate::simplicial_ale_fsi) fn build_step_jacobian_pattern<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &AleFsiBoundary<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    base_layout: &FsiLayout<D>,
) -> Result<StructuralJacobianPattern, Diagnostic> {
    let layout = base_layout.with_boundary(boundary)?;
    build_structural_jacobian_pattern(reference, partition, motion, &layout)
}

pub(super) fn build_structural_jacobian_pattern<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    layout: &FsiLayout<D>,
) -> Result<StructuralJacobianPattern, Diagnostic> {
    let cell_count = partition.cell_count();
    let mut pattern = StructuralJacobianPatternBuilder::new(
        layout.reduced_size(),
        layout.reduced_size(),
        cell_count,
    )?;
    for cell_index in 0..cell_count {
        let vertices = reference
            .entity_vertices(MeshEntity::new(D, cell_index))
            .ok_or_else(|| {
                invalid(format!(
                    "ALE FSI structural dependency cell {cell_index} has no vertex closure"
                ))
            })?;
        let (local_size, map) = match layout.cell_domain(cell_index)? {
            domain if layout.pressure_field(domain).is_some() => {
                let bubble_cell = CellId::new(cell_index);
                (
                    fluid_local_size::<D>(),
                    layout.fluid_map(bubble_cell, &vertices, true)?,
                )
            }
            domain if layout.state_field(domain).is_some() => (
                solid_local_size::<D>(),
                layout.solid_map(cell_index, &vertices, true)?,
            ),
            _ => {
                return Err(invalid(format!(
                    "ALE FSI structural dependency cell {cell_index} has no material assignment"
                )));
            }
        };
        pattern.include_dense_local(cell_index, local_size, &map)?;
    }

    // The sealed harmonic inverse can carry any interface-driver component
    // across the complete fluid region. Its numeric influence entries are not
    // inspected: every represented driver column conservatively becomes a
    // global singleton.
    for driver in motion.driver_vertices() {
        for component in 0..D {
            if let Some(dof) = layout.reduced_vertex_velocity(
                layout.state_rate(motion.policy().solid_displacement().erase())?,
                driver.index(),
                component,
            ) {
                pattern.mark_globally_coupled(dof.index())?;
            }
        }
    }
    pattern.finish()
}
