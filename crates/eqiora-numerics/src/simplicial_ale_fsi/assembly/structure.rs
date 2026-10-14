//! Exact structural dependencies of the prepared ALE FSI Jacobian.

use super::*;

pub(in crate::simplicial_ale_fsi) fn build_step_jacobian_pattern<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &FixedReferenceFsiBoundary<D>,
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
    layout.require_reference(reference, partition)?;
    let cell_count = partition.cell_count();
    let mut pattern = StructuralJacobianPatternBuilder::new(
        layout.reduced_size(),
        layout.reduced_size(),
        cell_count,
    )?;
    for cell_index in 0..cell_count {
        let map = layout.mapping().cell_map(cell_index, true)?;
        let local_size = map.unknowns().len();
        pattern.include_dense_local(cell_index, local_size, &map)?;
    }

    // The sealed harmonic inverse can carry any interface-driver component
    // across the complete fluid region. Its numeric influence entries are not
    // inspected: every represented driver column conservatively becomes a
    // global singleton.
    for &(domain, field) in motion.policy().drivers() {
        let vertices = partition
            .domain_vertices(domain)
            .ok_or_else(|| invalid("motion driver has no exact Domain support"))?;
        for driver in motion
            .driver_vertices()
            .iter()
            .filter(|vertex| vertices.contains(vertex))
        {
            for component in 0..D {
                if let Some(dof) = layout.reduced_vertex_velocity(
                    layout.state_rate(field.erase())?,
                    driver.index(),
                    component,
                ) {
                    pattern.mark_globally_coupled(dof.index())?;
                }
            }
        }
    }
    pattern.finish()
}
