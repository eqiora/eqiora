//! Exact Field support is independent of coefficient storage domain.
use super::*;

pub(super) fn field_support<S: crate::spatial_expression::Coefficient>(
    equations: &ExecutableLinearEquations<S>,
    mesh: &eqiora_meshing::CartesianMesh,
    field: eqiora_core::RawId,
    spatial: CommonSpatialPolicy,
) -> Result<(Vec<usize>, Vec<usize>), Diagnostic> {
    let region = equations
        .regions
        .iter()
        .find(|region| region.form.fields().iter().any(|(id, _)| *id == field))
        .ok_or_else(|| invalid("Field absent from exact Region inventory"))?;
    let mut shape = Vec::new();
    for (axis, bounds) in region.cartesian()?.bounds.iter().enumerate() {
        let coordinates = mesh.axis_coordinates(axis).expect("axis");
        let start = coordinates
            .iter()
            .position(|x| *x == bounds[0])
            .ok_or_else(|| invalid("Field support lower bound absent"))?;
        let end = coordinates
            .iter()
            .position(|x| *x == bounds[1])
            .ok_or_else(|| invalid("Field support upper bound absent"))?;
        shape.push(end - start + usize::from(spatial == CommonSpatialPolicy::Q1));
    }
    let domains = equations.cell_domains(mesh)?;
    let mut vertices = BTreeSet::new();
    for (index, domain) in domains.iter().enumerate() {
        if *domain == region.form.domain() {
            vertices.extend(
                mesh.incidence(
                    eqiora_meshing::MeshEntity::new(mesh.topological_dimension(), index),
                    0,
                )
                .expect("cell closure")
                .iter()
                .map(|vertex| vertex.entity.index()),
            );
        }
    }
    Ok((shape, vertices.into_iter().collect()))
}
