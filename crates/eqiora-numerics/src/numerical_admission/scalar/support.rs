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
        .find(|region| {
            region
                .form
                .represented_fields()
                .iter()
                .any(|(id, _)| *id == field)
        })
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
    let represented = region.form.represented_fields();
    let value_type = &represented
        .iter()
        .find(|(id, _)| *id == field)
        .expect("selected Field")
        .1;
    shape.extend(
        value_type
            .shape()
            .extents()
            .iter()
            .map(|extent| usize::try_from(extent.get()).expect("portable Field extent")),
    );
    let domains = equations.cell_domains(mesh)?;
    let mut entities = BTreeSet::new();
    for (index, domain) in domains.iter().enumerate() {
        if *domain == region.form.domain() {
            if spatial == CommonSpatialPolicy::CellCenteredTpfa {
                entities.insert(index);
                continue;
            }
            entities.extend(
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
    Ok((shape, entities.into_iter().collect()))
}

pub(super) fn simplicial_support<
    S: crate::spatial_expression::Coefficient + crate::finalized_spatial::ResidualScalar + Send,
>(
    equations: &ExecutableLinearEquations<S>,
    mesh: &SimplicialMeshEnvelopeV1,
    field: eqiora_core::RawId,
    space: Space,
) -> Result<(Vec<usize>, Vec<usize>), Diagnostic> {
    let (entities, _) = simplicial_topology(equations, mesh, field, space)?;
    let mut shape = vec![entities.len()];
    if space == Space::continuous_lagrange(std::num::NonZeroU16::MIN) {
        let fields = equations.represented_fields();
        let (_, value_type) = fields
            .iter()
            .find(|(id, _)| *id == field)
            .ok_or_else(|| invalid("Field absent from exact simplicial inventory"))?;
        shape.extend(
            value_type
                .shape()
                .extents()
                .iter()
                .map(|extent| usize::try_from(extent.get()).expect("portable Field extent")),
        );
    }
    Ok((
        shape,
        entities.into_iter().map(|entity| entity.index()).collect(),
    ))
}

pub(super) fn simplicial_topology<
    S: crate::spatial_expression::Coefficient + crate::finalized_spatial::ResidualScalar + Send,
>(
    equations: &ExecutableLinearEquations<S>,
    mesh: &SimplicialMeshEnvelopeV1,
    field: eqiora_core::RawId,
    space: Space,
) -> Result<
    (
        Vec<eqiora_meshing::MeshEntity>,
        Vec<eqiora_meshing::MeshEntity>,
    ),
    Diagnostic,
> {
    let (mapping, _, _) = equations.simplicial_assembly(mesh, space)?;
    let (domain, _) = mapping
        .field_layout(field)
        .ok_or_else(|| invalid("Field is absent from the exact simplicial layout"))?;
    let entities = mapping
        .keys()
        .filter(|key| key.field == field)
        .map(|key| key.entity)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let cells = mapping
        .cell_domains()
        .iter()
        .enumerate()
        .filter_map(|(index, cell_domain)| {
            (*cell_domain == domain)
                .then_some(eqiora_meshing::MeshEntity::new(mesh.dimension(), index))
        })
        .collect();
    Ok((entities, cells))
}
