//! Geometry-backed transient-flow semantic lowering.

use super::*;
use eqiora_geometry::CanonicalGeometryV1;

/// Lower one exact Geometry-backed two-dimensional transient incompressible flow.
pub(crate) fn lower_transient_incompressible_navier_stokes_geometry_2d(
    program: &KernelProgram,
    geometry: &CanonicalGeometryV1,
) -> Result<TransientIncompressibleNavierStokesModel<2>, Diagnostic> {
    let (domain, boundaries) =
        crate::canonical_stokes::recognize::unique_bound_geometry_domain_2d(program, geometry)?;
    let bounds = geometry.circular_hole_bounds().ok_or_else(|| {
        lowering_error(
            domain,
            "geometry-backed transient flow requires exact circular-hole geometry",
        )
    })?;
    lower_named(
        program,
        domain,
        *bounds,
        boundaries,
        Some(geometry.digest_bytes()),
    )
}

/// Recognize Geometry-backed transient-flow mathematics without selecting a method.
pub(crate) fn recognize_transient_incompressible_navier_stokes_geometry_mathematics(
    program: &KernelProgram,
) -> Result<(), Diagnostic> {
    let crate::canonical_stokes::domain::GeometrySupportInventory {
        domain,
        boundaries,
        geometry_digest,
    } = crate::canonical_stokes::domain::geometry_support_inventory(
        program,
        "geometry-backed transient-flow recognition requires exactly one GeometryRegion",
        "Geometry-backed equations require boundary supports",
    )?;
    lower_named(
        program,
        domain,
        [[0.0, 1.0], [0.0, 1.0]],
        boundaries,
        Some(geometry_digest),
    )?;
    Ok(())
}

fn lower_named(
    program: &KernelProgram,
    domain: RawId,
    bounds: [[f64; 2]; 2],
    boundaries: BTreeMap<String, RawId>,
    geometry_source_digest: Option<[u8; 32]>,
) -> Result<TransientIncompressibleNavierStokesModel<2>, Diagnostic> {
    let volume = lower_transient_volume::<2>(program, domain)?;
    let boundary = boundary::lower_named(
        program,
        domain,
        volume.velocity,
        volume.pressure,
        &volume.dynamic_viscosity,
        boundaries,
    )?;
    require_boundary_volume(
        &volume,
        &boundary.prescribed_velocity_fields,
        &boundary.prescribed_velocity_definitions,
    )?;
    let model = TransientIncompressibleNavierStokesModel {
        domain,
        velocity: volume.velocity,
        pressure: volume.pressure,
        force_source: volume.force_source,
        bounds,
        mass_density: volume.mass_density,
        dynamic_viscosity: volume.dynamic_viscosity,
        force_potential_expression: volume.force_potential_expression,
        momentum_relation: volume.momentum_relation,
        momentum_orientation: volume.momentum_orientation,
        incompressibility_relation: volume.incompressibility_relation,
        boundary_entries: boundary
            .entries
            .iter()
            .map(|(name, entry)| {
                (
                    PhysicalBoundarySupport::NamedEntitySet(name.clone()),
                    *entry,
                )
            })
            .collect(),
        boundary_relations: boundary.boundary_relations.clone(),
        normal_velocity_expressions: boundary
            .normal_velocity_expressions
            .iter()
            .filter_map(|(name, expression)| {
                boundary
                    .entries
                    .get(name)
                    .map(|entry| (entry.boundary(), expression.clone()))
            })
            .collect(),
        geometry_source_digest,
    };
    require_closed_model(program, &model, volume.representation, &boundary)?;
    Ok(model)
}

fn require_closed_model(
    program: &KernelProgram,
    model: &TransientIncompressibleNavierStokesModel<2>,
    representation: RawId,
    boundary: &boundary::LoweredNamedStokesBoundary2d,
) -> Result<(), Diagnostic> {
    let mut domains = BTreeSet::from([model.domain]);
    domains.extend(boundary.entries.values().map(|entry| entry.boundary()));
    domains.extend(boundary.connector_domains.iter().copied());
    let mut relations = BTreeSet::from([model.momentum_relation, model.incompressibility_relation]);
    relations.extend(model.force_potential_definition());
    relations.extend(
        model
            .boundary_relations
            .iter()
            .map(|binding| binding.relation()),
    );
    relations.extend(boundary.prescribed_velocity_definitions.iter().copied());
    let mut fields = BTreeSet::from([model.velocity, model.pressure]);
    fields.extend(model.force_potential());
    fields.extend(boundary.prescribed_velocity_fields.iter().copied());
    crate::canonical_stokes::inventory::FlowClosure {
        domains: &domains,
        fields: &fields,
        relations: &relations,
        representation,
        ports: &boundary.ports,
        connections: &boundary.connections,
        additional_parameters: &BTreeSet::new(),
    }
    .validate(program, "closed Geometry-backed transient lowering")
}
