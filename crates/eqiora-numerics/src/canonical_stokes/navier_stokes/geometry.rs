//! Geometry-backed transient-flow semantic lowering.

use super::*;
use eqiora_geometry::CanonicalGeometryV1;

/// Lower one exact Geometry-backed two-dimensional transient incompressible flow.
pub(crate) fn lower_transient_incompressible_navier_stokes_geometry_2d(
    program: &KernelProgram,
    geometry: &CanonicalGeometryV1,
) -> Result<TransientIncompressibleNavierStokesModel2d, Diagnostic> {
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
    let regions = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::GeometryRegion { .. }) =>
            {
                Some(domain)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let [region] = regions.as_slice() else {
        return Err(model_lowering_error(
            program,
            "geometry-backed transient-flow recognition requires exactly one GeometryRegion",
        ));
    };
    let DomainKind::GeometryRegion { geometry, .. } = region.kind() else {
        unreachable!("GeometryRegion filter is exact")
    };
    let domain = region.id().erase();
    let boundaries = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(boundary)
                if has_edge(program, boundary.id().erase(), domain, EdgeKind::BoundaryOf) =>
            {
                match boundary.kind() {
                    DomainKind::GeometryBoundary { entity_set } => {
                        Some((entity_set.clone(), boundary.id().erase()))
                    }
                    _ => None,
                }
            }
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    if boundaries.is_empty() {
        return Err(lowering_error(
            domain,
            "Geometry-backed equations require boundary supports",
        ));
    }
    lower_named(
        program,
        domain,
        [[0.0, 1.0], [0.0, 1.0]],
        boundaries,
        Some(geometry.bytes()),
    )?;
    Ok(())
}

fn lower_named(
    program: &KernelProgram,
    domain: RawId,
    bounds: [[f64; 2]; 2],
    boundaries: BTreeMap<String, RawId>,
    geometry_source_digest: Option<[u8; 32]>,
) -> Result<TransientIncompressibleNavierStokesModel2d, Diagnostic> {
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
    let model = TransientIncompressibleNavierStokesModel2d {
        domain,
        velocity: volume.velocity,
        pressure: volume.pressure,
        force_potential: volume.force_potential,
        bounds,
        mass_density: volume.mass_density,
        dynamic_viscosity: volume.dynamic_viscosity,
        force_potential_expression: volume.force_potential_expression,
        force_potential_definition: volume.force_potential_definition,
        momentum_relation: volume.momentum_relation,
        incompressibility_relation: volume.incompressibility_relation,
        boundary_dispositions: boundary
            .entries
            .values()
            .map(|entry| (entry.boundary(), entry.disposition()))
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
        named_boundary_ids: boundary
            .entries
            .iter()
            .map(|(name, entry)| (name.clone(), entry.boundary()))
            .collect(),
        stress_form: IncompressibleStressForm::SymmetricNewtonian,
        geometry_source_digest,
    };
    require_closed_model(program, &model, volume.representation, &boundary)?;
    Ok(model)
}

fn require_closed_model(
    program: &KernelProgram,
    model: &TransientIncompressibleNavierStokesModel2d,
    representation: RawId,
    boundary: &boundary::LoweredNamedStokesBoundary2d,
) -> Result<(), Diagnostic> {
    let mut domains = BTreeSet::from([model.domain]);
    domains.extend(boundary.entries.values().map(|entry| entry.boundary()));
    domains.extend(boundary.connector_domains.iter().copied());
    let mut relations = BTreeSet::from([
        model.force_potential_definition,
        model.momentum_relation,
        model.incompressibility_relation,
    ]);
    relations.extend(
        model
            .boundary_relations
            .iter()
            .map(|binding| binding.relation()),
    );
    relations.extend(boundary.prescribed_velocity_definitions.iter().copied());
    let activations = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Activates && relations.contains(&edge.to()))
        .map(|edge| edge.from())
        .collect::<BTreeSet<_>>();
    let parameters = parameters_referenced_by(program, &relations);
    let mut fields = BTreeSet::from([model.velocity, model.pressure, model.force_potential]);
    fields.extend(boundary.prescribed_velocity_fields.iter().copied());
    for node in program.nodes() {
        let admitted = match node {
            KernelNode::Domain(value) => domains.contains(&value.id().erase()),
            KernelNode::Representation(value) => value.id().erase() == representation,
            KernelNode::Field(value) => fields.contains(&value.id().erase()),
            KernelNode::Parameter(value) => parameters.contains(&value.id().erase()),
            KernelNode::Relation(value) => relations.contains(&value.id().erase()),
            KernelNode::Activation(value) => activations.contains(&value.id().erase()),
            KernelNode::Port(value) => boundary.ports.contains(&value.id().erase()),
            KernelNode::Connection(value) => boundary.connections.contains(&value.id().erase()),
            _ => false,
        };
        if !admitted {
            return Err(model_lowering_error(
                program,
                format!(
                    "closed Geometry-backed transient lowering would ignore unexpected {:?} node {}",
                    node.kind(),
                    node.id()
                ),
            ));
        }
    }
    Ok(())
}
