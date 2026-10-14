use crate::canonical_boundary::PhysicalBoundarySupport;
use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, RawId};
use eqiora_geometry::CanonicalGeometryV1;
use eqiora_schema::kernel::KernelNode;
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_sem::KernelProgram;

use crate::canonical_boundary::PhysicalBoundaryEntry;
use crate::form_compiler::vocabulary::{MixedGalerkinCorrespondence, MixedGalerkinSource};
use crate::spatial_expression;

use super::api::{SteadyIncompressibleStokesModel2d, SteadyStokesNormalPressure2d};
use super::boundary::{self, NormalPressureSource2d};
use super::domain::unique_named_geometry_domain;
pub(super) use super::domain::{unique_bound_geometry_domain_2d, unique_box, unique_box_2d};
use super::expression::{
    additive_load_definition_root, is_additive_divergence_of_field, load_definition_root,
    lower_exact_twice_viscosity, momentum_viscous_root,
};
use super::support::{
    continuum_representation, lowering_error, relation_expression, relations_on, unique_root,
};

use super::inventory::{VELOCITY, VELOCITY_POTENTIAL};

enum BoundarySource2d {
    Cartesian,
    Named(BTreeMap<String, RawId>),
}

struct LoweredBoundaryProjection2d {
    entries: BTreeMap<PhysicalBoundarySupport, PhysicalBoundaryEntry>,
    normal_pressure_sources: BTreeMap<PhysicalBoundarySupport, NormalPressureSource2d>,
    prescribed_velocity_traces: BTreeMap<
        PhysicalBoundarySupport,
        super::prescribed_velocity::SteadyStokesPrescribedVelocityTrace2d,
    >,
    prescribed_velocity_fields: BTreeSet<RawId>,
    prescribed_velocity_definitions: BTreeSet<RawId>,
    boundary_relations: Vec<crate::canonical_boundary::BoundaryRelationBinding>,
    ports: BTreeSet<RawId>,
    connections: BTreeSet<RawId>,
    connector_domains: BTreeSet<RawId>,
    uninterpreted_live_relations: BTreeSet<RawId>,
}

/// Lower the exact canonical 2D steady incompressible Stokes subset.
///
/// Recognition is identity-parametric and whole-model fail-closed. Source
/// names, package provenance, and declaration order are not part of the
/// contract.
///
/// # Errors
/// Returns `EQ0703` when the admitted Model is ambiguous, incomplete, or
/// contains meaning outside this deliberately narrow subset.
pub fn lower_steady_incompressible_stokes_cartesian_2d(
    program: &KernelProgram,
) -> Result<SteadyIncompressibleStokesModel2d, Diagnostic> {
    let (domain, bounds) = unique_box_2d(program)?;
    lower_steady_incompressible_stokes_2d_on(
        program,
        domain,
        bounds,
        BoundarySource2d::Cartesian,
        None,
        &BTreeSet::new(),
    )
}

/// Lower the exact geometry-backed 2D steady incompressible Stokes subset.
///
/// The supplied exact circular-hole geometry authenticates the Model's
/// `GeometryRegion` digest and supplies only exact bounds and fixed-side
/// parent-outward normals. Mesh entity membership is deliberately absent from
/// this method-neutral result and is resolved later through correspondence.
///
/// # Errors
/// Returns `EQ0703` for semantic ambiguity or unsupported boundary meaning,
/// and rejects a source digest or named-set inventory that differs from the
/// admitted Model before numerical realization.
pub(super) fn lower_steady_incompressible_stokes_geometry_2d(
    program: &KernelProgram,
    geometry: &CanonicalGeometryV1,
) -> Result<SteadyIncompressibleStokesModel2d, Diagnostic> {
    let (domain, boundaries) = unique_bound_geometry_domain_2d(program, geometry)?;
    let bounds = geometry.circular_hole_bounds().ok_or_else(|| {
        lowering_error(
            domain,
            "geometry-backed steady Stokes requires exact circular-hole geometry",
        )
    })?;
    lower_steady_incompressible_stokes_2d_on(
        program,
        domain,
        *bounds,
        BoundarySource2d::Named(boundaries),
        Some(geometry.digest_bytes()),
        &BTreeSet::new(),
    )
}

pub(crate) fn recognize_steady_incompressible_stokes_geometry_mathematics(
    program: &KernelProgram,
) -> Result<SteadyIncompressibleStokesModel2d, Diagnostic> {
    let crate::canonical_stokes::domain::GeometrySupportInventory {
        domain,
        boundaries,
        geometry_digest,
    } = crate::canonical_stokes::domain::geometry_support_inventory(
        program,
        "geometry-backed steady Stokes recognition requires exactly one GeometryRegion",
        "geometry-backed steady Stokes recognition requires boundary supports",
    )?;
    lower_steady_incompressible_stokes_2d_on(
        program,
        domain,
        [[0.0, 1.0], [0.0, 1.0]],
        BoundarySource2d::Named(boundaries),
        Some(geometry_digest),
        &BTreeSet::new(),
    )
}

fn lower_steady_incompressible_stokes_2d_on(
    program: &KernelProgram,
    domain: RawId,
    bounds: [[f64; 2]; 2],
    boundary_source: BoundarySource2d,
    geometry_source_digest: Option<[u8; 32]>,
    additional_parameters: &BTreeSet<RawId>,
) -> Result<SteadyIncompressibleStokesModel2d, Diagnostic> {
    let (velocity, scalar_fields, prescribed_velocity_fields, representation) =
        exact_steady_fields(program, domain)?;
    let volume_relations = relations_on(program, domain);
    if volume_relations.len() < 3 {
        return Err(lowering_error(
            domain,
            format!(
                "2D steady Stokes requires at least three volume Relations, found {}",
                volume_relations.len()
            ),
        ));
    }
    let typed_relations = super::inventory::typed_volume_relations(program, &volume_relations)?;
    let additive_inventory = additive_volume_inventory(&typed_relations, &volume_relations)?;

    let mut incompressibility = Vec::new();
    for relation in &volume_relations {
        let typed = &typed_relations[relation];
        let root = unique_root(typed.expression(), *relation)?;
        if is_additive_divergence_of_field(typed.expression(), root, velocity, *relation)? {
            incompressibility.push(*relation);
        }
    }
    if incompressibility.len() != 1 {
        return Err(lowering_error(
            domain,
            format!(
                "2D steady Stokes requires exactly one `div(u) = 0` Relation, found {}; unmatched signed leaves by owning Relation: {additive_inventory}",
                incompressibility.len(),
            ),
        ));
    }
    let incompressibility_relation = incompressibility[0];
    let mut candidates = Vec::new();
    for &force_potential in &scalar_fields {
        for &pressure in &scalar_fields {
            if pressure == force_potential {
                continue;
            }
            let mut definitions = Vec::new();
            for relation in &volume_relations {
                let expression = typed_relations[relation].expression();
                let root = unique_root(expression, *relation)?;
                if let Some(source) =
                    additive_load_definition_root(expression, root, force_potential, *relation)?
                {
                    definitions.push((*relation, source));
                }
            }
            let mut momenta = Vec::new();
            for relation in &volume_relations {
                let typed = &typed_relations[relation];
                let root = unique_root(typed.expression(), *relation)?;
                if let Some(viscous) = momentum_viscous_root(
                    typed,
                    root,
                    velocity,
                    pressure,
                    force_potential,
                    *relation,
                )? {
                    momenta.push((*relation, viscous));
                }
            }
            if definitions.len() == 1
                && momenta.len() == 1
                && definitions[0].0 != momenta[0].0
                && definitions[0].0 != incompressibility_relation
                && momenta[0].0 != incompressibility_relation
            {
                candidates.push((pressure, force_potential, definitions[0], momenta[0]));
            }
        }
    }
    if candidates.len() != 1 {
        return Err(lowering_error(
            domain,
            format!(
                "pressure, force-potential definition, and exact Stokes momentum balance must have one unique identity assignment, found {}; unmatched signed leaves by owning Relation: {additive_inventory}",
                candidates.len(),
            ),
        ));
    }
    let (
        pressure,
        force_potential,
        (force_potential_definition, source),
        (momentum_relation, viscous),
    ) = candidates.remove(0);
    let force_potential_expression = spatial_expression::lower(
        program,
        &relation_expression(program, force_potential_definition)?,
        source,
        force_potential_definition,
        2,
    )?;
    let dynamic_viscosity = lower_exact_twice_viscosity(
        program,
        &typed_relations[&momentum_relation],
        viscous,
        velocity,
        momentum_relation,
    )?
    .ok_or_else(|| {
        lowering_error(
            momentum_relation,
            "Stokes viscous stress must be exactly `2 * mu * symmetric_part(grad(u))` with one typed scalar coefficient",
        )
    })?;
    let Some(viscosity) = dynamic_viscosity.constant_value() else {
        return Err(lowering_error(
            momentum_relation,
            "dynamic viscosity must be a spatially constant scalar expression",
        ));
    };
    if !viscosity.is_finite() || viscosity <= 0.0 {
        return Err(lowering_error(
            momentum_relation,
            "steady Stokes requires finite positive dynamic viscosity",
        ));
    }

    let boundary = lower_boundary_projection(
        program,
        domain,
        velocity,
        pressure,
        &dynamic_viscosity,
        boundary_source,
    )?;
    if boundary.prescribed_velocity_fields != prescribed_velocity_fields {
        return Err(lowering_error(
            domain,
            "every velocity or velocity-potential scalar Field must define one prescribed velocity trace",
        ));
    }
    let volume_roles = StokesVolumeRoles {
        domain,
        representation,
        pressure,
        force_potential,
        scalar_fields: &scalar_fields,
        relations: &volume_relations,
        force_potential_definition,
        momentum_relation,
        incompressibility_relation,
    };
    let normal_pressures = resolve_normal_pressures(
        program,
        &volume_roles,
        &boundary.normal_pressure_sources,
        &boundary.prescribed_velocity_definitions,
    )?;
    let boundary_relation_ids = boundary
        .boundary_relations
        .iter()
        .map(|binding| binding.relation())
        .collect::<Vec<_>>();
    let boundary_dispositions = boundary
        .entries
        .values()
        .map(|entry| (entry.boundary(), entry.disposition()))
        .collect::<BTreeMap<_, _>>();
    let certificate_source = super::mixed_certificate::SteadyStokesCertificateSource {
        domain,
        velocity,
        pressure,
        source_definition: force_potential_definition,
        source_node: source,
        momentum_relation,
        incompressibility_relation,
        boundaries: &boundary.boundary_relations,
        boundary_dispositions: &boundary_dispositions,
    };
    let certificate = super::mixed_certificate::derive(program, &certificate_source)?;
    let correspondence = MixedGalerkinCorrespondence::derive(MixedGalerkinSource {
        domain,
        velocity,
        pressure,
        source: force_potential,
        source_definition: force_potential_definition,
        momentum_relation,
        incompressibility_relation,
        boundary_relations: &boundary_relation_ids,
    })
    .with_entries(certificate);
    let model = SteadyIncompressibleStokesModel2d {
        correspondence,
        bounds,
        dynamic_viscosity,
        force_potential_expression,
        boundary_entries: boundary.entries.clone(),
        boundary_relations: boundary.boundary_relations.clone(),
        normal_pressures: normal_pressures.by_key,
        prescribed_velocity_traces: boundary.prescribed_velocity_traces.clone(),
        geometry_source_digest,
    };
    require_closed_model(
        program,
        &model,
        representation,
        &boundary,
        &normal_pressures.fields,
        &normal_pressures.definitions,
        additional_parameters,
    )?;
    Ok(model)
}

fn additive_volume_inventory(
    typed_relations: &BTreeMap<RawId, TypedResidual<RawId>>,
    relations: &[RawId],
) -> Result<String, Diagnostic> {
    relations
        .iter()
        .map(|relation| {
            let expression = typed_relations[relation].expression();
            let root = unique_root(expression, *relation)?;
            let view = crate::additive_residual::AdditiveResidualView::derive(
                expression, root, *relation,
            )?;
            Ok(format!("Relation {relation}: [{}]", view.signed_leaves()))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()
        .map(|inventory| inventory.join("; "))
}

pub(super) fn lower_stokes_dissipation_profile_model_2d(
    program: &KernelProgram,
    geometry: &eqiora_artifact::GeometryDefinitionV1,
    bounds: [[f64; 2]; 2],
    design_parameters: [RawId; 3],
) -> Result<SteadyIncompressibleStokesModel2d, Diagnostic> {
    let required = BTreeSet::from([
        "body".to_owned(),
        "outer_x_lower".to_owned(),
        "outer_x_upper".to_owned(),
        "outer_y_lower".to_owned(),
        "outer_y_upper".to_owned(),
    ]);
    let (domain, boundaries) = unique_named_geometry_domain(
        program,
        geometry.canonical().digest_bytes(),
        "fluid",
        &required,
    )?;
    let design_parameters = design_parameters.into_iter().collect::<BTreeSet<_>>();
    if design_parameters.len() != 3 {
        return Err(lowering_error(
            domain,
            "Stokes dissipation profile requires three distinct r_A/a_2/a_4 Parameters",
        ));
    }
    for parameter in &design_parameters {
        if !matches!(program.node(*parameter), Some(KernelNode::Parameter(_)))
            || program.value(*parameter).is_none()
        {
            return Err(lowering_error(
                *parameter,
                "Stokes dissipation design identity must name one valued Parameter in the exact Model",
            ));
        }
    }
    lower_steady_incompressible_stokes_2d_on(
        program,
        domain,
        bounds,
        BoundarySource2d::Named(boundaries),
        Some(geometry.canonical().digest_bytes()),
        &design_parameters,
    )
}

fn lower_boundary_projection(
    program: &KernelProgram,
    domain: RawId,
    velocity: RawId,
    pressure: RawId,
    dynamic_viscosity: &crate::spatial_expression::ScalarSpatialExpression<f64>,
    source: BoundarySource2d,
) -> Result<LoweredBoundaryProjection2d, Diagnostic> {
    match source {
        BoundarySource2d::Cartesian => {
            let lowered = boundary::lower(program, domain, velocity, pressure, dynamic_viscosity)?;
            Ok(LoweredBoundaryProjection2d {
                entries: lowered
                    .inventory
                    .entries()
                    .map(|(&(axis, side), entry)| {
                        (
                            PhysicalBoundarySupport::CartesianSide { axis, side },
                            *entry,
                        )
                    })
                    .collect(),
                normal_pressure_sources: lowered
                    .normal_pressure_sources
                    .into_iter()
                    .map(|((axis, side), source)| {
                        (
                            PhysicalBoundarySupport::CartesianSide { axis, side },
                            source,
                        )
                    })
                    .collect(),
                prescribed_velocity_traces: lowered
                    .prescribed_velocity_traces
                    .into_iter()
                    .map(|((axis, side), trace)| {
                        (PhysicalBoundarySupport::CartesianSide { axis, side }, trace)
                    })
                    .collect(),
                prescribed_velocity_fields: lowered.prescribed_velocity_fields,
                prescribed_velocity_definitions: lowered.prescribed_velocity_definitions,
                boundary_relations: lowered.boundary_relations,
                ports: lowered.ports,
                connections: lowered.connections,
                connector_domains: lowered.connector_domains,
                uninterpreted_live_relations: lowered.uninterpreted_live_relations,
            })
        }
        BoundarySource2d::Named(boundaries) => {
            let lowered = boundary::lower_named(
                program,
                domain,
                velocity,
                pressure,
                dynamic_viscosity,
                boundaries,
            )?;
            Ok(LoweredBoundaryProjection2d {
                entries: lowered
                    .entries
                    .into_iter()
                    .map(|(name, entry)| (PhysicalBoundarySupport::NamedEntitySet(name), entry))
                    .collect(),
                normal_pressure_sources: lowered
                    .normal_pressure_sources
                    .into_iter()
                    .map(|(name, source)| (PhysicalBoundarySupport::NamedEntitySet(name), source))
                    .collect(),
                prescribed_velocity_traces: lowered
                    .prescribed_velocity_traces
                    .into_iter()
                    .map(|(name, trace)| (PhysicalBoundarySupport::NamedEntitySet(name), trace))
                    .collect(),
                prescribed_velocity_fields: lowered.prescribed_velocity_fields,
                prescribed_velocity_definitions: lowered.prescribed_velocity_definitions,
                boundary_relations: lowered.boundary_relations,
                ports: lowered.ports,
                connections: lowered.connections,
                connector_domains: lowered.connector_domains,
                uninterpreted_live_relations: lowered.uninterpreted_live_relations,
            })
        }
    }
}

fn exact_steady_fields(
    program: &KernelProgram,
    domain: RawId,
) -> Result<(RawId, Vec<RawId>, BTreeSet<RawId>, RawId), Diagnostic> {
    let inventory =
        super::inventory::FlowFields::collect(program, domain, 2, &[VELOCITY, VELOCITY_POTENTIAL]);
    let velocity = &inventory.velocities;
    let pressure_scalars = &inventory.pressures;
    let prescribed_velocity_scalars = &inventory.coefficients;
    if inventory.count
        != velocity.len() + pressure_scalars.len() + prescribed_velocity_scalars.len()
        || velocity.len() != 1
        || pressure_scalars.len() < 2
    {
        return Err(lowering_error(
            domain,
            "steady Stokes Fields must be exactly one spatial velocity vector, at least two pressure-valued invariant scalars, and zero or more invariant velocity or velocity-potential scalars",
        ));
    }
    let representation = inventory.representation(domain, "steady Stokes solution and boundary-coefficient Fields must share one continuum Representation")?;
    Ok((
        velocity[0],
        inventory.pressures,
        inventory.coefficients,
        representation,
    ))
}

pub(super) fn exact_fields(
    program: &KernelProgram,
    domain: RawId,
) -> Result<(RawId, Vec<RawId>, RawId), Diagnostic> {
    exact_fields_for_dimension::<2>(program, domain)
}

pub(super) fn exact_fields_for_dimension<const D: usize>(
    program: &KernelProgram,
    domain: RawId,
) -> Result<(RawId, Vec<RawId>, RawId), Diagnostic> {
    if !matches!(D, 2 | 3) {
        return Err(lowering_error(
            domain,
            format!("canonical fluid Fields support dimension two or three, received {D}"),
        ));
    }
    let inventory = super::inventory::FlowFields::collect(program, domain, D as u32, &[]);
    let velocity = &inventory.velocities;
    let scalars = &inventory.pressures;
    if inventory.count != velocity.len() + scalars.len() || velocity.len() != 1 || scalars.len() < 2
    {
        return Err(lowering_error(
            domain,
            format!(
                "Fields must be exactly one velocity-valued spatial Cartesian `[{D}]` Field and at least two pressure-valued invariant scalar Fields"
            ),
        ));
    }
    let representation = inventory.representation(
        domain,
        "velocity, pressure, and force potential must share one continuum Representation",
    )?;
    Ok((velocity[0], inventory.pressures, representation))
}

struct ResolvedNormalPressures2d {
    by_key: BTreeMap<PhysicalBoundarySupport, SteadyStokesNormalPressure2d>,
    fields: BTreeSet<RawId>,
    definitions: BTreeSet<RawId>,
}

struct StokesVolumeRoles<'a> {
    domain: RawId,
    representation: RawId,
    pressure: RawId,
    force_potential: RawId,
    scalar_fields: &'a [RawId],
    relations: &'a [RawId],
    force_potential_definition: RawId,
    momentum_relation: RawId,
    incompressibility_relation: RawId,
}

fn resolve_normal_pressures(
    program: &KernelProgram,
    volume: &StokesVolumeRoles<'_>,
    sources: &BTreeMap<PhysicalBoundarySupport, NormalPressureSource2d>,
    prescribed_velocity_definitions: &BTreeSet<RawId>,
) -> Result<ResolvedNormalPressures2d, Diagnostic> {
    let scalar_fields = volume
        .scalar_fields
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut coefficient_fields = BTreeSet::new();
    let mut definitions = BTreeSet::new();
    let mut lowered_fields: BTreeMap<
        RawId,
        (
            RawId,
            crate::spatial_expression::ScalarSpatialExpression<f64>,
        ),
    > = BTreeMap::new();
    let mut by_key = BTreeMap::new();

    for (key, source) in sources {
        let pressure_law = match *source {
            NormalPressureSource2d::Zero => SteadyStokesNormalPressure2d::zero(
                crate::spatial_expression::ScalarSpatialExpression::constant(2, 0.0),
            ),
            NormalPressureSource2d::Field {
                field,
                law_relation,
            } => {
                if field == volume.pressure || field == volume.force_potential {
                    return Err(lowering_error(
                        law_relation,
                        "normal-pressure coefficient Field must be distinct from solution pressure and force potential",
                    ));
                }
                if !scalar_fields.contains(&field)
                    || continuum_representation(program, field) != Some(volume.representation)
                {
                    return Err(lowering_error(
                        field,
                        "normal-pressure coefficient must be an invariant pressure-valued continuum Field on the exact Stokes volume",
                    ));
                }
                coefficient_fields.insert(field);
                let (definition, expression) = if let Some(lowered) = lowered_fields.get(&field) {
                    lowered.clone()
                } else {
                    let candidates = volume
                        .relations
                        .iter()
                        .copied()
                        .filter_map(|relation| {
                            let expression = &relation_expression(program, relation).ok()?;
                            let root = unique_root(expression, relation).ok()?;
                            load_definition_root(expression, root, field)
                                .map(|source| (relation, source))
                        })
                        .collect::<Vec<_>>();
                    if candidates.len() != 1 {
                        return Err(lowering_error(
                            field,
                            format!(
                                "normal-pressure coefficient Field requires exactly one scalar definition Relation, found {}",
                                candidates.len()
                            ),
                        ));
                    }
                    let (definition, root) = candidates[0];
                    let expression = spatial_expression::lower(
                        program,
                        &relation_expression(program, definition)?,
                        root,
                        definition,
                        2,
                    )?;
                    lowered_fields.insert(field, (definition, expression.clone()));
                    (definition, expression)
                };
                definitions.insert(definition);
                SteadyStokesNormalPressure2d::field(field, definition, expression)
            }
        };
        by_key.insert(key.clone(), pressure_law);
    }

    let expected_coefficient_fields = scalar_fields
        .iter()
        .copied()
        .filter(|field| *field != volume.pressure && *field != volume.force_potential)
        .collect::<BTreeSet<_>>();
    if coefficient_fields != expected_coefficient_fields {
        return Err(lowering_error(
            volume.domain,
            "every additional pressure-valued Field must be used by an exact normal-pressure boundary law",
        ));
    }

    let mut expected_relations = BTreeSet::from([
        volume.force_potential_definition,
        volume.momentum_relation,
        volume.incompressibility_relation,
    ]);
    expected_relations.extend(definitions.iter().copied());
    expected_relations.extend(prescribed_velocity_definitions.iter().copied());
    if volume.relations.iter().copied().collect::<BTreeSet<_>>() != expected_relations {
        return Err(lowering_error(
            volume.domain,
            "steady Stokes volume contains a Relation outside the force, momentum, incompressibility, and normal-pressure coefficient definitions",
        ));
    }

    Ok(ResolvedNormalPressures2d {
        by_key,
        fields: coefficient_fields,
        definitions,
    })
}

mod closure;
use closure::require_closed_model;
