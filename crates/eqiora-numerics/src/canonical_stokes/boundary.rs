//! Package-neutral normalization of exact steady-Stokes boundary meaning.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, RawId};
use eqiora_ir::{OperatorApplicationProof, StandardPureOperator};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};
use eqiora_sem::KernelProgram;

use crate::canonical_boundary::{
    BoundaryRelationBinding, CartesianBoundaryEntry, CartesianBoundaryInventory,
    PhysicalBoundaryDisposition, PhysicalBoundaryQuantity, PrescribedBoundaryLaw,
    exact_cartesian_boundaries, normalize_field_physical_interface,
};
use crate::spatial_expression::ScalarSpatialExpression;

use super::additive_boundary::{
    additive_prescribed_complete_velocity_parts, additive_prescribed_normal_velocity_parts,
};
use super::expression::{
    IncompressibleStressForm, load_definition_root, lower_incompressible_stress_viscosity,
};
use super::prescribed_velocity::SteadyStokesPrescribedVelocityTrace2d;
use super::support::{
    is_field, lowering_error, relation_expression, relations_on, require_continuous_relation,
    typed_relation, unique_root,
};
use crate::additive_residual::AdditiveResidualView;

#[derive(Debug)]
pub(crate) struct LoweredStokesBoundary<const D: usize> {
    pub(crate) inventory: CartesianBoundaryInventory<D>,
    pub(crate) normal_pressure_sources:
        BTreeMap<(usize, eqiora_schema::kernel::BoundarySide), NormalPressureSource2d>,
    pub(crate) prescribed_velocity_traces: BTreeMap<
        (usize, eqiora_schema::kernel::BoundarySide),
        SteadyStokesPrescribedVelocityTrace2d,
    >,
    pub(crate) prescribed_velocity_fields: BTreeSet<RawId>,
    pub(crate) prescribed_velocity_definitions: BTreeSet<RawId>,
    pub(crate) normal_velocity_expressions:
        BTreeMap<(usize, eqiora_schema::kernel::BoundarySide), ScalarSpatialExpression<f64>>,
    pub(crate) normal_velocity_fields: BTreeSet<RawId>,
    pub(crate) normal_velocity_definitions: BTreeSet<RawId>,
    pub(crate) relations: BTreeSet<RawId>,
    pub(crate) boundary_relations: Vec<BoundaryRelationBinding>,
    pub(crate) ports: BTreeSet<RawId>,
    pub(crate) connections: BTreeSet<RawId>,
    pub(crate) connector_domains: BTreeSet<RawId>,
    pub(crate) uninterpreted_live_relations: BTreeSet<RawId>,
}

pub(crate) type LoweredStokesBoundary2d = LoweredStokesBoundary<2>;

#[derive(Debug)]
pub(super) struct LoweredNamedStokesBoundary2d {
    pub(super) entries: BTreeMap<String, CartesianBoundaryEntry>,
    pub(super) normal_pressure_sources: BTreeMap<String, NormalPressureSource2d>,
    pub(super) prescribed_velocity_traces: BTreeMap<String, SteadyStokesPrescribedVelocityTrace2d>,
    pub(super) prescribed_velocity_fields: BTreeSet<RawId>,
    pub(super) prescribed_velocity_definitions: BTreeSet<RawId>,
    pub(super) normal_velocity_expressions: BTreeMap<String, ScalarSpatialExpression<f64>>,
    pub(super) boundary_relations: Vec<BoundaryRelationBinding>,
    pub(super) ports: BTreeSet<RawId>,
    pub(super) connections: BTreeSet<RawId>,
    pub(super) connector_domains: BTreeSet<RawId>,
    pub(super) uninterpreted_live_relations: BTreeSet<RawId>,
}

struct LoweredBoundaryEntries<K> {
    entries: BTreeMap<K, CartesianBoundaryEntry>,
    normal_pressure_sources: BTreeMap<K, NormalPressureSource2d>,
    prescribed_velocity_traces: BTreeMap<K, SteadyStokesPrescribedVelocityTrace2d>,
    prescribed_velocity_fields: BTreeSet<RawId>,
    prescribed_velocity_definitions: BTreeSet<RawId>,
    relations: BTreeSet<RawId>,
    boundary_relations: Vec<BoundaryRelationBinding>,
    ports: BTreeSet<RawId>,
    connections: BTreeSet<RawId>,
    connector_domains: BTreeSet<RawId>,
    uninterpreted_live_relations: BTreeSet<RawId>,
}

struct NormalVelocityProjection<K> {
    expressions: BTreeMap<K, ScalarSpatialExpression<f64>>,
    fields: BTreeSet<RawId>,
    definitions: BTreeSet<RawId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NormalPressureSource2d {
    Zero,
    Field { field: RawId, law_relation: RawId },
}

#[derive(Debug, Clone, Copy)]
struct BoundaryCandidate {
    disposition: PhysicalBoundaryDisposition,
    normal_pressure: Option<NormalPressureSource2d>,
}

#[derive(Clone, Copy)]
struct StressBoundaryContext<'a> {
    program: &'a KernelProgram,
    velocity: RawId,
    pressure: RawId,
    volume_viscosity: &'a ScalarSpatialExpression<f64>,
    stress_form: IncompressibleStressForm,
}

pub(super) fn lower(
    program: &KernelProgram,
    domain: RawId,
    velocity: RawId,
    pressure: RawId,
    volume_viscosity: &ScalarSpatialExpression<f64>,
) -> Result<LoweredStokesBoundary2d, Diagnostic> {
    lower_dimension::<2>(program, domain, velocity, pressure, volume_viscosity)
}

pub(super) fn lower_dimension<const D: usize>(
    program: &KernelProgram,
    domain: RawId,
    velocity: RawId,
    pressure: RawId,
    volume_viscosity: &ScalarSpatialExpression<f64>,
) -> Result<LoweredStokesBoundary<D>, Diagnostic> {
    lower_dimension_with_stress(
        program,
        domain,
        velocity,
        pressure,
        volume_viscosity,
        IncompressibleStressForm::SymmetricNewtonian,
    )
}

pub(crate) fn lower_with_boundaries(
    program: &KernelProgram,
    domain: RawId,
    velocity: RawId,
    pressure: RawId,
    volume_viscosity: &ScalarSpatialExpression<f64>,
    exact_boundaries: BTreeMap<(usize, eqiora_schema::kernel::BoundarySide), RawId>,
) -> Result<LoweredStokesBoundary<2>, Diagnostic> {
    let lowered = lower_entries::<2, _>(
        program,
        domain,
        velocity,
        pressure,
        volume_viscosity,
        exact_boundaries,
        IncompressibleStressForm::SymmetricNewtonian,
    )?;
    let normal_velocity = normal_velocity_projection(&lowered.prescribed_velocity_traces);
    Ok(LoweredStokesBoundary {
        inventory: CartesianBoundaryInventory::new(lowered.entries),
        normal_pressure_sources: lowered.normal_pressure_sources,
        prescribed_velocity_traces: lowered.prescribed_velocity_traces,
        prescribed_velocity_fields: lowered.prescribed_velocity_fields,
        prescribed_velocity_definitions: lowered.prescribed_velocity_definitions,
        normal_velocity_expressions: normal_velocity.expressions,
        normal_velocity_fields: normal_velocity.fields,
        normal_velocity_definitions: normal_velocity.definitions,
        relations: lowered.relations,
        boundary_relations: lowered.boundary_relations,
        ports: lowered.ports,
        connections: lowered.connections,
        connector_domains: lowered.connector_domains,
        uninterpreted_live_relations: lowered.uninterpreted_live_relations,
    })
}

fn lower_dimension_with_stress<const D: usize>(
    program: &KernelProgram,
    domain: RawId,
    velocity: RawId,
    pressure: RawId,
    volume_viscosity: &ScalarSpatialExpression<f64>,
    stress_form: IncompressibleStressForm,
) -> Result<LoweredStokesBoundary<D>, Diagnostic> {
    let exact_boundaries = exact_cartesian_boundaries::<D>(program, domain)?;
    let lowered = lower_entries::<D, _>(
        program,
        domain,
        velocity,
        pressure,
        volume_viscosity,
        exact_boundaries,
        stress_form,
    )?;
    let normal_velocity = normal_velocity_projection(&lowered.prescribed_velocity_traces);
    Ok(LoweredStokesBoundary {
        inventory: CartesianBoundaryInventory::new(lowered.entries),
        normal_pressure_sources: lowered.normal_pressure_sources,
        prescribed_velocity_traces: lowered.prescribed_velocity_traces,
        prescribed_velocity_fields: lowered.prescribed_velocity_fields,
        prescribed_velocity_definitions: lowered.prescribed_velocity_definitions,
        normal_velocity_expressions: normal_velocity.expressions,
        normal_velocity_fields: normal_velocity.fields,
        normal_velocity_definitions: normal_velocity.definitions,
        relations: lowered.relations,
        boundary_relations: lowered.boundary_relations,
        ports: lowered.ports,
        connections: lowered.connections,
        connector_domains: lowered.connector_domains,
        uninterpreted_live_relations: lowered.uninterpreted_live_relations,
    })
}

pub(super) fn lower_named(
    program: &KernelProgram,
    domain: RawId,
    velocity: RawId,
    pressure: RawId,
    volume_viscosity: &ScalarSpatialExpression<f64>,
    exact_boundaries: BTreeMap<String, RawId>,
) -> Result<LoweredNamedStokesBoundary2d, Diagnostic> {
    lower_named_with_stress(
        program,
        domain,
        velocity,
        pressure,
        volume_viscosity,
        exact_boundaries,
        IncompressibleStressForm::SymmetricNewtonian,
    )
}

pub(super) fn lower_named_with_stress(
    program: &KernelProgram,
    domain: RawId,
    velocity: RawId,
    pressure: RawId,
    volume_viscosity: &ScalarSpatialExpression<f64>,
    exact_boundaries: BTreeMap<String, RawId>,
    stress_form: IncompressibleStressForm,
) -> Result<LoweredNamedStokesBoundary2d, Diagnostic> {
    let lowered = lower_entries::<2, _>(
        program,
        domain,
        velocity,
        pressure,
        volume_viscosity,
        exact_boundaries,
        stress_form,
    )?;
    let normal_velocity = normal_velocity_projection(&lowered.prescribed_velocity_traces);
    Ok(LoweredNamedStokesBoundary2d {
        entries: lowered.entries,
        normal_pressure_sources: lowered.normal_pressure_sources,
        prescribed_velocity_traces: lowered.prescribed_velocity_traces,
        prescribed_velocity_fields: lowered.prescribed_velocity_fields,
        prescribed_velocity_definitions: lowered.prescribed_velocity_definitions,
        normal_velocity_expressions: normal_velocity.expressions,
        boundary_relations: lowered.boundary_relations,
        ports: lowered.ports,
        connections: lowered.connections,
        connector_domains: lowered.connector_domains,
        uninterpreted_live_relations: lowered.uninterpreted_live_relations,
    })
}

fn normal_velocity_projection<K: Clone + Ord>(
    traces: &BTreeMap<K, SteadyStokesPrescribedVelocityTrace2d>,
) -> NormalVelocityProjection<K> {
    let mut expressions = BTreeMap::new();
    let mut fields = BTreeSet::new();
    let mut definitions = BTreeSet::new();
    for (key, trace) in traces {
        if let SteadyStokesPrescribedVelocityTrace2d::Normal {
            coefficient_field,
            definition_relation,
            expression,
            ..
        } = trace
        {
            expressions.insert(key.clone(), expression.clone());
            fields.insert(*coefficient_field);
            definitions.insert(*definition_relation);
        }
    }
    NormalVelocityProjection {
        expressions,
        fields,
        definitions,
    }
}

fn lower_entries<const D: usize, K: Clone + Ord>(
    program: &KernelProgram,
    domain: RawId,
    velocity: RawId,
    pressure: RawId,
    volume_viscosity: &ScalarSpatialExpression<f64>,
    exact_boundaries: BTreeMap<K, RawId>,
    stress_form: IncompressibleStressForm,
) -> Result<LoweredBoundaryEntries<K>, Diagnostic> {
    let mut entries = BTreeMap::new();
    let mut normal_pressure_sources = BTreeMap::new();
    let mut prescribed_velocity_traces = BTreeMap::new();
    let mut prescribed_velocity_fields = BTreeSet::new();
    let mut prescribed_velocity_definitions = BTreeSet::new();
    let mut admitted_relations = BTreeSet::new();
    let mut boundary_relations = BTreeSet::new();
    let mut admitted_ports = BTreeSet::new();
    let mut admitted_connections = BTreeSet::new();
    let mut connector_domains = BTreeSet::new();
    let mut uninterpreted_live_relations = BTreeSet::new();

    for (key, boundary) in exact_boundaries {
        let relations = relations_on(program, boundary);
        for relation in &relations {
            require_continuous_relation(program, *relation)?;
        }

        let mut direct = Vec::new();
        for relation in &relations {
            if let Some(disposition) = direct_disposition(
                program,
                *relation,
                velocity,
                pressure,
                volume_viscosity,
                stress_form,
            )? {
                direct.push((*relation, disposition));
            }
        }
        let (candidate, side_relations) = if direct.len() == 1 && relations.len() == 1 {
            (direct[0].1, BTreeSet::from([direct[0].0]))
        } else {
            let normalized = normalize_physical_interface(
                program,
                boundary,
                velocity,
                pressure,
                volume_viscosity,
                &relations,
                stress_form,
            )?;
            let side_relations = normalized.relations.clone();
            admitted_ports.extend(normalized.ports);
            admitted_connections.insert(normalized.connection);
            connector_domains.extend(normalized.connector_domains);
            uninterpreted_live_relations.extend(normalized.uninterpreted_live_relations);
            let normal_pressure = normalized_pressure_source(
                program,
                normalized.disposition,
                normalized.interface_port,
            )?;
            (
                BoundaryCandidate {
                    disposition: normalized.disposition,
                    normal_pressure,
                },
                side_relations,
            )
        };
        admitted_relations.extend(side_relations.iter().copied());
        boundary_relations.extend(
            side_relations
                .into_iter()
                .map(|relation| BoundaryRelationBinding::new(boundary, relation)),
        );
        if let Some(normal_pressure) = candidate.normal_pressure {
            normal_pressure_sources.insert(key.clone(), normal_pressure);
        }
        if let PhysicalBoundaryDisposition::Prescribed(law) = candidate.disposition
            && law.quantity() == PhysicalBoundaryQuantity::Trace
        {
            let trace = prescribed_velocity_trace::<D>(
                program,
                law.relation(),
                velocity,
                domain,
                boundary,
            )?;
            prescribed_velocity_fields.insert(trace.coefficient_field());
            prescribed_velocity_definitions.insert(trace.definition_relation());
            prescribed_velocity_traces.insert(key.clone(), trace);
        }
        entries.insert(
            key,
            CartesianBoundaryEntry::new(boundary, candidate.disposition),
        );
    }

    Ok(LoweredBoundaryEntries {
        entries,
        normal_pressure_sources,
        prescribed_velocity_traces,
        prescribed_velocity_fields,
        prescribed_velocity_definitions,
        relations: admitted_relations,
        boundary_relations: boundary_relations.into_iter().collect(),
        ports: admitted_ports,
        connections: admitted_connections,
        connector_domains,
        uninterpreted_live_relations,
    })
}

fn direct_disposition(
    program: &KernelProgram,
    relation: RawId,
    velocity: RawId,
    pressure: RawId,
    volume_viscosity: &ScalarSpatialExpression<f64>,
    stress_form: IncompressibleStressForm,
) -> Result<Option<BoundaryCandidate>, Diagnostic> {
    let expression = &relation_expression(program, relation)?;
    let [root] = expression.roots() else {
        return Ok(None);
    };
    if additive_prescribed_complete_velocity_parts(expression, *root, velocity, relation)?.is_some()
        || additive_prescribed_normal_velocity_parts(
            program, expression, *root, velocity, relation,
        )?
        .is_some()
    {
        return Ok(Some(BoundaryCandidate {
            disposition: PhysicalBoundaryDisposition::Prescribed(PrescribedBoundaryLaw::new(
                PhysicalBoundaryQuantity::Trace,
                relation,
            )),
            normal_pressure: None,
        }));
    }
    let stress_context = StressBoundaryContext {
        program,
        velocity,
        pressure,
        volume_viscosity,
        stress_form,
    };
    let view = AdditiveResidualView::derive(expression, *root, relation)?;
    match view.leaves() {
        [leaf]
            if matches!(
                expression.node(leaf.value()),
                Some(ExprNode::Trace { value, .. }) if is_field(expression, *value, velocity)
            ) =>
        {
            Ok(Some(BoundaryCandidate {
                disposition: PhysicalBoundaryDisposition::TraceZero,
                normal_pressure: None,
            }))
        }
        [leaf]
            if matches!(
                expression.node(leaf.value()),
                Some(ExprNode::NormalComponent { .. })
            ) =>
        {
            let Some(ExprNode::NormalComponent { value: stress, .. }) =
                expression.node(leaf.value())
            else {
                unreachable!("guard proves normal component")
            };
            require_matching_stress(expression, *stress, relation, stress_context)?;
            Ok(Some(BoundaryCandidate {
                disposition: PhysicalBoundaryDisposition::FluxZero,
                normal_pressure: Some(NormalPressureSource2d::Zero),
            }))
        }
        [first, second] if first.sign() == second.sign() => {
            let Some(field) = direct_normal_pressure_field(
                expression,
                first.value(),
                second.value(),
                relation,
                stress_context,
            )?
            else {
                return Ok(None);
            };
            Ok(Some(BoundaryCandidate {
                disposition: PhysicalBoundaryDisposition::Prescribed(PrescribedBoundaryLaw::new(
                    PhysicalBoundaryQuantity::Flux,
                    relation,
                )),
                normal_pressure: Some(NormalPressureSource2d::Field {
                    field,
                    law_relation: relation,
                }),
            }))
        }
        _ => Ok(None),
    }
}

/// Two-dimensional entry point for the single retained trace owner.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn prescribed_velocity_trace_2d(
    program: &KernelProgram,
    relation: RawId,
    velocity: RawId,
    domain: RawId,
    boundary: RawId,
) -> Result<SteadyStokesPrescribedVelocityTrace2d, Diagnostic> {
    prescribed_velocity_trace::<2>(program, relation, velocity, domain, boundary)
}

fn prescribed_velocity_trace<const D: usize>(
    program: &KernelProgram,
    relation: RawId,
    velocity: RawId,
    domain: RawId,
    boundary: RawId,
) -> Result<SteadyStokesPrescribedVelocityTrace2d, Diagnostic> {
    let expression = &relation_expression(program, relation)?;
    let [root] = expression.roots() else {
        return Err(lowering_error(
            relation,
            "prescribed velocity trace requires exactly one residual root",
        ));
    };
    if let Some(potential) =
        additive_prescribed_complete_velocity_parts(expression, *root, velocity, relation)?
    {
        if D != 2 {
            return Err(lowering_error(
                relation,
                "complete affine-potential Stokes trace is restricted to two dimensions",
            ));
        }
        let potential = match expression.node(potential) {
            Some(ExprNode::Symbol(SymbolRef::Field(field))) => field.erase(),
            _ => unreachable!("complete trace matcher returns one Field symbol"),
        };
        let candidates = relations_on(program, domain)
            .into_iter()
            .filter_map(|definition| {
                let definition_expression = &relation_expression(program, definition).ok()?;
                let root = unique_root(definition_expression, definition).ok()?;
                load_definition_root(definition_expression, root, potential)
                    .map(|source| (definition, source))
            })
            .collect::<Vec<_>>();
        if candidates.len() != 1 {
            return Err(lowering_error(
                potential,
                format!(
                    "complete velocity potential requires exactly one scalar definition Relation, found {}",
                    candidates.len()
                ),
            ));
        }
        let (definition, source) = candidates[0];
        require_continuous_relation(program, definition)?;
        let speed_parameter = exact_complete_potential_source(
            &relation_expression(program, definition)?,
            source,
        )
        .ok_or_else(|| {
            lowering_error(
                definition,
                "complete velocity potential definition must be exactly `U * coordinate(0)`",
            )
        })?;
        let tape = crate::spatial_expression::lower(
            program,
            &relation_expression(program, definition)?,
            source,
            definition,
            D,
        )?;
        let [retained_speed] = tape.parameter_fields() else {
            return Err(lowering_error(
                definition,
                "complete velocity potential must retain exactly one speed Parameter",
            ));
        };
        if retained_speed.erase() != speed_parameter {
            return Err(lowering_error(
                definition,
                "complete velocity potential Parameter identity differs after tape lowering",
            ));
        }
        return SteadyStokesPrescribedVelocityTrace2d::complete_affine_potential(
            boundary,
            relation,
            potential,
            definition,
            speed_parameter,
            tape,
        )
        .map_err(|error| lowering_error(relation, error.message()));
    }

    let (expression, field, definition) =
        prescribed_normal_velocity_expression::<D>(program, relation, velocity, domain)?;
    Ok(SteadyStokesPrescribedVelocityTrace2d::normal(
        boundary, relation, field, definition, expression,
    ))
}

fn exact_complete_potential_source(expression: &ExprDag, source: ExprId) -> Option<RawId> {
    let ExprNode::Mul(left, right) = expression.node(source)? else {
        return None;
    };
    [(*left, *right), (*right, *left)]
        .into_iter()
        .find_map(|(parameter, coordinate)| {
            match (expression.node(parameter), expression.node(coordinate)) {
                (
                    Some(ExprNode::Symbol(SymbolRef::Parameter(parameter))),
                    Some(ExprNode::Symbol(SymbolRef::Coordinate { axis: 0, .. })),
                ) => Some(parameter.erase()),
                _ => None,
            }
        })
}

fn direct_normal_pressure_field(
    expression: &ExprDag,
    first: ExprId,
    second: ExprId,
    relation: RawId,
    context: StressBoundaryContext<'_>,
) -> Result<Option<RawId>, Diagnostic> {
    let typed = typed_relation(context.program, relation)?;
    debug_assert_eq!(typed.expression(), expression);
    for (traction, pressure_value) in [(first, second), (second, first)] {
        let Some(field) = normal_pressure_field(&typed, pressure_value, relation)? else {
            continue;
        };
        let Some(ExprNode::NormalComponent { value: stress, .. }) = expression.node(traction)
        else {
            continue;
        };
        require_matching_stress(expression, *stress, relation, context)?;
        return Ok(Some(field));
    }
    Ok(None)
}

fn normalized_pressure_source(
    program: &KernelProgram,
    disposition: PhysicalBoundaryDisposition,
    interface_port: RawId,
) -> Result<Option<NormalPressureSource2d>, Diagnostic> {
    match disposition {
        PhysicalBoundaryDisposition::TraceZero => Ok(None),
        PhysicalBoundaryDisposition::FluxZero => Ok(Some(NormalPressureSource2d::Zero)),
        PhysicalBoundaryDisposition::Prescribed(law) => {
            if law.quantity() == PhysicalBoundaryQuantity::Trace {
                return Ok(None);
            }
            let expression = &relation_expression(program, law.relation())?;
            let typed = typed_relation(program, law.relation())?;
            debug_assert_eq!(typed.expression(), expression);
            let [root] = expression.roots() else {
                return Err(lowering_error(
                    law.relation(),
                    "normal-pressure terminal requires exactly one residual root",
                ));
            };
            let view = AdditiveResidualView::derive(expression, *root, law.relation())?;
            let [first, second] = view.leaves() else {
                return Err(lowering_error(
                    law.relation(),
                    "normal-pressure terminal requires exactly one terminal-flux and one pressure leaf",
                ));
            };
            if !first.sign().is_opposite(second.sign()) {
                return Err(view.mismatch(
                    "normal-pressure terminal flux and pressure roles must have opposite signs, up to whole-equation reversal",
                ));
            }
            let (terminal_port, pressure_value) = if let Some(port) =
                port_flux(expression, first.value()).filter(|port| *port != interface_port)
            {
                (port, second.value())
            } else if let Some(port) =
                port_flux(expression, second.value()).filter(|port| *port != interface_port)
            {
                (port, first.value())
            } else {
                return Err(lowering_error(
                    law.relation(),
                    "normal-pressure terminal must prescribe its peer Port flux",
                ));
            };
            let Some(field) = normal_pressure_field(&typed, pressure_value, law.relation())? else {
                return Err(lowering_error(
                    law.relation(),
                    "normal-pressure terminal requires the parent-outward normal of one isotropic pressure Field",
                ));
            };
            debug_assert_ne!(terminal_port, interface_port);
            Ok(Some(NormalPressureSource2d::Field {
                field,
                law_relation: law.relation(),
            }))
        }
        PhysicalBoundaryDisposition::PortBinding { .. } => Ok(None),
    }
}

fn prescribed_normal_velocity_expression<const D: usize>(
    program: &KernelProgram,
    relation: RawId,
    velocity: RawId,
    domain: RawId,
) -> Result<(ScalarSpatialExpression<f64>, RawId, RawId), Diagnostic> {
    let expression = &relation_expression(program, relation)?;
    let [root] = expression.roots() else {
        return Err(lowering_error(
            relation,
            "prescribed velocity trace requires exactly one residual root",
        ));
    };
    let Some((value, sign)) =
        additive_prescribed_normal_velocity_parts(program, expression, *root, velocity, relation)?
    else {
        return Err(lowering_error(
            relation,
            "prescribed velocity trace must be the parent-outward normal of one scalar isotropic lift",
        ));
    };
    let Some(ExprNode::Symbol(SymbolRef::Field(field))) = expression.node(value) else {
        return Err(lowering_error(
            relation,
            "prescribed normal velocity must use one scalar volume coefficient Field",
        ));
    };
    let field = field.erase();
    let candidates = relations_on(program, domain)
        .into_iter()
        .filter_map(|definition| {
            let definition_expression = &relation_expression(program, definition).ok()?;
            let root = unique_root(definition_expression, definition).ok()?;
            load_definition_root(definition_expression, root, field)
                .map(|source| (definition, source))
        })
        .collect::<Vec<_>>();
    if candidates.len() != 1 {
        return Err(lowering_error(
            field,
            format!(
                "prescribed normal-velocity coefficient Field requires exactly one scalar definition Relation, found {}",
                candidates.len()
            ),
        ));
    }
    let (definition, source) = candidates[0];
    require_continuous_relation(program, definition)?;
    let expression = crate::spatial_expression::lower(
        program,
        &relation_expression(program, definition)?,
        source,
        definition,
        D,
    )?
    .multiply(ScalarSpatialExpression::constant(D, sign));
    Ok((expression, field, definition))
}

fn normal_pressure_field(
    residual: &TypedResidual<RawId>,
    value: ExprId,
    owner: RawId,
) -> Result<Option<RawId>, Diagnostic> {
    let expression = residual.expression();
    let Some(ExprNode::NormalComponent { value: tensor, .. }) = expression.node(value) else {
        return Ok(None);
    };
    let Some(proof) =
        OperatorApplicationProof::classify(residual, *tensor, StandardPureOperator::IsotropicLift)
            .map_err(|error| {
                lowering_error(
                    owner,
                    format!(
                        "isotropic_lift calculus proof failed at expression node {}: {error}",
                        tensor.index()
                    ),
                )
            })?
    else {
        return Ok(None);
    };
    Ok(match expression.node(proof.operand()) {
        Some(ExprNode::Symbol(SymbolRef::Field(field))) => Some(field.erase()),
        _ => None,
    })
}

fn normalize_physical_interface(
    program: &KernelProgram,
    boundary: RawId,
    velocity: RawId,
    pressure: RawId,
    volume_viscosity: &ScalarSpatialExpression<f64>,
    boundary_relations: &[RawId],
    stress_form: IncompressibleStressForm,
) -> Result<crate::canonical_boundary::NormalizedFieldPhysicalInterface, Diagnostic> {
    let mut interfaces = Vec::new();
    for relation in boundary_relations {
        if let Some(port) = interface_port(
            program,
            *relation,
            velocity,
            pressure,
            volume_viscosity,
            stress_form,
        )? {
            interfaces.push((*relation, port));
        }
    }
    if interfaces.len() != 1 {
        return Err(lowering_error(
            boundary,
            format!(
                "steady-Stokes boundary requires one direct law or one exact field-physical interface, found {} interface candidates",
                interfaces.len()
            ),
        ));
    }
    let (interface_relation, interface_port) = interfaces[0];
    normalize_field_physical_interface(
        program,
        boundary,
        boundary_relations,
        interface_relation,
        interface_port,
    )
}

fn interface_port(
    program: &KernelProgram,
    relation: RawId,
    velocity: RawId,
    pressure: RawId,
    volume_viscosity: &ScalarSpatialExpression<f64>,
    stress_form: IncompressibleStressForm,
) -> Result<Option<RawId>, Diagnostic> {
    let expression = &relation_expression(program, relation)?;
    let [first, second] = expression.roots() else {
        return Ok(None);
    };
    let stress_context = StressBoundaryContext {
        program,
        velocity,
        pressure,
        volume_viscosity,
        stress_form,
    };
    let mut trace_port = None;
    let mut flux_port = None;
    for root in [*first, *second] {
        let view = AdditiveResidualView::derive(expression, root, relation)?;
        let [left, right] = view.leaves() else {
            continue;
        };
        if !left.sign().is_opposite(right.sign()) {
            continue;
        }
        for (operator, port) in [(left, right), (right, left)] {
            if matches!(expression.node(operator.value()), Some(ExprNode::Trace { value, .. }) if is_field(expression, *value, velocity))
            {
                trace_port = port_trace(expression, port.value());
                continue;
            }
            if let Some(ExprNode::NormalComponent { value: stress, .. }) =
                expression.node(operator.value())
                && let Some(port) = port_flux(expression, port.value())
            {
                require_matching_stress(expression, *stress, relation, stress_context)?;
                flux_port = Some(port);
            }
        }
    }
    match (trace_port, flux_port) {
        (Some(trace), Some(flux)) if trace == flux => Ok(Some(trace)),
        (None, None) => Ok(None),
        _ => Err(lowering_error(
            relation,
            "steady-Stokes interface must pair exact velocity trace and Newtonian outward traction with one Port",
        )),
    }
}

fn require_matching_stress(
    expression: &ExprDag,
    stress: ExprId,
    relation: RawId,
    context: StressBoundaryContext<'_>,
) -> Result<(), Diagnostic> {
    let typed = typed_relation(context.program, relation)?;
    debug_assert_eq!(typed.expression(), expression);
    let viscosity = lower_incompressible_stress_viscosity(
        context.program,
        &typed,
        stress,
        context.velocity,
        context.pressure,
        relation,
        context.stress_form,
    )?
    .ok_or_else(|| {
        lowering_error(
            relation,
            match context.stress_form {
                IncompressibleStressForm::SymmetricNewtonian => {
                    "boundary traction must use the exact incompressible Newtonian stress"
                }
            },
        )
    })?;
    if !viscosity.is_same_coefficient_as(context.volume_viscosity) {
        return Err(lowering_error(
            relation,
            "boundary and volume dynamic viscosity coefficients differ",
        ));
    }
    Ok(())
}

fn port_trace(expression: &ExprDag, value: ExprId) -> Option<RawId> {
    match expression.node(value) {
        Some(ExprNode::Symbol(SymbolRef::PortTrace(port))) => Some(port.erase()),
        _ => None,
    }
}

fn port_flux(expression: &ExprDag, value: ExprId) -> Option<RawId> {
    match expression.node(value) {
        Some(ExprNode::Symbol(SymbolRef::PortFlux(port))) => Some(port.erase()),
        _ => None,
    }
}
