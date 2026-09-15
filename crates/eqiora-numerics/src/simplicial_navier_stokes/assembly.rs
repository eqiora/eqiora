use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPlan, AssemblyReport, AssemblyTarget,
    IndexedAssemblyWork, LocalContribution, LocalUnknown, TargetAssemblyMap,
};
use eqiora_core::Diagnostic;
use eqiora_meshing::{
    AffineGeometryMap, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, SimplicialMesh,
};
use eqiora_solver::{CanonicalCsrSystemView, LinearOperatorProperties};
use std::sync::{Arc, OnceLock};

#[cfg(test)]
thread_local! {
    static PACKET_EVALUATIONS: std::cell::Cell<[usize; 3]> = const { std::cell::Cell::new([0; 3]) };
}

#[cfg(test)]
pub(super) fn reset_packet_evaluations() {
    PACKET_EVALUATIONS.set([0; 3]);
}

#[cfg(test)]
pub(super) fn packet_evaluations() -> [usize; 3] {
    PACKET_EVALUATIONS.get()
}

use super::api::{MiniNavierStokesStepPlan2d, SimplicialMiniNavierStokesState2d};
use super::element::{FixedDomainViscousForm, MiniNavierStokesCell};
use super::{COMPONENTS, DIMENSION, invalid};
use crate::assembled_linearization::AssembledLinearizedRelation;
use crate::jacobian_audit::{StructuralJacobianPattern, StructuralJacobianPatternBuilder};
use crate::operator::LocalOperator;
use crate::simplicial_elliptic::SimplicialP1Field;
use crate::simplicial_stokes::boundary::{PreparedBoundary2d, PressureReferenceKind2d};
use crate::simplicial_stokes::constraint::MiniPressureMeanConstraintCell;
use crate::simplicial_stokes::layout::MixedLayout;
use crate::simplicial_stokes::{CELL_LOCAL_DOF_COUNT, CONSTRAINT_LOCAL_DOF_COUNT};
use crate::simplicial_stokes::{
    SimplicialMiniStokesBoundary2d, SimplicialMiniStokesPressureReference2d,
    SimplicialMiniVelocityField2d,
};

pub(super) struct StepAssembly {
    pub(super) relation: AssembledLinearizedRelation,
    pub(super) residual: Vec<f64>,
    pub(super) layout: Arc<MixedLayout>,
    pub(super) velocity: SimplicialMiniVelocityField2d,
    pub(super) pressure: SimplicialP1Field,
    pub(super) pressure_reference: SimplicialMiniStokesPressureReference2d,
    pub(super) gauge_multiplier: Option<f64>,
    pub(super) named_reaction_vertices: Arc<Vec<(String, Vec<usize>)>>,
    pub(super) assembly_report: AssemblyReport,
    evaluated_packets: Vec<EvaluatedStepPacket>,
    full_maps: Arc<Vec<Arc<AssemblyMap>>>,
}

impl StepAssembly {
    pub(super) fn residual_norm(&self) -> Result<f64, Diagnostic> {
        Ok(self
            .residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt())
    }

    pub(super) fn momentum_residual_norm(&self) -> f64 {
        self.residual[..self.layout.reduced_velocity_end()]
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    }

    pub(super) fn materialize_acceptance_data(&self) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        let mut full_residual = vec![0.0; self.layout.full_size];
        let mut gauge_weights = vec![0.0; self.layout.vertex_count];
        let gauge = self.layout.full_gauge();
        for (packet, evaluated) in self.evaluated_packets.iter().enumerate() {
            let map = self.full_maps.get(packet).ok_or_else(|| {
                invalid("accepted packet is outside the prepared full map inventory")
            })?;
            scatter_residual(&mut full_residual, map, &evaluated.residual)?;
            if let Some(gauge) = gauge {
                for (local_row, equation) in map.equations().iter().enumerate() {
                    let Some(equation) = equation else { continue };
                    let Some(pressure) = equation
                        .index()
                        .checked_sub(self.layout.full_pressure_offset)
                        .filter(|pressure| *pressure < self.layout.vertex_count)
                    else {
                        continue;
                    };
                    for (local_column, unknown) in map.unknowns().iter().enumerate() {
                        if matches!(unknown, LocalUnknown::Free(column) if column.index() == gauge)
                        {
                            gauge_weights[pressure] += evaluated
                                .assembly
                                .local()
                                .entry(local_row, local_column)
                                .expect("accepted packet map matches its local contribution");
                        }
                    }
                }
            }
        }
        if full_residual
            .iter()
            .chain(&gauge_weights)
            .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "accepted transient full residual data is non-finite",
            ));
        }
        Ok((full_residual, gauge_weights))
    }
}

#[derive(Clone)]
struct EvaluatedStepPacket {
    assembly: AssemblyPacket,
    residual: Vec<f64>,
}

pub(crate) struct PreparedStepStructure {
    boundary: PreparedBoundary2d,
    layout: Arc<MixedLayout>,
    named_reaction_vertices: Arc<Vec<(String, Vec<usize>)>>,
    reduced_maps: Vec<Arc<AssemblyMap>>,
    full_maps: Arc<Vec<Arc<AssemblyMap>>>,
    local_sizes: Vec<usize>,
    reduced_assembly_plan: AssemblyPlan,
    cell_geometries: Vec<AffineGeometryMap>,
    cell_quadrature: QuadratureRule,
    constraint_contributions: Vec<LocalContribution>,
    facet_geometries: Vec<AffineGeometryMap>,
    facet_incidence: Vec<eqiora_meshing::EntityIncidence>,
    facet_parent_vertices: Vec<Vec<usize>>,
    facet_quadrature: QuadratureRule,
    cell_count: usize,
    constraint_end: usize,
    packet_count: usize,
}

struct PreparedStepPoint<'a> {
    structure: &'a PreparedStepStructure,
    velocity: SimplicialMiniVelocityField2d,
    pressure: SimplicialP1Field,
    gauge_multiplier: Option<f64>,
}

impl PreparedStepPoint<'_> {
    fn reduced_map(&self, packet: usize) -> Result<&AssemblyMap, Diagnostic> {
        self.structure
            .reduced_maps
            .get(packet)
            .map(AsRef::as_ref)
            .ok_or_else(|| {
                invalid("transient MINI packet is outside the prepared contribution inventory")
            })
    }

    fn with_gauge(&self) -> bool {
        self.structure.boundary.pressure_reference == PressureReferenceKind2d::ZeroIntegral
    }
}

pub(super) fn initial_point<B>(
    mesh: &SimplicialMesh,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    state: &SimplicialMiniNavierStokesState2d,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
) -> Result<Vec<f64>, Diagnostic>
where
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let prepared = prepare_step_structure(
        mesh,
        boundary,
        essential_velocity,
        cell_quadrature,
        facet_quadrature,
    )?;
    initial_point_prepared(mesh, &prepared, state)
}

pub(crate) fn initial_point_prepared(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    state: &SimplicialMiniNavierStokesState2d,
) -> Result<Vec<f64>, Diagnostic> {
    require_same_mesh(mesh, state)?;
    require_pressure_policy(
        state,
        prepared.boundary.pressure_reference == PressureReferenceKind2d::ZeroIntegral,
    )?;
    prepared.layout.initial_point(
        &prepared.boundary.fixed_velocity,
        state.velocity().vertex_values(),
        state.velocity().cell_bubble_values(),
        state.pressure().vertex_values(),
        state.pressure_reference().gauge_multiplier(),
    )
}

pub(crate) fn prepare_step_structure<B>(
    mesh: &SimplicialMesh,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
) -> Result<PreparedStepStructure, Diagnostic>
where
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let _preparation =
        eqiora_execution::telemetry_span!(backend("assembly_preparation", "fixed-domain-mini"))
            .entered();
    let named_reaction_vertices = Arc::new(boundary.named_reaction_vertices(mesh));
    let boundary = boundary.prepare(mesh, essential_velocity)?;
    let with_gauge = boundary.pressure_reference == PressureReferenceKind2d::ZeroIntegral;
    let layout = Arc::new(MixedLayout::new(
        mesh,
        &boundary.fixed_velocity,
        with_gauge,
    )?);
    let cell_count = mesh
        .entity_count(DIMENSION)
        .expect("2D simplex mesh owns cells");
    let constraint_count = if with_gauge { cell_count } else { 0 };
    let constraint_end = cell_count
        .checked_add(constraint_count)
        .ok_or_else(|| invalid("transient MINI constraint packet count overflows usize"))?;
    let packet_count = constraint_end
        .checked_add(boundary.traction_facets.len())
        .ok_or_else(|| invalid("transient MINI packet count overflows usize"))?;
    let cell_geometries = (0..cell_count)
        .map(|cell| {
            mesh.geometry_map(MeshEntity::new(DIMENSION, cell))
                .expect("accepted simplex cell owns geometry")
        })
        .collect::<Vec<_>>();
    let constraint_contributions = if with_gauge {
        cell_geometries
            .iter()
            .map(|geometry| MiniPressureMeanConstraintCell.evaluate(geometry, cell_quadrature))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    let mut facet_geometries = Vec::new();
    let mut facet_incidence = Vec::new();
    let mut facet_parent_vertices = Vec::new();
    for facet in &boundary.traction_facets {
        let parents = mesh
            .incidence(facet.facet, DIMENSION)
            .ok_or_else(|| invalid("natural facet has no parent incidence"))?;
        let [parent] = parents.as_slice() else {
            return Err(invalid("natural facet requires exactly one parent cell"));
        };
        let vertices = mesh
            .entity_vertices(facet.facet)
            .expect("validated facet vertices");
        let cell_vertices = mesh
            .entity_vertices(parent.entity)
            .expect("accepted parent vertices");
        facet_parent_vertices.push(
            vertices
                .iter()
                .map(|vertex| {
                    cell_vertices
                        .iter()
                        .position(|value| value == vertex)
                        .expect("facet belongs to parent")
                })
                .collect(),
        );
        facet_geometries.push(
            mesh.geometry_map(facet.facet)
                .expect("validated facet geometry"),
        );
        facet_incidence.push(*parent);
    }
    let mut reduced_maps = Vec::with_capacity(packet_count);
    let mut full_maps = Vec::with_capacity(packet_count);
    let mut local_sizes = Vec::with_capacity(packet_count);
    for packet in 0..packet_count {
        let (local_size, reduced, full) = if packet < cell_count {
            let vertices = mesh
                .entity_vertices(MeshEntity::new(DIMENSION, packet))
                .expect("accepted simplex cell owns vertices");
            (
                CELL_LOCAL_DOF_COUNT,
                layout.reduced_cell_map(packet, &vertices, &boundary.fixed_velocity)?,
                layout.full_cell_map(packet, &vertices)?,
            )
        } else if packet < constraint_end {
            let vertices = mesh
                .entity_vertices(MeshEntity::new(DIMENSION, packet - cell_count))
                .expect("accepted simplex cell owns vertices");
            (
                CONSTRAINT_LOCAL_DOF_COUNT,
                layout.reduced_constraint_map(&vertices)?,
                layout.full_constraint_map(&vertices)?,
            )
        } else {
            let parent = facet_incidence[packet - constraint_end].entity;
            let vertices = mesh
                .entity_vertices(parent)
                .expect("accepted natural-facet parent vertices");
            (
                CELL_LOCAL_DOF_COUNT,
                layout.reduced_cell_map(parent.index(), &vertices, &boundary.fixed_velocity)?,
                layout.full_cell_map(parent.index(), &vertices)?,
            )
        };
        local_sizes.push(local_size);
        reduced_maps.push(Arc::new(reduced));
        full_maps.push(Arc::new(full));
    }
    let reduced_assembly_plan = AssemblyPlan::new(vec![AssemblyTarget::new(layout.reduced_size)?])?;
    let reduced_target = reduced_assembly_plan
        .target_id(0)
        .expect("one-target plan owns its target");
    let packet_structure = reduced_maps
        .iter()
        .map(|map| vec![TargetAssemblyMap::new(reduced_target, Arc::clone(map))])
        .collect::<Vec<_>>();
    let reduced_assembly_plan = reduced_assembly_plan.prepare(packet_structure)?;
    Ok(PreparedStepStructure {
        boundary,
        layout,
        named_reaction_vertices,
        reduced_maps,
        full_maps: Arc::new(full_maps),
        local_sizes,
        reduced_assembly_plan,
        cell_geometries,
        cell_quadrature: cell_quadrature.clone(),
        constraint_contributions,
        facet_geometries,
        facet_incidence,
        facet_parent_vertices,
        facet_quadrature: facet_quadrature.clone(),
        cell_count,
        constraint_end,
        packet_count,
    })
}

fn natural_facet_action(
    step: &PreparedStepPoint<'_>,
    facet: usize,
    plan: &MiniNavierStokesStepPlan2d,
    point: &[f64],
) -> Result<crate::form_compiler::region::RegionLinearization, Diagnostic> {
    let structure = step.structure;
    let incidence = structure.facet_incidence[facet];
    plan.form.natural_facet(
        &structure.cell_geometries[incidence.entity.index()],
        (
            &structure.facet_geometries[facet],
            incidence,
            &structure.facet_parent_vertices[facet],
        ),
        &structure.facet_quadrature,
        point,
        structure.boundary.traction_facets[facet].value,
    )
}

fn prepare_step_point<'a>(
    mesh: &SimplicialMesh,
    structure: &'a PreparedStepStructure,
    previous: &SimplicialMiniNavierStokesState2d,
    candidate: &[f64],
) -> Result<PreparedStepPoint<'a>, Diagnostic> {
    require_same_mesh(mesh, previous)?;
    let with_gauge = structure.boundary.pressure_reference == PressureReferenceKind2d::ZeroIntegral;
    require_pressure_policy(previous, with_gauge)?;
    if candidate.len() != structure.layout.reduced_size
        || candidate.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "MINI Navier--Stokes candidate must be finite and match the exact mixed layout",
        ));
    }
    let (vertex_values, bubble_values, pressure_values, gauge_multiplier) = structure
        .layout
        .reconstruct(candidate, &structure.boundary.fixed_velocity)?;
    Ok(PreparedStepPoint {
        structure,
        velocity: SimplicialMiniVelocityField2d::new(mesh.clone(), vertex_values, bubble_values)?,
        pressure: SimplicialP1Field::new(mesh.clone(), pressure_values)?,
        gauge_multiplier,
    })
}

pub(super) fn build_step_jacobian_pattern<B>(
    mesh: &SimplicialMesh,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
) -> Result<StructuralJacobianPattern, Diagnostic>
where
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let prepared = prepare_step_structure(
        mesh,
        boundary,
        essential_velocity,
        cell_quadrature,
        facet_quadrature,
    )?;
    build_step_jacobian_pattern_prepared(&prepared)
}

pub(crate) fn build_step_jacobian_pattern_prepared(
    prepared: &PreparedStepStructure,
) -> Result<StructuralJacobianPattern, Diagnostic> {
    let mut pattern = StructuralJacobianPatternBuilder::new(
        prepared.layout.reduced_size,
        prepared.layout.reduced_size,
        prepared.packet_count,
    )?;
    for packet in 0..prepared.packet_count {
        pattern.include_dense_local(
            packet,
            prepared.local_sizes[packet],
            &prepared.reduced_maps[packet],
        )?;
    }
    pattern.finish()
}

/// Reassemble the complete reduced residual without constructing a Jacobian.
///
/// This is the independent primal oracle for the accepted analytic
/// linearization. It shares only validation, field reconstruction, typed
/// assembly maps, and local weak-form kernels with the ordinary step path.
#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_step_residual<F, B>(
    mesh: &SimplicialMesh,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    body_force: &F,
    previous: &SimplicialMiniNavierStokesState2d,
    candidate: &[f64],
    plan: MiniNavierStokesStepPlan2d,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    viscous_form: FixedDomainViscousForm,
) -> Result<Vec<f64>, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let prepared = prepare_step_structure(
        mesh,
        boundary,
        essential_velocity,
        cell_quadrature,
        facet_quadrature,
    )?;
    assemble_step_residual_prepared(
        mesh,
        &prepared,
        body_force,
        previous,
        candidate,
        plan,
        viscous_form,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_step_residual_prepared<F>(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    body_force: &F,
    previous: &SimplicialMiniNavierStokesState2d,
    candidate: &[f64],
    plan: MiniNavierStokesStepPlan2d,
    viscous_form: FixedDomainViscousForm,
) -> Result<Vec<f64>, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let step = prepare_step_point(mesh, prepared, previous, candidate)?;
    let mut residual = vec![0.0; step.structure.layout.reduced_size];
    for packet in 0..step.structure.packet_count {
        let map = step.reduced_map(packet)?;
        let local_residual = if packet < step.structure.cell_count {
            let cell = MeshEntity::new(DIMENSION, packet);
            let geometry = &step.structure.cell_geometries[packet];
            let vertices = mesh
                .entity_vertices(cell)
                .expect("accepted simplex cell owns vertices");
            let cell = MiniNavierStokesCell {
                cell: packet,
                vertices: &vertices,
                form: &plan.form,
                previous_velocity: previous.velocity(),
                candidate_velocity: &step.velocity,
                candidate_pressure: step.pressure.vertex_values(),
                body_force,
            };
            match viscous_form {
                FixedDomainViscousForm::SymmetricNewtonian => {
                    cell.residual_prepared(geometry, &step.structure.cell_quadrature)?
                }
            }
        } else if packet < step.structure.constraint_end {
            let local_point = mapped_local_point(map, candidate)?;
            evaluate_local_residual(
                &step.structure.constraint_contributions[packet - step.structure.cell_count],
                &local_point,
            )?
        } else {
            let local_point = mapped_local_point(map, candidate)?;
            natural_facet_action(
                &step,
                packet - step.structure.constraint_end,
                &plan,
                &local_point,
            )?
            .residual
        };
        scatter_residual(&mut residual, map, &local_residual)?;
    }
    if residual.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "direct transient residual-only assembly produced a non-finite value",
        ));
    }
    Ok(residual)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_step_linearization<F, B>(
    mesh: &SimplicialMesh,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    body_force: &F,
    previous: &SimplicialMiniNavierStokesState2d,
    candidate: &[f64],
    plan: MiniNavierStokesStepPlan2d,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend,
    viscous_form: FixedDomainViscousForm,
) -> Result<StepAssembly, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let prepared = prepare_step_structure(
        mesh,
        boundary,
        essential_velocity,
        cell_quadrature,
        facet_quadrature,
    )?;
    assemble_step_linearization_prepared(
        mesh,
        &prepared,
        body_force,
        previous,
        candidate,
        plan,
        assembly,
        viscous_form,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_step_linearization_prepared<F>(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    body_force: &F,
    previous: &SimplicialMiniNavierStokesState2d,
    candidate: &[f64],
    plan: MiniNavierStokesStepPlan2d,
    assembly: &dyn AssemblyBackend,
    viscous_form: FixedDomainViscousForm,
) -> Result<StepAssembly, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let step = prepare_step_point(mesh, prepared, previous, candidate)?;
    let reduced_target = step
        .structure
        .reduced_assembly_plan
        .target_id(0)
        .expect("one-target plan owns reduced target");
    let evaluate_packet = |packet| {
        #[cfg(test)]
        PACKET_EVALUATIONS.set({
            let mut counts = PACKET_EVALUATIONS.get();
            counts[if packet < step.structure.cell_count {
                0
            } else if packet < step.structure.constraint_end {
                1
            } else {
                2
            }] += 1;
            counts
        });
        if packet < step.structure.cell_count {
            let cell = MeshEntity::new(DIMENSION, packet);
            let geometry = &step.structure.cell_geometries[packet];
            let vertices = mesh
                .entity_vertices(cell)
                .expect("accepted simplex cell owns vertices");
            let cell = MiniNavierStokesCell {
                cell: packet,
                vertices: &vertices,
                form: &plan.form,
                previous_velocity: previous.velocity(),
                candidate_velocity: &step.velocity,
                candidate_pressure: step.pressure.vertex_values(),
                body_force,
            };
            let linearization = match viscous_form {
                FixedDomainViscousForm::SymmetricNewtonian => {
                    cell.linearize_prepared(geometry, &step.structure.cell_quadrature)?
                }
            };
            let residual = linearization.residual().to_vec();
            let local = linearization.into_linear_contribution()?;
            let reduced = Arc::clone(&step.structure.reduced_maps[packet]);
            Ok(EvaluatedStepPacket {
                assembly: AssemblyPacket::new(
                    local,
                    vec![TargetAssemblyMap::new(reduced_target, reduced)],
                )?,
                residual,
            })
        } else if packet < step.structure.constraint_end {
            let cell_index = packet - step.structure.cell_count;
            let local = step.structure.constraint_contributions[cell_index].clone();
            let reduced = Arc::clone(&step.structure.reduced_maps[packet]);
            let residual = evaluate_linear_residual(&local, &reduced, candidate)?;
            Ok(EvaluatedStepPacket {
                assembly: AssemblyPacket::new(
                    local,
                    vec![TargetAssemblyMap::new(reduced_target, reduced)],
                )?,
                residual,
            })
        } else {
            let reduced = Arc::clone(&step.structure.reduced_maps[packet]);
            let point = mapped_local_point(&reduced, candidate)?;
            let action =
                natural_facet_action(&step, packet - step.structure.constraint_end, &plan, &point)?;
            let residual = action.residual.clone();
            let local = action.into_contribution(&point)?;
            Ok(EvaluatedStepPacket {
                assembly: AssemblyPacket::new(
                    local,
                    vec![TargetAssemblyMap::new(reduced_target, reduced)],
                )?,
                residual,
            })
        }
    };
    let evaluated = (0..step.structure.packet_count)
        .map(|_| OnceLock::<Result<EvaluatedStepPacket, Diagnostic>>::new())
        .collect::<Vec<_>>();
    let work = IndexedAssemblyWork::new(step.structure.packet_count, |packet: usize| {
        evaluated[packet]
            .get_or_init(|| {
                let _evaluation = eqiora_execution::telemetry_span!(backend(
                    "assembly_local_evaluation",
                    "fixed-domain-mini"
                ))
                .entered();
                evaluate_packet(packet)
            })
            .as_ref()
            .map(|evaluated| evaluated.assembly.clone())
            .map_err(Clone::clone)
    });
    let (systems, assembly_report) = {
        let _scatter = eqiora_execution::telemetry_span!(backend(
            "assembly_scatter_update",
            "fixed-domain-mini"
        ))
        .entered();
        assembly
            .assemble(&step.structure.reduced_assembly_plan, &work)?
            .into_parts()
    };
    let mut residual = vec![0.0; step.structure.layout.reduced_size];
    for (packet_index, evaluated) in evaluated.iter().enumerate() {
        let packet = evaluated
            .get()
            .expect("successful assembly evaluated every packet")
            .as_ref()
            .expect("successful assembly accepted every packet");
        scatter_residual(
            &mut residual,
            step.reduced_map(packet_index)?,
            &packet.residual,
        )?;
    }
    if residual.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "direct transient residual assembly produced a non-finite value",
        ));
    }
    let _finalization =
        eqiora_execution::telemetry_span!(backend("assembly_finalization", "fixed-domain-mini"))
            .entered();
    let [linear_system]: [eqiora_assembly::LinearSystem; 1] =
        systems.try_into().map_err(|systems: Vec<_>| {
            invalid(format!(
                "one-target transient MINI assembly returned {} systems",
                systems.len()
            ))
        })?;
    let canonical = CanonicalCsrSystemView::new(&linear_system, LinearOperatorProperties::General)?;
    let relation = AssembledLinearizedRelation::from_canonical(
        canonical,
        candidate.to_vec(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )?;
    let pressure_reference = if step.with_gauge() {
        SimplicialMiniStokesPressureReference2d::ZeroIntegral {
            multiplier: step
                .gauge_multiplier
                .expect("gauged layout reconstructs a multiplier"),
        }
    } else {
        SimplicialMiniStokesPressureReference2d::BoundaryTraction
    };
    let evaluated_packets = evaluated
        .into_iter()
        .enumerate()
        .map(|(packet, evaluated)| {
            evaluated.into_inner().ok_or_else(|| {
                invalid(format!(
                    "assembly backend omitted prepared transient packet {packet}"
                ))
            })?
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StepAssembly {
        relation,
        residual,
        layout: Arc::clone(&step.structure.layout),
        velocity: step.velocity,
        pressure: step.pressure,
        pressure_reference,
        gauge_multiplier: step.gauge_multiplier,
        named_reaction_vertices: Arc::clone(&step.structure.named_reaction_vertices),
        assembly_report,
        evaluated_packets,
        full_maps: Arc::clone(&step.structure.full_maps),
    })
}

fn evaluate_linear_residual(
    local: &LocalContribution,
    map: &AssemblyMap,
    global_point: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    let local_point = mapped_local_point(map, global_point)?;
    evaluate_local_residual(local, &local_point)
}

fn evaluate_local_residual(
    local: &LocalContribution,
    local_point: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    if local_point.len() != local.columns() {
        return Err(invalid(
            "local residual point does not match the prepared contribution shape",
        ));
    }
    Ok(local
        .matrix()
        .chunks_exact(local.columns())
        .zip(local.rhs())
        .map(|(row, rhs)| {
            row.iter()
                .zip(local_point)
                .map(|(entry, value)| entry * value)
                .sum::<f64>()
                - rhs
        })
        .collect())
}

fn mapped_local_point(map: &AssemblyMap, global_point: &[f64]) -> Result<Vec<f64>, Diagnostic> {
    map.unknowns()
        .iter()
        .map(|unknown| match unknown {
            LocalUnknown::Free(dof) => global_point.get(dof.index()).copied().ok_or_else(|| {
                invalid("local residual map references an unknown outside the candidate point")
            }),
            LocalUnknown::Fixed(value) => Ok(*value),
        })
        .collect()
}

fn scatter_residual(
    output: &mut [f64],
    map: &AssemblyMap,
    local_residual: &[f64],
) -> Result<(), Diagnostic> {
    if map.equations().len() != local_residual.len() {
        return Err(invalid(
            "direct transient residual shape differs from its assembly map",
        ));
    }
    for (equation, value) in map.equations().iter().zip(local_residual) {
        if let Some(equation) = equation {
            let destination = output.get_mut(equation.index()).ok_or_else(|| {
                invalid("direct transient residual equation is outside its target")
            })?;
            *destination += value;
        }
    }
    Ok(())
}

fn require_same_mesh(
    mesh: &SimplicialMesh,
    previous: &SimplicialMiniNavierStokesState2d,
) -> Result<(), Diagnostic> {
    if previous.velocity().mesh() != mesh || previous.pressure().mesh() != mesh {
        return Err(invalid(
            "MINI Navier--Stokes fixed-domain step rejects stale or moving mesh state",
        ));
    }
    Ok(())
}

fn require_pressure_policy(
    previous: &SimplicialMiniNavierStokesState2d,
    with_gauge: bool,
) -> Result<(), Diagnostic> {
    let expected_gauge = matches!(
        previous.pressure_reference(),
        SimplicialMiniStokesPressureReference2d::ZeroIntegral { .. }
    );
    if expected_gauge != with_gauge {
        return Err(invalid(
            "MINI Navier--Stokes pressure closure differs from the shaped initial state",
        ));
    }
    Ok(())
}
