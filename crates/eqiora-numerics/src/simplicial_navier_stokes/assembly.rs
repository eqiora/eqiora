use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPlan, AssemblyReport, AssemblyTarget,
    LocalContribution, LocalUnknown, TargetAssemblyMap,
};
use eqiora_core::Diagnostic;
use eqiora_meshing::{
    AffineGeometryMap, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, SimplicialMesh,
};
use eqiora_solver::{CanonicalCsrSystemView, LinearOperatorProperties};
use std::collections::BTreeMap;
use std::sync::Arc;

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
use super::layout::FlowLayout;
use super::{COMPONENTS, DIMENSION, invalid};
use crate::assembled_linearization::AssembledLinearizedRelation;
use crate::form_compiler::integral_constraint::zero_integral_constraint;
use crate::jacobian_audit::{StructuralJacobianPattern, StructuralJacobianPatternBuilder};
use crate::region_assembly::nonlinear::{
    EvaluatedRegionPacket, RegionAssemblyLinearization, assemble_linearization,
    evaluate_linear_residual, evaluate_local_residual, mapped_local_point, scatter_residual,
};
use crate::simplicial_elliptic::SimplicialP1Field;
use crate::simplicial_stokes::boundary::{PreparedBoundary2d, PressureReferenceKind2d};
use crate::simplicial_stokes::{CELL_LOCAL_DOF_COUNT, CONSTRAINT_LOCAL_DOF_COUNT};
use crate::simplicial_stokes::{
    SimplicialMiniStokesBoundary2d, SimplicialMiniStokesPressureReference2d,
    SimplicialMiniVelocityField2d,
};

pub(super) struct StepAssembly {
    pub(super) relation: AssembledLinearizedRelation,
    pub(super) residual: Vec<f64>,
    pub(super) layout: Arc<FlowLayout>,
    pub(super) velocity: SimplicialMiniVelocityField2d,
    pub(super) pressure: SimplicialP1Field,
    pub(super) pressure_reference: SimplicialMiniStokesPressureReference2d,
    pub(super) gauge_multiplier: Option<f64>,
    pub(super) named_reaction_vertices: Arc<Vec<(String, Vec<usize>)>>,
    pub(super) assembly_report: AssemblyReport,
    evaluated_packets: Vec<EvaluatedRegionPacket>,
    full_maps: Arc<Vec<Arc<AssemblyMap<f64>>>>,
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
        self.layout
            .velocity_rows()
            .iter()
            .map(|dof| self.residual[dof.index()].powi(2))
            .sum::<f64>()
            .sqrt()
    }

    pub(super) fn materialize_acceptance_data(&self) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        let mut full_residual = vec![0.0; self.layout.full_size()];
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
                    let Some(pressure) = self.layout.pressure_vertex(equation.index()) else {
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

pub(crate) struct PreparedStepStructure {
    mesh: SimplicialMesh,
    form: Arc<super::form::StepForm>,
    bound: Option<Vec<crate::form_compiler::region::PreparedRegionCell<f64>>>,
    boundary: PreparedBoundary2d,
    layout: Arc<FlowLayout>,
    named_reaction_vertices: Arc<Vec<(String, Vec<usize>)>>,
    packet_mappings: Vec<Arc<[TargetAssemblyMap<f64>]>>,
    full_maps: Arc<Vec<Arc<AssemblyMap<f64>>>>,
    local_sizes: Vec<usize>,
    reduced_assembly_plan: AssemblyPlan,
    cell_geometries: Vec<AffineGeometryMap>,
    cell_quadrature: QuadratureRule,
    constraint_contributions: Vec<LocalContribution<f64>>,
    facet_geometries: Vec<AffineGeometryMap>,
    facet_incidence: Vec<eqiora_meshing::EntityIncidence>,
    facet_parent_vertices: Vec<Vec<usize>>,
    facet_quadrature: QuadratureRule,
    cell_count: usize,
    constraint_end: usize,
    packet_count: usize,
}

impl PreparedStepStructure {
    pub(crate) fn require_quadrature(
        &self,
        cell: &QuadratureRule,
        facet: &QuadratureRule,
    ) -> Result<(), Diagnostic> {
        if cell != &self.cell_quadrature || facet != &self.facet_quadrature {
            return Err(invalid(
                "prepared step quadrature differs from its exact numerical binding",
            ));
        }
        Ok(())
    }

    pub(crate) fn bind<F>(mut self, load: &F) -> Result<Self, Diagnostic>
    where
        F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    {
        if self.bound.is_some() {
            return Err(invalid("step structure already has an exact form binding"));
        }
        let cells = self
            .cell_geometries
            .iter()
            .map(|geometry| {
                self.form
                    .prepare_cell(geometry, &self.cell_quadrature, load)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.bound = Some(cells);
        Ok(self)
    }

    fn cells(
        &self,
        plan: &MiniNavierStokesStepPlan2d,
    ) -> Result<&[crate::form_compiler::region::PreparedRegionCell<f64>], Diagnostic> {
        let Some(cells) = &self.bound else {
            return Err(invalid(
                "step assembly requires its exact prepared form binding",
            ));
        };
        if !Arc::ptr_eq(&self.form, &plan.form) && self.form != plan.form {
            return Err(invalid(
                "prepared step form differs from the requested numerical policy",
            ));
        }
        Ok(cells)
    }
}

struct PreparedStepPoint<'a> {
    history: Vec<BTreeMap<eqiora_core::RawId, Vec<f64>>>,
    structure: &'a PreparedStepStructure,
    velocity: SimplicialMiniVelocityField2d,
    pressure: SimplicialP1Field,
    gauge_multiplier: Option<f64>,
}

impl PreparedStepPoint<'_> {
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
    plan: &MiniNavierStokesStepPlan2d,
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
        plan,
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
    plan: &MiniNavierStokesStepPlan2d,
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
    let layout = Arc::new(FlowLayout::new(
        mesh,
        &boundary.fixed_velocity,
        with_gauge,
        &plan.form,
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
            .map(|geometry| {
                crate::simplicial_stokes::acceptance::require_local_geometry(
                    geometry,
                    cell_quadrature,
                )?;
                zero_integral_constraint(
                    eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                    geometry,
                    cell_quadrature,
                )
            })
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
            (
                CELL_LOCAL_DOF_COUNT,
                layout.cell_map(packet, true)?,
                layout.cell_map(packet, false)?,
            )
        } else if packet < constraint_end {
            (
                CONSTRAINT_LOCAL_DOF_COUNT,
                layout.constraint_map(packet - cell_count, true)?,
                layout.constraint_map(packet - cell_count, false)?,
            )
        } else {
            let parent = facet_incidence[packet - constraint_end].entity;
            (
                CELL_LOCAL_DOF_COUNT,
                layout.cell_map(parent.index(), true)?,
                layout.cell_map(parent.index(), false)?,
            )
        };
        local_sizes.push(local_size);
        reduced_maps.push(Arc::new(reduced));
        full_maps.push(Arc::new(full));
    }
    let reduced_assembly_plan =
        AssemblyPlan::new(vec![AssemblyTarget::new(layout.reduced_size())?])?;
    let reduced_target = reduced_assembly_plan
        .target_id(0)
        .expect("one-target plan owns its target");
    let packet_mappings = reduced_maps
        .into_iter()
        .map(|map| Arc::<[_]>::from([TargetAssemblyMap::new(reduced_target, map)]))
        .collect::<Vec<_>>();
    let reduced_assembly_plan = reduced_assembly_plan.prepare(packet_mappings.clone())?;
    Ok(PreparedStepStructure {
        mesh: mesh.clone(),
        form: Arc::clone(&plan.form),
        bound: None,
        boundary,
        layout,
        named_reaction_vertices,
        packet_mappings,
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
    derivative: bool,
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
        derivative,
    )
}

fn prepare_step_point<'a>(
    mesh: &SimplicialMesh,
    structure: &'a PreparedStepStructure,
    previous: &SimplicialMiniNavierStokesState2d,
    candidate: &[f64],
) -> Result<PreparedStepPoint<'a>, Diagnostic> {
    if mesh != &structure.mesh {
        return Err(invalid(
            "prepared step structure belongs to another exact mesh",
        ));
    }
    require_same_mesh(mesh, previous)?;
    let with_gauge = structure.boundary.pressure_reference == PressureReferenceKind2d::ZeroIntegral;
    require_pressure_policy(previous, with_gauge)?;
    if candidate.len() != structure.layout.reduced_size()
        || candidate.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "MINI Navier--Stokes candidate must be finite and match the exact mixed layout",
        ));
    }
    let (vertex_values, bubble_values, pressure_values, gauge_multiplier) =
        structure.layout.reconstruct(candidate)?;
    let fields = structure.layout.physical_fields(
        &structure.form,
        previous.velocity().vertex_values(),
        previous.velocity().cell_bubble_values(),
        previous.pressure().vertex_values(),
    )?;
    let history = (0..structure.cell_count)
        .map(|cell| {
            structure
                .layout
                .cell_history(cell, &structure.form, &fields)
        })
        .collect::<Result<_, _>>()?;
    Ok(PreparedStepPoint {
        history,
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
    plan: &MiniNavierStokesStepPlan2d,
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
        plan,
    )?;
    build_step_jacobian_pattern_prepared(&prepared)
}

pub(crate) fn build_step_jacobian_pattern_prepared(
    prepared: &PreparedStepStructure,
) -> Result<StructuralJacobianPattern, Diagnostic> {
    let mut pattern = StructuralJacobianPatternBuilder::new(
        prepared.layout.reduced_size(),
        prepared.layout.reduced_size(),
        prepared.packet_count,
    )?;
    for packet in 0..prepared.packet_count {
        pattern.include_dense_local(
            packet,
            prepared.local_sizes[packet],
            prepared.packet_mappings[packet][0].map(),
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
        &plan,
    )?
    .bind(body_force)?;
    assemble_step_residual_prepared(mesh, &prepared, previous, candidate, plan)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_step_residual_prepared(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    previous: &SimplicialMiniNavierStokesState2d,
    candidate: &[f64],
    plan: MiniNavierStokesStepPlan2d,
) -> Result<Vec<f64>, Diagnostic> {
    let forms = prepared.cells(&plan)?;
    let step = prepare_step_point(mesh, prepared, previous, candidate)?;
    let mut residual = vec![0.0; step.structure.layout.reduced_size()];
    for (packet, mappings) in step.structure.packet_mappings.iter().enumerate() {
        let map = mappings[0].map();
        let local_residual = if packet < step.structure.cell_count {
            let local_point = mapped_local_point(map, candidate)?;
            forms[packet].residual(&step.history[packet], &local_point)?
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
                false,
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
    assembly: &dyn AssemblyBackend<f64>,
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
        &plan,
    )?
    .bind(body_force)?;
    assemble_step_linearization_prepared(mesh, &prepared, previous, candidate, plan, assembly)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_step_linearization_prepared(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    previous: &SimplicialMiniNavierStokesState2d,
    candidate: &[f64],
    plan: MiniNavierStokesStepPlan2d,
    assembly: &dyn AssemblyBackend<f64>,
) -> Result<StepAssembly, Diagnostic> {
    let forms = prepared.cells(&plan)?;
    let step = prepare_step_point(mesh, prepared, previous, candidate)?;
    let evaluate_packet = |packet: usize| {
        let _evaluation = eqiora_execution::telemetry_span!(backend(
            "assembly_local_evaluation",
            "fixed-domain-mini"
        ))
        .entered();
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
            let point =
                mapped_local_point(step.structure.packet_mappings[packet][0].map(), candidate)?;
            let linearization = forms[packet].linearize(&step.history[packet], &point)?;
            let residual = linearization.residual.clone();
            let local = linearization.into_contribution(&point)?;
            Ok(EvaluatedRegionPacket {
                assembly: AssemblyPacket::new(
                    local,
                    Arc::clone(&step.structure.packet_mappings[packet]),
                )?,
                residual,
            })
        } else if packet < step.structure.constraint_end {
            let cell_index = packet - step.structure.cell_count;
            let local = step.structure.constraint_contributions[cell_index].clone();
            let reduced = step.structure.packet_mappings[packet][0].map();
            let residual = evaluate_linear_residual(&local, reduced, candidate)?;
            Ok(EvaluatedRegionPacket {
                assembly: AssemblyPacket::new(
                    local,
                    Arc::clone(&step.structure.packet_mappings[packet]),
                )?,
                residual,
            })
        } else {
            let reduced = step.structure.packet_mappings[packet][0].map();
            let point = mapped_local_point(reduced, candidate)?;
            let action = natural_facet_action(
                &step,
                packet - step.structure.constraint_end,
                &plan,
                &point,
                true,
            )?;
            let residual = action.residual.clone();
            let local = action.into_contribution(&point)?;
            Ok(EvaluatedRegionPacket {
                assembly: AssemblyPacket::new(
                    local,
                    Arc::clone(&step.structure.packet_mappings[packet]),
                )?,
                residual,
            })
        }
    };
    let RegionAssemblyLinearization {
        systems,
        mut residuals,
        report: assembly_report,
        packets: evaluated_packets,
    } = {
        let _scatter = eqiora_execution::telemetry_span!(backend(
            "assembly_scatter_update",
            "fixed-domain-mini"
        ))
        .entered();
        assemble_linearization(
            &step.structure.reduced_assembly_plan,
            eqiora_assembly::AssemblyPacketSetIdentityV1::Unbound,
            step.structure.packet_count,
            assembly,
            evaluate_packet,
        )?
    };
    let residual = residuals.remove(0);
    let _finalization =
        eqiora_execution::telemetry_span!(backend("assembly_finalization", "fixed-domain-mini"))
            .entered();
    let [linear_system]: [eqiora_assembly::LinearSystem<f64>; 1] =
        systems.try_into().map_err(|systems: Vec<_>| {
            invalid(format!(
                "one-target transient MINI assembly returned {} systems",
                systems.len()
            ))
        })?;
    let canonical = CanonicalCsrSystemView::new(
        &linear_system,
        linear_system.rhs(),
        LinearOperatorProperties::General,
    )?;
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
