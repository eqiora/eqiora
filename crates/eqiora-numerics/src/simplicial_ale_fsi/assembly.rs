//! Monolithic reduced residual and exact ALE state Jacobian assembly.
//!
//! The algebraic candidate uses the unchanged dimensionless FSI quotient
//! layout.  Solid displacement and current geometry are not independent
//! unknowns: backward Euler derives the former from shared solid velocity and
//! the sealed harmonic action derives the latter.  Every Jacobian column
//! follows that same composition analytically.

mod fields;
use fields::{
    fluid_local_size, fluid_row_scales, local_pressure_coefficients, local_velocity_coefficients,
    solid_local_size,
};

use std::sync::Arc;

use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPlan, AssemblyReport, AssemblyTarget,
    DofId, IndexedAssemblyWork, LocalContribution, LocalUnknown, TargetAssemblyMap,
};
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::FixedTopologyGeometryAction;
use eqiora_meshing::{
    CellId, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, SimplicialMesh,
};
use eqiora_solver::{CanonicalCsrSystemView, LinearOperatorProperties};

use super::boundary_step::{AlgebraicDirection, PreparedAleFsiBoundaryStep};
use super::contract::{AleFsiBoundary, AleFsiState, AleFsiStepPlan};
use super::element::{AleMiniFluidCell, AleMiniFluidDirection};
use super::{P1HarmonicMeshMotionAction, invalid};
use crate::assembled_linearization::AssembledLinearizedRelation;
use crate::jacobian_audit::{StructuralJacobianPattern, StructuralJacobianPatternBuilder};
use crate::simplicial_fsi::{FixedReferenceFsiPartition, FixedReferenceFsiState};
use crate::simplicial_fsi::{element::solid_local, layout::FsiLayout};

/// One assembled Newton point and the independently evaluated physical split.
pub(super) struct StepAssembly<const D: usize> {
    pub(super) relation: AssembledLinearizedRelation,
    pub(super) current: AleFsiState<D>,
    pub(super) geometry_action: FixedTopologyGeometryAction<D>,
    pub(super) residual: Vec<f64>,
    pub(super) full_fluid_residual: Vec<f64>,
    pub(super) full_solid_residual: Vec<f64>,
    pub(super) layout: Arc<FsiLayout<D>>,
    pub(super) assembly_report: AssemblyReport,
}

struct PreparedAleFsiCell {
    vertices: Vec<MeshEntity>,
    reduced_map: AssemblyMap,
    full_map: AssemblyMap,
    dense_map: AssemblyMap,
    bubble_cell: Option<CellId>,
}

/// Immutable layout and assembly structure shared by every action in one Run.
pub(super) struct PreparedAleFsiStructure<const D: usize> {
    boundary_template: PreparedAleFsiBoundaryStep<D>,
    layout: Arc<FsiLayout<D>>,
    directions: Vec<AlgebraicDirection<D>>,
    assembly_plan: AssemblyPlan,
    reduced_target: eqiora_assembly::AssemblyTargetId,
    cells: Vec<PreparedAleFsiCell>,
    #[cfg(test)]
    phases: AleFsiStructuralPhaseCounts,
}

#[cfg(test)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct AleFsiStructuralPhaseCounts {
    pub(super) authentication: usize,
    pub(super) layout: usize,
    pub(super) maps: usize,
    pub(super) quadrature: usize,
    pub(super) sparsity: usize,
}

/// Action-local endpoint binding over one immutable prepared structure.
pub(super) struct PreparedAleFsiAction<const D: usize> {
    boundary: PreparedAleFsiBoundaryStep<D>,
    previous_reference: FixedReferenceFsiState<D>,
}

impl<const D: usize> StepAssembly<D> {
    pub(super) fn residual_norm(&self) -> Result<f64, Diagnostic> {
        finite_norm(&self.residual, "ALE FSI reduced residual")
    }

    pub(super) fn residual(&self) -> &[f64] {
        &self.residual
    }

    pub(super) fn algebraic_values(&self) -> &[f64] {
        self.relation.accepted_unknowns()
    }

    pub(super) const fn current_state(&self) -> &AleFsiState<D> {
        &self.current
    }

    pub(super) const fn geometry_action(&self) -> &FixedTopologyGeometryAction<D> {
        &self.geometry_action
    }

    pub(super) fn full_fluid_residual(&self) -> &[f64] {
        &self.full_fluid_residual
    }

    pub(super) fn full_solid_residual(&self) -> &[f64] {
        &self.full_solid_residual
    }

    pub(super) const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_ale_fsi_structure<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &PreparedAleFsiBoundaryStep<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    initial: &AleFsiState<D>,
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    base_layout: &FsiLayout<D>,
) -> Result<PreparedAleFsiStructure<D>, Diagnostic> {
    #[cfg(test)]
    let mut phases = AleFsiStructuralPhaseCounts::default();
    boundary.validate_inputs(reference, partition, motion, initial, plan, quadrature)?;
    base_layout.require_reference(reference, partition)?;
    base_layout.require_scale(plan.scale())?;
    #[cfg(test)]
    {
        phases.authentication += 1;
        phases.quadrature += 1;
    }
    let layout = Arc::new(boundary.layout(base_layout)?);
    #[cfg(test)]
    {
        phases.layout += 1;
    }
    let directions = boundary.build_directions(partition, motion, plan, &layout)?;
    let assembly_plan = AssemblyPlan::new(vec![AssemblyTarget::new(layout.reduced_size())?])?;
    #[cfg(test)]
    {
        phases.sparsity += 1;
    }
    let reduced_target = assembly_plan
        .target_id(0)
        .expect("one-target ALE FSI plan owns its reduced target");
    let cell_count = reference.entity_count(D).ok_or_else(|| {
        invalid(format!(
            "ALE FSI reference mesh omits its {D}D cell stratum"
        ))
    })?;
    if cell_count != partition.cell_count() {
        return Err(invalid(
            "ALE FSI material partition differs from the reference cell inventory",
        ));
    }
    let mut cells = Vec::with_capacity(cell_count);
    for cell_index in 0..cell_count {
        let vertices = reference
            .entity_vertices(MeshEntity::new(D, cell_index))
            .ok_or_else(|| {
                invalid(format!(
                    "ALE FSI cell packet {cell_index} has no reference vertex closure"
                ))
            })?;
        require_simplex_closure::<D>(&vertices, reference.vertices().len())?;
        let (reduced_map, full_map, bubble_cell) = match layout.cell_domain(cell_index)? {
            domain if layout.pressure_field(domain).is_some() => {
                let cell = CellId::new(cell_index);
                (
                    layout.fluid_map(cell, &vertices, true)?,
                    layout.fluid_map(cell, &vertices, false)?,
                    Some(cell),
                )
            }
            domain if layout.state_field(domain).is_some() => (
                layout.solid_map(cell_index, &vertices, true)?,
                layout.solid_map(cell_index, &vertices, false)?,
                None,
            ),
            _ => {
                return Err(invalid(format!(
                    "ALE FSI cell packet {cell_index} has no material assignment"
                )));
            }
        };
        let dense_map = AssemblyMap::new(
            reduced_map.equations().to_vec(),
            (0..layout.reduced_size())
                .map(|index| LocalUnknown::Free(DofId::new(index)))
                .collect(),
        )?;
        cells.push(PreparedAleFsiCell {
            vertices,
            reduced_map,
            full_map,
            dense_map,
            bubble_cell,
        });
    }
    #[cfg(test)]
    {
        phases.maps += 1;
    }
    Ok(PreparedAleFsiStructure {
        boundary_template: boundary.clone(),
        layout,
        directions,
        assembly_plan,
        reduced_target,
        cells,
        #[cfg(test)]
        phases,
    })
}

impl<const D: usize> PreparedAleFsiStructure<D> {
    #[cfg(test)]
    pub(super) const fn phase_counts(&self) -> AleFsiStructuralPhaseCounts {
        self.phases
    }
    pub(super) fn prepare_action(
        &self,
        _reference: &SimplicialMesh,
        _partition: &FixedReferenceFsiPartition<D>,
        boundary: PreparedAleFsiBoundaryStep<D>,
        previous: &AleFsiState<D>,
        plan: &AleFsiStepPlan<D>,
    ) -> Result<PreparedAleFsiAction<D>, Diagnostic> {
        if !self.boundary_template.has_same_structure(&boundary) {
            return Err(invalid(
                "ALE FSI action boundary differs from its prepared Run structure",
            ));
        }
        boundary.validate_action(previous, plan)?;
        let previous_reference = previous.physical_state().clone();
        Ok(PreparedAleFsiAction {
            boundary,
            previous_reference,
        })
    }

    pub(super) fn initial_point(
        &self,
        action: &PreparedAleFsiAction<D>,
        previous: &AleFsiState<D>,
        plan: &AleFsiStepPlan<D>,
    ) -> Result<Vec<f64>, Diagnostic> {
        action
            .boundary
            .reduce_initial_point(previous, plan, &self.layout)
    }
}

/// Map one accepted state to the unchanged dimensionless quotient coordinates.
#[allow(dead_code)]
pub(super) fn initial_point<const D: usize>(
    reference: &SimplicialMesh,
    boundary: &AleFsiBoundary<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    base_layout: &FsiLayout<D>,
) -> Result<Vec<f64>, Diagnostic> {
    let prepared = match PreparedAleFsiBoundaryStep::from_boundary(boundary) {
        Some(prepared) => prepared,
        None => PreparedAleFsiBoundaryStep::homogeneous(
            reference,
            boundary,
            previous.time(),
            previous.time() + plan.time_step(),
            plan.scale().velocity(),
        )?,
    };
    initial_point_prepared(
        reference,
        &prepared,
        motion,
        previous,
        plan,
        quadrature,
        base_layout,
    )
}

pub(super) fn initial_point_prepared<const D: usize>(
    reference: &SimplicialMesh,
    boundary: &PreparedAleFsiBoundaryStep<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    base_layout: &FsiLayout<D>,
) -> Result<Vec<f64>, Diagnostic> {
    let partition = base_layout.partition();
    base_layout.require_reference(reference, partition)?;
    base_layout.require_scale(plan.scale())?;
    boundary.validate_inputs(reference, partition, motion, previous, plan, quadrature)?;
    let layout = boundary.layout(base_layout)?;
    boundary.reduce_initial_point(previous, plan, &layout)
}

/// Assemble the exact reduced Newton action at one dimensionless candidate.
///
/// Each cell packet owns its physical residual rows and a rectangular block
/// containing every reduced state column.  The assembly backend therefore
/// sees the same general sparse action used by the solver, while direct
/// residual assembly remains independent of `A x - b` reconstruction.
#[allow(dead_code, clippy::too_many_arguments)]
pub(super) fn assemble_step_linearization<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &AleFsiBoundary<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    candidate: &[f64],
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend,
    base_layout: &FsiLayout<D>,
) -> Result<StepAssembly<D>, Diagnostic> {
    let prepared_boundary = match PreparedAleFsiBoundaryStep::from_boundary(boundary) {
        Some(prepared) => prepared,
        None => PreparedAleFsiBoundaryStep::homogeneous(
            reference,
            boundary,
            previous.time(),
            previous.time() + plan.time_step(),
            plan.scale().velocity(),
        )?,
    };
    assemble_step_linearization_prepared(
        reference,
        partition,
        &prepared_boundary,
        motion,
        previous,
        candidate,
        plan,
        quadrature,
        assembly,
        base_layout,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_step_linearization_prepared<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &PreparedAleFsiBoundaryStep<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    candidate: &[f64],
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend,
    base_layout: &FsiLayout<D>,
) -> Result<StepAssembly<D>, Diagnostic> {
    let structure = prepare_ale_fsi_structure(
        reference,
        partition,
        boundary,
        motion,
        previous,
        plan,
        quadrature,
        base_layout,
    )?;
    let action =
        structure.prepare_action(reference, partition, boundary.clone(), previous, plan)?;
    assemble_step_linearization_with_structure(
        reference, partition, &structure, &action, motion, previous, candidate, plan, quadrature,
        assembly,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_step_linearization_with_structure<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    structure: &PreparedAleFsiStructure<D>,
    action: &PreparedAleFsiAction<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    candidate: &[f64],
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend,
) -> Result<StepAssembly<D>, Diagnostic> {
    let prepared = prepare_step(
        reference, partition, structure, action, motion, previous, plan, candidate,
    )?;
    let evaluate = |cell_index| {
        evaluate_cell(
            &structure.layout,
            cell_index,
            reference,
            partition,
            quadrature,
            previous,
            &action.previous_reference,
            &prepared.current,
            &prepared.geometry_action,
            plan,
            candidate,
            &structure.directions,
            structure
                .cells
                .get(cell_index)
                .ok_or_else(|| invalid("ALE FSI cell is outside prepared Run structure"))?,
            structure.reduced_target,
        )
    };
    let work = IndexedAssemblyWork::new(structure.cells.len(), |cell_index| {
        evaluate(cell_index).map(|evaluated| evaluated.packet)
    });
    let (systems, assembly_report) = assembly
        .assemble(&structure.assembly_plan, &work)?
        .into_parts();
    if assembly_report.packet_count() != structure.cells.len()
        || assembly_report.target_count() != 1
    {
        return Err(invalid(
            "ALE FSI assembly evidence differs from its exact cell and target inventory",
        ));
    }

    let direct = assemble_direct_residuals(
        reference, partition, structure, action, previous, candidate, plan, quadrature, &prepared,
    )?;

    let [linear_system]: [eqiora_assembly::LinearSystem; 1] =
        systems.try_into().map_err(|systems: Vec<_>| {
            invalid(format!(
                "one-target ALE FSI assembly returned {} systems",
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
    let mut reconstructed = zeroed(direct.reduced.len(), "captured relation residual")?;
    eqiora_ir::LinearizedRelation::primal(&relation, &mut reconstructed)?;
    require_same_residual(&direct.reduced, &reconstructed)?;

    Ok(StepAssembly {
        relation,
        current: prepared.current,
        geometry_action: prepared.geometry_action,
        residual: direct.reduced,
        full_fluid_residual: direct.full_fluid,
        full_solid_residual: direct.full_solid,
        layout: Arc::clone(&structure.layout),
        assembly_report,
    })
}

/// Reassemble only the direct reduced residual at one dimensionless point.
///
/// This is the finite-difference path used to verify the analytic ALE state
/// Jacobian. It deliberately performs the identical validation, state and
/// geometry reconstruction, cell traversal, local primal evaluation, and
/// scatter as assemble_step_linearization, without constructing dense
/// directions, a Jacobian, assembly packets, or an assembly backend request.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(super) fn assemble_step_residual<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &AleFsiBoundary<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    candidate: &[f64],
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    base_layout: &FsiLayout<D>,
) -> Result<Vec<f64>, Diagnostic> {
    let prepared_boundary = match PreparedAleFsiBoundaryStep::from_boundary(boundary) {
        Some(prepared) => prepared,
        None => PreparedAleFsiBoundaryStep::homogeneous(
            reference,
            boundary,
            previous.time(),
            previous.time() + plan.time_step(),
            plan.scale().velocity(),
        )?,
    };
    assemble_step_residual_prepared(
        reference,
        partition,
        &prepared_boundary,
        motion,
        previous,
        candidate,
        plan,
        quadrature,
        base_layout,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_step_residual_prepared<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &PreparedAleFsiBoundaryStep<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    candidate: &[f64],
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    base_layout: &FsiLayout<D>,
) -> Result<Vec<f64>, Diagnostic> {
    let structure = prepare_ale_fsi_structure(
        reference,
        partition,
        boundary,
        motion,
        previous,
        plan,
        quadrature,
        base_layout,
    )?;
    let action =
        structure.prepare_action(reference, partition, boundary.clone(), previous, plan)?;
    let prepared = prepare_step(
        reference, partition, &structure, &action, motion, previous, plan, candidate,
    )?;
    Ok(assemble_direct_residuals(
        reference, partition, &structure, &action, previous, candidate, plan, quadrature, &prepared,
    )?
    .reduced)
}

struct PreparedStep<const D: usize> {
    current: AleFsiState<D>,
    geometry_action: FixedTopologyGeometryAction<D>,
}

#[allow(clippy::too_many_arguments)]
fn prepare_step<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    structure: &PreparedAleFsiStructure<D>,
    action: &PreparedAleFsiAction<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    plan: &AleFsiStepPlan<D>,
    candidate: &[f64],
) -> Result<PreparedStep<D>, Diagnostic> {
    if candidate.len() != structure.layout.reduced_size()
        || candidate.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "ALE FSI candidate must be finite and match the exact reduced quotient layout",
        ));
    }
    let current = action.boundary.reconstruct_current_state(
        reference,
        partition,
        motion,
        previous,
        candidate,
        plan,
        &structure.layout,
    )?;
    let geometry_action = plan.geometry_action(reference, partition, motion, previous, &current)?;
    Ok(PreparedStep {
        current,
        geometry_action,
    })
}

struct DirectResiduals {
    reduced: Vec<f64>,
    full_fluid: Vec<f64>,
    full_solid: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn assemble_direct_residuals<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    structure: &PreparedAleFsiStructure<D>,
    action: &PreparedAleFsiAction<D>,
    previous: &AleFsiState<D>,
    candidate: &[f64],
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    prepared: &PreparedStep<D>,
) -> Result<DirectResiduals, Diagnostic> {
    let mut reduced = zeroed(structure.layout.reduced_size(), "reduced residual")?;
    let mut full_fluid = zeroed(structure.layout.full_size(), "full fluid residual")?;
    let mut full_solid = zeroed(structure.layout.full_size(), "full solid residual")?;
    for (cell_index, cell) in structure.cells.iter().enumerate() {
        let evaluated = evaluate_cell_residual(
            &structure.layout,
            cell_index,
            reference,
            partition,
            quadrature,
            previous,
            &action.previous_reference,
            &prepared.current,
            &prepared.geometry_action,
            plan,
            candidate,
            cell,
        )?;
        scatter_residual(
            &mut reduced,
            evaluated.reduced_map.equations(),
            &evaluated.residual,
        )?;
        let full = match evaluated.domain {
            domain if structure.layout.pressure_field(domain).is_some() => &mut full_fluid,
            domain if structure.layout.state_field(domain).is_some() => &mut full_solid,
            _ => {
                return Err(invalid(format!(
                    "ALE FSI cell {cell_index} has no material assignment"
                )));
            }
        };
        scatter_residual(full, evaluated.full_map.equations(), &evaluated.residual)?;
    }
    if reduced
        .iter()
        .chain(&full_fluid)
        .chain(&full_solid)
        .any(|value| !value.is_finite())
    {
        return Err(invalid(
            "direct ALE FSI residual assembly produced a non-finite value",
        ));
    }
    Ok(DirectResiduals {
        reduced,
        full_fluid,
        full_solid,
    })
}

struct EvaluatedCell {
    packet: AssemblyPacket,
}

struct EvaluatedCellResidual {
    residual: Vec<f64>,
    reduced_map: AssemblyMap,
    full_map: AssemblyMap,
    domain: RawId,
    source: CellResidualSource,
}

enum CellResidualSource {
    Fluid { bubble_cell: CellId },
    Solid(LocalContribution),
}

#[allow(clippy::too_many_arguments)]
fn evaluate_cell<const D: usize>(
    layout: &FsiLayout<D>,
    cell_index: usize,
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    quadrature: &QuadratureRule,
    previous: &AleFsiState<D>,
    previous_reference: &FixedReferenceFsiState<D>,
    current: &AleFsiState<D>,
    geometry_action: &FixedTopologyGeometryAction<D>,
    plan: &AleFsiStepPlan<D>,
    candidate: &[f64],
    directions: &[AlgebraicDirection<D>],
    cell: &PreparedAleFsiCell,
    reduced_target: eqiora_assembly::AssemblyTargetId,
) -> Result<EvaluatedCell, Diagnostic> {
    let evaluated = evaluate_cell_residual(
        layout,
        cell_index,
        reference,
        partition,
        quadrature,
        previous,
        previous_reference,
        current,
        geometry_action,
        plan,
        candidate,
        cell,
    )?;
    let matrix = match &evaluated.source {
        CellResidualSource::Fluid { bubble_cell } => evaluate_fluid_jacobian(
            layout,
            cell_index,
            *bubble_cell,
            reference,
            partition,
            quadrature,
            previous,
            current,
            geometry_action,
            plan,
            candidate.len(),
            directions,
            &cell.vertices,
        )?,
        CellResidualSource::Solid(local) => {
            embed_solid_jacobian(local, &evaluated.reduced_map, candidate.len())?
        }
    };
    let rhs = affine_rhs_from_residual(&matrix, candidate, &evaluated.residual)?;
    Ok(EvaluatedCell {
        packet: AssemblyPacket::new(
            LocalContribution::new(evaluated.residual.len(), candidate.len(), matrix, rhs)?,
            vec![TargetAssemblyMap::new(
                reduced_target,
                cell.dense_map.clone(),
            )],
        )?,
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_cell_residual<const D: usize>(
    layout: &FsiLayout<D>,
    cell_index: usize,
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    quadrature: &QuadratureRule,
    previous: &AleFsiState<D>,
    previous_reference: &FixedReferenceFsiState<D>,
    current: &AleFsiState<D>,
    geometry_action: &FixedTopologyGeometryAction<D>,
    plan: &AleFsiStepPlan<D>,
    candidate: &[f64],
    cell: &PreparedAleFsiCell,
) -> Result<EvaluatedCellResidual, Diagnostic> {
    let domain = layout.cell_domain(cell_index)?;
    let (residual, reduced_map, full_map, source) = match domain {
        domain if layout.pressure_field(domain).is_some() => evaluate_fluid_residual(
            layout,
            cell_index,
            cell,
            partition,
            quadrature,
            previous,
            current,
            geometry_action,
            plan,
        )?,
        domain if layout.state_field(domain).is_some() => evaluate_solid_residual(
            layout,
            cell_index,
            MeshEntity::new(D, cell_index),
            cell,
            reference,
            quadrature,
            previous_reference,
            plan,
            candidate,
        )?,
        _ => {
            return Err(invalid(format!(
                "ALE FSI cell packet {cell_index} has no material assignment"
            )));
        }
    };
    if residual.len() != reduced_map.equations().len()
        || residual.len() != full_map.equations().len()
        || residual.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(format!(
            "{D}D ALE FSI cell residual differs from its exact equation closures"
        )));
    }
    Ok(EvaluatedCellResidual {
        residual,
        reduced_map,
        full_map,
        domain,
        source,
    })
}

struct PreparedFluidCell<'a, const D: usize> {
    velocity_field: eqiora_core::RawId,
    geometry: &'a eqiora_meshing::FixedTopologyCellGeometryAction<D>,
    previous_velocity: Vec<[f64; D]>,
    current_velocity: Vec<[f64; D]>,
    current_pressure: Vec<f64>,
}

impl<const D: usize> PreparedFluidCell<'_, D> {
    fn operator(&self, plan: &AleFsiStepPlan<D>) -> Result<AleMiniFluidCell<'_, D>, Diagnostic> {
        let field = self.velocity_field.downcast().expect("Field");
        Ok(AleMiniFluidCell::<D> {
            geometry: self.geometry,
            density: plan
                .material()
                .density(field)
                .ok_or_else(|| invalid("missing exact ALE density Field"))?,
            viscosity: plan
                .material()
                .viscosity(field)
                .ok_or_else(|| invalid("missing exact ALE viscosity Field"))?,
            time_step: plan.time_step(),
            previous_velocity: &self.previous_velocity,
            current_velocity: &self.current_velocity,
            current_pressure: &self.current_pressure,
        })
    }
}

fn prepare_fluid_cell<'a, const D: usize>(
    layout: &FsiLayout<D>,
    cell_index: usize,
    vertices: &[MeshEntity],
    _partition: &FixedReferenceFsiPartition<D>,
    previous: &AleFsiState<D>,
    current: &AleFsiState<D>,
    geometry_action: &'a FixedTopologyGeometryAction<D>,
) -> Result<(CellId, PreparedFluidCell<'a, D>), Diagnostic> {
    let bubble_cell = CellId::new(cell_index);
    let domain = layout.cell_domain(cell_index)?;
    let velocity_field = layout.velocity_field(domain)?;
    let pressure_field = layout
        .pressure_field(domain)
        .ok_or_else(|| invalid("ALE cell has no exact constraint witness"))?;
    let previous_velocity = local_velocity_coefficients(
        vertices,
        velocity_field,
        &previous.physical_state().fields[&velocity_field].coefficients,
        bubble_cell,
    )?;
    let current_velocity = local_velocity_coefficients(
        vertices,
        velocity_field,
        &current.physical_state().fields[&velocity_field].coefficients,
        bubble_cell,
    )?;
    let current_pressure = local_pressure_coefficients::<D>(
        vertices,
        pressure_field,
        &current.physical_state().fields[&pressure_field].coefficients,
    )?;
    let geometry = geometry_action.cell(cell_index).ok_or_else(|| {
        invalid(format!(
            "ALE FSI geometry action omits fluid cell {cell_index}"
        ))
    })?;
    Ok((
        bubble_cell,
        PreparedFluidCell {
            velocity_field,
            geometry,
            previous_velocity,
            current_velocity,
            current_pressure,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_fluid_residual<const D: usize>(
    layout: &FsiLayout<D>,
    cell_index: usize,
    cell: &PreparedAleFsiCell,
    partition: &FixedReferenceFsiPartition<D>,
    quadrature: &QuadratureRule,
    previous: &AleFsiState<D>,
    current: &AleFsiState<D>,
    geometry_action: &FixedTopologyGeometryAction<D>,
    plan: &AleFsiStepPlan<D>,
) -> Result<(Vec<f64>, AssemblyMap, AssemblyMap, CellResidualSource), Diagnostic> {
    let (bubble_cell, prepared) = prepare_fluid_cell(
        layout,
        cell_index,
        &cell.vertices,
        partition,
        previous,
        current,
        geometry_action,
    )?;
    if cell.bubble_cell != Some(bubble_cell) {
        return Err(invalid(
            "ALE FSI fluid cell identity changed after structural preparation",
        ));
    }
    let primal = prepared.operator(plan)?.residual(quadrature)?;
    let row_scales = fluid_row_scales(plan);
    if primal.len() != row_scales.len() {
        return Err(invalid(format!(
            "{D}D ALE FSI fluid residual differs from its typed row-scale inventory"
        )));
    }
    let residual = primal
        .iter()
        .zip(row_scales.iter().copied())
        .map(|(value, scale)| value * scale)
        .collect::<Vec<_>>();
    if residual.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "{D}D ALE FSI fluid residual is non-finite after typed row scaling"
        )));
    }
    Ok((
        residual,
        cell.reduced_map.clone(),
        cell.full_map.clone(),
        CellResidualSource::Fluid { bubble_cell },
    ))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_fluid_jacobian<const D: usize>(
    layout: &FsiLayout<D>,
    cell_index: usize,
    expected_bubble_cell: CellId,
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    quadrature: &QuadratureRule,
    previous: &AleFsiState<D>,
    current: &AleFsiState<D>,
    geometry_action: &FixedTopologyGeometryAction<D>,
    plan: &AleFsiStepPlan<D>,
    candidate_width: usize,
    directions: &[AlgebraicDirection<D>],
    vertices: &[MeshEntity],
) -> Result<Vec<f64>, Diagnostic> {
    if directions.len() != candidate_width {
        return Err(invalid(
            "ALE FSI analytic direction inventory differs from the candidate width",
        ));
    }
    let (bubble_cell, prepared) = prepare_fluid_cell(
        layout,
        cell_index,
        vertices,
        partition,
        previous,
        current,
        geometry_action,
    )?;
    if bubble_cell != expected_bubble_cell {
        return Err(invalid(
            "ALE FSI fluid cell identity changed between residual and Jacobian evaluation",
        ));
    }
    let cell_operator = prepared.operator(plan)?;
    let row_scales = fluid_row_scales(plan);
    let entry_count = fluid_local_size::<D>()
        .checked_mul(candidate_width)
        .ok_or_else(|| invalid("ALE FSI fluid cell Jacobian shape overflows usize"))?;
    let mut matrix = zeroed(entry_count, "fluid cell Jacobian")?;
    for (column, direction) in directions.iter().enumerate() {
        let velocity_direction = local_velocity_coefficients(
            vertices,
            prepared.velocity_field,
            &direction.physical,
            bubble_cell,
        )?;
        let pressure_field = layout
            .pressure_field(layout.cell_domain(cell_index)?)
            .ok_or_else(|| invalid("ALE Jacobian cell has no exact constraint witness"))?;
        let pressure_direction =
            local_pressure_coefficients::<D>(vertices, pressure_field, &direction.physical)?;
        if direction.coordinate.len() != reference.vertices().len() {
            return Err(invalid(
                "ALE FSI coordinate direction differs from the mesh vertex inventory",
            ));
        }
        let coordinate_direction = direction
            .coordinate
            .iter()
            .map(|value| value.to_vec())
            .collect::<Vec<_>>();
        let geometry_direction = geometry_action
            .current_mesh()
            .linearized_geometry_map(MeshEntity::new(D, cell_index), &coordinate_direction)?;
        let evaluated = cell_operator.evaluate(
            AleMiniFluidDirection::<D> {
                current_velocity: &velocity_direction,
                current_pressure: &pressure_direction,
                current_geometry: &geometry_direction,
            },
            quadrature,
        )?;
        if evaluated.jvp().len() != row_scales.len() {
            return Err(invalid(format!(
                "{D}D ALE FSI fluid JVP differs from its typed row-scale inventory"
            )));
        }
        for (row, (&value, scale)) in evaluated
            .jvp()
            .iter()
            .zip(row_scales.iter().copied())
            .enumerate()
        {
            matrix[row * candidate_width + column] = value * scale;
        }
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(invalid("ALE FSI fluid Jacobian is non-finite"));
    }
    Ok(matrix)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_solid_residual<const D: usize>(
    layout: &FsiLayout<D>,
    cell_index: usize,
    entity: MeshEntity,
    cell: &PreparedAleFsiCell,
    reference: &SimplicialMesh,
    quadrature: &QuadratureRule,
    previous: &FixedReferenceFsiState<D>,
    plan: &AleFsiStepPlan<D>,
    candidate: &[f64],
) -> Result<(Vec<f64>, AssemblyMap, AssemblyMap, CellResidualSource), Diagnostic> {
    let geometry = reference.geometry_map(entity).ok_or_else(|| {
        invalid(format!(
            "ALE FSI solid cell {cell_index} has no reference affine geometry"
        ))
    })?;
    let local = solid_local(
        &geometry,
        quadrature,
        plan.fixed_reference_config(),
        &cell.vertices,
        previous,
        layout
            .velocity_field(layout.cell_domain(cell_index)?)?
            .downcast()
            .expect("Field"),
        layout
            .state_field(layout.cell_domain(cell_index)?)
            .ok_or_else(|| invalid("ALE elastic cell has no exact state"))?
            .downcast()
            .expect("Field"),
    )?;
    let reduced_map = cell.reduced_map.clone();
    let local_point = local_point(&reduced_map, candidate)?;
    let residual = evaluate_affine_residual(&local, &local_point)?;
    let full_map = cell.full_map.clone();
    Ok((
        residual,
        reduced_map,
        full_map,
        CellResidualSource::Solid(local),
    ))
}

fn embed_solid_jacobian(
    local: &LocalContribution,
    reduced_map: &AssemblyMap,
    candidate_width: usize,
) -> Result<Vec<f64>, Diagnostic> {
    if reduced_map.unknowns().len() != local.columns() {
        return Err(invalid(
            "ALE FSI solid local unknown closure differs from its matrix columns",
        ));
    }
    let entry_count = local
        .rows()
        .checked_mul(candidate_width)
        .ok_or_else(|| invalid("ALE FSI solid cell Jacobian shape overflows usize"))?;
    let mut matrix = zeroed(entry_count, "solid cell Jacobian")?;
    for (local_column, unknown) in reduced_map.unknowns().iter().enumerate() {
        if let LocalUnknown::Free(dof) = unknown {
            if dof.index() >= candidate_width {
                return Err(invalid(
                    "ALE FSI solid local unknown lies outside the candidate width",
                ));
            }
            for row in 0..local.rows() {
                matrix[row * candidate_width + dof.index()] +=
                    local.matrix()[row * local.columns() + local_column];
            }
        }
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(invalid("ALE FSI solid Jacobian is non-finite"));
    }
    Ok(matrix)
}

pub(super) fn build_step_jacobian_pattern<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &AleFsiBoundary<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    base_layout: &FsiLayout<D>,
) -> Result<StructuralJacobianPattern, Diagnostic> {
    let layout = base_layout.with_boundary(boundary)?;
    build_structural_jacobian_pattern(reference, partition, motion, &layout)
}

#[allow(dead_code)]
pub(super) fn build_step_jacobian_pattern_prepared<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &PreparedAleFsiBoundaryStep<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    base_layout: &FsiLayout<D>,
) -> Result<StructuralJacobianPattern, Diagnostic> {
    let layout = boundary.layout(base_layout)?;
    build_structural_jacobian_pattern(reference, partition, motion, &layout)
}

fn build_structural_jacobian_pattern<const D: usize>(
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

fn local_point(map: &AssemblyMap, candidate: &[f64]) -> Result<Vec<f64>, Diagnostic> {
    map.unknowns()
        .iter()
        .map(|unknown| match unknown {
            LocalUnknown::Free(dof) => candidate.get(dof.index()).copied().ok_or_else(|| {
                invalid("ALE FSI local map references an unknown outside the candidate")
            }),
            LocalUnknown::Fixed(value) => Ok(*value),
        })
        .collect()
}

fn require_simplex_closure<const D: usize>(
    vertices: &[MeshEntity],
    vertex_count: usize,
) -> Result<(), Diagnostic> {
    if vertices.len() != D + 1
        || vertices
            .iter()
            .any(|vertex| vertex.dimension() != 0 || vertex.index() >= vertex_count)
    {
        return Err(invalid(format!(
            "{D}D ALE FSI cell requires one exact {}-vertex simplex closure",
            D + 1
        )));
    }
    Ok(())
}

fn evaluate_affine_residual(
    local: &LocalContribution,
    point: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    let entry_count = local
        .rows()
        .checked_mul(local.columns())
        .ok_or_else(|| invalid("ALE FSI affine local matrix shape overflows usize"))?;
    if local.columns() != point.len() || local.matrix().len() != entry_count {
        return Err(invalid(
            "ALE FSI affine residual point differs from its local matrix shape",
        ));
    }
    let residual = local
        .matrix()
        .chunks_exact(local.columns())
        .zip(local.rhs())
        .map(|(row, rhs)| {
            row.iter()
                .zip(point)
                .map(|(entry, value)| entry * value)
                .sum::<f64>()
                - rhs
        })
        .collect::<Vec<_>>();
    if residual.len() != local.rows() || residual.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "ALE FSI affine local residual is non-finite or differs from its row closure",
        ));
    }
    Ok(residual)
}

fn affine_rhs_from_residual(
    matrix: &[f64],
    point: &[f64],
    residual: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    let entry_count = residual
        .len()
        .checked_mul(point.len())
        .ok_or_else(|| invalid("ALE FSI dense local matrix shape overflows usize"))?;
    if matrix.len() != entry_count {
        return Err(invalid(
            "ALE FSI dense Jacobian differs from its residual and candidate shape",
        ));
    }
    let rhs = matrix
        .chunks_exact(point.len())
        .zip(residual)
        .map(|(row, residual)| {
            row.iter()
                .zip(point)
                .map(|(entry, value)| entry * value)
                .sum::<f64>()
                - residual
        })
        .collect::<Vec<_>>();
    if rhs.len() != residual.len() || rhs.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "ALE FSI captured local right-hand side is non-finite or has the wrong row shape",
        ));
    }
    Ok(rhs)
}

fn scatter_residual(
    output: &mut [f64],
    equations: &[Option<DofId>],
    local: &[f64],
) -> Result<(), Diagnostic> {
    if equations.len() != local.len() {
        return Err(invalid(
            "ALE FSI direct residual shape differs from its equation map",
        ));
    }
    for (equation, value) in equations.iter().zip(local) {
        if let Some(equation) = equation {
            let destination = output.get_mut(equation.index()).ok_or_else(|| {
                invalid("ALE FSI residual equation is outside its assembly target")
            })?;
            *destination += value;
        }
    }
    Ok(())
}

fn require_same_residual(direct: &[f64], reconstructed: &[f64]) -> Result<(), Diagnostic> {
    if direct.len() != reconstructed.len() {
        return Err(invalid(
            "captured ALE FSI relation residual shape differs from direct assembly",
        ));
    }
    let scale = direct
        .iter()
        .chain(reconstructed)
        .fold(1.0_f64, |scale, value| scale.max(value.abs()));
    let defect = direct
        .iter()
        .zip(reconstructed)
        .fold(0.0_f64, |defect, (direct, reconstructed)| {
            defect.max((direct - reconstructed).abs())
        });
    if defect > 65_536.0 * f64::EPSILON * scale {
        return Err(invalid(
            "captured ALE FSI relation does not reproduce its independently assembled residual",
        ));
    }
    Ok(())
}

fn finite_norm(values: &[f64], name: &'static str) -> Result<f64, Diagnostic> {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() {
        return Err(invalid(format!("{name} norm is non-finite")));
    }
    Ok(norm)
}

fn zeroed(length: usize, name: &'static str) -> Result<Vec<f64>, Diagnostic> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| invalid(format!("ALE FSI {name} allocation failed")))?;
    values.resize(length, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests;
