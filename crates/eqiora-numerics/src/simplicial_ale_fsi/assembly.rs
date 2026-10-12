//! Monolithic reduced residual and exact ALE state Jacobian assembly.
//!
//! The algebraic candidate uses the unchanged dimensionless FSI quotient
//! layout.  Solid displacement and current geometry are not independent
//! unknowns: backward Euler derives the former from shared solid velocity and
//! the sealed harmonic action derives the latter.  Every Jacobian column
//! follows that same composition analytically.
//!
//! This module prepares structure, evaluates cells, and captures the linear
//! system. The private result module owns the completed assembly value and
//! its read-only inspection methods, keeping result access separate from the
//! assembly procedure.

mod fluid;
use fluid::{evaluate_fluid_jacobian, evaluate_fluid_residual};
mod residual;
mod structure;
pub(super) use structure::build_step_jacobian_pattern;
mod result;
use residual::{finite_norm, require_same_residual, zeroed};
pub(super) use result::StepAssembly;

use std::sync::Arc;

use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan,
    AssemblyTarget, DofId, LocalContribution, LocalUnknown, TargetAssemblyMap,
};
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::FixedTopologyGeometryAction;
use eqiora_meshing::{
    CellId, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, SimplicialMesh,
};
use eqiora_solver::{CanonicalCsrSystemView, LinearOperatorProperties};

use super::boundary_step::PreparedAleFsiBoundaryStep;
use super::contract::{AleFsiBoundary, AleFsiState, AleFsiStepPlan};
use super::{P1HarmonicMeshMotionAction, invalid};
use crate::assembled_linearization::AssembledLinearizedRelation;
use crate::form_compiler::region::RegionLinearization;
use crate::jacobian_audit::{StructuralJacobianPattern, StructuralJacobianPatternBuilder};
use crate::region_assembly::mapping::RegionDirection;
use crate::region_assembly::nonlinear::{
    EvaluatedRegionPacket, assemble_linearization, evaluate_linear_residual, scatter_residual,
};
use crate::simplicial_fsi::layout::FsiLayout;
use crate::simplicial_fsi::{FixedReferenceFsiPartition, FixedReferenceFsiState};

type MappedCellResidual = (
    Vec<f64>,
    AssemblyMap<f64>,
    AssemblyMap<f64>,
    CellResidualSource,
);

struct PreparedAleFsiCell {
    reduced_map: AssemblyMap<f64>,
    full_map: AssemblyMap<f64>,
    dense_map: AssemblyMap<f64>,
    bubble_cell: Option<CellId>,
    reference_action: Option<crate::form_compiler::region::PreparedRegionCell<f64>>,
}

/// Immutable layout and assembly structure shared by every action in one Run.
pub(super) struct PreparedAleFsiStructure<const D: usize> {
    boundary_template: PreparedAleFsiBoundaryStep<D>,
    layout: Arc<FsiLayout<D>>,
    directions: Vec<RegionDirection<D>>,
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
                    layout.mapping().cell_map(cell.index(), true)?,
                    layout.mapping().cell_map(cell.index(), false)?,
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
        let reference_action = if bubble_cell.is_none() {
            let geometry = reference
                .geometry_map(MeshEntity::new(D, cell_index))
                .ok_or_else(|| invalid("reference cell has no affine geometry"))?;
            Some(
                layout
                    .region_form(layout.cell_domain(cell_index)?)?
                    .prepare_cell(&geometry, quadrature)?,
            )
        } else {
            None
        };
        cells.push(PreparedAleFsiCell {
            reduced_map,
            full_map,
            dense_map,
            bubble_cell,
            reference_action,
        });
    }
    #[cfg(test)]
    {
        phases.maps += 1;
    }
    let assembly_plan = assembly_plan.prepare(
        cells
            .iter()
            .map(|cell| {
                vec![TargetAssemblyMap::new(
                    reduced_target,
                    cell.dense_map.clone(),
                )]
            })
            .collect(),
    )?;
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
    assembly: &dyn AssemblyBackend<f64>,
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
    assembly: &dyn AssemblyBackend<f64>,
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
    assembly: &dyn AssemblyBackend<f64>,
) -> Result<StepAssembly<D>, Diagnostic> {
    let prepared = prepare_step(
        reference, partition, structure, action, motion, previous, plan, candidate,
    )?;
    let evaluate = |cell_index| {
        evaluate_cell(
            &structure.layout,
            cell_index,
            quadrature,
            previous,
            &action.previous_reference,
            &prepared.current,
            &prepared.geometry_action,
            candidate,
            &structure.directions,
            structure
                .cells
                .get(cell_index)
                .ok_or_else(|| invalid("ALE FSI cell is outside prepared Run structure"))?,
            structure.reduced_target,
        )
    };
    let assembled = assemble_linearization(
        &structure.assembly_plan,
        AssemblyPacketSetIdentityV1::Unbound,
        structure.cells.len(),
        assembly,
        evaluate,
    )?;
    let systems = assembled.systems;
    let assembly_report = assembled.report;
    if assembly_report.packet_count() != structure.cells.len()
        || assembly_report.target_count() != 1
    {
        return Err(invalid(
            "ALE FSI assembly evidence differs from its exact cell and target inventory",
        ));
    }

    let direct = assemble_direct_residuals(
        structure, action, previous, candidate, quadrature, &prepared,
    )?;
    require_same_residual(&direct.reduced, &assembled.residuals[0])?;

    let [linear_system]: [eqiora_assembly::LinearSystem<f64>; 1] =
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
        &structure, &action, previous, candidate, quadrature, &prepared,
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
    structure: &PreparedAleFsiStructure<D>,
    action: &PreparedAleFsiAction<D>,
    previous: &AleFsiState<D>,
    candidate: &[f64],
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
            quadrature,
            previous,
            &action.previous_reference,
            &prepared.current,
            &prepared.geometry_action,
            candidate,
            cell,
        )?;
        scatter_residual(&mut reduced, &evaluated.reduced_map, &evaluated.residual)?;
        let full = match evaluated.domain {
            domain if structure.layout.pressure_field(domain).is_some() => &mut full_fluid,
            domain if structure.layout.state_field(domain).is_some() => &mut full_solid,
            _ => {
                return Err(invalid(format!(
                    "ALE FSI cell {cell_index} has no material assignment"
                )));
            }
        };
        scatter_residual(full, &evaluated.full_map, &evaluated.residual)?;
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

struct EvaluatedCellResidual {
    residual: Vec<f64>,
    reduced_map: AssemblyMap<f64>,
    full_map: AssemblyMap<f64>,
    domain: RawId,
    source: CellResidualSource,
}

enum CellResidualSource {
    Fluid { bubble_cell: CellId },
    Solid(LocalContribution<f64>),
}

#[allow(clippy::too_many_arguments)]
fn evaluate_cell<const D: usize>(
    layout: &FsiLayout<D>,
    cell_index: usize,
    quadrature: &QuadratureRule,
    previous: &AleFsiState<D>,
    previous_reference: &FixedReferenceFsiState<D>,
    current: &AleFsiState<D>,
    geometry_action: &FixedTopologyGeometryAction<D>,
    candidate: &[f64],
    directions: &[RegionDirection<D>],
    cell: &PreparedAleFsiCell,
    reduced_target: eqiora_assembly::AssemblyTargetId,
) -> Result<EvaluatedRegionPacket, Diagnostic> {
    let evaluated = evaluate_cell_residual(
        layout,
        cell_index,
        quadrature,
        previous,
        previous_reference,
        current,
        geometry_action,
        candidate,
        cell,
    )?;
    let matrix = match &evaluated.source {
        CellResidualSource::Fluid { bubble_cell } => evaluate_fluid_jacobian(
            layout,
            cell_index,
            *bubble_cell,
            quadrature,
            previous,
            current,
            geometry_action,
            candidate.len(),
            directions,
        )?,
        CellResidualSource::Solid(local) => {
            embed_solid_jacobian(local, &evaluated.reduced_map, candidate.len())?
        }
    };
    let local = RegionLinearization {
        jacobian: matrix,
        residual: evaluated.residual.clone(),
    }
    .into_contribution(candidate)?;
    Ok(EvaluatedRegionPacket {
        residual: evaluated.residual,
        assembly: AssemblyPacket::new(
            local,
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
    quadrature: &QuadratureRule,
    previous: &AleFsiState<D>,
    previous_reference: &FixedReferenceFsiState<D>,
    current: &AleFsiState<D>,
    geometry_action: &FixedTopologyGeometryAction<D>,
    candidate: &[f64],
    cell: &PreparedAleFsiCell,
) -> Result<EvaluatedCellResidual, Diagnostic> {
    let domain = layout.cell_domain(cell_index)?;
    let (residual, reduced_map, full_map, source) = match domain {
        domain if layout.pressure_field(domain).is_some() => evaluate_fluid_residual(
            layout,
            cell_index,
            cell,
            quadrature,
            previous,
            current,
            geometry_action,
        )?,
        domain if layout.state_field(domain).is_some() => {
            evaluate_solid_residual(layout, cell_index, cell, previous_reference, candidate)?
        }
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

#[allow(clippy::too_many_arguments)]
fn evaluate_solid_residual<const D: usize>(
    layout: &FsiLayout<D>,
    cell_index: usize,
    cell: &PreparedAleFsiCell,
    previous: &FixedReferenceFsiState<D>,
    candidate: &[f64],
) -> Result<MappedCellResidual, Diagnostic> {
    let form = layout.region_form(layout.cell_domain(cell_index)?)?;
    let previous =
        layout
            .mapping()
            .physical_cell_history(cell_index, form, Some(&previous.fields))?;
    let local = cell
        .reference_action
        .as_ref()
        .ok_or_else(|| invalid("reference cell lacks its prepared equation action"))?
        .evaluate(&previous)?;
    let reduced_map = cell.reduced_map.clone();
    let residual = evaluate_linear_residual(&local, &reduced_map, candidate)?;
    let full_map = cell.full_map.clone();
    Ok((
        residual,
        reduced_map,
        full_map,
        CellResidualSource::Solid(local),
    ))
}

fn embed_solid_jacobian(
    local: &LocalContribution<f64>,
    reduced_map: &AssemblyMap<f64>,
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

#[cfg(test)]
mod tests;
