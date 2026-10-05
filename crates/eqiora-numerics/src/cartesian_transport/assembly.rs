use std::num::NonZeroUsize;
use std::sync::Arc;

use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPlan, AssemblyTarget, DofId,
    IndexedAssemblyWork, LocalContribution, LocalUnknown, REFERENCE_ASSEMBLY_BACKEND,
    TargetAssemblyMap,
};
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, DynQuantity, GraphPath};
use eqiora_meshing::MeshTopology;
use eqiora_realization::{
    CellCenteredConvectionScheme, ResolvedTransientCellCenteredTransportRealization,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::{
    CanonicalCsrSystemView, ExecutionTopology, LinearOperator, LinearOperatorProperties,
    LinearSolution, LinearSolverBackend,
};

use super::admission::{require_exact_realization, time_dimension, validate_previous};
use super::api::{
    FinalizedScalarTransportFvmStep2d, ScalarTransportCellState2d, ScalarTransportFvmStep2d,
    ScalarTransportFvmStepEvidence2d,
};
use super::faces::{cell_geometry, face_packet, transport_faces, validate_side_roles};
use super::replay::{
    advective_face_range, boundary_flux, face_counts, integrated_mass,
    replay_interior_cancellation, replay_physical_residual, require_complete_operator,
};
use crate::canonical_transport::{
    ScalarTransportCartesianModel2d, lower_scalar_transport_cartesian_2d,
};
use crate::finalized_spatial::FinalizedLinearCore;
use eqiora_meshing::CartesianMesh;

const DIMENSION: usize = 2;

/// Construct exact cell averages from the admitted constant-on-support scalar
/// initial equation. A constant scalar function has that same average in each cell.
///
/// This bounded slice has no second callback or array-valued initialization
/// channel. Shaped provided initial data requires a future typed Run contract.
///
/// # Errors
/// Rejects lineage/plan drift or a missing, non-scalar, non-finite, or
/// dimensionally inconsistent or inapplicable initial equation.
pub fn initialize_resolved_scalar_transport_fvm_2d(
    program: &KernelProgram,
    resolved: &ResolvedTransientCellCenteredTransportRealization,
) -> Result<(ScalarTransportCartesianModel2d, ScalarTransportCellState2d), Diagnostic> {
    let model = lower_scalar_transport_cartesian_2d(program)?;
    let selection = require_exact_realization(program, &model, resolved)?;
    let initial = super::initial::constant_initial_value(program, &model)?;
    let mesh = CartesianMesh::uniform(model.bounds(), &[selection.cells; DIMENSION])?;
    let cell_count = mesh
        .entity_count(DIMENSION)
        .expect("2D Cartesian mesh owns top cells");
    let state = ScalarTransportCellState2d {
        model: resolved.model(),
        semantic_revision: resolved.semantic_revision(),
        realization_revision: resolved.realization_revision(),
        field: model.state(),
        mesh,
        time: DynQuantity::new(0.0, time_dimension()),
        value_dimension: selection.field_dimension,
        values: vec![initial; cell_count],
    };
    Ok((model, state))
}

/// Finalize one resolved conservative transport step through reference assembly.
///
/// # Errors
/// Preserves exact admission, boundary-closure, geometry, and assembly failures.
pub fn finalize_resolved_scalar_transport_fvm_step_2d(
    program: &KernelProgram,
    resolved: &ResolvedTransientCellCenteredTransportRealization,
    previous: &ScalarTransportCellState2d,
) -> Result<
    (
        ScalarTransportCartesianModel2d,
        FinalizedScalarTransportFvmStep2d,
    ),
    Diagnostic,
> {
    finalize_resolved_scalar_transport_fvm_step_2d_with_assembly(
        program,
        resolved,
        previous,
        &REFERENCE_ASSEMBLY_BACKEND,
    )
}

/// Finalize one resolved conservative transport step through explicit assembly.
///
/// Every cell contributes one mass packet and every canonical facet contributes
/// one oriented flux packet. Boundary roles are validated before the backend
/// sees any work.
///
/// # Errors
/// Preserves reference finalization failures and the selected assembly
/// backend's complete-operation diagnostic.
pub fn finalize_resolved_scalar_transport_fvm_step_2d_with_assembly(
    program: &KernelProgram,
    resolved: &ResolvedTransientCellCenteredTransportRealization,
    previous: &ScalarTransportCellState2d,
    assembly: &dyn AssemblyBackend<f64>,
) -> Result<
    (
        ScalarTransportCartesianModel2d,
        FinalizedScalarTransportFvmStep2d,
    ),
    Diagnostic,
> {
    let model = lower_scalar_transport_cartesian_2d(program)?;
    let selection = require_exact_realization(program, &model, resolved)?;
    let mesh = CartesianMesh::uniform(model.bounds(), &[selection.cells; DIMENSION])?;
    validate_previous(&model, resolved, &mesh, selection.field_dimension, previous)?;
    let duration = selection.duration;
    if !(previous.time.value() + duration).is_finite() {
        return Err(invalid_realization(
            "scalar transport step endpoint time must be finite",
        ));
    }
    let cell_count = mesh
        .entity_count(DIMENSION)
        .expect("2D Cartesian mesh owns top cells");
    let (cell_centers, cell_measures) = cell_geometry(&mesh)?;
    let (faces, role_by_side, reconstruction, periodic_face_count) = transport_faces(
        &model,
        &mesh,
        &cell_centers,
        &previous.values,
        selection.convection_scheme,
        duration,
    )?;
    validate_side_roles(&model, &role_by_side)?;

    let row_scale = selection.state_scale / selection.weak_scale;
    let matrix_scale = selection.state_scale * row_scale;
    if !row_scale.is_finite() || !matrix_scale.is_finite() || row_scale <= 0.0 {
        return Err(invalid_realization(
            "transport congruence scaling produced a non-finite or non-positive factor",
        ));
    }
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(cell_count)?])?;
    let target = plan
        .target_id(0)
        .expect("one-target transport assembly owns target zero");
    let packet_count = cell_count
        .checked_add(faces.len())
        .ok_or_else(|| invalid_numerics("transport packet count overflows usize"))?;
    let work = IndexedAssemblyWork::new(packet_count, |packet| {
        let (local, map) = if packet < cell_count {
            let mass = cell_measures[packet] / duration;
            (
                LocalContribution::new(
                    1,
                    1,
                    vec![matrix_scale * mass],
                    vec![row_scale * mass * previous.values[packet]],
                )?,
                AssemblyMap::new(
                    vec![Some(DofId::new(packet))],
                    vec![LocalUnknown::Free(DofId::new(packet))],
                )?,
            )
        } else {
            face_packet(&faces[packet - cell_count], matrix_scale, row_scale)?
        };
        AssemblyPacket::new(local, vec![TargetAssemblyMap::new(target, map)])
    });
    let (systems, assembly_report) = assembly.assemble(&plan, &work)?.into_parts();
    if assembly_report.packet_count() != packet_count
        || assembly_report.target_count() != plan.target_count()
        || assembly_report.execution().topology()
            != (ExecutionTopology::Host {
                workers: NonZeroUsize::MIN,
            })
    {
        return Err(invalid_realization(
            "transport assembly receipt differs from the exact packet, target, or portable placement inventory",
        ));
    }
    let mut systems = systems.into_iter();
    let system = systems
        .next()
        .expect("one-target transport assembly returns one system");
    debug_assert!(systems.next().is_none());
    let canonical_system = Arc::new(CanonicalCsrSystemView::new(
        &system,
        LinearOperatorProperties::General,
    )?);
    let operator_replay = require_complete_operator(
        &canonical_system,
        &cell_measures,
        &faces,
        &previous.values,
        duration,
        selection.state_scale,
        selection.weak_scale,
    )?;
    let core = FinalizedLinearCore::new(
        selection.solver_plan,
        resolved
            .requirements()
            .fieldwise()
            .execution()
            .vector_layout(),
        selection.target,
        canonical_system,
    );
    let finalized = FinalizedScalarTransportFvmStep2d {
        core,
        realization: resolved.clone(),
        mesh,
        field: model.state(),
        previous_time: previous.time,
        value_dimension: previous.value_dimension,
        duration,
        previous_values: previous.values.clone(),
        cell_measures,
        faces,
        periodic_face_count,
        state_scale: selection.state_scale,
        weak_scale: selection.weak_scale,
        maximum_operator_replay_defect: operator_replay.maximum_defect,
        operator_replay_tolerance: operator_replay.tolerance,
        assembly_report,
        convection_scheme: selection.convection_scheme,
        maximum_courant_number: reconstruction.maximum_courant_number,
        limited_face_count: reconstruction.limited_face_count,
        advective_bounds: reconstruction.bounds,
    };
    Ok((model, finalized))
}

/// Solve one resolved step through reference assembly and an explicit solver.
///
/// # Errors
/// Preserves finalization, solver capability, true-residual, and conservation
/// diagnostics.
pub fn solve_resolved_scalar_transport_fvm_step_2d(
    program: &KernelProgram,
    resolved: &ResolvedTransientCellCenteredTransportRealization,
    previous: &ScalarTransportCellState2d,
    backend: &dyn LinearSolverBackend,
) -> Result<(ScalarTransportCartesianModel2d, ScalarTransportFvmStep2d), Diagnostic> {
    solve_resolved_scalar_transport_fvm_step_2d_with_assembly(
        program,
        resolved,
        previous,
        &REFERENCE_ASSEMBLY_BACKEND,
        backend,
    )
}

/// Solve one resolved step through independently selected assembly and solver adapters.
///
/// # Errors
/// Preserves all typed admission and adapter diagnostics.
pub fn solve_resolved_scalar_transport_fvm_step_2d_with_assembly(
    program: &KernelProgram,
    resolved: &ResolvedTransientCellCenteredTransportRealization,
    previous: &ScalarTransportCellState2d,
    assembly: &dyn AssemblyBackend<f64>,
    backend: &dyn LinearSolverBackend,
) -> Result<(ScalarTransportCartesianModel2d, ScalarTransportFvmStep2d), Diagnostic> {
    let (model, finalized) = finalize_resolved_scalar_transport_fvm_step_2d_with_assembly(
        program, resolved, previous, assembly,
    )?;
    let solved = backend.solve(&finalized.linear_problem()?, finalized.solver_plan())?;
    Ok((model, finalized.finish(solved)?))
}

pub(super) fn finish_step(
    finalized: FinalizedScalarTransportFvmStep2d,
    solved: LinearSolution,
) -> Result<ScalarTransportFvmStep2d, Diagnostic> {
    finalized.core.validate_solution(&solved)?;
    if solved.values().len() != finalized.cell_measures.len() {
        return Err(invalid_numerics(
            "transport solution shape differs from its finalized cell system",
        ));
    }
    let (dimensionless_values, solve_report) = solved.into_parts();
    let values = dimensionless_values
        .iter()
        .map(|value| finalized.state_scale * value)
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_numerics(
            "transport physical reconstruction is non-finite",
        ));
    }

    let mut residual = vec![0.0; dimensionless_values.len()];
    finalized
        .core
        .canonical_csr_system_view()
        .apply(&dimensionless_values, &mut residual)?;
    for (value, right) in residual
        .iter_mut()
        .zip(finalized.core.canonical_csr_system_view().right_hand_side())
    {
        *value -= right;
    }
    let replayed_residual_norm = residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !replayed_residual_norm.is_finite()
        || replayed_residual_norm > solve_report.residual_target()
    {
        return Err(invalid_numerics(
            "transport CSR replay does not satisfy the accepted solver target",
        ));
    }

    let physical_residual = replay_physical_residual(
        &finalized.cell_measures,
        &finalized.faces,
        &finalized.previous_values,
        &values,
        finalized.duration,
    )?;
    let csr_to_physical = finalized.weak_scale / finalized.state_scale;
    let mut maximum_assembly_replay_defect: f64 = 0.0;
    let mut assembly_replay_scale: f64 = 1.0;
    for (csr_value, physical_value) in residual.iter().zip(&physical_residual) {
        let reconstructed = csr_to_physical * csr_value;
        maximum_assembly_replay_defect =
            maximum_assembly_replay_defect.max((reconstructed - physical_value).abs());
        assembly_replay_scale = assembly_replay_scale
            .max(reconstructed.abs())
            .max(physical_value.abs());
    }
    let assembly_replay_tolerance = 4096.0 * f64::EPSILON * assembly_replay_scale;
    if !maximum_assembly_replay_defect.is_finite()
        || maximum_assembly_replay_defect > assembly_replay_tolerance
    {
        return Err(invalid_numerics(format!(
            "transport independent assembly replay defect {:e} exceeds tolerance {:e}",
            maximum_assembly_replay_defect, assembly_replay_tolerance
        )));
    }

    let old_mass = integrated_mass(&finalized.cell_measures, &finalized.previous_values);
    let new_mass = integrated_mass(&finalized.cell_measures, &values);
    let outward_boundary_flux =
        boundary_flux(&finalized.faces, &finalized.previous_values, &values)?;
    let conservation_defect = (new_mass - old_mass) / finalized.duration + outward_boundary_flux;
    let physical_scale = ((new_mass - old_mass) / finalized.duration)
        .abs()
        .max(outward_boundary_flux.abs())
        .max(1.0);
    let residual_bound = (dimensionless_values.len() as f64).sqrt() * finalized.weak_scale
        / finalized.state_scale
        * solve_report.true_residual_norm();
    let conservation_tolerance = 8.0 * residual_bound + 4096.0 * f64::EPSILON * physical_scale;
    if !conservation_defect.is_finite() || conservation_defect.abs() > conservation_tolerance {
        return Err(invalid_numerics(format!(
            "transport global conservation defect {:e} exceeds tolerance {:e}",
            conservation_defect, conservation_tolerance
        )));
    }

    let minimum_value = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (
        coupled_face_count,
        boundary_face_count,
        inflow_face_count,
        outflow_face_count,
        wall_face_count,
    ) = face_counts(&finalized.faces);
    let interior_face_count = coupled_face_count
        .checked_sub(finalized.periodic_face_count)
        .ok_or_else(|| invalid_numerics("transport periodic face count exceeds coupled faces"))?;
    let maximum_interior_cancellation_defect =
        replay_interior_cancellation(&finalized.faces, &finalized.previous_values, &values)?;
    let (advective_face_value_range, advective_face_bound_defect) = advective_face_range(
        &finalized.faces,
        &finalized.previous_values,
        &values,
        finalized.advective_bounds,
    )?;
    let advective_bound_scale = finalized.advective_bounds[0]
        .abs()
        .max(finalized.advective_bounds[1].abs())
        .max(1.0);
    let advective_face_bound_tolerance = (finalized.convection_scheme
        == CellCenteredConvectionScheme::ExplicitPreviousStateCartesianMinmod)
        .then_some(4096.0 * f64::EPSILON * advective_bound_scale);
    if let Some(tolerance) = advective_face_bound_tolerance
        && (!advective_face_bound_defect.is_finite() || advective_face_bound_defect > tolerance)
    {
        return Err(invalid_numerics(format!(
            "limited advective face bound defect {advective_face_bound_defect:e} exceeds tolerance {tolerance:e}"
        )));
    }
    let evidence = ScalarTransportFvmStepEvidence2d {
        convection_scheme: finalized.convection_scheme,
        maximum_courant_number: finalized.maximum_courant_number,
        limited_face_count: finalized.limited_face_count,
        advective_face_value_range,
        advective_face_bound_defect,
        advective_face_bound_tolerance,
        old_mass,
        new_mass,
        outward_boundary_flux,
        conservation_defect,
        conservation_tolerance,
        minimum_value,
        maximum_value,
        replayed_residual_norm,
        maximum_assembly_replay_defect,
        assembly_replay_tolerance,
        maximum_operator_replay_defect: finalized.maximum_operator_replay_defect,
        operator_replay_tolerance: finalized.operator_replay_tolerance,
        interior_face_count,
        periodic_face_count: finalized.periodic_face_count,
        boundary_face_count,
        inflow_face_count,
        outflow_face_count,
        wall_face_count,
        maximum_interior_cancellation_defect,
        assembly_report: finalized.assembly_report,
        solve_report,
    };
    let state = ScalarTransportCellState2d {
        model: finalized.realization.model(),
        semantic_revision: finalized.realization.semantic_revision(),
        realization_revision: finalized.realization.realization_revision(),
        field: finalized.field,
        mesh: finalized.mesh,
        time: DynQuantity::new(
            finalized.previous_time.value() + finalized.duration,
            time_dimension(),
        ),
        value_dimension: finalized.value_dimension,
        values,
    };
    Ok(ScalarTransportFvmStep2d {
        realization: finalized.realization,
        state,
        evidence,
    })
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message).with_graph_path(GraphPath::new([
        "realization".to_owned(),
        "scalar-transport-fvm-2d".to_owned(),
    ]))
}

fn invalid_numerics(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message).with_graph_path(GraphPath::new([
        "numerics".to_owned(),
        "scalar-transport-fvm-2d".to_owned(),
    ]))
}
