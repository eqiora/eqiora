use eqiora_assembly::{AssemblyBackend, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_core::{Diagnostic, ScalarDomain, ScalarType};
use eqiora_meshing::{QuadratureRule, SimplicialMesh};
use eqiora_solver::{LinearOperatorProperties, LinearSolverBackend};

use super::acceptance::{accept_step, require_consistent_initial_state};
use super::api::{
    MiniNavierStokesStepPlan2d, SimplicialMiniNavierStokesState2d,
    SimplicialMiniNavierStokesTrajectory2d,
};
use super::assembly::{
    PreparedStepStructure, assemble_step_linearization_prepared, assemble_step_residual_prepared,
    initial_point_prepared, prepare_step_structure,
};
use super::{COMPONENTS, DIMENSION, invalid, norm};
use crate::nonlinear_execution::{NewtonTrial, solve_damped_newton};
use crate::simplicial_stokes::SimplicialMiniStokesBoundary2d;
use crate::step_count::NonZeroStepCount;

/// Advance a fixed mesh through one or more accepted implicit steps with the
/// deterministic reference assembler.
///
/// # Errors
/// Fails closed on initial-state/mesh/pressure drift, insufficient quadrature,
/// analytic linearization failure, Krylov nonconvergence, unsuccessful
/// globalization, or non-monotone model time. A failed step does not mutate
/// the last accepted state. The convection audit requires cell exactness of at
/// least eight and one-dimensional facet exactness of at least three.
#[allow(clippy::too_many_arguments)]
pub fn advance_simplicial_mini_navier_stokes_2d<F, B>(
    mesh: &SimplicialMesh,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    body_force: &F,
    initial: SimplicialMiniNavierStokesState2d,
    step_count: NonZeroStepCount,
    plan: MiniNavierStokesStepPlan2d,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    solver: &dyn LinearSolverBackend,
) -> Result<SimplicialMiniNavierStokesTrajectory2d, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    advance_simplicial_mini_navier_stokes_2d_with_assembly(
        mesh,
        boundary,
        essential_velocity,
        body_force,
        initial,
        step_count,
        plan,
        cell_quadrature,
        facet_quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
    )
}

/// Advance through an explicit assembly adapter and common linear backend.
///
/// # Errors
/// Preserves all fixed-domain, assembly, Newton, Krylov, and acceptance
/// diagnostics without fallback, including the convection audit's minimum
/// cell exactness of eight and one-dimensional facet exactness of three.
#[allow(clippy::too_many_arguments)]
pub fn advance_simplicial_mini_navier_stokes_2d_with_assembly<F, B>(
    mesh: &SimplicialMesh,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    body_force: &F,
    initial: SimplicialMiniNavierStokesState2d,
    step_count: NonZeroStepCount,
    plan: MiniNavierStokesStepPlan2d,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: &dyn LinearSolverBackend,
) -> Result<SimplicialMiniNavierStokesTrajectory2d, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    super::element::require_convective_evidence_quadrature(cell_quadrature, facet_quadrature)?;
    let prepared = prepare_step_structure(
        mesh,
        boundary,
        essential_velocity,
        cell_quadrature,
        facet_quadrature,
        &plan,
    )?
    .bind(body_force)?;
    advance_simplicial_mini_navier_stokes_2d_with_prepared_structure(
        mesh,
        &prepared,
        initial,
        step_count,
        plan,
        cell_quadrature,
        facet_quadrature,
        assembly,
        solver,
        |_| Ok(()),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn advance_simplicial_mini_navier_stokes_2d_with_prepared_structure(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    initial: SimplicialMiniNavierStokesState2d,
    step_count: NonZeroStepCount,
    plan: MiniNavierStokesStepPlan2d,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: &dyn LinearSolverBackend,
    mut accept: impl FnMut(&SimplicialMiniNavierStokesState2d) -> Result<(), Diagnostic>,
) -> Result<SimplicialMiniNavierStokesTrajectory2d, Diagnostic> {
    solver.capabilities().require_problem(
        plan.linear_solver(),
        ScalarDomain::Real,
        ScalarType::F64,
        LinearOperatorProperties::General,
    )?;
    super::element::require_convective_evidence_quadrature(cell_quadrature, facet_quadrature)?;
    prepared.require_quadrature(cell_quadrature, facet_quadrature)?;
    let mut linear = solver.prepare_linear(plan.linear_solver())?;
    let mut trajectory = SimplicialMiniNavierStokesTrajectory2d::new(initial);
    for _ in 0..step_count.get() {
        if let Some(linear) = &mut linear {
            linear.begin_action()?;
        }
        let result = (|| {
            let previous = trajectory
                .states()
                .last()
                .expect("trajectory owns its initial state");
            let mut borrowed = linear
                .as_mut()
                .map(|linear| &mut **linear as &mut dyn eqiora_solver::PreparedLinearSolver);
            let (next, evidence) = solve_one_step(
                mesh,
                prepared,
                previous,
                plan.clone(),
                cell_quadrature,
                facet_quadrature,
                assembly,
                solver,
                &mut borrowed,
            )?;
            accept(&next)?;
            trajectory.push(next, evidence)
        })();
        if let Err(error) = result {
            if let Some(linear) = &mut linear {
                let _ = linear.discard_action();
            }
            return Err(error);
        }
        if let Some(linear) = &mut linear
            && let Err(error) = linear.commit_action()
        {
            let _ = linear.discard_action();
            return Err(error);
        }
    }
    Ok(trajectory)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn advance_simplicial_mini_navier_stokes_2d_with_prepared_structure_and_linear(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    initial: SimplicialMiniNavierStokesState2d,
    step_count: NonZeroStepCount,
    plan: MiniNavierStokesStepPlan2d,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: &dyn LinearSolverBackend,
    prepared_linear: Option<&mut (dyn eqiora_solver::PreparedLinearSolver + '_)>,
) -> Result<SimplicialMiniNavierStokesTrajectory2d, Diagnostic> {
    solver.capabilities().require_problem(
        plan.linear_solver(),
        ScalarDomain::Real,
        ScalarType::F64,
        LinearOperatorProperties::General,
    )?;
    super::element::require_convective_evidence_quadrature(cell_quadrature, facet_quadrature)?;
    prepared.require_quadrature(cell_quadrature, facet_quadrature)?;
    let mut prepared_linear = prepared_linear;
    let mut trajectory = SimplicialMiniNavierStokesTrajectory2d::new(initial);
    for _ in 0..step_count.get() {
        let previous = trajectory
            .states()
            .last()
            .expect("trajectory owns its initial state");
        let (next, evidence) = solve_one_step(
            mesh,
            prepared,
            previous,
            plan.clone(),
            cell_quadrature,
            facet_quadrature,
            assembly,
            solver,
            &mut prepared_linear,
        )?;
        trajectory.push(next, evidence)?;
    }
    Ok(trajectory)
}

#[allow(clippy::too_many_arguments)]
fn solve_one_step(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    previous: &SimplicialMiniNavierStokesState2d,
    plan: MiniNavierStokesStepPlan2d,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly_backend: &dyn AssemblyBackend<f64>,
    solver: &dyn LinearSolverBackend,
    prepared_linear: &mut Option<&mut (dyn eqiora_solver::PreparedLinearSolver + '_)>,
) -> Result<
    (
        SimplicialMiniNavierStokesState2d,
        super::api::SimplicialMiniNavierStokesStepEvidence2d,
    ),
    Diagnostic,
> {
    let point = initial_point_prepared(mesh, prepared, previous)?;
    require_consistent_initial_state(mesh, cell_quadrature, previous, plan.clone())?;
    let nonlinear = eqiora_realization::NonlinearSolvePlan::new(
        plan.nonlinear_relative_tolerance(),
        plan.nonlinear_absolute_tolerance(),
        plan.maximum_newton_iterations(),
        plan.maximum_line_search_steps(),
    )?;
    let (current, evidence) = solve_damped_newton(
        point,
        nonlinear.into(),
        |point| {
            assemble_step_linearization_prepared(
                mesh,
                prepared,
                previous,
                point,
                plan.clone(),
                assembly_backend,
            )
        },
        |current| current.residual_norm(),
        |_, current| {
            let right_hand_side = current
                .residual
                .iter()
                .map(|value| -value)
                .collect::<Vec<_>>();
            let linear_problem = current
                .relation
                .state_jacobian()
                .linear_problem_with_right_hand_side(&right_hand_side)?;
            let solution =
                if let Some(prepared_linear) = prepared_linear.as_deref_mut() {
                    let structure = current.assembly_report.structure_identity().ok_or_else(|| {
                    invalid("prepared linear solve requires an exact assembly structure identity")
                })?;
                    prepared_linear.solve(structure, &linear_problem)?
                } else {
                    solver.solve(&linear_problem, plan.linear_solver())?
                };
            Ok(solution.into_parts())
        },
        |candidate, _| {
            let residual =
                assemble_step_residual_prepared(mesh, prepared, previous, candidate, plan.clone())?;
            Ok(Some(NewtonTrial {
                residual_norm: norm(&residual)?,
                linearization: None,
            }))
        },
    )?;
    accept_step(
        mesh,
        previous,
        plan,
        cell_quadrature,
        facet_quadrature,
        current,
        evidence,
    )
}
