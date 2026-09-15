use eqiora_assembly::{AssemblyBackend, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_core::Diagnostic;
use eqiora_meshing::{QuadratureRule, SimplicialMesh};
use eqiora_solver::LinearSolverBackend;

use super::acceptance::{NewtonEvidence, accept_step, require_consistent_initial_state};
use super::api::{
    MiniNavierStokesStepPlan2d, SimplicialMiniNavierStokesState2d,
    SimplicialMiniNavierStokesTrajectory2d,
};
use super::assembly::{
    PreparedStepStructure, assemble_step_linearization_prepared, initial_point_prepared,
    prepare_step_structure,
};
use super::element::FixedDomainViscousForm;
use super::{COMPONENTS, DIMENSION, invalid, solve_failed};
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
    assembly: &dyn AssemblyBackend,
    solver: &dyn LinearSolverBackend,
) -> Result<SimplicialMiniNavierStokesTrajectory2d, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    advance_with_viscous_form(
        mesh,
        boundary,
        essential_velocity,
        body_force,
        initial,
        step_count,
        plan,
        cell_quadrature,
        facet_quadrature,
        assembly,
        solver,
    )
}

#[allow(clippy::too_many_arguments)]
fn advance_with_viscous_form<F, B>(
    mesh: &SimplicialMesh,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    body_force: &F,
    initial: SimplicialMiniNavierStokesState2d,
    step_count: NonZeroStepCount,
    plan: MiniNavierStokesStepPlan2d,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend,
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
    )?;
    let mut prepared_linear = solver.prepare_linear(plan.linear_solver())?;
    advance_simplicial_mini_navier_stokes_2d_with_prepared_structure_and_linear(
        mesh,
        &prepared,
        body_force,
        initial,
        step_count,
        plan,
        cell_quadrature,
        facet_quadrature,
        assembly,
        solver,
        prepared_linear.as_deref_mut(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn advance_simplicial_mini_navier_stokes_2d_with_prepared_structure_and_linear<F>(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    body_force: &F,
    initial: SimplicialMiniNavierStokesState2d,
    step_count: NonZeroStepCount,
    plan: MiniNavierStokesStepPlan2d,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend,
    solver: &dyn LinearSolverBackend,
    mut prepared_linear: Option<&mut (dyn eqiora_solver::PreparedLinearSolver + '_)>,
) -> Result<SimplicialMiniNavierStokesTrajectory2d, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    super::element::require_convective_evidence_quadrature(cell_quadrature, facet_quadrature)?;
    let mut trajectory = SimplicialMiniNavierStokesTrajectory2d::new(initial);
    for _ in 0..step_count.get() {
        let previous = trajectory
            .states()
            .last()
            .expect("trajectory owns its initial state");
        let (next, evidence) = solve_one_step(
            mesh,
            prepared,
            body_force,
            previous,
            plan.clone(),
            cell_quadrature,
            facet_quadrature,
            assembly,
            solver,
            &mut prepared_linear,
            FixedDomainViscousForm::SymmetricNewtonian,
        )?;
        trajectory.push(next, evidence)?;
    }
    Ok(trajectory)
}

#[allow(clippy::too_many_arguments)]
fn solve_one_step<F>(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    body_force: &F,
    previous: &SimplicialMiniNavierStokesState2d,
    plan: MiniNavierStokesStepPlan2d,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly_backend: &dyn AssemblyBackend,
    solver: &dyn LinearSolverBackend,
    prepared_linear: &mut Option<&mut (dyn eqiora_solver::PreparedLinearSolver + '_)>,
    viscous_form: FixedDomainViscousForm,
) -> Result<
    (
        SimplicialMiniNavierStokesState2d,
        super::api::SimplicialMiniNavierStokesStepEvidence2d,
    ),
    Diagnostic,
>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let mut point = initial_point_prepared(mesh, prepared, previous)?;
    require_consistent_initial_state(mesh, cell_quadrature, previous, plan.clone())?;
    let mut current = {
        let _assembly =
            eqiora_execution::telemetry_span!(assembly("initial_linearization")).entered();
        assemble_step_linearization_prepared(
            mesh,
            prepared,
            body_force,
            previous,
            &point,
            plan.clone(),
            assembly_backend,
            viscous_form,
        )?
    };
    let initial_residual_norm = current.residual_norm()?;
    let residual_target = plan.nonlinear_target(initial_residual_norm)?;
    eqiora_execution::telemetry_event!(
        "newton",
        0,
        initial_residual_norm,
        residual_target,
        initial_residual_norm <= residual_target,
    );
    if initial_residual_norm <= residual_target {
        return accept_step(
            mesh,
            previous,
            plan.clone(),
            cell_quadrature,
            facet_quadrature,
            current,
            NewtonEvidence {
                iterations: 0,
                initial_residual_norm,
                residual_target,
                linear_solves: Vec::new(),
            },
        );
    }

    let mut reports = Vec::new();
    for iteration in 1..=plan.maximum_newton_iterations().get() {
        let previous_norm = current.residual_norm()?;
        let _iteration = eqiora_execution::telemetry_span!(nonlinear_iteration(
            "newton",
            iteration,
            previous_norm
        ))
        .entered();
        let right_hand_side = current
            .residual
            .iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        let linear_problem = current
            .relation
            .state_jacobian()
            .linear_problem_with_right_hand_side(&right_hand_side)?;
        let solution = if let Some(prepared_linear) = prepared_linear.as_deref_mut() {
            let structure = current
                .assembly_report
                .structure_identity()
                .ok_or_else(|| {
                    invalid("prepared linear solve requires an exact assembly structure identity")
                })?;
            prepared_linear.solve(structure, &linear_problem)?
        } else {
            solver.solve(&linear_problem, plan.linear_solver())?
        };
        reports.push(solution.report().clone());
        let correction = solution.values();
        let mut accepted = None;
        let mut best_trial = (f64::INFINITY, 0.0);
        let mut scale = 1.0;
        for _ in 0..=plan.maximum_line_search_steps() {
            let candidate = point
                .iter()
                .zip(correction)
                .map(|(point, correction)| point + scale * correction)
                .collect::<Vec<_>>();
            let assembled = {
                let _assembly =
                    eqiora_execution::telemetry_span!(assembly("line_search_trial")).entered();
                assemble_step_linearization_prepared(
                    mesh,
                    prepared,
                    body_force,
                    previous,
                    &candidate,
                    plan.clone(),
                    assembly_backend,
                    viscous_form,
                )?
            };
            let norm = assembled.residual_norm()?;
            if norm < best_trial.0 {
                best_trial = (norm, scale);
            }
            if norm <= residual_target || norm < previous_norm {
                accepted = Some((candidate, assembled, norm));
                break;
            }
            scale *= 0.5;
        }
        let Some((candidate, assembled, norm)) = accepted else {
            return Err(solve_failed(format!(
                "MINI Navier--Stokes Newton line search failed at iteration {iteration}: \
                     previous residual {previous_norm:e}, best trial residual {:e} at scale {:e}, \
                     target {residual_target:e}",
                best_trial.0, best_trial.1,
            )));
        };
        point = candidate;
        current = assembled;
        eqiora_execution::telemetry_event!(
            "newton",
            iteration,
            norm,
            residual_target,
            norm <= residual_target,
        );
        if norm <= residual_target {
            return accept_step(
                mesh,
                previous,
                plan.clone(),
                cell_quadrature,
                facet_quadrature,
                current,
                NewtonEvidence {
                    iterations: iteration,
                    initial_residual_norm,
                    residual_target,
                    linear_solves: reports,
                },
            );
        }
    }
    Err(solve_failed(format!(
        "MINI Navier--Stokes Newton solve reached {} iterations above target {residual_target:e}",
        plan.maximum_newton_iterations()
    )))
}
