use eqiora_core::Diagnostic;
use eqiora_meshing::{MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, SimplicialMesh};

use super::api::{
    MiniNavierStokesStepPlan2d, SimplicialMiniNavierStokesState2d,
    SimplicialMiniNavierStokesStepEvidence2d,
};
use super::assembly::StepAssembly;
use super::element::{
    evaluate_velocity, integrate_convective_evidence, local_velocity_coefficients,
};
use super::{invalid, solve_failed};
use crate::simplicial_stokes::acceptance::{
    integrate_pressure, require_weak_incompressibility_from_gauge_weights,
    require_zero_gauge_multiplier,
};
use crate::simplicial_stokes::element::{MiniSpaces, physical_gradients};

pub(super) struct NewtonEvidence {
    pub(super) iterations: usize,
    pub(super) initial_residual_norm: f64,
    pub(super) residual_target: f64,
    pub(super) linear_solves: Vec<eqiora_solver::SolveReport>,
}

pub(super) fn require_consistent_initial_state(
    mesh: &SimplicialMesh,
    quadrature: &QuadratureRule,
    state: &SimplicialMiniNavierStokesState2d,
    plan: MiniNavierStokesStepPlan2d,
) -> Result<(), Diagnostic> {
    let coefficient_scale = state
        .velocity()
        .vertex_values()
        .iter()
        .chain(state.velocity().cell_bubble_values())
        .flatten()
        .chain(state.pressure().vertex_values())
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let tolerance = (16.0 * plan.nonlinear_absolute_tolerance())
        .max(8192.0 * f64::EPSILON * (1.0 + coefficient_scale));
    let continuity = initial_weak_continuity_norm(mesh, quadrature, state.velocity())?;
    if !continuity.is_finite() || continuity > tolerance {
        return Err(invalid(format!(
            "initial MINI weak continuity residual {continuity:e} exceeds consistency tolerance {tolerance:e}"
        )));
    }
    let pressure_integral = integrate_pressure(mesh, quadrature, state.pressure().vertex_values())?;
    if let Some(multiplier) = state.pressure_reference().gauge_multiplier() {
        require_zero_gauge_multiplier(mesh, quadrature, multiplier, tolerance)?;
        let pressure_tolerance = tolerance
            + 8192.0
                * f64::EPSILON
                * (1.0
                    + state
                        .pressure()
                        .vertex_values()
                        .iter()
                        .map(|value| value.abs())
                        .fold(0.0, f64::max));
        if pressure_integral.abs() > pressure_tolerance {
            return Err(invalid(format!(
                "initial MINI pressure integral {pressure_integral:e} exceeds consistency tolerance {pressure_tolerance:e}"
            )));
        }
    }
    Ok(())
}

fn initial_weak_continuity_norm(
    mesh: &SimplicialMesh,
    quadrature: &QuadratureRule,
    velocity: &crate::simplicial_stokes::SimplicialMiniVelocityField2d,
) -> Result<f64, Diagnostic> {
    let spaces = MiniSpaces::new()?;
    let mut residual = vec![0.0; mesh.vertices().len()];
    for cell in 0..mesh
        .entity_count(super::DIMENSION)
        .expect("mesh owns cells")
    {
        let entity = MeshEntity::new(super::DIMENSION, cell);
        let geometry = mesh
            .geometry_map(entity)
            .expect("accepted simplex cell owns geometry");
        let vertices = mesh
            .entity_vertices(entity)
            .expect("accepted simplex cell owns vertices");
        let coefficients = local_velocity_coefficients(velocity, cell, &vertices);
        let inverse = geometry.inverse_jacobian()?;
        for point in quadrature.points() {
            let basis = spaces.tabulate(&point.coordinates)?;
            let gradients = physical_gradients(&basis, &inverse);
            let (_, gradient) = evaluate_velocity(&coefficients, &basis.values, &gradients);
            let divergence = (0..super::DIMENSION)
                .map(|axis| gradient[axis][axis])
                .sum::<f64>();
            let scale = point.weight * geometry.measure_scale();
            for (local, pressure_basis) in basis.pressure_values.iter().enumerate() {
                residual[vertices[local].index()] -= scale * pressure_basis * divergence;
            }
        }
    }
    Ok(residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt())
}

pub(super) fn accept_step(
    mesh: &SimplicialMesh,
    previous: &SimplicialMiniNavierStokesState2d,
    plan: MiniNavierStokesStepPlan2d,
    quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: StepAssembly,
    newton: NewtonEvidence,
) -> Result<
    (
        SimplicialMiniNavierStokesState2d,
        SimplicialMiniNavierStokesStepEvidence2d,
    ),
    Diagnostic,
> {
    let final_residual_norm = assembly.residual_norm()?;
    let momentum_residual_norm = assembly.momentum_residual_norm();
    if !final_residual_norm.is_finite() || final_residual_norm > newton.residual_target {
        return Err(solve_failed(format!(
            "accepted nonlinear residual {final_residual_norm:e} exceeds target {:e}",
            newton.residual_target
        )));
    }
    let (full_residual, gauge_weights) = assembly.materialize_acceptance_data()?;
    let continuity_residual_norm = require_weak_incompressibility_from_gauge_weights(
        &full_residual,
        &assembly.layout,
        &gauge_weights,
        assembly.gauge_multiplier,
        newton.residual_target,
    )?;
    let pressure_integral =
        integrate_pressure(mesh, quadrature, assembly.pressure.vertex_values())?;
    if let Some(multiplier) = assembly.gauge_multiplier {
        require_zero_gauge_multiplier(mesh, quadrature, multiplier, newton.residual_target)?;
        let pressure_tolerance = 4096.0
            * f64::EPSILON
            * (1.0
                + assembly
                    .pressure
                    .vertex_values()
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0, f64::max));
        if pressure_integral.abs() > pressure_tolerance + newton.residual_target {
            return Err(invalid(format!(
                "zero-integral pressure closure produced integral {pressure_integral:e}"
            )));
        }
    }
    let convective = integrate_convective_evidence(
        mesh,
        &assembly.velocity,
        plan.density(),
        quadrature,
        facet_quadrature,
    )?;
    let coefficient_scale = assembly
        .velocity
        .vertex_values()
        .iter()
        .chain(assembly.velocity.cell_bubble_values())
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let power_tolerance =
        16384.0 * f64::EPSILON * (1.0 + coefficient_scale * convective.skew_residual_norm);
    if convective.skew_power.abs() > power_tolerance {
        return Err(invalid(format!(
            "skew convective self-work {:e} exceeds algebraic tolerance {power_tolerance:e}",
            convective.skew_power
        )));
    }
    let defect_tolerance = 32768.0
        * f64::EPSILON
        * (1.0 + convective.skew_residual_norm + convective.conservative_defect_norm);
    if convective.defect_identity_error > defect_tolerance {
        return Err(invalid(format!(
            "skew/conservative advection defect identity error {:e} exceeds algebraic tolerance {defect_tolerance:e}",
            convective.defect_identity_error
        )));
    }
    let named_boundary_reactions = assembly
        .named_reaction_vertices
        .iter()
        .map(|(name, vertices)| {
            let mut reaction = [0.0; super::COMPONENTS];
            for vertex in vertices {
                for (component, value) in reaction.iter_mut().enumerate() {
                    *value +=
                        full_residual[assembly.layout.full_vertex_velocity(*vertex, component)];
                }
            }
            (name.clone(), reaction)
        })
        .collect::<Vec<_>>();
    if named_boundary_reactions
        .iter()
        .flat_map(|(_, reaction)| reaction)
        .any(|value| !value.is_finite())
    {
        return Err(invalid(
            "accepted transient named boundary reaction is non-finite",
        ));
    }
    let next_time = previous.time() + plan.time_step();
    if !next_time.is_finite() || next_time <= previous.time() {
        return Err(invalid(
            "MINI Navier--Stokes step cannot advance representable model time",
        ));
    }
    let state = SimplicialMiniNavierStokesState2d::accepted(
        next_time,
        assembly.velocity,
        assembly.pressure,
        assembly.pressure_reference,
    );
    let evidence = SimplicialMiniNavierStokesStepEvidence2d::new(
        newton.iterations,
        newton.initial_residual_norm,
        final_residual_norm,
        momentum_residual_norm,
        newton.residual_target,
        continuity_residual_norm,
        pressure_integral,
        convective.skew_residual_norm,
        convective.skew_power,
        convective.conservative_defect_norm,
        named_boundary_reactions,
        assembly.assembly_report,
        newton.linear_solves,
    );
    Ok((state, evidence))
}
