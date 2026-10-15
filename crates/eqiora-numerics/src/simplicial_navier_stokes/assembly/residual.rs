//! Shared residual evaluation with solver and constrained-reaction projections.

use super::*;

pub(crate) fn assemble_step_residual_prepared(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    previous: &SimplicialMiniNavierStokesState2d,
    candidate: &[f64],
    plan: MiniNavierStokesStepPlan2d,
) -> Result<Vec<f64>, Diagnostic> {
    let mut residual = vec![0.0; prepared.layout.reduced_size()];
    visit_step_residuals(
        mesh,
        prepared,
        previous,
        candidate,
        &plan,
        |_, map, local| scatter_residual(&mut residual, map, local),
    )?;
    require_finite_residual(&residual)?;
    Ok(residual)
}

/// Recover only the requested constrained-surface reaction, without a solve or Jacobian.
pub(crate) fn boundary_reaction_prepared(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    previous: &SimplicialMiniNavierStokesState2d,
    current: &SimplicialMiniNavierStokesState2d,
    plan: &MiniNavierStokesStepPlan2d,
    name: &str,
) -> Result<[f64; COMPONENTS], Diagnostic> {
    let vertices = prepared
        .named_reaction_vertices
        .iter()
        .find_map(|(candidate, vertices)| (candidate == name).then_some(vertices))
        .ok_or_else(|| invalid("selected boundary has no constrained reaction"))?;
    let point = initial_point_prepared(mesh, prepared, current)?;
    let mut residual = vec![0.0; prepared.layout.full_size()];
    visit_step_residuals(
        mesh,
        prepared,
        previous,
        &point,
        plan,
        |packet, _, local| scatter_residual(&mut residual, &prepared.full_maps[packet], local),
    )?;
    require_finite_residual(&residual)?;
    let force = crate::simplicial_boundary::sum_vertex_reaction(
        &residual,
        vertices.iter().copied(),
        |vertex, component| prepared.layout.full_vertex_velocity(vertex, component),
    );
    require_finite_residual(&force)?;
    Ok(force)
}

fn require_finite_residual(residual: &[f64]) -> Result<(), Diagnostic> {
    if residual.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "direct transient residual-only assembly produced a non-finite value",
        ));
    }
    Ok(())
}

fn visit_step_residuals(
    mesh: &SimplicialMesh,
    prepared: &PreparedStepStructure,
    previous: &SimplicialMiniNavierStokesState2d,
    candidate: &[f64],
    plan: &MiniNavierStokesStepPlan2d,
    mut visit: impl FnMut(usize, &AssemblyMap<f64>, &[f64]) -> Result<(), Diagnostic>,
) -> Result<(), Diagnostic> {
    let forms = prepared.cells(plan)?;
    let step = prepare_step_point(mesh, prepared, previous, candidate)?;
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
                plan,
                &local_point,
                false,
            )?
            .residual
        };
        visit(packet, map, &local_residual)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplicial_stokes::{
        SimplicialMiniStokesBoundaryCondition2d, SimplicialMiniStokesBoundaryFacet2d,
    };
    use eqiora_meshing::{MeshQualityGate, simplex_duffy_gauss_legendre};
    use eqiora_realization::Target;
    use eqiora_solver::{LinearSolver, ReductionPolicy, SolverPlan};
    use std::num::NonZeroUsize;

    #[test]
    fn constrained_reaction_integrates_previous_bubble_mass_without_solving() {
        let mesh = SimplicialMesh::new(
            2,
            vec![vec![0., 0.], vec![1., 0.], vec![0., 1.]],
            vec![vec![0, 1, 2]],
            MeshQualityGate::new(0.01).unwrap(),
        )
        .unwrap();
        let facets = (0..mesh.entity_count(1).unwrap()).map(|i| MeshEntity::new(1, i));
        let boundary = SimplicialMiniStokesBoundary2d::new(
            &mesh,
            facets.map(|facet| {
                SimplicialMiniStokesBoundaryFacet2d::new(
                    facet,
                    SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity,
                )
            }),
        )
        .unwrap()
        .with_named_reaction_surface(&mesh, "edge", [MeshEntity::new(1, 0)])
        .unwrap();
        let state = |time, bubble| {
            SimplicialMiniNavierStokesState2d::new(
                time,
                SimplicialMiniVelocityField2d::new(mesh.clone(), vec![[0.; 2]; 3], vec![bubble])
                    .unwrap(),
                SimplicialP1Field::new(mesh.clone(), vec![0.; 3]).unwrap(),
                SimplicialMiniStokesPressureReference2d::ZeroIntegral { multiplier: 0. },
            )
            .unwrap()
        };
        let previous = state(0., [3., -2.]);
        let current = state(0.5, [0.; 2]);
        let plan = MiniNavierStokesStepPlan2d::new(
            2.,
            1.,
            0.5,
            1e-9,
            1e-11,
            NonZeroUsize::new(8).unwrap(),
            4,
            SolverPlan::new(
                LinearSolver::SparseLu,
                1e-10,
                1e-12,
                NonZeroUsize::new(100).unwrap(),
            )
            .unwrap()
            .with_reduction(ReductionPolicy::Fast),
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
        )
        .unwrap();
        let prepared = prepare_step_structure(
            &mesh,
            &boundary,
            &|_| Ok([0.; 2]),
            &simplex_duffy_gauss_legendre(2, 5).unwrap(),
            &simplex_duffy_gauss_legendre(1, 3).unwrap(),
            &plan,
        )
        .unwrap()
        .bind(&|_| Ok([0.; 2]))
        .unwrap();
        let force =
            boundary_reaction_prepared(&mesh, &prepared, &previous, &current, &plan, "edge")
                .unwrap();
        // B = 27 λ0 λ1 λ2. Each vertex mass is ∫ λ_i B = 3 A / 20.
        // A = 1/2; the edge owns two vertices. With current velocity/pressure
        // zero, only -ρ/dt times the previous bubble contributes: -0.6 [3,-2].
        // This probes the residual at supplied coefficients, not an accepted
        // divergence-free trajectory or a new solver-admissibility claim.
        for (actual, expected) in force.into_iter().zip([-1.8_f64, 1.2]) {
            assert!((actual - expected).abs() <= 1024. * f64::EPSILON * expected.abs());
        }
        assert_eq!(
            boundary_reaction_prepared(&mesh, &prepared, &current, &current, &plan, "edge")
                .unwrap(),
            [0.; 2]
        );
        assert!(
            boundary_reaction_prepared(&mesh, &prepared, &previous, &current, &plan, "missing")
                .is_err()
        );
    }
}
