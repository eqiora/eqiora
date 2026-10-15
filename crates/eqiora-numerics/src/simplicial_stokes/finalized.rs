use std::sync::Arc;

use eqiora_assembly::{AssemblyReport, LinearSystem};
use eqiora_core::Diagnostic;
use eqiora_meshing::{QuadratureRule, SimplicialMesh};
use eqiora_solver::{CanonicalCsrSystemView, LinearSolution};

use super::acceptance::{
    integrate_pressure, require_weak_incompressibility, require_zero_gauge_multiplier,
};
use super::api::{
    SimplicialMiniStokesPressureReference2d, SimplicialMiniStokesSolution2d,
    SimplicialMiniVelocityField2d,
};
use super::layout::MixedLayout;
use super::{COMPONENTS, invalid};
use crate::simplicial_boundary::sum_vertex_reaction;
use crate::simplicial_elliptic::SimplicialP1Field;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedMiniStokesAssembly {
    pub(super) mesh: SimplicialMesh,
    pub(super) layout: MixedLayout,
    pub(super) fixed_velocity: Vec<Option<[f64; COMPONENTS]>>,
    pub(super) named_reaction_vertices: Vec<(String, Vec<usize>)>,
    pub(super) linear_system: LinearSystem<f64>,
    pub(super) full_system: LinearSystem<f64>,
    pub(super) volume_only_system: LinearSystem<f64>,
    pub(super) integrated_body_force: [f64; COMPONENTS],
    pub(super) integrated_boundary_traction: [f64; COMPONENTS],
    pub(super) quadrature: QuadratureRule,
    pub(super) assembly_report: AssemblyReport,
}

impl FinalizedMiniStokesAssembly {
    pub(crate) fn into_canonical(
        self,
    ) -> Result<(Arc<CanonicalCsrSystemView>, FinalizedMiniStokesState), Diagnostic> {
        let Self {
            mesh,
            layout,
            fixed_velocity,
            named_reaction_vertices,
            linear_system,
            full_system,
            volume_only_system,
            integrated_body_force,
            integrated_boundary_traction,
            quadrature,
            assembly_report,
        } = self;
        let canonical_system = Arc::new(CanonicalCsrSystemView::new(
            &linear_system,
            linear_system.rhs(),
            eqiora_solver::LinearOperatorProperties::SymmetricIndefinite,
        )?);
        Ok((
            canonical_system,
            FinalizedMiniStokesState {
                mesh,
                layout,
                fixed_velocity,
                named_reaction_vertices,
                full_system,
                volume_only_system,
                integrated_body_force,
                integrated_boundary_traction,
                quadrature,
                assembly_report,
            },
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedMiniStokesState {
    mesh: SimplicialMesh,
    layout: MixedLayout,
    fixed_velocity: Vec<Option<[f64; COMPONENTS]>>,
    named_reaction_vertices: Vec<(String, Vec<usize>)>,
    full_system: LinearSystem<f64>,
    volume_only_system: LinearSystem<f64>,
    integrated_body_force: [f64; COMPONENTS],
    integrated_boundary_traction: [f64; COMPONENTS],
    quadrature: QuadratureRule,
    assembly_report: AssemblyReport,
}

impl FinalizedMiniStokesState {
    pub(crate) const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    pub(crate) fn boundary_reaction(
        &self,
        vertices: Vec<[f64; COMPONENTS]>,
        bubbles: Vec<[f64; COMPONENTS]>,
        pressure: Vec<f64>,
        name: &str,
    ) -> Result<Option<[f64; COMPONENTS]>, Diagnostic> {
        let velocity = SimplicialMiniVelocityField2d::new(self.mesh.clone(), vertices, bubbles)?;
        let pressure = SimplicialP1Field::new(self.mesh.clone(), pressure)?;
        named_reaction(
            &self.full_system,
            &self.layout,
            &self.named_reaction_vertices,
            &velocity,
            &pressure,
            name,
        )
    }

    pub(crate) fn finish(
        self,
        solved: LinearSolution,
        canonical_system: Arc<CanonicalCsrSystemView>,
    ) -> Result<SimplicialMiniStokesSolution2d, Diagnostic> {
        if solved.values().len() != canonical_system.rows() {
            return Err(invalid(
                "MINI Stokes solution shape differs from its finalized system",
            ));
        }
        let residual_target = solved.report().residual_target();
        let (algebraic_values, solve_report) = solved.into_parts();
        let (vertex_values, cell_bubble_values, pressure_values, gauge_multiplier) = self
            .layout
            .reconstruct(&algebraic_values, &self.fixed_velocity)?;
        let residual = full_residual(
            &self.full_system,
            &self.layout,
            &vertex_values,
            &cell_bubble_values,
            &pressure_values,
            gauge_multiplier,
        )?;
        let row = |vertex, component| self.layout.full_vertex_velocity(vertex, component);
        let boundary_reaction = sum_vertex_reaction(
            &residual,
            self.fixed_velocity
                .iter()
                .enumerate()
                .filter_map(|(vertex, value)| value.is_some().then_some(vertex)),
            row,
        );
        let pressure_integral = integrate_pressure(&self.mesh, &self.quadrature, &pressure_values)?;
        let continuity_residual_norm = require_weak_incompressibility(
            &self.full_system,
            &residual,
            &self.layout,
            gauge_multiplier,
            residual_target,
        )?;
        let pressure_reference = match gauge_multiplier {
            Some(multiplier) => {
                require_zero_gauge_multiplier(
                    &self.mesh,
                    &self.quadrature,
                    multiplier,
                    residual_target,
                )?;
                SimplicialMiniStokesPressureReference2d::ZeroIntegral { multiplier }
            }
            None => SimplicialMiniStokesPressureReference2d::BoundaryTraction,
        };
        if boundary_reaction
            .iter()
            .chain(&self.integrated_body_force)
            .chain(&self.integrated_boundary_traction)
            .chain([&pressure_integral, &continuity_residual_norm])
            .chain(gauge_multiplier.as_ref())
            .any(|value| !value.is_finite())
        {
            return Err(invalid("MINI Stokes evidence is non-finite"));
        }
        if self.full_system.matrix() != self.volume_only_system.matrix() {
            return Err(invalid(
                "MINI Stokes traction packets changed the volume operator matrix",
            ));
        }
        Ok(SimplicialMiniStokesSolution2d {
            velocity: SimplicialMiniVelocityField2d {
                mesh: self.mesh.clone(),
                vertex_values,
                cell_bubble_values,
            },
            pressure: SimplicialP1Field::new(self.mesh, pressure_values)?,
            pressure_reference,
            algebraic_values,
            canonical_system,
            full_system: self.full_system,
            volume_only_system: self.volume_only_system,
            boundary_reaction,
            layout: self.layout,
            named_reaction_vertices: self.named_reaction_vertices,
            integrated_body_force: self.integrated_body_force,
            integrated_boundary_traction: self.integrated_boundary_traction,
            pressure_integral,
            continuity_residual_norm,
            assembly_report: self.assembly_report,
            solve_report,
        })
    }
}

fn full_residual(
    system: &LinearSystem<f64>,
    layout: &MixedLayout,
    vertices: &[[f64; COMPONENTS]],
    bubbles: &[[f64; COMPONENTS]],
    pressure: &[f64],
    gauge: Option<f64>,
) -> Result<Vec<f64>, Diagnostic> {
    let mut values = vec![0.; layout.full_size];
    layout.fill_full_values(&mut values, vertices, bubbles, pressure, gauge);
    let mut residual = system.matrix().multiply(&values)?;
    for (value, rhs) in residual.iter_mut().zip(system.rhs()) {
        *value -= rhs;
    }
    Ok(residual)
}

pub(super) fn named_reaction(
    system: &LinearSystem<f64>,
    layout: &MixedLayout,
    selections: &[(String, Vec<usize>)],
    velocity: &SimplicialMiniVelocityField2d,
    pressure: &SimplicialP1Field,
    name: &str,
) -> Result<Option<[f64; COMPONENTS]>, Diagnostic> {
    let Some((_, vertices)) = selections.iter().find(|(key, _)| key == name) else {
        return Ok(None);
    };
    if velocity.mesh() != pressure.mesh() {
        return Err(invalid(
            "Stokes observation fields belong to different meshes",
        ));
    }
    // The zero-integral pressure multiplier acts only on pressure equations.
    // Its value does not contribute to a constrained momentum reaction.
    let residual = full_residual(
        system,
        layout,
        velocity.vertex_values(),
        velocity.cell_bubble_values(),
        pressure.vertex_values(),
        layout.full_gauge().map(|_| 0.),
    )?;
    let force = sum_vertex_reaction(&residual, vertices.iter().copied(), |vertex, component| {
        layout.full_vertex_velocity(vertex, component)
    });
    if force.iter().any(|value| !value.is_finite()) {
        return Err(invalid("Stokes boundary observation is non-finite"));
    }
    Ok(Some(force))
}

#[cfg(test)]
mod tests {
    use crate::fluid::{
        SimplicialMiniStokesBoundary2d, SimplicialMiniStokesBoundaryCondition2d,
        SimplicialMiniStokesBoundaryFacet2d, solve_simplicial_mini_stokes_2d,
    };
    use crate::simplicial_boundary::reaction_reductions;
    use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
    use eqiora_meshing::{
        MeshEntity, MeshQualityGate, MeshTopology, SimplicialMesh, simplex_centroid_rule,
        simplex_duffy_gauss_legendre,
    };
    use eqiora_solver::{
        LinearSolveRequest, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
        ReductionPolicy, SolverPlan,
    };

    #[test]
    fn acceptance_reduces_only_global_force_until_a_named_surface_is_requested() {
        let mesh = SimplicialMesh::new(
            2,
            vec![vec![0., 0.], vec![1., 0.], vec![1., 1.], vec![0., 1.]],
            vec![vec![0, 1, 2], vec![0, 2, 3]],
            MeshQualityGate::new(0.4).unwrap(),
        )
        .unwrap();
        let facets = (0..mesh.entity_count(1).unwrap())
            .map(|index| MeshEntity::new(1, index))
            .filter(|facet| mesh.is_boundary_entity(*facet).unwrap())
            .collect::<Vec<_>>();
        let mut boundary = SimplicialMiniStokesBoundary2d::new(
            &mesh,
            facets.iter().map(|facet| {
                SimplicialMiniStokesBoundaryFacet2d::new(
                    *facet,
                    SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity,
                )
            }),
        )
        .unwrap();
        for (name, x) in [("left", 0.), ("right", 1.)] {
            let selected = facets.iter().copied().filter(|facet| {
                mesh.entity_vertices(*facet)
                    .unwrap()
                    .iter()
                    .all(|vertex| mesh.vertices()[vertex.index()][0] == x)
            });
            boundary = boundary
                .with_named_reaction_surface(&mesh, name, selected)
                .unwrap();
        }
        let policy = SolverPlan::new(
            LinearSolver::MinimumResidual,
            1e-11,
            1e-13,
            std::num::NonZeroUsize::new(100).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(ReductionPolicy::Reproducible);
        let before = reaction_reductions();
        let solution = solve_simplicial_mini_stokes_2d(
            &mesh,
            1.,
            &|_| Ok([0., 0.]),
            &boundary,
            &|_| Ok([0., 0.]),
            &simplex_duffy_gauss_legendre(2, 3).unwrap(),
            &simplex_centroid_rule(1).unwrap(),
            &REFERENCE_ASSEMBLY_BACKEND,
            LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, policy),
        )
        .unwrap();
        // Acceptance must still compute its one global conservation reduction.
        // Neither of the two available named observations belongs to that work.
        assert_eq!(reaction_reductions() - before, 1);
        let accepted = solution.clone();
        assert_eq!(solution.named_boundary_reaction("missing").unwrap(), None);
        assert_eq!(reaction_reductions() - before, 1);
        assert_eq!(
            solution.named_boundary_reaction("left").unwrap(),
            Some([0., 0.])
        );
        assert_eq!(reaction_reductions() - before, 2);
        assert_eq!(solution, accepted);
        assert_eq!(
            solution.named_boundary_reaction("right").unwrap(),
            Some([0., 0.])
        );
        assert_eq!(reaction_reductions() - before, 3);
        assert_eq!(solution, accepted);
    }
}
