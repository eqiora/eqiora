//! Explicit implementation verification for the transient MINI realization.

use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
use eqiora_core::Diagnostic;
use eqiora_ir::{LinearizedRelation, RelationTangent};
use eqiora_meshing::{QuadratureRule, SimplicialMesh};

use super::api::{MiniNavierStokesStepPlan2d, SimplicialMiniNavierStokesState2d};
use super::assembly::{
    assemble_step_linearization, assemble_step_residual, build_step_jacobian_pattern, initial_point,
};
use super::{COMPONENTS, DIMENSION, invalid};
use crate::jacobian_audit::{CenteredJacobianVerification, audit_centered_jacobian};
use crate::simplicial_stokes::SimplicialMiniStokesBoundary2d;

impl MiniNavierStokesStepPlan2d {
    /// Reconstruct every analytic column at one accepted transient MINI step.
    ///
    /// This is an explicit product verifier, not part of ordinary numerical
    /// acceptance. It reuses the production residual and analytic JVP owners.
    ///
    /// Returns `(columns, colors, globally_coupled_singletons,
    /// residual_assemblies, maximum_error)`.
    ///
    /// # Errors
    /// Returns a diagnostic when the states do not describe this plan's step,
    /// when assembly fails, or when an analytic column differs from the
    /// centered residual reconstruction.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_accepted_jacobian<F, B>(
        self,
        mesh: &SimplicialMesh,
        boundary: &SimplicialMiniStokesBoundary2d,
        essential_velocity: &B,
        body_force: &F,
        previous: &SimplicialMiniNavierStokesState2d,
        accepted: &SimplicialMiniNavierStokesState2d,
        cell_quadrature: &QuadratureRule,
        facet_quadrature: &QuadratureRule,
    ) -> Result<(usize, usize, usize, usize, f64), Diagnostic>
    where
        F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
        B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    {
        verify_simplicial_mini_navier_stokes_2d_jacobian(
            mesh,
            boundary,
            essential_velocity,
            body_force,
            previous,
            accepted,
            self,
            cell_quadrature,
            facet_quadrature,
        )
        .map(|verification| {
            (
                verification.column_count(),
                verification.color_count(),
                verification.globally_coupled_singleton_count(),
                verification.residual_assembly_count(),
                verification.maximum_error(),
            )
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_simplicial_mini_navier_stokes_2d_jacobian<F, B>(
    mesh: &SimplicialMesh,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    body_force: &F,
    previous: &SimplicialMiniNavierStokesState2d,
    accepted: &SimplicialMiniNavierStokesState2d,
    plan: MiniNavierStokesStepPlan2d,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
) -> Result<CenteredJacobianVerification, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    if accepted.time().to_bits() != (previous.time() + plan.time_step()).to_bits() {
        return Err(invalid(
            "explicit transient MINI verification requires the exact accepted successor state",
        ));
    }
    super::element::require_convective_evidence_quadrature(cell_quadrature, facet_quadrature)?;
    let point = initial_point(
        mesh,
        boundary,
        essential_velocity,
        accepted,
        cell_quadrature,
        facet_quadrature,
    )?;
    let assembled = assemble_step_linearization(
        mesh,
        boundary,
        essential_velocity,
        body_force,
        previous,
        &point,
        plan.clone(),
        cell_quadrature,
        facet_quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
    )?;
    let pattern = build_step_jacobian_pattern(
        mesh,
        boundary,
        essential_velocity,
        cell_quadrature,
        facet_quadrature,
    )?;
    audit_centered_jacobian(
        &point,
        &pattern,
        8.0e-6,
        "transient MINI",
        |candidate| {
            assemble_step_residual(
                mesh,
                boundary,
                essential_velocity,
                body_force,
                previous,
                candidate,
                plan.clone(),
                cell_quadrature,
                facet_quadrature,
            )
        },
        |column, analytic| {
            let mut direction = vec![0.0; point.len()];
            direction[column] = 1.0;
            assembled
                .relation
                .jvp(RelationTangent::Unknown(&direction), analytic)
        },
    )
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use eqiora_core::diagnostic::codes;
    use eqiora_meshing::MeshQualityGate;
    use eqiora_realization::Target;
    use eqiora_solver::{
        BackendId, LinearOperatorProperties, LinearProblem, LinearSolution, LinearSolver,
        LinearSolverBackend, PreconditionerPolicy, ReductionPolicy, ReplicatedLinearExecution,
        SolverCapabilities, SolverCapability, SolverPlan, SolverProvider,
    };

    use super::*;
    use crate::simplicial_elliptic::SimplicialP1Field;
    use crate::simplicial_stokes::{
        SimplicialMiniStokesPressureReference2d, SimplicialMiniVelocityField2d,
    };
    use crate::step_count::NonZeroStepCount;

    #[test]
    fn equation_derived_open_boundary_preserves_uniform_throughflow() {
        use crate::simplicial_stokes::{
            SimplicialMiniStokesBoundaryCondition2d, SimplicialMiniStokesBoundaryFacet2d,
        };
        use eqiora_meshing::{
            MeshEntity, MeshTopology, simplex_duffy_gauss_legendre, triangle_duffy_gauss_legendre,
        };
        let vertices = (0..3)
            .flat_map(|y| (0..3).map(move |x| vec![x as f64 / 2.0, y as f64 / 2.0]))
            .collect();
        let mut cells = Vec::new();
        for y in 0..2 {
            for x in 0..2 {
                let a = 3 * y + x;
                cells.push(vec![a, a + 1, a + 4]);
                cells.push(vec![a, a + 4, a + 3]);
            }
        }
        let mesh =
            SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.01).unwrap()).unwrap();
        let facets: Vec<_> = (0..mesh.entity_count(1).unwrap())
            .map(|i| MeshEntity::new(1, i))
            .filter(|facet| mesh.is_boundary_entity(*facet) == Some(true))
            .map(|facet| {
                let outlet = mesh
                    .entity_vertices(facet)
                    .unwrap()
                    .iter()
                    .all(|v| mesh.vertices()[v.index()][0] == 1.0);
                SimplicialMiniStokesBoundaryFacet2d::new(
                    facet,
                    if outlet {
                        SimplicialMiniStokesBoundaryCondition2d::ConstantTraction { value: [0.; 2] }
                    } else {
                        SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity
                    },
                )
            })
            .collect();
        let boundary = SimplicialMiniStokesBoundary2d::new(&mesh, facets).unwrap();
        let velocity =
            SimplicialMiniVelocityField2d::new(mesh.clone(), vec![[2., 3.]; 9], vec![[0.; 2]; 8])
                .unwrap();
        let pressure = SimplicialP1Field::new(mesh.clone(), vec![0.; 9]).unwrap();
        let previous = SimplicialMiniNavierStokesState2d::new(
            0.,
            velocity.clone(),
            pressure.clone(),
            SimplicialMiniStokesPressureReference2d::BoundaryTraction,
        )
        .unwrap();
        let accepted = SimplicialMiniNavierStokesState2d::new(
            0.1,
            velocity,
            pressure,
            SimplicialMiniStokesPressureReference2d::BoundaryTraction,
        )
        .unwrap();
        let plan = MiniNavierStokesStepPlan2d::new(
            4.,
            0.1,
            0.1,
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
        let cell_rule = triangle_duffy_gauss_legendre(5).unwrap();
        let facet_rule = simplex_duffy_gauss_legendre(1, 3).unwrap();
        let essential = |_| Ok([2., 3.]);
        let load = |_| Ok([0.; 2]);
        let point = initial_point(
            &mesh,
            &boundary,
            &essential,
            &previous,
            &cell_rule,
            &facet_rule,
        )
        .unwrap();
        let prepared = super::super::assembly::prepare_step_structure(
            &mesh,
            &boundary,
            &essential,
            &cell_rule,
            &facet_rule,
        )
        .unwrap()
        .bind(&plan, &load)
        .unwrap();
        prepared
            .require_quadrature(&cell_rule, &facet_rule)
            .unwrap();
        assert!(
            prepared
                .require_quadrature(&triangle_duffy_gauss_legendre(6).unwrap(), &facet_rule)
                .is_err()
        );
        for (density, viscosity, step) in [(8., 0.1, 0.1), (4., 0.2, 0.1), (4., 0.1, 0.2)] {
            let mut changed = plan.clone();
            changed.form = std::sync::Arc::new(
                super::super::form::StepForm::reference(density, viscosity, step).unwrap(),
            );
            assert!(
                super::super::assembly::assemble_step_residual_prepared(
                    &mesh, &prepared, &previous, &point, changed,
                )
                .is_err()
            );
        }
        let residual = assemble_step_residual(
            &mesh,
            &boundary,
            &essential,
            &load,
            &previous,
            &point,
            plan.clone(),
            &cell_rule,
            &facet_rule,
        )
        .unwrap();
        for value in residual {
            assert!(
                value.abs() < 1e-10,
                "uniform throughflow residual {value:e}"
            );
        }
        let verification = plan
            .verify_accepted_jacobian(
                &mesh,
                &boundary,
                &essential,
                &load,
                &previous,
                &accepted,
                &cell_rule,
                &facet_rule,
            )
            .unwrap();
        assert!(verification.4 < 1e-5);
    }

    struct NoSolveBackend;

    impl fmt::Debug for NoSolveBackend {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("NoSolveBackend")
        }
    }

    impl LinearSolverBackend for NoSolveBackend {
        fn provider(&self) -> SolverProvider {
            SolverProvider::new(
                BackendId::new("eqiora.test.no-solve-transient-mini"),
                env!("CARGO_PKG_VERSION"),
                &[],
            )
        }

        fn capabilities(&self) -> SolverCapabilities {
            SolverCapabilities::exact([SolverCapability {
                algorithm: LinearSolver::SparseLu,
                operator_properties: LinearOperatorProperties::General,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: eqiora_core::ScalarType::F64,
            }])
            .unwrap()
        }

        fn solve_with_execution(
            &self,
            _problem: &LinearProblem<'_>,
            _plan: SolverPlan,
            _execution: &dyn ReplicatedLinearExecution,
        ) -> Result<LinearSolution, Diagnostic> {
            Err(Diagnostic::error(
                codes::NUMERICAL_SOLVE_FAILED,
                "zero equilibrium must not invoke the test solver",
            ))
        }
    }

    #[test]
    fn explicit_verifier_reconstructs_the_production_mini_jacobian() {
        let mesh = SimplicialMesh::new(
            2,
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 1.0],
            ],
            vec![vec![0, 1, 2], vec![2, 3, 0]],
            MeshQualityGate::new(0.05).unwrap(),
        )
        .unwrap();
        let boundary = SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap();
        let state = |time| {
            SimplicialMiniNavierStokesState2d::new(
                time,
                SimplicialMiniVelocityField2d::new(
                    mesh.clone(),
                    vec![[0.0; 2]; mesh.vertices().len()],
                    vec![[0.0; 2]; 2],
                )
                .unwrap(),
                SimplicialP1Field::new(mesh.clone(), vec![0.0; mesh.vertices().len()]).unwrap(),
                SimplicialMiniStokesPressureReference2d::ZeroIntegral { multiplier: 0.0 },
            )
            .unwrap()
        };
        let plan = MiniNavierStokesStepPlan2d::new(
            1.0,
            0.1,
            0.01,
            1.0e-9,
            1.0e-11,
            NonZeroUsize::new(8).unwrap(),
            4,
            SolverPlan::new(
                LinearSolver::SparseLu,
                1.0e-10,
                1.0e-12,
                NonZeroUsize::new(100).unwrap(),
            )
            .unwrap()
            .with_reduction(ReductionPolicy::Fast),
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
        )
        .unwrap();
        let cell_quadrature = eqiora_meshing::triangle_duffy_gauss_legendre(5).unwrap();
        let facet_quadrature = eqiora_meshing::simplex_duffy_gauss_legendre(1, 3).unwrap();
        let prepared_trace_evaluations = AtomicUsize::new(0);
        crate::jacobian_audit::reset_centered_residual_assembly_count();
        super::super::assembly::reset_packet_evaluations();
        let trajectory = super::super::advance_simplicial_mini_navier_stokes_2d(
            &mesh,
            &boundary,
            &|_| {
                prepared_trace_evaluations.fetch_add(1, Ordering::Relaxed);
                Ok([0.0; 2])
            },
            &|_| Ok([0.0; 2]),
            state(0.0),
            NonZeroStepCount::new(NonZeroUsize::new(10).unwrap()),
            plan.clone(),
            &cell_quadrature,
            &facet_quadrature,
            &NoSolveBackend,
        )
        .unwrap();
        assert_eq!(trajectory.states().len(), 11);
        assert_eq!(
            super::super::assembly::packet_evaluations(),
            [20, 20, 0],
            "each of ten assemblies evaluates two cells and two gauge packets exactly once",
        );
        let structure = trajectory.steps()[0]
            .assembly_report()
            .structure_identity()
            .expect("fixed transient assembly owns a prepared structure");
        assert!(trajectory.steps().iter().all(|step| {
            step.assembly_report().target_count() == 1
                && step.assembly_report().structure_identity() == Some(structure)
        }));
        assert_eq!(
            prepared_trace_evaluations.load(Ordering::Relaxed),
            mesh.vertices().len(),
            "ten actions must evaluate the invariant essential trace only during preparation",
        );
        assert_eq!(
            crate::jacobian_audit::centered_residual_assembly_count(),
            0,
            "ordinary acceptance must perform no centered-Jacobian residual assemblies",
        );
        let verification = verify_simplicial_mini_navier_stokes_2d_jacobian(
            &mesh,
            &boundary,
            &|_| Ok([0.0; 2]),
            &|_| Ok([0.0; 2]),
            &trajectory.states()[0],
            &trajectory.states()[1],
            plan.clone(),
            &cell_quadrature,
            &facet_quadrature,
        )
        .unwrap();
        assert!(verification.column_count() > 0);
        assert_eq!(
            verification.residual_assembly_count(),
            2 * verification.color_count()
        );
        assert_eq!(
            crate::jacobian_audit::centered_residual_assembly_count(),
            verification.residual_assembly_count(),
        );
        assert!(verification.maximum_error() < 1.0e-3);
    }
}
