use super::*;
use eqiora_solver::{
    LinearProblem, LinearSolverBackend, ReplicatedLinearExecution, SolverCapabilities,
    SolverProvider,
};

fn enclosed_harmonic_patch() -> (SimplicialMesh, FixedReferenceFsiPartition<2>) {
    let vertices = (0..7)
        .flat_map(|y| (0..7).map(move |x| vec![f64::from(x) / 2., f64::from(y) / 2.]))
        .collect();
    let mut cells = Vec::new();
    let mut harmonic = Vec::new();
    let mut driven = Vec::new();
    for y in 0..6 {
        for x in 0..6 {
            let lower = y * 7 + x;
            for triangle in [
                vec![lower, lower + 1, lower + 8],
                vec![lower, lower + 8, lower + 7],
            ] {
                let cell = CellId::new(cells.len());
                if (1..5).contains(&x) && (1..5).contains(&y) {
                    harmonic.push(cell);
                } else {
                    driven.push(cell);
                }
                cells.push(triangle);
            }
        }
    }
    let mesh = SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.1).unwrap()).unwrap();
    let partition = exact_partition(&mesh, harmonic, driven);
    (mesh, partition)
}

#[test]
fn compatible_reference_plans_preserve_affine_harmonic_extension_and_exact_reports() {
    let (mesh, partition) = enclosed_harmonic_patch();
    // Every harmonic boundary vertex is driven by the surrounding Domain.
    // An affine vector field has zero Laplacian, and P1 represents it exactly;
    // uniqueness of the Dirichlet problem fixes its interior extension.
    let affine = |point: &[f64]| {
        [
            1. / 256. + point[0] / 128. - point[1] / 64.,
            -1. / 128. + point[0] / 64. + point[1] / 256.,
        ]
    };
    let driver = partition
        .domain_vertices(solid_domain())
        .unwrap()
        .iter()
        .map(|&vertex| (vertex, affine(&mesh.vertices()[vertex.index()])))
        .collect::<BTreeMap<_, _>>();
    for (algorithm, preconditioner) in [
        (
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Identity,
        ),
        (
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Jacobi,
        ),
        (
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Identity,
        ),
        (
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Jacobi,
        ),
        (
            LinearSolver::MinimumResidual,
            PreconditionerPolicy::Identity,
        ),
        (LinearSolver::MinimumResidual, PreconditionerPolicy::Jacobi),
    ] {
        let plan = SolverPlan::new(algorithm, 1e-13, 1e-15, NonZeroUsize::new(100).unwrap())
            .unwrap()
            .with_preconditioner(preconditioner);
        let solver = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan);
        let action =
            P1HarmonicMeshMotionAction::new(&mesh, &partition, motion_policy(plan), solver)
                .unwrap();
        assert!(action.fixed_exterior_vertices().is_empty());
        assert_eq!(action.harmonic_interior_vertices().len(), 9);
        assert_eq!(action.policy().solver(), plan);
        action.validate_reference(&mesh, &partition).unwrap();
        for report in action.influence_solve_reports() {
            assert_eq!(report.solver_plan(), plan);
            assert_eq!(report.solver_provider(), REFERENCE_LINEAR_SOLVER.provider());
            assert_eq!(
                report.execution(),
                eqiora_solver::ExecutionReport::host_serial()
            );
            assert!(report.completed_iterations() <= plan.maximum_iterations().get());
            assert!(report.true_residual_norm() <= report.residual_target());
        }
        let actual = action.apply([(solid_displacement(), &driver)]).unwrap();
        assert_eq!(
            actual,
            action.apply_jvp([(solid_displacement(), &driver)]).unwrap()
        );
        for (point, value) in mesh.vertices().iter().zip(&actual) {
            for (actual, expected) in value.iter().zip(affine(point)) {
                assert!(
                    (actual - expected).abs() < 2e-13,
                    "{algorithm:?}/{preconditioner:?}: {actual} != {expected}"
                );
            }
        }
        let residual = independently_assemble_fluid_residual(&mesh, &partition, &actual);
        for vertex in action.harmonic_interior_vertices() {
            assert!(
                residual[vertex.index()]
                    .iter()
                    .all(|value| value.abs() < 2e-13)
            );
        }
    }
}

#[derive(Debug)]
struct GeneralOnly;
impl LinearSolverBackend for GeneralOnly {
    fn provider(&self) -> SolverProvider {
        SolverProvider::new(eqiora_solver::BackendId::new("test.general-only"), "0", &[])
    }
    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities::exact(
            REFERENCE_LINEAR_SOLVER
                .capabilities()
                .combinations()
                .iter()
                .filter(|entry| entry.operator_properties == LinearOperatorProperties::General)
                .copied(),
        )
        .unwrap()
    }
    fn solve_with_execution(
        &self,
        _: &LinearProblem<'_>,
        _: SolverPlan,
        _: &dyn ReplicatedLinearExecution,
    ) -> Result<eqiora_solver::LinearSolution, Diagnostic> {
        panic!("unavailable SPD tuple must reject before any influence solve")
    }
}

#[test]
fn general_only_and_unavailable_exact_profiles_reject_before_motion_assembly() {
    // No solved interior: capability errors must precede relation assembly.
    let (mesh, partition) = coarse_partition();
    let bicg = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    let minres_jacobi = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi)
    .with_reduction(ReductionPolicy::Fast);
    let sparse = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    for request in [
        LinearSolveRequest::new(&GeneralOnly, bicg),
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, minres_jacobi),
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, sparse),
        LinearSolveRequest::new(
            &REFERENCE_LINEAR_SOLVER,
            bicg.with_reduction(ReductionPolicy::Fast),
        ),
    ] {
        let error = P1HarmonicMeshMotionAction::<2>::new(
            &mesh,
            &partition,
            motion_policy(request.plan()),
            request,
        )
        .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("exact"));
    }
}

#[test]
fn canonical_capture_coefficient_budget_is_checked_before_allocating() {
    // 3*(1600²) + 2*(1600*100) = 8,000,000 exactly. One extra
    // driver needs 3,200 more coefficients; no large fixture is allocated.
    require_motion_coefficient_budget(1600, 100).unwrap();
    let error = require_motion_coefficient_budget(1600, 101).unwrap_err();
    assert_eq!(error.code(), codes::INVALID_DISCRETIZATION);
    assert!(error.message().contains("8003200"));
    assert!(require_motion_coefficient_budget(usize::MAX, 1).is_err());
    assert!(require_motion_coefficient_budget(1, usize::MAX).is_err());
}
