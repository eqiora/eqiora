use eqiora::solver::{
    LinearProblem, LinearSolution, LinearSolverBackend, ReplicatedLinearExecution, SolverProvider,
};

use super::*;

/// Same admitted provider and numerical algorithm, without a prepared session.
#[derive(Debug)]
struct ColdFaer;

impl LinearSolverBackend for ColdFaer {
    fn provider(&self) -> SolverProvider {
        FaerLinearSolver.provider()
    }

    fn capabilities(&self) -> SolverCapabilities {
        FaerLinearSolver.capabilities()
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, eqiora::Diagnostic> {
        FaerLinearSolver.solve_with_execution(problem, plan, execution)
    }
}

#[test]
fn prepared_sparse_lu_closes_the_ordinary_ale_path_and_matches_cold_execution() {
    let document = ModelDocument::compile("prepared-ale.eqi", DIRECT_SOURCE).unwrap();
    let canonical = lower_ale_fsi_cartesian_2d(document.program()).unwrap();
    let fixture = Fixture::new(document, canonical);
    let time_step = 0.01;
    let plan = SolverPlan::new(LinearSolver::SparseLu, 1.0e-9, 1.0e-11, NonZeroUsize::MIN)
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(ReductionPolicy::Fast);
    assert!(FaerLinearSolver.prepare_linear(plan).unwrap().is_some());
    let resolved = resolve_fixed_topology_ale_coupled(
        &FixedTopologyAleCoupledRealizationRequest::explicit(
            fixture.canonical.model(),
            SemanticRevision::new(fixture.canonical.semantic_revision()),
            RealizationRevision::new(1),
            realization_plan(&fixture.canonical, fixture.mesh_reference, time_step, plan),
        ),
        fixed_topology_ale_fsi_requirements_2d(&fixture.canonical),
        &capabilities(),
    )
    .unwrap();
    let finalized = finalize_resolved_fixed_topology_ale_fsi_2d(
        &fixture.canonical,
        &resolved,
        fixture.mesh_reference,
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        fixture.initial_physical(time_step),
        &FaerLinearSolver,
    )
    .unwrap();
    let steps = NonZeroStepCount::new(NonZeroUsize::new(2).unwrap());
    let warm = finalized.clone().solve(steps, &FaerLinearSolver).unwrap();
    let cold = finalized.clone().solve(steps, &ColdFaer).unwrap();
    assert_eq!(warm.states(), cold.states());
    assert!(
        warm.steps()
            .iter()
            .all(|step| !step.nonlinear_linear_solves().is_empty())
    );
    assert_harmonic_geometry_replays(&fixture, finalized.motion(), &warm);
    assert_consecutive_geometry_and_evidence(&fixture, &warm, time_step, LinearSolver::SparseLu);
}
