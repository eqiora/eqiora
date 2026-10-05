use std::sync::{Arc, Mutex};

use eqiora_solver::{PreparedLinearSolver, PreparedLinearStructureIdentity};

use super::*;

#[derive(Debug, Default)]
struct Observations {
    preparations: usize,
    solves: usize,
    structure: Option<PreparedLinearStructureIdentity>,
}

#[derive(Debug, Default)]
struct PreparingSolver(Arc<Mutex<Observations>>);

impl LinearSolverBackend for PreparingSolver {
    fn provider(&self) -> SolverProvider {
        DenseGeneralSolver.provider()
    }

    fn capabilities(&self) -> SolverCapabilities {
        DenseGeneralSolver.capabilities()
    }

    fn prepare_linear(
        &self,
        plan: SolverPlan,
    ) -> Result<Option<Box<dyn PreparedLinearSolver>>, Diagnostic> {
        self.0.lock().unwrap().preparations += 1;
        Ok(Some(Box::new(Prepared {
            observations: self.0.clone(),
            plan,
        })))
    }

    fn solve_with_execution(
        &self,
        _: &LinearProblem<'_>,
        _: SolverPlan,
        _: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        panic!("prepared occurrence must not call the cold backend")
    }
}

#[derive(Debug)]
struct Prepared {
    observations: Arc<Mutex<Observations>>,
    plan: SolverPlan,
}

impl PreparedLinearSolver for Prepared {
    fn solve(
        &mut self,
        structure: &PreparedLinearStructureIdentity,
        problem: &LinearProblem<'_>,
    ) -> Result<LinearSolution, Diagnostic> {
        assert!(problem.canonical_csr_system().is_some());
        let mut observations = self.observations.lock().unwrap();
        if let Some(expected) = &observations.structure {
            assert_eq!(expected, structure);
        } else {
            observations.structure = Some(structure.clone());
        }
        observations.solves += 1;
        DenseGeneralSolver.solve(problem, self.plan)
    }
}

#[test]
fn one_provider_preparation_serves_all_newton_actions_with_cold_equivalence() {
    let fixture = fixture();
    let quadrature = triangle_duffy_gauss_legendre(5).unwrap();
    let prepared_solver = PreparingSolver::default();
    let run = |solver: &dyn LinearSolverBackend| {
        advance_simplicial_ale_fsi_2d(
            &fixture.mesh,
            &fixture.partition,
            &fixture.boundary,
            &fixture.motion,
            fixture.initial.clone(),
            NonZeroStepCount::new(NonZeroUsize::new(2).unwrap()),
            fixture.plan.clone(),
            &quadrature,
            solver,
            &fixture.layout,
        )
        .unwrap()
    };
    let warm = run(&prepared_solver);
    let cold = run(&DenseGeneralSolver);
    assert_eq!(warm.states(), cold.states());
    let observed = prepared_solver.0.lock().unwrap();
    assert_eq!(observed.preparations, 1);
    assert_eq!(
        observed.solves,
        warm.steps()
            .iter()
            .map(|step| step.nonlinear_linear_solves().len())
            .sum::<usize>()
    );
    assert!(observed.solves >= 2);
    assert!(observed.structure.is_some());
}
