use std::sync::Arc;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_solver::{
    LinearProblem, LinearSolution, PreparedLinearSolver, PreparedLinearStructureIdentity,
    SolverPlan,
};

use crate::FAER_SOLVER_PROVIDER;
use crate::sparse_lu::{accepted_initial, solve_factored};
use crate::sparse_lu_factor::{
    SparseLuNumericFactor, SparseLuScalar, SparseLuSymbolicFactor, factor_numeric, factor_symbolic,
};

pub(super) fn prepare<S: SparseLuScalar + 'static>(
    plan: SolverPlan,
) -> Result<Option<Box<dyn PreparedLinearSolver<S>>>, Diagnostic> {
    if plan.algorithm() != eqiora_solver::LinearSolver::SparseLu
        || plan.preconditioner() != eqiora_solver::PreconditionerPolicy::Identity
        || plan.reduction() != eqiora_solver::ReductionPolicy::Fast
    {
        return Ok(None);
    }
    crate::FaerLinearSolver.capabilities().require_problem(
        plan,
        S::DOMAIN,
        S::STORAGE,
        eqiora_solver::LinearOperatorProperties::General,
    )?;
    Ok(Some(Box::new(FaerPreparedSparseLu::<S>::new(plan))))
}

#[derive(Debug)]
pub(super) struct FaerPreparedSparseLu<S> {
    plan: SolverPlan,
    ready: Option<Arc<Ready<S>>>,
    tentative: Option<Arc<Ready<S>>>,
    action: Action,
    #[cfg(test)]
    symbolic_factorizations: usize,
    #[cfg(test)]
    numeric_factorizations: usize,
}

#[derive(Debug, PartialEq, Eq)]
enum Action {
    Idle,
    Active,
    Failed,
}

#[derive(Debug)]
struct Ready<S> {
    structure: PreparedLinearStructureIdentity,
    rows: usize,
    columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    coefficients: Vec<S>,
    symbolic: Arc<SparseLuSymbolicFactor>,
    numeric: Arc<SparseLuNumericFactor<S>>,
}

impl<S: SparseLuScalar> FaerPreparedSparseLu<S> {
    pub(super) const fn new(plan: SolverPlan) -> Self {
        Self {
            plan,
            ready: None,
            tentative: None,
            action: Action::Idle,
            #[cfg(test)]
            symbolic_factorizations: 0,
            #[cfg(test)]
            numeric_factorizations: 0,
        }
    }

    fn solve_candidate(
        &mut self,
        structure: &PreparedLinearStructureIdentity,
        problem: &LinearProblem<'_, S>,
    ) -> Result<LinearSolution<S>, Diagnostic> {
        let system = problem.canonical_csr_system().ok_or_else(|| {
            invalid("prepared faer sparse LU requires an exact canonical CSR coefficient source")
        })?;
        if problem.operator().rows() != system.rows()
            || problem.operator().columns() != system.columns()
            || problem.properties() != system.properties()
        {
            return Err(invalid(
                "prepared faer sparse LU requires a matching canonical CSR action",
            ));
        }

        if let Some(initial) = accepted_initial(problem, self.plan)? {
            return Ok(initial);
        }

        let same_symbolic = self.tentative.as_ref().is_some_and(|ready| {
            ready.structure == *structure
                && ready.rows == system.rows()
                && ready.columns == system.columns()
                && ready.row_offsets == system.row_offsets()
                && ready.column_indices == system.column_indices()
        });
        // Captured coefficients are finite and normalize signed zeros. Exact
        // typed equality retains both complex components without another codec.
        let coefficients = system.values().to_vec();
        let same_numeric = same_symbolic
            && self
                .tentative
                .as_ref()
                .is_some_and(|ready| ready.coefficients == coefficients);

        let symbolic = if same_symbolic {
            Arc::clone(
                &self
                    .tentative
                    .as_ref()
                    .expect("matching structure")
                    .symbolic,
            )
        } else {
            #[cfg(test)]
            {
                self.symbolic_factorizations += 1;
            }
            Arc::new(factor_symbolic(system)?)
        };
        let numeric = if same_numeric {
            Arc::clone(
                &self
                    .tentative
                    .as_ref()
                    .expect("matching coefficients")
                    .numeric,
            )
        } else {
            #[cfg(test)]
            {
                self.numeric_factorizations += 1;
            }
            Arc::new(factor_numeric(&symbolic, system)?)
        };
        let accepted = solve_factored(problem, self.plan, &symbolic, &numeric)?;
        if !same_numeric {
            self.tentative = Some(Arc::new(Ready {
                structure: structure.clone(),
                rows: system.rows(),
                columns: system.columns(),
                row_offsets: system.row_offsets().to_vec(),
                column_indices: system.column_indices().to_vec(),
                coefficients,
                symbolic,
                numeric,
            }));
        }
        Ok(accepted)
    }
}

impl<S: SparseLuScalar> PreparedLinearSolver<S> for FaerPreparedSparseLu<S> {
    fn begin_action(&mut self) -> Result<(), Diagnostic> {
        if self.action != Action::Idle {
            return Err(invalid("prepared sparse LU action is already open"));
        }
        self.tentative = self.ready.clone();
        self.action = Action::Active;
        Ok(())
    }

    fn commit_action(&mut self) -> Result<(), Diagnostic> {
        if self.action != Action::Active {
            return Err(invalid(
                "prepared sparse LU commit requires a successful active action",
            ));
        }
        self.ready = self.tentative.take();
        self.action = Action::Idle;
        Ok(())
    }

    fn discard_action(&mut self) -> Result<(), Diagnostic> {
        if self.action == Action::Idle {
            return Err(invalid(
                "prepared sparse LU discard requires an open action",
            ));
        }
        self.tentative = None;
        self.action = Action::Idle;
        Ok(())
    }

    fn solve(
        &mut self,
        structure: &PreparedLinearStructureIdentity,
        problem: &LinearProblem<'_, S>,
    ) -> Result<LinearSolution<S>, Diagnostic> {
        if self.action != Action::Active {
            return Err(invalid(
                "prepared sparse LU solve requires an active action",
            ));
        }
        let algorithm = format!("{:?}", self.plan.algorithm());
        let _linear = eqiora_execution::telemetry_span!(linear_solve(
            &algorithm,
            FAER_SOLVER_PROVIDER.id().as_str()
        ))
        .entered();
        let result = self.solve_candidate(structure, problem);
        if result.is_err() {
            self.action = Action::Failed;
        }
        result
    }
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use eqiora_solver::{
        CanonicalCsrSystemView, CsrStorage, LinearOperatorProperties, LinearSolver,
        PreconditionerPolicy, ReductionPolicy,
    };

    use super::*;

    #[derive(Debug)]
    struct Storage<S> {
        offsets: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<S>,
        rhs: Vec<S>,
    }

    impl<S> CsrStorage<S> for Storage<S> {
        fn rows(&self) -> usize {
            self.rhs.len()
        }
        fn columns(&self) -> usize {
            self.rhs.len()
        }
        fn row_offsets(&self) -> &[usize] {
            &self.offsets
        }
        fn column_indices(&self) -> &[usize] {
            &self.columns
        }
        fn values(&self) -> &[S] {
            &self.values
        }
    }

    impl<S> Storage<S> {
        pub(crate) fn rhs(&self) -> &[S] {
            &self.rhs
        }
    }

    fn plan() -> SolverPlan {
        SolverPlan::new(LinearSolver::SparseLu, 0.0, 1.0e-12, NonZeroUsize::MIN)
            .unwrap()
            .with_preconditioner(PreconditionerPolicy::Identity)
            .with_reduction(ReductionPolicy::Fast)
    }

    fn system<S: SparseLuScalar>(storage: &Storage<S>) -> CanonicalCsrSystemView<S> {
        CanonicalCsrSystemView::new(storage, storage.rhs(), LinearOperatorProperties::General)
            .unwrap()
    }

    #[test]
    fn registered_exact_structure_reuse_falsifiers() {
        exact_structure_and_topology_control_symbolic_reuse();
        failed_numeric_candidate_does_not_replace_accepted_factors();
        action_rollback_preserves_committed_factors_and_rejects_stale_commit();
        malformed_actions_reject_before_mutation_and_empty_actions_commit();
    }

    fn exact_structure_and_topology_control_symbolic_reuse() {
        let a = PreparedLinearStructureIdentity::new(&b"ordering-a"[..]).unwrap();
        let b = PreparedLinearStructureIdentity::new(&b"ordering-b"[..]).unwrap();
        let diagonal = Storage {
            offsets: vec![0, 1, 2],
            columns: vec![0, 1],
            values: vec![2.0, 4.0],
            rhs: vec![2.0, 8.0],
        };
        let changed_values = Storage {
            offsets: vec![0, 1, 2],
            columns: vec![0, 1],
            values: vec![4.0, 8.0],
            rhs: vec![4.0, 16.0],
        };
        let changed_topology = Storage {
            offsets: vec![0, 2, 4],
            columns: vec![0, 1, 0, 1],
            values: vec![3.0, 1.0, 1.0, 3.0],
            rhs: vec![4.0, 4.0],
        };
        let systems = [
            system(&diagonal),
            system(&changed_values),
            system(&changed_topology),
        ];
        let mut prepared = FaerPreparedSparseLu::new(plan());
        prepared.begin_action().unwrap();
        prepared
            .solve(&a, &systems[0].linear_problem().unwrap())
            .unwrap();
        prepared
            .solve(&a, &systems[1].linear_problem().unwrap())
            .unwrap();
        assert_eq!(prepared.symbolic_factorizations, 1);
        assert_eq!(prepared.numeric_factorizations, 2);
        prepared
            .solve(&b, &systems[1].linear_problem().unwrap())
            .unwrap();
        assert_eq!(prepared.symbolic_factorizations, 2);
        prepared
            .solve(&b, &systems[2].linear_problem().unwrap())
            .unwrap();
        assert_eq!(prepared.symbolic_factorizations, 3);
        prepared.commit_action().unwrap();
    }

    fn failed_numeric_candidate_does_not_replace_accepted_factors() {
        let identity = PreparedLinearStructureIdentity::new(&b"stable-structure"[..]).unwrap();
        let accepted = Storage {
            offsets: vec![0, 1],
            columns: vec![0],
            values: vec![4.0],
            rhs: vec![1.0],
        };
        let singular = Storage {
            offsets: vec![0, 1],
            columns: vec![0],
            values: vec![0.0],
            rhs: vec![1.0],
        };
        let same_coefficients_new_rhs = Storage {
            offsets: vec![0, 1],
            columns: vec![0],
            values: vec![4.0],
            rhs: vec![2.0],
        };
        let systems = [
            system(&accepted),
            system(&singular),
            system(&same_coefficients_new_rhs),
        ];
        let mut prepared = FaerPreparedSparseLu::new(plan());
        prepared.begin_action().unwrap();
        prepared
            .solve(&identity, &systems[0].linear_problem().unwrap())
            .unwrap();
        prepared.commit_action().unwrap();
        prepared.begin_action().unwrap();
        assert!(
            prepared
                .solve(&identity, &systems[1].linear_problem().unwrap())
                .is_err()
        );
        prepared.discard_action().unwrap();
        prepared.begin_action().unwrap();
        let retried = prepared
            .solve(&identity, &systems[2].linear_problem().unwrap())
            .unwrap();
        assert_eq!(retried.values(), &[0.5]);
        prepared.commit_action().unwrap();
        assert_eq!(prepared.symbolic_factorizations, 1);
    }
    fn action_rollback_preserves_committed_factors_and_rejects_stale_commit() {
        let identity = PreparedLinearStructureIdentity::new(&b"action-structure"[..]).unwrap();
        let accepted = Storage {
            offsets: vec![0, 1],
            columns: vec![0],
            values: vec![4.0],
            rhs: vec![2.0],
        };
        let changed = Storage {
            offsets: vec![0, 1],
            columns: vec![0],
            values: vec![8.0],
            rhs: vec![2.0],
        };
        let singular = Storage {
            offsets: vec![0, 1],
            columns: vec![0],
            values: vec![0.0],
            rhs: vec![2.0],
        };
        let a = system(&accepted);
        let b = system(&changed);
        let bad = system(&singular);
        let mut prepared = FaerPreparedSparseLu::new(plan());
        prepared.begin_action().unwrap();
        assert_eq!(
            prepared
                .solve(&identity, &a.linear_problem().unwrap())
                .unwrap()
                .values(),
            &[0.5]
        );
        prepared.commit_action().unwrap();
        let committed = Arc::clone(prepared.ready.as_ref().unwrap());
        prepared.begin_action().unwrap();
        assert_eq!(
            prepared
                .solve(&identity, &b.linear_problem().unwrap())
                .unwrap()
                .values(),
            &[0.25]
        );
        assert!(Arc::ptr_eq(prepared.ready.as_ref().unwrap(), &committed));
        assert!(
            prepared
                .solve(&identity, &bad.linear_problem().unwrap())
                .is_err()
        );
        assert!(prepared.commit_action().is_err());
        assert!(
            prepared
                .solve(&identity, &a.linear_problem().unwrap())
                .is_err()
        );
        assert!(prepared.begin_action().is_err());
        prepared.discard_action().unwrap();
        let counts = (
            prepared.symbolic_factorizations,
            prepared.numeric_factorizations,
        );
        prepared.begin_action().unwrap();
        assert_eq!(
            prepared
                .solve(&identity, &a.linear_problem().unwrap())
                .unwrap()
                .values(),
            &[0.5]
        );
        prepared.commit_action().unwrap();
        assert_eq!(
            (
                prepared.symbolic_factorizations,
                prepared.numeric_factorizations
            ),
            counts
        );
        assert!(Arc::ptr_eq(prepared.ready.as_ref().unwrap(), &committed));
        // Caller rejection after a successful solve also discards its numeric update.
        prepared.begin_action().unwrap();
        prepared
            .solve(&identity, &b.linear_problem().unwrap())
            .unwrap();
        prepared.discard_action().unwrap();
        assert!(Arc::ptr_eq(prepared.ready.as_ref().unwrap(), &committed));
        // Rejection before factorization must poison the entire action too.
        prepared.begin_action().unwrap();
        let noncanonical =
            LinearProblem::new(&a, accepted.rhs(), LinearOperatorProperties::General).unwrap();
        assert!(prepared.solve(&identity, &noncanonical).is_err());
        assert!(prepared.commit_action().is_err());
        prepared.discard_action().unwrap();
        assert!(Arc::ptr_eq(prepared.ready.as_ref().unwrap(), &committed));
    }

    fn malformed_actions_reject_before_mutation_and_empty_actions_commit() {
        let identity = PreparedLinearStructureIdentity::new(&b"empty-action"[..]).unwrap();
        let storage = Storage {
            offsets: vec![0, 1],
            columns: vec![0],
            values: vec![4.0],
            rhs: vec![2.0],
        };
        let source = system(&storage);
        let mut prepared = FaerPreparedSparseLu::new(plan());
        assert!(
            prepared
                .solve(&identity, &source.linear_problem().unwrap())
                .is_err()
        );
        assert!(prepared.commit_action().is_err());
        assert!(prepared.discard_action().is_err());
        assert_eq!(prepared.action, Action::Idle);
        assert!(prepared.ready.is_none());
        assert!(prepared.tentative.is_none());
        assert_eq!(prepared.symbolic_factorizations, 0);
        assert_eq!(prepared.numeric_factorizations, 0);
        prepared.begin_action().unwrap();
        assert!(prepared.begin_action().is_err());
        assert_eq!(prepared.action, Action::Active);
        prepared.commit_action().unwrap();
        assert!(prepared.ready.is_none());
        prepared.begin_action().unwrap();
        prepared
            .solve(&identity, &source.linear_problem().unwrap())
            .unwrap();
        prepared.commit_action().unwrap();
        let committed = Arc::clone(prepared.ready.as_ref().unwrap());
        prepared.begin_action().unwrap();
        assert!(prepared.begin_action().is_err());
        assert!(Arc::ptr_eq(
            prepared.tentative.as_ref().unwrap(),
            &committed
        ));
        prepared.commit_action().unwrap();
        assert!(Arc::ptr_eq(prepared.ready.as_ref().unwrap(), &committed));
        assert!(prepared.commit_action().is_err());
        assert!(prepared.discard_action().is_err());
        assert!(Arc::ptr_eq(prepared.ready.as_ref().unwrap(), &committed));
    }

    #[test]
    fn complex_reuse_retains_imaginary_coefficients_and_requested_orientation() {
        use eqiora_solver::{
            LinearOperatorOrientation as Orientation, LinearSolveRequest, LinearSolverBackend,
        };
        use faer::complex::Complex64 as C;
        // A=[[2+i,1-2i],[3+i,4-i]], x=[1+2i,-2+i]. Each RHS below
        // follows directly from integer complex multiplication, without a solver.
        let mut storage = Storage {
            offsets: vec![0, 2, 4],
            columns: vec![0, 1, 0, 1],
            values: vec![
                C::new(2., 1.),
                C::new(1., -2.),
                C::new(3., 1.),
                C::new(4., -1.),
            ],
            rhs: vec![C::new(0., 10.), C::new(-6., 13.)],
        };
        let expected = [C::new(1., 2.), C::new(-2., 1.)];
        let near = |solution: &LinearSolution<C>| {
            for (actual, expected) in solution.values().iter().zip(expected) {
                assert!((*actual - expected).norm() < 1e-12);
            }
            assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
        };
        let identity = PreparedLinearStructureIdentity::new(&b"complex-structure"[..]).unwrap();
        let source = system(&storage);
        // Route the ordinary oriented request into its provider-private prepared
        // session; canonical action provenance stays owned by LinearSolveRequest.
        #[derive(Debug)]
        struct PreparedAdapter {
            session: std::sync::Mutex<FaerPreparedSparseLu<C>>,
            identity: PreparedLinearStructureIdentity,
        }
        impl LinearSolverBackend<C> for PreparedAdapter {
            fn provider(&self) -> eqiora_solver::SolverProvider {
                FAER_SOLVER_PROVIDER
            }
            fn capabilities(&self) -> eqiora_solver::SolverCapabilities {
                crate::FaerLinearSolver.capabilities()
            }
            fn solve_with_execution(
                &self,
                problem: &LinearProblem<'_, C>,
                policy: SolverPlan,
                execution: &dyn eqiora_solver::ReplicatedLinearExecution<C>,
            ) -> Result<LinearSolution<C>, Diagnostic> {
                crate::require_serial_execution(execution, policy)?;
                let mut session = self.session.lock().unwrap();
                assert_eq!(session.plan, policy);
                session.begin_action()?;
                session.solve(&self.identity, problem)
            }
        }
        let adapter = PreparedAdapter {
            session: std::sync::Mutex::new(FaerPreparedSparseLu::new(plan())),
            identity: identity.clone(),
        };
        for (orientation, rhs) in [
            (Orientation::Normal, [C::new(0., 10.), C::new(-6., 13.)]),
            (Orientation::Transposed, [C::new(-7., 6.), C::new(-2., 6.)]),
            (
                Orientation::ConjugateTransposed,
                [C::new(-1., 8.), C::new(-12., 6.)],
            ),
        ] {
            for solution in [
                LinearSolveRequest::new(&crate::FaerLinearSolver, plan())
                    .solve_canonical_oriented(&source, &rhs, orientation)
                    .unwrap(),
                LinearSolveRequest::new(&adapter, plan())
                    .solve_canonical_oriented(&source, &rhs, orientation)
                    .unwrap(),
            ] {
                near(&solution);
                assert_eq!(solution.report().orientation(), orientation);
                assert_eq!(solution.report().solver_provider(), FAER_SOLVER_PROVIDER);
            }
            adapter.session.lock().unwrap().commit_action().unwrap();
        }
        let mut prepared = adapter.session.into_inner().unwrap();
        assert_eq!(prepared.symbolic_factorizations, 1);
        assert_eq!(prepared.numeric_factorizations, 1);
        // Changing only the imaginary part must rebuild numeric factors.
        storage.values[0].im = 2.;
        storage.rhs[0] = C::new(-2., 11.);
        let changed = system(&storage);
        prepared.begin_action().unwrap();
        near(
            &prepared
                .solve(&identity, &changed.linear_problem().unwrap())
                .unwrap(),
        );
        assert_eq!(prepared.symbolic_factorizations, 1);
        prepared.commit_action().unwrap();
        assert_eq!(prepared.numeric_factorizations, 2);
        // Failed numeric candidates cannot displace those accepted factors.
        storage.values.fill(C::new(0., 0.));
        let singular = system(&storage);
        prepared.begin_action().unwrap();
        assert!(
            prepared
                .solve(&identity, &singular.linear_problem().unwrap())
                .is_err()
        );
        prepared.discard_action().unwrap();
        prepared.begin_action().unwrap();
        near(
            &prepared
                .solve(&identity, &changed.linear_problem().unwrap())
                .unwrap(),
        );
        assert_eq!(prepared.symbolic_factorizations, 1);
        prepared.commit_action().unwrap();
        assert_eq!(prepared.numeric_factorizations, 3);
        let mut public = <crate::FaerLinearSolver as LinearSolverBackend<C>>::prepare_linear(
            &crate::FaerLinearSolver,
            plan(),
        )
        .unwrap()
        .unwrap();
        public.begin_action().unwrap();
        near(
            &public
                .solve(&identity, &source.linear_problem().unwrap())
                .unwrap(),
        );
        public.commit_action().unwrap();
        let problem = source
            .linear_problem()
            .unwrap()
            .with_initial_guess(&expected)
            .unwrap();
        let solution = crate::FaerLinearSolver.solve(&problem, plan()).unwrap();
        near(&solution);
        assert_eq!(solution.report().completed_iterations(), 0);
        let mut untouched = FaerPreparedSparseLu::new(plan());
        untouched.begin_action().unwrap();
        near(&untouched.solve(&identity, &problem).unwrap());
        untouched.commit_action().unwrap();
        assert_eq!(untouched.symbolic_factorizations, 0);
        assert_eq!(untouched.numeric_factorizations, 0);
        for unsupported in [
            plan().with_reduction(ReductionPolicy::Reproducible),
            plan().with_preconditioner(PreconditionerPolicy::Jacobi),
        ] {
            assert!(
                crate::FaerLinearSolver
                    .solve(&problem, unsupported)
                    .is_err()
            );
        }
    }
}
