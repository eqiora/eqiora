use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_solver::{
    ConvergenceReason, LinearOperatorOrientation, LinearProblem, LinearSolution,
    PreparedLinearSolver, PreparedLinearStructureIdentity, SolverPlan, accept_linear_solution,
};

use crate::FAER_SOLVER_PROVIDER;
use crate::sparse_lu::{fixed_norm, fixed_residual_norm};
use crate::sparse_lu_factor::{
    SparseLuNumericFactor, SparseLuSymbolicFactor, factor_numeric, factor_symbolic,
    solve_factored_oriented,
};

#[derive(Debug)]
pub(super) struct FaerPreparedSparseLu {
    plan: SolverPlan,
    ready: Option<Ready>,
    #[cfg(test)]
    symbolic_factorizations: usize,
    #[cfg(test)]
    numeric_factorizations: usize,
}

#[derive(Debug)]
struct Ready {
    structure: PreparedLinearStructureIdentity,
    rows: usize,
    columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    coefficient_bits: Vec<u64>,
    symbolic: SparseLuSymbolicFactor,
    numeric: SparseLuNumericFactor,
}

impl FaerPreparedSparseLu {
    pub(super) const fn new(plan: SolverPlan) -> Self {
        Self {
            plan,
            ready: None,
            #[cfg(test)]
            symbolic_factorizations: 0,
            #[cfg(test)]
            numeric_factorizations: 0,
        }
    }

    fn solve_candidate(
        &mut self,
        structure: &PreparedLinearStructureIdentity,
        problem: &LinearProblem<'_>,
    ) -> Result<LinearSolution, Diagnostic> {
        let system = problem.canonical_csr_system().ok_or_else(|| {
            invalid("prepared faer sparse LU requires an exact canonical CSR coefficient source")
        })?;
        if problem.operator().orientation() != LinearOperatorOrientation::Normal
            || problem.operator().rows() != system.rows()
            || problem.operator().columns() != system.columns()
            || problem.properties() != system.properties()
        {
            return Err(invalid(
                "prepared faer sparse LU requires a matching normal canonical CSR action",
            ));
        }

        let initial = problem
            .initial_guess()
            .map_or_else(|| vec![0.0; system.columns()], <[f64]>::to_vec);
        let initial_residual_norm = fixed_residual_norm(problem, &initial)?;
        let residual_target = self
            .plan
            .residual_target(fixed_norm(problem.right_hand_side())?)?;
        if initial_residual_norm <= residual_target {
            return accept_linear_solution(
                problem,
                self.plan,
                FAER_SOLVER_PROVIDER,
                ConvergenceReason::InitialResidualSatisfied,
                0,
                initial_residual_norm,
                initial,
            );
        }

        let same_symbolic = self.ready.as_ref().is_some_and(|ready| {
            ready.structure == *structure
                && ready.rows == system.rows()
                && ready.columns == system.columns()
                && ready.row_offsets == system.row_offsets()
                && ready.column_indices == system.column_indices()
        });
        let coefficient_bits = system
            .values()
            .iter()
            .map(|value| normalized_bits(*value))
            .collect::<Vec<_>>();
        let same_numeric = same_symbolic
            && self
                .ready
                .as_ref()
                .is_some_and(|ready| ready.coefficient_bits == coefficient_bits);

        let candidate_symbolic = if same_symbolic {
            None
        } else {
            #[cfg(test)]
            {
                self.symbolic_factorizations += 1;
            }
            Some(factor_symbolic(system)?)
        };
        let symbolic = candidate_symbolic.as_ref().unwrap_or_else(|| {
            &self
                .ready
                .as_ref()
                .expect("same symbolic identity retains a factor")
                .symbolic
        });
        let candidate_numeric = if same_numeric {
            None
        } else {
            #[cfg(test)]
            {
                self.numeric_factorizations += 1;
            }
            Some(factor_numeric(symbolic, system)?)
        };
        let numeric = candidate_numeric.as_ref().unwrap_or_else(|| {
            &self
                .ready
                .as_ref()
                .expect("same coefficient identity retains a factor")
                .numeric
        });
        let values = solve_factored_oriented(
            symbolic,
            numeric,
            problem.right_hand_side(),
            LinearOperatorOrientation::Normal,
        )?;
        let reported_residual_norm = fixed_residual_norm(problem, &values)?;
        let accepted = accept_linear_solution(
            problem,
            self.plan,
            FAER_SOLVER_PROVIDER,
            ConvergenceReason::ResidualToleranceSatisfied,
            1,
            reported_residual_norm,
            values,
        )?;

        match (candidate_symbolic, candidate_numeric) {
            (Some(symbolic), Some(numeric)) => {
                self.ready = Some(Ready {
                    structure: structure.clone(),
                    rows: system.rows(),
                    columns: system.columns(),
                    row_offsets: system.row_offsets().to_vec(),
                    column_indices: system.column_indices().to_vec(),
                    coefficient_bits,
                    symbolic,
                    numeric,
                });
            }
            (None, Some(numeric)) => {
                let ready = self
                    .ready
                    .as_mut()
                    .expect("numeric rebuild retains an accepted symbolic factor");
                ready.coefficient_bits = coefficient_bits;
                ready.numeric = numeric;
            }
            (None, None) => {}
            (Some(_), None) => unreachable!("new symbolic state requires numeric factorization"),
        }
        Ok(accepted)
    }
}

impl PreparedLinearSolver for FaerPreparedSparseLu {
    fn solve(
        &mut self,
        structure: &PreparedLinearStructureIdentity,
        problem: &LinearProblem<'_>,
    ) -> Result<LinearSolution, Diagnostic> {
        self.solve_candidate(structure, problem)
    }
}

const fn normalized_bits(value: f64) -> u64 {
    if value == 0.0 { 0 } else { value.to_bits() }
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use eqiora_solver::{
        CanonicalCsrSystemView, CompleteCsrStorage, LinearOperatorProperties, LinearSolver,
        PreconditionerPolicy, ReductionPolicy,
    };

    use super::*;

    #[derive(Debug)]
    struct Storage {
        offsets: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<f64>,
        rhs: Vec<f64>,
    }

    impl CompleteCsrStorage for Storage {
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
        fn values(&self) -> &[f64] {
            &self.values
        }
        fn right_hand_side(&self) -> &[f64] {
            &self.rhs
        }
    }

    fn plan() -> SolverPlan {
        SolverPlan::new(LinearSolver::SparseLu, 0.0, 1.0e-12, NonZeroUsize::MIN)
            .unwrap()
            .with_preconditioner(PreconditionerPolicy::Identity)
            .with_reduction(ReductionPolicy::Fast)
    }

    fn system(storage: &Storage) -> CanonicalCsrSystemView {
        CanonicalCsrSystemView::new(storage, LinearOperatorProperties::General).unwrap()
    }

    #[test]
    fn registered_exact_structure_reuse_falsifiers() {
        exact_structure_and_topology_control_symbolic_reuse();
        failed_numeric_candidate_does_not_replace_accepted_factors();
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
        prepared
            .solve(&identity, &systems[0].linear_problem().unwrap())
            .unwrap();
        assert!(
            prepared
                .solve(&identity, &systems[1].linear_problem().unwrap())
                .is_err()
        );
        let retried = prepared
            .solve(&identity, &systems[2].linear_problem().unwrap())
            .unwrap();
        assert_eq!(retried.values(), &[0.5]);
        assert_eq!(prepared.symbolic_factorizations, 1);
    }
}
