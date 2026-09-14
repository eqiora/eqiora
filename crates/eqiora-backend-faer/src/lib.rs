//! Isolated production host linear-algebra adapter for
//! [`faer`](https://docs.rs/faer).
//!
//! No faer matrix, workspace, or error type crosses this crate's public API.
//! The adapter consumes Eqiora's host-local operator/problem contract and
//! returns Eqiora-owned convergence evidence after independent true-residual
//! verification.

mod prepared_sparse_lu;
mod sparse_lu;
mod sparse_lu_factor;

use std::sync::{Arc, Mutex};

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, ScalarType};
use eqiora_solver::{
    BackendId, ConvergenceReason, DiagonalAvailability, ExecutionReport, LinearOperator,
    LinearOperatorProperties, LinearProblem, LinearSolution, LinearSolver, LinearSolverBackend,
    PreconditionerPolicy, ProviderLibrary, ReductionPolicy, ReplicatedLinearExecution,
    SolverCapabilities, SolverCapability, SolverPlan, SolverProvider, accept_linear_solution,
};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::{
    InitialGuessStatus, LinOp, Precond,
    bicgstab::{BicgParams, bicgstab, bicgstab_scratch},
    conjugate_gradient::{CgParams, conjugate_gradient, conjugate_gradient_scratch},
};
use faer::reborrow::ReborrowMut;
use faer::{Mat, Par, mat::MatMut, mat::MatRef};

/// Exact Eqiora faer adapter package version compiled into this binary.
pub const FAER_ADAPTER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Exact faer dependency version compiled into this adapter.
pub const FAER_VERSION: &str = "0.24.4";

/// Complete identity of the faer solver provider compiled into this binary.
pub const FAER_SOLVER_PROVIDER: SolverProvider = SolverProvider::new(
    BackendId::new("eqiora.faer"),
    FAER_ADAPTER_VERSION,
    &[ProviderLibrary::new("faer", FAER_VERSION)],
);

/// Stateless faer adapter for host-local `f64` CG, BiCGSTAB, and sparse LU.
#[derive(Debug, Clone, Copy, Default)]
pub struct FaerLinearSolver;

// Materialize every initial residual through Eqiora's operator instead of
// relying on a library-specific implicit-zero workspace path.
const EXPLICIT_INITIAL_GUESS: InitialGuessStatus = InitialGuessStatus::MaybeNonZero;

impl LinearSolverBackend for FaerLinearSolver {
    fn provider(&self) -> SolverProvider {
        FAER_SOLVER_PROVIDER
    }

    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities::exact([
            SolverCapability {
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Jacobi,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::BiConjugateGradientStabilized,
                operator_properties: LinearOperatorProperties::General,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::BiConjugateGradientStabilized,
                operator_properties: LinearOperatorProperties::General,
                preconditioner: PreconditionerPolicy::Jacobi,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::SparseLu,
                operator_properties: LinearOperatorProperties::General,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::SparseLu,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::SparseLu,
                operator_properties: LinearOperatorProperties::SymmetricIndefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
        ])
        .expect("faer exact capability set is nonempty")
    }

    fn prepare_linear(
        &self,
        plan: SolverPlan,
    ) -> Result<Option<Box<dyn eqiora_solver::PreparedLinearSolver + '_>>, Diagnostic> {
        if plan.algorithm() != LinearSolver::SparseLu
            || plan.preconditioner() != PreconditionerPolicy::Identity
            || plan.reduction() != ReductionPolicy::Fast
        {
            return Ok(None);
        }
        self.capabilities().require_problem(
            plan,
            ScalarType::F64,
            LinearOperatorProperties::General,
        )?;
        Ok(Some(Box::new(
            prepared_sparse_lu::FaerPreparedSparseLu::new(plan),
        )))
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        let algorithm = format!("{:?}", plan.algorithm());
        let _linear = eqiora_execution::telemetry_span!(linear_solve(
            &algorithm,
            self.provider().id().as_str()
        ))
        .entered();
        if execution.report() != ExecutionReport::host_serial() {
            return Err(invalid_realization(
                "the faer adapter currently admits only direct serial execution",
            ));
        }
        self.capabilities()
            .require_problem(plan, ScalarType::F64, problem.properties())?;
        let inverse_diagonal = inverse_diagonal(problem, plan)?;
        match plan.algorithm() {
            LinearSolver::ConjugateGradient => {
                solve_cg(self.provider(), problem, plan, inverse_diagonal)
            }
            LinearSolver::MinimumResidual => Err(invalid_realization(
                "the faer adapter does not implement MINRES",
            )),
            LinearSolver::BiConjugateGradientStabilized => {
                solve_bicgstab(self.provider(), problem, plan, inverse_diagonal)
            }
            LinearSolver::SparseLu => sparse_lu::solve_sparse_lu(self.provider(), problem, plan),
        }
    }
}

fn solve_cg(
    provider: SolverProvider,
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    inverse_diagonal: Option<Vec<f64>>,
) -> Result<LinearSolution, Diagnostic> {
    let dimension = problem.operator().columns();
    let mut output = initial_matrix(problem);
    let right_hand_side = right_hand_side_matrix(problem);
    let failure = ApplyFailure::default();
    let operator = FaerOperator::new(problem.operator(), failure.clone());
    let preconditioner = FaerPreconditioner::new(dimension, inverse_diagonal);
    let parallelism = Par::Seq;
    let scratch =
        conjugate_gradient_scratch(preconditioner.clone(), operator.clone(), 1, parallelism);
    let mut workspace = MemBuffer::new(scratch);
    let params = CgParams {
        initial_guess: EXPLICIT_INITIAL_GUESS,
        abs_tolerance: plan.absolute_tolerance(),
        rel_tolerance: plan.relative_tolerance(),
        max_iters: plan.maximum_iterations().get(),
        ..Default::default()
    };
    let result = conjugate_gradient(
        output.as_mut(),
        preconditioner,
        operator,
        right_hand_side.as_ref(),
        params,
        |_| {},
        parallelism,
        MemStack::new(&mut workspace),
    );
    failure.take()?;
    let info = result.map_err(|error| solve_failed(format!("faer CG failed: {error:?}")))?;
    accept_linear_solution(
        problem,
        plan,
        provider,
        convergence_reason(info.iter_count),
        info.iter_count,
        info.abs_residual,
        output.col_as_slice(0).to_vec(),
    )
}

fn solve_bicgstab(
    provider: SolverProvider,
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    inverse_diagonal: Option<Vec<f64>>,
) -> Result<LinearSolution, Diagnostic> {
    let dimension = problem.operator().columns();
    let mut output = initial_matrix(problem);
    let right_hand_side = right_hand_side_matrix(problem);
    let failure = ApplyFailure::default();
    let operator = FaerOperator::new(problem.operator(), failure.clone());
    let left_preconditioner = FaerPreconditioner::new(dimension, inverse_diagonal);
    let right_preconditioner = FaerPreconditioner::new(dimension, None);
    let parallelism = Par::Seq;
    let scratch = bicgstab_scratch(
        left_preconditioner.clone(),
        right_preconditioner.clone(),
        operator.clone(),
        1,
        parallelism,
    );
    let mut workspace = MemBuffer::new(scratch);
    let params = BicgParams {
        initial_guess: EXPLICIT_INITIAL_GUESS,
        abs_tolerance: plan.absolute_tolerance(),
        rel_tolerance: plan.relative_tolerance(),
        max_iters: plan.maximum_iterations().get(),
        ..Default::default()
    };
    let result = bicgstab(
        output.as_mut(),
        left_preconditioner,
        right_preconditioner,
        operator,
        right_hand_side.as_ref(),
        params,
        |_| {},
        parallelism,
        MemStack::new(&mut workspace),
    );
    failure.take()?;
    let info = result.map_err(|error| solve_failed(format!("faer BiCGSTAB failed: {error:?}")))?;
    accept_linear_solution(
        problem,
        plan,
        provider,
        convergence_reason(info.iter_count),
        info.iter_count,
        info.abs_residual,
        output.col_as_slice(0).to_vec(),
    )
}

fn inverse_diagonal(
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
) -> Result<Option<Vec<f64>>, Diagnostic> {
    if plan.preconditioner() == PreconditionerPolicy::Identity {
        return Ok(None);
    }
    let mut diagonal = vec![0.0; problem.operator().rows()];
    if problem.operator().diagonal(&mut diagonal)? == DiagonalAvailability::Unavailable {
        return Err(invalid_realization(
            "faer Jacobi preconditioning requires an available operator diagonal",
        ));
    }
    let invalid = diagonal.iter().any(|value| {
        !value.is_finite()
            || *value == 0.0
            || (plan.algorithm() == LinearSolver::ConjugateGradient && *value < 0.0)
    });
    if invalid {
        return Err(solve_failed(
            "Jacobi diagonal must be finite and nonzero, and positive for CG",
        ));
    }
    Ok(Some(
        diagonal.into_iter().map(|value| 1.0 / value).collect(),
    ))
}

fn initial_matrix(problem: &LinearProblem<'_>) -> Mat<f64> {
    Mat::from_fn(problem.operator().columns(), 1, |row, _| {
        problem.initial_guess().map_or(0.0, |values| values[row])
    })
}

fn right_hand_side_matrix(problem: &LinearProblem<'_>) -> Mat<f64> {
    Mat::from_fn(problem.operator().rows(), 1, |row, _| {
        problem.right_hand_side()[row]
    })
}

const fn convergence_reason(iterations: usize) -> ConvergenceReason {
    if iterations == 0 {
        ConvergenceReason::InitialResidualSatisfied
    } else {
        ConvergenceReason::ResidualToleranceSatisfied
    }
}

#[derive(Debug, Clone, Default)]
struct ApplyFailure(Arc<Mutex<Option<Diagnostic>>>);

impl ApplyFailure {
    fn record(&self, diagnostic: Diagnostic) {
        let mut slot = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot.is_none() {
            *slot = Some(diagnostic);
        }
    }

    fn take(&self) -> Result<(), Diagnostic> {
        let mut slot = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match slot.take() {
            Some(diagnostic) => Err(diagnostic),
            None => Ok(()),
        }
    }
}

#[derive(Debug, Clone)]
struct FaerOperator<'a> {
    operator: &'a dyn LinearOperator,
    failure: ApplyFailure,
}

impl<'a> FaerOperator<'a> {
    const fn new(operator: &'a dyn LinearOperator, failure: ApplyFailure) -> Self {
        Self { operator, failure }
    }

    fn apply_real(&self, mut output: MatMut<'_, f64>, input: MatRef<'_, f64>) {
        output.fill(f64::NAN);
        let Some(input) = input.try_as_col_major() else {
            self.failure.record(solve_failed(
                "faer supplied a non-contiguous operator input",
            ));
            return;
        };
        let Some(mut output) = output.try_as_col_major_mut() else {
            self.failure.record(solve_failed(
                "faer supplied a non-contiguous operator output",
            ));
            return;
        };
        for column in 0..input.ncols() {
            let input_column = input.col(column).as_slice();
            let output_column = output.rb_mut().col_mut(column).as_slice_mut();
            if let Err(diagnostic) = self.operator.apply(input_column, output_column) {
                self.failure.record(diagnostic);
                return;
            }
        }
    }
}

impl LinOp<f64> for FaerOperator<'_> {
    fn apply_scratch(&self, _right_hand_side_columns: usize, _parallelism: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.operator.rows()
    }

    fn ncols(&self) -> usize {
        self.operator.columns()
    }

    fn apply(
        &self,
        output: MatMut<'_, f64>,
        input: MatRef<'_, f64>,
        _parallelism: Par,
        _stack: &mut MemStack,
    ) {
        self.apply_real(output, input);
    }

    fn conj_apply(
        &self,
        output: MatMut<'_, f64>,
        input: MatRef<'_, f64>,
        _parallelism: Par,
        _stack: &mut MemStack,
    ) {
        self.apply_real(output, input);
    }
}

#[derive(Debug, Clone)]
struct FaerPreconditioner {
    dimension: usize,
    inverse_diagonal: Option<Vec<f64>>,
}

impl FaerPreconditioner {
    const fn new(dimension: usize, inverse_diagonal: Option<Vec<f64>>) -> Self {
        Self {
            dimension,
            inverse_diagonal,
        }
    }

    fn apply_real(&self, mut output: MatMut<'_, f64>, input: MatRef<'_, f64>) {
        for column in 0..input.ncols() {
            for row in 0..self.dimension {
                output[(row, column)] = input[(row, column)]
                    * self
                        .inverse_diagonal
                        .as_ref()
                        .map_or(1.0, |inverse| inverse[row]);
            }
        }
    }
}

impl LinOp<f64> for FaerPreconditioner {
    fn apply_scratch(&self, _right_hand_side_columns: usize, _parallelism: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.dimension
    }

    fn ncols(&self) -> usize {
        self.dimension
    }

    fn apply(
        &self,
        output: MatMut<'_, f64>,
        input: MatRef<'_, f64>,
        _parallelism: Par,
        _stack: &mut MemStack,
    ) {
        self.apply_real(output, input);
    }

    fn conj_apply(
        &self,
        output: MatMut<'_, f64>,
        input: MatRef<'_, f64>,
        _parallelism: Par,
        _stack: &mut MemStack,
    ) {
        self.apply_real(output, input);
    }
}

impl Precond<f64> for FaerPreconditioner {}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use eqiora_solver::{LinearOperatorProperties, REFERENCE_LINEAR_SOLVER};

    use super::*;

    #[test]
    fn faer_provider_names_the_exact_adapter_and_library_release() {
        FAER_SOLVER_PROVIDER.validate().unwrap();
        assert_eq!(FAER_SOLVER_PROVIDER.id().as_str(), "eqiora.faer");
        assert_eq!(
            FAER_SOLVER_PROVIDER.implementation_version(),
            FAER_ADAPTER_VERSION
        );
        assert_eq!(FAER_SOLVER_PROVIDER.libraries().len(), 1);
        assert_eq!(FAER_SOLVER_PROVIDER.libraries()[0].name(), "faer");
        assert_eq!(FAER_SOLVER_PROVIDER.libraries()[0].version(), FAER_VERSION);
        assert_eq!(FaerLinearSolver.provider(), FAER_SOLVER_PROVIDER);
    }

    #[derive(Debug)]
    struct DenseOperator {
        entries: [[f64; 2]; 2],
    }

    impl LinearOperator for DenseOperator {
        fn rows(&self) -> usize {
            2
        }

        fn columns(&self) -> usize {
            2
        }

        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            if input.len() != 2 || output.len() != 2 {
                return Err(solve_failed("test operator shape mismatch"));
            }
            for (row, output) in output.iter_mut().enumerate() {
                *output = self.entries[row][0] * input[0] + self.entries[row][1] * input[1];
            }
            Ok(())
        }

        fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
            if output.len() != 2 {
                return Err(solve_failed("test diagonal shape mismatch"));
            }
            output.copy_from_slice(&[self.entries[0][0], self.entries[1][1]]);
            Ok(DiagonalAvailability::Available)
        }
    }

    fn plan(algorithm: LinearSolver) -> SolverPlan {
        SolverPlan::new(algorithm, 1.0e-12, 1.0e-14, NonZeroUsize::new(100).unwrap()).unwrap()
    }

    #[test]
    fn faer_operator_adapter_preserves_the_eqiora_action() {
        let operator = DenseOperator {
            entries: [[4.0, 1.0], [2.0, 3.0]],
        };
        let failure = ApplyFailure::default();
        let adapter = FaerOperator::new(&operator, failure.clone());
        let input = Mat::from_fn(2, 1, |row, _| [1.0, -2.0][row]);
        let mut output = Mat::zeros(2, 1);
        adapter.apply(
            output.as_mut(),
            input.as_ref(),
            Par::Seq,
            MemStack::new(&mut MemBuffer::new(StackReq::EMPTY)),
        );
        failure.take().unwrap();
        assert_eq!(output.col_as_slice(0), &[2.0, -4.0]);
    }

    #[test]
    fn faer_preconditioner_adapter_applies_the_inverse_diagonal() {
        let adapter = FaerPreconditioner::new(2, Some(vec![0.25, 0.5]));
        let input = Mat::from_fn(2, 1, |row, _| [4.0, 6.0][row]);
        let mut output = Mat::zeros(2, 1);
        adapter.apply(
            output.as_mut(),
            input.as_ref(),
            Par::Seq,
            MemStack::new(&mut MemBuffer::new(StackReq::EMPTY)),
        );
        assert_eq!(output.col_as_slice(0), &[1.0, 3.0]);
    }

    #[test]
    fn faer_cg_matches_the_independent_reference_oracle() {
        let operator = DenseOperator {
            entries: [[4.0, 1.0], [1.0, 3.0]],
        };
        let problem = LinearProblem::new(
            &operator,
            &[1.0, 2.0],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        let reference = REFERENCE_LINEAR_SOLVER
            .solve(&problem, plan(LinearSolver::ConjugateGradient))
            .unwrap();
        let faer_plan = plan(LinearSolver::ConjugateGradient)
            .with_preconditioner(PreconditionerPolicy::Jacobi)
            .with_reduction(ReductionPolicy::Fast);
        let faer = FaerLinearSolver.solve(&problem, faer_plan).unwrap();
        for (reference, faer) in reference.values().iter().zip(faer.values()) {
            assert!((reference - faer).abs() < 1.0e-13);
        }
        assert_eq!(faer.report().backend().as_str(), "eqiora.faer");
        assert_eq!(faer.report().solver_provider(), FAER_SOLVER_PROVIDER);
        assert_eq!(
            faer.report().execution_provider(),
            eqiora_solver::SERIAL_EXECUTION_PROVIDER
        );
        assert_eq!(
            faer.report().verification_provider(),
            eqiora_solver::SERIAL_EXECUTION_PROVIDER
        );
        assert!(faer.report().true_residual_norm() <= faer.report().residual_target());
        assert_eq!(
            REFERENCE_LINEAR_SOLVER.capabilities().reductions(),
            &std::collections::BTreeSet::from([ReductionPolicy::Reproducible])
        );
    }

    #[test]
    fn faer_bicgstab_solves_a_nonsymmetric_manufactured_system() {
        let operator = DenseOperator {
            entries: [[4.0, 1.0], [2.0, 3.0]],
        };
        let problem =
            LinearProblem::new(&operator, &[2.0, -4.0], LinearOperatorProperties::General).unwrap();
        let plan = plan(LinearSolver::BiConjugateGradientStabilized)
            .with_preconditioner(PreconditionerPolicy::Jacobi)
            .with_reduction(ReductionPolicy::Fast);
        let solution = FaerLinearSolver.solve(&problem, plan).unwrap();
        assert!((solution.values()[0] - 1.0).abs() < 1.0e-12);
        assert!((solution.values()[1] + 2.0).abs() < 1.0e-12);
        assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
    }

    #[test]
    fn faer_does_not_claim_a_reproducible_reduction_it_cannot_control() {
        let operator = DenseOperator {
            entries: [[4.0, 1.0], [1.0, 3.0]],
        };
        let problem = LinearProblem::new(
            &operator,
            &[1.0, 2.0],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        assert_eq!(
            FaerLinearSolver
                .solve(&problem, plan(LinearSolver::ConjugateGradient))
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }

    #[test]
    fn faer_rejects_minres_symmetric_indefinite_at_exact_tuple_preflight() {
        let unsupported = plan(LinearSolver::MinimumResidual)
            .with_preconditioner(PreconditionerPolicy::Identity)
            .with_reduction(ReductionPolicy::Fast);
        let error = FaerLinearSolver
            .capabilities()
            .require_problem(
                unsupported,
                ScalarType::F64,
                LinearOperatorProperties::SymmetricIndefinite,
            )
            .unwrap_err();

        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("MinimumResidual"));
        assert!(error.message().contains("SymmetricIndefinite"));
        assert!(error.message().contains("exact"));
    }

    #[test]
    fn faer_does_not_infer_unverified_cross_product_tuples() {
        let unsupported = plan(LinearSolver::BiConjugateGradientStabilized)
            .with_preconditioner(PreconditionerPolicy::Identity)
            .with_reduction(ReductionPolicy::Fast);
        let error = FaerLinearSolver
            .capabilities()
            .require_problem(
                unsupported,
                ScalarType::F64,
                LinearOperatorProperties::SymmetricPositiveDefinite,
            )
            .unwrap_err();

        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("BiConjugateGradientStabilized"));
        assert!(error.message().contains("SymmetricPositiveDefinite"));
        assert!(error.message().contains("exact"));
    }

    #[test]
    fn operator_diagnostics_cross_the_adapter_without_a_backend_error_leak() {
        #[derive(Debug)]
        struct FailingOperator;

        impl LinearOperator for FailingOperator {
            fn rows(&self) -> usize {
                1
            }

            fn columns(&self) -> usize {
                1
            }

            fn apply(&self, _input: &[f64], _output: &mut [f64]) -> Result<(), Diagnostic> {
                Err(Diagnostic::error(
                    codes::NUMERICAL_SOLVE_FAILED,
                    "intentional operator failure",
                ))
            }
        }

        let problem = LinearProblem::new(
            &FailingOperator,
            &[1.0],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        let error = FaerLinearSolver
            .solve(
                &problem,
                plan(LinearSolver::ConjugateGradient).with_reduction(ReductionPolicy::Fast),
            )
            .unwrap_err();
        assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
        assert_eq!(error.message(), "intentional operator failure");
    }
}
