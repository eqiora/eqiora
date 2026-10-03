mod bicgstab;
mod hpd;

use eqiora_core::Diagnostic;
use num_complex::Complex64;

use crate::{
    CanonicalCsrSystemView, CompleteCsrStorage, LinearOperatorOrientation,
    LinearOperatorProperties, LinearProblem, LinearSolution, LinearSolverBackend,
    ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION, SolveReport, SolverPlan,
};

use super::{ReferenceLinearSolver, invalid_realization, solve_failed};

impl LinearSolverBackend<Complex64> for ReferenceLinearSolver {
    fn provider(&self) -> crate::SolverProvider {
        Self::provider(self)
    }

    fn capabilities(&self) -> crate::SolverCapabilities {
        Self::capabilities(self)
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_, Complex64>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution<Complex64>, Diagnostic> {
        self.capabilities().require_problem(
            plan,
            problem.scalar_domain(),
            problem.scalar_type(),
            problem.properties(),
        )?;
        if execution.provider() != SERIAL_LINEAR_EXECUTION.provider()
            || execution.report() != SERIAL_LINEAR_EXECUTION.report()
        {
            return Err(invalid_realization(
                "complex reference solves require the host-serial execution profile",
            ));
        }
        execution.require_reduction(plan.reduction())?;
        let source = problem.canonical_csr_system().ok_or_else(|| {
            invalid_realization(
                "complex reference solves require captured canonical complex coefficients",
            )
        })?;
        if problem.properties() == LinearOperatorProperties::HermitianPositiveDefinite {
            hpd::require_positive_pivots(source)?;
        }
        let produced = if plan.algorithm() == crate::LinearSolver::BiConjugateGradientStabilized {
            bicgstab::solve(problem, plan)?
        } else {
            let storage = RealBlockStorage::new(source, problem)?;
            let block = CanonicalCsrSystemView::new(
                &storage,
                LinearOperatorProperties::SymmetricPositiveDefinite,
            )?;
            let mut block_problem = block.linear_problem()?;
            let initial = problem.initial_guess().map(pack);
            if let Some(initial) = &initial {
                block_problem = block_problem.with_initial_guess(initial)?;
            }
            let real = <Self as LinearSolverBackend<f64>>::solve_with_execution(
                self,
                &block_problem,
                plan,
                execution,
            )?;
            bicgstab::Produced {
                values: real
                    .values()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| Complex64::new(pair[0], pair[1]))
                    .collect(),
                reason: real.report().reason(),
                iterations: real.report().completed_iterations(),
                residual: real.report().reported_residual_norm(),
            }
        };
        let values = produced.values;
        // Replay the original typed action, not the lowering used by production.
        let initial_values = problem.initial_guess().map_or_else(
            || vec![Complex64::new(0., 0.); source.columns()],
            <[Complex64]>::to_vec,
        );
        let initial_norm = residual_norm(problem, &initial_values)?;
        let true_norm = residual_norm(problem, &values)?;
        let rhs_norm = norm(problem.right_hand_side())?;
        let report = SolveReport::accepted(
            self.provider(),
            execution.provider(),
            execution.report(),
            problem.operator().orientation(),
            plan,
            produced.reason,
            produced.iterations,
            initial_norm,
            produced.residual,
            true_norm,
            plan.residual_target(rhs_norm)?,
        )?;
        LinearSolution::new(values, report)
    }
}

// For each complex coordinate j, [Re(x_j), Im(x_j)] are adjacent.
// Each a+ib becomes [[a,-b],[b,a]]. No imaginary component is discarded.
struct RealBlockStorage {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
    rhs: Vec<f64>,
}

impl RealBlockStorage {
    fn new(
        source: &CanonicalCsrSystemView<Complex64>,
        problem: &LinearProblem<'_, Complex64>,
    ) -> Result<Self, Diagnostic> {
        let dimension = source
            .rows()
            .checked_mul(2)
            .ok_or_else(|| invalid_realization("complex real-block dimension overflowed"))?;
        let mut rows = vec![Vec::new(); dimension];
        for row in 0..source.rows() {
            for entry in source.row_offsets()[row]..source.row_offsets()[row + 1] {
                let column = source.column_indices()[entry];
                let value = source.values()[entry];
                let (row, column, value) = match problem.operator().orientation() {
                    LinearOperatorOrientation::Normal => (row, column, value),
                    LinearOperatorOrientation::Transposed => (column, row, value),
                    LinearOperatorOrientation::ConjugateTransposed => (column, row, value.conj()),
                };
                rows[2 * row].extend([(2 * column, value.re), (2 * column + 1, -value.im)]);
                rows[2 * row + 1].extend([(2 * column, value.im), (2 * column + 1, value.re)]);
            }
        }
        let mut storage = Self {
            offsets: vec![0],
            columns: Vec::new(),
            values: Vec::new(),
            rhs: pack(problem.right_hand_side()),
        };
        for row in &mut rows {
            row.sort_unstable_by_key(|(column, _)| *column);
            for &(column, value) in row.iter() {
                storage.columns.push(column);
                storage.values.push(value);
            }
            storage.offsets.push(storage.values.len());
        }
        Ok(storage)
    }
}

impl CompleteCsrStorage for RealBlockStorage {
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

fn pack(values: &[Complex64]) -> Vec<f64> {
    values
        .iter()
        .flat_map(|value| [value.re, value.im])
        .collect()
}

fn norm(values: &[Complex64]) -> Result<f64, Diagnostic> {
    let coordinates = pack(values);
    SERIAL_LINEAR_EXECUTION
        .inner_product(crate::FixedOrderInnerProduct::new(
            &coordinates,
            &coordinates,
        )?)
        .map(f64::sqrt)
}

fn residual_norm(
    problem: &LinearProblem<'_, Complex64>,
    values: &[Complex64],
) -> Result<f64, Diagnostic> {
    let mut applied = vec![Complex64::new(0., 0.); values.len()];
    problem.operator().apply(values, &mut applied)?;
    for (applied, rhs) in applied.iter_mut().zip(problem.right_hand_side()) {
        *applied = *rhs - *applied;
    }
    norm(&applied)
}

#[cfg(test)]
mod tests;
