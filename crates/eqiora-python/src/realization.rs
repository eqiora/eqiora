//! Python projections of common numerical solve observations.

use eqiora::solver::{
    ConvergenceReason, LinearOperatorOrientation, LinearSolver, PreconditionerPolicy,
    ReductionPolicy,
};
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyclass(
    name = "ConvergenceReason",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
/// Accepted reason for linear-solve convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyConvergenceReason {
    InitialResidualSatisfied,
    ResidualToleranceSatisfied,
}

impl From<ConvergenceReason> for PyConvergenceReason {
    fn from(value: ConvergenceReason) -> Self {
        match value {
            ConvergenceReason::InitialResidualSatisfied => Self::InitialResidualSatisfied,
            ConvergenceReason::ResidualToleranceSatisfied => Self::ResidualToleranceSatisfied,
        }
    }
}

#[pyclass(
    name = "LinearSolveSummary",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
/// Bounded projection of one accepted linear-solve report.
#[derive(Debug, Clone)]
pub(crate) struct PyLinearSolveSummary {
    backend: String,
    adapter: String,
    verification_adapter: String,
    orientation: String,
    algorithm: String,
    preconditioner: String,
    reduction: String,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: usize,
    reason: PyConvergenceReason,
    completed_iterations: usize,
    initial_residual_norm: f64,
    reported_residual_norm: f64,
    true_residual_norm: f64,
    residual_target: f64,
}

impl PyLinearSolveSummary {
    pub(crate) fn from_evidence(report: &eqiora_numerics::CommonSolveEvidence) -> Self {
        Self {
            backend: report.solver().id().to_owned(),
            adapter: report.execution().adapter().to_owned(),
            verification_adapter: report.verification().adapter().to_owned(),
            orientation: match report.orientation() {
                LinearOperatorOrientation::Normal => "normal",
                LinearOperatorOrientation::Transposed => "transposed",
                LinearOperatorOrientation::ConjugateTransposed => "conjugate-transposed",
            }
            .to_owned(),
            algorithm: match report.algorithm() {
                LinearSolver::ConjugateGradient => "conjugate-gradient",
                LinearSolver::MinimumResidual => "minimum-residual",
                LinearSolver::BiConjugateGradientStabilized => "bicgstab",
                LinearSolver::SparseLu => "sparse-lu",
            }
            .to_owned(),
            preconditioner: match report.preconditioner() {
                PreconditionerPolicy::Identity => "identity",
                PreconditionerPolicy::Jacobi => "jacobi",
            }
            .to_owned(),
            reduction: match report.reduction() {
                ReductionPolicy::Reproducible => "reproducible",
                ReductionPolicy::Fast => "fast",
            }
            .to_owned(),
            relative_tolerance: report.relative_tolerance(),
            absolute_tolerance: report.absolute_tolerance(),
            maximum_iterations: report.maximum_iterations(),
            reason: report.reason().into(),
            completed_iterations: report.completed_iterations(),
            initial_residual_norm: report.initial_residual_norm(),
            reported_residual_norm: report.reported_residual_norm(),
            true_residual_norm: report.true_residual_norm(),
            residual_target: report.residual_target(),
        }
    }

    pub(crate) fn from_common_result(
        result: &eqiora_numerics::CommonResult,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<Self> {
        let mut summary = Self::from_evidence(result.solve_evidence(output_index, iteration)?);
        // A retained spatial step may carry a separately replayed original
        // residual. Keep that Result observation distinct from producer history.
        summary.true_residual_norm = result.solve_true_residual_norm(output_index, iteration)?;
        Some(summary)
    }
}

#[pymethods]
impl PyLinearSolveSummary {
    fn __repr__(&self) -> String {
        format!(
            "LinearSolveSummary(algorithm={:?}, preconditioner={:?}, completed_iterations={}, true_residual_norm={:e}, residual_target={:e})",
            self.algorithm,
            self.preconditioner,
            self.completed_iterations,
            self.true_residual_norm,
            self.residual_target,
        )
    }
    #[getter]
    fn backend(&self) -> &str {
        &self.backend
    }
    #[getter]
    fn adapter(&self) -> &str {
        &self.adapter
    }
    #[getter]
    fn verification_adapter(&self) -> &str {
        &self.verification_adapter
    }
    #[getter]
    fn orientation(&self) -> &str {
        &self.orientation
    }
    #[getter]
    fn algorithm(&self) -> &str {
        &self.algorithm
    }
    #[getter]
    fn preconditioner(&self) -> &str {
        &self.preconditioner
    }
    #[getter]
    fn reduction(&self) -> &str {
        &self.reduction
    }
    #[getter]
    const fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }
    #[getter]
    const fn absolute_tolerance(&self) -> f64 {
        self.absolute_tolerance
    }
    #[getter]
    const fn maximum_iterations(&self) -> usize {
        self.maximum_iterations
    }
    #[getter]
    const fn reason(&self) -> PyConvergenceReason {
        self.reason
    }
    #[getter]
    const fn completed_iterations(&self) -> usize {
        self.completed_iterations
    }
    #[getter]
    const fn initial_residual_norm(&self) -> f64 {
        self.initial_residual_norm
    }
    #[getter]
    const fn reported_residual_norm(&self) -> f64 {
        self.reported_residual_norm
    }
    #[getter]
    const fn true_residual_norm(&self) -> f64 {
        self.true_residual_norm
    }
    #[getter]
    const fn residual_target(&self) -> f64 {
        self.residual_target
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyConvergenceReason>()?;
    module.add_class::<PyLinearSolveSummary>()?;
    Ok(())
}
