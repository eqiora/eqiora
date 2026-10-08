use std::sync::Arc;

use eqiora_assembly::AssemblyReport;
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, Scalar};
use eqiora_realization::{Target, VectorLayoutKind};
use eqiora_solver::{
    CanonicalCsrSystemView, ExecutionTopology, FixedOrderInnerProduct, LinearOperator,
    LinearOperatorOrientation, LinearOperatorProperties, LinearProblem, LinearSolution,
    ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION, SolverPlan,
};
use num_complex::{Complex64, ComplexFloat};

use crate::discrete_block::{BlockMaterialization, DiscreteBlockSystem};

/// Physics-neutral ownership and acceptance boundary shared by finalized
/// linear spatial problems.
///
/// Reconstruction state deliberately remains in each public wrapper. This
/// core owns only the exact Realization-selected execution contract and the
/// single canonical algebraic source accepted by every backend.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedLinearCore<S> {
    solver: SolverPlan,
    vector_layout: VectorLayoutKind,
    target: Target,
    canonical_system: Arc<CanonicalCsrSystemView<S>>,
    block_materialization: Option<BlockMaterialization>,
}

impl<S: ResidualScalar> FinalizedLinearCore<S> {
    pub(crate) fn new(
        solver: SolverPlan,
        vector_layout: VectorLayoutKind,
        target: Target,
        canonical_system: Arc<CanonicalCsrSystemView<S>>,
    ) -> Self {
        Self {
            solver,
            vector_layout,
            target,
            canonical_system,
            block_materialization: None,
        }
    }

    pub(super) fn operator_properties(&self) -> LinearOperatorProperties {
        self.canonical_system.properties()
    }

    pub(crate) const fn solver_plan(&self) -> SolverPlan {
        self.solver
    }

    pub(crate) const fn vector_layout(&self) -> VectorLayoutKind {
        self.vector_layout
    }

    pub(crate) fn canonical_csr_system_view(&self) -> &CanonicalCsrSystemView<S> {
        self.canonical_system.as_ref()
    }

    pub(crate) fn linear_problem(&self) -> Result<LinearProblem<'_, S>, Diagnostic> {
        self.canonical_system.linear_problem()
    }

    pub(crate) fn validate_solution(&self, solution: &LinearSolution<S>) -> Result<(), Diagnostic> {
        if let Some(materialization) = self.block_materialization {
            materialization.validate(&self.canonical_system)?;
        }
        let report = solution.report();
        if solution.values().len() != self.canonical_system.rows() {
            return Err(invalid_realization(
                "accepted linear solution shape differs from the finalized spatial system",
            ));
        }
        if report.orientation() != LinearOperatorOrientation::Normal {
            return Err(invalid_realization(
                "finalized spatial problem requires a normal-orientation solution",
            ));
        }
        if report.solver_plan() != self.solver {
            return Err(invalid_realization(
                "accepted linear solution used a different SolverPlan than the finalized spatial problem",
            ));
        }
        if report.completed_iterations() > self.solver.maximum_iterations().get() {
            return Err(invalid_realization(
                "accepted linear solution exceeds the finalized SolverPlan iteration limit",
            ));
        }
        validate_producer_topology(
            self.vector_layout,
            self.target,
            report.execution().topology(),
        )?;
        validate_verifier_topology(
            self.vector_layout,
            self.target,
            report.verification().topology(),
        )?;

        let rhs_squared = S::squared_norm(self.canonical_system.right_hand_side())?;
        let expected_target = self.solver.residual_target(rhs_squared.sqrt())?;
        if report.residual_target().to_bits() != expected_target.to_bits() {
            return Err(invalid_realization(
                "accepted linear solution tolerance evidence differs from the finalized SolverPlan",
            ));
        }
        let mut residual = fallible_residual(self.canonical_system.rows())?;
        self.canonical_system
            .apply(solution.values(), &mut residual)?;
        for (value, rhs) in residual
            .iter_mut()
            .zip(self.canonical_system.right_hand_side())
        {
            *value = *rhs - *value;
        }
        let residual_squared = S::squared_norm(&residual)?;
        let exact_residual = residual_squared.sqrt();
        if exact_residual > expected_target {
            return Err(Diagnostic::error(
                codes::NUMERICAL_SOLVE_FAILED,
                format!(
                    "accepted linear solution residual {exact_residual:e} exceeds this finalized spatial problem target {expected_target:e}"
                ),
            ));
        }
        Ok(())
    }

    pub(super) fn into_canonical_system(self) -> Arc<CanonicalCsrSystemView<S>> {
        self.canonical_system
    }
}

// Block execution is still real-only; this specialization does not widen its
// semantic Field admission merely because the finalized CSR can be complex.
impl FinalizedLinearCore<f64> {
    pub(crate) fn with_block_system(
        mut self,
        block_system: &DiscreteBlockSystem,
        assembly_report: &AssemblyReport,
    ) -> Result<Self, Diagnostic> {
        self.block_materialization =
            Some(block_system.bind_materialization(&self.canonical_system, assembly_report)?);
        Ok(self)
    }
}

/// Scalar-specific coordinate projection into the existing fixed-order real
/// reduction. The rest of spatial reacceptance has one implementation.
pub(crate) trait ResidualScalar: Scalar + ComplexFloat<Real = f64> + Sync {
    fn squared_norm(values: &[Self]) -> Result<f64, Diagnostic>;
}

impl ResidualScalar for f64 {
    fn squared_norm(values: &[Self]) -> Result<f64, Diagnostic> {
        SERIAL_LINEAR_EXECUTION.inner_product(FixedOrderInnerProduct::new(values, values)?)
    }
}

impl ResidualScalar for Complex64 {
    fn squared_norm(values: &[Self]) -> Result<f64, Diagnostic> {
        let length = values.len().checked_mul(2).ok_or_else(allocation_error)?;
        let mut coordinates = fallible_residual::<f64>(length)?;
        for (pair, value) in coordinates.as_chunks_mut::<2>().0.iter_mut().zip(values) {
            *pair = [value.re, value.im];
        }
        f64::squared_norm(&coordinates)
    }
}

fn validate_producer_topology(
    vector_layout: VectorLayoutKind,
    target: Target,
    topology: ExecutionTopology,
) -> Result<(), Diagnostic> {
    match (vector_layout, target, topology) {
        (
            VectorLayoutKind::Replicated,
            Target::HostCpu { threads },
            ExecutionTopology::Host { workers },
        ) if workers <= threads => Ok(()),
        (
            VectorLayoutKind::Replicated,
            Target::CudaGpu { device },
            ExecutionTopology::Cuda { device: produced },
        ) if produced == device => Ok(()),
        (
            VectorLayoutKind::Distributed,
            Target::HostCpu { threads },
            ExecutionTopology::Distributed {
                workers_per_partition,
                ..
            },
        ) if threads == std::num::NonZeroUsize::MIN
            && workers_per_partition == std::num::NonZeroUsize::MIN =>
        {
            Ok(())
        }
        (
            VectorLayoutKind::Distributed,
            Target::CudaGpu { device: 0 },
            ExecutionTopology::Distributed {
                workers_per_partition,
                ..
            },
        ) if workers_per_partition == std::num::NonZeroUsize::MIN => Ok(()),
        (VectorLayoutKind::Replicated, Target::HostCpu { .. }, _) => Err(invalid_realization(
            "a replicated host realization requires a host producer within its admitted worker bound",
        )),
        (VectorLayoutKind::Replicated, Target::CudaGpu { .. }, _) => Err(invalid_realization(
            "a replicated CUDA realization requires a producer on the exact resolved CUDA device",
        )),
        (VectorLayoutKind::Distributed, Target::HostCpu { .. }, _) => Err(invalid_realization(
            "a distributed host realization requires a distributed producer with exactly one worker per partition",
        )),
        (VectorLayoutKind::Distributed, Target::CudaGpu { .. }, _) => Err(invalid_realization(
            "a distributed CUDA realization requires a distributed producer with one host control worker per partition and deployment-local device ordinal zero",
        )),
    }
}

fn validate_verifier_topology(
    vector_layout: VectorLayoutKind,
    target: Target,
    topology: ExecutionTopology,
) -> Result<(), Diagnostic> {
    match (vector_layout, target, topology) {
        (
            VectorLayoutKind::Replicated,
            Target::HostCpu { threads },
            ExecutionTopology::Host { workers },
        ) if workers <= threads => Ok(()),
        (
            VectorLayoutKind::Replicated,
            Target::CudaGpu { .. },
            ExecutionTopology::Host { workers },
        )
        | (
            VectorLayoutKind::Distributed,
            Target::HostCpu { .. },
            ExecutionTopology::Host { workers },
        )
        | (
            VectorLayoutKind::Distributed,
            Target::CudaGpu { device: 0 },
            ExecutionTopology::Host { workers },
        ) if workers == std::num::NonZeroUsize::MIN => Ok(()),
        (VectorLayoutKind::Replicated, Target::HostCpu { .. }, _) => Err(invalid_realization(
            "a replicated host realization requires host verification within its admitted worker bound",
        )),
        (VectorLayoutKind::Replicated, Target::CudaGpu { .. }, _) => Err(invalid_realization(
            "a replicated CUDA realization requires independent one-worker host verification",
        )),
        (VectorLayoutKind::Distributed, Target::HostCpu { .. }, _) => Err(invalid_realization(
            "a distributed host realization requires independent one-worker complete-host verification",
        )),
        (VectorLayoutKind::Distributed, Target::CudaGpu { .. }, _) => Err(invalid_realization(
            "a distributed CUDA realization requires independent one-worker complete-host verification and deployment-local device ordinal zero",
        )),
    }
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

fn fallible_residual<S: Scalar + ComplexFloat<Real = f64>>(
    length: usize,
) -> Result<Vec<S>, Diagnostic> {
    let mut residual = Vec::new();
    residual
        .try_reserve_exact(length)
        .map_err(|_| allocation_error())?;
    residual.resize(length, S::zero());
    Ok(residual)
}

fn allocation_error() -> Diagnostic {
    Diagnostic::error(
        codes::NUMERICAL_SOLVE_FAILED,
        "finalized spatial residual allocation exceeds platform capacity",
    )
}

#[cfg(test)]
mod complex_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_capacity_failure_is_a_stable_diagnostic() {
        let diagnostic = fallible_residual::<f64>(usize::MAX).unwrap_err();

        assert_eq!(diagnostic.code(), codes::NUMERICAL_SOLVE_FAILED);
        assert_eq!(
            diagnostic.message(),
            "finalized spatial residual allocation exceeds platform capacity"
        );
    }

    #[test]
    fn distributed_cuda_keeps_distributed_production_and_host_verification_distinct() {
        let distributed = ExecutionTopology::Distributed {
            ranks: std::num::NonZeroUsize::new(4).unwrap(),
            workers_per_partition: std::num::NonZeroUsize::MIN,
        };
        let host = ExecutionTopology::Host {
            workers: std::num::NonZeroUsize::MIN,
        };

        validate_producer_topology(
            VectorLayoutKind::Distributed,
            Target::CudaGpu { device: 0 },
            distributed,
        )
        .unwrap();
        validate_verifier_topology(
            VectorLayoutKind::Distributed,
            Target::CudaGpu { device: 0 },
            host,
        )
        .unwrap();

        let foreign_device = validate_producer_topology(
            VectorLayoutKind::Distributed,
            Target::CudaGpu { device: 1 },
            distributed,
        )
        .unwrap_err();
        assert!(foreign_device.message().contains("ordinal zero"));

        let device_verifier = validate_verifier_topology(
            VectorLayoutKind::Distributed,
            Target::CudaGpu { device: 0 },
            ExecutionTopology::Cuda { device: 0 },
        )
        .unwrap_err();
        assert!(device_verifier.message().contains("complete-host"));
    }
}
