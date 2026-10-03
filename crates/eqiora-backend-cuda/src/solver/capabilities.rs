use eqiora_core::ScalarType;
use eqiora_solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, ReductionPolicy,
    SolverCapabilities, SolverCapability,
};

use super::CudaLinearSolver;

impl CudaLinearSolver {
    /// Exact numerical policies admitted by the first CUDA solver slice.
    #[must_use]
    pub fn capabilities() -> SolverCapabilities {
        SolverCapabilities::exact([
            SolverCapability {
                scalar_domain: eqiora_core::ScalarDomain::Real,
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Jacobi,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                scalar_domain: eqiora_core::ScalarDomain::Real,
                algorithm: LinearSolver::BiConjugateGradientStabilized,
                operator_properties: LinearOperatorProperties::General,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                scalar_domain: eqiora_core::ScalarDomain::Real,
                algorithm: LinearSolver::MinimumResidual,
                operator_properties: LinearOperatorProperties::SymmetricIndefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
        ])
        .expect("CUDA solver exact capability set is nonempty")
    }
}
