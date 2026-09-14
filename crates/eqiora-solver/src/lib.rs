//! **eqiora-solver** — backend-neutral linear-solver contracts.
//!
//! This crate owns the single solver-plan vocabulary, host-local `f64`
//! operator seam, capability negotiation, convergence evidence, and a small
//! deterministic reference oracle. Production library and device adapters
//! live in dedicated L3 crates.

mod backend;
mod csr;
mod execution;
mod operator;
mod plan;
mod planning;
mod provider;
mod reference;
mod report;
mod structure;

pub use backend::{
    BackendId, LinearSolveRequest, LinearSolverBackend, SolverCapabilities, SolverCapability,
};
pub use csr::{CanonicalCsrAgreementFingerprintV1, CanonicalCsrSystemView, CompleteCsrStorage};
use eqiora_core::ScalarType;
pub use execution::{
    FixedOrderInnerProduct, REPRODUCIBLE_INNER_PRODUCT_CHUNK_LENGTH, ReplicatedLinearExecution,
    SERIAL_EXECUTION_PROVIDER, SERIAL_LINEAR_EXECUTION,
};
pub use operator::{
    DiagonalAvailability, LinearOperator, LinearOperatorOrientation, LinearOperatorProperties,
    LinearProblem, RowLinearAction, TransposeLinearOperator, Transposed,
};
pub use plan::{LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverPlan};
pub use planning::{
    HostSerialSolverProfile, ResolvedHostSerialSolverPlan, SolverPlanningObjective,
    plan_host_serial_solver_v2,
};
pub use provider::{ExecutionProvider, ProviderLibrary, SolverProvider};
pub use reference::{REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER};
pub use report::{
    ConvergenceReason, ExecutionId, ExecutionReport, ExecutionTopology, LinearAcceptanceWorkspace,
    LinearSolution, SolveReport, accept_linear_solution, accept_linear_solution_with_execution,
    accept_linear_solution_with_verifier, accept_linear_solution_with_verifier_in,
};
pub use structure::{AlgebraicBlock, AlgebraicConstraint, AlgebraicStructure};
