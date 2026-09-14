//! **eqiora-time** — backend-neutral contracts for time execution.
//!
//! Canonical model meaning remains an activation-controlled network of
//! implicit Relations. A compiler may classify a continuous subsystem into
//! the narrower first-order form `M(t) y_dot = f(t, y)`. General
//! `F(t, y, y_dot) = 0` systems use a distinct residual/JVP problem and must
//! never be disguised as mass-matrix problems to satisfy an adapter.

mod diagnostic;
mod event_linearization;
mod history;
mod lowering;
mod plan;
mod problem;
mod reference_implicit;
mod root_outcome;
mod root_sensitivity_outcome;
pub use root_sensitivity_outcome::TimeRootSensitivityOutcome;
mod root_registration;
pub use root_outcome::TimeRootOutcome;
mod solution;
pub use history::{AcceptedTimeHistory, TimeEventDiscontinuity, TimeHistoryStep};
mod system;

#[cfg(test)]
mod tests;

pub use event_linearization::{
    EventFlowLinearization, EventForwardSensitivity, EventGuardLinearization,
    EventResetLinearization, TransversalEventLinearization,
};
pub use lowering::{
    ConstantDerivativeMatrixProof, DaeVariableKind, GeneralImplicitLoweringProof,
    GeneralImplicitReason, MassMatrixRank, MonomialDerivativeRow, TimeEquationClass,
    TimeLoweringProof,
};
pub use plan::{ForwardSensitivityPlan, TimeMethod, TimePlan};
pub use problem::{
    ForwardSensitivityProblem, ImplicitDaeInitialization, ImplicitDaeProblem,
    InitialConditionPolicy, TimeProblem,
};
pub use reference_implicit::ReferenceImplicitTimeBackend;
pub use root_registration::{
    RegisteredRootProblem, RootActivationGroup, RootProposal, RootRegistrationId,
    RootRegistrationProof,
};
pub use solution::{
    ForwardSensitivitySolution, TimeBackendIdentity, TimeExecutionReport, TimeSolution,
};
pub use system::{
    ImplicitTimeSystem, MassParameterDependence, ParametricTimeSystem, RootFunctions, TimeSystem,
};
