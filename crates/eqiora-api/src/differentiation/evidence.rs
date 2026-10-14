//! Compact provenance for accepted primal and derivative occurrences.
use super::*;
use eqiora_solver::{CanonicalCsrAgreementFingerprintV1, SolveReport};

/// Action represented by one differentiation result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentiationMode {
    /// Accepted selected output values.
    Primal,
    /// Forward output Jacobian action.
    Jvp,
    /// Reverse output Jacobian action.
    Vjp,
}

/// Lowered derivative implementation used by this bounded slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeImplementation {
    /// Analytically assembled `R_w`, `R_p`, `O_w`, and `O_p` actions.
    AnalyticAssembled,
    /// Exact original-expression Operator IR partial actions.
    OperatorIr,
}

/// Whether an occurrence reused the evaluation's accepted linearization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearizationState {
    /// The action publishes the primal that established this evaluation.
    Established,
    /// The derivative action reused the immutable state owned by the evaluation.
    Reused,
}

#[derive(Debug, Clone, PartialEq)]
enum PrimalEvidence {
    Affine(Box<SolveReport>),
    Linear(Box<ExecutionReceipt>),
    Nonlinear {
        initial_state_identity: String,
        iterations: usize,
        initial_residual_norm: f64,
    },
}

/// Typed in-memory provenance for one primal or derivative occurrence.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiationEvidence {
    identity: DifferentiableProgramIdentity,
    point: DifferentiableParameterPoint,
    mode: DifferentiationMode,
    implementation: DerivativeImplementation,
    linearization_state: LinearizationState,
    primal_residual_norm: f64,
    residual_tolerance: f64,
    state_system: CanonicalCsrAgreementFingerprintV1,
    accepted_state: AcceptedOutputFingerprintV1,
    accepted_output: AcceptedOutputFingerprintV1,
    primal: PrimalEvidence,
    derivative_solve: Option<SolveReport>,
}

impl DifferentiationEvidence {
    /// Exact program identity.
    #[must_use]
    pub const fn identity(&self) -> &DifferentiableProgramIdentity {
        &self.identity
    }

    /// Exact accepted numerical point used by this occurrence.
    #[must_use]
    pub const fn point(&self) -> &DifferentiableParameterPoint {
        &self.point
    }

    /// Primal, JVP, or VJP action.
    #[must_use]
    pub const fn mode(&self) -> DifferentiationMode {
        self.mode
    }

    /// Source of derivative actions.
    #[must_use]
    pub const fn implementation(&self) -> DerivativeImplementation {
        self.implementation
    }

    /// Primal establishment or derivative reuse of the accepted state.
    #[must_use]
    pub const fn linearization_state(&self) -> LinearizationState {
        self.linearization_state
    }

    /// Exact algebraic state-system identity at the accepted point.
    #[must_use]
    pub const fn state_system(&self) -> CanonicalCsrAgreementFingerprintV1 {
        self.state_system
    }

    /// Independently evaluated primal residual norm.
    #[must_use]
    pub const fn primal_residual_norm(&self) -> f64 {
        self.primal_residual_norm
    }

    /// Threshold used to admit the linearization.
    #[must_use]
    pub const fn residual_tolerance(&self) -> f64 {
        self.residual_tolerance
    }

    /// Solve that established the accepted primal point.
    #[must_use]
    pub fn primal_solve(&self) -> Option<&SolveReport> {
        match &self.primal {
            PrimalEvidence::Affine(report) => Some(report),
            _ => self.receipt().map(ExecutionReceipt::report),
        }
    }

    /// Exact deployment, operator, plan, output, and accepted-solve linkage.
    #[must_use]
    pub fn receipt(&self) -> Option<&ExecutionReceipt> {
        match &self.primal {
            PrimalEvidence::Linear(receipt) => Some(receipt),
            PrimalEvidence::Nonlinear { .. } | PrimalEvidence::Affine(_) => None,
        }
    }

    /// Exact nonlinear initial State, absent for a linear primal.
    #[must_use]
    pub fn nonlinear_initial_state_identity(&self) -> Option<&str> {
        match &self.primal {
            PrimalEvidence::Nonlinear {
                initial_state_identity,
                ..
            } => Some(initial_state_identity),
            PrimalEvidence::Linear(_) | PrimalEvidence::Affine(_) => None,
        }
    }

    /// Accepted nonlinear update count, including zero-update acceptance.
    #[must_use]
    pub const fn nonlinear_iterations(&self) -> Option<usize> {
        match &self.primal {
            PrimalEvidence::Nonlinear { iterations, .. } => Some(*iterations),
            PrimalEvidence::Linear(_) | PrimalEvidence::Affine(_) => None,
        }
    }

    /// Original nonlinear residual norm at the bound seed and Parameter point.
    #[must_use]
    pub const fn nonlinear_initial_residual_norm(&self) -> Option<f64> {
        match &self.primal {
            PrimalEvidence::Nonlinear {
                initial_residual_norm,
                ..
            } => Some(*initial_residual_norm),
            PrimalEvidence::Linear(_) | PrimalEvidence::Affine(_) => None,
        }
    }

    /// Canonical identity of the accepted unknown coordinates paired with the Jacobian.
    #[must_use]
    pub const fn accepted_state(&self) -> AcceptedOutputFingerprintV1 {
        self.accepted_state
    }

    /// Canonical identity of the selected output coordinates in signature order.
    #[must_use]
    pub const fn accepted_output(&self) -> AcceptedOutputFingerprintV1 {
        self.accepted_output
    }

    /// Normal or transposed derivative solve, absent for primal publication.
    #[must_use]
    pub const fn derivative_solve(&self) -> Option<&SolveReport> {
        self.derivative_solve.as_ref()
    }
}

impl DifferentiableEvaluation {
    pub(crate) fn primal_evidence(&self) -> DifferentiationEvidence {
        self.evidence(
            DifferentiationMode::Primal,
            None,
            LinearizationState::Established,
        )
    }

    pub(super) fn evidence(
        &self,
        mode: DifferentiationMode,
        derivative_solve: Option<SolveReport>,
        linearization_state: LinearizationState,
    ) -> DifferentiationEvidence {
        DifferentiationEvidence {
            identity: self.identity.clone(),
            point: self.point.clone(),
            mode,
            implementation: if self.native.receipt().is_some() {
                DerivativeImplementation::AnalyticAssembled
            } else {
                DerivativeImplementation::OperatorIr
            },
            linearization_state,
            primal_residual_norm: self.primal_residual_norm,
            residual_tolerance: self.residual_tolerance,
            state_system: self
                .native
                .relation()
                .state_jacobian()
                .agreement_fingerprint(),
            accepted_state: self.accepted_state,
            accepted_output: self.accepted_output,
            primal: match self.native.receipt() {
                Some(receipt) => PrimalEvidence::Linear(Box::new(receipt.clone())),
                None if self.native.nonlinear_initial_state().is_none() => {
                    PrimalEvidence::Affine(Box::new(
                        self.native
                            .linear_solves()
                            .first()
                            .expect("accepted affine solve")
                            .clone(),
                    ))
                }
                None => PrimalEvidence::Nonlinear {
                    initial_state_identity: self
                        .native
                        .nonlinear_initial_state()
                        .expect("accepted nonlinear seed")
                        .identity()
                        .to_owned(),
                    iterations: self
                        .native
                        .nonlinear_iterations()
                        .expect("accepted nonlinear count"),
                    initial_residual_norm: self
                        .native
                        .nonlinear_initial_residual_norm()
                        .expect("accepted nonlinear initial residual"),
                },
            },
            derivative_solve,
        }
    }
}
