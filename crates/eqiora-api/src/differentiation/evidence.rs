//! Compact provenance for accepted primal and derivative occurrences.
use super::*;
use eqiora_numerics::CommonSolveEvidence;
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

impl DerivativeImplementation {
    pub(super) fn from_linearization(value: &CommonDifferentiationLinearization) -> Self {
        if value.field_output().is_some() {
            Self::AnalyticAssembled
        } else {
            Self::OperatorIr
        }
    }
}

/// Whether an occurrence reused the evaluation's accepted linearization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearizationState {
    /// The action publishes the primal that established this evaluation.
    Established,
    /// The derivative action reused the immutable state owned by the evaluation.
    Reused,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "record", deny_unknown_fields)]
pub(super) enum PrimalEvidence {
    Linear(Box<CommonSolveEvidence>),
    Nonlinear {
        initial_state_identity: Option<String>,
        linear_solves: Vec<CommonSolveEvidence>,
        iterations: usize,
        initial_residual_norm: f64,
    },
}

/// Typed in-memory provenance for one primal or derivative occurrence.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiationEvidence {
    identity: DifferentiableProgramIdentity,
    point: DifferentiableValues,
    mode: DifferentiationMode,
    implementation: DerivativeImplementation,
    linearization_state: LinearizationState,
    steps: Vec<DifferentiationStepEvidence>,
}

/// Acceptance and actual solver reports for one step, in execution order.
/// Stationary evaluations contain one such entry. A trajectory retains every
/// accepted step, including steps absent from the selected output schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiationStepEvidence {
    acceptance: Acceptance,
    pub(super) primal: PrimalEvidence,
    derivative_solve: Option<CommonSolveEvidence>,
}

impl DifferentiationEvidence {
    pub(crate) fn same_primal(&self, other: &Self) -> bool {
        self.implementation == other.implementation
            && self.steps.len() == other.steps.len()
            && self
                .steps
                .iter()
                .zip(&other.steps)
                .all(|(a, b)| a.acceptance == b.acceptance && a.primal == b.primal)
    }

    /// Exact program identity.
    #[must_use]
    pub const fn identity(&self) -> &DifferentiableProgramIdentity {
        &self.identity
    }

    /// Exact accepted numerical point used by this occurrence.
    #[must_use]
    pub const fn point(&self) -> &DifferentiableValues {
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

    /// Complete ordered step acceptances, independent of the output schedule.
    #[must_use]
    pub fn steps(&self) -> &[DifferentiationStepEvidence] {
        &self.steps
    }
}

impl DifferentiationStepEvidence {
    /// Exact algebraic state-system identity at the accepted point.
    #[must_use]
    pub const fn state_system(&self) -> CanonicalCsrAgreementFingerprintV1 {
        self.acceptance.state_system
    }

    /// Independently evaluated primal residual norm.
    #[must_use]
    pub const fn primal_residual_norm(&self) -> f64 {
        self.acceptance.primal_residual_norm
    }

    /// Threshold used to admit the linearization.
    #[must_use]
    pub const fn residual_tolerance(&self) -> f64 {
        self.acceptance.residual_tolerance
    }

    /// Solve that established the accepted primal point.
    #[must_use]
    pub fn primal_solve(&self) -> Option<&CommonSolveEvidence> {
        match &self.primal {
            PrimalEvidence::Linear(report) => Some(report),
            PrimalEvidence::Nonlinear { .. } => None,
        }
    }

    /// Actual primal linear solves, including every accepted Newton update.
    #[must_use]
    pub fn primal_solves(&self) -> &[CommonSolveEvidence] {
        match &self.primal {
            PrimalEvidence::Linear(solve) => std::slice::from_ref(solve.as_ref()),
            PrimalEvidence::Nonlinear { linear_solves, .. } => linear_solves,
        }
    }

    /// External nonlinear initial State. Later trajectory steps instead use the
    /// preceding accepted step; absence alone does not identify a linear primal.
    #[must_use]
    pub fn nonlinear_initial_state_identity(&self) -> Option<&str> {
        match &self.primal {
            PrimalEvidence::Nonlinear {
                initial_state_identity,
                ..
            } => initial_state_identity.as_deref(),
            PrimalEvidence::Linear(_) => None,
        }
    }

    /// Accepted nonlinear update count, including zero-update acceptance.
    #[must_use]
    pub const fn nonlinear_iterations(&self) -> Option<usize> {
        match &self.primal {
            PrimalEvidence::Nonlinear { iterations, .. } => Some(*iterations),
            PrimalEvidence::Linear(_) => None,
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
            PrimalEvidence::Linear(_) => None,
        }
    }

    /// Canonical identity of the accepted unknown coordinates paired with the Jacobian.
    #[must_use]
    pub const fn accepted_state(&self) -> AcceptedOutputFingerprintV1 {
        self.acceptance.accepted_state
    }

    /// Canonical identity of the selected output coordinates in signature order.
    #[must_use]
    pub const fn accepted_output(&self) -> AcceptedOutputFingerprintV1 {
        self.acceptance.accepted_output
    }

    /// Normal or transposed derivative solve, absent for primal publication.
    #[must_use]
    pub const fn derivative_solve(&self) -> Option<&CommonSolveEvidence> {
        self.derivative_solve.as_ref()
    }
}

impl DifferentiableEvaluation {
    pub(crate) fn primal_evidence(&self) -> DifferentiationEvidence {
        self.evidence(
            DifferentiationMode::Primal,
            Vec::new(),
            LinearizationState::Established,
        )
    }

    pub(super) fn evidence(
        &self,
        mode: DifferentiationMode,
        derivative_solves: Vec<SolveReport>,
        linearization_state: LinearizationState,
    ) -> DifferentiationEvidence {
        assert!(
            derivative_solves.is_empty() || derivative_solves.len() == self.native.points().len()
        );
        let steps = self
            .native
            .points()
            .iter()
            .zip(&self.acceptance)
            .enumerate()
            .map(|(index, (point, acceptance))| {
                let primal = match point.nonlinear_iterations() {
                    None => PrimalEvidence::Linear(Box::new(CommonSolveEvidence::from_report(
                        point
                            .linear_solves()
                            .first()
                            .expect("accepted linear primal solve"),
                    ))),
                    Some(iterations) => PrimalEvidence::Nonlinear {
                        initial_state_identity: point
                            .nonlinear_initial_state()
                            .map(|state| state.identity().to_owned()),
                        linear_solves: point
                            .linear_solves()
                            .iter()
                            .map(CommonSolveEvidence::from_report)
                            .collect(),
                        iterations,
                        initial_residual_norm: point
                            .linearization()
                            .initial_residual_norm()
                            .expect("accepted nonlinear initial residual"),
                    },
                };
                DifferentiationStepEvidence {
                    acceptance: acceptance.clone(),
                    primal,
                    derivative_solve: derivative_solves
                        .get(index)
                        .map(CommonSolveEvidence::from_report),
                }
            })
            .collect();
        DifferentiationEvidence {
            identity: self.identity.clone(),
            point: self.point.clone(),
            mode,
            implementation: DerivativeImplementation::from_linearization(
                self.native.points()[0].linearization(),
            ),
            linearization_state,
            steps,
        }
    }
}

/// Mathematical admission shared by live evaluations and read-only records.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Acceptance {
    primal_residual_norm: f64,
    pub(super) residual_tolerance: f64,
    state_system: CanonicalCsrAgreementFingerprintV1,
    accepted_state: AcceptedOutputFingerprintV1,
    accepted_output: AcceptedOutputFingerprintV1,
}

impl Acceptance {
    pub(super) fn new(
        identity: &DifferentiableProgramIdentity,
        linearization: &CommonDifferentiationLinearization,
    ) -> Result<Self, Diagnostic> {
        let relation = linearization.relation();
        if identity.input_dimension() != relation.parameter_dimension()
            || identity
                .outputs()
                .iter()
                .map(|value| {
                    value.coordinate_count() / value.sample_times_s().map_or(1, <[f64]>::len)
                })
                .sum::<usize>()
                != linearization.output_dimension()
        {
            return Err(invalid(
                "accepted linearization differs from the Model/Plan value signatures",
            ));
        }
        let accepted = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
            relation,
            linearization,
            relation.state_jacobian(),
            linearization.state_scalar_domain(),
            linearization.residual_target(),
        )?;
        Ok(Self {
            primal_residual_norm: accepted.relation().primal_residual_norm(),
            residual_tolerance: linearization.residual_target(),
            state_system: relation.state_jacobian().agreement_fingerprint(),
            accepted_state: AcceptedOutputFingerprintV1::from_values(relation.accepted_unknowns())?,
            accepted_output: AcceptedOutputFingerprintV1::from_values(
                linearization.output_values(),
            )?,
        })
    }
}

impl DifferentiationEvidence {
    pub(super) fn from_record(
        identity: DifferentiableProgramIdentity,
        point: DifferentiableValues,
        implementation: DerivativeImplementation,
        acceptance: Vec<Acceptance>,
        primal: Vec<PrimalEvidence>,
    ) -> Self {
        Self {
            identity,
            point,
            implementation,
            steps: acceptance
                .into_iter()
                .zip(primal)
                .map(|(acceptance, primal)| DifferentiationStepEvidence {
                    acceptance,
                    primal,
                    derivative_solve: None,
                })
                .collect(),
            mode: DifferentiationMode::Primal,
            linearization_state: LinearizationState::Established,
        }
    }
}
