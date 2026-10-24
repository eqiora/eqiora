//! One accepted output point for spatial linear or finite algebraic Plans.
use super::*;
use crate::nonlinear_execution::NewtonInitialPoint;
use eqiora_ir::{DenseOutputLinearization, LinearizedOutput, LinearizedRelation};

mod output;
pub(super) use output::Output;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Primal {
    Affine(eqiora_solver::SolveReport),
    Linear(Box<ExecutionReceipt>),
    Nonlinear {
        initial: crate::CommonInitialState<'static>,
        iterations: usize,
        linear_solves: Vec<eqiora_solver::SolveReport>,
    },
}

/// One accepted exact Parameter point, its original residual and output partial actions.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonDifferentiationPoint {
    linearization: CommonDifferentiationLinearization,
    primal: Primal,
}

impl CommonDifferentiationPoint {
    pub(super) fn from_primal(
        linearization: CommonDifferentiationLinearization,
        primal: Primal,
    ) -> Self {
        Self {
            linearization,
            primal,
        }
    }

    /// Accepted mathematical point and partial actions, independent of execution records.
    #[must_use]
    pub fn linearization(&self) -> &CommonDifferentiationLinearization {
        &self.linearization
    }

    /// Deployment receipt when the primal used that execution contract.
    /// Native complex and finite solves retain their actual solver reports instead.
    #[must_use]
    pub fn receipt(&self) -> Option<&ExecutionReceipt> {
        match &self.primal {
            Primal::Linear(receipt) => Some(receipt),
            _ => None,
        }
    }

    /// Reports for the linear solves used to establish this accepted point.
    /// A nonlinear point accepted at its seed has no linear updates.
    #[must_use]
    pub fn linear_solves(&self) -> &[eqiora_solver::SolveReport] {
        match &self.primal {
            Primal::Affine(report) => std::slice::from_ref(report),
            Primal::Linear(receipt) => std::slice::from_ref(receipt.report()),
            Primal::Nonlinear { linear_solves, .. } => linear_solves,
        }
    }

    /// Share an admitted linear deployment binding without duplicating receipt ownership.
    pub fn with_shared_receipt_binding(mut self, template: &Self) -> Result<Self, Diagnostic> {
        let (Primal::Linear(receipt), Primal::Linear(source)) =
            (&mut self.primal, &template.primal)
        else {
            return Err(invalid(
                "receipt sharing requires two accepted linear points",
            ));
        };
        **receipt = receipt.as_ref().clone().with_shared_binding(source)?;
        Ok(self)
    }

    /// Exact retained initial State, absent for linear execution.
    #[must_use]
    pub fn nonlinear_initial_state(&self) -> Option<&crate::CommonInitialState<'static>> {
        match &self.primal {
            Primal::Nonlinear { initial, .. } => Some(initial),
            _ => None,
        }
    }

    /// Accepted nonlinear update count; a zero-update solve has no linear primal receipt.
    #[must_use]
    pub fn nonlinear_iterations(&self) -> Option<usize> {
        match &self.primal {
            Primal::Nonlinear { iterations, .. } => Some(*iterations),
            _ => None,
        }
    }
}

/// Original-equation acceptance and partial actions at an exact Parameter point.
///
/// This owns mathematical values, not solver reports or provider handles. A Plan
/// can derive it from a supplied candidate without executing a solve. Acceptance
/// does not authenticate a claimed producer or its iteration history.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonDifferentiationLinearization {
    relation: AssembledLinearizedRelation,
    previous_state_jacobian: Option<eqiora_assembly::CsrMatrix<f64>>,
    state_scalar_domain: eqiora_core::ScalarDomain,
    output: Output,
    residual_target: f64,
    initial: Option<NewtonInitialPoint>,
}

impl CommonDifferentiationLinearization {
    /// Complete coefficient projection when every selected output is a Field.
    /// This describes the mathematical action independently of execution receipts.
    #[must_use]
    pub const fn field_output(&self) -> Option<&FieldCoefficientLinearization> {
        match &self.output {
            Output::Field(output) => Some(output),
            Output::Dense(_) | Output::Sequence { .. } => None,
        }
    }

    pub(super) fn from_spatial(
        relation: AssembledLinearizedRelation,
        state_scalar_domain: eqiora_core::ScalarDomain,
        output: Output,
        residual_target: f64,
        initial: Option<NewtonInitialPoint>,
    ) -> Self {
        Self {
            relation,
            previous_state_jacobian: None,
            state_scalar_domain,
            output,
            residual_target,
            initial,
        }
    }

    pub(super) fn from_finite(
        relation: AssembledLinearizedRelation,
        state_scalar_domain: eqiora_core::ScalarDomain,
        output: DenseOutputLinearization,
        residual_target: f64,
        initial: Option<NewtonInitialPoint>,
    ) -> Self {
        Self {
            relation,
            previous_state_jacobian: None,
            state_scalar_domain,
            output: Output::Dense(output),
            residual_target,
            initial,
        }
    }

    /// Sparse residual derivative in the preceding State's free coordinates.
    /// Present for a fixed-history spatial step. This is a partial action, not
    /// a derivative through earlier accepted steps; Model Parameter products
    /// continue to hold that physical predecessor fixed.
    #[must_use]
    pub const fn previous_state_jacobian(&self) -> Option<&eqiora_assembly::CsrMatrix<f64>> {
        self.previous_state_jacobian.as_ref()
    }

    pub(super) fn with_previous_state_jacobian(
        mut self,
        jacobian: eqiora_assembly::CsrMatrix<f64>,
    ) -> Result<Self, Diagnostic> {
        if jacobian.rows() != self.relation.residual_dimension()
            || jacobian.columns() != self.relation.unknown_dimension()
        {
            return Err(invalid(
                "previous-state Jacobian differs from the accepted coordinate layout",
            ));
        }
        self.previous_state_jacobian = Some(jacobian);
        Ok(self)
    }

    /// Original accepted residual and the paired `R_w` and `R_p` actions.
    #[must_use]
    pub fn relation(&self) -> &AssembledLinearizedRelation {
        &self.relation
    }

    /// Numerical scalar domain retained from this point's state system.
    /// Tangents, Parameters and outputs still use their full real coordinates.
    #[must_use]
    pub const fn state_scalar_domain(&self) -> eqiora_core::ScalarDomain {
        self.state_scalar_domain
    }

    /// Complete selected output coordinates at this point.
    #[must_use]
    pub fn output_values(&self) -> &[f64] {
        self.output.values()
    }

    /// Threshold derived from the original equations and the Plan's controls.
    #[must_use]
    pub const fn residual_target(&self) -> f64 {
        self.residual_target
    }

    /// Nonlinear seed residual; absent for products without a retained seed.
    #[must_use]
    pub const fn initial_residual_norm(&self) -> Option<f64> {
        match self.initial {
            Some(initial) => Some(initial.residual_norm),
            None => None,
        }
    }
    /// Validate only the update-count facts implied by the rederived seed and
    /// accepted point. This does not authenticate a producer or solve history.
    pub fn validate_nonlinear_update_count(&self, iterations: usize) -> Result<(), Diagnostic> {
        self.initial
            .ok_or_else(|| invalid("nonlinear update count requires a nonlinear accepted point"))?
            .validate_updates(self.residual_target, iterations)
    }
}

impl LinearizedOutput<f64> for CommonDifferentiationLinearization {
    fn unknown_dimension(&self) -> usize {
        self.relation.unknown_dimension()
    }
    fn parameter_dimension(&self) -> usize {
        self.relation.parameter_dimension()
    }
    fn output_dimension(&self) -> usize {
        self.output.output_dimension()
    }
    fn primal(&self, values: &mut [f64]) -> Result<(), Diagnostic> {
        self.output.primal(values)
    }
    fn jvp(
        &self,
        unknown: &[f64],
        parameter: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.output.jvp(unknown, parameter, output)
    }
    fn vjp(
        &self,
        output: &[f64],
        unknown: &mut [f64],
        parameter: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.output.vjp(output, unknown, parameter)
    }
}
