//! One accepted scalar-output point for spatial linear or finite nonlinear Plans.
use super::*;
use crate::finite_constraints::FiniteNonlinearSolution;
use eqiora_ir::{LinearizedOutput, LinearizedRelation, ScalarObjectiveLinearization};

#[derive(Debug, Clone, PartialEq)]
enum Output {
    Field(CartesianScalarFieldLinearization),
    Observable(ScalarObjectiveLinearization),
}

#[derive(Debug, Clone, PartialEq)]
enum Primal {
    Linear(Box<ExecutionReceipt>),
    Nonlinear {
        initial: CommonAlgebraicState,
        solution: FiniteNonlinearSolution,
    },
}

/// One accepted exact Parameter point, its original residual and output partial actions.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonScalarDifferentiationPoint {
    relation: AssembledLinearizedRelation,
    output: Output,
    primal: Primal,
}

impl CommonScalarDifferentiationPoint {
    pub(super) fn from_linear(
        relation: AssembledLinearizedRelation,
        output: CartesianScalarFieldLinearization,
        receipt: ExecutionReceipt,
    ) -> Self {
        Self {
            relation,
            output: Output::Field(output),
            primal: Primal::Linear(Box::new(receipt)),
        }
    }

    pub(super) fn from_nonlinear(
        relation: AssembledLinearizedRelation,
        output: ScalarObjectiveLinearization,
        initial: CommonAlgebraicState,
        solution: FiniteNonlinearSolution,
    ) -> Self {
        Self {
            relation,
            output: Output::Observable(output),
            primal: Primal::Nonlinear { initial, solution },
        }
    }

    /// Original accepted residual and the paired `R_w` and `R_p` actions.
    #[must_use]
    pub fn relation(&self) -> &AssembledLinearizedRelation {
        &self.relation
    }

    /// Complete selected Field or scalar Observable value at this point.
    #[must_use]
    pub fn output_values(&self) -> Vec<f64> {
        match &self.output {
            Output::Field(output) => output.values().to_vec(),
            Output::Observable(output) => vec![output.value()],
        }
    }

    /// Linear primal receipt, absent when nonlinear acceptance established the point.
    #[must_use]
    pub fn receipt(&self) -> Option<&ExecutionReceipt> {
        match &self.primal {
            Primal::Linear(receipt) => Some(receipt),
            _ => None,
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

    /// Exact finite initial State, absent for spatial linear execution.
    #[must_use]
    pub fn nonlinear_initial_state(&self) -> Option<&CommonAlgebraicState> {
        match &self.primal {
            Primal::Nonlinear { initial, .. } => Some(initial),
            _ => None,
        }
    }

    /// Accepted nonlinear update count; a zero-update solve has no linear primal receipt.
    #[must_use]
    pub fn nonlinear_iterations(&self) -> Option<usize> {
        match &self.primal {
            Primal::Nonlinear { solution, .. } => Some(solution.iterations),
            _ => None,
        }
    }

    /// Original initial residual at this evaluation's exact Parameter point.
    #[must_use]
    pub fn nonlinear_initial_residual_norm(&self) -> Option<f64> {
        match &self.primal {
            Primal::Nonlinear { solution, .. } => Some(solution.initial_residual_norm),
            _ => None,
        }
    }

    /// Residual target independently retained by the primal acceptance owner.
    #[must_use]
    pub fn residual_target(&self) -> f64 {
        match &self.primal {
            Primal::Linear(receipt) => receipt.report().residual_target(),
            Primal::Nonlinear { solution, .. } => solution.assessment.residual_target(),
        }
    }
}

impl LinearizedOutput<f64> for CommonScalarDifferentiationPoint {
    fn unknown_dimension(&self) -> usize {
        self.relation.unknown_dimension()
    }
    fn parameter_dimension(&self) -> usize {
        self.relation.parameter_dimension()
    }
    fn output_dimension(&self) -> usize {
        match &self.output {
            Output::Field(output) => output.output_dimension(),
            Output::Observable(output) => output.output_dimension(),
        }
    }
    fn primal(&self, values: &mut [f64]) -> Result<(), Diagnostic> {
        match &self.output {
            Output::Field(output) => output.primal(values),
            Output::Observable(output) => output.primal(values),
        }
    }
    fn jvp(
        &self,
        unknown: &[f64],
        parameter: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        match &self.output {
            Output::Field(value) => value.jvp(unknown, parameter, output),
            Output::Observable(value) => value.jvp(unknown, parameter, output),
        }
    }
    fn vjp(
        &self,
        output: &[f64],
        unknown: &mut [f64],
        parameter: &mut [f64],
    ) -> Result<(), Diagnostic> {
        match &self.output {
            Output::Field(value) => value.vjp(output, unknown, parameter),
            Output::Observable(value) => value.vjp(output, unknown, parameter),
        }
    }
}
