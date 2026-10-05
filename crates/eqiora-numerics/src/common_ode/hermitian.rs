//! Model-bound operator contracts use the existing complete Hermitian validator.
use super::*;

impl CommonOdePolicy {
    /// Require an exact Hermitian finite-map Parameter owned by this Model.
    ///
    /// Validation precedes division by time/energy scales and matrix actions.
    /// This declares a mathematical property of the bound Parameter; evolution
    /// and state-normalization conditions remain separate explicit contracts.
    ///
    /// # Errors
    /// Rejects duplicate declarations. Resolution rejects missing Parameters
    /// and non-Hermitian or incompatible finite-map values.
    pub fn with_hermitian_parameter(
        mut self,
        parameter: Id<kinds::Parameter>,
    ) -> Result<Self, Diagnostic> {
        if self.hermitian_parameters.contains(&parameter) {
            return Err(invalid("Hermitian Parameter contract is repeated"));
        }
        self.hermitian_parameters.push(parameter);
        self.hermitian_parameters.sort_by_key(|p| p.ulid());
        Ok(self)
    }
}

impl CommonOdePlan {
    pub(super) fn admit_hermitian_parameters(
        &self,
        kernel: &KernelProgram,
    ) -> Result<(), Diagnostic> {
        if !self.temporal.hermitian_parameters.is_empty()
            && (self.temporal.events().is_some() || self.temporal.forward_sensitivities().is_some())
        {
            return Err(invalid(
                "Hermitian Parameter contracts do not yet admit event resets or parameter sensitivity directions",
            ));
        }
        for parameter in &self.temporal.hermitian_parameters {
            let value = kernel
                .typed_value(parameter.erase())
                .ok_or_else(|| invalid("Hermitian Parameter has no complete Model value"))?;
            eqiora_solver::HermitianEigenproblem::check_operator(value)?;
        }
        Ok(())
    }
}
