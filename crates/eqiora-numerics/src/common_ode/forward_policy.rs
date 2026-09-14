//! Explicit derivative controls retain exact physical coordinates and units.
use super::*;
use eqiora_core::DynQuantity;
use eqiora_time::ForwardSensitivityPlan;

/// Positive absolute tolerance for one exact state/Parameter derivative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommonSensitivityTolerance {
    field: Id<kinds::Field>,
    parameter: Id<kinds::Parameter>,
    quantity: DynQuantity,
}
impl CommonSensitivityTolerance {
    pub fn new(
        field: Id<kinds::Field>,
        parameter: Id<kinds::Parameter>,
        quantity: DynQuantity,
    ) -> Result<Self, Diagnostic> {
        require_positive(quantity.value(), "forward sensitivity absolute tolerance")?;
        Ok(Self {
            field,
            parameter,
            quantity,
        })
    }
    #[must_use]
    pub const fn field(self) -> Id<kinds::Field> {
        self.field
    }
    #[must_use]
    pub const fn parameter(self) -> Id<kinds::Parameter> {
        self.parameter
    }
    #[must_use]
    pub const fn quantity(self) -> DynQuantity {
        self.quantity
    }
}
/// Explicit forward derivative error controls, before exact Model admission.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonForwardSensitivity {
    relative_tolerance: f64,
    absolute_tolerances: Vec<CommonSensitivityTolerance>,
}
impl CommonForwardSensitivity {
    pub fn new(
        relative_tolerance: f64,
        mut absolute_tolerances: Vec<CommonSensitivityTolerance>,
    ) -> Result<Self, Diagnostic> {
        require_positive(relative_tolerance, "forward sensitivity relative tolerance")?;
        absolute_tolerances.sort_by_key(|entry| (entry.parameter().ulid(), entry.field().ulid()));
        if absolute_tolerances.is_empty()
            || absolute_tolerances.windows(2).any(|pair| {
                pair[0].field() == pair[1].field() && pair[0].parameter() == pair[1].parameter()
            })
        {
            return Err(invalid(
                "forward sensitivity requires nonempty, unique exact Field/Parameter tolerances",
            ));
        }
        Ok(Self {
            relative_tolerance,
            absolute_tolerances,
        })
    }
    #[must_use]
    pub const fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }
    #[must_use]
    pub fn absolute_tolerances(&self) -> &[CommonSensitivityTolerance] {
        &self.absolute_tolerances
    }
    pub(super) fn identity_bytes(&self) -> Vec<u8> {
        let mut bytes = b"forward-sensitivity/v1\0".to_vec();
        bytes.extend_from_slice(&self.relative_tolerance.to_bits().to_be_bytes());
        for entry in &self.absolute_tolerances {
            push(&mut bytes, entry.field().ulid().to_string().as_bytes());
            push(&mut bytes, entry.parameter().ulid().to_string().as_bytes());
            bytes.extend_from_slice(&entry.quantity().value().to_bits().to_be_bytes());
            bytes.extend_from_slice(&dimension_bytes(entry.quantity().dim()));
        }
        bytes
    }
}
impl CommonOdePlan {
    /// Explicit derivative execution controls in Parameter-major state order.
    #[must_use]
    pub fn forward_sensitivity_plan(&self) -> Option<&ForwardSensitivityPlan> {
        self.forward_sensitivity_plan.as_ref()
    }

    pub(super) fn admit_forward_policy(
        &mut self,
        kernel: &KernelProgram,
    ) -> Result<(), Diagnostic> {
        let Some(policy) = self.temporal.forward_sensitivities() else {
            return Ok(());
        };
        let parameters = if let Some(roots) = self.root_set()? {
            parameter_system::GlobalParameterSystem::new(self, &roots)?
                .parameter_ids()
                .to_vec()
        } else {
            self.program.forward_sensitivity_problem()?;
            self.program.parameter_fields().to_vec()
        };
        let fields = self.program.state_fields();
        if fields.len().checked_mul(parameters.len()) != Some(policy.absolute_tolerances().len()) {
            return Err(invalid(
                "forward sensitivity tolerances must cover every admitted Field/Parameter pair exactly",
            ));
        }
        let mut ordered = Vec::with_capacity(policy.absolute_tolerances().len());
        for parameter in &parameters {
            let dimension = kernel
                .typed_value((*parameter).into())
                .ok_or_else(|| invalid("forward sensitivity Parameter has no exact typed value"))?
                .value_type()
                .dimension();
            for (field, field_dimension) in fields.iter().zip(&self.field_dimensions) {
                let tolerance = policy.absolute_tolerances().iter().find(|entry| entry.field() == *field && entry.parameter() == *parameter).ok_or_else(|| invalid("forward sensitivity tolerance omits an exact Field/Parameter coordinate"))?;
                let expected = field_dimension
                    .div(dimension)
                    .ok_or_else(|| invalid("forward sensitivity dimension quotient overflow"))?;
                if tolerance.quantity().dim() != expected {
                    return Err(invalid(
                        "forward sensitivity absolute tolerance dimension must be Field dimension divided by Parameter dimension",
                    ));
                }
                ordered.push(tolerance.quantity().value());
            }
        }
        self.forward_sensitivity_plan = Some(ForwardSensitivityPlan::new(
            policy.relative_tolerance(),
            ordered,
        )?);
        self.forward_parameter_ids = Some(parameters);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
