//! Explicit squared-norm contracts for finite homogeneous linear evolution.
use super::*;
use eqiora_core::{DynQuantity, ScalarDomain};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConservedNorm {
    pub(crate) fields: Vec<Id<kinds::Field>>,
    pub(crate) target: DynQuantity,
    pub(crate) tolerance: DynQuantity,
}

impl CommonOdePolicy {
    /// Require conservation of the sum of squared magnitudes of complete Fields.
    ///
    /// Fields must share a physical dimension and have admitted real or complex
    /// component storage. The target and absolute tolerance carry its square.
    /// Admission proves a homogeneous autonomous linear generator preserves the
    /// selected norm; every accepted State must meet the stated tolerance.
    /// Values are never rescaled. Other ODEs remain available without this contract.
    ///
    /// # Errors
    /// Rejects empty/repeated Fields, invalid targets or incompatible tolerances.
    pub fn with_conserved_norm(
        mut self,
        mut fields: Vec<Id<kinds::Field>>,
        target: DynQuantity,
        tolerance: DynQuantity,
    ) -> Result<Self, Diagnostic> {
        fields.sort_by_key(|field| field.ulid());
        if fields.is_empty()
            || fields.windows(2).any(|pair| pair[0] == pair[1])
            || !target.value().is_finite()
            || target.value() < 0.
            || !tolerance.value().is_finite()
            || tolerance.value() <= 0.
            || target.dim() != tolerance.dim()
            || self
                .conserved_norms
                .iter()
                .any(|norm| norm.fields == fields)
        {
            return Err(invalid(
                "conserved norm requires unique Fields, a nonnegative target and positive same-dimension tolerance",
            ));
        }
        self.conserved_norms.push(ConservedNorm {
            fields,
            target,
            tolerance,
        });
        self.conserved_norms.sort_by(|a, b| {
            a.fields
                .iter()
                .map(|f| f.ulid())
                .cmp(b.fields.iter().map(|f| f.ulid()))
        });
        Ok(self)
    }
}

impl ConservedNorm {
    pub(crate) fn identity_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.fields.len() as u64).to_be_bytes());
        for field in &self.fields {
            push(&mut bytes, field.ulid().to_string().as_bytes());
        }
        bytes.extend_from_slice(&dimension_bytes(self.target.dim()));
        bytes.extend_from_slice(&self.target.value().to_bits().to_be_bytes());
        bytes.extend_from_slice(&self.tolerance.value().to_bits().to_be_bytes());
        bytes
    }

    fn indices(&self, program: &FirstOrderProgram) -> Vec<usize> {
        program
            .state_coordinates()
            .iter()
            .enumerate()
            .filter_map(|(index, c)| {
                (c.derivative_order() == 0 && self.fields.contains(&c.field())).then_some(index)
            })
            .collect()
    }

    fn admit(
        &self,
        kernel: &KernelProgram,
        program: &FirstOrderProgram,
        generator: &[f64],
    ) -> Result<(), Diagnostic> {
        for field in &self.fields {
            let Some(KernelNode::Field(definition)) = kernel.node(field.erase()) else {
                return Err(invalid("conserved norm refers to a missing Field"));
            };
            let ty = definition.value_type();
            if !matches!(
                ty.scalar_domain(),
                ScalarDomain::Real | ScalarDomain::Complex
            ) || ty.dimension().pow(2, 1) != Some(self.target.dim())
            {
                return Err(invalid(
                    "conserved norm requires real or complex Fields with matching squared physical dimensions",
                ));
            }
            let count = ty
                .shape()
                .component_count()
                .and_then(|count| {
                    count.checked_mul(if ty.scalar_domain() == ScalarDomain::Complex {
                        2
                    } else {
                        1
                    })
                })
                .ok_or_else(|| invalid("conserved norm shape overflows"))?;
            if program
                .state_coordinates()
                .iter()
                .filter(|c| c.field() == *field && c.derivative_order() == 0)
                .count()
                != count
            {
                return Err(invalid(
                    "conserved norm Field is not a complete current-value time state",
                ));
            }
        }
        let selected = self.indices(program);
        let n = program.state_coordinates().len();
        // D selects the declared coordinates. d(y^T D y)/dt=0 iff
        // D A + A^T D=0. Check both triangles and coupling to unselected Fields.
        for row in 0..n {
            for column in 0..n {
                let a = if selected.contains(&row) {
                    generator[row * n + column]
                } else {
                    0.
                };
                let b = if selected.contains(&column) {
                    generator[column * n + row]
                } else {
                    0.
                };
                if a != -b {
                    return Err(invalid(
                        "constant generator does not preserve the declared squared norm",
                    ));
                }
            }
        }
        Ok(())
    }

    fn check_state(&self, program: &FirstOrderProgram, values: &[f64]) -> Result<(), Diagnostic> {
        // hypot avoids squaring large components before summation. The target
        // is finite, so an unrepresentable squared norm necessarily rejects.
        let norm = self
            .indices(program)
            .into_iter()
            .fold(0_f64, |n, i| n.hypot(values[i]));
        let squared = norm * norm;
        if !squared.is_finite() || (squared - self.target.value()).abs() > self.tolerance.value() {
            return Err(invalid(
                "State violates its declared squared-norm tolerance",
            ));
        }
        Ok(())
    }
}

impl CommonOdePlan {
    pub(super) fn admit_conserved_norms(&self, kernel: &KernelProgram) -> Result<(), Diagnostic> {
        if self.temporal.conserved_norms.is_empty() {
            return Ok(());
        }
        if self.temporal.events().is_some() || self.temporal.forward_sensitivities().is_some() {
            return Err(invalid(
                "conserved norm admission does not yet prove event resets or parameter sensitivity directions",
            ));
        }
        let generator = self.program.constant_generator()?;
        for norm in &self.temporal.conserved_norms {
            norm.admit(kernel, &self.program, &generator)?;
        }
        Ok(())
    }
    pub(super) fn check_conserved_norms(&self, values: &[f64]) -> Result<(), Diagnostic> {
        for norm in &self.temporal.conserved_norms {
            norm.check_state(&self.program, values)?;
        }
        Ok(())
    }
}
