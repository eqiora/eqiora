//! Tangent and dual products share storage admission, never their mathematical role.

use super::*;

/// A direction in the ordered product of the selected base spaces.
///
/// Signatures retain the exact Model entities, mathematical types and coefficient
/// support. Components use the base coordinates with a dimensionless perturbation
/// parameter; no flattened buffer can substitute for that association.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiableTangent {
    pub(super) values: DifferentiableValues,
}

impl DifferentiableTangent {
    /// Bind each leaf's logical real/imaginary components to its exact base space.
    ///
    /// # Errors
    /// Rejects mixed Model owners, duplicate entities, wrong counts, nonfinite components,
    /// and imaginary components in a real base space.
    pub fn from_components(
        signatures: &[DifferentiableValueSignature],
        components: &[&[(f64, f64)]],
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            values: DifferentiableValues::from_components(signatures, components)?,
        })
    }

    pub(crate) fn from_coordinates(
        signatures: Arc<[DifferentiableValueSignature]>,
        coordinates: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            values: DifferentiableValues::from_coordinates(signatures, coordinates)?,
        })
    }

    pub(crate) fn coordinates(&self) -> &[f64] {
        self.values.coordinates()
    }

    /// Exact ordered base-space signatures, excluding any map axes.
    #[must_use]
    pub fn base_signatures(&self) -> &[DifferentiableValueSignature] {
        self.values.signatures()
    }

    /// Logical components of one base-space leaf, in coefficient/component order.
    pub fn components(
        &self,
        index: usize,
    ) -> Option<impl ExactSizeIterator<Item = (f64, f64)> + '_> {
        self.values.components(index)
    }
}

/// A real-linear covector over the ordered selected base spaces.
///
/// Complex components pair with a tangent through `Re(conj(c) * t)`, summed over
/// logical components. Signatures identify the **base** spaces: their physical
/// units are not the units of a covector. No primal `ValueLiteral`, implicit
/// metric, holomorphic derivative, or objective unit is inferred here.
///
/// A reverse seed cannot silently become a forward direction:
///
/// ```compile_fail
/// use eqiora_api::{DifferentiableCotangent, DifferentiableProgram};
/// fn wrong_role(program: &DifferentiableProgram, seed: &DifferentiableCotangent) {
///     program.jvp(seed);
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiableCotangent {
    pub(super) values: DifferentiableValues,
}

impl DifferentiableCotangent {
    /// Bind real-linear dual components to their exact ordered base spaces.
    ///
    /// # Errors
    /// Rejects mixed Model owners, duplicate entities, wrong counts, nonfinite components,
    /// and imaginary components in a real base space.
    pub fn from_components(
        signatures: &[DifferentiableValueSignature],
        components: &[&[(f64, f64)]],
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            values: DifferentiableValues::from_components(signatures, components)?,
        })
    }

    pub(crate) fn from_coordinates(
        signatures: Arc<[DifferentiableValueSignature]>,
        coordinates: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            values: DifferentiableValues::from_coordinates(signatures, coordinates)?,
        })
    }

    pub(crate) fn coordinates(&self) -> &[f64] {
        self.values.coordinates()
    }

    /// Base-space signatures; these do not assign primal units to the covector.
    #[must_use]
    pub fn base_signatures(&self) -> &[DifferentiableValueSignature] {
        self.values.signatures()
    }

    /// Dual components paired with the tangent in this exact base-space leaf.
    pub fn components(
        &self,
        index: usize,
    ) -> Option<impl ExactSizeIterator<Item = (f64, f64)> + '_> {
        self.values.components(index)
    }
}

impl DifferentiableProgramIdentity {
    pub(crate) fn tangent_from_coordinates(
        &self,
        coordinates: &[f64],
    ) -> Result<DifferentiableTangent, Diagnostic> {
        DifferentiableTangent::from_coordinates(self.inputs.clone(), coordinates.to_vec())
    }

    #[cfg(test)]
    pub(crate) fn cotangent_from_coordinates(
        &self,
        coordinates: &[f64],
    ) -> Result<DifferentiableCotangent, Diagnostic> {
        DifferentiableCotangent::from_coordinates(self.outputs.clone(), coordinates.to_vec())
    }
}
