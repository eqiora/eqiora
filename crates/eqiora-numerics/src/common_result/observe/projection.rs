//! Select mathematical components through the existing typed value owner.
use super::{CommonObservation, Diagnostic, DynQuantity, invalid};

impl CommonObservation {
    /// Select a declared row-major component without inventing a physical convention.
    /// Phase is the principal argument in radians, undefined exactly at zero.
    /// # Errors
    /// Rejects unknown projections, invalid indices and unsupported numerical ranges.
    pub fn project_component(
        &self,
        projection: &str,
        index: usize,
    ) -> Result<Option<DynQuantity>, Diagnostic> {
        let value = match projection {
            "real" => self.value.component_real(index).map(Some),
            "imaginary" => self.value.component_imaginary(index).map(Some),
            "magnitude" => self.value.component_magnitude(index).map(Some),
            "squared_magnitude" => self.value.component_squared_magnitude(index).map(Some),
            "phase" => self.value.component_phase(index),
            _ => {
                return Err(invalid(
                    "projection must be real, imaginary, magnitude, squared_magnitude or phase",
                ));
            }
        };
        value.map_err(|error| invalid(error.to_string()))
    }
}
