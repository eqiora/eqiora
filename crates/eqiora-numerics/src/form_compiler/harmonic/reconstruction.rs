use super::*;
use eqiora_core::{DimExponents, DynQuantity};

impl HarmonicReduction {
    /// One peak-amplitude, negative-exponential reconstruction for finite and spatial values.
    pub(crate) fn reconstruct(
        &self,
        time: DynQuantity,
        components: impl IntoIterator<Item = (f64, f64)>,
    ) -> Result<Vec<f64>, Diagnostic> {
        if time.dim() != DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap()
            || !time.value().is_finite()
        {
            return Err(invalid(
                "harmonic reconstruction requires finite original model time in seconds",
            ));
        }
        let phase = self.angular_frequency * time.value();
        if !phase.is_finite() {
            return Err(invalid("harmonic reconstruction phase is not finite"));
        }
        let (sin, cos) = phase.sin_cos();
        components
            .into_iter()
            .map(|(real, imag)| {
                let value = real * cos + imag * sin;
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(invalid(
                        "harmonic reconstruction produced a nonfinite real value",
                    ))
                }
            })
            .collect()
    }
}
