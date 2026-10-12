//! Independent residual agreement and finite allocation checks.

use eqiora_core::Diagnostic;

use super::invalid;

pub(super) fn require_same_residual(
    direct: &[f64],
    reconstructed: &[f64],
) -> Result<(), Diagnostic> {
    if direct.len() != reconstructed.len() {
        return Err(invalid(
            "captured ALE FSI relation residual shape differs from direct assembly",
        ));
    }
    let scale = direct
        .iter()
        .chain(reconstructed)
        .fold(1.0_f64, |scale, value| scale.max(value.abs()));
    let defect = direct
        .iter()
        .zip(reconstructed)
        .fold(0.0_f64, |defect, (direct, reconstructed)| {
            defect.max((direct - reconstructed).abs())
        });
    if defect > 65_536.0 * f64::EPSILON * scale {
        return Err(invalid(
            "captured ALE FSI relation does not reproduce its independently assembled residual",
        ));
    }
    Ok(())
}

pub(super) fn finite_norm(values: &[f64], name: &'static str) -> Result<f64, Diagnostic> {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() {
        return Err(invalid(format!("{name} norm is non-finite")));
    }
    Ok(norm)
}

pub(super) fn zeroed(length: usize, name: &'static str) -> Result<Vec<f64>, Diagnostic> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| invalid(format!("ALE FSI {name} allocation failed")))?;
    values.resize(length, 0.0);
    Ok(values)
}
