//! Residual reconstruction, scatter and finite allocation checks.

use eqiora_assembly::{DofId, LocalContribution};
use eqiora_core::Diagnostic;

use super::invalid;

pub(super) fn evaluate_affine_residual(
    local: &LocalContribution<f64>,
    point: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    let entry_count = local
        .rows()
        .checked_mul(local.columns())
        .ok_or_else(|| invalid("ALE FSI affine local matrix shape overflows usize"))?;
    if local.columns() != point.len() || local.matrix().len() != entry_count {
        return Err(invalid(
            "ALE FSI affine residual point differs from its local matrix shape",
        ));
    }
    let residual = local
        .matrix()
        .chunks_exact(local.columns())
        .zip(local.rhs())
        .map(|(row, rhs)| {
            row.iter()
                .zip(point)
                .map(|(entry, value)| entry * value)
                .sum::<f64>()
                - rhs
        })
        .collect::<Vec<_>>();
    if residual.len() != local.rows() || residual.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "ALE FSI affine local residual is non-finite or differs from its row closure",
        ));
    }
    Ok(residual)
}

pub(super) fn affine_rhs_from_residual(
    matrix: &[f64],
    point: &[f64],
    residual: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    let entry_count = residual
        .len()
        .checked_mul(point.len())
        .ok_or_else(|| invalid("ALE FSI dense local matrix shape overflows usize"))?;
    if matrix.len() != entry_count {
        return Err(invalid(
            "ALE FSI dense Jacobian differs from its residual and candidate shape",
        ));
    }
    let rhs = matrix
        .chunks_exact(point.len())
        .zip(residual)
        .map(|(row, residual)| {
            row.iter()
                .zip(point)
                .map(|(entry, value)| entry * value)
                .sum::<f64>()
                - residual
        })
        .collect::<Vec<_>>();
    if rhs.len() != residual.len() || rhs.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "ALE FSI captured local right-hand side is non-finite or has the wrong row shape",
        ));
    }
    Ok(rhs)
}

pub(super) fn scatter_residual(
    output: &mut [f64],
    equations: &[Option<DofId>],
    local: &[f64],
) -> Result<(), Diagnostic> {
    if equations.len() != local.len() {
        return Err(invalid(
            "ALE FSI direct residual shape differs from its equation map",
        ));
    }
    for (equation, value) in equations.iter().zip(local) {
        if let Some(equation) = equation {
            let destination = output.get_mut(equation.index()).ok_or_else(|| {
                invalid("ALE FSI residual equation is outside its assembly target")
            })?;
            *destination += value;
        }
    }
    Ok(())
}

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
