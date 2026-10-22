//! Local affine residual actions and their constrained gather/scatter.
use super::invalid;
use crate::spatial_expression::Coefficient;
use eqiora_assembly::{AssemblyMap, LocalContribution, LocalUnknown};
use eqiora_core::Diagnostic;

pub(crate) fn evaluate_linear_residual<S: Coefficient>(
    local: &LocalContribution<S>,
    map: &AssemblyMap<S>,
    global_point: &[S],
) -> Result<Vec<S>, Diagnostic> {
    let local_point = mapped_local_point(map, global_point)?;
    evaluate_local_residual(local, &local_point)
}

pub(crate) fn evaluate_local_residual<S: Coefficient>(
    local: &LocalContribution<S>,
    local_point: &[S],
) -> Result<Vec<S>, Diagnostic> {
    if local_point.len() != local.columns() {
        return Err(invalid(
            "local residual point does not match the prepared contribution shape",
        ));
    }
    let residual = local
        .matrix()
        .chunks_exact(local.columns())
        .zip(local.rhs())
        .map(|(row, rhs)| {
            row.iter()
                .zip(local_point)
                .map(|(entry, value)| *entry * *value)
                .sum::<S>()
                - *rhs
        })
        .collect::<Vec<_>>();
    if residual.iter().any(|value| !value.is_finite()) {
        return Err(invalid("local primal residual is non-finite"));
    }
    Ok(residual)
}

pub(crate) fn mapped_local_point<S: Coefficient>(
    map: &AssemblyMap<S>,
    global_point: &[S],
) -> Result<Vec<S>, Diagnostic> {
    map.unknowns()
        .iter()
        .map(|unknown| match unknown {
            LocalUnknown::Free(dof) => global_point.get(dof.index()).copied().ok_or_else(|| {
                invalid("local residual map references an unknown outside the candidate point")
            }),
            LocalUnknown::Fixed(value) => Ok(*value),
        })
        .collect()
}

pub(crate) fn scatter_residual<S: Coefficient>(
    output: &mut [S],
    map: &AssemblyMap<S>,
    local_residual: &[S],
) -> Result<(), Diagnostic> {
    if map.equations().len() != local_residual.len() {
        return Err(invalid(
            "region residual shape differs from its assembly map",
        ));
    }
    for (equation, value) in map.equations().iter().zip(local_residual) {
        if let Some(equation) = equation {
            let destination = output
                .get_mut(equation.index())
                .ok_or_else(|| invalid("region residual equation is outside its target"))?;
            *destination += *value;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
