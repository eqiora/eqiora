//! A quotient jet is linear in its numerator coefficients for a fixed denominator.
use super::*;
use affine_analysis::SummaryFailure;

fn value(a: &[Option<AffineValue>], b: &[Option<AffineValue>]) -> Result<AffineValue, Diagnostic> {
    let (mut av, mut bv) = (
        a.iter().map(|v| v.map(|v| v.value)).collect::<Vec<_>>(),
        b.iter().map(|v| v.map(|v| v.value)).collect::<Vec<_>>(),
    );
    let value = jet(&av, &bv)?;
    let tangent = if a.iter().chain(b).flatten().any(|v| v.tangent.is_some()) {
        av.extend(a.iter().map(|v| v.and_then(|v| v.tangent)));
        bv.extend(b.iter().map(|v| v.and_then(|v| v.tangent)));
        Some(jet(&av, &bv)?)
    } else {
        None
    };
    Ok(AffineValue { value, tangent })
}

pub(in crate::scalar) fn evaluate(
    a: &[Option<AffineSummary>],
    b: &[Option<AffineSummary>],
    dimension: usize,
    instruction: usize,
) -> Result<AffineSummary, SummaryFailure> {
    if b.iter().flatten().any(AffineSummary::depends_on_selected) {
        return Err(SymbolicLinearityFailure::Nonlinear { instruction }.into());
    }
    let constants = |values: &[Option<AffineSummary>]| {
        values
            .iter()
            .map(|value| match value {
                None => Some(None),
                Some(value) => value.constant.map(Some),
            })
            .collect::<Option<Vec<_>>>()
    };
    let Some(denominator) = constants(b) else {
        if a.iter().flatten().any(AffineSummary::depends_on_selected) {
            return Err(SymbolicLinearityFailure::VariableCoefficient { instruction }.into());
        }
        return Ok(AffineSummary::independent(dimension));
    };
    let numerical = |diagnostic| SummaryFailure::Numerical {
        instruction,
        diagnostic,
    };
    let constant = constants(a)
        .map(|a| value(&a, &denominator))
        .transpose()
        .map_err(numerical)?;
    let coefficients = (0..dimension)
        .map(|index| {
            let numerator = a
                .iter()
                .map(|value| value.as_ref().map(|v| v.coefficients[index]))
                .collect::<Vec<_>>();
            value(&numerator, &denominator)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(numerical)?;
    Ok(AffineSummary::finite(constant, coefficients, instruction)?)
}
