//! A solve jet is linear in its right-hand coefficients for a fixed matrix.
use super::*;
use affine_analysis::SummaryFailure;

fn value(
    shape: Shape,
    a: &[Option<AffineValue>],
    b: &[Option<AffineValue>],
) -> Result<AffineValue, Diagnostic> {
    let (av, bv) = (
        a.iter().map(|v| v.map(|v| v.value)).collect::<Vec<_>>(),
        b.iter().map(|v| v.map(|v| v.value)).collect::<Vec<_>>(),
    );
    let value = shape.jet(&av, &bv)?;
    let tangent = if a.iter().chain(b).flatten().any(|v| v.tangent.is_some()) {
        let mut av = av
            .into_iter()
            .map(|v| v.map(Scaled::new))
            .collect::<Vec<_>>();
        let mut bv = bv
            .into_iter()
            .map(|v| v.map(Scaled::new))
            .collect::<Vec<_>>();
        av.extend(a.iter().map(|v| v.and_then(|v| v.tangent)));
        bv.extend(b.iter().map(|v| v.and_then(|v| v.tangent)));
        Some(shape.scaled_jet(&av, &bv)?)
    } else {
        None
    };
    Ok(AffineValue { value, tangent })
}

pub(in crate::scalar) fn evaluate(
    shape: Shape,
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
        .map(|a| value(shape, &a, &denominator))
        .transpose()
        .map_err(numerical)?;
    let coefficients = if matches!(shape, Shape::ComplexMath { .. }) {
        if a.iter().flatten().any(AffineSummary::depends_on_selected) {
            return Err(SymbolicLinearityFailure::Nonlinear { instruction }.into());
        }
        vec![AffineValue::constant(0.); dimension]
    } else {
        (0..dimension)
            .map(|index| {
                let numerator = a
                    .iter()
                    .map(|value| value.as_ref().map(|v| v.coefficients[index]))
                    .collect::<Vec<_>>();
                value(shape, &numerator, &denominator)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(numerical)?
    };
    Ok(AffineSummary::finite(constant, coefficients, instruction)?)
}
