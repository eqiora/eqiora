//! Real/complex arithmetic over the same typed expression nodes.
use super::check_component_work;
use eqiora_core::{Diagnostic, ScalarDomain, ValueFrame, ValueLiteral, diagnostic::codes};
use eqiora_schema::kernel::{
    ExprNode,
    typing::{self, ExpressionType},
};
use num_complex::Complex64;

fn ty(value: &ValueLiteral) -> Result<ExpressionType<()>, Diagnostic> {
    let ty = value.value_type();
    if !matches!(
        ty.scalar_domain(),
        ScalarDomain::Real | ScalarDomain::Complex
    ) || ty.frame() != ValueFrame::Invariant
        || ty.array_rank() != ty.shape().rank()
    {
        return Err(Diagnostic::error(
            codes::NOT_IMPLEMENTED,
            "numeric reference execution requires invariant real or complex channels",
        ));
    }
    Ok(ExpressionType::new(ty.clone(), None))
}

fn typing_error(error: typing::TypeViolation<()>) -> Diagnostic {
    Diagnostic::error(codes::DIMENSION_MISMATCH, error.to_string())
}

pub(super) fn binary(
    node: &ExprNode,
    left: &ValueLiteral,
    right: &ValueLiteral,
) -> Result<ValueLiteral, Diagnostic> {
    let (left_type, right_type) = (ty(left)?, ty(right)?);
    let output = match node {
        ExprNode::Add(..) | ExprNode::Sub(..) => typing::additive(&left_type, &right_type),
        ExprNode::Mul(..) => typing::multiply(&left_type, &right_type),
        ExprNode::Div(..) => typing::divide(&left_type, &right_type),
        ExprNode::Complex { .. } => left_type.complex(right_type),
        _ => unreachable!("numeric binary dispatch"),
    }
    .map_err(typing_error)?
    .value_type;
    let count = output
        .shape()
        .component_count()
        .expect("checked output type");
    check_component_work(0, count)?;
    let scalar_left = left.value_type().shape().is_scalar();
    let scalar_right = right.value_type().shape().is_scalar();
    let real_output = output.scalar_domain() == ScalarDomain::Real;
    ValueLiteral::new(
        output,
        (0..count).map(|index| {
            let (ar, ai) = left
                .component(if scalar_left { 0 } else { index })
                .expect("typed component");
            let (br, bi) = right
                .component(if scalar_right { 0 } else { index })
                .expect("typed component");
            if real_output {
                let real = match node {
                    ExprNode::Add(..) => ar + br,
                    ExprNode::Sub(..) => ar - br,
                    ExprNode::Mul(..) => ar * br,
                    ExprNode::Div(..) => ar / br,
                    _ => unreachable!("real arithmetic"),
                };
                return (real, 0.0);
            }
            let (a, b) = (Complex64::new(ar, ai), Complex64::new(br, bi));
            let result = match node {
                ExprNode::Add(..) => a + b,
                ExprNode::Sub(..) => a - b,
                ExprNode::Mul(..) => a * b,
                ExprNode::Div(..) => complex_divide(a, b),
                ExprNode::Complex { .. } => Complex64::new(ar, br),
                _ => unreachable!("complex arithmetic"),
            };
            (result.re, result.im)
        }),
    )
    .map_err(numeric_error)
}

pub(super) fn unary(node: &ExprNode, value: &ValueLiteral) -> Result<ValueLiteral, Diagnostic> {
    let input = ty(value)?;
    let output = match node {
        ExprNode::Neg(_) => input,
        ExprNode::PowI(_, exponent) => typing::power(&input, *exponent).map_err(typing_error)?,
        _ => unreachable!("numeric unary dispatch"),
    }
    .value_type;
    let count = output
        .shape()
        .component_count()
        .expect("checked output type");
    check_component_work(0, count)?;
    let real_output = output.scalar_domain() == ScalarDomain::Real;
    ValueLiteral::new(
        output,
        (0..count).map(|index| {
            let (real, imag) = value.component(index).expect("typed component");
            match node {
                ExprNode::Neg(_) => (-real, -imag),
                ExprNode::PowI(_, exponent) if real_output => (real.powi(*exponent), 0.0),
                ExprNode::PowI(_, exponent) => {
                    let value = Complex64::new(real, imag);
                    let z = if *exponent < 0 {
                        complex_divide(Complex64::new(1., 0.), value).powu(exponent.unsigned_abs())
                    } else {
                        value.powu(exponent.unsigned_abs())
                    };
                    (z.re, z.im)
                }
                _ => unreachable!("numeric unary dispatch"),
            }
        }),
    )
    .map_err(numeric_error)
}

fn numeric_error(error: eqiora_core::InvalidValueLiteral) -> Diagnostic {
    let code = if matches!(error, eqiora_core::InvalidValueLiteral::NonFinite) {
        codes::NONFINITE_EVALUATION
    } else {
        codes::DIMENSION_MISMATCH
    };
    Diagnostic::error(code, format!("numeric expression rejected: {error:?}"))
}

// Keep the denominator norm and reciprocal representable before using the
// library's floating division. Scaling both operands leaves the quotient unchanged.
fn complex_divide(mut numerator: Complex64, mut denominator: Complex64) -> Complex64 {
    let norm = denominator.norm();
    if norm.is_infinite() {
        numerator *= 0.5;
        denominator *= 0.5;
    } else if norm != 0. && norm < f64::MIN_POSITIVE {
        let scale = f64::from_bits((1023 + 512) << 52);
        numerator *= scale;
        denominator *= scale;
    }
    numerator.fdiv(denominator)
}
