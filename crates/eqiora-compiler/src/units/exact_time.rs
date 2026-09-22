//! Bounded exact literal arithmetic for concrete source clocks.

use eqiora_core::{Diagnostic, DimExponents};
use eqiora_lang::{BinaryOp, Expr, ExprKind, UnaryOp};
use eqiora_schema::kernel::RationalTime;
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::diagnostics::source_error;

pub(crate) fn lower_clock(
    file: &str,
    period: &Expr,
    phase: &Expr,
) -> Result<(RationalTime, RationalTime), Diagnostic> {
    Ok((
        lower_time(file, period, true)?,
        lower_time(file, phase, false)?,
    ))
}

fn error(file: &str, expression: &Expr, message: &str) -> Diagnostic {
    source_error(
        eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR,
        file,
        expression.range(),
        message,
    )
}

fn lower_time(file: &str, expression: &Expr, period: bool) -> Result<RationalTime, Diagnostic> {
    let (value, dimension) = evaluate(file, expression, 0)?;
    if dimension != super::coherent_dimension("s").expect("time unit") {
        return Err(error(
            file,
            expression,
            "clock quantity must have time dimension",
        ));
    }
    if value.is_negative() || (period && value.is_zero()) {
        return Err(error(
            file,
            expression,
            if period {
                "periodic ClockDomain requires a strictly positive period"
            } else {
                "periodic ClockDomain requires a nonnegative phase"
            },
        ));
    }
    RationalTime::new(
        value.numer().to_u64().expect("bounded numerator"),
        value.denom().to_u64().expect("bounded denominator"),
    )
    .map_err(|diagnostic| error(file, expression, diagnostic.message()))
}

fn evaluate(
    file: &str,
    expression: &Expr,
    depth: usize,
) -> Result<(Ratio<BigInt>, DimExponents), Diagnostic> {
    let fail = |message| error(file, expression, message);
    if depth > 256 {
        return Err(fail("clock literal expression exceeds depth 256"));
    }
    let (value, dimension) = match expression.kind() {
        ExprKind::Quantity { value, unit } => {
            let unit = super::lower_unit(unit, depth + 1).map_err(fail)?;
            if value.is_zero() {
                (Ratio::zero(), unit.dimension)
            } else {
                let power = value
                    .exponent10()
                    .checked_add(i64::from(unit.decimal_power))
                    .ok_or_else(|| fail("clock decimal exponent exceeds exact time bounds"))?;
                // Canonical coefficients have no factor ten. A negative exponent below
                // -63 leaves at least 2^64 or 5^64 in the reduced denominator.
                if !(-63..=19).contains(&power) || value.coefficient().len() > 256 {
                    return Err(fail("clock decimal exceeds reduced u64 rational bounds"));
                }
                let mut coefficient = BigInt::parse_bytes(value.coefficient().as_bytes(), 10)
                    .expect("validated decimal");
                if value.is_negative() {
                    coefficient = -coefficient;
                }
                let scale = BigInt::from(10u8).pow(power.unsigned_abs() as u32);
                let value = if power >= 0 {
                    Ratio::from_integer(coefficient * scale)
                } else {
                    Ratio::new(coefficient, scale)
                };
                (value, unit.dimension)
            }
        }
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value: operand,
        } => {
            let (value, dimension) = evaluate(file, operand, depth + 1)?;
            (-value, dimension)
        }
        ExprKind::Binary { op, left, right }
            if matches!(
                op,
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
            ) =>
        {
            let (left, left_dimension) = evaluate(file, left, depth + 1)?;
            let (right, right_dimension) = evaluate(file, right, depth + 1)?;
            match op {
                BinaryOp::Add | BinaryOp::Sub => {
                    if left_dimension != right_dimension {
                        return Err(fail(
                            "clock addition and subtraction require equal dimensions",
                        ));
                    }
                    (
                        if *op == BinaryOp::Add {
                            left + right
                        } else {
                            left - right
                        },
                        left_dimension,
                    )
                }
                BinaryOp::Mul => (
                    left * right,
                    left_dimension
                        .mul(right_dimension)
                        .ok_or_else(|| fail("clock dimension exceeds exponent bounds"))?,
                ),
                BinaryOp::Div => {
                    if right.is_zero() {
                        return Err(fail("clock literal division by zero"));
                    }
                    (
                        left / right,
                        left_dimension
                            .div(right_dimension)
                            .ok_or_else(|| fail("clock dimension exceeds exponent bounds"))?,
                    )
                }
                _ => unreachable!(),
            }
        }
        _ => {
            return Err(fail(
                "clock time requires exact literal quantity arithmetic; bindings and binary64 numbers are not admitted",
            ));
        }
    };
    if value.numer().abs().to_u64().is_none() || value.denom().to_u64().is_none() {
        return Err(fail("clock expression exceeds reduced u64 rational bounds"));
    }
    Ok((value, dimension))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(period: &str, phase: &str) -> Result<(RationalTime, RationalTime), Diagnostic> {
        let source = format!("model M() {{ clock tick = periodic({period}, phase = {phase}); }}");
        let document = eqiora_lang::parse("clock.eqi", &source)
            .into_document()
            .expect("clock syntax");
        let model = &document.models()[0];
        let eqiora_lang::Item::Clock(clock) = &model.items()[0] else {
            panic!("clock")
        };
        lower_clock("clock.eqi", clock.period(), clock.phase())
    }

    #[test]
    fn native_binary64_clock_values_are_rejected() {
        let expression = eqiora_lang::SourceAstFactory::expression(
            ExprKind::Number(eqiora_lang::DecimalLiteral::parse("1.0").expect("exact literal")),
            eqiora_lang::TextRange::new(0, 1),
        )
        .unwrap();
        let error = lower_clock("native.eqi", &expression, &expression).unwrap_err();
        assert!(error.message().contains("binary64"));
    }

    #[test]
    fn exact_decimal_units_and_rational_arithmetic() {
        for period in [
            "10[ms]",
            "0.01[s]",
            "(1[s] / 100)",
            "(3[ms] + 7[ms])",
            "(2[ms] * 5)",
        ] {
            let (period, phase) = clock(period, "1[s] / 3").expect("exact clock");
            assert_eq!(period, RationalTime::new(1, 100).unwrap());
            assert_eq!(phase, RationalTime::new(1, 3).unwrap());
        }
        assert_eq!(
            clock("1[s] - 999[ms]", "0[s]").unwrap().0,
            RationalTime::new(1, 1000).unwrap()
        );
    }

    #[test]
    fn exact_time_falsifiers() {
        for period in [
            "0[s]",
            "-1[s]",
            "1[m]",
            "1[s] + 1[m]",
            "1[s] / 0",
            "1e100000[s]",
            "1e-100000[s]",
            "18446744073709551616[s]",
            "1[s] / 18446744073709551616",
            "unknown",
        ] {
            assert!(clock(period, "0[s]").is_err(), "{period}");
        }
        assert!(clock("1[s]", "-1[s]").is_err());
        // Zero does not allocate exponent-sized powers, but still carries units.
        assert!(clock("1[s]", "0e999999[s]").is_ok());
        assert!(clock("1[s]", "0[m]").is_err());
    }

    #[test]
    fn reduction_precedes_bounds_but_every_node_is_bounded() {
        let max = u64::MAX;
        // 5^63 * 10^-63 = 1 / 2^63: a long coefficient can reduce
        // into the admitted range without any binary floating conversion.
        assert_eq!(
            clock(
                "108420217248550443400745280086994171142578125e-63[s]",
                "0[s]"
            )
            .unwrap()
            .0,
            RationalTime::new(1, 1u64 << 63).unwrap(),
        );
        assert_eq!(
            clock(&format!("{max}[s]"), "0[s]").unwrap().0,
            RationalTime::new(max, 1).unwrap()
        );
        assert_eq!(
            clock("18446744073709551615000[ms]", "0[s]").unwrap().0,
            RationalTime::new(max, 1).unwrap()
        );
        assert!(clock(&format!("({max}[s] + 1[s]) / 2"), "0[s]").is_err());
    }
}
