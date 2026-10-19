//! Analytic functions compose in the same nilpotent direction algebra as solves.
use super::*;
use eqiora_core::{DimExponents, ScalarDomain, ValueLiteral, ValueType};
use eqiora_schema::kernel::UnaryMathFunction;

type Complex = [Scaled; 2];
const ZERO: Complex = [Scaled::ZERO; 2];

pub(super) fn admit(function: UnaryMathFunction, count: usize) -> Result<usize, Diagnostic> {
    real_math::admit(function)?;
    let work = count
        .is_power_of_two()
        .then(|| {
            let order = count.trailing_zeros();
            3usize
                .checked_pow(order)?
                .checked_mul((order as usize + 1) * 8)
        })
        .flatten();
    if work.is_none_or(|work| work > MAX_COMPONENT_PRODUCTS) {
        return Err(ir_builder_error(
            "function coefficient work exceeds scalar product budget",
        ));
    }
    Ok(count)
}

fn value(function: UnaryMathFunction, z: Complex) -> Result<Complex, Diagnostic> {
    let ty = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
        .map_err(|error| ir_builder_error(error.to_string()))?;
    let input = ValueLiteral::new(ty, [(z[0].value(), z[1].value())])
        .map_err(|error| ir_builder_error(error.to_string()))?;
    let (real, imaginary) = function
        .evaluate(&input)?
        .component(0)
        .expect("scalar result");
    Ok([Scaled::new(real), Scaled::new(imaginary)])
}

fn multiply([a, b]: Complex, [c, d]: Complex) -> Complex {
    [a * c - b * d, a * d + b * c]
}

fn scale(z: Complex, factor: f64) -> Complex {
    z.map(|value| value * factor)
}

pub(super) fn coefficients(
    function: UnaryMathFunction,
    real: &[Option<Scaled>],
    imaginary: &[Option<Scaled>],
) -> Result<Vec<Complex>, Diagnostic> {
    let count = admit(function, real.len())?;
    require_length(imaginary, count, "complex function coefficients")?;
    if real
        .iter()
        .chain(imaginary)
        .flatten()
        .any(|v| !v.is_finite())
    {
        return Ok(vec![[Scaled::new(f64::NAN); 2]; count]);
    }
    let z = |index: usize| {
        [
            real[index].unwrap_or(Scaled::ZERO),
            imaginary[index].unwrap_or(Scaled::ZERO),
        ]
    };
    let base = z(0);
    let initial = value(function, base)?;
    let mut result = vec![ZERO; count];
    result[0] = initial;
    if count == 1 {
        return Ok(result);
    }
    // Primal principal-branch values remain admitted on the cut. Active
    // derivatives, including sampled zero directions, require regularity.
    if matches!(function, UnaryMathFunction::Log | UnaryMathFunction::Sqrt)
        && base[1].is_zero()
        && base[0].value() <= 0.
    {
        return Err(Diagnostic::error(
            codes::NONFINITE_EVALUATION,
            "complex root/log derivative is undefined at zero or on the principal branch cut",
        ));
    }
    let order = count.trailing_zeros() as usize;
    let mut taylor = initial;
    let mut companion = match function {
        UnaryMathFunction::Sin => value(UnaryMathFunction::Cos, base)?,
        UnaryMathFunction::Cos => scale(value(UnaryMathFunction::Sin, base)?, -1.),
        _ => ZERO,
    };
    let mut power = vec![ZERO; count];
    power[0] = [Scaled::new(1.), Scaled::ZERO];
    for k in 1..=order {
        taylor = match function {
            UnaryMathFunction::Sin | UnaryMathFunction::Cos => {
                let next = scale(companion, 1. / k as f64);
                companion = scale(taylor, -1. / k as f64);
                next
            }
            UnaryMathFunction::Exp => scale(taylor, 1. / k as f64),
            UnaryMathFunction::Log => {
                let numerator = if k == 1 {
                    [Scaled::new(1.), Scaled::ZERO]
                } else {
                    scale(taylor, -(k as f64 - 1.) / k as f64)
                };
                complex_quotient::scaled_quotient(numerator, base)
            }
            UnaryMathFunction::Sqrt => {
                complex_quotient::scaled_quotient(scale(taylor, (1.5 - k as f64) / k as f64), base)
            }
            _ => unreachable!("admitted analytic function"),
        };
        let mut next = vec![ZERO; count];
        for mask in 1..count {
            let mut subset = mask;
            while subset != 0 {
                let term = multiply(z(subset), power[mask ^ subset]);
                for (sum, term) in next[mask].iter_mut().zip(term) {
                    *sum += term;
                }
                subset = (subset - 1) & mask;
            }
            let term = multiply(taylor, next[mask]);
            for (sum, term) in result[mask].iter_mut().zip(term) {
                *sum += term;
            }
        }
        power = next;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: Complex, expected: [f64; 2], scale: f64) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            let actual = (actual / scale).value();
            assert!(
                (actual - expected).abs() <= 16. * f64::EPSILON * (1. + expected.abs()),
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn complex_taylor_coefficients_follow_closed_form_series_through_order_four() {
        use UnaryMathFunction::*;
        // Four independent copies of x: coefficients are ordinary repeated
        // derivatives. sin(i*x)=i*sinh(x), cos(i*x)=cosh(x), exp(x), log(1+x).
        for function in [Sin, Cos, Exp, Log] {
            let mut a = vec![None; 16];
            let mut b = vec![None; 16];
            a[0] = Some(Scaled::new(f64::from(function == Log)));
            b[0] = Some(Scaled::ZERO);
            for bit in [1, 2, 4, 8] {
                if matches!(function, Sin | Cos) {
                    b[bit] = Some(Scaled::new(1.));
                } else {
                    a[bit] = Some(Scaled::new(1.));
                }
            }
            for (mask, actual) in coefficients(function, &a, &b)
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let order = mask.count_ones() as usize;
                let expected = match function {
                    Sin => [0., f64::from(!order.is_multiple_of(2))],
                    Cos => [f64::from(order.is_multiple_of(2)), 0.],
                    Exp => [1., 0.],
                    Log => [[0., 1., -1., 2., -6.][order], 0.],
                    _ => unreachable!(),
                };
                close(actual, expected, 1.);
            }
        }
    }

    #[test]
    fn complex_root_coefficients_keep_scales_and_principal_branch_regularity() {
        // sqrt((3+4i)*s*(1+x))=(2+i)*sqrt(s)*sqrt(1+x).
        for exponent in [-900, -400, 400, 900] {
            let scale = 2f64.powi(exponent);
            let mut a = vec![None; 16];
            let mut b = vec![None; 16];
            for mask in [0, 1, 2, 4, 8] {
                a[mask] = Some(Scaled::new(3. * scale));
                b[mask] = Some(Scaled::new(4. * scale));
            }
            for (mask, actual) in coefficients(UnaryMathFunction::Sqrt, &a, &b)
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let factor = [1., 0.5, -0.25, 0.375, -0.9375][mask.count_ones() as usize];
                close(actual, [2. * factor, factor], 2f64.powi(exponent / 2));
            }
        }
        let a = [Some(Scaled::new(-4.)), Some(Scaled::ZERO)];
        let b = [Some(Scaled::ZERO); 2];
        close(
            coefficients(UnaryMathFunction::Sqrt, &a[..1], &b[..1]).unwrap()[0],
            [0., 2.],
            1.,
        );
        assert!(coefficients(UnaryMathFunction::Sqrt, &a, &b).is_err());
        assert!(coefficients(UnaryMathFunction::Log, &a, &b).is_err());
    }
}
