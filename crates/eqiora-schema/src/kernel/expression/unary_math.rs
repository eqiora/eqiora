//! Shared finite scalar mathematics for constant folding and reference execution.
use super::UnaryMathFunction;
use crate::kernel::typing::{self, ExpressionType};
use eqiora_core::{Diagnostic, ScalarDomain, ValueLiteral, diagnostic::codes};
use num_complex::Complex64;

impl UnaryMathFunction {
    /// Evaluate one finite typed scalar with the canonical domain and branch rules.
    /// Real roots/logarithms never promote invalid real inputs to complex values.
    /// Complex roots/logarithms use the principal branch; zero phase is rejected.
    pub fn evaluate(self, value: &ValueLiteral) -> Result<ValueLiteral, Diagnostic> {
        let result_type = typing::unary_math(
            self,
            &ExpressionType::<()>::new(value.value_type().clone(), None),
        )
        .map_err(|error| Diagnostic::error(codes::DIMENSION_MISMATCH, error.to_string()))?
        .value_type;
        let (re, im) = value.component(0).expect("scalar numeric type");
        if matches!(self, Self::Arg | Self::Log) && re == 0. && im == 0. {
            return Err(domain_error(
                "phase and logarithm require nonzero amplitude",
            ));
        }
        let result = if value.value_type().scalar_domain() == ScalarDomain::Real {
            let real = match self {
                Self::Sin => re.sin(),
                Self::Cos => re.cos(),
                Self::Exp => re.exp(),
                Self::Log if re > 0. => re.ln(),
                Self::Sqrt if re >= 0. => re.sqrt(),
                Self::Log | Self::Sqrt => {
                    return Err(domain_error("real root or logarithm is outside its domain"));
                }
                Self::Conj | Self::Real => re,
                Self::Imag => 0.,
                Self::Abs => re.abs(),
                Self::Abs2 => re * re,
                Self::Arg => {
                    if re < 0. {
                        std::f64::consts::PI
                    } else {
                        0.
                    }
                }
            };
            Complex64::new(real, 0.)
        } else {
            let z = Complex64::new(re, im);
            match self {
                Self::Sin => z.sin(),
                Self::Cos => z.cos(),
                Self::Exp => z.exp(),
                Self::Log => complex_log(z),
                Self::Sqrt => complex_sqrt(z),
                Self::Conj => z.conj(),
                Self::Real => Complex64::new(re, 0.),
                Self::Imag => Complex64::new(im, 0.),
                Self::Abs => Complex64::new(z.norm(), 0.),
                Self::Abs2 => Complex64::new(z.norm_sqr(), 0.),
                Self::Arg => Complex64::new(z.arg(), 0.),
            }
        };
        ValueLiteral::new(result_type, [(result.re, result.im)])
            .map_err(|_| domain_error("unary mathematics produced a nonfinite result"))
    }
}

// Scale only when forming the intermediate magnitude would overflow or lose
// subnormal precision. Powers of two preserve the branch and do not round a
// normal input's significand; the final ValueLiteral still checks finiteness.
fn complex_log(z: Complex64) -> Complex64 {
    let magnitude = z.norm();
    if magnitude.is_normal() {
        return z.ln();
    }
    let high = z.re.abs().max(z.im.abs());
    let low = z.re.abs().min(z.im.abs());
    let ratio = low / high;
    Complex64::new(high.ln() + 0.5 * (ratio * ratio).ln_1p(), z.arg())
}

fn complex_sqrt(z: Complex64) -> Complex64 {
    let high = z.re.abs().max(z.im.abs());
    if high > f64::MAX / 2. {
        return (z * 2_f64.powi(-512)).sqrt() * 2_f64.powi(256);
    }
    if high != 0. && high < f64::MIN_POSITIVE * 2. {
        return (z * 2_f64.powi(512)).sqrt() * 2_f64.powi(-256);
    }
    z.sqrt()
}

fn domain_error(message: &str) -> Diagnostic {
    Diagnostic::error(codes::NONFINITE_EVALUATION, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, ValueType};

    fn value(domain: ScalarDomain, dimension: DimExponents, re: f64, im: f64) -> ValueLiteral {
        ValueLiteral::new(ValueType::scalar(domain, dimension).unwrap(), [(re, im)]).unwrap()
    }
    fn close(actual: (f64, f64), expected: (f64, f64)) {
        for (a, e) in [(actual.0, expected.0), (actual.1, expected.1)] {
            assert!(
                (a - e).abs() <= 16. * f64::EPSILON * e.abs().max(1.),
                "{a} versus {e}"
            );
        }
    }
    #[test]
    fn projections_preserve_units_and_explicitly_select_real_domain() {
        use UnaryMathFunction::*;
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let z = value(ScalarDomain::Complex, length, 3., 4.);
        for (function, pair, domain, dimension) in [
            (Conj, (3., -4.), ScalarDomain::Complex, length),
            (Real, (3., 0.), ScalarDomain::Real, length),
            (Imag, (4., 0.), ScalarDomain::Real, length),
            (Abs, (5., 0.), ScalarDomain::Real, length),
            (
                Abs2,
                (25., 0.),
                ScalarDomain::Real,
                length.pow(2, 1).unwrap(),
            ),
        ] {
            let result = function.evaluate(&z).unwrap();
            assert_eq!(result.component(0), Some(pair));
            assert_eq!(result.value_type().scalar_domain(), domain);
            assert_eq!(result.value_type().dimension(), dimension);
        }
    }
    #[test]
    fn analytic_values_use_principal_branches_and_independent_exact_identities() {
        use UnaryMathFunction::*;
        let unit = DimExponents::DIMENSIONLESS;
        let pi = std::f64::consts::PI;
        // exp(log(2))=2 gives sinh(log(2))=3/4, cosh(log(2))=5/4.
        for (function, input, expected) in [
            (Sin, (0., std::f64::consts::LN_2), (0., 0.75)),
            (Cos, (0., std::f64::consts::LN_2), (1.25, 0.)),
            (Exp, (0., pi), (-1., 0.)),
            (Log, (-1., 0.), (0., pi)),
            (Sqrt, (3., 4.), (2., 1.)),
            (Sqrt, (-4., 0.), (0., 2.)),
            (Arg, (1., 1.), (pi / 4., 0.)),
        ] {
            let result = function
                .evaluate(&value(ScalarDomain::Complex, unit, input.0, input.1))
                .unwrap();
            close(result.component(0).unwrap(), expected);
        }
        for (function, input, expected) in [
            (Sin, 0., 0.),
            (Cos, 0., 1.),
            (Exp, 0., 1.),
            (Log, 1., 0.),
            (Sqrt, 4., 2.),
        ] {
            let result = function
                .evaluate(&value(ScalarDomain::Real, unit, input, 0.))
                .unwrap();
            assert_eq!(result.real_scalar_value().unwrap().value(), expected);
        }
    }
    #[test]
    fn finite_roots_and_logs_survive_extreme_intermediate_magnitudes() {
        let unit = DimExponents::DIMENSIONLESS;
        // For z = a(1+i), log(z) = log(a) + log(2)/2 + i*pi/4.
        // The exact norm can exceed binary64 even though every log component is finite.
        let a = f64::MAX;
        let z = value(ScalarDomain::Complex, unit, a, a);
        let logarithm = UnaryMathFunction::Log.evaluate(&z).unwrap();
        close(
            logarithm.component(0).unwrap(),
            (
                a.ln() + std::f64::consts::LN_2 / 2.,
                std::f64::consts::FRAC_PI_4,
            ),
        );
        let root = UnaryMathFunction::Sqrt.evaluate(&z).unwrap();
        // sqrt(a(1+i))/sqrt(a) has components sqrt((sqrt(2)+1)/2),
        // sqrt((sqrt(2)-1)/2), without ever forming the overflowing norm.
        let (re, im) = root.component(0).unwrap();
        close(
            (re / a.sqrt(), im / a.sqrt()),
            (
                ((std::f64::consts::SQRT_2 + 1.) / 2.).sqrt(),
                ((std::f64::consts::SQRT_2 - 1.) / 2.).sqrt(),
            ),
        );
        // sqrt(i*2^-1074) = (1+i)*2^-537/sqrt(2); dividing the input by two
        // first would incorrectly produce zero before taking the root.
        let tiny = value(ScalarDomain::Complex, unit, 0., f64::from_bits(1));
        let root = UnaryMathFunction::Sqrt.evaluate(&tiny).unwrap();
        let scale = 2_f64.powi(-537);
        let (re, im) = root.component(0).unwrap();
        close(
            (re / scale, im / scale),
            (
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ),
        );
    }

    #[test]
    fn invalid_domains_and_nonfinite_outputs_reject_without_complex_promotion() {
        use UnaryMathFunction::*;
        let unit = DimExponents::DIMENSIONLESS;
        for (function, domain, input) in [
            (Sqrt, ScalarDomain::Real, -1.),
            (Log, ScalarDomain::Real, -1.),
            (Log, ScalarDomain::Real, 0.),
            (Log, ScalarDomain::Complex, 0.),
            (Arg, ScalarDomain::Real, 0.),
            (Arg, ScalarDomain::Complex, 0.),
            (Exp, ScalarDomain::Real, 1e308),
            (Exp, ScalarDomain::Complex, 1e308),
        ] {
            assert_eq!(
                function
                    .evaluate(&value(domain, unit, input, 0.))
                    .unwrap_err()
                    .code(),
                codes::NONFINITE_EVALUATION
            );
        }
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        for function in [Sin, Cos, Exp, Log] {
            assert_eq!(
                function
                    .evaluate(&value(ScalarDomain::Complex, length, 1., 1.))
                    .unwrap_err()
                    .code(),
                codes::DIMENSION_MISMATCH
            );
        }
    }
}
