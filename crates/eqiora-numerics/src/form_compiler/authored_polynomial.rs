//! Exact bounded comparison of a first variation with the independently derived weak Law.
//! Live energy replay and exact boundary-discharge checks precede this comparison.
use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};
use eqiora_schema::kernel::pure_operator::{ExactPolynomial, ExactRational};

mod source;

type Polynomial = ExactPolynomial<Atom>;
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Atom {
    Field(String, Vec<usize>),
    Parameter(String),
    Coordinate(usize),
    Test(Vec<usize>),
    FieldGradient(String, Vec<usize>),
    TestGradient(Vec<usize>),
}

pub(super) fn matches_variation(
    projection: &AuthoredFormulationProjection,
    dimensions: usize,
    left: &E,
    right: &E,
) -> bool {
    let [(name, field, _, _)] = projection.test_restrictions() else {
        return false;
    };
    let Some(domain) = projection.domain_ulid() else {
        return false;
    };
    let (_, authored_left, authored_right) = &projection.equations()[0];
    let mut context = Context {
        name,
        field,
        domain,
        dimensions,
        remaining: 65536,
    };
    let compare = || -> Option<bool> {
        let actual = context
            .integral(authored_left)?
            .checked_add(&context.integral(authored_right)?.checked_neg().ok()?)
            .ok()?;
        let expected = context
            .integral(left)?
            .checked_add(&context.integral(right)?.checked_neg().ok()?)
            .ok()?;
        Some(actual == expected)
    };
    let mut compare = compare;
    compare().unwrap_or(false)
}

pub(super) fn matches_elastic_variation(
    projection: &AuthoredFormulationProjection,
    typed: &eqiora_schema::kernel::typing::TypedResidual<eqiora_core::RawId>,
    stress: eqiora_schema::kernel::ExprId,
    load: eqiora_schema::kernel::ExprId,
) -> bool {
    let [(name, field, _, _)] = projection.test_restrictions() else {
        return false;
    };
    let Some(domain) = projection.domain_ulid() else {
        return false;
    };
    let [(_, left, right)] = projection.equations() else {
        return false;
    };
    let mut context = Context {
        name,
        field,
        domain,
        dimensions: 2,
        remaining: 65536,
    };
    let mut compare = || -> Option<bool> {
        let actual = context
            .integral(left)?
            .checked_add(&context.integral(right)?.checked_neg().ok()?)
            .ok()?;
        let mut expected = Polynomial::constant(ExactRational::integer(0));
        // Integration by parts of -div(stress)-load pairs stress_ij with w_i,j
        // and load_i with w_i. Boundary discharge is authenticated separately.
        for i in 0..2 {
            for j in 0..2 {
                let term = context
                    .source(typed, stress, &[i, j], 0)?
                    .checked_mul(&Polynomial::atom(Atom::TestGradient(vec![i, j])))
                    .ok()?;
                expected = expected.checked_add(&term).ok()?;
            }
            let term = context
                .source(typed, load, &[i], 0)?
                .checked_mul(&Polynomial::atom(Atom::Test(vec![i])))
                .ok()?;
            expected = expected.checked_add(&term.checked_neg().ok()?).ok()?;
        }
        Some(actual == expected)
    };
    compare().unwrap_or(false)
}

struct Context<'a> {
    name: &'a str,
    field: &'a str,
    domain: &'a str,
    dimensions: usize,
    remaining: usize,
}
impl Context<'_> {
    fn integral(&mut self, value: &E) -> Option<Polynomial> {
        match value {
            E::Number { value } if *value == 0.0 => {
                Some(Polynomial::constant(ExactRational::integer(0)))
            }
            E::Integrate {
                domain_ulid,
                integrand,
            } if domain_ulid == self.domain => self.scalar(integrand, 0),
            E::Variation {
                wrt_ulid,
                directions,
                value,
                ..
            } if wrt_ulid == self.field && directions.as_slice() == [self.name] => {
                // Only a root first variation is admitted; nested wrappers cannot
                // masquerade as a checked body or as a stationarity equation.
                let E::Integrate {
                    domain_ulid,
                    integrand,
                } = value.as_ref()
                else {
                    return None;
                };
                (domain_ulid == self.domain).then_some(())?;
                self.scalar(integrand, 0)
            }
            _ => None,
        }
    }
    fn step(&mut self, depth: usize) -> Option<()> {
        if depth > 96 || self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(())
    }
    fn scalar(&mut self, value: &E, depth: usize) -> Option<Polynomial> {
        self.step(depth)?;
        Some(match value {
            E::Number { value } => Polynomial::constant(number(*value)?),
            E::Rational {
                numerator,
                denominator,
                ..
            } => Polynomial::constant(
                ExactRational::new(*numerator, i64::try_from(*denominator).ok()?).ok()?,
            ),
            E::Field { ulid } => Polynomial::atom(Atom::Field(ulid.clone(), vec![])),
            E::Parameter { ulid } => Polynomial::atom(Atom::Parameter(ulid.clone())),
            E::Coordinate { axis } if *axis < self.dimensions => {
                Polynomial::atom(Atom::Coordinate(*axis))
            }
            E::Test { field_ulid } if field_ulid == self.field => {
                Polynomial::atom(Atom::Test(vec![]))
            }
            E::Direction { name, field_ulid } if name == self.name && field_ulid == self.field => {
                Polynomial::atom(Atom::Test(vec![]))
            }
            E::Neg { value } => self.scalar(value, depth + 1)?.checked_neg().ok()?,
            E::Add { left, right } => self
                .scalar(left, depth + 1)?
                .checked_add(&self.scalar(right, depth + 1)?)
                .ok()?,
            E::Sub { left, right } => self
                .scalar(left, depth + 1)?
                .checked_add(&self.scalar(right, depth + 1)?.checked_neg().ok()?)
                .ok()?,
            E::Mul { left, right } => self
                .scalar(left, depth + 1)?
                .checked_mul(&self.scalar(right, depth + 1)?)
                .ok()?,
            E::Component { value, indices } if indices.len() == 1 => {
                self.vector(value, usize::try_from(indices[0]).ok()?, depth + 1)?
            }
            E::Component { value, indices } if indices.len() == 2 => self.tensor(
                value,
                usize::try_from(indices[0]).ok()?,
                usize::try_from(indices[1]).ok()?,
                depth + 1,
            )?,
            E::Dot { left, right } => {
                let mut sum = Polynomial::constant(ExactRational::integer(0));
                for axis in 0..self.dimensions {
                    let term = self
                        .vector(left, axis, depth + 1)?
                        .checked_mul(&self.vector(right, axis, depth + 1)?)
                        .ok()?;
                    sum = sum.checked_add(&term).ok()?;
                }
                sum
            }
            _ => return None,
        })
    }
    fn tensor(
        &mut self,
        value: &E,
        component: usize,
        axis: usize,
        depth: usize,
    ) -> Option<Polynomial> {
        self.step(depth)?;
        if component >= self.dimensions || axis >= self.dimensions {
            return None;
        }
        match value {
            E::Gradient { value } => Some(Polynomial::atom(match value.as_ref() {
                E::Field { ulid } => Atom::FieldGradient(ulid.clone(), vec![component, axis]),
                E::Direction { name, field_ulid }
                    if name == self.name && field_ulid == self.field =>
                {
                    Atom::TestGradient(vec![component, axis])
                }
                E::Test { field_ulid } if field_ulid == self.field => {
                    Atom::TestGradient(vec![component, axis])
                }
                _ => return None,
            })),
            _ => None,
        }
    }
    fn vector(&mut self, value: &E, axis: usize, depth: usize) -> Option<Polynomial> {
        self.step(depth)?;
        if axis >= self.dimensions {
            return None;
        }
        match value {
            E::Field { ulid } => Some(Polynomial::atom(Atom::Field(ulid.clone(), vec![axis]))),
            E::Direction { name, field_ulid } if name == self.name && field_ulid == self.field => {
                Some(Polynomial::atom(Atom::Test(vec![axis])))
            }
            E::Test { field_ulid } if field_ulid == self.field => {
                Some(Polynomial::atom(Atom::Test(vec![axis])))
            }
            E::Gradient { value } => Some(Polynomial::atom(match value.as_ref() {
                E::Field { ulid } => Atom::FieldGradient(ulid.clone(), vec![axis]),
                E::Test { field_ulid } if field_ulid == self.field => {
                    Atom::TestGradient(vec![axis])
                }
                E::Direction { name, field_ulid }
                    if name == self.name && field_ulid == self.field =>
                {
                    Atom::TestGradient(vec![axis])
                }
                _ => return None,
            })),
            E::Neg { value } => self.vector(value, axis, depth + 1)?.checked_neg().ok(),
            E::Mul { left, right } => {
                if let Some(scalar) = self.scalar(left, depth + 1) {
                    scalar
                        .checked_mul(&self.vector(right, axis, depth + 1)?)
                        .ok()
                } else {
                    self.vector(left, axis, depth + 1)?
                        .checked_mul(&self.scalar(right, depth + 1)?)
                        .ok()
                }
            }
            _ => None,
        }
    }
}

// Decode the binary64 value exactly, without a tolerance or decimal rounding.
fn number(value: f64) -> Option<ExactRational> {
    if !value.is_finite() {
        return None;
    }
    if value == 0.0 {
        return Some(ExactRational::integer(0));
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mut mantissa = bits & ((1u64 << 52) - 1);
    let mut power = if exponent == 0 {
        -1074
    } else {
        mantissa |= 1u64 << 52;
        exponent - 1075
    };
    let zeros = mantissa.trailing_zeros();
    mantissa >>= zeros;
    power += zeros as i32;
    let (numerator, denominator) = if power >= 0 {
        let scale = 1i128.checked_shl(u32::try_from(power).ok()?)?;
        (i128::from(mantissa).checked_mul(scale)?, 1)
    } else {
        let shift = u32::try_from(-power).ok()?;
        if shift > 62 {
            return None;
        }
        (i128::from(mantissa), 1i64 << shift)
    };
    let numerator = if value.is_sign_negative() {
        numerator.checked_neg()?
    } else {
        numerator
    };
    ExactRational::new(i64::try_from(numerator).ok()?, denominator).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_coefficients_are_exact_and_fail_closed_outside_portable_bounds() {
        for (value, numerator, denominator) in [
            (0.0, 0, 1),
            (-0.0, 0, 1),
            (0.5, 1, 2),
            (-3.25, -13, 4),
            (0.1, 3602879701896397, 36028797018963968),
        ] {
            assert_eq!(
                number(value),
                Some(ExactRational::new(numerator, denominator).unwrap())
            );
        }
        for value in [
            f64::NAN,
            f64::INFINITY,
            f64::MIN_POSITIVE,
            f64::MAX,
            -2.0_f64.powi(127),
        ] {
            assert!(number(value).is_none());
        }
    }
}
