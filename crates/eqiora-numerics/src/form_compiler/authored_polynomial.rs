//! Exact bounded comparison of authored residuals and first variations with the weak Law.
//! Live authored-dependence replay and exact boundary-discharge checks precede this comparison.
use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};
use eqiora_core::ValueType;
use eqiora_schema::kernel::pure_operator::ExactRational;
use eqiora_sem::KernelProgram;
use std::collections::BTreeMap;

mod coefficients;
mod source;
mod typing;
use coefficients::Polynomial;
use typing::symbol_types;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Atom {
    Measure(String),
    TraceField(String, Vec<usize>),
    TraceTest(Vec<usize>),
    Field(String, Vec<usize>),
    Parameter(String, Vec<usize>),
    Coordinate(String, String, usize),
    Test(Vec<usize>),
    FieldGradient(String, Vec<usize>),
    TestGradient(Vec<usize>),
}

pub(super) fn matches_weak_residual(
    projection: &AuthoredFormulationProjection,
    program: &KernelProgram,
    dimensions: usize,
    left: &E,
    right: &E,
) -> bool {
    let [(name, field, _, _)] = projection.test_restrictions() else {
        return false;
    };
    if projection.domain_ulid().is_none() {
        return false;
    }
    let (_, authored_left, authored_right) = &projection.equations()[0];
    let mut context = Context {
        name,
        field,
        dimensions,
        remaining: 65536,
        symbols: symbol_types(program),
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
        // A whole residual reversal preserves the same equation; individual
        // term sign or phase changes still fail exact channel equality.
        Some(actual == expected || actual == expected.checked_neg().ok()?)
    };
    let mut compare = compare;
    compare().unwrap_or(false)
}

pub(super) struct ElasticTractionTerm {
    pub(super) boundary: eqiora_core::RawId,
    pub(super) typed: eqiora_schema::kernel::typing::TypedResidual<eqiora_core::RawId>,
    pub(super) datum: eqiora_schema::kernel::ExprId,
    pub(super) negative: bool,
}

pub(super) fn matches_elastic_variation(
    projection: &AuthoredFormulationProjection,
    program: &KernelProgram,
    typed: &eqiora_schema::kernel::typing::TypedResidual<eqiora_core::RawId>,
    stress: eqiora_schema::kernel::ExprId,
    load: eqiora_schema::kernel::ExprId,
    tractions: &[ElasticTractionTerm],
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
        dimensions: 2,
        remaining: 65536,
        symbols: symbol_types(program),
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
        let mut expected = expected
            .checked_mul(&Polynomial::atom(Atom::Measure(domain.into())))
            .ok()?;
        for traction in tractions {
            for i in 0..2 {
                let term = context
                    .source(&traction.typed, traction.datum, &[i], 0)?
                    .checked_mul(&Polynomial::atom(Atom::TraceTest(vec![i])))
                    .ok()?
                    .checked_mul(&Polynomial::atom(Atom::Measure(
                        traction.boundary.ulid().to_string(),
                    )))
                    .ok()?;
                let term = if traction.negative {
                    term.checked_neg().ok()?
                } else {
                    term
                };
                expected = expected.checked_add(&term).ok()?;
            }
        }
        Some(actual == expected)
    };
    compare().unwrap_or(false)
}

struct Context<'a> {
    name: &'a str,
    field: &'a str,
    dimensions: usize,
    remaining: usize,
    symbols: BTreeMap<String, ValueType>,
}
impl Context<'_> {
    fn integral(&mut self, value: &E) -> Option<Polynomial> {
        self.integral_at(value, 0, true)
    }
    fn integral_at(
        &mut self,
        value: &E,
        depth: usize,
        allow_variation: bool,
    ) -> Option<Polynomial> {
        self.step(depth)?;
        match value {
            E::Number { value } if *value == 0.0 => {
                Some(Polynomial::constant(ExactRational::integer(0)))
            }
            E::Integrate {
                domain_ulid,
                integrand,
            } => self
                .scalar(integrand, depth + 1)?
                .checked_mul(&Polynomial::atom(Atom::Measure(domain_ulid.clone())))
                .ok(),
            E::Add { left, right } => self
                .integral_at(left, depth + 1, allow_variation)?
                .checked_add(&self.integral_at(right, depth + 1, allow_variation)?)
                .ok(),
            E::Sub { left, right } => self
                .integral_at(left, depth + 1, allow_variation)?
                .checked_add(
                    &self
                        .integral_at(right, depth + 1, allow_variation)?
                        .checked_neg()
                        .ok()?,
                )
                .ok(),
            E::Neg { value } => self
                .integral_at(value, depth + 1, allow_variation)?
                .checked_neg()
                .ok(),
            E::Variation {
                wrt_ulid,
                directions,
                value,
                ..
            } if allow_variation
                && wrt_ulid == self.field
                && directions.as_slice() == [self.name] =>
            {
                // Replay authenticates the integral sum. Nested variation wrappers
                // cannot masquerade as generated first-variation terms.
                self.integral_at(value, depth + 1, false)
            }
            _ => None,
        }
    }
    fn trace(&self, value: &E, indices: Vec<usize>) -> Option<Polynomial> {
        self.atom(match value {
            E::Field { ulid } => Atom::TraceField(ulid.clone(), indices),
            E::Test { field_ulid } if field_ulid == self.field => Atom::TraceTest(indices),
            E::Direction { name, field_ulid } if name == self.name && field_ulid == self.field => {
                Atom::TraceTest(indices)
            }
            _ => return None,
        })
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
            E::Field { ulid } => self.atom(Atom::Field(ulid.clone(), vec![]))?,
            E::Trace { value } => self.trace(value, vec![])?,
            E::Parameter { ulid } => self.atom(Atom::Parameter(ulid.clone(), vec![]))?,
            E::Coordinate {
                support_ulid,
                factor_ulid,
                axis,
            } if *axis < self.dimensions => Polynomial::atom(Atom::Coordinate(
                support_ulid.clone(),
                factor_ulid.clone(),
                *axis,
            )),
            E::Test { field_ulid } if field_ulid == self.field => self.atom(Atom::Test(vec![]))?,
            E::Direction { name, field_ulid } if name == self.name && field_ulid == self.field => {
                self.atom(Atom::Test(vec![]))?
            }
            E::Complex { real, imag } => {
                Polynomial::complex(self.scalar(real, depth + 1)?, self.scalar(imag, depth + 1)?)?
            }
            E::Conjugate { value } => self.scalar(value, depth + 1)?.conjugate().ok()?,
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
            E::Inner { left, right } => self.contract(left, right, true, depth + 1)?,
            E::Dot { left, right } => self.contract(left, right, false, depth + 1)?,
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
            E::Gradient { value } => self.atom(match value.as_ref() {
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
            }),
            _ => None,
        }
    }
    fn vector(&mut self, value: &E, axis: usize, depth: usize) -> Option<Polynomial> {
        self.step(depth)?;
        match value {
            E::Trace { value } => self.trace(value, vec![axis]),
            E::Field { ulid } => self.atom(Atom::Field(ulid.clone(), vec![axis])),
            E::Parameter { ulid } => self.atom(Atom::Parameter(ulid.clone(), vec![axis])),
            E::Direction { name, field_ulid } if name == self.name && field_ulid == self.field => {
                self.atom(Atom::Test(vec![axis]))
            }
            E::Test { field_ulid } if field_ulid == self.field => self.atom(Atom::Test(vec![axis])),
            E::Gradient { value } => self.atom(match value.as_ref() {
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
            }),
            E::Add { left, right } => self
                .vector(left, axis, depth + 1)?
                .checked_add(&self.vector(right, axis, depth + 1)?)
                .ok(),
            E::Sub { left, right } => self
                .vector(left, axis, depth + 1)?
                .checked_add(&self.vector(right, axis, depth + 1)?.checked_neg().ok()?)
                .ok(),
            E::Conjugate { value } => self.vector(value, axis, depth + 1)?.conjugate().ok(),
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
mod tests;
