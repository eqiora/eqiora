//! Exact real/imaginary channels over the existing bounded polynomial owner.
use super::Atom;
use eqiora_schema::kernel::pure_operator::{ExactPolynomial, ExactPolynomialError, ExactRational};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Channel {
    Real,
    Imaginary,
}
type Component = ExactPolynomial<(Atom, Channel)>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Polynomial {
    real: Component,
    imaginary: Component,
}
impl Polynomial {
    pub(super) fn constant(value: ExactRational) -> Self {
        Self {
            real: Component::constant(value),
            imaginary: Component::constant(ExactRational::integer(0)),
        }
    }
    pub(super) fn atom(atom: Atom) -> Self {
        Self::symbol(atom, false)
    }
    pub(super) fn symbol(atom: Atom, complex: bool) -> Self {
        Self {
            real: Component::atom((atom.clone(), Channel::Real)),
            imaginary: if complex {
                Component::atom((atom, Channel::Imaginary))
            } else {
                Component::constant(ExactRational::integer(0))
            },
        }
    }
    pub(super) fn complex(real: Self, imaginary: Self) -> Option<Self> {
        // A Complex constructor has real-valued operands; do not silently discard channels.
        let zero = Component::constant(ExactRational::integer(0));
        (real.imaginary == zero && imaginary.imaginary == zero).then_some(Self {
            real: real.real,
            imaginary: imaginary.real,
        })
    }
    pub(super) fn checked_neg(&self) -> Result<Self, ExactPolynomialError> {
        Ok(Self {
            real: self.real.checked_neg()?,
            imaginary: self.imaginary.checked_neg()?,
        })
    }
    pub(super) fn conjugate(&self) -> Result<Self, ExactPolynomialError> {
        Ok(Self {
            real: self.real.clone(),
            imaginary: self.imaginary.checked_neg()?,
        })
    }
    pub(super) fn checked_add(&self, other: &Self) -> Result<Self, ExactPolynomialError> {
        Ok(Self {
            real: self.real.checked_add(&other.real)?,
            imaginary: self.imaginary.checked_add(&other.imaginary)?,
        })
    }
    pub(super) fn checked_mul(&self, other: &Self) -> Result<Self, ExactPolynomialError> {
        Ok(Self {
            real: self.real.checked_mul(&other.real)?.checked_add(
                &self
                    .imaginary
                    .checked_mul(&other.imaginary)?
                    .checked_neg()?,
            )?,
            imaginary: self
                .real
                .checked_mul(&other.imaginary)?
                .checked_add(&self.imaginary.checked_mul(&other.real)?)?,
        })
    }

    /// Division stays polynomial only for a nonzero constant divisor.
    pub(super) fn divide_constant(&self, divisor: &Self) -> Option<Self> {
        let constant = |component: &Component| {
            let mut terms = component.terms();
            match terms.next() {
                None => Some(ExactRational::integer(0)),
                Some(([], value)) if terms.next().is_none() => Some(value),
                _ => None,
            }
        };
        let real = constant(&divisor.real)?;
        let imaginary = constant(&divisor.imaginary)?;
        let norm = real
            .checked_mul(real.clone())
            .ok()?
            .checked_add(imaginary.checked_mul(imaginary.clone()).ok()?)
            .ok()?;
        let reciprocal = Self::complex(
            Self::constant(real.checked_div(norm.clone()).ok()?),
            Self::constant(imaginary.checked_neg().ok()?.checked_div(norm).ok()?),
        )?;
        self.checked_mul(&reciprocal).ok()
    }

    pub(super) fn checked_pow(&self, mut exponent: u32) -> Result<Self, ExactPolynomialError> {
        let mut result = Self::constant(ExactRational::integer(1));
        let mut base = self.clone();
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = result.checked_mul(&base)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.checked_mul(&base)?;
            }
        }
        Ok(result)
    }
}
