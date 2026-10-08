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
}
