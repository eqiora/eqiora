//! Independent coefficient/exponent correspondence for first time derivatives.

#[cfg(test)]
mod map_tests;
#[cfg(test)]
mod mapped_field_tests;
mod projection;
#[cfg(test)]
mod tests;
mod transport;

use std::fmt;

use eqiora_core::entity::kinds;
use eqiora_core::{Id, ValueType};

use super::{ExprDag, ExprId};
use crate::kernel::pure_operator::{ExactPolynomial, ExactPolynomialError, ExactRational};

const MAX_PROOF_NODES: usize = 4096;
const MAX_PROOF_WORK: usize = 262_144;

/// The time-derivative correspondence could not be proved within its limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeDerivativeProofError {
    /// A supplied expression ID does not exist in this expression arena.
    InvalidExpression,
    /// The reachable expression is outside the smooth real or complex scalar polynomial profile.
    UnsupportedExpression,
    /// Storage reads a derivative, discrete value, Port, or other unsupported symbol.
    UnsupportedStorageSymbol,
    /// A binary64 literal is not finite or exceeds the exact rational representation.
    NonExactConstant,
    /// Exact polynomial arithmetic failed or exceeded its own work bound.
    Polynomial(ExactPolynomialError),
    /// Input nodes or cumulative polynomial work exceeded the proof bound.
    Limit,
    /// The normalized accumulation does not have the required coefficients and atoms.
    Mismatch,
}

impl fmt::Display for TimeDerivativeProofError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidExpression => "time-derivative proof references an invalid expression",
            Self::UnsupportedExpression => {
                "time-derivative proof requires a typed scalar polynomial"
            }
            Self::UnsupportedStorageSymbol => "storage uses an unsupported time-dependent symbol",
            Self::NonExactConstant => "time-derivative proof literal exceeds exact rational bounds",
            Self::Polynomial(_) => "time-derivative proof exact polynomial arithmetic failed",
            Self::Limit => "time-derivative proof exceeds its work bound",
            Self::Mismatch => "accumulation is not the exact first time derivative of storage",
        })
    }
}

impl std::error::Error for TimeDerivativeProofError {}

impl From<ExactPolynomialError> for TimeDerivativeProofError {
    fn from(error: ExactPolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Atom {
    Field(Id<kinds::Field>),
    Derivative(Id<kinds::Field>),
    Parameter(Id<kinds::Parameter>),
    Time,
    Coordinate(Id<kinds::Domain>, Id<kinds::Domain>, usize),
    Map(usize),
    MapRate(usize),
    Partial(usize),
    MappedField(usize),
    ImaginaryField(Id<kinds::Field>),
    ImaginaryDerivative(Id<kinds::Field>),
    ImaginaryParameter(Id<kinds::Parameter>),
}

impl Atom {
    fn key(self) -> (u8, u128, u128, usize) {
        match self {
            Self::Field(id) => (0, id.ulid().into(), 0, 0),
            Self::Derivative(id) => (1, id.ulid().into(), 0, 0),
            Self::Parameter(id) => (2, id.ulid().into(), 0, 0),
            Self::Time => (3, 0, 0, 0),
            Self::Coordinate(support, factor, axis) => {
                (4, support.ulid().into(), factor.ulid().into(), axis)
            }
            Self::Map(index) => (5, 0, 0, index),
            Self::MapRate(index) => (6, 0, 0, index),
            Self::Partial(index) => (7, 0, 0, index),
            Self::MappedField(index) => (8, 0, 0, index),
            Self::ImaginaryField(id) => (9, id.ulid().into(), 0, 0),
            Self::ImaginaryDerivative(id) => (10, id.ulid().into(), 0, 0),
            Self::ImaginaryParameter(id) => (11, id.ulid().into(), 0, 0),
        }
    }
}

impl Ord for Atom {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key().cmp(&other.key())
    }
}

impl PartialOrd for Atom {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

type Polynomial = ExactPolynomial<Atom>;

impl ExprDag {
    /// Check that `accumulation` is the first time derivative of `storage`.
    ///
    /// Both IDs are interpreted in this DAG, whether or not they are output roots.
    /// Fields are independent unknowns whose time derivatives are the matching
    /// `SymbolRef::Derivative`; Parameters are constant, and model time has derivative
    /// one. `value_type` supplies the independently checked type of each reachable
    /// node. The caller must already have established consistent dimensions and supports, continuous Field eligibility, and the activation context
    /// of the owning Relation. This check does not establish them, and it does not
    /// permit floating-point reassociation.
    ///
    /// Supported expressions are real/complex scalar constants, Fields, Parameters,
    /// time, negation, addition, subtraction, multiplication, nonnegative integer powers,
    /// real/imaginary projections, conjugation, squared magnitude and retained scalar
    /// polynomial pure definitions. Real and imaginary coefficients are proved separately. Coordinate-map factors keep their
    /// selectors and normalized polynomial rows; an accumulation action must match each
    /// mapped row's time rate and use model time. Polynomial pullbacks substitute the
    /// coordinate identities. For first mapped continuous scalar Fields, the chain rule
    /// uses the map, the physical time rate, and the coordinate partials. Nested
    /// pullbacks of unknown Fields and nested factor-valued motion are not supported.
    /// Only `accumulation` may read Field time derivatives and first mapped coordinate
    /// partials. Branches, guards, divisions, other spatial operators, discrete reads,
    /// Ports, and other scalar functions are rejected, even when algebraic cancellation
    /// would remove them.
    ///
    /// Binary64 literals are taken as their exact dyadic values, never as a guessed
    /// decimal rational. Non-finite values and dyadics outside `ExactRational` bounds
    /// are rejected. Retained definitions keep their original rational coefficients.
    /// Inputs are limited to 4096 DAG nodes, and the cumulative count of normalized
    /// terms plus factors to 262144, in addition to the polynomial work limits.
    ///
    /// The check compares normalized coefficients and exponents directly; it never
    /// calls executable differentiation or builds an execution DAG.
    pub fn verify_time_derivative(
        &self,
        storage: ExprId,
        accumulation: ExprId,
        value_type: impl Fn(ExprId) -> Option<ValueType>,
    ) -> Result<(), TimeDerivativeProofError> {
        if self.nodes().len() > MAX_PROOF_NODES {
            return Err(TimeDerivativeProofError::Limit);
        }
        let mut budget = Budget(MAX_PROOF_WORK);
        let mut maps = projection::Maps::default();
        let storage =
            projection::components(self, storage, false, &mut budget, &mut maps, &value_type)?;
        let accumulation = projection::components(
            self,
            accumulation,
            true,
            &mut budget,
            &mut maps,
            &value_type,
        )?;
        let expected = projection::Parts {
            real: derivative(&storage.real, &mut maps, &mut budget)?,
            imag: derivative(&storage.imag, &mut maps, &mut budget)?,
        };
        if expected == accumulation {
            Ok(())
        } else {
            Err(TimeDerivativeProofError::Mismatch)
        }
    }
}

fn derivative(
    value: &Polynomial,
    maps: &mut projection::Maps,
    budget: &mut Budget,
) -> Result<Polynomial, TimeDerivativeProofError> {
    let mut expected = Polynomial::constant(ExactRational::integer(0));
    for (atoms, coefficient) in value.terms() {
        let mut start = 0;
        while start < atoms.len() {
            let atom = atoms[start];
            let end = start + atoms[start..].partition_point(|other| *other == atom);
            let rate = match atom {
                Atom::Field(field) => Some(Polynomial::atom(Atom::Derivative(field))),
                Atom::ImaginaryField(field) => {
                    Some(Polynomial::atom(Atom::ImaginaryDerivative(field)))
                }
                Atom::Time => Some(Polynomial::constant(ExactRational::integer(1))),
                Atom::Map(index) if maps.dynamic(index) => {
                    Some(Polynomial::atom(Atom::MapRate(index)))
                }
                Atom::MappedField(index) => Some(maps.field_rate(index, budget)?),
                _ => None,
            };
            if let Some(rate) = rate {
                let multiplicity =
                    i64::try_from(end - start).map_err(|_| TimeDerivativeProofError::Limit)?;
                let coefficient = coefficient
                    .checked_mul(ExactRational::integer(multiplicity))
                    .map_err(ExactPolynomialError::from)?;
                budget.charge(atoms.len() + 1)?;
                let mut factors = atoms.to_vec();
                factors.remove(start);
                let mut term = Polynomial::constant(ExactRational::integer(0));
                term.add_term(factors, coefficient)?;
                let term = term.checked_mul(&rate)?;
                budget.polynomial(&term)?;
                expected = expected.checked_add(&term)?;
                budget.polynomial(&expected)?;
            }
            start = end;
        }
    }
    Ok(expected)
}

struct Budget(usize);

impl Budget {
    fn charge(&mut self, amount: usize) -> Result<(), TimeDerivativeProofError> {
        self.0 = self
            .0
            .checked_sub(amount)
            .ok_or(TimeDerivativeProofError::Limit)?;
        Ok(())
    }

    fn polynomial(&mut self, polynomial: &Polynomial) -> Result<(), TimeDerivativeProofError> {
        self.charge(1 + polynomial.terms().len() + polynomial.factor_count())
    }
}

#[cfg(test)]
fn real_scalar_type(_: ExprId) -> Option<ValueType> {
    Some(
        ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            eqiora_core::DimExponents::DIMENSIONLESS,
        )
        .expect("independently declared scalar test atoms"),
    )
}
