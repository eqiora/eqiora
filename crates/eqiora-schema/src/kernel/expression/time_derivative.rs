//! Independent coefficient/exponent correspondence for first time derivatives.

mod projection;
#[cfg(test)]
mod tests;

use std::fmt;

use eqiora_core::Id;
use eqiora_core::entity::kinds;

use super::{ExprDag, ExprId};
use crate::kernel::pure_operator::{ExactPolynomial, ExactPolynomialError, ExactRational};

const MAX_PROOF_NODES: usize = 4096;
const MAX_PROOF_WORK: usize = 262_144;

/// A bounded mathematical time-derivative correspondence could not be proved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeDerivativeProofError {
    /// A supplied expression ID does not exist in this expression arena.
    InvalidExpression,
    /// The reachable expression is outside the smooth real scalar polynomial profile.
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
                "time-derivative proof requires a real scalar polynomial"
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
}

impl Atom {
    fn key(self) -> (u8, u128) {
        match self {
            Self::Field(id) => (0, id.ulid().into()),
            Self::Derivative(id) => (1, id.ulid().into()),
            Self::Parameter(id) => (2, id.ulid().into()),
            Self::Time => (3, 0),
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
    /// Check that `accumulation` is the exact first time derivative of `storage`.
    ///
    /// Both IDs are interpreted in this DAG, whether or not they are output roots.
    /// Field identities are independent indeterminates whose time derivatives are
    /// the corresponding exact `SymbolRef::Derivative`; Parameters are fixed and
    /// model time has derivative one. The caller must first establish real scalar
    /// types, consistent dimensions/support, continuous Field eligibility and the
    /// owning Relation's activation context. This check does not establish those
    /// properties or authorize floating-point reassociation.
    ///
    /// The admitted expressions are real scalar constants, Fields, Parameters,
    /// time, negation, addition, subtraction, multiplication, nonnegative integer
    /// powers, and retained scalar polynomial pure definitions. Only accumulation
    /// may read Field derivatives. Branches, guards, divisions, spatial operators,
    /// discrete reads, Ports and other scalar functions are rejected, even when
    /// an algebraic cancellation would hide them.
    ///
    /// Binary64 literals mean their exact dyadic values, never a guessed decimal
    /// rational. Nonfinite values or dyadics outside `ExactRational` bounds reject.
    /// Retained definitions keep their original exact rational coefficients.
    /// Inputs are bounded to 4096 DAG nodes; cumulative normalized terms plus
    /// factors are bounded to 262144, in addition to the polynomial work limits.
    ///
    /// This independently checks normalized coefficient/exponent correspondence;
    /// it never calls executable differentiation or produces an execution DAG.
    pub fn verify_time_derivative(
        &self,
        storage: ExprId,
        accumulation: ExprId,
    ) -> Result<(), TimeDerivativeProofError> {
        if self.nodes().len() > MAX_PROOF_NODES {
            return Err(TimeDerivativeProofError::Limit);
        }
        let mut budget = Budget(MAX_PROOF_WORK);
        let storage = projection::normalize(self, storage, false, &mut budget)?;
        let accumulation = projection::normalize(self, accumulation, true, &mut budget)?;
        let mut expected = Polynomial::constant(ExactRational::integer(0));
        // For c * product(a^n), each dynamic atom contributes c*n times the
        // monomial with one a removed, and (for a Field) its exact rate inserted.
        for (atoms, coefficient) in storage.terms() {
            let mut start = 0;
            while start < atoms.len() {
                let atom = atoms[start];
                let end = start + atoms[start..].partition_point(|other| *other == atom);
                if matches!(atom, Atom::Field(_) | Atom::Time) {
                    let multiplicity =
                        i64::try_from(end - start).map_err(|_| TimeDerivativeProofError::Limit)?;
                    let coefficient = coefficient
                        .checked_mul(ExactRational::integer(multiplicity))
                        .map_err(ExactPolynomialError::from)?;
                    budget.charge(atoms.len() + 1)?;
                    let mut factors = atoms.to_vec();
                    factors.remove(start);
                    if let Atom::Field(field) = atom {
                        factors.push(Atom::Derivative(field));
                        factors.sort();
                    }
                    expected.add_term(factors, coefficient)?;
                }
                start = end;
            }
        }
        if expected == accumulation {
            Ok(())
        } else {
            Err(TimeDerivativeProofError::Mismatch)
        }
    }
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
