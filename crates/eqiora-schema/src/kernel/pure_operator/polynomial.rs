//! Exact polynomial arithmetic, with work limits, used only for semantic classification.

use std::collections::BTreeMap;

use super::{CalculusNode, ExactRational, PureOperatorDefinition, PureOperatorError};

const MAX_TERMS: usize = 16_384;
const MAX_FACTORS: usize = 65_536;

/// Failure of exact polynomial classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactPolynomialError {
    /// Rational arithmetic exceeded the portable representation.
    Arithmetic(PureOperatorError),
    /// Polynomial terms or total monomial factors exceeded the fixed work limit.
    Limit,
}

impl From<PureOperatorError> for ExactPolynomialError {
    fn from(error: PureOperatorError) -> Self {
        Self::Arithmetic(error)
    }
}

/// Canonical commutative polynomial over ordered semantic atoms, with exact
/// coefficients.
///
/// Zero coefficients are omitted; each monomial keeps its repeated atoms sorted.
/// This type classifies mathematical expressions. It never permits floating-point
/// reassociation and does not build executable derivatives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactPolynomial<A> {
    terms: BTreeMap<Vec<A>, ExactRational>,
    factors: usize,
}

impl<A: Clone + Ord> ExactPolynomial<A> {
    /// A constant polynomial; zero is the empty polynomial.
    #[must_use]
    pub fn constant(value: ExactRational) -> Self {
        Self {
            terms: if value.is_zero() {
                BTreeMap::new()
            } else {
                BTreeMap::from([(Vec::new(), value)])
            },
            factors: 0,
        }
    }

    /// One semantic indeterminate.
    #[must_use]
    pub fn atom(atom: A) -> Self {
        Self {
            terms: BTreeMap::from([(vec![atom], ExactRational::integer(1))]),
            factors: 1,
        }
    }

    /// Canonical monomials in lexicographic atom order.
    pub fn terms(&self) -> impl ExactSizeIterator<Item = (&[A], ExactRational)> {
        self.terms
            .iter()
            .map(|(atoms, coefficient)| (atoms.as_slice(), coefficient.clone()))
    }

    /// Coefficient negation.
    pub fn checked_neg(&self) -> Result<Self, ExactPolynomialError> {
        Ok(Self {
            terms: self
                .terms
                .iter()
                .map(|(atoms, coefficient)| Ok((atoms.clone(), coefficient.checked_neg()?)))
                .collect::<Result<_, PureOperatorError>>()?,
            factors: self.factors,
        })
    }

    /// Sum, collecting like monomials.
    pub fn checked_add(&self, other: &Self) -> Result<Self, ExactPolynomialError> {
        let mut result = self.clone();
        for (atoms, coefficient) in &other.terms {
            result.add_term(atoms.clone(), coefficient.clone())?;
        }
        Ok(result)
    }

    /// Distributive product. The work limit is checked before allocating the product terms.
    pub fn checked_mul(&self, other: &Self) -> Result<Self, ExactPolynomialError> {
        let term_work = self.terms.len().saturating_mul(other.terms.len());
        let factor_work = self
            .factor_count()
            .saturating_mul(other.terms.len())
            .saturating_add(other.factor_count().saturating_mul(self.terms.len()));
        if term_work > MAX_TERMS || factor_work > MAX_FACTORS {
            return Err(ExactPolynomialError::Limit);
        }
        let mut result = Self::constant(ExactRational::integer(0));
        for (left_atoms, left_coefficient) in &self.terms {
            for (right_atoms, right_coefficient) in &other.terms {
                let mut atoms = left_atoms.clone();
                atoms.extend(right_atoms.iter().cloned());
                atoms.sort();
                result.add_term(
                    atoms,
                    left_coefficient.checked_mul(right_coefficient.clone())?,
                )?;
            }
        }
        Ok(result)
    }

    /// Substitute polynomials into a scalar polynomial operator definition.
    /// Returns `None` for a definition outside this classification profile. This does
    /// not reassociate executable arithmetic or differentiate anything.
    /// The caller charges each intermediate result against its cumulative budget.
    ///
    /// # Errors
    /// Returns exact arithmetic failures or the caller's budget failure.
    pub fn substitute<E: From<ExactPolynomialError>>(
        definition: &PureOperatorDefinition,
        arguments: &[&Self],
        mut charge: impl FnMut(&Self) -> Result<(), E>,
    ) -> Result<Option<Self>, E> {
        if definition.formals().len() != arguments.len()
            || !definition.result_rule().is_invariant_scalar()
            || definition
                .formals()
                .iter()
                .any(|formal| !formal.is_invariant_scalar())
        {
            return Ok(None);
        }
        let Ok(definition) = definition.polynomial_derivative_view() else {
            return Ok(None);
        };
        let mut values: Vec<Self> = Vec::with_capacity(definition.nodes().len());
        for node in definition.nodes() {
            let get = |id: super::CalculusNodeId| &values[id.index() as usize];
            let value = match node {
                CalculusNode::Rational { value, .. } => Self::constant(value.clone()),
                CalculusNode::FormalComponent { formal, axes } if axes.is_empty() => {
                    (*arguments[usize::from(*formal)]).clone()
                }
                CalculusNode::BoundInput(value) => get(*value).clone(),
                CalculusNode::Neg(value) => get(*value).checked_neg()?,
                CalculusNode::Add(left, right) => get(*left).checked_add(get(*right))?,
                CalculusNode::Mul(left, right) => get(*left).checked_mul(get(*right))?,
                _ => return Ok(None),
            };
            charge(&value)?;
            values.push(value);
        }
        Ok(values.get(definition.root().index() as usize).cloned())
    }

    pub(crate) fn factor_count(&self) -> usize {
        self.factors
    }

    pub(crate) fn add_term(
        &mut self,
        atoms: Vec<A>,
        coefficient: ExactRational,
    ) -> Result<(), ExactPolynomialError> {
        let sum = self
            .terms
            .get(&atoms)
            .cloned()
            .unwrap_or(ExactRational::integer(0))
            .checked_add(coefficient)?;
        if sum.is_zero() {
            if self.terms.remove(&atoms).is_some() {
                self.factors -= atoms.len();
            }
        } else {
            if !self.terms.contains_key(&atoms) {
                if self.terms.len() == MAX_TERMS
                    || self.factor_count().saturating_add(atoms.len()) > MAX_FACTORS
                {
                    return Err(ExactPolynomialError::Limit);
                }
                self.factors += atoms.len();
            }
            self.terms.insert(atoms, sum);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_binomial_coefficients_and_zero_are_canonical() {
        let x = ExactPolynomial::atom('x');
        let y = ExactPolynomial::atom('y');
        let sum = x.checked_add(&y).unwrap();
        let square = sum.checked_mul(&sum).unwrap();
        let terms = square.terms().collect::<Vec<_>>();
        assert_eq!(
            terms,
            vec![
                (&['x', 'x'][..], ExactRational::integer(1)),
                (&['x', 'y'][..], ExactRational::integer(2)),
                (&['y', 'y'][..], ExactRational::integer(1)),
            ]
        );
        let zero = ExactPolynomial::constant(ExactRational::integer(0));
        assert_eq!(
            square.checked_add(&square.checked_neg().unwrap()).unwrap(),
            zero
        );
        assert_eq!(zero.checked_mul(&square).unwrap(), zero);
        assert_eq!(zero.factor_count(), 0);
    }

    #[test]
    fn rational_overflow_and_factor_growth_fail_closed() {
        // Twelve squarings produce 2^4096 (4097 bits). The next square needs
        // 8193 bits, beyond the declared 8192-bit coefficient budget.
        let mut large = ExactPolynomial::<char>::constant(ExactRational::integer(2));
        for _ in 0..12 {
            large = large.checked_mul(&large).unwrap();
        }
        assert_eq!(
            large.checked_mul(&large),
            Err(ExactPolynomialError::Arithmetic(
                PureOperatorError::RationalOverflow
            ))
        );
        let mut monomial = ExactPolynomial::atom('x');
        for _ in 0..16 {
            monomial = monomial.checked_mul(&monomial).unwrap();
        }
        assert_eq!(
            monomial.checked_mul(&monomial),
            Err(ExactPolynomialError::Limit)
        );
    }
}
