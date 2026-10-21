//! Exact real-coordinate polynomials for one numeric scalar.
use super::*;
use crate::kernel::UnaryMathFunction;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in super::super) struct Parts {
    pub(in super::super) real: Polynomial,
    pub(in super::super) imag: Polynomial,
}
impl Parts {
    pub(super) fn real(real: Polynomial) -> Self {
        Self {
            real,
            imag: Polynomial::constant(ExactRational::integer(0)),
        }
    }
    pub(super) fn as_real(&self) -> Result<&Polynomial, Error> {
        if self.imag.terms().next().is_none() {
            Ok(&self.real)
        } else {
            Err(Error::UnsupportedExpression)
        }
    }
    pub(super) fn charge(&self, budget: &mut Budget) -> Result<(), Error> {
        budget.polynomial(&self.real)?;
        if self.imag.terms().next().is_some() {
            budget.polynomial(&self.imag)?;
        }
        Ok(())
    }
    pub(super) fn neg(&self) -> Result<Self, Error> {
        Ok(Self {
            real: self.real.checked_neg()?,
            imag: self.imag.checked_neg()?,
        })
    }
    pub(super) fn add(&self, rhs: &Self) -> Result<Self, Error> {
        Ok(Self {
            real: self.real.checked_add(&rhs.real)?,
            imag: self.imag.checked_add(&rhs.imag)?,
        })
    }
    pub(super) fn mul(&self, rhs: &Self) -> Result<Self, Error> {
        Ok(Self {
            real: self
                .real
                .checked_mul(&rhs.real)?
                .checked_add(&self.imag.checked_mul(&rhs.imag)?.checked_neg()?)?,
            imag: self
                .real
                .checked_mul(&rhs.imag)?
                .checked_add(&self.imag.checked_mul(&rhs.real)?)?,
        })
    }
    pub(super) fn unary(&self, function: UnaryMathFunction) -> Result<Self, Error> {
        Ok(match function {
            UnaryMathFunction::Real => Self::real(self.real.clone()),
            UnaryMathFunction::Imag => Self::real(self.imag.clone()),
            UnaryMathFunction::Conj => Self {
                real: self.real.clone(),
                imag: self.imag.checked_neg()?,
            },
            UnaryMathFunction::Abs2 => Self::real(
                self.real
                    .checked_mul(&self.real)?
                    .checked_add(&self.imag.checked_mul(&self.imag)?)?,
            ),
            _ => return Err(Error::UnsupportedExpression),
        })
    }
}

pub(super) fn definition(
    definition: &PureOperatorDefinition,
    arguments: &[&Parts],
    budget: &mut Budget,
) -> Result<Parts, Error> {
    use crate::kernel::pure_operator::CalculusNode;
    if definition.formals().len() != arguments.len()
        || !definition.result_rule().is_invariant_scalar()
        || definition
            .formals()
            .iter()
            .any(|formal| !formal.is_invariant_scalar())
    {
        return Err(Error::UnsupportedExpression);
    }
    let definition = definition
        .polynomial_derivative_view()
        .map_err(|_| Error::UnsupportedExpression)?;
    let mut values = Vec::<Parts>::with_capacity(definition.nodes().len());
    for node in definition.nodes() {
        let get = |id: crate::kernel::pure_operator::CalculusNodeId| &values[id.index() as usize];
        let value = match node {
            CalculusNode::Rational { value, .. } => {
                Parts::real(Polynomial::constant(value.clone()))
            }
            CalculusNode::FormalComponent { formal, axes } if axes.is_empty() => {
                (*arguments[usize::from(*formal)]).clone()
            }
            CalculusNode::BoundInput(value) | CalculusNode::Demand { value, .. } => {
                get(*value).clone()
            }
            CalculusNode::Complex { real, imag } => Parts {
                real: get(*real).as_real()?.clone(),
                imag: get(*imag).as_real()?.clone(),
            },
            CalculusNode::Neg(value) => get(*value).neg()?,
            CalculusNode::Add(left, right) => get(*left).add(get(*right))?,
            CalculusNode::Mul(left, right) => get(*left).mul(get(*right))?,
            CalculusNode::UnaryMath(function, value) => get(*value).unary(*function)?,
            _ => return Err(Error::UnsupportedExpression),
        };
        value.charge(budget)?;
        values.push(value);
    }
    Ok(values[definition.root().index() as usize].clone())
}
