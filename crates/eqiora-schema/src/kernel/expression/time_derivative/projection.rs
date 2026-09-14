use super::{Atom, Budget, ExactRational, Polynomial, TimeDerivativeProofError as Error};
use crate::kernel::pure_operator::{CalculusNode, PureOperatorDefinition};
use crate::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};

pub(super) fn normalize(
    dag: &ExprDag,
    root: ExprId,
    allow_derivatives: bool,
    budget: &mut Budget,
) -> Result<Polynomial, Error> {
    let root = root.index() as usize;
    let mut reachable = vec![false; dag.nodes().len()];
    *reachable.get_mut(root).ok_or(Error::InvalidExpression)? = true;
    for index in (0..=root).rev() {
        if reachable[index] {
            dag.nodes()[index].try_for_each_operand(|operand| {
                let slot = reachable
                    .get_mut(operand.index() as usize)
                    .ok_or(Error::InvalidExpression)?;
                *slot = true;
                Ok::<_, Error>(())
            })?;
        }
    }
    let mut values: Vec<Option<Polynomial>> = vec![None; root + 1];
    for index in 0..=root {
        if !reachable[index] {
            continue;
        }
        let get = |id: ExprId| {
            values
                .get(id.index() as usize)
                .and_then(Option::as_ref)
                .ok_or(Error::InvalidExpression)
        };
        let value = match &dag.nodes()[index] {
            ExprNode::Constant(value) => Polynomial::constant(exact_binary64(
                value
                    .real_scalar_value()
                    .ok_or(Error::UnsupportedExpression)?
                    .value(),
            )?),
            ExprNode::Symbol(symbol) => Polynomial::atom(match symbol {
                SymbolRef::Field(field) => Atom::Field(*field),
                SymbolRef::Parameter(parameter) => Atom::Parameter(*parameter),
                SymbolRef::Time => Atom::Time,
                SymbolRef::Derivative(field) if allow_derivatives => Atom::Derivative(*field),
                _ if !allow_derivatives => return Err(Error::UnsupportedStorageSymbol),
                _ => return Err(Error::UnsupportedExpression),
            }),
            ExprNode::Neg(value) => get(*value)?.checked_neg()?,
            ExprNode::Add(left, right) => get(*left)?.checked_add(get(*right)?)?,
            ExprNode::Sub(left, right) => get(*left)?.checked_add(&get(*right)?.checked_neg()?)?,
            ExprNode::Mul(left, right) => get(*left)?.checked_mul(get(*right)?)?,
            ExprNode::PowI(base, exponent) if *exponent >= 0 => {
                power(get(*base)?, *exponent as u32, budget)?
            }
            ExprNode::PureOperatorApplication(application) => {
                let definition = dag
                    .definition(application.definition())
                    .ok_or(Error::InvalidExpression)?;
                let arguments = application
                    .arguments()
                    .iter()
                    .map(|argument| get(*argument))
                    .collect::<Result<Vec<_>, _>>()?;
                pure_definition(definition, &arguments, budget)?
            }
            _ => return Err(Error::UnsupportedExpression),
        };
        budget.polynomial(&value)?;
        values[index] = Some(value);
    }
    values[root].take().ok_or(Error::InvalidExpression)
}

fn power(base: &Polynomial, mut exponent: u32, budget: &mut Budget) -> Result<Polynomial, Error> {
    let mut value = Polynomial::constant(ExactRational::integer(1));
    let mut factor = base.clone();
    while exponent > 0 {
        if exponent & 1 != 0 {
            value = value.checked_mul(&factor)?;
            budget.polynomial(&value)?;
        }
        exponent >>= 1;
        if exponent > 0 {
            factor = factor.checked_mul(&factor)?;
            budget.polynomial(&factor)?;
        }
    }
    Ok(value)
}

fn pure_definition(
    definition: &PureOperatorDefinition,
    arguments: &[&Polynomial],
    budget: &mut Budget,
) -> Result<Polynomial, Error> {
    if definition.formals().len() != arguments.len()
        || !definition.result_rule().is_invariant_scalar()
        || definition
            .formals()
            .iter()
            .any(|formal| !formal.is_invariant_scalar())
    {
        return Err(Error::UnsupportedExpression);
    }
    let mut values: Vec<Polynomial> = Vec::with_capacity(definition.nodes().len());
    for node in definition.nodes() {
        let get = |id: crate::kernel::pure_operator::CalculusNodeId| {
            values
                .get(id.index() as usize)
                .ok_or(Error::InvalidExpression)
        };
        let value = match node {
            CalculusNode::Rational { value, .. } => Polynomial::constant(*value),
            CalculusNode::FormalComponent { formal, axes } if axes.is_empty() => (**arguments
                .get(usize::from(*formal))
                .ok_or(Error::InvalidExpression)?)
            .clone(),
            CalculusNode::Neg(value) => get(*value)?.checked_neg()?,
            CalculusNode::Add(left, right) => get(*left)?.checked_add(get(*right)?)?,
            CalculusNode::Mul(left, right) => get(*left)?.checked_mul(get(*right)?)?,
            _ => return Err(Error::UnsupportedExpression),
        };
        budget.polynomial(&value)?;
        values.push(value);
    }
    values
        .get(definition.root().index() as usize)
        .cloned()
        .ok_or(Error::InvalidExpression)
}

fn exact_binary64(value: f64) -> Result<ExactRational, Error> {
    if !value.is_finite() {
        return Err(Error::NonExactConstant);
    }
    if value == 0.0 {
        return Ok(ExactRational::integer(0));
    }
    let bits = value.to_bits();
    let encoded_exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (mut significand, mut exponent) = if encoded_exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1_u64 << 52), encoded_exponent - 1023 - 52)
    };
    let shift = significand.trailing_zeros();
    significand >>= shift;
    exponent += shift as i32;
    let signed = if bits >> 63 == 0 {
        i128::from(significand)
    } else {
        -i128::from(significand)
    };
    let (numerator, denominator) = if exponent >= 0 {
        let scale = 1_i128
            .checked_shl(exponent as u32)
            .filter(|scale| *scale > 0)
            .ok_or(Error::NonExactConstant)?;
        let numerator = signed.checked_mul(scale).ok_or(Error::NonExactConstant)?;
        (
            i64::try_from(numerator).map_err(|_| Error::NonExactConstant)?,
            1,
        )
    } else {
        let denominator = 1_u64
            .checked_shl((-exponent) as u32)
            .ok_or(Error::NonExactConstant)?;
        (signed as i64, denominator)
    };
    ExactRational::from_canonical_parts(numerator, denominator).map_err(|_| Error::NonExactConstant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary64_literals_use_exact_dyadics_and_reject_unrepresentable_extremes() {
        assert_eq!(exact_binary64(0.5), Ok(ExactRational::new(1, 2).unwrap()));
        assert_eq!(exact_binary64(-0.0), Ok(ExactRational::integer(0)));
        assert_eq!(
            exact_binary64(0.1),
            Ok(ExactRational::from_canonical_parts(3602879701896397, 36028797018963968).unwrap())
        );
        assert_eq!(
            exact_binary64(-9223372036854775808.0),
            Ok(ExactRational::integer(i64::MIN))
        );
        for value in [
            f64::NAN,
            f64::INFINITY,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
        ] {
            assert_eq!(exact_binary64(value), Err(Error::NonExactConstant));
        }
    }
}
