mod parts;
pub(super) use parts::Parts;
mod maps;
pub(super) use maps::Maps;

use super::ExprId;
use super::{Atom, Budget, ExactRational, Polynomial, TimeDerivativeProofError as Error};
use crate::kernel::pure_operator::PureOperatorDefinition;
use crate::kernel::{ExprDag, ExprNode, SymbolRef};
use eqiora_core::{DimExponents, ScalarDomain, ValueType};

pub(super) fn normalize(
    dag: &ExprDag,
    root: ExprId,
    allow_derivatives: bool,
    budget: &mut Budget,
    maps: &mut Maps,
) -> Result<Polynomial, Error> {
    components(dag, root, allow_derivatives, budget, maps, &|_| {
        Some(
            ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .expect("real coordinate"),
        )
    })?
    .as_real()
    .cloned()
}

pub(super) fn components(
    dag: &ExprDag,
    root: ExprId,
    allow_derivatives: bool,
    budget: &mut Budget,
    maps: &mut Maps,
    value_type: &impl Fn(ExprId) -> Option<ValueType>,
) -> Result<Parts, Error> {
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
    let mut values: Vec<Option<Parts>> = vec![None; root + 1];
    for index in 0..=root {
        if !reachable[index] {
            continue;
        }
        let ty = value_type(ExprId(index as u32)).ok_or(Error::UnsupportedExpression)?;
        if !ty.shape().is_scalar()
            || !matches!(
                ty.scalar_domain(),
                ScalarDomain::Real | ScalarDomain::Complex
            )
        {
            return Err(Error::UnsupportedExpression);
        }
        let get = |id: ExprId| {
            values
                .get(id.index() as usize)
                .and_then(Option::as_ref)
                .ok_or(Error::InvalidExpression)
        };
        let get_real = |id| get(id)?.as_real();
        let value = match &dag.nodes()[index] {
            ExprNode::Constant(value) => {
                if !value.value_type().shape().is_scalar() {
                    return Err(Error::UnsupportedExpression);
                }
                let (real, imag) = value.component(0).ok_or(Error::UnsupportedExpression)?;
                let exact = |value| {
                    ExactRational::from_binary64(value)
                        .map(Polynomial::constant)
                        .ok_or(Error::NonExactConstant)
                };
                Parts {
                    real: exact(real)?,
                    imag: exact(imag)?,
                }
            }
            ExprNode::Symbol(symbol) => {
                let atom = match symbol {
                    SymbolRef::Field(field) => Atom::Field(*field),
                    SymbolRef::Parameter(parameter) => Atom::Parameter(*parameter),
                    SymbolRef::Time => Atom::Time,
                    SymbolRef::Coordinate {
                        support,
                        factor,
                        axis,
                    } => Atom::Coordinate(*support, *factor, *axis),
                    SymbolRef::Derivative(field, std::num::NonZeroU32::MIN)
                        if allow_derivatives =>
                    {
                        Atom::Derivative(*field)
                    }
                    _ if !allow_derivatives => return Err(Error::UnsupportedStorageSymbol),
                    _ => return Err(Error::UnsupportedExpression),
                };
                let imag = if ty.scalar_domain() == ScalarDomain::Complex {
                    Polynomial::atom(match atom {
                        Atom::Field(id) => Atom::ImaginaryField(id),
                        Atom::Derivative(id) => Atom::ImaginaryDerivative(id),
                        Atom::Parameter(id) => Atom::ImaginaryParameter(id),
                        _ => return Err(Error::UnsupportedExpression),
                    })
                } else {
                    Polynomial::constant(ExactRational::integer(0))
                };
                Parts {
                    real: Polynomial::atom(atom),
                    imag,
                }
            }
            ExprNode::CoordinateMapFactor { factor, source, at } => {
                let index = maps.intern(dag, Some(*factor), source, at, &get_real, budget)?;
                Parts::real(Polynomial::atom(Atom::Map(index)))
            }
            ExprNode::CoordinateMapFactorAction {
                value,
                parameter,
                directions,
            } if allow_derivatives => Parts::real(maps.action(
                dag,
                get_real(*value)?,
                *parameter,
                directions,
                &get_real,
                budget,
            )?),
            ExprNode::CoordinatePartial { value, wrt }
                if allow_derivatives && ty.scalar_domain() == ScalarDomain::Real =>
            {
                Parts::real(maps.partial(dag, *value, *wrt, budget)?)
            }
            ExprNode::Pullback { value, source, at }
                if ty.scalar_domain() == ScalarDomain::Real =>
            {
                Parts::real(maps.pullback(dag, *value, source, at, &get_real, budget)?)
            }
            ExprNode::Complex { real, imag } => Parts {
                real: get_real(*real)?.clone(),
                imag: get_real(*imag)?.clone(),
            },
            ExprNode::UnaryMath(function, value) => get(*value)?.unary(*function)?,
            ExprNode::Neg(value) => get(*value)?.neg()?,
            ExprNode::Add(left, right) => get(*left)?.add(get(*right)?)?,
            ExprNode::Sub(left, right) => get(*left)?.add(&get(*right)?.neg()?)?,
            ExprNode::Mul(left, right) => get(*left)?.mul(get(*right)?)?,
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
                parts::definition(definition, &arguments, budget)?
            }
            _ => return Err(Error::UnsupportedExpression),
        };
        value.charge(budget)?;
        values[index] = Some(value);
    }
    values[root].take().ok_or(Error::InvalidExpression)
}

fn power(base: &Parts, mut exponent: u32, budget: &mut Budget) -> Result<Parts, Error> {
    let mut value = Parts::real(Polynomial::constant(ExactRational::integer(1)));
    let mut factor = base.clone();
    while exponent > 0 {
        if exponent & 1 != 0 {
            value = value.mul(&factor)?;
            value.charge(budget)?;
        }
        exponent >>= 1;
        if exponent > 0 {
            factor = factor.mul(&factor)?;
            factor.charge(budget)?;
        }
    }
    Ok(value)
}
