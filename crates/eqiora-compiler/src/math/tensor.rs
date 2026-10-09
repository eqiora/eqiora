//! Structural tensor arguments select shared Kernel calculus definitions.
use eqiora_lang::{CallArguments, Expr, ExprKind};
use eqiora_schema::kernel::{
    pure_operator::{PureOperatorDefinition, PureOperatorError},
    typing::ExpressionType,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Operation {
    Component(Vec<u32>),
    Permute(Vec<u16>),
    Contract(Vec<(u16, u16)>),
    Outer,
    Cross,
    ComponentwiseProduct,
}

pub(crate) fn named(name: &str) -> bool {
    matches!(
        name,
        "component" | "permute_axes" | "contract" | "outer" | "componentwise_product" | "cross"
    )
}

pub(crate) fn source<'a>(
    name: &str,
    arguments: &'a CallArguments,
) -> Result<(Operation, Vec<&'a Expr>), &'static str> {
    let (operands, options) = arguments.parts();
    let (arity, key) = match name {
        "component" => (1, Some("indices")),
        "permute_axes" => (1, Some("order")),
        "contract" => (2, Some("axes")),
        "outer" | "componentwise_product" | "cross" => (2, None),
        _ => return Err("unknown tensor operation"),
    };
    if operands.len() != arity || options.len() != usize::from(key.is_some()) {
        return Err(
            "tensor operation requires its exact positional operands and named axis option",
        );
    }
    let option = match key {
        Some(key) if options[0].name() == key => Some(tuple(options[0].value())?),
        Some(_) => return Err("tensor operation has an unknown axis option"),
        None => None,
    };
    let operation = match name {
        "component" => Operation::Component(
            option
                .unwrap()
                .iter()
                .map(index)
                .collect::<Result<_, _>>()?,
        ),
        "permute_axes" => {
            Operation::Permute(option.unwrap().iter().map(axis).collect::<Result<_, _>>()?)
        }
        "contract" => Operation::Contract(
            option
                .unwrap()
                .iter()
                .map(|pair| {
                    let [a, b] = tuple(pair)? else {
                        return Err("each contraction pair requires two axes");
                    };
                    Ok((axis(a)?, axis(b)?))
                })
                .collect::<Result<_, _>>()?,
        ),
        "outer" => Operation::Outer,
        "cross" => Operation::Cross,
        "componentwise_product" => Operation::ComponentwiseProduct,
        _ => unreachable!("checked operation"),
    };
    Ok((operation, operands.iter().collect()))
}

fn tuple(expression: &Expr) -> Result<&[Expr], &'static str> {
    match expression.kind() {
        ExprKind::Tuple(values) => Ok(values),
        _ => Err("tensor axes require an explicit compile-time tuple"),
    }
}
fn index(expression: &Expr) -> Result<u32, &'static str> {
    match expression.kind() {
        ExprKind::Number(value) => value
            .canonical_text()
            .parse()
            .map_err(|_| "tensor indices require nonnegative bounded integer literals"),
        _ => Err("tensor indices require nonnegative bounded integer literals"),
    }
}
fn axis(expression: &Expr) -> Result<u16, &'static str> {
    u16::try_from(index(expression)?).map_err(|_| "tensor axis exceeds the bounded rank")
}

impl Operation {
    pub(crate) fn definition<I>(
        &self,
        arguments: &[ExpressionType<I>],
    ) -> Result<PureOperatorDefinition, PureOperatorError> {
        let extent = arguments
            .iter()
            .find_map(|argument| {
                argument
                    .shape()
                    .extents()
                    .first()
                    .map(|extent| extent.get())
            })
            .ok_or(PureOperatorError::FormalTypeMismatch)?;
        let ranks = arguments
            .iter()
            .map(|argument| {
                u16::try_from(argument.shape().rank())
                    .map_err(|_| PureOperatorError::InvalidResultRule)
            })
            .collect::<Result<Vec<_>, _>>()?;
        match (self, ranks.as_slice()) {
            (Self::Cross, [1, 1]) if extent == 3 => PureOperatorDefinition::cross_product(),
            (Self::ComponentwiseProduct, [left, right]) if left == right => {
                PureOperatorDefinition::componentwise_product(extent, *left)
            }

            (Self::Component(indices), [rank]) if usize::from(*rank) == indices.len() => {
                PureOperatorDefinition::tensor_component(extent, indices)
            }
            (Self::Permute(order), [rank]) if usize::from(*rank) == order.len() => {
                PureOperatorDefinition::permute_axes(extent, order)
            }
            (Self::Contract(pairs), [left, right]) => {
                PureOperatorDefinition::contract(extent, *left, *right, pairs)
            }
            (Self::Outer, [left, right]) => {
                PureOperatorDefinition::contract(extent, *left, *right, &[])
            }
            _ => Err(PureOperatorError::FormalTypeMismatch),
        }
    }
}
