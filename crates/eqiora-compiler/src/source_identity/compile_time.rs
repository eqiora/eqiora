use super::*;

pub(super) fn encode_parameter(
    encoder: &mut Encoder,
    declaration: &ParameterDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(2, |encoder| {
        super::value_type::encode_value_type(encoder, declaration.value_type(), budget, 1)
    })?;
    match declaration.value().kind() {
        ExprKind::Number(value) => encoder.field(3, |encoder| {
            super::expression::encode_decimal(encoder, value, false)
        }),
        ExprKind::Quantity { value, unit } => {
            encoder.field(3, |encoder| {
                super::expression::encode_decimal(encoder, value, false)
            })?;
            encoder.field(4, |encoder| encode_expression(encoder, unit, budget, 1))
        }
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } if matches!(
            value.kind(),
            ExprKind::Number(_) | ExprKind::Quantity { .. }
        ) =>
        {
            // Canonicalize a signed native literal and a parsed unary minus identically.
            match value.kind() {
                ExprKind::Number(value) => encoder.field(3, |encoder| {
                    super::expression::encode_decimal(encoder, value, true)
                }),
                ExprKind::Quantity { value, unit } => {
                    encoder.field(3, |encoder| {
                        super::expression::encode_decimal(encoder, value, true)
                    })?;
                    encoder.field(4, |encoder| encode_expression(encoder, unit, budget, 1))
                }
                _ => unreachable!("literal guard"),
            }
        }
        _ => encoder.field(5, |encoder| {
            encode_expression(encoder, declaration.value(), budget, 1)
        }),
    }
}

pub(super) fn encode_let(
    encoder: &mut Encoder,
    declaration: &eqiora_lang::NamedDefinitionDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    if let Some(value_type) = declaration.value_type() {
        encoder.field(2, |encoder| {
            super::value_type::encode_value_type(encoder, value_type, budget, 1)
        })?;
    }
    encoder.field(3, |encoder| {
        encode_expression(encoder, declaration.value(), budget, 1)
    })?;
    if let Some(domain) = declaration.domain() {
        encoder.field(4, |encoder| encode_name(encoder, domain, budget))?;
    }
    if let Some(activation) = declaration.activation() {
        encoder.field(5, |encoder| encode_name(encoder, activation, budget))?;
    }
    Ok(())
}

pub(super) fn encode_observable(
    encoder: &mut Encoder,
    declaration: &eqiora_lang::ObservableDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(2, |encoder| {
        super::value_type::encode_value_type(encoder, declaration.value_type(), budget, 1)
    })?;
    if let Some(domain) = declaration.domain() {
        encoder.field(6, |encoder| encode_name(encoder, domain, budget))?;
    }
    match declaration.value().kind() {
        ExprKind::Number(value) => encoder.field(3, |encoder| {
            super::expression::encode_decimal(encoder, value, false)
        }),
        ExprKind::Quantity { value, unit } => {
            encoder.field(3, |encoder| {
                super::expression::encode_decimal(encoder, value, false)
            })?;
            encoder.field(4, |encoder| encode_expression(encoder, unit, budget, 1))
        }
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } if matches!(
            value.kind(),
            ExprKind::Number(_) | ExprKind::Quantity { .. }
        ) =>
        {
            // Canonicalize a signed native literal and a parsed unary minus identically.
            match value.kind() {
                ExprKind::Number(value) => encoder.field(3, |encoder| {
                    super::expression::encode_decimal(encoder, value, true)
                }),
                ExprKind::Quantity { value, unit } => {
                    encoder.field(3, |encoder| {
                        super::expression::encode_decimal(encoder, value, true)
                    })?;
                    encoder.field(4, |encoder| encode_expression(encoder, unit, budget, 1))
                }
                _ => unreachable!("literal guard"),
            }
        }
        _ => encoder.field(5, |encoder| {
            encode_expression(encoder, declaration.value(), budget, 1)
        }),
    }
}
