use super::*;

pub(super) fn encode_expression(
    encoder: &mut Encoder,
    expression: &Expr,
    budget: &mut Budget,
    depth: usize,
) -> Result<(), Diagnostic> {
    budget.account_expression(depth)?;
    match expression.kind() {
        ExprKind::Partial {
            value,
            wrt,
            holding,
        } => {
            encoder.u16(20)?;
            encode_expression(encoder, value, budget, next_depth(depth)?)?;
            encode_path(encoder, wrt, budget)?;
            encoder.u32(as_u32(holding.len(), "partial holding bindings")?)?;
            let mut holding = holding.iter().collect::<Vec<_>>();
            holding.sort_by(|left, right| left.as_str().cmp(right.as_str()));
            for binding in holding {
                encode_path(encoder, binding, budget)?;
            }
            Ok(())
        }
        ExprKind::Case { value, arms } => {
            encoder.u8(18)?;
            encode_expression(encoder, value, budget, next_depth(depth)?)?;
            encoder.u32(as_u32(arms.len(), "case arms")?)?;
            for arm in arms {
                encode_type_path(encoder, arm.pattern(), budget)?;
                encode_expression(encoder, arm.value(), budget, next_depth(depth)?)?;
            }
            Ok(())
        }
        ExprKind::Select {
            condition,
            then_value,
            else_value,
        } => {
            encoder.u16(17)?;
            let child_depth = next_depth(depth)?;
            encoder.field(1, |encoder| {
                encode_expression(encoder, condition, budget, child_depth)
            })?;
            encoder.field(2, |encoder| {
                encode_expression(encoder, then_value, budget, child_depth)
            })?;
            encoder.field(3, |encoder| {
                encode_expression(encoder, else_value, budget, child_depth)
            })
        }
        ExprKind::Reduction {
            operation,
            binder,
            value,
        } => {
            encoder.u16(15)?;
            encoder.u8(match operation {
                eqiora_lang::ReductionOp::Sum => 1,
                eqiora_lang::ReductionOp::Product => 2,
                eqiora_lang::ReductionOp::Min => 3,
                eqiora_lang::ReductionOp::Max => 4,
            })?;
            budget.account_name(binder.member())?;
            encoder.field(1, |encoder| encode_path(encoder, binder.set(), budget))?;
            budget.reduction_binders.push(binder.member().to_owned());
            let result = encoder.field(2, |encoder| {
                encode_expression(encoder, value, budget, next_depth(depth)?)
            });
            budget.reduction_binders.pop();
            result
        }
        ExprKind::Name(name) if budget.reduction_binders.iter().any(|bound| bound == name) => {
            budget.account_name(name)?;
            let ordinal = budget
                .reduction_binders
                .iter()
                .rev()
                .position(|bound| bound == name)
                .expect("bound name");
            encoder.u16(14)?;
            encoder.u32(
                u32::try_from(ordinal)
                    .map_err(|_| source_identity_error("binder depth overflows"))?,
            )
        }
        ExprKind::Boolean(value) => {
            encoder.u16(13)?;
            encoder.u8(u8::from(*value))
        }
        ExprKind::Number(value) => {
            encoder.u16(1)?;
            encode_decimal(encoder, value, false)
        }
        ExprKind::Quantity { value, unit } => {
            encoder.u16(9)?;
            encoder.field(1, |encoder| encode_decimal(encoder, value, false))?;
            encoder.field(2, |encoder| {
                encode_expression(encoder, unit, budget, next_depth(depth)?)
            })
        }
        ExprKind::Array(elements) => {
            encoder.u16(10)?;
            let child_depth = next_depth(depth)?;
            let mut encoded = Vec::with_capacity(elements.len());
            for element in elements {
                let mut element_encoder = Encoder::new(budget.limits.max_canonical_bytes);
                encode_expression(&mut element_encoder, element, budget, child_depth)?;
                let bytes = element_encoder.finish()?;
                budget.account_materialized_bytes(bytes.len())?;
                encoded.push(bytes);
            }
            encoder.field(1, |encoder| encoder.records(&encoded))
        }
        ExprKind::Member { value, member } => {
            encoder.u16(12)?;
            encoder.field(1, |encoder| {
                encode_expression(encoder, value, budget, next_depth(depth)?)
            })?;
            encoder.field(2, |encoder| encode_name(encoder, member, budget))
        }
        ExprKind::Index { value, index } => {
            encoder.u16(11)?;
            let child_depth = next_depth(depth)?;
            encoder.field(1, |encoder| {
                encode_expression(encoder, value, budget, child_depth)
            })?;
            encoder.field(2, |encoder| {
                encode_expression(encoder, index, budget, child_depth)
            })
        }
        ExprKind::Slice {
            value,
            lower,
            upper,
        } => {
            encoder.u16(19)?;
            for (tag, expression) in [(1, value), (2, lower), (3, upper)] {
                encoder.field(tag, |encoder| {
                    encode_expression(encoder, expression, budget, next_depth(depth)?)
                })?;
            }
            Ok(())
        }
        ExprKind::Name(name) => {
            encoder.u16(2)?;
            encoder.field(1, |encoder| encode_name(encoder, name, budget))
        }
        ExprKind::Path(path) => {
            encoder.u16(3)?;
            encoder.field(1, |encoder| encode_path(encoder, path, budget))
        }
        ExprKind::BoundaryPortSelection { port, selector } => {
            encoder.u16(7)?;
            encoder.field(1, |encoder| encode_path(encoder, port, budget))?;
            encoder.field(2, |encoder| {
                encode_boundary_port_selector(encoder, selector, budget)
            })
        }
        ExprKind::Unary { op, value } => {
            if let (
                UnaryOp::Neg,
                ExprKind::Quantity {
                    value: literal,
                    unit,
                },
            ) = (op, value.kind())
            {
                // Native signed decimals and parsed literal negation share one
                // quantity record. Keep the authored node/depth budget intact.
                let literal_depth = next_depth(depth)?;
                budget.account_expression(literal_depth)?;
                encoder.u16(9)?;
                encoder.field(1, |encoder| encode_decimal(encoder, literal, true))?;
                return encoder.field(2, |encoder| {
                    encode_expression(encoder, unit, budget, next_depth(literal_depth)?)
                });
            }
            if let (UnaryOp::Neg, ExprKind::Number(literal)) = (op, value.kind()) {
                budget.account_expression(next_depth(depth)?)?;
                encoder.u16(1)?;
                return encode_decimal(encoder, literal, true);
            }
            encoder.u16(4)?;
            encoder.field(1, |encoder| {
                encoder.u8(match op {
                    UnaryOp::Neg => 1,
                    UnaryOp::Not => 2,
                })
            })?;
            let child_depth = next_depth(depth)?;
            encoder.field(2, |encoder| {
                encode_expression(encoder, value, budget, child_depth)
            })
        }
        ExprKind::Binary { op, left, right } => {
            encoder.u16(5)?;
            encoder.field(1, |encoder| {
                encoder.u8(match op {
                    BinaryOp::Add => 1,
                    BinaryOp::Sub => 2,
                    BinaryOp::Mul => 3,
                    BinaryOp::Div => 4,
                    BinaryOp::Pow => 5,
                    BinaryOp::Equal => 6,
                    BinaryOp::NotEqual => 7,
                    BinaryOp::Less => 8,
                    BinaryOp::LessEqual => 9,
                    BinaryOp::Greater => 10,
                    BinaryOp::GreaterEqual => 11,
                    BinaryOp::And => 12,
                    BinaryOp::Or => 13,
                })
            })?;
            let child_depth = next_depth(depth)?;
            encoder.field(2, |encoder| {
                encode_expression(encoder, left, budget, child_depth)
            })?;
            encoder.field(3, |encoder| {
                encode_expression(encoder, right, budget, child_depth)
            })
        }
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if !callee.is_qualified() && arguments.len() == 1 => {
            // Byte-for-byte compatibility with source identity v1.
            encoder.u16(6)?;
            encoder.field(1, |encoder| encode_name(encoder, callee.as_str(), budget))?;
            let child_depth = next_depth(depth)?;
            encoder.field(2, |encoder| {
                encode_expression(encoder, &arguments[0], budget, child_depth)
            })
        }
        ExprKind::Call { callee, arguments } => {
            let named = arguments.named();
            let tensor = callee.as_str() == "tensor_value" && named.is_some();
            encoder.u16(if named.is_some() && !tensor { 16 } else { 8 })?;
            encoder.field(1, |encoder| encode_type_path(encoder, callee, budget))?;
            let child_depth = next_depth(depth)?;
            let mut encoded = Vec::with_capacity(arguments.expressions().len());
            if let Some(formals) = budget
                .operator_formals
                .get(callee.as_str())
                .cloned()
                .filter(|_| named.is_some() && !tensor)
            {
                let ordered = crate::pure_operator::ordered_arguments(
                    "<source-identity>",
                    expression.range(),
                    formals.iter().map(String::as_str),
                    arguments,
                )?;
                for (slot, value) in ordered.into_iter().enumerate() {
                    let mut argument_encoder = Encoder::new(budget.limits.max_canonical_bytes);
                    argument_encoder.field(1, |encoder| {
                        encoder.u32(as_u32(slot, "operator formal slot")?)
                    })?;
                    argument_encoder.field(2, |encoder| {
                        encode_expression(encoder, value, budget, child_depth)
                    })?;
                    encoded.push(argument_encoder.finish()?);
                }
            } else if let Some(bindings) = named {
                let bindings = if tensor {
                    ["frame", "components"]
                        .iter()
                        .map(|name| {
                            bindings
                                .iter()
                                .find(|binding| binding.name() == *name)
                                .ok_or_else(|| {
                                    source_identity_error(
                                        "tensor_value requires named frame and components",
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    bindings.iter().collect()
                };
                for binding in bindings {
                    let mut argument_encoder = Encoder::new(budget.limits.max_canonical_bytes);
                    if !tensor {
                        argument_encoder
                            .field(1, |encoder| encode_name(encoder, binding.name(), budget))?;
                    }
                    if tensor {
                        encode_expression(
                            &mut argument_encoder,
                            binding.value(),
                            budget,
                            child_depth,
                        )?;
                    } else {
                        argument_encoder.field(2, |encoder| {
                            encode_expression(encoder, binding.value(), budget, child_depth)
                        })?;
                    }
                    encoded.push(argument_encoder.finish()?);
                }
                if !tensor {
                    encoded.sort_unstable();
                }
            } else {
                for argument in arguments.expressions() {
                    let mut argument_encoder = Encoder::new(budget.limits.max_canonical_bytes);
                    encode_expression(&mut argument_encoder, argument, budget, child_depth)?;
                    encoded.push(argument_encoder.finish()?);
                }
            }
            if callee.as_str() == "boundaries" {
                encoded.sort_unstable();
            }
            let materialized = encoded.iter().try_fold(0_usize, |total, value| {
                total.checked_add(value.len()).ok_or_else(|| {
                    source_identity_error("call argument encoding bytes overflow usize")
                })
            })?;
            budget.account_materialized_bytes(materialized)?;
            encoder.field(2, |encoder| encoder.records(&encoded))
        }
        _ => Err(source_identity_error(
            "expression syntax is newer than source identity v1",
        )),
    }
}

pub(super) fn encode_relation(
    encoder: &mut Encoder,
    declaration: &RelationDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    let count = declaration.conditions().map_or(1, |values| values.len());
    if count > budget.limits.max_residuals_per_relation {
        return Err(source_identity_error("Relation exceeds residual limit"));
    }
    encoder.field(1, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(2, |encoder| match declaration.activation() {
        ActivationSyntax::Continuous => encoder.u16(1),
        ActivationSyntax::Named(clock) => {
            encoder.u16(2)?;
            encoder.field(1, |encoder| encode_name(encoder, clock, budget))
        }
        _ => Err(source_identity_error(
            "Activation syntax is newer than source identity v1",
        )),
    })?;
    encoder.field(3, |encoder| {
        encode_optional_name(encoder, declaration.domain(), budget)
    })?;
    encoder.field(4, |encoder| {
        let conditions = match declaration.body() {
            eqiora_lang::RelationBody::Conservation(terms) => {
                encoder.u16(2)?;
                encoder.field(1, |encoder| match terms.storage() {
                    Some(value) => {
                        encoder.u16(1)?;
                        encode_expression(encoder, value, budget, 1)
                    }
                    None => encoder.u16(0),
                })?;
                encoder.field(2, |encoder| {
                    encode_expression(encoder, terms.flux(), budget, 1)
                })?;
                return encoder.field(3, |encoder| {
                    encode_expression(encoder, terms.source(), budget, 1)
                });
            }
            eqiora_lang::RelationBody::Conditions(conditions) => {
                encoder.u16(1)?;
                conditions
            }
        };
        encoder.u32(as_u32(conditions.len(), "Relation residual count")?)?;
        for equation in conditions {
            encoder.field(1, |encoder| {
                encoder.field(1, |encoder| {
                    encode_expression(encoder, equation.left(), budget, 1)
                })?;
                encoder.field(3, |encoder| {
                    encoder.u16(match equation.kind() {
                        eqiora_schema::kernel::RelationConditionKind::Equality => 1,
                        eqiora_schema::kernel::RelationConditionKind::Inequality => 2,
                        eqiora_schema::kernel::RelationConditionKind::Complementarity => 3,
                    })
                })?;
                encoder.field(2, |encoder| {
                    encode_expression(encoder, equation.right(), budget, 1)
                })
            })?;
        }
        Ok(())
    })
}

pub(super) fn encode_relation_family(
    encoder: &mut Encoder,
    declaration: &RelationFamilyDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| {
        encode_relation(encoder, declaration.relation(), budget)
    })?;
    encoder.field(2, |encoder| {
        encode_boundary_family_binder(encoder, declaration.binder(), budget)
    })
}

pub(super) fn encode_decimal(
    encoder: &mut Encoder,
    value: &eqiora_lang::DecimalLiteral,
    negate: bool,
) -> Result<(), Diagnostic> {
    let negative = !value.is_zero() && (value.is_negative() != negate);
    encoder.string(&format!(
        "{}{}e{}",
        if negative { "-" } else { "" },
        value.coefficient(),
        value.exponent10(),
    ))
}
