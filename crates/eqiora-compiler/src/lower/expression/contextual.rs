//! Resolve source decimal literals only after an exact operand context is known.
use super::*;
use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};

pub(super) fn equation(
    file: &str,
    left: &LoweringExpression,
    right: &LoweringExpression,
    bindings: &BTreeMap<String, Binding>,
    support: Option<&SpatialSupport<RawId>>,
) -> Result<(LoweringExpression, LoweringExpression), Diagnostic> {
    let mut resolver = Resolver {
        file,
        bindings,
        support,
        anchors: HashMap::new(),
        resolved: HashMap::new(),
    };
    let target = resolver.anchor(left).or_else(|| resolver.anchor(right));
    Ok((
        resolver.resolve(left, target)?,
        resolver.resolve(right, target)?,
    ))
}

pub(super) fn value(
    file: &str,
    expression: &LoweringExpression,
    bindings: &BTreeMap<String, Binding>,
) -> Result<LoweringExpression, Diagnostic> {
    Resolver {
        file,
        bindings,
        support: None,
        anchors: HashMap::new(),
        resolved: HashMap::new(),
    }
    .resolve(expression, None)
}

pub(super) fn typed_value(
    file: &str,
    expression: &LoweringExpression,
    bindings: &BTreeMap<String, Binding>,
    support: Option<&SpatialSupport<RawId>>,
    scalar_domain: ScalarDomain,
) -> Result<LoweringExpression, Diagnostic> {
    Resolver {
        file,
        bindings,
        support,
        anchors: HashMap::new(),
        resolved: HashMap::new(),
    }
    .resolve(expression, Some(scalar_domain))
}

// Caches live for one equation: support and binding interpretation are fixed,
// while one shared node may legitimately be used in different scalar contexts.
struct Resolver<'a> {
    file: &'a str,
    bindings: &'a BTreeMap<String, Binding>,
    support: Option<&'a SpatialSupport<RawId>>,
    anchors: HashMap<usize, Option<ScalarDomain>>,
    resolved: HashMap<(usize, Option<ScalarDomain>), LoweringExpression>,
}

impl Resolver<'_> {
    fn anchor(&mut self, value: &LoweringExpression) -> Option<ScalarDomain> {
        let key = Arc::as_ptr(&value.node) as usize;
        if let Some(result) = self.anchors.get(&key) {
            return *result;
        }
        let result = self.anchor_node(value);
        self.anchors.insert(key, result);
        result
    }
    fn anchor_node(&mut self, value: &LoweringExpression) -> Option<ScalarDomain> {
        match value.node.as_ref() {
            LoweringExpressionNode::CoordinateMapFactor { .. } => Some(ScalarDomain::Real),
            LoweringExpressionNode::Evaluate { value, .. }
            | LoweringExpressionNode::Pullback { value, .. } => self.anchor(value),
            LoweringExpressionNode::Case { arms, .. } => {
                arms.iter().find_map(|(_, value)| self.anchor(value))
            }
            LoweringExpressionNode::Select {
                then_value,
                else_value,
                ..
            } => self.anchor(then_value).or_else(|| self.anchor(else_value)),
            LoweringExpressionNode::Require { value, .. } => self.anchor(value),
            LoweringExpressionNode::Number(_) => None,
            LoweringExpressionNode::Array(elements) => {
                elements.iter().find_map(|element| self.anchor(element))
            }
            LoweringExpressionNode::Index { value, .. } => self.anchor(value),
            LoweringExpressionNode::Neg(value) => self.anchor(value),
            LoweringExpressionNode::Not(_) => Some(ScalarDomain::Boolean),
            LoweringExpressionNode::Binary { operator, .. }
                if super::super::comparison_operator(*operator).is_some()
                    || matches!(operator, BinaryOp::And | BinaryOp::Or) =>
            {
                Some(ScalarDomain::Boolean)
            }
            LoweringExpressionNode::Binary { left, right, .. }
            | LoweringExpressionNode::Extremum { left, right, .. } => {
                self.anchor(left).or_else(|| self.anchor(right))
            }
            _ => expression_type(self.file, value, self.bindings, self.support)
                .ok()
                .map(|ty| ty.value_type.scalar_domain()),
        }
    }

    fn resolve(
        &mut self,
        expression: &LoweringExpression,
        expected: Option<ScalarDomain>,
    ) -> Result<LoweringExpression, Diagnostic> {
        let key = (Arc::as_ptr(&expression.node) as usize, expected);
        if let Some(result) = self.resolved.get(&key) {
            return Ok(LoweringExpression {
                node: result.node.clone(),
                range: expression.range(),
                structural_parameters: expression.structural_parameters.clone(),
            });
        }
        let result = self.resolve_node(expression, expected)?;
        self.resolved.insert(key, result.clone());
        Ok(result)
    }
    fn resolve_node(
        &mut self,
        expression: &LoweringExpression,
        expected: Option<ScalarDomain>,
    ) -> Result<LoweringExpression, Diagnostic> {
        let file = self.file;
        let node = match expression.node.as_ref() {
            LoweringExpressionNode::CoordinateMapFactor { factor, source, at } => {
                LoweringExpressionNode::CoordinateMapFactor {
                    factor: *factor,
                    source: source.clone(),
                    at: at
                        .iter()
                        .map(|(coordinate, mapped)| {
                            Ok((
                                coordinate.clone(),
                                self.resolve(mapped, Some(ScalarDomain::Real))?,
                            ))
                        })
                        .collect::<Result<_, Diagnostic>>()?,
                }
            }
            LoweringExpressionNode::Pullback { value, source, at } => {
                LoweringExpressionNode::Pullback {
                    value: self.resolve(value, expected)?,
                    source: source.clone(),
                    at: at
                        .iter()
                        .map(|(coordinate, mapped)| {
                            Ok((
                                coordinate.clone(),
                                self.resolve(mapped, Some(ScalarDomain::Real))?,
                            ))
                        })
                        .collect::<Result<_, Diagnostic>>()?,
                }
            }
            LoweringExpressionNode::Evaluate { value, at, side } => {
                LoweringExpressionNode::Evaluate {
                    value: self.resolve(value, expected)?,
                    at: at
                        .iter()
                        .map(|(coordinate, point)| {
                            Ok((
                                coordinate.clone(),
                                self.resolve(point, Some(ScalarDomain::Real))?,
                            ))
                        })
                        .collect::<Result<_, Diagnostic>>()?,
                    side: *side,
                }
            }
            LoweringExpressionNode::Number(value) => {
                return literal(file, expression, value, expected);
            }
            LoweringExpressionNode::Case { value, arms } => {
                let domain = arms
                    .iter()
                    .find_map(|(_, value)| self.anchor(value))
                    .or(expected);
                LoweringExpressionNode::Case {
                    value: self.resolve(value, None)?,
                    arms: arms
                        .iter()
                        .map(|(pattern, value)| Ok((pattern.clone(), self.resolve(value, domain)?)))
                        .collect::<Result<Vec<_>, Diagnostic>>()?,
                }
            }
            LoweringExpressionNode::Select {
                condition,
                then_value,
                else_value,
            } => {
                let domain = self
                    .anchor(then_value)
                    .or_else(|| self.anchor(else_value))
                    .or(expected);
                LoweringExpressionNode::Select {
                    condition: self.resolve(condition, Some(ScalarDomain::Boolean))?,
                    then_value: self.resolve(then_value, domain)?,
                    else_value: self.resolve(else_value, domain)?,
                }
            }
            LoweringExpressionNode::Require { condition, value } => {
                LoweringExpressionNode::Require {
                    condition: self.resolve(condition, Some(ScalarDomain::Boolean))?,
                    value: self.resolve(value, expected)?,
                }
            }
            LoweringExpressionNode::Finite {
                operation,
                arguments,
            } => LoweringExpressionNode::Finite {
                operation: *operation,
                arguments: arguments
                    .iter()
                    .map(|value| self.resolve(value, None))
                    .collect::<Result<Vec<_>, _>>()?,
            },
            LoweringExpressionNode::Piecewise { name, arguments } => {
                LoweringExpressionNode::Piecewise {
                    name: name.clone(),
                    arguments: arguments
                        .iter()
                        .map(|value| self.resolve(value, Some(ScalarDomain::Real)))
                        .collect::<Result<Vec<_>, _>>()?,
                }
            }
            LoweringExpressionNode::Not(value) => {
                LoweringExpressionNode::Not(self.resolve(value, None)?)
            }
            LoweringExpressionNode::Neg(value) => {
                if let LoweringExpressionNode::Number(number) = value.node.as_ref() {
                    let spelling = number.canonical_text();
                    let signed = if let Some(value) = spelling.strip_prefix('-') {
                        value.to_owned()
                    } else {
                        format!("-{spelling}")
                    };
                    let number = eqiora_lang::DecimalLiteral::parse(&signed).map_err(|error| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            expression.range(),
                            error.to_string(),
                        )
                    })?;
                    return literal(file, expression, &number, expected);
                }
                LoweringExpressionNode::Neg(self.resolve(value, expected)?)
            }
            LoweringExpressionNode::Extremum {
                minimum,
                left,
                right,
            } => {
                let domain = self
                    .anchor(left)
                    .or_else(|| self.anchor(right))
                    .or(expected);
                LoweringExpressionNode::Extremum {
                    minimum: *minimum,
                    left: self.resolve(left, domain)?,
                    right: self.resolve(right, domain)?,
                }
            }
            LoweringExpressionNode::Binary {
                operator,
                left,
                right,
            } => {
                let domain = if matches!(operator, BinaryOp::And | BinaryOp::Or) {
                    None
                } else {
                    self.anchor(left)
                        .or_else(|| self.anchor(right))
                        .or_else(|| {
                            if super::super::comparison_operator(*operator).is_some() {
                                None
                            } else {
                                expected
                            }
                        })
                };
                LoweringExpressionNode::Binary {
                    operator: *operator,
                    left: self.resolve(left, domain)?,
                    right: self.resolve(
                        right,
                        if *operator == BinaryOp::Pow {
                            None
                        } else {
                            domain
                        },
                    )?,
                }
            }
            LoweringExpressionNode::IntegerCall {
                operator,
                arguments,
            } => {
                if arguments.len() != operator.arity() {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        expression.range(),
                        "invalid integer operation arity",
                    ));
                }
                LoweringExpressionNode::IntegerCall {
                    operator: *operator,
                    arguments: arguments
                        .iter()
                        .map(|argument| self.resolve(argument, Some(operator.operand_domain())))
                        .collect::<Result<_, _>>()?,
                }
            }
            LoweringExpressionNode::Array(elements) => LoweringExpressionNode::Array(
                elements
                    .iter()
                    .map(|value| self.resolve(value, expected))
                    .collect::<Result<_, _>>()?,
            ),
            LoweringExpressionNode::Index { value, index } => LoweringExpressionNode::Index {
                value: self.resolve(value, expected)?,
                index: *index,
            },
            LoweringExpressionNode::Complex { real, imag } => LoweringExpressionNode::Complex {
                real: self.resolve(real, Some(ScalarDomain::Real))?,
                imag: self.resolve(imag, Some(ScalarDomain::Real))?,
            },
            LoweringExpressionNode::Call { callee, argument } => LoweringExpressionNode::Call {
                callee: callee.clone(),
                argument: self.resolve(argument, None)?,
            },
            LoweringExpressionNode::Boundary {
                operation,
                argument,
                on,
                from,
            } => LoweringExpressionNode::Boundary {
                operation: *operation,
                argument: self.resolve(argument, None)?,
                on: on.clone(),
                from: from.clone(),
            },
            LoweringExpressionNode::Partial { value, wrt } => LoweringExpressionNode::Partial {
                value: self.resolve(value, Some(ScalarDomain::Real))?,
                wrt: wrt.clone(),
            },
            LoweringExpressionNode::Sample { value, clock } => LoweringExpressionNode::Sample {
                value: self.resolve(value, None)?,
                clock: clock.clone(),
            },
            LoweringExpressionNode::Property { release, arguments } => {
                LoweringExpressionNode::Property {
                    release: release.clone(),
                    arguments: arguments
                        .iter()
                        .map(|value| self.resolve(value, None))
                        .collect::<Result<_, _>>()?,
                }
            }
            LoweringExpressionNode::PureOperator {
                definition,
                arguments,
            } => LoweringExpressionNode::PureOperator {
                definition: definition.clone(),
                arguments: arguments
                    .iter()
                    .map(|value| self.resolve(value, None))
                    .collect::<Result<_, _>>()?,
            },
            LoweringExpressionNode::Tensor {
                operation,
                arguments,
            } => LoweringExpressionNode::Tensor {
                operation: operation.clone(),
                arguments: arguments
                    .iter()
                    .map(|value| self.resolve(value, None))
                    .collect::<Result<_, _>>()?,
            },
            _ => return Ok(expression.clone()),
        };
        Ok(LoweringExpression {
            node: Arc::new(node),
            range: expression.range(),
            structural_parameters: expression.structural_parameters.clone(),
        })
    }
}

fn literal(
    file: &str,
    expression: &LoweringExpression,
    value: &eqiora_lang::DecimalLiteral,
    expected: Option<ScalarDomain>,
) -> Result<LoweringExpression, Diagnostic> {
    let literal = if expected == Some(ScalarDomain::Integer) {
        ValueLiteral::from_integer(
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("admitted numeric scalar type"),
            value.to_i64().map_err(|error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    error.to_string(),
                )
            })?,
        )
    } else {
        ValueLiteral::from_real(
            ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .expect("admitted numeric scalar type"),
            value.to_f64().map_err(|error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    error.to_string(),
                )
            })?,
        )
    }
    .map_err(|error| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            error.to_string(),
        )
    })?;
    Ok(LoweringExpression::literal(literal, expression.range()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_shares_nodes_but_separates_scalar_contexts() {
        let range = TextRange::new(0, 1);
        let literal =
            LoweringExpression::number(eqiora_lang::DecimalLiteral::parse("2").unwrap(), range);
        let bindings = BTreeMap::new();
        let mut resolver = Resolver {
            file: "context.eqi",
            bindings: &bindings,
            support: None,
            anchors: HashMap::new(),
            resolved: HashMap::new(),
        };
        let real = resolver
            .resolve(&literal, Some(ScalarDomain::Real))
            .unwrap();
        let integer = resolver
            .resolve(&literal, Some(ScalarDomain::Integer))
            .unwrap();
        let LoweringExpressionNode::Literal(real_value) = real.node.as_ref() else {
            panic!("real literal")
        };
        let LoweringExpressionNode::Literal(integer_value) = integer.node.as_ref() else {
            panic!("integer literal")
        };
        assert_eq!(real_value.real_scalar_value().unwrap().value(), 2.0);
        assert_eq!(integer_value.integer_scalar_value(), Some(2));
        assert!(!Arc::ptr_eq(&real.node, &integer.node));
        let mut diamond = literal;
        for _ in 0..12 {
            diamond = LoweringExpression::binary(BinaryOp::Add, diamond.clone(), diamond, range);
        }
        let mut normalized = resolver
            .resolve(&diamond, Some(ScalarDomain::Real))
            .unwrap();
        for _ in 0..12 {
            let LoweringExpressionNode::Binary { left, right, .. } = normalized.node.as_ref()
            else {
                panic!("binary diamond")
            };
            assert!(Arc::ptr_eq(&left.node, &right.node));
            normalized = left.clone();
        }
        assert_eq!(resolver.resolved.len(), 14);
    }
}
