//! Full operand admission through the shared Kernel typing rules.

use super::*;

pub(in crate::lower) fn relation_support(
    file: &str,
    range: TextRange,
    name: &str,
    bindings: &BTreeMap<String, Binding>,
) -> Result<SpatialSupport<RawId>, Diagnostic> {
    match bindings.get(name) {
        Some(Binding::Domain(
            id,
            DomainContract::Spatial {
                dimensions: Some(dimensions),
                parent: None,
            },
        )) => Ok(SpatialSupport::Volume {
            domain: id.erase(),
            dimensions: *dimensions,
        }),
        Some(Binding::Domain(
            id,
            DomainContract::Spatial {
                parent: Some(parent),
                ..
            },
        )) => {
            let Some(Binding::Domain(
                parent_id,
                DomainContract::Spatial {
                    dimensions: Some(dimensions),
                    parent: None,
                },
            )) = bindings.get(parent)
            else {
                return Err(unresolved(
                    file,
                    range,
                    parent,
                    "boundary parent volume Domain",
                ));
            };
            Ok(SpatialSupport::Boundary {
                domain: id.erase(),
                parent: parent_id.erase(),
                dimensions: *dimensions,
            })
        }
        _ => Err(unresolved(file, range, name, "spatial Relation Domain")),
    }
}

pub(super) fn expression_type(
    file: &str,
    expression: &LoweringExpression,
    bindings: &BTreeMap<String, Binding>,
    support: Option<&SpatialSupport<RawId>>,
) -> Result<ExpressionType<RawId>, Diagnostic> {
    expression_type_cached(file, expression, bindings, support, &mut HashMap::new())
}

fn expression_type_cached(
    file: &str,
    expression: &LoweringExpression,
    bindings: &BTreeMap<String, Binding>,
    support: Option<&SpatialSupport<RawId>>,
    cache: &mut HashMap<usize, ExpressionType<RawId>>,
) -> Result<ExpressionType<RawId>, Diagnostic> {
    let key = Arc::as_ptr(&expression.node) as usize;
    if let Some(inferred) = cache.get(&key) {
        return Ok(inferred.clone());
    }
    let mut infer = |operand| expression_type_cached(file, operand, bindings, support, cache);
    let violation = |error| spatial_type_error(file, expression, error);
    let inferred = match expression.node.as_ref() {
        LoweringExpressionNode::Partial { value, wrt } => {
            let value = infer(value)?;
            let selected = infer(&LoweringExpression::name(wrt.clone(), expression.range()))?;
            super::partial::result_type(&value, &selected).map_err(|message| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    message,
                )
            })
        }
        LoweringExpressionNode::IntegerCall {
            operator,
            arguments,
        } => {
            let operands = arguments
                .iter()
                .map(&mut infer)
                .collect::<Result<Vec<_>, _>>()?;
            operator.infer(&operands).map_err(violation)
        }
        LoweringExpressionNode::Array(elements) => {
            let elements = elements
                .iter()
                .map(&mut infer)
                .collect::<Result<Vec<_>, _>>()?;
            ExpressionType::array(&elements).map_err(violation)
        }
        LoweringExpressionNode::Index { value, index } => {
            ExpressionType::index(infer(value)?, *index).map_err(violation)
        }
        LoweringExpressionNode::Complex { real, imag } => {
            ExpressionType::complex(infer(real)?, infer(imag)?).map_err(violation)
        }
        LoweringExpressionNode::Number(_) => {
            Ok(ExpressionType::scalar(DimExponents::DIMENSIONLESS, None))
        }
        LoweringExpressionNode::Literal(value) => {
            Ok(ExpressionType::new(value.value_type().clone(), None))
        }
        LoweringExpressionNode::Name(name) if name == "time" => {
            Ok(ExpressionType::scalar(time_dimension(), None))
        }
        LoweringExpressionNode::Name(name) => match bindings.get(name) {
            Some(Binding::Field(_, contract)) => {
                let value_type =
                    resolve_field_contract(file, expression.range(), contract, bindings)?;
                let support = contract
                    .domain
                    .as_deref()
                    .map(|name| relation_support(file, expression.range(), name, bindings))
                    .transpose()?;
                Ok(ExpressionType::new(value_type, support))
            }
            Some(Binding::Parameter(_, value_type)) => {
                Ok(ExpressionType::new(value_type.clone(), None))
            }
            Some(Binding::Port(_, contract)) => {
                match resolve_port_contract(file, expression.range(), contract, bindings)? {
                    ResolvedPortContract::Signal {
                        value_type,
                        support,
                        ..
                    } => Ok(ExpressionType::new(value_type, support)),
                    ResolvedPortContract::ScalarPhysical { .. } => Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        expression.range(),
                        format!(
                            "scalar physical Port `{name}` requires a declared quantity member (`port.member`)"
                        ),
                    )),
                    ResolvedPortContract::BoundaryPhysical { .. } => Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        expression.range(),
                        format!(
                            "field-physical Port `{name}` requires a declared quantity member (`port.member`)"
                        ),
                    )),
                }
            }
            _ => Err(unresolved(
                file,
                expression.range(),
                name,
                "expression symbol",
            )),
        },
        LoweringExpressionNode::Neg(value) => infer(value),
        LoweringExpressionNode::Not(value) => infer(value)?.logical_not().map_err(violation),
        LoweringExpressionNode::Case { value, arms } => {
            let selector = infer(value)?;
            crate::enumeration::validate_patterns(
                file,
                expression.range(),
                &selector.value_type,
                &arms
                    .iter()
                    .map(|(pattern, _)| pattern.clone())
                    .collect::<Vec<_>>(),
            )?;
            let branches = arms
                .iter()
                .map(|(_, value)| infer(value))
                .collect::<Result<Vec<_>, _>>()?;
            crate::enumeration::result_type(selector, &branches).map_err(violation)
        }
        LoweringExpressionNode::Select {
            condition,
            then_value,
            else_value,
        } => infer(condition)?
            .select(infer(then_value)?, infer(else_value)?)
            .map_err(violation),
        LoweringExpressionNode::Require { condition, value } => {
            infer(condition)?.require(infer(value)?).map_err(violation)
        }
        LoweringExpressionNode::Piecewise { name, arguments } => {
            crate::math::piecewise::result_type(
                name,
                &arguments.iter().map(infer).collect::<Result<Vec<_>, _>>()?,
            )
            .map_err(violation)
        }
        LoweringExpressionNode::Extremum { left, right, .. } => infer(left)?
            .ordered_selection(infer(right)?)
            .map_err(violation),
        LoweringExpressionNode::Binary {
            operator,
            left,
            right,
        } => {
            let left_type = infer(left)?;
            let right_type = infer(right)?;
            match operator {
                BinaryOp::And => left_type.and(right_type),
                BinaryOp::Or => left_type.or(right_type),
                op if super::super::comparison_operator(*op).is_some() => {
                    left_type.compare(super::super::comparison_operator(*op).unwrap(), right_type)
                }
                BinaryOp::Add => left_type.sum(right_type),
                BinaryOp::Sub => typing::additive(&left_type, &right_type),
                BinaryOp::Mul => typing::multiply(&left_type, &right_type),
                BinaryOp::Div => typing::divide(&left_type, &right_type),
                BinaryOp::Pow => {
                    let exponent = lowering_integer_literal(right).ok_or_else(|| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            right.range(),
                            "power exponent must be an i32 integer literal",
                        )
                    })?;
                    typing::power(&left_type, exponent)
                }
                _ => unreachable!("comparison handled above"),
            }
            .map_err(violation)
        }
        LoweringExpressionNode::Call { callee, argument } => {
            if callee == "sin" {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    "bare `sin` is not language vocabulary; use compiler-owned `math.sin`",
                ));
            }
            if callee.starts_with("math.") && !matches!(callee.as_str(), "math.sin" | "math.sqrt") {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    format!("unknown compiler-owned scalar mathematics member `{callee}`"),
                ));
            }
            if callee == "coordinate" {
                let axis = lowering_integer_literal(argument)
                    .and_then(|axis| usize::try_from(axis).ok())
                    .ok_or_else(|| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            argument.range(),
                            "coordinate(...) requires a non-negative integer literal axis",
                        )
                    })?;
                return typing::coordinate(axis, support).map_err(violation);
            }
            let mut boundary_name = None;
            let port_contract = if let LoweringExpressionNode::Name(name) = argument.node.as_ref() {
                if let Some(Binding::Port(_, contract)) = bindings.get(name) {
                    if let PortContract::BoundaryPhysical { boundary, .. } = contract {
                        boundary_name = Some(boundary.as_str());
                    }
                    Some(resolve_port_contract(
                        file,
                        argument.range(),
                        contract,
                        bindings,
                    )?)
                } else {
                    None
                }
            } else {
                None
            };
            match (callee.as_str(), port_contract) {
                ("across", Some(ResolvedPortContract::ScalarPhysical { across_type, .. })) => {
                    return Ok(ExpressionType::new(across_type, None));
                }
                ("through", Some(ResolvedPortContract::ScalarPhysical { through_type, .. })) => {
                    return Ok(ExpressionType::new(through_type, None));
                }
                (
                    "trace" | "flux",
                    Some(ResolvedPortContract::BoundaryPhysical {
                        trace_type,
                        flux_type,
                        ..
                    }),
                ) => {
                    let support = relation_support(
                        file,
                        argument.range(),
                        boundary_name.expect("resolved boundary contract retains its name"),
                        bindings,
                    )?;
                    return Ok(ExpressionType::new(
                        if callee == "trace" {
                            trace_type
                        } else {
                            flux_type
                        },
                        Some(support),
                    ));
                }
                ("across" | "through" | "flux", _) => {
                    let family = if callee == "flux" { "field" } else { "scalar" };
                    let message = match argument.node.as_ref() {
                        LoweringExpressionNode::Name(name) => {
                            format!("`{name}` is not a {family} physical Port")
                        }
                        _ => {
                            format!("`{callee}(...)` requires one bare {family} physical Port name")
                        }
                    };
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        argument.range(),
                        message,
                    ));
                }
                _ => {}
            }
            if matches!(callee.as_str(), "pre" | "next")
                && !matches!(argument.node.as_ref(), LoweringExpressionNode::Name(name) if matches!(bindings.get(name), Some(Binding::Field(..))))
            {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    argument.range(),
                    "time operator requires one Field name",
                ));
            }
            if callee == "period" {
                if !matches!(argument.node.as_ref(),LoweringExpressionNode::Name(name) if matches!(bindings.get(name),Some(Binding::Clock(..))))
                {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        argument.range(),
                        "period requires one clock name",
                    ));
                }
                return Ok(ExpressionType::new(
                    eqiora_core::ValueType::scalar(
                        eqiora_core::ScalarDomain::Real,
                        crate::dimensions::time_dimension(),
                    )
                    .expect("admitted numeric scalar type"),
                    None,
                ));
            }
            let operand = infer(argument)?;
            match callee.as_str() {
                "grad" => typing::gradient(&operand),
                "div" => typing::divergence(&operand),
                "symmetric_part" => typing::symmetric_part(&operand),
                "isotropic_lift" => typing::isotropic_lift(&operand),
                "trace" => typing::trace(&operand, support),
                "normal" => typing::normal(&operand, support),
                "math.sin" => typing::unary_math(UnaryMathFunction::Sin, &operand),
                "math.sqrt" => typing::unary_math(UnaryMathFunction::Sqrt, &operand),
                "derivative" => typing::time_derivative(&operand),
                "pre" | "next" | "hold" => Ok(operand),
                _ => {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        expression.range(),
                        format!("unknown scalar operator `{callee}`"),
                    ));
                }
            }
            .map_err(violation)
        }
        LoweringExpressionNode::Sample { value, .. } => infer(value),
        LoweringExpressionNode::Property { release, arguments } => {
            let arguments = arguments.iter().map(infer).collect::<Result<Vec<_>, _>>()?;
            super::property::infer(release, &arguments).map_err(|message| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    message,
                )
            })
        }
        LoweringExpressionNode::PureOperator {
            definition,
            arguments,
        } => {
            let arguments = arguments.iter().map(infer).collect::<Result<Vec<_>, _>>()?;
            definition
                .instantiate(&arguments)
                .map(|application| application.result_type().clone())
                .map_err(|error| spatial_type_error(file, expression, error))
        }
        LoweringExpressionNode::InvalidValue(message) => Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            *message,
        )),
        LoweringExpressionNode::UnknownMath(path) => Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            format!("unknown compiler-owned scalar mathematics member `{path}`"),
        )),
        LoweringExpressionNode::Unsupported => Err(source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            expression.range(),
            "expression syntax is newer than this compiler",
        )),
    }?;
    cache.insert(key, inferred.clone());
    Ok(inferred)
}
