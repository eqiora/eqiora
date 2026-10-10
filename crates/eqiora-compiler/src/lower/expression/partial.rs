//! Formalize explicit scalar expressions without cutting alias dependencies.
use super::*;
use eqiora_schema::kernel::pure_operator::{
    CalculusBuilder, CalculusNode, CalculusNodeId, PureValueClass,
};

pub(crate) fn result_type<I: Clone + PartialEq>(
    value: &ExpressionType<I>,
    selected: &ExpressionType<I>,
) -> Result<ExpressionType<I>, &'static str> {
    if [value, selected].into_iter().any(|ty| {
        !ty.shape().is_scalar() || ty.value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
    }) {
        return Err("partial requires real scalar expressions and independent values");
    }
    if value.support.is_some() && selected.support.is_some() && value.support != selected.support {
        return Err("partial operands have incompatible spatial supports");
    }
    let dimension = value
        .dimension()
        .div(selected.dimension())
        .ok_or("partial result dimension exceeds exact exponent bounds")?;
    // Even an independent zero is requested on the selected coordinate support.
    // This agrees with the retained operator's coordinate argument and native typing.
    Ok(ExpressionType::scalar(
        dimension,
        value.support.clone().or_else(|| selected.support.clone()),
    ))
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Input {
    Name(String),
    TimeDerivative(String, std::num::NonZeroU32),
    Coordinate(String, String, usize),
    Pullback(usize),
}
pub(super) fn input(value: &LoweringExpression) -> Option<Input> {
    match value.node.as_ref() {
        LoweringExpressionNode::Name(name) => Some(Input::Name(name.clone())),
        LoweringExpressionNode::Call { callee, argument } if callee == "derivative" => {
            match input(argument)? {
                Input::Name(name) => Some(Input::TimeDerivative(name, std::num::NonZeroU32::MIN)),
                Input::TimeDerivative(name, order) => {
                    Some(Input::TimeDerivative(name, order.checked_add(1)?))
                }
                _ => None,
            }
        }
        LoweringExpressionNode::Coordinate {
            support,
            factor,
            axis,
        } => Some(Input::Coordinate(support.clone(), factor.clone(), *axis)),
        LoweringExpressionNode::Pullback { .. } => {
            Some(Input::Pullback(Arc::as_ptr(&value.node) as usize))
        }
        _ => None,
    }
}

impl ExpressionLowerer<'_> {
    pub(super) fn lower_partial(
        &mut self,
        expression: &LoweringExpression,
        value: &LoweringExpression,
        selected: &LoweringExpression,
    ) -> Result<TypedExpression, Diagnostic> {
        // Formalization creates temporary reciprocals and chain-rule terms.
        // Their pointer keys must never escape the lifetime of this call.
        let outer_cache = std::mem::take(&mut self.cache);
        let lowered = self.lower_partial_scoped(expression, value, selected);
        self.cache = outer_cache;
        lowered
    }

    fn lower_partial_scoped(
        &mut self,
        expression: &LoweringExpression,
        value: &LoweringExpression,
        selected: &LoweringExpression,
    ) -> Result<TypedExpression, Diagnostic> {
        if value
            .referenced_names()
            .iter()
            .any(|name| matches!(self.bindings.get(name), Some(Binding::Observable(..))))
        {
            return Err(error(
                self.file,
                expression,
                "partial of an Observable requires differentiation through its retained definition",
            ));
        }
        let value_type = types::expression_type(self.file, value, self.bindings, None)?;
        let input_type = types::expression_type(self.file, selected, self.bindings, None)?;
        let result = result_type(&value_type, &input_type)
            .map_err(|message| error(self.file, expression, message))?;
        if matches!(input(selected), Some(Input::Coordinate(..)))
            && let LoweringExpressionNode::Pullback { value, source, at } = value.node.as_ref()
        {
            // Each target coordinate contributes its own dimensioned Jacobian
            // entry. Keep the map on the differentiated value, rather than
            // treating mapped coordinates as independent formal inputs.
            let terms = at.iter().map(|(coordinate, mapped)| {
                let derivative = LoweringExpression::partial(
                    value.clone(),
                    coordinate.clone(),
                    expression.range(),
                );
                let pulled = LoweringExpression::pullback(
                    derivative,
                    source.clone(),
                    at.clone(),
                    expression.range(),
                );
                let jacobian = LoweringExpression::partial(
                    mapped.clone(),
                    selected.clone(),
                    expression.range(),
                );
                LoweringExpression::binary(BinaryOp::Mul, pulled, jacobian, expression.range())
            });
            let sum = terms
                .reduce(|left, right| {
                    LoweringExpression::binary(BinaryOp::Add, left, right, expression.range())
                })
                .ok_or_else(|| {
                    error(self.file, expression, "pullback has no target coordinates")
                })?;
            return self.lower(&sum);
        }
        if matches!(input(selected), Some(Input::Coordinate(..)))
            && let LoweringExpressionNode::Name(name) = value.node.as_ref()
            && matches!(self.bindings.get(name), Some(Binding::Field(_, contract)) if contract.domain.is_some() && matches!(contract.activation, ActivationSyntax::Continuous))
        {
            let value = self.lower_name(value, name)?;
            let wrt = self.lower(selected)?;
            let id = self
                .builder
                .coordinate_partial(value.id, wrt.id)
                .map_err(|failure| self.builder_error(expression, failure))?;
            return Ok(TypedExpression {
                id,
                dimension: result.dimension(),
            });
        }
        let derived = self.formal_derivative_expression(expression, value, selected, false)?;
        self.lower(&derived)
    }

    pub(super) fn formal_derivative_expression(
        &self,
        expression: &LoweringExpression,
        value: &LoweringExpression,
        selected: &LoweringExpression,
        total_time: bool,
    ) -> Result<LoweringExpression, Diagnostic> {
        let value_type = types::expression_type(self.file, value, self.bindings, None)?;
        let input_type = types::expression_type(self.file, selected, self.bindings, None)?;
        let result = result_type(&value_type, &input_type)
            .map_err(|message| error(self.file, expression, message))?;
        let mut inputs = vec![selected.clone()];
        let mut names = BTreeMap::from([(
            input(selected).ok_or_else(|| {
                error(
                    self.file,
                    expression,
                    "partial selector is not an independent input",
                )
            })?,
            0u16,
        )]);
        let mut pending = vec![value];
        let mut coordinate_derivative = matches!(input(selected), Some(Input::Coordinate(..)));
        let mut nested_partial = false;
        let mut literals = BTreeMap::new();
        let mut divisors = BTreeMap::new();
        // The borrowed root keeps every source Arc alive throughout this call.
        // Both pointer-keyed maps are local to this formalization and its scope.
        let mut visited = BTreeSet::new();
        while let Some(value) = pending.pop() {
            if !visited.insert(Arc::as_ptr(&value.node) as usize) {
                continue;
            }
            match value.node.as_ref() {
                _ if input(value).is_some() => {
                    let key = input(value).expect("matched independent input");
                    if let std::collections::btree_map::Entry::Vacant(entry) = names.entry(key) {
                        let index = input_slot(inputs.len())
                            .map_err(|message| error(self.file, expression, message))?;
                        entry.insert(index);
                        inputs.push(value.clone());
                    }
                }
                LoweringExpressionNode::Partial { value, wrt } => {
                    nested_partial = true;
                    coordinate_derivative |= matches!(input(wrt), Some(Input::Coordinate(..)));
                    pending.extend([value, wrt]);
                }
                LoweringExpressionNode::Literal(_) => {
                    let key = Arc::as_ptr(&value.node) as usize;
                    if let std::collections::btree_map::Entry::Vacant(entry) = literals.entry(key) {
                        let index = input_slot(inputs.len())
                            .map_err(|message| error(self.file, expression, message))?;
                        entry.insert(index);
                        inputs.push(value.clone());
                    }
                }
                LoweringExpressionNode::Neg(value) => pending.push(value),
                LoweringExpressionNode::Binary {
                    operator: BinaryOp::Div,
                    left,
                    right,
                } => {
                    let reciprocal = reciprocal_literal(self.file, right)?;
                    let index = input_slot(inputs.len())
                        .map_err(|message| error(self.file, expression, message))?;
                    divisors.insert(Arc::as_ptr(&value.node) as usize, index);
                    inputs.push(reciprocal);
                    pending.push(left);
                }
                LoweringExpressionNode::Binary { left, right, .. } => pending.extend([right, left]),
                LoweringExpressionNode::PureOperator { arguments, .. } => {
                    pending.extend(arguments.iter().rev())
                }
                _ => {
                    return Err(error(
                        self.file,
                        expression,
                        "partial admits explicit scalar polynomial arithmetic and operator composition",
                    ));
                }
            }
        }
        let mut directions = Vec::new();
        for (index, value) in inputs.iter().enumerate() {
            if matches!(value.node.as_ref(), LoweringExpressionNode::Pullback { .. }) {
                if !total_time && (!coordinate_derivative || nested_partial) {
                    return Err(error(
                        self.file,
                        expression,
                        "composed pullback differentiation requires an explicit first coordinate derivative",
                    ));
                }
                directions.push(index);
            }
        }
        if total_time {
            for (index, value) in inputs.iter().enumerate() {
                let name = match input(value) {
                    Some(Input::Name(name) | Input::TimeDerivative(name, _)) => name,
                    _ => continue,
                };
                if let Some(Binding::Field(_, contract)) = self.bindings.get(&name) {
                    if nested_partial && contract.domain.is_some() {
                        return Err(error(
                            self.file,
                            expression,
                            "time differentiation of unknown Field coordinate partials requires a mixed-derivative representation",
                        ));
                    }
                    directions.push(index);
                }
            }
        } else if coordinate_derivative {
            for (index, value) in inputs.iter().enumerate() {
                if let LoweringExpressionNode::Name(name) = value.node.as_ref()
                    && let Some(Binding::Field(_, contract)) = self.bindings.get(name)
                    && contract.domain.is_some()
                {
                    if nested_partial
                        || !matches!(contract.activation, ActivationSyntax::Continuous)
                    {
                        return Err(error(
                            self.file,
                            expression,
                            "coordinate Field derivatives require a continuous Field and the admitted first-order profile",
                        ));
                    }
                    directions.push(index);
                }
            }
        }
        let formal_types = inputs
            .iter()
            .map(|value| types::expression_type(self.file, value, self.bindings, None))
            .collect::<Result<Vec<_>, _>>()?;
        let class = |dimension| {
            PureValueClass::invariant_scalar()
                .with_dimension(dimension)
                .with_scalar_domain(eqiora_core::ScalarDomain::Real)
        };
        let mut formals = formal_types
            .iter()
            .map(|ty| {
                result_type(ty, &input_type)
                    .map_err(|message| error(self.file, expression, message))?;
                class(ty.dimension())
                    .map_err(|failure| error(self.file, expression, failure.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for index in &directions {
            input_slot(formals.len()).map_err(|message| error(self.file, expression, message))?;
            let ty = result_type(&formal_types[*index], &input_type)
                .map_err(|message| error(self.file, expression, message))?;
            formals.push(
                class(ty.dimension())
                    .map_err(|failure| error(self.file, expression, failure.to_string()))?,
            );
        }
        let mut calculus = CalculusBuilder::new(
            formals,
            class(result.dimension())
                .map_err(|failure| error(self.file, expression, failure.to_string()))?,
        )
        .map_err(|failure| error(self.file, expression, failure.to_string()))?;
        let root = scalar(
            self.file,
            value,
            &names,
            &literals,
            &divisors,
            &mut calculus,
            &mut BTreeMap::new(),
        )?;
        let mut derivative = calculus
            .partial(root, 0)
            .map_err(|failure| error(self.file, expression, failure.to_string()))?;
        // Chain the common exact polynomial derivative with each retained Field
        // or pullback derivative. Total time differentiation supplies their exact
        // rates; spatial differentiation supplies their coordinate directions.
        // Numerical basis choice stays outside calculus.
        for (direction, index) in directions.iter().enumerate() {
            let coefficient = calculus
                .partial(root, *index as u16)
                .map_err(|failure| error(self.file, expression, failure.to_string()))?;
            let value = calculus
                .push(CalculusNode::FormalComponent {
                    formal: (inputs.len() + direction) as u16,
                    axes: Box::new([]),
                })
                .map_err(|failure| error(self.file, expression, failure.to_string()))?;
            let term = calculus
                .push(CalculusNode::Mul(coefficient, value))
                .map_err(|failure| error(self.file, expression, failure.to_string()))?;
            derivative = calculus
                .push(CalculusNode::Add(derivative, term))
                .map_err(|failure| error(self.file, expression, failure.to_string()))?;
        }
        let definition = calculus
            .finish(derivative)
            .map_err(|failure| error(self.file, expression, failure.to_string()))?;
        let mut arguments = inputs.clone();
        for index in directions {
            arguments.push(if total_time {
                LoweringExpression::call(
                    "derivative".to_owned(),
                    inputs[index].clone(),
                    expression.range(),
                )
            } else {
                LoweringExpression::partial(
                    inputs[index].clone(),
                    selected.clone(),
                    expression.range(),
                )
            });
        }
        Ok(LoweringExpression::pure_operator(
            definition,
            arguments,
            expression.range(),
        ))
    }
}

fn reciprocal_literal(
    file: &str,
    expression: &LoweringExpression,
) -> Result<LoweringExpression, Diagnostic> {
    let mut value = expression;
    let mut sign = 1.0;
    while let LoweringExpressionNode::Neg(inner) = value.node.as_ref() {
        value = inner;
        sign = -sign;
    }
    let invalid = || {
        error(
            file,
            expression,
            "polynomial partial division requires a fixed nonzero real literal denominator",
        )
    };
    let LoweringExpressionNode::Literal(literal) = value.node.as_ref() else {
        return Err(invalid());
    };
    let quantity = literal
        .real_scalar_value()
        .filter(|value| value.value() != 0.0)
        .ok_or_else(invalid)?;
    let dimension = quantity.dim().pow(-1, 1).ok_or_else(|| {
        error(
            file,
            expression,
            "partial reciprocal dimension exceeds exact bounds",
        )
    })?;
    let reciprocal = eqiora_core::ValueLiteral::from_real(
        literal
            .value_type()
            .clone()
            .with_dimension(dimension)
            .map_err(|failure| error(file, expression, failure.to_string()))?,
        sign / quantity.value(),
    )
    .map_err(|failure| error(file, expression, failure.to_string()))?;
    Ok(LoweringExpression::literal(reciprocal, expression.range))
}

fn input_slot(count: usize) -> Result<u16, &'static str> {
    if count >= eqiora_schema::kernel::pure_operator::MAX_FORMALS {
        return Err("partial input occurrences exceed the calculus formal bound");
    }
    u16::try_from(count).map_err(|_| "partial formal index overflows")
}

fn scalar(
    file: &str,
    expression: &LoweringExpression,
    names: &BTreeMap<Input, u16>,
    literals: &BTreeMap<usize, u16>,
    divisors: &BTreeMap<usize, u16>,
    builder: &mut CalculusBuilder,
    cache: &mut BTreeMap<usize, CalculusNodeId>,
) -> Result<CalculusNodeId, Diagnostic> {
    let key = Arc::as_ptr(&expression.node) as usize;
    if let Some(value) = cache.get(&key) {
        return Ok(*value);
    }
    let node = match expression.node.as_ref() {
        _ if input(expression).is_some() => CalculusNode::FormalComponent {
            formal: names[&input(expression).expect("matched independent input")],
            axes: Box::new([]),
        },
        LoweringExpressionNode::Literal(_) => CalculusNode::FormalComponent {
            formal: literals[&(Arc::as_ptr(&expression.node) as usize)],
            axes: Box::new([]),
        },
        LoweringExpressionNode::Partial { value, wrt } => {
            let root = scalar(file, value, names, literals, divisors, builder, cache)?;
            let id = builder
                .partial(root, names[&input(wrt).expect("validated partial input")])
                .map_err(|failure| error(file, expression, failure.to_string()))?;
            cache.insert(key, id);
            return Ok(id);
        }
        LoweringExpressionNode::Neg(value) => CalculusNode::Neg(scalar(
            file, value, names, literals, divisors, builder, cache,
        )?),
        LoweringExpressionNode::Binary {
            operator: BinaryOp::Pow,
            left,
            right,
        } => {
            let exponent = lowering_integer_literal(right)
                .filter(|n| {
                    *n > 0
                        && *n
                            <= i32::from(eqiora_schema::kernel::pure_operator::MAX_FORMAL_EXPONENT)
                })
                .ok_or_else(|| {
                    error(
                        file,
                        right,
                        "polynomial exponent exceeds the positive bounded calculus range",
                    )
                })?;
            let value = scalar(file, left, names, literals, divisors, builder, cache)?;
            let mut product = value;
            for _ in 1..exponent {
                product = builder
                    .push(CalculusNode::Mul(product, value))
                    .map_err(|failure| error(file, expression, failure.to_string()))?;
            }
            return Ok(product);
        }
        LoweringExpressionNode::Binary {
            operator: BinaryOp::Div,
            left,
            ..
        } => {
            let left = scalar(file, left, names, literals, divisors, builder, cache)?;
            let right = builder
                .push(CalculusNode::FormalComponent {
                    formal: divisors[&key],
                    axes: Box::new([]),
                })
                .map_err(|failure| error(file, expression, failure.to_string()))?;
            CalculusNode::Mul(left, right)
        }
        LoweringExpressionNode::Binary {
            operator,
            left,
            right,
        } => {
            let left = scalar(file, left, names, literals, divisors, builder, cache)?;
            let mut right = scalar(file, right, names, literals, divisors, builder, cache)?;
            match operator {
                BinaryOp::Add => CalculusNode::Add(left, right),
                BinaryOp::Sub => {
                    right = builder
                        .push(CalculusNode::Neg(right))
                        .map_err(|failure| error(file, expression, failure.to_string()))?;
                    CalculusNode::Add(left, right)
                }
                BinaryOp::Mul => CalculusNode::Mul(left, right),
                _ => {
                    return Err(error(
                        file,
                        expression,
                        "partial arithmetic requires an admitted polynomial rule",
                    ));
                }
            }
        }
        LoweringExpressionNode::PureOperator {
            definition,
            arguments,
        } => {
            let arguments = arguments
                .iter()
                .map(|value| scalar(file, value, names, literals, divisors, builder, cache))
                .collect::<Result<Vec<_>, _>>()?;
            let id = builder
                .apply_scalar(definition, &arguments)
                .map_err(|failure| error(file, expression, failure.to_string()))?;
            cache.insert(key, id);
            return Ok(id);
        }
        _ => {
            return Err(error(
                file,
                expression,
                "partial expression is outside the admitted polynomial profile",
            ));
        }
    };
    let id = builder
        .push(node)
        .map_err(|failure| error(file, expression, failure.to_string()))?;
    cache.insert(key, id);
    Ok(id)
}

fn error(file: &str, expression: &LoweringExpression, message: impl Into<String>) -> Diagnostic {
    source_error(
        codes::LANGUAGE_TYPE_ERROR,
        file,
        expression.range(),
        message,
    )
}
