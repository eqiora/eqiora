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
enum Input {
    Name(String),
    Coordinate(String, String, usize),
}
fn input(value: &LoweringExpression) -> Option<Input> {
    match value.node.as_ref() {
        LoweringExpressionNode::Name(name) => Some(Input::Name(name.clone())),
        LoweringExpressionNode::Coordinate {
            support,
            factor,
            axis,
        } => Some(Input::Coordinate(support.clone(), factor.clone(), *axis)),
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
        // The borrowed root keeps every source Arc alive throughout this call.
        // Both pointer-keyed maps are local to this formalization and its scope.
        let mut visited = BTreeSet::new();
        while let Some(value) = pending.pop() {
            if !visited.insert(Arc::as_ptr(&value.node) as usize) {
                continue;
            }
            match value.node.as_ref() {
                LoweringExpressionNode::Name(_) | LoweringExpressionNode::Coordinate { .. } => {
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
        let mut field_directions = Vec::new();
        if coordinate_derivative {
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
                    field_directions.push(index);
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
        for index in &field_directions {
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
            &mut calculus,
            &mut BTreeMap::new(),
        )?;
        let mut derivative = calculus
            .partial(root, 0)
            .map_err(|failure| error(self.file, expression, failure.to_string()))?;
        // Chain the common exact polynomial derivative with each retained Field
        // coordinate derivative. Numerical basis choice remains outside calculus.
        for (direction, index) in field_directions.iter().enumerate() {
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
        let mut arguments = inputs
            .iter()
            .map(|value| {
                // Selectors can be synthetic: their Arcs do not outlive this call.
                // Never enter names in the source-occurrence pointer cache.
                if let LoweringExpressionNode::Name(name) = value.node.as_ref() {
                    if name == "time" {
                        return self
                            .builder
                            .symbol(SymbolRef::Time)
                            .map_err(|failure| self.builder_error(expression, failure));
                    }
                    return self.lower_name(value, name).map(|value| value.id);
                }
                self.lower(value).map(|value| value.id)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for index in field_directions {
            let direction = self
                .builder
                .coordinate_partial(arguments[index], arguments[0])
                .map_err(|failure| self.builder_error(expression, failure))?;
            arguments.push(direction);
        }
        let id = self
            .builder
            .pure_operator(&definition, arguments)
            .map_err(|failure| self.builder_error(expression, failure))?;
        Ok(TypedExpression {
            id,
            dimension: result.dimension(),
        })
    }
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
    builder: &mut CalculusBuilder,
    cache: &mut BTreeMap<usize, CalculusNodeId>,
) -> Result<CalculusNodeId, Diagnostic> {
    let key = Arc::as_ptr(&expression.node) as usize;
    if let Some(value) = cache.get(&key) {
        return Ok(*value);
    }
    let node = match expression.node.as_ref() {
        LoweringExpressionNode::Name(_) | LoweringExpressionNode::Coordinate { .. } => {
            CalculusNode::FormalComponent {
                formal: names[&input(expression).expect("matched independent input")],
                axes: Box::new([]),
            }
        }
        LoweringExpressionNode::Literal(_) => CalculusNode::FormalComponent {
            formal: literals[&(Arc::as_ptr(&expression.node) as usize)],
            axes: Box::new([]),
        },
        LoweringExpressionNode::Partial { value, wrt } => {
            let root = scalar(file, value, names, literals, builder, cache)?;
            let id = builder
                .partial(root, names[&input(wrt).expect("validated partial input")])
                .map_err(|failure| error(file, expression, failure.to_string()))?;
            cache.insert(key, id);
            return Ok(id);
        }
        LoweringExpressionNode::Neg(value) => {
            CalculusNode::Neg(scalar(file, value, names, literals, builder, cache)?)
        }
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
            let value = scalar(file, left, names, literals, builder, cache)?;
            let mut product = value;
            for _ in 1..exponent {
                product = builder
                    .push(CalculusNode::Mul(product, value))
                    .map_err(|failure| error(file, expression, failure.to_string()))?;
            }
            return Ok(product);
        }
        LoweringExpressionNode::Binary {
            operator,
            left,
            right,
        } => {
            let left = scalar(file, left, names, literals, builder, cache)?;
            let mut right = scalar(file, right, names, literals, builder, cache)?;
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
                .map(|value| scalar(file, value, names, literals, builder, cache))
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
