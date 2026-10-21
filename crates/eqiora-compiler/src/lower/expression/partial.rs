//! Formalize explicit typed expressions without cutting their dependencies.
use super::*;
use eqiora_schema::kernel::pure_operator::{
    CalculusArgument, CalculusBuilder, CalculusNode, CalculusNodeId, ComponentIndex, PureValueClass,
};
mod coordinate_actions;

pub(crate) fn result_type<I: Clone + Eq>(
    value: &ExpressionType<I>,
    selected: &ExpressionType<I>,
) -> Result<ExpressionType<I>, &'static str> {
    value.partial(selected).map_err(|error| match error {
        eqiora_schema::kernel::typing::TypeViolation::IncompatibleSupport { .. } => {
            "partial operands have incompatible spatial supports"
        }
        eqiora_schema::kernel::typing::TypeViolation::DimensionOverflow { .. } => {
            "partial result dimension exceeds exact exponent bounds"
        }
        _ => "partial requires a numeric value and an independent real scalar",
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FormalMode {
    Value,
    Partial,
    TotalTime,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Input {
    Name(String),
    TimeDerivative(String, std::num::NonZeroU32),
    Coordinate(String, String, usize),
    Pullback(usize),
    MapFactor(usize),
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
        LoweringExpressionNode::CoordinateMapFactor { .. } => {
            Some(Input::MapFactor(Arc::as_ptr(&value.node) as usize))
        }
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
        // Formalization creates temporary chain-rule terms.
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
        self.formal_expression(
            expression,
            value,
            selected,
            if total_time {
                FormalMode::TotalTime
            } else {
                FormalMode::Partial
            },
        )
    }

    pub(super) fn lower_action(
        &mut self,
        expression: &LoweringExpression,
    ) -> Result<TypedExpression, Diagnostic> {
        let selected = match expression.node.as_ref() {
            LoweringExpressionNode::Jvp { directions, .. } => &directions[0].0,
            LoweringExpressionNode::Vjp { wrt, .. } => wrt,
            _ => unreachable!("action lowering"),
        };
        let outer_cache = std::mem::take(&mut self.cache);
        let result = self
            .coordinate_action(expression)
            .and_then(|coordinate| match coordinate {
                Some(derived) => self.lower(&derived),
                None => self
                    .formal_expression(expression, expression, selected, FormalMode::Value)
                    .and_then(|derived| self.lower(&derived)),
            });
        self.cache = outer_cache;
        result
    }

    fn formal_expression(
        &self,
        expression: &LoweringExpression,
        value: &LoweringExpression,
        selected: &LoweringExpression,
        mode: FormalMode,
    ) -> Result<LoweringExpression, Diagnostic> {
        let total_time = mode == FormalMode::TotalTime;
        let derive = mode != FormalMode::Value;
        if derive {
            require_retained_definitions(self.file, value, self.bindings)?;
        }
        let value_type = types::expression_type(self.file, value, self.bindings, None)?;
        let input_type = types::expression_type(self.file, selected, self.bindings, None)?;
        let result = if derive {
            result_type(&value_type, &input_type)
                .map_err(|message| error(self.file, expression, message))?
        } else {
            value_type
        };
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
                _ if input(value).is_some() => {
                    let key = input(value).expect("matched independent input");
                    if let std::collections::btree_map::Entry::Vacant(entry) = names.entry(key) {
                        let index = input_slot(inputs.len())
                            .map_err(|message| error(self.file, expression, message))?;
                        entry.insert(index);
                        inputs.push(value.clone());
                    }
                }
                LoweringExpressionNode::Jvp { .. }
                | LoweringExpressionNode::Vjp { .. }
                | LoweringExpressionNode::Index { .. }
                | LoweringExpressionNode::Select { .. }
                | LoweringExpressionNode::Require { .. }
                | LoweringExpressionNode::Not(_)
                | LoweringExpressionNode::Complex { .. } => {
                    value.node.push_children(&mut pending);
                }
                LoweringExpressionNode::Partial { value, wrt } => {
                    nested_partial = true;
                    coordinate_derivative |= matches!(input(wrt), Some(Input::Coordinate(..)));
                    pending.extend([value, wrt]);
                }
                LoweringExpressionNode::Literal(literal) if literal.as_bool().is_some() => {}
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
                LoweringExpressionNode::Call { callee, argument }
                    if crate::math::unary_function(callee).is_some() =>
                {
                    pending.push(argument);
                }
                LoweringExpressionNode::PureOperator { arguments, .. } => {
                    pending.extend(arguments.iter().rev())
                }
                _ => {
                    return Err(error(
                        self.file,
                        expression,
                        "differentiation admits explicit numeric component calculus and operator composition",
                    ));
                }
            }
        }
        let mut directions = Vec::new();
        if derive {
            for (index, value) in inputs.iter().enumerate() {
                if matches!(
                    value.node.as_ref(),
                    LoweringExpressionNode::CoordinateMapFactor { .. }
                ) {
                    if !total_time {
                        return Err(error(
                            self.file,
                            expression,
                            "coordinate-map factors currently admit only first time differentiation",
                        ));
                    }
                    directions.push(index);
                }
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
        }
        let formal_types = inputs
            .iter()
            .map(|value| types::expression_type(self.file, value, self.bindings, None))
            .collect::<Result<Vec<_>, _>>()?;
        let class =
            |ty: &ExpressionType<RawId>| PureValueClass::from_value_type(ty.value_type.clone());
        let mut formals = formal_types
            .iter()
            .map(|ty| {
                class(ty).map_err(|failure| error(self.file, expression, failure.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for index in &directions {
            input_slot(formals.len()).map_err(|message| error(self.file, expression, message))?;
            let ty = result_type(&formal_types[*index], &input_type)
                .map_err(|message| error(self.file, expression, message))?;
            formals.push(
                class(&ty).map_err(|failure| error(self.file, expression, failure.to_string()))?,
            );
        }
        let mut calculus = CalculusBuilder::new(
            formals.clone(),
            class(&result).map_err(|failure| error(self.file, expression, failure.to_string()))?,
        )
        .map_err(|failure| error(self.file, expression, failure.to_string()))?;
        let root = ComponentCompiler {
            file: self.file,
            bindings: self.bindings,
            names: &names,
            literals: &literals,
            formals: &formals,
        }
        .component(value, &mut calculus, &mut BTreeMap::new())?;
        let mut derivative = if derive {
            calculus
                .partial(root, 0)
                .map_err(|failure| error(self.file, expression, failure.to_string()))?
        } else {
            root
        };
        // Chain the common formal derivative with each retained Field
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
        if total_time && !value.referenced_names().contains("time") {
            // Time supplies the derivative unit even when it is absent from the
            // expression. Do not create a runtime dependency on that unused slot.
            let zero = eqiora_core::ValueLiteral::from_real(input_type.value_type.clone(), 0.0)
                .map_err(|failure| error(self.file, expression, failure.to_string()))?;
            arguments[0] = LoweringExpression::literal(zero, expression.range());
        }
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

fn input_slot(count: usize) -> Result<u16, &'static str> {
    if count >= eqiora_schema::kernel::pure_operator::MAX_FORMALS {
        return Err("partial input occurrences exceed the calculus formal bound");
    }
    u16::try_from(count).map_err(|_| "partial formal index overflows")
}

fn require_retained_definitions(
    file: &str,
    value: &LoweringExpression,
    bindings: &BTreeMap<String, Binding>,
) -> Result<(), Diagnostic> {
    if value
        .referenced_names()
        .iter()
        .any(|name| matches!(bindings.get(name), Some(Binding::Observable(..))))
    {
        return Err(error(
            file,
            value,
            "derivative of an Observable requires differentiation through its retained definition",
        ));
    }
    Ok(())
}

// Model partials and actions compile complete values through the same component
// calculus. Each nested value gets its own result axes and local component cache.
struct ComponentCompiler<'a> {
    file: &'a str,
    bindings: &'a BTreeMap<String, Binding>,
    names: &'a BTreeMap<Input, u16>,
    literals: &'a BTreeMap<usize, u16>,
    formals: &'a [PureValueClass],
}

impl ComponentCompiler<'_> {
    fn axes(&self, formal: u16) -> Box<[ComponentIndex]> {
        (0..self.formals[usize::from(formal)].rank())
            .map(|axis| ComponentIndex::Result(axis as u16))
            .collect()
    }

    fn argument(
        &self,
        expression: &LoweringExpression,
        builder: &mut CalculusBuilder,
        cache: &mut BTreeMap<usize, CalculusNodeId>,
    ) -> Result<CalculusArgument, Diagnostic> {
        let ty = types::expression_type(self.file, expression, self.bindings, None)?;
        let class = PureValueClass::from_value_type(ty.value_type)
            .map_err(|failure| error(self.file, expression, failure.to_string()))?;
        if class.is_invariant_scalar() {
            return self.component(expression, builder, cache).map(Into::into);
        }
        let mut value = CalculusBuilder::new(self.formals.iter().cloned(), class)
            .map_err(|failure| error(self.file, expression, failure.to_string()))?;
        let root = self.component(expression, &mut value, &mut BTreeMap::new())?;
        value
            .finish(root)
            .map(Into::into)
            .map_err(|failure| error(self.file, expression, failure.to_string()))
    }

    fn component(
        &self,
        expression: &LoweringExpression,
        builder: &mut CalculusBuilder,
        cache: &mut BTreeMap<usize, CalculusNodeId>,
    ) -> Result<CalculusNodeId, Diagnostic> {
        let file = self.file;
        let names = self.names;
        let literals = self.literals;
        let key = Arc::as_ptr(&expression.node) as usize;
        if let Some(value) = cache.get(&key) {
            return Ok(*value);
        }
        let node = match expression.node.as_ref() {
            _ if input(expression).is_some() => CalculusNode::FormalComponent {
                formal: names[&input(expression).expect("matched independent input")],
                axes: self.axes(names[&input(expression).expect("matched independent input")]),
            },
            LoweringExpressionNode::Literal(literal) if literal.as_bool().is_some() => {
                CalculusNode::Boolean(literal.as_bool().expect("matched Boolean"))
            }
            LoweringExpressionNode::Literal(_) => CalculusNode::FormalComponent {
                formal: literals[&(Arc::as_ptr(&expression.node) as usize)],
                axes: self.axes(literals[&(Arc::as_ptr(&expression.node) as usize)]),
            },
            LoweringExpressionNode::Index { value, index } => {
                let input = types::expression_type(file, value, self.bindings, None)?;
                let argument = self.argument(value, builder, cache)?;
                let id = crate::pure_operator::composition::indexed_value(
                    builder,
                    argument,
                    input.value_type,
                    *index,
                )
                .map_err(|failure| error(file, expression, failure.to_string()))?;
                cache.insert(key, id);
                return Ok(id);
            }
            LoweringExpressionNode::Jvp { value, directions } => {
                for (selected, _) in directions {
                    self.require_formal_dependencies(value, selected)?;
                }
                let root = self.argument(value, builder, cache)?;
                let directions = directions
                    .iter()
                    .map(|(selected, direction)| {
                        let formal = names[&input(selected).expect("validated action input")];
                        Ok((formal, self.argument(direction, builder, cache)?))
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                let id = builder
                    .jvp(&root, &directions)
                    .map_err(|failure| error(file, expression, failure.to_string()))?;
                cache.insert(key, id);
                return Ok(id);
            }
            LoweringExpressionNode::Vjp {
                value,
                wrt,
                cotangent,
            } => {
                self.require_formal_dependencies(value, wrt)?;
                let root = self.argument(value, builder, cache)?;
                let seed = self.argument(cotangent, builder, cache)?;
                let formal = names[&input(wrt).expect("validated action input")];
                let id = builder
                    .vjp(&root, &seed, formal)
                    .map_err(|failure| error(file, expression, failure.to_string()))?;
                cache.insert(key, id);
                return Ok(id);
            }
            LoweringExpressionNode::Complex { real, imag } => CalculusNode::Complex {
                real: self.component(real, builder, cache)?,
                imag: self.component(imag, builder, cache)?,
            },
            LoweringExpressionNode::Partial { value, wrt } => {
                self.require_formal_dependencies(value, wrt)?;
                let root = self.component(value, builder, cache)?;
                let id = builder
                    .partial(root, names[&input(wrt).expect("validated partial input")])
                    .map_err(|failure| error(file, expression, failure.to_string()))?;
                cache.insert(key, id);
                return Ok(id);
            }
            LoweringExpressionNode::Neg(value) => {
                CalculusNode::Neg(self.component(value, builder, cache)?)
            }
            LoweringExpressionNode::Not(value) => {
                CalculusNode::Not(self.component(value, builder, cache)?)
            }
            LoweringExpressionNode::Select {
                condition,
                then_value,
                else_value,
            } => CalculusNode::Select {
                condition: self.component(condition, builder, cache)?,
                then_value: self.component(then_value, builder, cache)?,
                else_value: self.component(else_value, builder, cache)?,
            },
            LoweringExpressionNode::Require { condition, value } => CalculusNode::Require {
                condition: self.component(condition, builder, cache)?,
                value: self.component(value, builder, cache)?,
            },
            LoweringExpressionNode::Binary {
                operator: BinaryOp::Pow,
                left,
                right,
            } => {
                let exponent = lowering_integer_literal(right)
                    .filter(|n| {
                        *n > 0
                            && *n
                                <= i32::from(
                                    eqiora_schema::kernel::pure_operator::MAX_FORMAL_EXPONENT,
                                )
                    })
                    .ok_or_else(|| {
                        error(
                            file,
                            right,
                            "polynomial exponent exceeds the positive bounded calculus range",
                        )
                    })?;
                let value = self.component(left, builder, cache)?;
                let mut product = value;
                for _ in 1..exponent {
                    product = builder
                        .push(CalculusNode::Mul(product, value))
                        .map_err(|failure| error(file, expression, failure.to_string()))?;
                }
                return Ok(product);
            }
            LoweringExpressionNode::Call { callee, argument }
                if crate::math::unary_function(callee).is_some() =>
            {
                CalculusNode::UnaryMath(
                    crate::math::unary_function(callee).expect("matched unary mathematics"),
                    self.component(argument, builder, cache)?,
                )
            }
            LoweringExpressionNode::Binary {
                operator,
                left,
                right,
            } => {
                let left = self.component(left, builder, cache)?;
                let mut right = self.component(right, builder, cache)?;
                match operator {
                    BinaryOp::Add => CalculusNode::Add(left, right),
                    BinaryOp::Sub => {
                        right = builder
                            .push(CalculusNode::Neg(right))
                            .map_err(|failure| error(file, expression, failure.to_string()))?;
                        CalculusNode::Add(left, right)
                    }
                    BinaryOp::Mul => CalculusNode::Mul(left, right),
                    BinaryOp::Div => CalculusNode::Div(left, right),
                    BinaryOp::And => CalculusNode::And(left, right),
                    BinaryOp::Or => CalculusNode::Or(left, right),
                    op if crate::lower::comparison_operator(*op).is_some() => {
                        CalculusNode::Compare(
                            crate::lower::comparison_operator(*op).expect("matched comparison"),
                            left,
                            right,
                        )
                    }
                    _ => {
                        return Err(error(
                            file,
                            expression,
                            "differentiation requires an admitted component arithmetic rule",
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
                    .map(|value| self.argument(value, builder, cache))
                    .collect::<Result<Vec<_>, _>>()?;
                let id = builder
                    .apply_inputs(definition, &arguments)
                    .map_err(|failure| error(file, expression, failure.to_string()))?;
                cache.insert(key, id);
                return Ok(id);
            }
            _ => {
                return Err(error(
                    file,
                    expression,
                    "expression is outside the admitted numeric component calculus",
                ));
            }
        };
        let id = builder
            .push(node)
            .map_err(|failure| error(file, expression, failure.to_string()))?;
        cache.insert(key, id);
        Ok(id)
    }
}

fn error(file: &str, expression: &LoweringExpression, message: impl Into<String>) -> Diagnostic {
    source_error(
        codes::LANGUAGE_TYPE_ERROR,
        file,
        expression.range(),
        message,
    )
}
