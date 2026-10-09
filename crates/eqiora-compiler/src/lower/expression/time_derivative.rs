//! Total time differentiation composes the shared formal partial transform.
use super::*;

impl ExpressionLowerer<'_> {
    pub(super) fn lower_time_derivative(
        &mut self,
        expression: &LoweringExpression,
        value: &LoweringExpression,
    ) -> Result<TypedExpression, Diagnostic> {
        if self.sampling || (self.allow_discrete_symbols && !self.initial) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                expression.range(),
                "smooth time differentiation requires a continuous evolution context",
            ));
        }
        // Preserve the authored Field and exact order. First-order numerical
        // coordinates are selected later by Formulation, never by adding States
        // to the source Model.
        let mut terminal = value;
        let mut order = std::num::NonZeroU32::MIN;
        while let LoweringExpressionNode::Call { callee, argument } = terminal.node.as_ref() {
            if callee != "derivative" {
                break;
            }
            order = order.checked_add(1).ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    "time derivative order exceeds its exact representation",
                )
            })?;
            terminal = argument;
        }
        if order.get() > 1
            && let LoweringExpressionNode::Name(name) = terminal.node.as_ref()
            && let Some(Binding::Field(field, contract)) = self.bindings.get(name)
        {
            let field = *field;
            let operand = ExpressionType::<String>::scalar(contract.dimension, None);
            let dimension = typing::time_derivative(&operand, order)
                .map_err(|_| dimension_overflow(self.file, expression.range()))?
                .dimension();
            // Validate without emitting an unused first-derivative node. The
            // original Model's semantic projection must remain a closed DAG.
            self.require_eligible_evolution(expression, "derivative", contract)?;
            self.dependencies.insert(field.erase());
            let id = self
                .builder
                .symbol(SymbolRef::Derivative(field, order))
                .map_err(|failure| self.builder_error(expression, failure))?;
            return Ok(TypedExpression { id, dimension });
        }
        // Expanded chain-rule expressions are temporary. Keep their pointer
        // cache entries within the lifetime of the expansion.
        let outer_cache = std::mem::take(&mut self.cache);
        let lowered = (|| {
            let value = self.expand_time_derivatives(value, &mut BTreeMap::new())?;
            let total = self.total_time_expression(expression, &value)?;
            self.lower(&total)
        })();
        self.cache = outer_cache;
        lowered
    }

    fn total_time_expression(
        &self,
        expression: &LoweringExpression,
        value: &LoweringExpression,
    ) -> Result<LoweringExpression, Diagnostic> {
        let mut evolving = Vec::new();
        for name in value.referenced_names() {
            if name == "time" {
                evolving.push(name);
                continue;
            }
            match self.bindings.get(&name) {
                // Coordinate support and factor dependencies are fixed domains.
                Some(Binding::Parameter(..) | Binding::Domain(..)) => {}
                Some(Binding::Field(_, contract))
                    if self.eligible_evolution("derivative", contract) =>
                {
                    evolving.push(name)
                }
                _ => {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        expression.range(),
                        "time derivative requires declared continuous states, fixed Parameters, and the enclosing time coordinate",
                    ));
                }
            }
        }
        // The time coordinate also supplies the quotient dimension for a fixed expression.
        if evolving.is_empty() {
            evolving.push("time".to_owned());
        }
        let mut inputs = evolving
            .into_iter()
            .map(|name| LoweringExpression::name(name, expression.range()))
            .collect::<Vec<_>>();
        let mut rates = BTreeMap::new();
        let mut pending = vec![value];
        let mut seen = BTreeSet::new();
        while let Some(value) = pending.pop() {
            if !seen.insert(Arc::as_ptr(&value.node) as usize) {
                continue;
            }
            if let Some(key @ partial::Input::TimeDerivative(..)) = partial::input(value) {
                let partial::Input::TimeDerivative(name, _) = &key else {
                    unreachable!()
                };
                // Fixed Parameters have zero rates and time has constant unit
                // rate. Neither introduces an evolving higher-order coordinate.
                if matches!(self.bindings.get(name), Some(Binding::Field(..))) {
                    rates.entry(key).or_insert_with(|| value.clone());
                }
                continue;
            }
            match value.node.as_ref() {
                LoweringExpressionNode::Neg(inner) => pending.push(inner),
                LoweringExpressionNode::Partial { value, wrt } => pending.extend([value, wrt]),
                LoweringExpressionNode::Binary { left, right, .. } => pending.extend([left, right]),
                LoweringExpressionNode::PureOperator { arguments, .. } => pending.extend(arguments),
                _ => {}
            }
        }
        inputs.extend(rates.into_values());
        let mut terms = Vec::with_capacity(inputs.len());
        for selected in inputs {
            let partial =
                LoweringExpression::partial(value.clone(), selected.clone(), expression.range());
            let term = if matches!(selected.node.as_ref(), LoweringExpressionNode::Name(name) if name == "time")
            {
                partial
            } else {
                let rate =
                    LoweringExpression::call("derivative".to_owned(), selected, expression.range());
                LoweringExpression::binary(BinaryOp::Mul, partial, rate, expression.range())
            };
            terms.push(term);
        }
        Ok(terms
            .into_iter()
            .reduce(|left, right| {
                LoweringExpression::binary(BinaryOp::Add, left, right, expression.range())
            })
            .expect("at least the fixed-time partial"))
    }

    fn expand_time_derivatives(
        &self,
        value: &LoweringExpression,
        cache: &mut BTreeMap<usize, LoweringExpression>,
    ) -> Result<LoweringExpression, Diagnostic> {
        let key = Arc::as_ptr(&value.node) as usize;
        if let Some(expanded) = cache.get(&key) {
            return Ok(expanded.clone());
        }
        let expanded = match value.node.as_ref() {
            // A Field rate is an independent coordinate for the formal partial
            // transform. Only compound total derivatives need expansion here.
            LoweringExpressionNode::Call { callee, argument }
                if callee == "derivative" && partial::input(value).is_none() =>
            {
                let argument = self.expand_time_derivatives(argument, cache)?;
                self.total_time_expression(value, &argument)?
            }
            LoweringExpressionNode::Neg(inner) => {
                LoweringExpression::neg(self.expand_time_derivatives(inner, cache)?, value.range())
            }
            LoweringExpressionNode::Binary {
                operator,
                left,
                right,
            } => LoweringExpression::binary(
                *operator,
                self.expand_time_derivatives(left, cache)?,
                self.expand_time_derivatives(right, cache)?,
                value.range(),
            ),
            LoweringExpressionNode::Partial { value: inner, wrt } => LoweringExpression::partial(
                self.expand_time_derivatives(inner, cache)?,
                wrt.clone(),
                value.range(),
            ),
            LoweringExpressionNode::PureOperator {
                definition,
                arguments,
            } => LoweringExpression::pure_operator(
                definition.clone(),
                arguments
                    .iter()
                    .map(|argument| self.expand_time_derivatives(argument, cache))
                    .collect::<Result<Vec<_>, _>>()?,
                value.range(),
            ),
            _ => value.clone(),
        };
        cache.insert(key, expanded.clone());
        Ok(expanded)
    }
}
