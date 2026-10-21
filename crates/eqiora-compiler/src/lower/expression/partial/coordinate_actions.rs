//! Coordinate actions use the existing represented-Field/pullback chain rule.
//! Formal slots remain independent; Model coordinates need their physical dependencies.
use super::*;

impl ExpressionLowerer<'_> {
    pub(super) fn coordinate_action(
        &self,
        expression: &LoweringExpression,
    ) -> Result<Option<LoweringExpression>, Diagnostic> {
        let coordinate = |selected| matches!(input(selected), Some(Input::Coordinate(..)));
        let range = expression.range();
        match expression.node.as_ref() {
            LoweringExpressionNode::Jvp { value, directions }
                if directions.iter().any(|(selected, _)| coordinate(selected)) =>
            {
                // Validate the whole action before decomposing its ordered blocks.
                types::expression_type(self.file, expression, self.bindings, None)?;
                let mut seen = BTreeSet::new();
                let mut sum = None;
                for (selected, direction) in directions {
                    if !seen.insert(input(selected)) {
                        return Err(error(
                            self.file,
                            expression,
                            "jvp selectors must be distinct independent bindings",
                        ));
                    }
                    let term = if coordinate(selected) {
                        LoweringExpression::binary(
                            BinaryOp::Mul,
                            LoweringExpression::partial(value.clone(), selected.clone(), range),
                            direction.clone(),
                            range,
                        )
                    } else {
                        LoweringExpression::jvp(
                            value.clone(),
                            vec![(selected.clone(), direction.clone())],
                            range,
                        )
                    };
                    sum = Some(match sum {
                        Some(prior) => {
                            LoweringExpression::binary(BinaryOp::Add, prior, term, range)
                        }
                        None => term,
                    });
                }
                Ok(sum)
            }
            LoweringExpressionNode::Vjp {
                value,
                wrt,
                cotangent,
            } if coordinate(wrt) => {
                types::expression_type(self.file, expression, self.bindings, None)?;
                let derivative = LoweringExpression::partial(value.clone(), wrt.clone(), range);
                let slope = types::expression_type(self.file, &derivative, self.bindings, None)?;
                let seed = types::expression_type(self.file, cotangent, self.bindings, None)?;
                let pairing =
                    PureOperatorDefinition::real_pairing(seed.value_type, slope.value_type)
                        .map_err(|failure| error(self.file, expression, failure.to_string()))?;
                Ok(Some(LoweringExpression::pure_operator(
                    pairing,
                    vec![cotangent.clone(), derivative],
                    range,
                )))
            }
            _ => Ok(None),
        }
    }
}

impl ComponentCompiler<'_> {
    // Nested formal transforms cannot freeze a physical derivative request.
    // Ordinary coordinate actions leave this arena and use lower_partial; mixed
    // represented-Field derivatives still require their own admitted representation.
    pub(super) fn require_formal_dependencies(
        &self,
        value: &LoweringExpression,
        selected: &LoweringExpression,
    ) -> Result<(), Diagnostic> {
        require_retained_definitions(self.file, value, self.bindings)?;
        let coordinate = matches!(input(selected), Some(Input::Coordinate(..)));
        let mut pending = vec![value];
        let mut seen = BTreeSet::new();
        while let Some(value) = pending.pop() {
            if !seen.insert(Arc::as_ptr(&value.node)) {
                continue;
            }
            let missing = match value.node.as_ref() {
                LoweringExpressionNode::Name(name)
                    if coordinate
                        && matches!(self.bindings.get(name), Some(Binding::Field(_, contract)) if contract.domain.is_some()) =>
                {
                    Some(
                        "nested coordinate actions require an admitted mixed Field derivative representation",
                    )
                }
                LoweringExpressionNode::Pullback { .. }
                | LoweringExpressionNode::CoordinateMapFactor { .. } => Some(
                    "formal differentiation of a pullback or coordinate-map factor requires its retained dependency transform",
                ),
                _ => None,
            };
            if let Some(message) = missing {
                return Err(error(self.file, value, message));
            }
            value.node.push_children(&mut pending);
        }
        Ok(())
    }
}
