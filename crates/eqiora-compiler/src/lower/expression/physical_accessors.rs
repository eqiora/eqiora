//! Lower accessors on existing physical Port contracts.
use super::*;

impl ExpressionLowerer<'_> {
    pub(super) fn lower_physical_accessor(
        &mut self,
        expression: &LoweringExpression,
        callee: &str,
        argument: &LoweringExpression,
    ) -> Result<TypedExpression, Diagnostic> {
        let LoweringExpressionNode::Name(name) = argument.node.as_ref() else {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                argument.range(),
                if matches!(callee, "across" | "through") {
                    format!("`{callee}(...)` requires one bare scalar physical Port name")
                } else {
                    format!("`{callee}(...)` requires one bare field-physical Port name")
                },
            ));
        };
        let Some(binding) = self.bindings.get(name) else {
            return Err(unresolved(
                self.file,
                argument.range(),
                name,
                "scalar physical Port",
            ));
        };
        let Binding::Port(port, contract) = binding else {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                argument.range(),
                format!("`{name}` is not a scalar physical Port"),
            ));
        };
        let contract = resolve_port_contract(self.file, argument.range(), contract, self.bindings)?;
        self.dependencies.insert(port.erase());
        self.ports.insert(port.erase());
        let (symbol, dimension) = match (callee, contract) {
            ("across", ResolvedPortContract::ScalarPhysical { across_type, .. }) => {
                (SymbolRef::Across(*port), across_type.dimension())
            }
            ("through", ResolvedPortContract::ScalarPhysical { through_type, .. }) => {
                (SymbolRef::Through(*port), through_type.dimension())
            }
            ("trace", ResolvedPortContract::BoundaryPhysical { trace_type, .. }) => {
                (SymbolRef::PortTrace(*port), trace_type.dimension())
            }
            ("flux", ResolvedPortContract::BoundaryPhysical { flux_type, .. }) => {
                (SymbolRef::PortFlux(*port), flux_type.dimension())
            }
            _ => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    argument.range(),
                    if matches!(callee, "across" | "through") {
                        format!("`{name}` is not a scalar physical Port")
                    } else {
                        format!("`{name}` is not a field-physical Port")
                    },
                ));
            }
        };
        self.builder
            .symbol(symbol)
            .map(|id| TypedExpression { id, dimension })
            .map_err(|diagnostic| self.builder_error(expression, diagnostic))
    }
}

impl LoweringExpression {
    pub(crate) fn collect_physical_port_names(&self, names: &mut BTreeSet<String>) -> bool {
        let mut pending = vec![self];
        let mut seen = BTreeSet::new();
        while let Some(expression) = pending.pop() {
            if !seen.insert(Arc::as_ptr(&expression.node) as usize) {
                continue;
            }
            match expression.node.as_ref() {
                LoweringExpressionNode::Call { callee, argument } => {
                    if matches!(callee.as_str(), "across" | "through" | "trace" | "flux")
                        && let LoweringExpressionNode::Name(name) = argument.node.as_ref()
                    {
                        names.insert(name.clone());
                    }
                    pending.push(argument);
                }
                LoweringExpressionNode::Partial { value, .. }
                | LoweringExpressionNode::Neg(value)
                | LoweringExpressionNode::Not(value)
                | LoweringExpressionNode::Index { value, .. }
                | LoweringExpressionNode::Sample { value, .. } => pending.push(value),
                LoweringExpressionNode::Array(elements) => pending.extend(elements),
                LoweringExpressionNode::IntegerCall { arguments, .. } => pending.extend(arguments),
                LoweringExpressionNode::Complex { real, imag } => pending.extend([real, imag]),
                LoweringExpressionNode::Case { value, arms } => {
                    pending.push(value);
                    pending.extend(arms.iter().map(|(_, value)| value));
                }
                LoweringExpressionNode::Select {
                    condition,
                    then_value,
                    else_value,
                } => pending.extend([condition, then_value, else_value]),
                LoweringExpressionNode::Require { condition, value } => {
                    pending.extend([condition, value])
                }
                LoweringExpressionNode::Binary { left, right, .. }
                | LoweringExpressionNode::Extremum { left, right, .. } => {
                    pending.push(left);
                    pending.push(right);
                }
                LoweringExpressionNode::Property { arguments, .. }
                | LoweringExpressionNode::PureOperator { arguments, .. }
                | LoweringExpressionNode::Finite { arguments, .. }
                | LoweringExpressionNode::Piecewise { arguments, .. } => pending.extend(arguments),
                LoweringExpressionNode::Number(_)
                | LoweringExpressionNode::Literal(_)
                | LoweringExpressionNode::Name(_) => {}
                _ => return false,
            }
        }
        true
    }
}
