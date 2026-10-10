//! Smooth time dependency admission before ordinary evolution lowering.
use super::*;

impl ExpressionChecker<'_, '_, '_> {
    pub(super) fn check_time_derivative(
        &mut self,
        expression: &Expr,
        value: &Expr,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        if self.sampling || (!self.intrinsic && self.allow_discrete_symbols && !self.initial) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                "smooth time differentiation requires a continuous evolution context",
            ));
        }
        let inferred = self.check(value)?;
        let mut pending = vec![value.clone()];
        let mut aliases = std::collections::BTreeSet::new();
        while let Some(value) = pending.pop() {
            match value.kind() {
                ExprKind::Name(name) if name == "time" => {}
                ExprKind::Name(name) => match self.scope.symbols.get(name).cloned() {
                    Some(SymbolContract::Parameter(_) | SymbolContract::Coordinate(_)) => {}
                    Some(SymbolContract::Alias(alias)) => {
                        if aliases.insert(name.clone()) {
                            pending.push(alias.expression.clone());
                        }
                    }
                    Some(SymbolContract::Field(..)) => {
                        self.check_evolution("derivative", expression, &value)?;
                    }
                    _ => return Err(self.time_rule_error(&value)),
                },
                ExprKind::Number(_) | ExprKind::Quantity { .. } => {}
                ExprKind::CoordinateMapFactor { at, .. } => {
                    pending.extend(at.iter().map(|(_, mapped)| mapped.clone()));
                }
                ExprKind::Pullback { value, at, .. } => {
                    pending.push(*value.clone());
                    pending.extend(at.iter().map(|(_, mapped)| mapped.clone()));
                }
                ExprKind::Unary {
                    op: UnaryOp::Neg,
                    value,
                } => pending.push(*value.clone()),
                ExprKind::Binary {
                    op: BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Pow,
                    left,
                    right,
                } => {
                    pending.extend([*left.clone(), *right.clone()]);
                }
                ExprKind::Call { callee, arguments } if !is_builtin_operator(callee) => {
                    pending.extend(arguments.expressions().cloned());
                }
                ExprKind::Call { callee, arguments } if callee.as_str() == "derivative" => {
                    pending.extend(arguments.expressions().cloned());
                }
                ExprKind::Call { callee, arguments }
                    if callee.as_str() == "time" && arguments.expressions().next().is_none() => {}
                _ => return Err(self.time_rule_error(&value)),
            }
        }
        typing::time_derivative(&inferred, std::num::NonZeroU32::MIN)
            .map_err(|error| type_error(self.scope.file, expression, error))
    }

    fn time_rule_error(&self, value: &Expr) -> Diagnostic {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            self.scope.file,
            value.range(),
            "time derivative admits explicit smooth scalar polynomial expressions of continuous states, fixed Parameters, declared coordinates, explicit pullbacks and coordinate-map factors; discrete, algebraic, and unsupported derivative products reject",
        )
    }
}
