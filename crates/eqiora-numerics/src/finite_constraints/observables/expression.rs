//! Scoped coordinate expansion retains parameter expressions for ordinary native AD.
use super::*;
use eqiora_schema::kernel::ComparisonOp;

impl Context<'_> {
    pub(super) fn expression(
        &mut self,
        expression: &ExprDag,
        point: &Point,
        depth: usize,
    ) -> Result<Vec<ExprId>, Diagnostic> {
        self.expression_roots(expression, expression.roots(), point, depth)
    }

    fn expression_roots(
        &mut self,
        expression: &ExprDag,
        roots: &[ExprId],
        point: &Point,
        depth: usize,
    ) -> Result<Vec<ExprId>, Diagnostic> {
        if depth > 32 || !expression.properties().is_empty() {
            return Err(invalid(
                "finite Observable expansion requires bounded unannotated scalar expressions",
            ));
        }
        self.charge(expression.nodes().len())?;
        let mut values = vec![None; expression.nodes().len()];
        for &root in roots {
            let mut pending = vec![(root, false)];
            while let Some((id, ready)) = pending.pop() {
                if values[id.index() as usize].is_some() {
                    continue;
                }
                let node = expression.node(id).expect("validated expression operand");
                if !ready {
                    pending.push((id, true));
                    // A point binds its value in a fresh context. Its mapped
                    // coordinates are all evaluated in the original context.
                    if let ExprNode::Evaluate { at, .. } = node {
                        pending.extend(at.iter().rev().map(|(_, value)| (*value, false)));
                    } else {
                        node.try_for_each_operand(|operand| {
                            pending.push((operand, false));
                            Ok::<_, Diagnostic>(())
                        })?;
                    }
                    continue;
                }
                let value = |id: &ExprId| values[id.index() as usize].expect("expanded operand");
                let copied = match node {
                    ExprNode::Constant(literal) if literal.real_scalar_value().is_some() => {
                        self.builder.constant(literal.clone())?
                    }
                    ExprNode::Symbol(SymbolRef::Observable(id)) => {
                        self.observable(*id, point, depth + 1)?
                    }
                    ExprNode::Symbol(SymbolRef::Coordinate { factor, axis, .. }) => *point
                        .get(&(factor.erase(), *axis))
                        .ok_or_else(|| invalid("finite Observable leaves an unbound coordinate"))?,
                    ExprNode::Symbol(symbol @ (SymbolRef::Field(_) | SymbolRef::Parameter(_))) => {
                        self.builder.symbol(*symbol)?
                    }
                    ExprNode::Neg(a) => self.builder.neg(value(a))?,
                    ExprNode::Add(a, b) => self.builder.add(value(a), value(b))?,
                    ExprNode::Sub(a, b) => self.builder.sub(value(a), value(b))?,
                    ExprNode::Mul(a, b) => self.builder.mul(value(a), value(b))?,
                    ExprNode::Div(a, b) => self.builder.div(value(a), value(b))?,
                    ExprNode::PowI(a, power) => self.builder.powi(value(a), *power)?,
                    ExprNode::UnaryMath(function, a) => {
                        self.builder.unary_math(*function, value(a))?
                    }
                    ExprNode::PureOperatorApplication(application) => self.builder.pure_operator(
                        expression
                            .definition(application.definition())
                            .expect("retained definition"),
                        application.arguments().iter().map(value),
                    )?,
                    ExprNode::Evaluate {
                        value: root,
                        at,
                        side: None,
                    } => {
                        let mut local = point.clone();
                        let mut condition = None;
                        for (selector, mapped) in at {
                            self.charge(6)?;
                            let Some(ExprNode::Symbol(SymbolRef::Coordinate {
                                factor, axis, ..
                            })) = expression.node(*selector)
                            else {
                                return Err(invalid("point selector must be an exact coordinate"));
                            };
                            let mapped = value(mapped);
                            let key = (factor.erase(), *axis);
                            let bounds = axes(self.kernel, *factor)?
                                .into_iter()
                                .find(|(candidate, _)| *candidate == key)
                                .ok_or_else(|| {
                                    invalid("point coordinate is outside its bounded factor")
                                })?
                                .1;
                            let lower = self.builder.constant(bounds.lower())?;
                            let upper = self.builder.constant(bounds.upper())?;
                            let above =
                                self.builder
                                    .compare(ComparisonOp::GreaterEqual, mapped, lower)?;
                            let below =
                                self.builder
                                    .compare(ComparisonOp::LessEqual, mapped, upper)?;
                            let inside = self.builder.and(above, below)?;
                            condition = Some(match condition {
                                None => inside,
                                Some(previous) => self.builder.and(previous, inside)?,
                            });
                            local.insert(key, mapped);
                        }
                        let result =
                            self.expression_roots(expression, &[*root], &local, depth + 1)?[0];
                        self.charge(1)?;
                        self.builder
                            .require(condition.expect("nonempty validated point"), result)?
                    }
                    _ => {
                        return Err(invalid(
                            "finite Observable coupling requires smooth real scalar arithmetic and complete point bindings",
                        ));
                    }
                };
                values[id.index() as usize] = Some(copied);
            }
        }
        Ok(roots
            .iter()
            .map(|id| values[id.index() as usize].expect("expanded root"))
            .collect())
    }
}
