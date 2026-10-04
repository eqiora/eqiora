//! Conservative per-factor degree admission; no floating-point expression reassociation.
use super::*;

impl Context<'_> {
    pub(super) fn degree_expression(
        &mut self,
        expression: &ExprDag,
        depth: usize,
    ) -> Result<Degree, Diagnostic> {
        if depth > 32 {
            return Err(invalid(
                "finite integral degree dependency depth exceeds 32",
            ));
        }
        self.charge(expression.nodes().len())?;
        let mut degrees: Vec<Degree> = Vec::with_capacity(expression.nodes().len());
        for node in expression.nodes() {
            let at = |id: &ExprId| &degrees[id.index() as usize];
            let degree = match node {
                ExprNode::Constant(_)
                | ExprNode::Symbol(SymbolRef::Field(_) | SymbolRef::Parameter(_)) => Degree::new(),
                ExprNode::Symbol(SymbolRef::Coordinate { factor, axis, .. }) => {
                    Degree::from([((factor.erase(), *axis), 1)])
                }
                ExprNode::Symbol(SymbolRef::Observable(id)) => {
                    self.degree_observable(*id, depth + 1)?
                }
                ExprNode::Neg(value) => at(value).clone(),
                ExprNode::Add(a, b) | ExprNode::Sub(a, b) => combine(at(a), at(b), false)?,
                ExprNode::Mul(a, b) => combine(at(a), at(b), true)?,
                ExprNode::Div(a, b) if at(b).is_empty() => at(a).clone(),
                ExprNode::PowI(value, power) if *power >= 0 => {
                    if *power == 0 {
                        Degree::new()
                    } else {
                        at(value)
                            .iter()
                            .map(|(axis, degree)| {
                                let power = u16::try_from(*power).map_err(|_| {
                                    invalid("finite polynomial exponent exceeds u16")
                                })?;
                                Ok((
                                    *axis,
                                    degree.checked_mul(power).ok_or_else(|| {
                                        invalid("finite polynomial degree overflow")
                                    })?,
                                ))
                            })
                            .collect::<Result<Degree, Diagnostic>>()?
                    }
                }
                ExprNode::PowI(value, _) | ExprNode::UnaryMath(_, value)
                    if at(value).is_empty() =>
                {
                    Degree::new()
                }
                _ => {
                    return Err(invalid(
                        "finite integral requires a polynomial in its coordinate factors with coordinate-independent denominators",
                    ));
                }
            };
            degrees.push(degree);
        }
        if expression.roots().len() != 1 {
            return Err(invalid("integral density requires one root"));
        }
        Ok(degrees[expression.roots()[0].index() as usize].clone())
    }

    fn degree_observable(
        &mut self,
        id: Id<kinds::Observable>,
        depth: usize,
    ) -> Result<Degree, Diagnostic> {
        if let Some(degree) = self.degrees.get(&id.erase()) {
            return Ok(degree.clone());
        }
        let Some(KernelNode::Observable(definition)) = self.kernel.node(id.erase()) else {
            return Err(invalid("polynomial degree references a foreign Observable"));
        };
        let definition = definition.clone();
        let mut degree = self.degree_expression(definition.expression(), depth)?;
        if let ObservableReduction::SpatialIntegral { domain, .. } = definition.reduction() {
            for (axis, _) in axes(self.kernel, domain)? {
                degree.remove(&axis);
            }
        }
        self.degrees.insert(id.erase(), degree.clone());
        Ok(degree)
    }
}

fn combine(left: &Degree, right: &Degree, multiply: bool) -> Result<Degree, Diagnostic> {
    let mut result = left.clone();
    for (axis, degree) in right {
        let current = result.entry(*axis).or_default();
        *current = if multiply {
            current
                .checked_add(*degree)
                .ok_or_else(|| invalid("finite polynomial degree overflow"))?
        } else {
            (*current).max(*degree)
        };
    }
    Ok(result)
}
