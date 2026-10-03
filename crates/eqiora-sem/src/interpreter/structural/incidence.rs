//! Raw scalar occurrence incidence; derivative and value coordinates stay distinct.
use super::*;

pub(in crate::interpreter) fn variables(
    dag: &ExprDag,
    roots: &[ExprId],
    signal_sources: &BTreeMap<RawId, RawId>,
) -> Result<BTreeSet<Variable>, Diagnostic> {
    let mut pending = roots.to_vec();
    let mut seen = vec![false; dag.nodes().len()];
    let mut coordinates = BTreeSet::new();
    while let Some(id) = pending.pop() {
        let index = id.index() as usize;
        if std::mem::replace(&mut seen[index], true) {
            continue;
        }
        match &dag.nodes()[index] {
            ExprNode::Symbol(symbol) => {
                let variable = match symbol {
                    SymbolRef::Field(field) => Some(Variable::Field(field.erase())),
                    SymbolRef::Derivative(field) => Some(Variable::Derivative(field.erase())),
                    SymbolRef::Port(port) => Some(Variable::Port(
                        signal_sources
                            .get(&port.erase())
                            .copied()
                            .unwrap_or_else(|| port.erase()),
                    )),
                    SymbolRef::Across(port) => {
                        Some(Variable::Physical(PhysicalUnknown::Across(*port)))
                    }
                    SymbolRef::Through(port) => {
                        Some(Variable::Physical(PhysicalUnknown::Through(*port)))
                    }
                    _ => None,
                };
                if let Some(variable) = variable {
                    coordinates.insert(variable);
                }
            }
            ExprNode::Constant(_) | ExprNode::SpatialCoordinate(_) => {}
            ExprNode::Require { condition, value } => pending.extend([*condition, *value]),
            ExprNode::Select {
                condition,
                then_value,
                else_value,
            } => pending.extend([*condition, *then_value, *else_value]),
            ExprNode::Array { elements } => pending.extend(elements),
            ExprNode::Index { value, .. }
            | ExprNode::Sample { value, .. }
            | ExprNode::Not(value)
            | ExprNode::Ordinal(value)
            | ExprNode::ToReal(value)
            | ExprNode::ToInteger(value)
            | ExprNode::Hold(value)
            | ExprNode::Neg(value)
            | ExprNode::PowI(value, _)
            | ExprNode::UnaryMath(_, value)
            | ExprNode::FiniteUnary(_, value)
            | ExprNode::Gradient(value)
            | ExprNode::Divergence(value)
            | ExprNode::SymmetricPart(value)
            | ExprNode::IsotropicLift(value)
            | ExprNode::Trace(value)
            | ExprNode::NormalComponent(value) => pending.push(*value),
            ExprNode::Complex {
                real: left,
                imag: right,
            }
            | ExprNode::FiniteBinary(_, left, right)
            | ExprNode::Compare(_, left, right)
            | ExprNode::And(left, right)
            | ExprNode::Or(left, right)
            | ExprNode::Add(left, right)
            | ExprNode::Sub(left, right)
            | ExprNode::Mul(left, right)
            | ExprNode::Div(left, right)
            | ExprNode::Quotient(left, right)
            | ExprNode::Remainder(left, right) => pending.extend([*left, *right]),
            ExprNode::PureOperatorApplication(application) => {
                pending.extend(application.arguments())
            }
            _ => {
                return Err(Diagnostic::error(
                    codes::NOT_IMPLEMENTED,
                    "expression is newer than scalar structural incidence analysis",
                ));
            }
        }
    }
    Ok(coordinates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{Id, entity::kinds};
    use eqiora_schema::kernel::ExprDagBuilder;

    #[test]
    fn value_and_derivative_occurrences_are_distinct_before_balance_projection() {
        let field = Id::<kinds::Field>::new();
        let mut builder = ExprDagBuilder::new();
        let value = builder.symbol(SymbolRef::Field(field)).unwrap();
        let derivative = builder.symbol(SymbolRef::Derivative(field)).unwrap();
        let dag = builder.finish([value, derivative]).unwrap();
        assert_eq!(
            variables(&dag, dag.roots(), &BTreeMap::new()).unwrap(),
            BTreeSet::from([
                Variable::Field(field.erase()),
                Variable::Derivative(field.erase())
            ])
        );
    }
}
