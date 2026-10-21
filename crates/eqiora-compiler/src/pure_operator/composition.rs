//! Whole call arguments use the existing closed component-definition representation.
use super::*;
use eqiora_schema::kernel::pure_operator::CalculusArgument;

pub(super) fn compile_argument(
    file: &str,
    expression: &Expr,
    formals: &FormalBindings<'_>,
    sources: &BTreeMap<&str, &PureOperatorDecl>,
    compiled: &BTreeMap<String, PureOperatorDefinition>,
    builder: &mut CalculusBuilder,
) -> Result<CalculusArgument, Diagnostic> {
    let formal = match expression.kind() {
        ExprKind::Name(name) => formals.names.get(name.as_str()).copied(),
        _ => None,
    };
    let class = formal
        .map(|index| &formals.classes[usize::from(index)])
        .or_else(|| {
            let ExprKind::Call { callee, .. } = expression.kind() else {
                return None;
            };
            compiled
                .get(callee.as_str())
                .map(PureOperatorDefinition::result_rule)
        });
    let Some(class) = class.filter(|class| !class.is_invariant_scalar()) else {
        return compile_expression(file, expression, formals, sources, compiled, builder)
            .map(CalculusArgument::Scalar);
    };
    let mut value = CalculusBuilder::new(formals.classes.clone(), class.clone())
        .map_err(|error| kernel_error(file, expression.range(), error))?;
    let root = if let Some(formal) = formal {
        value
            .push(CalculusNode::FormalComponent {
                formal,
                axes: (0..class.rank())
                    .map(|axis| ComponentIndex::Result(axis as u16))
                    .collect(),
            })
            .map_err(|error| kernel_error(file, expression.range(), error))?
    } else {
        compile_expression(file, expression, formals, sources, compiled, &mut value)?
    };
    value
        .finish(root)
        .map(CalculusArgument::from)
        .map_err(|error| kernel_error(file, expression.range(), error))
}
