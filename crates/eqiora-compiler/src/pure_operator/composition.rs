//! Whole call arguments use the existing closed component-definition representation.
use super::*;
use eqiora_schema::kernel::pure_operator::CalculusArgument;
use eqiora_schema::kernel::typing::ExpressionType;

/// Static channel selection uses the same complete index contract and typed
/// projection for parsed local definitions and retained Model expressions.
pub(crate) fn indexed_value(
    caller: &mut CalculusBuilder,
    argument: CalculusArgument,
    input: eqiora_core::ValueType,
    index: u32,
) -> Result<CalculusNodeId, PureOperatorError> {
    let output = ExpressionType::<()>::new(input.clone(), None)
        .index(index)
        .map_err(|_| PureOperatorError::FormalTypeMismatch)?
        .value_type;
    let mut projection = CalculusBuilder::new(
        [PureValueClass::from_value_type(input)?],
        PureValueClass::from_value_type(output.clone())?,
    )?;
    let root = projection.push(CalculusNode::FormalComponent {
        formal: 0,
        axes: std::iter::once(ComponentIndex::Fixed(index))
            .chain((0..output.shape().rank()).map(|axis| ComponentIndex::Result(axis as u16)))
            .collect(),
    })?;
    caller.apply_inputs(&projection.finish(root)?, &[argument])
}

pub(super) fn compile_index(
    file: &str,
    expression: &Expr,
    formals: &FormalBindings<'_>,
    sources: &BTreeMap<&str, &PureOperatorDecl>,
    compiled: &BTreeMap<String, PureOperatorDefinition>,
    builder: &mut CalculusBuilder,
) -> Result<CalculusNodeId, Diagnostic> {
    let ExprKind::Index { value, index } = expression.kind() else {
        unreachable!("indexed source expression")
    };
    let class = argument_class(file, value, formals, compiled)?
        .and_then(|class| class.exact_type().cloned())
        .ok_or_else(|| {
            pure_error(
                file,
                value.range(),
                "channel index requires a complete array type",
            )
        })?;
    let index = crate::hierarchy::closed_index(index).map_err(|_| {
        pure_error(
            file,
            index.range(),
            "channel index requires a constant nonnegative integer",
        )
    })?;
    let value = compile_argument(file, value, formals, sources, compiled, builder)?;
    indexed_value(builder, value, class, index)
        .map_err(|error| kernel_error(file, expression.range(), error))
}

fn argument_class(
    file: &str,
    expression: &Expr,
    formals: &FormalBindings<'_>,
    compiled: &BTreeMap<String, PureOperatorDefinition>,
) -> Result<Option<PureValueClass>, Diagnostic> {
    match expression.kind() {
        ExprKind::Name(name) => Ok(formals
            .names
            .get(name.as_str())
            .map(|index| formals.classes[usize::from(*index)].clone())),
        ExprKind::Call { callee, arguments } if callee.as_str() == "jvp" => {
            let (value, _, _) = actions::arguments(file, expression, arguments)?;
            argument_class(file, value, formals, compiled)
        }
        ExprKind::Call { callee, arguments } if callee.as_str() == "vjp" => {
            let (_, selected, _) = actions::vjp_arguments(file, expression, arguments)?;
            let Some(input) = argument_class(file, selected, formals, compiled)?
                .and_then(|class| class.exact_type().cloned())
            else {
                return Ok(None);
            };
            let output = ExpressionType::<()>::new(input, None)
                .dual()
                .map_err(|error| pure_error(file, expression.range(), error.to_string()))?;
            PureValueClass::from_value_type(output.value_type)
                .map(Some)
                .map_err(|error| kernel_error(file, expression.range(), error))
        }
        ExprKind::Call { callee, .. } => Ok(compiled
            .get(callee.as_str())
            .map(|definition| definition.result_rule().clone())),
        ExprKind::Index { value, index } => {
            let Some(input) = argument_class(file, value, formals, compiled)?
                .and_then(|class| class.exact_type().cloned())
            else {
                return Ok(None);
            };
            let index = crate::hierarchy::closed_index(index).map_err(|_| {
                pure_error(
                    file,
                    index.range(),
                    "channel index requires a constant nonnegative integer",
                )
            })?;
            let output = ExpressionType::<()>::new(input, None)
                .index(index)
                .map_err(|error| pure_error(file, expression.range(), error.to_string()))?;
            PureValueClass::from_value_type(output.value_type)
                .map(Some)
                .map_err(|error| kernel_error(file, expression.range(), error))
        }
        _ => Ok(None),
    }
}

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
    let class = argument_class(file, expression, formals, compiled)?;
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
