//! Property release bodies use exactly the existing typed scalar calculus.
use super::*;

pub(crate) fn compile_property(
    file: &str,
    document: &eqiora_lang::Document,
    inputs: &[(String, eqiora_lang::ValueTypeSyntax)],
    result: &eqiora_lang::ValueTypeSyntax,
    value: &Expr,
    scale: &Expr,
    validity: Option<&Expr>,
) -> Result<PureOperatorDefinition, Diagnostic> {
    let compiled = compile_definitions(file, document)?;
    let sources = document
        .pure_operators()
        .iter()
        .map(|value| (value.name(), value))
        .collect::<BTreeMap<_, _>>();
    let mut names = BTreeMap::new();
    let mut rules = Vec::new();
    for (index, (name, syntax)) in inputs.iter().enumerate() {
        let index = u16::try_from(index)
            .map_err(|_| pure_error(file, syntax.range(), "property formal limit exceeded"))?;
        if names.insert(name.as_str(), index).is_some() {
            return Err(pure_error(
                file,
                syntax.range(),
                "duplicate independent property input",
            ));
        }
        rules.push(value_class(
            file,
            syntax.range(),
            &PureValueClassSyntax::Typed(syntax.clone()),
        )?);
    }
    let result = value_class(
        file,
        result.range(),
        &PureValueClassSyntax::Typed(result.clone()),
    )?;
    let mut builder = CalculusBuilder::new(rules, result)
        .map_err(|error| kernel_error(file, value.range(), error))?;
    let root = compile_expression(file, value, &names, &sources, &compiled, &mut builder)?;
    let scale = compile_expression(
        file,
        scale,
        &BTreeMap::new(),
        &sources,
        &compiled,
        &mut builder,
    )?;
    let root = builder
        .push(CalculusNode::Mul(root, scale))
        .map_err(|error| kernel_error(file, value.range(), error))?;
    let root = if let Some(validity) = validity {
        let condition =
            compile_expression(file, validity, &names, &sources, &compiled, &mut builder)?;
        builder
            .push(CalculusNode::Require {
                condition,
                value: root,
            })
            .map_err(|error| kernel_error(file, validity.range(), error))?
    } else {
        root
    };
    builder
        .finish(root)
        .map_err(|error| kernel_error(file, value.range(), error))
}
