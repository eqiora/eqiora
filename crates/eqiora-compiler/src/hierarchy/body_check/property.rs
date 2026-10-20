//! Definition-only property typing: no release body or fabricated material value.
use super::scope::DefinitionScope;
use crate::{diagnostics::source_error, property::Contract};
use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_lang::{CallArguments, Expr, SignatureItem, TextRange};
use eqiora_schema::kernel::typing::{ExpressionType, SpatialSupport};

pub(super) fn bind(
    scope: &mut DefinitionScope<'_, '_>,
    signature: &[SignatureItem],
) -> Vec<Diagnostic> {
    let mut errors = Vec::new();
    for item in signature {
        if let SignatureItem::Property(requirement) = item {
            match scope.elaborator.resolve_property_contract(
                &scope.namespace,
                requirement.contract(),
                scope.file,
            ) {
                Ok(contract) => {
                    scope
                        .properties
                        .insert(requirement.name().to_owned(), contract.clone());
                }
                Err(error) => errors.push(error),
            }
        }
    }
    errors
}

pub(super) fn bare(
    file: &str,
    range: TextRange,
    contract: &Contract,
) -> Result<ExpressionType<String>, Diagnostic> {
    if !contract.inputs.is_empty() {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            "property requires its declared named inputs",
        ));
    }
    contract
        .result_type::<String>(None)
        .map(|value_type| ExpressionType::new(value_type, None))
}

pub(super) fn application(
    file: &str,
    range: TextRange,
    contract: &Contract,
    arguments: &CallArguments,
    mut infer: impl FnMut(&Expr) -> Result<ExpressionType<String>, Diagnostic>,
) -> Result<ExpressionType<String>, Diagnostic> {
    let arguments = crate::pure_operator::ordered_arguments(
        file,
        range,
        contract.inputs.iter().map(|(name, _)| name.as_str()),
        arguments,
    )?;
    let mut support = None;
    for ((name, syntax), argument) in contract.inputs.iter().zip(arguments) {
        let actual = infer(argument)?;
        let expected =
            crate::value_types::lower_value_type(&contract.file, syntax, actual.support.as_ref())?;
        if actual.value_type != expected {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                argument.range(),
                format!("property input `{name}` differs from its exact declared type"),
            ));
        }
        if let Some(actual_support) = actual.support {
            if !matches!(actual_support, SpatialSupport::Volume { .. }) {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    argument.range(),
                    "analytic property inputs require a common volume or uniform values",
                ));
            }
            if support
                .as_ref()
                .is_some_and(|expected| expected != &actual_support)
            {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    argument.range(),
                    "property inputs have different exact volume supports",
                ));
            }
            support = Some(actual_support);
        }
    }
    let result = contract.result_type(support.as_ref())?;
    Ok(ExpressionType::new(result, support))
}
