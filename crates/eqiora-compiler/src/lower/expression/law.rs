//! Lower exact Law terms and derive their single fixed-domain balance.

use super::*;
use eqiora_schema::kernel::ConservationTerms;

pub(in crate::lower) fn lower_law(
    file: &str,
    range: TextRange,
    domain: &str,
    storage: Option<&LoweringExpression>,
    flux: &LoweringExpression,
    source: &LoweringExpression,
    bindings: &BTreeMap<String, Binding>,
) -> Result<(LoweredRelation, ConservationTerms), Diagnostic> {
    let support = relation_support(file, range, domain, bindings)?;
    if !matches!(support, SpatialSupport::Volume { .. }) {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            "fixed-domain Law requires a volume support",
        ));
    }
    let storage = storage
        .map(|value| contextual::value(file, value, bindings))
        .transpose()?;
    let flux = contextual::value(file, flux, bindings)?;
    let source = contextual::value(file, source, bindings)?;
    let divergence = LoweringExpression::call("div".to_owned(), flux.clone(), range);
    let accumulation = storage
        .as_ref()
        .map(|value| LoweringExpression::call("derivative".to_owned(), value.clone(), range));
    let left = if let Some(accumulation) = &accumulation {
        LoweringExpression::binary(BinaryOp::Add, accumulation.clone(), divergence, range)
    } else {
        divergence
    };
    let left_type = expression_type(file, &left, bindings, Some(&support))?;
    let source_type = expression_type(file, &source, bindings, Some(&support))?;
    // An explicit source zero is a mathematical zero, with the balance's units.
    let source = if lowering_integer_literal(&source) == Some(0) {
        LoweringExpression::literal(
            eqiora_core::ValueLiteral::from_real(left_type.value_type.clone(), 0.0).map_err(
                |error| source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.to_string()),
            )?,
            source.range(),
        )
    } else {
        typing::additive(&left_type, &source_type).map_err(|error| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                source.range(),
                error.to_string(),
            )
        })?;
        source
    };
    typing::residual(&left_type, Some(&support)).map_err(|error| {
        source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.to_string())
    })?;
    let activation = ActivationSyntax::Continuous;
    let mut lowerer = ExpressionLowerer {
        file,
        bindings,
        builder: ExprDagBuilder::new(),
        dependencies: BTreeSet::new(),
        ports: BTreeSet::new(),
        cache: HashMap::new(),
        sampling: false,
        allow_discrete_symbols: false,
        activation: &activation,
        initial: false,
    };
    // Keep all source expression owners live while using the pointer-keyed cache.
    let storage = storage
        .as_ref()
        .zip(accumulation.as_ref())
        .map(|(stored, accumulation)| {
            Ok::<_, Diagnostic>((lowerer.lower(stored)?.id, lowerer.lower(accumulation)?.id))
        })
        .transpose()?;
    let flux = lowerer.lower(&flux)?.id;
    let source = lowerer.lower(&source)?.id;
    let left = lowerer.lower(&left)?.id;
    let expression = lowerer.builder.finish([left, source])?;
    let terms = ConservationTerms::new(storage, flux, source);
    terms.validate_balance(&expression)?;
    Ok((
        LoweredRelation {
            expression,
            dependencies: lowerer.dependencies,
            ports: lowerer.ports,
        },
        terms,
    ))
}
