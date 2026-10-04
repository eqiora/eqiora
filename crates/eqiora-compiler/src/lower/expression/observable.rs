//! Observable roots share ordinary expression lowering and the Kernel type contract.
use super::*;
use eqiora_schema::kernel::{ObservableDef, ObservableMeasure, ObservableReduction};

pub(in crate::lower) fn lower_observable(
    file: &str,
    id: Id<kinds::Observable>,
    value_type: &eqiora_lang::ValueTypeSyntax,
    value: &LoweringExpression,
    reduction: Option<&LoweringIntegral>,
    domain: Option<&str>,
    bindings: &BTreeMap<String, Binding>,
) -> Result<(ObservableDef, BTreeSet<RawId>), Diagnostic> {
    let range = value.range;
    let support = reduction
        .map(|reduction| relation_support(file, range, &reduction.domain, bindings))
        .transpose()?;
    let output = domain
        .map(|name| relation_support(file, range, name, bindings))
        .transpose()?;
    let value_type = crate::value_types::lower_value_type(
        file,
        value_type,
        output.as_ref().or(support.as_ref()),
    )?;
    let value = contextual::typed_value(
        file,
        value,
        bindings,
        support.as_ref().or(output.as_ref()),
        value_type.scalar_domain(),
    )?;
    let inferred = expression_type(file, &value, bindings, support.as_ref().or(output.as_ref()))?;
    let input = reduction
        .and(inferred.support.as_ref().or(support.as_ref()))
        .cloned();
    let typed_limits = reduction
        .and_then(|reduction| reduction.limits.as_ref())
        .map(|limits| {
            limits
                .iter()
                .map(|limit| {
                    let value = contextual::typed_value(
                        file,
                        limit,
                        bindings,
                        support.as_ref(),
                        eqiora_core::ScalarDomain::Real,
                    )?;
                    let ty = expression_type(file, &value, bindings, support.as_ref())?;
                    Ok((value, ty))
                })
                .collect::<Result<Vec<_>, Diagnostic>>()
        })
        .transpose()?;
    let mut reduction = match reduction {
        None => ObservableReduction::Value,
        Some(reduction) => {
            let name = &reduction.domain;
            let Some(Binding::Domain(domain, _)) = bindings.get(name) else {
                return Err(unresolved(file, range, name, "Observable Domain"));
            };
            ObservableReduction::SpatialIntegral {
                input: input
                    .as_ref()
                    .expect("integral input")
                    .domain()
                    .downcast()
                    .expect("admitted Domain"),
                domain: *domain,
                limits: None,
                measure: reduction.measure.unwrap_or({
                    if matches!(support.as_ref(), Some(SpatialSupport::Boundary { .. })) {
                        ObservableMeasure::Boundary
                    } else {
                        ObservableMeasure::Volume
                    }
                }),
            }
        }
    };
    let mut lowerer = ExpressionLowerer {
        file,
        bindings,
        support: input.clone().or(output.clone()),
        builder: ExprDagBuilder::new(),
        dependencies: BTreeSet::new(),
        ports: BTreeSet::new(),
        cache: HashMap::new(),
        sampling: false,
        allow_discrete_symbols: true,
        allow_observables: true,
        activation: &ActivationSyntax::Continuous,
        initial: false,
    };
    let root = lowerer.lower(&value)?.id;
    if let Some(values) = typed_limits.as_ref() {
        let lower = lowerer.lower(&values[0].0)?.id;
        let upper = lowerer.lower(&values[1].0)?.id;
        let ObservableReduction::SpatialIntegral { limits, .. } = &mut reduction else {
            unreachable!()
        };
        *limits = Some([lower, upper]);
    }
    let expression = lowerer
        .builder
        .finish([root])
        .map_err(|error| source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.message()))?;
    let definition = ObservableDef::new(id, value_type, expression, reduction)?;
    definition
        .validate_type(
            &inferred,
            typed_limits
                .as_ref()
                .map(|values| [&values[0].1, &values[1].1]),
            input.as_ref(),
            support.as_ref(),
            output.as_ref(),
        )
        .map_err(|error| source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.message()))?;
    Ok((definition, lowerer.dependencies))
}

/// References read the declared reduced value type, not its density type.
pub(super) fn reference_type(
    file: &str,
    range: TextRange,
    value_type: &eqiora_lang::ValueTypeSyntax,
    domain: Option<&str>,
    reduction: Option<&LoweringIntegral>,
    bindings: &BTreeMap<String, Binding>,
) -> Result<ExpressionType<RawId>, Diagnostic> {
    let support = domain
        .map(|name| relation_support(file, range, name, bindings))
        .transpose()?;
    let measure_support = reduction
        .map(|reduction| relation_support(file, range, &reduction.domain, bindings))
        .transpose()?;
    let value_type = crate::value_types::lower_value_type(
        file,
        value_type,
        support.as_ref().or(measure_support.as_ref()),
    )?;
    Ok(ExpressionType::new(value_type, support))
}
