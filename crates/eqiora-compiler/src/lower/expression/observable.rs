//! Observable roots share ordinary expression lowering and the Kernel type contract.
use super::*;
use eqiora_schema::kernel::{ObservableDef, ObservableMeasure, ObservableReduction};

pub(in crate::lower) fn lower_observable(
    file: &str,
    range: TextRange,
    id: Id<kinds::Observable>,
    value_type: &eqiora_lang::ValueTypeSyntax,
    value: &LoweringExpression,
    reduction: Option<&String>,
    bindings: &BTreeMap<String, Binding>,
) -> Result<(ObservableDef, BTreeSet<RawId>), Diagnostic> {
    let support = reduction
        .map(|name| relation_support(file, range, name, bindings))
        .transpose()?;
    let reduction = match reduction {
        None => ObservableReduction::Value,
        Some(name) => {
            let Some(Binding::Domain(domain, _)) = bindings.get(name) else {
                return Err(unresolved(file, range, name, "Observable Domain"));
            };
            ObservableReduction::SpatialIntegral {
                domain: *domain,
                measure: if matches!(
                    support.as_ref(),
                    Some(eqiora_schema::kernel::typing::SpatialSupport::Boundary { .. })
                ) {
                    ObservableMeasure::Boundary
                } else {
                    ObservableMeasure::Volume
                },
            }
        }
    };
    let value_type = crate::value_types::lower_value_type(file, value_type, support.as_ref())?;
    let value = contextual::typed_value(
        file,
        value,
        bindings,
        support.as_ref(),
        value_type.scalar_domain(),
    )?;
    let inferred = expression_type(file, &value, bindings, support.as_ref())?;
    let mut lowerer = ExpressionLowerer {
        file,
        bindings,
        support: support.clone(),
        builder: ExprDagBuilder::new(),
        dependencies: BTreeSet::new(),
        ports: BTreeSet::new(),
        cache: HashMap::new(),
        sampling: false,
        allow_discrete_symbols: true,
        activation: &ActivationSyntax::Continuous,
        initial: false,
    };
    let root = lowerer.lower(&value)?.id;
    let expression = lowerer
        .builder
        .finish([root])
        .map_err(|error| source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.message()))?;
    let definition = ObservableDef::new(id, value_type, expression, reduction)?;
    definition
        .validate_type(&inferred, support.as_ref())
        .map_err(|error| source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.message()))?;
    Ok((definition, lowerer.dependencies))
}
