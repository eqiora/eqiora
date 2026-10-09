//! Selected authoring arguments enter the same dependency-ordered Parameter owner.
use super::*;
use crate::StaticBindingValue;
use eqiora_lang::SignatureItem;

pub(in crate::hierarchy) fn resolve_selected_parameters(
    file: &str,
    signature: &[SignatureItem],
    authored: &[(&str, StaticBindingValue<'_>)],
    frames: BTreeMap<String, SpatialSupport<String>>,
    records: &RecordContext,
) -> Result<Vec<super::super::ExternalParameterBinding>, Vec<Diagnostic>> {
    let authored_declarations = signature
        .iter()
        .filter_map(|item| match item {
            SignatureItem::Parameter(parameter) => {
                Some((parameter.name().to_owned(), parameter.clone()))
            }
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let declarations = records::expand(file, authored_declarations.clone(), records)?;
    let mut expressions = BTreeMap::new();
    let mut bound_values = BTreeMap::new();
    let mut clocks = BTreeMap::new();
    for &(name, value) in authored {
        if let Some(record) = authored_declarations
            .get(name)
            .and_then(|declaration| records.record_for_type(declaration.value_type()))
        {
            let StaticBindingValue::Expression(expression) = value else {
                return Err(vec![source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    authored_declarations[name].range(),
                    "selected record Parameter requires a closed named constructor expression",
                )]);
            };
            let values = crate::record::parameters::member_initializers(
                file,
                record,
                expression,
                |path| {
                    records
                        .visible
                        .get(path.as_str())
                        .map(|value| value.definition.id())
                },
                |_| None,
            )
            .map_err(|error| vec![error])?;
            for ((member, _), value) in record.definition.members().iter().zip(values) {
                expressions.insert(format!("{name}.{member}"), value);
            }
            continue;
        }
        match value {
            StaticBindingValue::Expression(value) if declarations.contains_key(name) => {
                expressions.insert(name.to_owned(), value.clone());
            }
            StaticBindingValue::Value(value) if declarations.contains_key(name) => {
                bound_values.insert(
                    name.to_owned(),
                    SymbolicParameterValue {
                        value: Some(value.clone()),
                        value_type: value.value_type().clone(),
                        expression: Some(LoweringExpression::literal(
                            value.clone(),
                            declarations[name].range(),
                        )),
                        lineage: Some(ParameterLineage::Constant),
                    },
                );
            }
            StaticBindingValue::Clock(clock) => {
                if let eqiora_schema::kernel::ClockKind::Periodic { period, .. } = clock.kind() {
                    clocks.insert(name.to_owned(), Some(period));
                }
            }
            _ => {}
        }
    }
    let resolver = SymbolicParameterResolver {
        declaration_file: file,
        declarations,
        bindings: Some(bindings::Bindings::closed(
            file,
            expressions,
            clocks.clone(),
            frames.clone(),
        )),
        bound_values,
        resolved: BTreeMap::new(),
        required_policy: RequiredParameterPolicy::RejectUnbound,
        frames,
    };
    let resolved = resolver.resolve_all(&mut |name| clocks.get(name).copied())?;
    resolved
        .iter()
        .filter(|(name, _)| {
            authored.iter().any(|(root, _)| {
                name.as_str() == *root
                    || (records.parameters.contains_key(*root)
                        && name.starts_with(&format!("{root}.")))
            })
        })
        .map(|(name, value)| {
            value
                .value
                .clone()
                .map(|value| super::super::ExternalParameterBinding::new(name, value))
                .ok_or_else(|| {
                    vec![hierarchy_error(format!(
                        "selected Parameter `{name}` remained symbolic"
                    ))]
                })
        })
        .collect()
}
