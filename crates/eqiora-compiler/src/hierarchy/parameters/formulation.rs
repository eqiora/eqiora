//! Shared live coefficient aliases for Model and Component forms.
use super::*;

/// Preserve the same selected Parameter expressions used by strong-law lowering.
pub(in crate::hierarchy) fn resolve_formulation_coefficients(
    file: &str,
    component: &ComponentDecl,
    records: &RecordContext,
    compiled: &crate::CompiledModel,
) -> Result<BTreeMap<String, crate::formulation::AuthoredFormExpression>, Vec<Diagnostic>> {
    let resolver = SymbolicParameterResolver::component_interface(file, component, records)?;
    resolve(file, component, resolver, compiled, &mut |name| {
        super::super::clocks::component(file, component, name)
    })
}

pub(in crate::hierarchy) fn resolve_model_formulation_coefficients(
    file: &str,
    model: &eqiora_lang::ModelDecl,
    records: &RecordContext,
    compiled: &crate::CompiledModel,
) -> Result<BTreeMap<String, crate::formulation::AuthoredFormExpression>, Vec<Diagnostic>> {
    let resolver = super::model_parameters::resolver(
        file,
        model,
        RequiredParameterPolicy::RejectUnbound,
        super::super::supports::model_spatial_supports(file, model)?,
        records,
    )?;
    resolve(file, model, resolver, compiled, &mut |name| {
        super::super::clocks::model(file, model, name)
    })
}

fn resolve(
    file: &str,
    component: &impl crate::formulation::source::FormulationSource,
    mut resolver: SymbolicParameterResolver<'_>,
    compiled: &crate::CompiledModel,
    clock: &mut dyn FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
) -> Result<BTreeMap<String, crate::formulation::AuthoredFormExpression>, Vec<Diagnostic>> {
    resolver.required_policy = RequiredParameterPolicy::RejectUnbound;
    let mut parameters = BTreeMap::new();
    for name in resolver.declarations.keys() {
        let id = if let Some(id) = compiled.symbols().get(name) {
            id
        } else {
            let mut candidates = compiled
                .symbols()
                .iter()
                .filter(|(label, _)| label.ends_with(&format!(".{name}")));
            let Some((_, id)) = candidates.next() else {
                continue;
            };
            if candidates.next().is_some() {
                return Err(vec![hierarchy_error(format!(
                    "ambiguous authored Parameter `{name}`"
                ))]);
            }
            id
        };
        let Some((parameter, value)) = compiled.transaction().ops().iter().find_map(|op| {
            let eqiora_graph::Op::DefineKernelNode {
                node: eqiora_schema::kernel::KernelNode::Parameter(parameter),
            } = op
            else {
                return None;
            };
            if parameter.id().erase() != id {
                return None;
            }
            let value = compiled.transaction().ops().iter().find_map(|op| match op {
                eqiora_graph::Op::SetValue { target, value } if *target == id => {
                    Some(value.clone())
                }
                _ => None,
            });
            Some((parameter, value))
        }) else {
            continue;
        };
        parameters.insert(
            name.clone(),
            (parameter.id(), parameter.value_type().clone()),
        );
        resolver.bound_values.insert(
            name.clone(),
            SymbolicParameterValue {
                value,
                value_type: parameter.value_type().clone(),
                expression: Some(LoweringExpression::name(name.clone(), component.range())),
                // Live dependencies are not constants, even when a current value exists.
                lineage: Some(ParameterLineage::Derived),
            },
        );
    }
    let mut referenced = BTreeSet::new();
    for (name, _, equations, _) in component.formulations() {
        if let Some(eqiora_lang::FormulationBinding::Harmonic {
            angular_frequency,
            excitations,
            ..
        }) = component.formulation_binding(name)
        {
            for expression in
                std::iter::once(angular_frequency).chain(excitations.iter().map(|(_, value)| value))
            {
                let _ = expression.rewrite_name_paths(|name| {
                    referenced.insert(name.as_str().to_owned());
                    None
                });
            }
        }
        let gauge = component.formulation_gauge(name);
        for (left, right) in equations.iter().chain(
            gauge
                .into_iter()
                .flat_map(|(_, equations)| equations.iter()),
        ) {
            for expression in [left, right] {
                let _ = expression.rewrite_name_paths(|name| {
                    referenced.insert(name.as_str().to_owned());
                    None
                });
            }
        }
    }
    let resolved = resolver.resolve_all(clock)?;
    resolved
        .into_iter()
        .filter(|(name, _)| referenced.contains(name) && !parameters.contains_key(name))
        .map(|(name, value)| {
            let expression = value.expression.ok_or_else(|| {
                vec![hierarchy_error(format!(
                    "authored coefficient `{name}` has no specialized expression"
                ))]
            })?;
            let (dag, value_type) =
                crate::lower::lower_parameter_coefficient(file, &expression, &parameters)
                    .map_err(|error| vec![error])?;
            if value_type != value.value_type {
                return Err(vec![hierarchy_error(
                    "authored coefficient lost its exact declared type",
                )]);
            }
            crate::formulation::coefficient_alias(&dag, value_type)
                .map(|value| (name, value))
                .map_err(|error| vec![error])
        })
        .collect()
}
