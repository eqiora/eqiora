//! Static validation of distributable hierarchy definitions.
//!
//! This phase owns no occurrence. It therefore creates no Model, Transaction,
//! graph identity, instance path, or provenance. Required public Parameters
//! are checked separately as symbolic interface obligations.

mod specialization;

use std::collections::BTreeMap;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{ComponentItem, ConnectorSyntax, DomainSyntax, Expr, ExprKind, Item};
use eqiora_schema::kernel::BoundarySide;

use crate::diagnostics::{BoundedDiagnostics, source_error};
use crate::dimensions::lower_dimension;

use super::complete_exterior::CartesianDomain;
use super::definition_graph::CheckedDefinitionGraph;
use super::field_slots::{
    component_field_contracts, component_field_interface, model_field_contracts,
    resolve_instance_fields,
};
use super::parameters::{
    SymbolicParameterMap, resolve_component_lets, resolve_component_parameters_symbolically,
    resolve_model_lets,
};
use super::preflight::{DefinitionKey, Elaborator};
use super::supports::{
    CompleteExteriorMembershipBudget, ResolvedBoundaryTarget, SupportInterface,
    component_support_interface, model_spatial_supports, resolve_instance_support_bindings,
    symbolic_complete_exterior_set,
};

pub(super) fn validate(
    elaborator: &Elaborator<'_>,
) -> Result<CheckedDefinitionGraph, Vec<Diagnostic>> {
    let mut diagnostics = BoundedDiagnostics::new(elaborator.limits.max_definition_diagnostics);
    let checked = match super::definition_graph::validate(elaborator) {
        Ok(checked) => Some(checked),
        Err(errors) => {
            diagnostics.extend(errors);
            None
        }
    };
    validate_connectors(elaborator, &mut diagnostics);
    let body_proofs =
        validate_definition_bodies_and_parameters(elaborator, checked.as_ref(), &mut diagnostics);
    if let Some(checked) = checked.as_ref() {
        super::physical_closure::validate(checked, &body_proofs, &mut diagnostics);
    }
    let valid = diagnostics.is_empty();
    let diagnostics = diagnostics.finish("package definition validation");
    match (valid, checked) {
        (true, Some(checked)) => Ok(checked),
        (false, _) => Err(diagnostics),
        (true, None) => unreachable!("definition graph failure emits a diagnostic"),
    }
}

fn validate_definition_bodies_and_parameters(
    elaborator: &Elaborator<'_>,
    checked: Option<&CheckedDefinitionGraph>,
    diagnostics: &mut BoundedDiagnostics,
) -> super::body_check::DefinitionBodyProofs {
    let mut body_proofs = super::body_check::DefinitionBodyProofs::default();
    let mut complete_exterior_budget =
        CompleteExteriorMembershipBudget::new(elaborator.limits.complete_exteriors);
    if let Err(error) = enforce_parameter_term_limit(elaborator) {
        diagnostics.push(error);
        return body_proofs;
    }

    let mut body_values = BTreeMap::<DefinitionKey, SymbolicParameterMap>::new();
    let mut support_interfaces = BTreeMap::<DefinitionKey, SupportInterface>::new();
    for (key, definition) in elaborator.components() {
        match component_support_interface(definition.file, definition.declaration) {
            Ok(interface) => {
                support_interfaces.insert(key.clone(), interface);
            }
            Err(errors) => diagnostics.extend(errors),
        }
        match resolve_component_parameters_symbolically(
            definition.file,
            definition.declaration,
            |name| super::clocks::component(definition.file, definition.declaration, name),
            &super::parameters::RecordContext::component(elaborator, definition),
        ) {
            Ok(parameters) => {
                let mut values = parameters.clone();
                if let Err(errors) = elaborator.bind_symbolic_properties(
                    &definition.namespace,
                    definition.file,
                    definition.declaration.signature(),
                    &mut values,
                ) {
                    diagnostics.extend(errors);
                    continue;
                }
                if let Err(errors) = resolve_component_lets(
                    definition.file,
                    definition.declaration,
                    &mut values,
                    |name| super::clocks::component(definition.file, definition.declaration, name),
                ) {
                    diagnostics.extend(errors);
                    continue;
                }
                body_values.insert(key.clone(), values);
            }
            Err(errors) => diagnostics.extend(errors),
        }
    }

    let contexts = if let Some(checked) = checked
        && elaborator.components().any(|(key, definition)| {
            body_values
                .get(key)
                .is_some_and(|values| specialization::needs_context(definition, values))
        }) {
        match super::definition_graph::component_contexts(elaborator, checked) {
            Ok(contexts) => contexts,
            Err(errors) => {
                diagnostics.extend(errors);
                return body_proofs;
            }
        }
    } else {
        BTreeMap::new()
    };
    for (key, definition) in elaborator.components() {
        if let (Some(values), Some(supports)) = (body_values.get(key), support_interfaces.get(key))
        {
            match specialization::validate(
                elaborator,
                definition,
                values,
                contexts.get(key).map(Vec::as_slice),
                supports,
            ) {
                Ok(proof) => {
                    body_proofs.components.insert(key.clone(), proof);
                }
                Err(errors) => diagnostics.extend(errors),
            }
        }
    }

    for (key, definition) in elaborator.components() {
        let Some(base_values) = body_values.get(key) else {
            continue;
        };
        let parents = contexts
            .get(key)
            .filter(|values| !values.is_empty())
            .map_or_else(|| std::slice::from_ref(base_values), Vec::as_slice);
        for parent in parents {
            let Some(parent_supports) = support_interfaces.get(key) else {
                continue;
            };
            let fields = match component_field_interface(
                definition.file,
                definition.declaration,
                parent_supports,
                parent,
            ) {
                Ok(fields) => fields,
                Err(errors) => {
                    diagnostics.extend(errors);
                    continue;
                }
            };
            let parent_fields = component_field_contracts(
                definition.file,
                definition.declaration,
                parent_supports,
                &fields,
                parent,
                &super::parameters::RecordContext::component(elaborator, definition),
            );
            let mut occurrences_valid = true;
            let mut parent_boundary_sets = BTreeMap::new();
            if let Some(parent_supports) = support_interfaces.get(key) {
                for (name, contract) in parent_supports.complete_exteriors() {
                    match symbolic_complete_exterior_set(
                        definition.file,
                        name,
                        contract,
                        definition.declaration.range(),
                    ) {
                        Ok(set) => {
                            parent_boundary_sets.insert(name.to_owned(), set);
                        }
                        Err(error) => {
                            occurrences_valid = false;
                            diagnostics.push(error);
                        }
                    }
                }
            }
            for item in definition.declaration.items() {
                let ComponentItem::Instance(instance) = item else {
                    continue;
                };
                let extent = match specialization::instance_extent(
                    definition.file,
                    instance,
                    definition
                        .declaration
                        .items()
                        .iter()
                        .filter_map(|item| match item {
                            ComponentItem::IndexSet(set) => Some(set),
                            _ => None,
                        }),
                    parent,
                    elaborator.limits.max_instances,
                ) {
                    Ok(extent) => extent,
                    Err(error) => {
                        occurrences_valid = false;
                        diagnostics.push(error);
                        continue;
                    }
                };
                let ordinals = (0..extent.unwrap_or(0))
                    .map(Some)
                    .chain(extent.is_none().then_some(None));
                for ordinal in ordinals {
                    let member = match ordinal.map_or_else(
                        || Ok(instance.clone()),
                        |ordinal| {
                            super::reductions::instantiate_instance(
                                definition.file,
                                instance,
                                ordinal,
                            )
                        },
                    ) {
                        Ok(member) => member,
                        Err(error) => {
                            occurrences_valid = false;
                            diagnostics.push(error);
                            continue;
                        }
                    };
                    let instance = &member;
                    let Ok(child) = elaborator.resolve_component(
                        &definition.namespace,
                        instance.definition(),
                        definition.file,
                        instance.range(),
                    ) else {
                        continue;
                    };
                    let child_key = DefinitionKey {
                        namespace: child.namespace.clone(),
                        name: child.declaration.name().to_owned(),
                    };
                    let child_values =
                        match super::parameters::resolve_instance_parameters_symbolically(
                            (child.file, definition.file),
                            child.declaration,
                            instance,
                            parent,
                            &mut |name| {
                                super::clocks::component(
                                    definition.file,
                                    definition.declaration,
                                    name,
                                )
                            },
                            &mut |name| {
                                parent_supports
                                    .get(name)
                                    .map(|contract| contract.support().clone())
                            },
                            (
                                &super::parameters::RecordContext::component(elaborator, &child),
                                &super::parameters::RecordContext::component(
                                    elaborator, definition,
                                ),
                            ),
                        ) {
                            Ok(values) => values,
                            Err(errors) => {
                                occurrences_valid = false;
                                diagnostics.extend(errors);
                                continue;
                            }
                        };
                    let (Some(parent_supports), Some(child_supports)) = (
                        support_interfaces.get(key),
                        support_interfaces.get(&child_key),
                    ) else {
                        occurrences_valid = false;
                        continue;
                    };
                    let support_bindings = match resolve_instance_support_bindings(
                        definition.file,
                        child.declaration,
                        child_supports,
                        instance,
                        |name| parent_supports.visible_support(name).cloned(),
                        |_| None,
                        |_| None,
                        |name| parent_boundary_sets.get(name).cloned(),
                        &mut complete_exterior_budget,
                    ) {
                        Ok(bindings) => Some(bindings),
                        Err(errors) => {
                            occurrences_valid = false;
                            diagnostics.extend(errors);
                            None
                        }
                    };
                    let child_fields = match component_field_interface(
                        child.file,
                        child.declaration,
                        child_supports,
                        &child_values,
                    ) {
                        Ok(fields) => fields,
                        Err(errors) => {
                            occurrences_valid = false;
                            diagnostics.extend(errors);
                            continue;
                        }
                    };
                    if let Some(support_bindings) = support_bindings
                        && let Err(errors) = resolve_instance_fields(
                            definition.file,
                            child.declaration,
                            &child_fields,
                            instance,
                            |name| {
                                definition
                            .declaration
                            .items()
                            .iter()
                            .any(|item| match item {
                                ComponentItem::Clock(c) => c.name() == name,
                                _ => false,
                            })
                            .then(|| name.to_owned())
                            .or_else(|| definition.declaration.signature().iter().any(|item|matches!(item,eqiora_lang::SignatureItem::Clock(c) if c.name()==name)).then(||name.to_owned()))
                            },
                            |slot| {
                                support_bindings
                                    .singular_targets()
                                    .get(slot)
                                    .and_then(|target| parent_supports.visible_support(target))
                                    .cloned()
                            },
                            |name| parent_fields.get(name).cloned(),
                        )
                    {
                        occurrences_valid = false;
                        diagnostics.extend(errors);
                    }
                }
                if !occurrences_valid {
                    body_proofs.components.remove(key);
                }
            }
        }
    }

    for (key, definition) in elaborator.models() {
        let mut parameters = SymbolicParameterMap::new();
        let mut occurrences_valid = true;
        let model_supports = match model_spatial_supports(definition.file, definition.declaration) {
            Ok(supports) => Some(supports),
            Err(errors) => {
                occurrences_valid = false;
                diagnostics.extend(errors);
                None
            }
        };
        match super::parameters::resolve_model_parameters_symbolically(
            definition.file,
            definition.declaration,
            |name| super::clocks::model(definition.file, definition.declaration, name),
            &super::parameters::RecordContext::model(elaborator, definition),
        ) {
            Ok(values) => parameters = values,
            Err(errors) => {
                occurrences_valid = false;
                diagnostics.extend(errors);
            }
        }
        if let Err(errors) = elaborator.bind_symbolic_properties(
            &definition.namespace,
            definition.file,
            definition.declaration.signature(),
            &mut parameters,
        ) {
            occurrences_valid = false;
            diagnostics.extend(errors);
        }
        if let Err(errors) = resolve_model_lets(
            definition.file,
            definition.declaration,
            &mut parameters,
            |name| super::clocks::model(definition.file, definition.declaration, name),
        ) {
            occurrences_valid = false;
            diagnostics.extend(errors);
        }
        let mut model_boundary_sets = BTreeMap::new();
        match super::supports::signature_support_interface(definition.file, definition.signature())
        {
            Ok(interface) => {
                for (name, contract) in interface.complete_exteriors() {
                    match symbolic_complete_exterior_set(
                        definition.file,
                        name,
                        contract,
                        definition.range(),
                    ) {
                        Ok(set) => {
                            model_boundary_sets.insert(name.to_owned(), set);
                        }
                        Err(error) => {
                            occurrences_valid = false;
                            diagnostics.push(error);
                        }
                    }
                }
            }
            Err(errors) => {
                occurrences_valid = false;
                diagnostics.extend(errors);
            }
        }
        let model_fields = model_supports
            .as_ref()
            .map(|supports| {
                model_field_contracts(
                    definition.file,
                    definition.declaration,
                    supports,
                    &parameters,
                    &super::parameters::RecordContext::model(elaborator, definition),
                )
            })
            .unwrap_or_default();
        match super::body_check::validate_model_body(elaborator, definition, &parameters) {
            Ok(proof) => {
                body_proofs.models.insert(key.clone(), proof);
            }
            Err(errors) => diagnostics.extend(errors),
        }
        for item in definition.declaration.items() {
            let Item::Instance(instance) = item else {
                continue;
            };
            let extent = match specialization::instance_extent(
                definition.file,
                instance,
                definition
                    .declaration
                    .items()
                    .iter()
                    .filter_map(|item| match item {
                        Item::IndexSet(set) => Some(set),
                        _ => None,
                    }),
                &parameters,
                elaborator.limits.max_instances,
            ) {
                Ok(extent) => extent,
                Err(error) => {
                    occurrences_valid = false;
                    diagnostics.push(error);
                    continue;
                }
            };
            let ordinals = (0..extent.unwrap_or(0))
                .map(Some)
                .chain(extent.is_none().then_some(None));
            for ordinal in ordinals {
                let member = match ordinal.map_or_else(
                    || Ok(instance.clone()),
                    |ordinal| {
                        super::reductions::instantiate_instance(definition.file, instance, ordinal)
                    },
                ) {
                    Ok(member) => member,
                    Err(error) => {
                        occurrences_valid = false;
                        diagnostics.push(error);
                        continue;
                    }
                };
                let instance = &member;
                let Ok(child) = elaborator.resolve_component(
                    &definition.namespace,
                    instance.definition(),
                    definition.file,
                    instance.range(),
                ) else {
                    continue;
                };
                let child_key = DefinitionKey {
                    namespace: child.namespace.clone(),
                    name: child.declaration.name().to_owned(),
                };
                let child_values = match super::parameters::resolve_instance_parameters_symbolically(
                    (child.file, definition.file),
                    child.declaration,
                    instance,
                    &parameters,
                    &mut |name| super::clocks::model(definition.file, definition.declaration, name),
                    &mut |name| {
                        model_supports
                            .as_ref()
                            .and_then(|supports| supports.get(name))
                            .cloned()
                    },
                    (
                        &super::parameters::RecordContext::component(elaborator, &child),
                        &super::parameters::RecordContext::model(elaborator, definition),
                    ),
                ) {
                    Ok(values) => values,
                    Err(errors) => {
                        occurrences_valid = false;
                        diagnostics.extend(errors);
                        continue;
                    }
                };
                let Some(child_supports) = support_interfaces.get(&child_key) else {
                    occurrences_valid = false;
                    continue;
                };
                let Some(model_supports) = model_supports.as_ref() else {
                    occurrences_valid = false;
                    continue;
                };
                let support_bindings = match super::supports::resolve_definition_support_bindings(
                    definition.file,
                    child.declaration,
                    child_supports,
                    instance,
                    |name| model_supports.get(name).cloned(),
                    |name| match model_supports.get(name) {
                        Some(eqiora_schema::kernel::typing::SpatialSupport::Boundary {
                            domain,
                            ..
                        }) => Some(ResolvedBoundaryTarget::new(name.to_owned(), domain.clone())),
                        _ => None,
                    },
                    |identity| match model_supports.get(identity)? {
                        eqiora_schema::kernel::typing::SpatialSupport::Volume {
                            dimensions,
                            ..
                        } => Some(CartesianDomain::Volume {
                            ambient_dimension: *dimensions,
                        }),
                        eqiora_schema::kernel::typing::SpatialSupport::Boundary {
                            parent,
                            dimensions,
                            ..
                        } => {
                            if let Some((axis, side)) = elaborator
                                .selected_boundary_sides
                                .get(&(key.clone(), identity.clone()))
                            {
                                return Some(CartesianDomain::Boundary {
                                    exact_parent: parent.clone(),
                                    ambient_dimension: *dimensions,
                                    axis: *axis,
                                    side: *side,
                                });
                            }
                            let declaration = definition.declaration.items().iter().find_map(
                                |item| match item {
                                    Item::Domain(domain) if domain.name() == identity => {
                                        Some(domain)
                                    }
                                    _ => None,
                                },
                            )?;
                            let DomainSyntax::Boundary { axis, side, .. } = declaration.syntax()
                            else {
                                return None;
                            };
                            Some(CartesianDomain::Boundary {
                                exact_parent: parent.clone(),
                                ambient_dimension: *dimensions,
                                axis: *axis,
                                side: match side {
                                    eqiora_lang::BoundarySideSyntax::Lower => BoundarySide::Lower,
                                    eqiora_lang::BoundarySideSyntax::Upper => BoundarySide::Upper,
                                },
                            })
                        }
                        eqiora_schema::kernel::typing::SpatialSupport::Interface { .. } => None,
                    },
                    |name| model_boundary_sets.get(name).cloned(),
                    |name| {
                        !elaborator.selected_models.contains(key) && definition.signature().iter().any(|item| {
                        matches!(item, eqiora_lang::SignatureItem::Support(slot) if slot.name() == name && matches!(slot.syntax(), eqiora_lang::SupportSlotSyntax::Boundary { .. }))
                    })
                    },
                    &mut complete_exterior_budget,
                ) {
                    Ok(bindings) => Some(bindings),
                    Err(errors) => {
                        occurrences_valid = false;
                        diagnostics.extend(errors);
                        None
                    }
                };
                let child_fields = match component_field_interface(
                    child.file,
                    child.declaration,
                    child_supports,
                    &child_values,
                ) {
                    Ok(fields) => fields,
                    Err(errors) => {
                        occurrences_valid = false;
                        diagnostics.extend(errors);
                        continue;
                    }
                };
                if let Some(support_bindings) = support_bindings
                    && let Err(errors) = resolve_instance_fields(
                        definition.file,
                        child.declaration,
                        &child_fields,
                        instance,
                        |name| {
                            definition
                                .declaration
                                .items()
                                .iter()
                                .any(|item| matches!(item, Item::Clock(c) if c.name() == name))
                                .then(|| name.to_owned())
                        },
                        |slot| {
                            support_bindings
                                .singular_targets()
                                .get(slot)
                                .and_then(|target| model_supports.get(target))
                                .cloned()
                        },
                        |name| model_fields.get(name).cloned(),
                    )
                {
                    occurrences_valid = false;
                    diagnostics.extend(errors);
                }
            }
            if !occurrences_valid {
                body_proofs.models.remove(key);
            }
        }
    }
    body_proofs
}

pub(super) fn enforce_parameter_term_limit(elaborator: &Elaborator<'_>) -> Result<(), Diagnostic> {
    let mut terms = 0_usize;
    for (_, definition) in elaborator.components() {
        for item in definition.declaration.items() {
            match item {
                ComponentItem::Let(declaration) => {
                    count_expression_terms(declaration.value(), &mut terms, elaborator)?;
                }
                ComponentItem::Parameter(parameter) => {
                    increment_parameter_terms(&mut terms, 1, elaborator)?;
                    if let Some(default) = parameter.default() {
                        count_expression_terms(default, &mut terms, elaborator)?;
                    }
                }
                ComponentItem::Instance(instance) => {
                    for binding in instance.bindings() {
                        count_expression_terms(binding.value(), &mut terms, elaborator)?;
                    }
                }
                _ => {}
            }
        }
    }
    for (_, definition) in elaborator.models() {
        for item in definition.declaration.items() {
            match item {
                Item::Let(declaration) => {
                    count_expression_terms(declaration.value(), &mut terms, elaborator)?;
                }
                Item::Instance(instance) => {
                    for binding in instance.bindings() {
                        count_expression_terms(binding.value(), &mut terms, elaborator)?;
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn count_expression_terms(
    expression: &Expr,
    terms: &mut usize,
    elaborator: &Elaborator<'_>,
) -> Result<(), Diagnostic> {
    let mut pending = vec![expression];
    while let Some(expression) = pending.pop() {
        let cost = match expression.kind() {
            ExprKind::Case { arms, .. } => arms.len().saturating_sub(1).max(1).saturating_mul(3),
            ExprKind::Call { callee, .. } => {
                crate::math::piecewise::cost(callee.as_str()).map_or(1, |cost| cost.0)
            }
            _ => 1,
        };
        increment_parameter_terms(terms, cost, elaborator)?;
        match expression.kind() {
            ExprKind::Case { value, arms } => {
                pending.push(value);
                pending.extend(arms.iter().map(|arm| arm.value()));
            }
            ExprKind::Select {
                condition,
                then_value,
                else_value,
            } => pending.extend([condition.as_ref(), then_value.as_ref(), else_value.as_ref()]),
            ExprKind::Reduction { value, .. } => pending.push(value),
            ExprKind::Array(elements) => pending.extend(elements),
            ExprKind::Index { value, index } => pending.extend([value.as_ref(), index.as_ref()]),
            ExprKind::Slice {
                value,
                lower,
                upper,
            } => pending.extend([value.as_ref(), lower.as_ref(), upper.as_ref()]),
            ExprKind::Unary { value, .. } => pending.push(value),
            ExprKind::Call { arguments, .. } => pending.extend(arguments.expressions()),
            ExprKind::Binary { left, right, .. } => {
                pending.push(right);
                pending.push(left);
            }
            _ => {}
        }
    }
    Ok(())
}

fn increment_parameter_terms(
    terms: &mut usize,
    additional: usize,
    elaborator: &Elaborator<'_>,
) -> Result<(), Diagnostic> {
    *terms = terms
        .checked_add(additional)
        .ok_or_else(|| definition_error("symbolic Parameter term count overflows usize"))?;
    if *terms > elaborator.limits.max_parameter_terms {
        return Err(definition_error(format!(
            "resolved hierarchy exceeds the {} symbolic Parameter term limit",
            elaborator.limits.max_parameter_terms
        )));
    }
    Ok(())
}

fn validate_connectors(elaborator: &Elaborator<'_>, diagnostics: &mut BoundedDiagnostics) {
    for (_, definition) in elaborator.connectors() {
        match definition.declaration.syntax() {
            ConnectorSyntax::ScalarPhysical {
                across_type,
                through_type,
                ..
            } => {
                if let Err(error) =
                    crate::value_types::lower_scalar_type(definition.file, across_type)
                {
                    diagnostics.push(error);
                }
                if let Err(error) =
                    crate::value_types::lower_scalar_type(definition.file, through_type)
                {
                    diagnostics.push(error);
                }
            }
            ConnectorSyntax::FieldPhysical { trace, flux, .. } => {
                if let Err(error) = lower_dimension(definition.file, trace.dimension()) {
                    diagnostics.push(error);
                }
                if let Err(error) = lower_dimension(definition.file, flux.dimension()) {
                    diagnostics.push(error);
                }
            }
            _ => diagnostics.push(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                definition.file,
                definition.declaration.range(),
                "Connector syntax is newer than definition validation",
            )),
        }
    }
}

fn definition_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::LANGUAGE_LOWERING_ERROR, message)
}

#[cfg(test)]
mod tests {
    use eqiora_lang::parse;

    use crate::source_identity::LocalSourceIdentity;

    use super::*;
    use crate::hierarchy::HierarchyLimits;

    #[test]
    fn definition_diagnostics_are_bounded_with_an_explicit_truncation() {
        let source = "model A() { relation bad { missing_a = 0; } } model B() { relation bad { missing_b = 0; } }";
        let document = parse("bounded-diagnostics.eqi", source)
            .into_document()
            .expect("fixture parses");
        let identity = LocalSourceIdentity::from_document(&document).expect("source identity");
        let limits = HierarchyLimits {
            max_definition_diagnostics: 1,
            ..HierarchyLimits::default()
        };
        let elaborator = Elaborator::new(
            "bounded-diagnostics.eqi",
            source.len(),
            &document,
            identity,
            limits,
        )
        .expect("definitions index");

        let diagnostics = validate(&elaborator).expect_err("both Models are invalid");
        assert_eq!(diagnostics.len(), 2);
        assert!(diagnostics[0].source_span().is_some());
        assert!(diagnostics[1].message().contains("omitted 1"));
        assert!(diagnostics[1].source_span().is_none());
    }
}
