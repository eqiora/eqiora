//! Bound selected-root footprint before any occurrence graph allocation.
struct ChildFootprint<'a> {
    edges: Vec<Edge<'a>>,
    summaries: Vec<Option<DefinitionSummary>>,
    extra_connections: usize,
}

use super::super::{clocks, parameters};
use super::super::{
    support_environment::SupportEnvironment, supports::CompleteExteriorMembershipBudget,
};
use super::*;
use eqiora_lang::{ExprKind, InstanceDecl, NamedDefinitionDecl};
use parameters::SymbolicParameterMap;

type Values = Vec<(String, Option<eqiora_core::ValueLiteral>)>;

pub(in crate::hierarchy) fn selected_expansion_size(
    elaborator: &Elaborator<'_>,
    checked: &CheckedDefinitionGraph,
    model: &ModelDefinition<'_>,
) -> Result<super::super::preflight::ExpansionSize, Vec<Diagnostic>> {
    let values = model_values(elaborator, model)?;
    selected_expansion_size_with_contexts(elaborator, checked, model, values, None)
}

pub(in crate::hierarchy) type ComponentContexts =
    BTreeMap<DefinitionKey, Vec<SymbolicParameterMap>>;

pub(in crate::hierarchy) fn component_contexts(
    elaborator: &Elaborator<'_>,
    checked: &CheckedDefinitionGraph,
) -> Result<ComponentContexts, Vec<Diagnostic>> {
    let mut contexts = ComponentContexts::new();
    for (_, model) in elaborator.models() {
        let values = model_values(elaborator, model)?;
        selected_expansion_size_with_contexts(
            elaborator,
            checked,
            model,
            values,
            Some(&mut contexts),
        )?;
    }
    if let Some((key, values)) = &elaborator.selected_component {
        let component = elaborator
            .components()
            .find_map(|(candidate, definition)| (candidate == key).then_some(definition))
            .expect("selected checked component");
        let context_count = contexts.values().map(Vec::len).sum();
        Selected {
            elaborator,
            checked,
            cache: Vec::new(),
            indexed_subtrees: BTreeMap::new(),
            contexts: Some(&mut contexts),
            context_count,
            instances_visited: 0,
            memberships: CompleteExteriorMembershipBudget::new(
                elaborator.limits.complete_exteriors,
            ),
        }
        .component(
            component,
            values.clone(),
            elaborator.selected_supports.get(key).cloned().unwrap_or(
                SupportEnvironment::component(component.file, component.declaration)?,
            ),
            0,
        )?;
    }
    Ok(contexts)
}

pub(in crate::hierarchy) fn selected_component_summary(
    elaborator: &Elaborator<'_>,
    checked: &CheckedDefinitionGraph,
    component: &ComponentDefinition<'_>,
    values: SymbolicParameterMap,
    supports: &[crate::external::ExternalSupportBinding],
) -> Result<DefinitionSummary, Vec<Diagnostic>> {
    Selected {
        elaborator,
        checked,
        cache: Vec::new(),
        indexed_subtrees: BTreeMap::new(),
        contexts: None,
        context_count: 0,
        instances_visited: 0,
        memberships: CompleteExteriorMembershipBudget::new(elaborator.limits.complete_exteriors),
    }
    .component(
        component,
        values,
        SupportEnvironment::external(
            supports,
            &mut CompleteExteriorMembershipBudget::new(elaborator.limits.complete_exteriors),
        )?,
        0,
    )
}

fn model_values(
    elaborator: &Elaborator<'_>,
    model: &ModelDefinition<'_>,
) -> Result<SymbolicParameterMap, Vec<Diagnostic>> {
    let mut values = parameters::resolve_model_parameters_symbolically(
        model.file,
        model.declaration,
        |name| clocks::model(model.file, model.declaration, name),
        &parameters::RecordContext::model(elaborator, model),
    )?;
    elaborator.bind_symbolic_properties(
        &model.namespace,
        model.file,
        model.declaration.signature(),
        &mut values,
    )?;
    parameters::resolve_model_lets(model.file, model.declaration, &mut values, |name| {
        clocks::model(model.file, model.declaration, name)
    })?;
    Ok(values)
}

fn selected_expansion_size_with_contexts(
    elaborator: &Elaborator<'_>,
    checked: &CheckedDefinitionGraph,
    model: &ModelDefinition<'_>,
    values: SymbolicParameterMap,
    contexts: Option<&mut ComponentContexts>,
) -> Result<super::super::preflight::ExpansionSize, Vec<Diagnostic>> {
    let context_count = contexts
        .as_ref()
        .map_or(0, |contexts| contexts.values().map(Vec::len).sum());
    let mut preflight = Selected {
        elaborator,
        checked,
        cache: Vec::new(),
        indexed_subtrees: BTreeMap::new(),
        contexts,
        context_count,
        instances_visited: 0,
        memberships: CompleteExteriorMembershipBudget::new(elaborator.limits.complete_exteriors),
    };
    let key = DefinitionKey {
        namespace: model.namespace.clone(),
        name: model.name().to_owned(),
    };
    let mut supports = SupportEnvironment::model(model.file, model.declaration)?;
    if let Some(bound) = elaborator.selected_supports.get(&key) {
        supports.overlay(bound);
    }
    let mut diagnostics = Vec::new();
    let symbolic = preflight.contexts.is_some()
        && unresolved_extents(
            model.file,
            model.owned_items().filter_map(|item| match item {
                Item::IndexSet(set) => Some(set),
                _ => None,
            }),
            &values,
        )?;
    let mut local = model_local_footprint(
        elaborator,
        model,
        &mut diagnostics,
        if symbolic { None } else { Some(&values) },
    );
    if preflight.contexts.is_none() {
        local.instantiate(&supports.counts()?)?;
    }
    local.fixed.declarations = local
        .fixed
        .declarations
        .checked_add(
            elaborator
                .finite_spaces
                .values()
                .map(BTreeMap::len)
                .sum::<usize>(),
        )
        .ok_or_else(|| vec![definition_error("finite-space footprint overflows")])?;
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }
    let sets = model
        .owned_items()
        .filter_map(|item| match item {
            Item::IndexSet(value) => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    let instances = model.owned_items().filter_map(|item| match item {
        Item::Instance(value) => Some(value),
        _ => None,
    });
    let ChildFootprint {
        edges,
        summaries,
        extra_connections,
    } = preflight.children(
        &model.namespace,
        model.file,
        instances,
        &sets,
        (
            &values,
            &parameters::RecordContext::model(elaborator, model),
            &supports,
        ),
        0,
    )?;
    local.fixed.connections = local
        .fixed
        .connections
        .checked_add(extra_connections)
        .ok_or_else(|| vec![definition_error("connection footprint overflows")])?;
    let node = ModelNode {
        key: DefinitionKey {
            namespace: model.namespace.clone(),
            name: model.name().to_owned(),
        },
        file: model.file,
        range: model.range(),
        local,
        local_connectors: BTreeSet::new(),
        edges,
    };
    let summary = summarize_model(
        &node,
        &summaries,
        elaborator.limits.into(),
        &mut ReachabilityMemberships::new(elaborator.limits.max_definition_reachability_pairs),
    )?;
    append_limit_diagnostics(
        &mut diagnostics,
        &node.key,
        node.file,
        node.range,
        &summary,
        false,
        elaborator.limits,
    );
    if let Some(external) = elaborator.selected_supports.get(&key) {
        // Reserve external support entries before RootExpansion stages its root.
        for (resource, count, limit) in [
            (
                "staged identities",
                summary.staged_identities(),
                elaborator.limits.identity.max_staged_identities,
            ),
            (
                "provenance entries",
                summary.provenance_entries(),
                elaborator.limits.provenance.max_entries,
            ),
        ] {
            super::super::checked_external_footprint(
                resource,
                count,
                external.singular.len(),
                limit,
            )?;
        }
    }
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }
    Ok(super::super::preflight::ExpansionSize {
        declarations: summary.declarations(),
        connections: summary.connections(),
    })
}

struct Selected<'a, 'd, 'c> {
    elaborator: &'a Elaborator<'d>,
    checked: &'a CheckedDefinitionGraph,
    cache: Vec<(DefinitionKey, Values, SupportEnvironment, DefinitionSummary)>,
    indexed_subtrees: BTreeMap<DefinitionKey, (bool, bool)>,
    contexts: Option<&'c mut ComponentContexts>,
    context_count: usize,
    instances_visited: usize,
    memberships: CompleteExteriorMembershipBudget,
}
impl Selected<'_, '_, '_> {
    fn structural_profile(
        &mut self,
        component: &ComponentDefinition<'_>,
    ) -> Result<(bool, bool), Vec<Diagnostic>> {
        let key = DefinitionKey {
            namespace: component.namespace.clone(),
            name: component.name().to_owned(),
        };
        if let Some(found) = self.indexed_subtrees.get(&key) {
            return Ok(*found);
        }
        let mut found = if component.signature().iter().any(|item| matches!(item, eqiora_lang::SignatureItem::Support(slot) if matches!(slot.syntax(), eqiora_lang::SupportSlotSyntax::CompleteExterior { .. }))) { (true, true) } else { (false, false) };
        if component.signature().iter().any(|item| match item {
            eqiora_lang::SignatureItem::Field(value) => {
                parameters::extent_expressions(value.value_type())
                    .iter()
                    .any(|extent| crate::hierarchy::closed_index(extent).is_err())
            }
            _ => false,
        }) {
            found = (true, true);
        }
        for item in component.owned_items() {
            let syntax = match item {
                ComponentItem::Parameter(value) => Some(value.value_type()),
                ComponentItem::Observable(value) => Some(value.value_type()),
                ComponentItem::Field(value) => Some(value.value_type()),
                ComponentItem::Port(value) => match value.syntax() {
                    PortSyntax::Signal { value_type, .. } => Some(value_type),
                    _ => None,
                },
                _ => None,
            };
            if syntax.is_some_and(|syntax| {
                parameters::extent_expressions(syntax)
                    .iter()
                    .any(|extent| crate::hierarchy::closed_index(extent).is_err())
            }) {
                found = (true, true);
            }
            match item {
                ComponentItem::IndexSet(set) => {
                    found.0 = true;
                    let closed = match set.value().kind() {
                        ExprKind::Call {
                            callee,
                            arguments: eqiora_lang::CallArguments::Positional(arguments),
                        } if callee.as_str() == "range" => match arguments.as_slice() {
                            [extent] => matches!(
                                parameters::structural_extent(
                                    component.file,
                                    extent,
                                    &SymbolicParameterMap::new(),
                                ),
                                Ok(Some(_))
                            ),
                            _ => false,
                        },
                        _ => false,
                    };
                    found.1 |= !closed;
                }
                ComponentItem::Instance(instance) => {
                    let child = self
                        .elaborator
                        .resolve_component(
                            &component.namespace,
                            instance.definition(),
                            component.file,
                            instance.range(),
                        )
                        .map_err(|e| vec![e])?;
                    let child_profile = self.structural_profile(&child)?;
                    found.0 |= instance.family().is_some() || child_profile.0;
                    found.1 |= instance.family().is_some() || child_profile.1;
                }
                _ => {}
            }
        }
        self.indexed_subtrees.insert(key, found);
        Ok(found)
    }
    fn children<'i>(
        &mut self,
        namespace: &super::super::preflight::DefinitionNamespace,
        file: &'i str,
        instances: impl IntoIterator<Item = &'i InstanceDecl>,
        sets: &[&NamedDefinitionDecl],
        (values, records, supports): (
            &SymbolicParameterMap,
            &parameters::RecordContext,
            &SupportEnvironment,
        ),
        depth: usize,
    ) -> Result<ChildFootprint<'i>, Vec<Diagnostic>> {
        let mut edges = Vec::new();
        let mut summaries = Vec::new();
        let mut extra_connections = 0usize;
        for instance in instances {
            let child = self
                .elaborator
                .resolve_component(namespace, instance.definition(), file, instance.range())
                .map_err(|e| vec![e])?;
            let key = DefinitionKey {
                namespace: child.namespace.clone(),
                name: child.name().to_owned(),
            };
            let extent = if let Some(family) = instance.family() {
                let set = sets
                    .iter()
                    .find(|set| set.name() == family.set().as_str())
                    .ok_or_else(|| {
                        vec![source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            family.range(),
                            "indexed instance requires an enclosing IndexSet",
                        )]
                    })?;
                match parameters::index_set_extent(file, set, values).map_err(|e| vec![e])? {
                    Some((extent, _)) => extent as usize,
                    None if self.contexts.is_some() => {
                        edges.push(Edge {
                            target: summaries.len(),
                            multiplicity: 1,
                            occurrence: instance.name().to_owned(),
                            file,
                            range: instance.range(),
                        });
                        summaries.push(Some(
                            self.checked
                                .component_summary(&key)
                                .expect("checked definition")
                                .clone(),
                        ));
                        continue;
                    }
                    None => {
                        return Err(vec![source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            family.range(),
                            "selected indexed extent remains unresolved",
                        )]);
                    }
                }
            } else {
                1
            };
            // Every member has at least one occurrence; reject aggregate work before
            // constructing substituted bindings or walking parameter-dependent children.
            if extent
                > self
                    .elaborator
                    .limits
                    .max_instances
                    .saturating_sub(edges.len())
            {
                return Err(vec![source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    file,
                    instance.range(),
                    "indexed footprint exceeds the Component instances limit",
                )]);
            }
            self.instances_visited = self
                .instances_visited
                .checked_add(extent)
                .filter(|count| *count <= self.elaborator.limits.max_instances)
                .ok_or_else(|| {
                    vec![source_error(
                        codes::LANGUAGE_LOWERING_ERROR,
                        file,
                        instance.range(),
                        "static specialization exceeds the Component instances work limit",
                    )]
                })?;
            let profile = self.structural_profile(&child)?;
            if extent > 1 {
                let mut errors = Vec::new();
                let inputs = footprint::input_binding_count(
                    self.elaborator,
                    namespace,
                    file,
                    instance,
                    &mut errors,
                );
                if !errors.is_empty() {
                    return Err(errors);
                }
                extra_connections = inputs
                    .checked_mul(extent - 1)
                    .and_then(|count| extra_connections.checked_add(count))
                    .filter(|count| *count <= self.elaborator.limits.max_connections)
                    .ok_or_else(|| {
                        vec![definition_error(
                            "indexed Input connection footprint exceeds limit",
                        )]
                    })?;
            }
            for ordinal in 0..extent {
                let summary = if !profile.0 && self.contexts.is_none() {
                    self.checked
                        .component_summary(&key)
                        .expect("checked definition")
                        .clone()
                } else {
                    let member = super::super::reductions::instantiate_instance(
                        file,
                        instance,
                        ordinal as u32,
                    )
                    .map_err(|error| vec![error])?;
                    let mut child_values = parameters::resolve_instance_parameters_symbolically(
                        (child.file, file),
                        child.declaration,
                        &member,
                        values,
                        &mut |name| clocks::component(child.file, child.declaration, name),
                        &mut |_| None,
                        (
                            &parameters::RecordContext::component(self.elaborator, &child),
                            records,
                        ),
                    )?;
                    self.elaborator.bind_instance_property_values(
                        &child,
                        &member,
                        (namespace, file),
                        values,
                        &mut child_values,
                    )?;
                    let child_supports = supports.bind(
                        file,
                        child.declaration,
                        &member,
                        self.contexts.is_some(),
                        &mut self.memberships,
                    )?;
                    self.component(&child, child_values, child_supports, depth + 1)?
                };
                edges.push(Edge {
                    target: summaries.len(),
                    multiplicity: 1,
                    occurrence: if instance.family().is_some() {
                        format!("{}[{ordinal}]", instance.name())
                    } else {
                        instance.name().to_owned()
                    },
                    file,
                    range: instance.range(),
                });
                summaries.push(Some(summary));
            }
        }
        Ok(ChildFootprint {
            edges,
            summaries,
            extra_connections,
        })
    }
    fn component(
        &mut self,
        component: &ComponentDefinition<'_>,
        mut values: SymbolicParameterMap,
        supports: SupportEnvironment,
        depth: usize,
    ) -> Result<DefinitionSummary, Vec<Diagnostic>> {
        if depth > self.elaborator.limits.max_instance_depth {
            return Err(vec![definition_error(
                "selected instance depth exceeds limit",
            )]);
        }
        let key = DefinitionKey {
            namespace: component.namespace.clone(),
            name: component.name().to_owned(),
        };
        let cache_values = values
            .iter()
            .map(|(name, value)| (name.clone(), value.value.clone()))
            .collect::<Values>();
        if let Some((_, _, _, summary)) =
            self.cache
                .iter()
                .find(|(definition, inputs, environment, _)| {
                    definition == &key && inputs == &cache_values && environment == &supports
                })
        {
            return Ok(summary.clone());
        }
        self.elaborator.bind_symbolic_properties(
            &component.namespace,
            component.file,
            component.declaration.signature(),
            &mut values,
        )?;
        parameters::resolve_component_lets(
            component.file,
            component.declaration,
            &mut values,
            |name| clocks::component(component.file, component.declaration, name),
        )?;
        if self.contexts.is_some()
            && unresolved_extents(
                component.file,
                component.owned_items().filter_map(|item| match item {
                    ComponentItem::IndexSet(set) => Some(set),
                    _ => None,
                }),
                &values,
            )?
        {
            // Keep the unresolved definition on the ordinary symbolic checking path.
            return Ok(self
                .checked
                .component_summary(&key)
                .expect("validated definition graph")
                .clone());
        }
        if let Some(contexts) = self.contexts.as_mut() {
            if self.context_count >= self.elaborator.limits.max_definition_reachability_pairs {
                return Err(vec![definition_error(
                    "concrete Component contexts exceed the definition reachability limit",
                )]);
            }
            self.context_count += 1;
            contexts
                .entry(key.clone())
                .or_default()
                .push(values.clone());
        }
        let mut diagnostics = Vec::new();
        let (mut local, local_connectors) =
            component_local_footprint(self.elaborator, component, &mut diagnostics, Some(&values));
        if !diagnostics.is_empty() {
            return Err(diagnostics);
        }
        let sets = component
            .owned_items()
            .filter_map(|item| match item {
                ComponentItem::IndexSet(value) => Some(value),
                _ => None,
            })
            .collect::<Vec<_>>();
        let instances = component.owned_items().filter_map(|item| match item {
            ComponentItem::Instance(value) => Some(value),
            _ => None,
        });
        let ChildFootprint {
            edges,
            summaries,
            extra_connections,
        } = self.children(
            &component.namespace,
            component.file,
            instances,
            &sets,
            (
                &values,
                &parameters::RecordContext::component(self.elaborator, component),
                &supports,
            ),
            depth,
        )?;
        if self.contexts.is_none() {
            local.instantiate(&supports.counts()?)?;
        }
        local.fixed.connections = local
            .fixed
            .connections
            .checked_add(extra_connections)
            .ok_or_else(|| vec![definition_error("connection footprint overflows")])?;
        let node = ComponentNode {
            key: key.clone(),
            file: component.file,
            range: component.range(),
            local,
            local_connectors,
            edges,
        };
        let summary = summarize_component(
            &node,
            &summaries,
            self.elaborator.limits.into(),
            &mut ReachabilityMemberships::new(
                self.elaborator.limits.max_definition_reachability_pairs,
            ),
        )?;
        append_limit_diagnostics(
            &mut diagnostics,
            &key,
            node.file,
            node.range,
            &summary,
            false,
            self.elaborator.limits,
        );
        if !diagnostics.is_empty() {
            return Err(diagnostics);
        }
        self.cache
            .push((key, cache_values, supports, summary.clone()));
        Ok(summary)
    }
}

fn unresolved_extents<'a>(
    file: &str,
    sets: impl IntoIterator<Item = &'a NamedDefinitionDecl>,
    values: &SymbolicParameterMap,
) -> Result<bool, Vec<Diagnostic>> {
    for set in sets {
        let ExprKind::Call {
            arguments: eqiora_lang::CallArguments::Positional(arguments),
            ..
        } = set.value().kind()
        else {
            return Err(vec![definition_error("index set requires range(extent)")]);
        };
        let [extent] = arguments.as_slice() else {
            return Err(vec![definition_error("range requires one extent")]);
        };
        if parameters::structural_extent(file, extent, values)
            .map_err(|error| vec![error])?
            .is_none()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn size(
        source: &str,
        limits: HierarchyLimits,
    ) -> Result<super::super::super::preflight::ExpansionSize, Vec<Diagnostic>> {
        let document = eqiora_lang::parse("indexed-budget.eqi", source)
            .into_compilation_document()
            .unwrap();
        let identity =
            crate::source_identity::LocalSourceIdentity::from_document(&document).unwrap();
        let elaborator = Elaborator::new(
            "indexed-budget.eqi",
            source.len(),
            &document,
            identity,
            limits,
        )?;
        let checked = validate(&elaborator)?;
        let model = elaborator.find_entry_model("M").unwrap().unwrap();
        selected_expansion_size(&elaborator, &checked, &model)
    }
    #[test]
    fn actual_extent_multiplies_the_existing_child_footprint() {
        let source = "component Cell(parameter value:1) {} model M(parameter n:integer=3) { indexset Stages=range(n); instance cell[i in Stages]:Cell(value=to_real(ordinal(i))); }";
        let admitted = size(source, HierarchyLimits::default()).unwrap();
        assert_eq!(admitted.declarations, 5); // n + IndexSet + three Cell Parameters.
        assert_eq!(admitted.connections, 0);
        for limits in [
            HierarchyLimits {
                max_instances: 2,
                ..Default::default()
            },
            HierarchyLimits {
                max_declarations: 4,
                ..Default::default()
            },
            HierarchyLimits {
                identity: crate::identity::ElaborationIdentityLimits {
                    max_staged_identities: 5,
                    ..Default::default()
                },
                ..Default::default()
            },
            HierarchyLimits {
                provenance: crate::provenance::ProvenanceLimits {
                    max_entries: 5,
                    ..Default::default()
                },
                ..Default::default()
            },
        ] {
            assert!(size(source, limits).is_err());
        }
    }
    #[test]
    fn sibling_products_are_added_before_occurrence_allocation() {
        let source = "component Cell(parameter value:integer) {} model M() { indexset S=range(3); instance a[i in S]:Cell(value=ordinal(i)); instance b[j in S]:Cell(value=ordinal(j)); }";
        let errors = size(
            source,
            HierarchyLimits {
                max_instances: 4,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("Component instances")),
            "{errors:?}"
        );
    }
    #[test]
    fn closed_child_index_sets_have_a_fixed_per_member_footprint() {
        let source = "component Cell(parameter value:integer) { indexset Local=range(2); } model M() { indexset S=range(3); instance cell[i in S]:Cell(value=ordinal(i)); }";
        let admitted = size(source, HierarchyLimits::default()).unwrap();
        // Root IndexSet plus three copies of a Parameter and a local IndexSet.
        assert_eq!(admitted.declarations, 7);
        assert!(
            size(
                source,
                HierarchyLimits {
                    max_declarations: 6,
                    ..Default::default()
                }
            )
            .is_err()
        );
        let dependent = "component Cell(parameter value:integer) { indexset Local=range(value); } model M() { indexset S=range(3); instance cell[i in S]:Cell(value=ordinal(i)+1); }";
        assert_eq!(
            size(dependent, HierarchyLimits::default())
                .unwrap()
                .declarations,
            7
        );
    }
    #[test]
    fn nested_structural_profile_counts_every_actual_ordinal() {
        let source = "component Cell(parameter value:integer) {} component Row(parameter n:integer) { indexset Columns=range(n); instance cell[j in Columns]:Cell(value=ordinal(j)); } model M() { indexset Rows=range(2); instance row[i in Rows]:Row(n=ordinal(i)+1); }";
        // One root IndexSet, two Row Parameters and IndexSets, and 1 + 2 Cells.
        assert_eq!(
            size(source, HierarchyLimits::default())
                .unwrap()
                .declarations,
            8
        );
        assert!(
            size(
                source,
                HierarchyLimits {
                    max_declarations: 7,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    #[test]
    fn expression_forwarding_keeps_instance_expansion_limits() {
        let source = "component Leaf(support body:volume(ambient_dimension=2),expression source:K/m^2 on body){} component Outer(support body:volume(ambient_dimension=2),expression source:K/m^2 on body){instance leaf:Leaf(body=body,source=source);} model M(){domain body=box(0,1,0,1);instance outer:Outer(body=body,source=1[K/m^2]);}";
        assert!(size(source, HierarchyLimits::default()).is_ok());
        let errors = size(
            source,
            HierarchyLimits {
                max_instances: 1,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("Component instances")),
            "{errors:?}"
        );
    }
}
