use std::collections::BTreeSet;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{ComponentItem, SignatureItem, SupportSlotSyntax};
use eqiora_schema::kernel::typing::{ExpressionType, SpatialSupport};

use crate::diagnostics::source_error;

use super::expression::{validate_relation_expression, validate_relation_family_expression};
use super::scope::{
    DefinitionScope, SymbolContract, component_port_contract, component_port_family_contract,
    field_expression_type, validate_boundary_connection,
};
use super::{
    ChildInstanceProof, DefinitionBodyProof, LocalPhysicalPortProof, SymbolicParameterMap,
    validate_clock,
};
use crate::hierarchy::field_slots::FieldInterface;
use crate::hierarchy::preflight::{ComponentDefinition, DefinitionKey, Elaborator};
use crate::hierarchy::supports::SupportInterface;

pub(super) fn validate(
    elaborator: &Elaborator<'_>,
    definition: &ComponentDefinition<'_>,
    compile_time_values: &SymbolicParameterMap,
    supports: &SupportInterface,
    fields: &FieldInterface,
) -> Result<DefinitionBodyProof, Vec<Diagnostic>> {
    let mut checker = ComponentBodyChecker::new(
        elaborator,
        definition,
        compile_time_values,
        supports,
        fields,
    );
    checker.validate();
    if checker.diagnostics.is_empty() {
        Ok(checker.proof)
    } else {
        Err(checker.diagnostics)
    }
}

// Editor preparation retains only the declaration symbols. It never binds
// child instances or validates/expands an occurrence graph.
pub(super) fn declaration_symbols(
    elaborator: &Elaborator<'_>,
    definition: &ComponentDefinition<'_>,
    compile_time_values: &SymbolicParameterMap,
    supports: &SupportInterface,
    fields: &FieldInterface,
) -> Result<std::collections::BTreeMap<String, SymbolContract>, Vec<Diagnostic>> {
    let mut checker = ComponentBodyChecker::new(
        elaborator,
        definition,
        compile_time_values,
        supports,
        fields,
    );
    checker.bind_declaration_scope();
    if checker.diagnostics.is_empty() {
        Ok(checker.scope.symbols)
    } else {
        Err(checker.diagnostics)
    }
}

struct ComponentBodyChecker<'e, 'd> {
    definition: &'e ComponentDefinition<'d>,
    compile_time_values: &'e SymbolicParameterMap,
    supports: &'e SupportInterface,
    fields: &'e FieldInterface,
    scope: DefinitionScope<'e, 'd>,
    connected_ports: BTreeSet<Vec<String>>,
    proof: DefinitionBodyProof,
    diagnostics: Vec<Diagnostic>,
}

impl<'e, 'd> ComponentBodyChecker<'e, 'd> {
    fn new(
        elaborator: &'e Elaborator<'d>,
        definition: &'e ComponentDefinition<'d>,
        compile_time_values: &'e SymbolicParameterMap,
        supports: &'e SupportInterface,
        fields: &'e FieldInterface,
    ) -> Self {
        Self {
            definition,
            compile_time_values,
            supports,
            fields,
            scope: DefinitionScope::new(elaborator, definition.namespace.clone(), definition.file),
            connected_ports: BTreeSet::new(),
            proof: DefinitionBodyProof::new(
                definition.file,
                definition.declaration.range(),
                elaborator.limits.connection_sets,
            ),
            diagnostics: Vec::new(),
        }
    }

    fn bind_declaration_scope(&mut self) {
        self.diagnostics.extend(super::property::bind(
            &mut self.scope,
            self.definition.declaration.signature(),
        ));
        self.scope.static_values = self.compile_time_values.clone();
        self.scope.record_context = super::super::parameters::RecordContext::component(
            self.scope.elaborator,
            self.definition,
        );
        for (root, record) in self.scope.record_context.records() {
            for (member, _) in record.definition.members() {
                let name = format!("{root}.{member}");
                if let Some(value) = self.compile_time_values.get(&name) {
                    self.scope.symbols.insert(
                        name,
                        SymbolContract::Parameter(ExpressionType::new(
                            value.value_type.clone(),
                            None,
                        )),
                    );
                }
            }
        }
        for item in self.definition.declaration.items() {
            if let ComponentItem::IndexSet(declaration) = item
                && let Err(error) = self.scope.bind_index_set(declaration)
            {
                self.diagnostics.push(error);
            }
        }
        self.bind_borrowed_interfaces();
        self.bind_complete_exteriors();
        self.bind_owned_interfaces();
    }

    fn validate(&mut self) {
        self.bind_declaration_scope();
        self.bind_child_instances();
        if let Err(errors) = super::expression::validate_aliases(
            &mut self.scope,
            self.definition
                .declaration
                .items()
                .iter()
                .filter_map(|item| match item {
                    ComponentItem::Let(d) => Some(d),
                    _ => None,
                }),
            self.compile_time_values,
        ) {
            self.diagnostics.extend(errors);
        }
        self.validate_declarations();
    }

    fn bind_complete_exteriors(&mut self) {
        for item in self.definition.declaration.signature() {
            let SignatureItem::Support(declaration) = item else {
                continue;
            };
            let SupportSlotSyntax::CompleteExterior { parent } = declaration.syntax() else {
                continue;
            };
            let Some(parent) = self.supports.get(parent) else {
                self.diagnostics.push(self.scope.wrong_local_kind(
                    declaration.range(),
                    parent,
                    "complete-exterior volume parent",
                ));
                continue;
            };
            let SpatialSupport::Volume { .. } = parent.support() else {
                self.diagnostics.push(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.definition.file,
                    declaration.range(),
                    "complete-exterior support requires a volume parent",
                ));
                continue;
            };
            self.scope.symbols.insert(
                declaration.name().to_owned(),
                SymbolContract::CompleteExterior {
                    parent: parent.support().clone(),
                },
            );
        }
    }

    fn bind_borrowed_interfaces(&mut self) {
        for item in self.definition.declaration.signature() {
            match item {
                SignatureItem::Support(declaration) => {
                    if matches!(
                        declaration.syntax(),
                        SupportSlotSyntax::CompleteExterior { .. }
                    ) {
                        continue;
                    }
                    let Some(contract) = self.supports.get(declaration.name()) else {
                        self.diagnostics.push(source_error(
                            codes::LANGUAGE_LOWERING_ERROR,
                            self.definition.file,
                            declaration.range(),
                            format!(
                                "typed support interface is missing slot `{}`",
                                declaration.name()
                            ),
                        ));
                        continue;
                    };
                    self.scope.symbols.insert(
                        declaration.name().to_owned(),
                        SymbolContract::Support(contract.support().clone()),
                    );
                }
                SignatureItem::Input(value) | SignatureItem::Output(value) => {
                    self.scope.exposed_signals.insert(value.name().to_owned());
                }
                SignatureItem::Clock(declaration) => {
                    self.scope
                        .borrowed_clocks
                        .insert(declaration.name().to_owned());
                    self.scope
                        .symbols
                        .insert(declaration.name().to_owned(), SymbolContract::Clock);
                }
                SignatureItem::Field(declaration) => {
                    let Some(contract) = self.fields.field(declaration.name()) else {
                        self.diagnostics.push(source_error(
                            codes::LANGUAGE_LOWERING_ERROR,
                            self.definition.file,
                            declaration.range(),
                            format!(
                                "typed Field interface is missing slot `{}`",
                                declaration.name()
                            ),
                        ));
                        continue;
                    };
                    self.scope.symbols.insert(
                        declaration.name().to_owned(),
                        SymbolContract::Field(
                            contract.value().clone(),
                            declaration.role(),
                            declaration.activation().clone(),
                        ),
                    );
                }
                _ => {}
            }
        }
    }

    fn bind_owned_interfaces(&mut self) {
        for item in self.definition.owned_items() {
            match item {
                ComponentItem::Let(declaration) => {
                    if let Some(value) = self.compile_time_values.get(declaration.name()) {
                        self.scope.symbols.insert(
                            declaration.name().to_owned(),
                            SymbolContract::Parameter(ExpressionType::new(
                                value.value_type.clone(),
                                None,
                            )),
                        );
                    }
                }
                ComponentItem::Parameter(declaration) => {
                    if self
                        .scope
                        .record_context
                        .record_for_type(declaration.value_type())
                        .is_some()
                    {
                        continue;
                    }
                    let Some(parameter) = self.compile_time_values.get(declaration.name()) else {
                        self.diagnostics.push(source_error(
                            codes::LANGUAGE_LOWERING_ERROR,
                            self.definition.file,
                            declaration.range(),
                            format!(
                                "symbolic interface is missing component Parameter `{}`",
                                declaration.name()
                            ),
                        ));
                        continue;
                    };
                    self.scope.symbols.insert(
                        declaration.name().to_owned(),
                        SymbolContract::Parameter(ExpressionType::new(
                            parameter.value_type.clone(),
                            None,
                        )),
                    );
                }
                ComponentItem::Port(declaration) => match component_port_contract(
                    self.scope.elaborator,
                    self.definition,
                    declaration,
                    self.compile_time_values,
                ) {
                    Ok(contract) => {
                        if contract.is_physical() {
                            self.proof.local_physical_ports.insert(
                                declaration.name().to_owned(),
                                LocalPhysicalPortProof {
                                    public: declaration.visibility()
                                        == eqiora_lang::VisibilitySyntax::Public,
                                    range: declaration.range(),
                                },
                            );
                        }
                        self.scope.symbols.insert(
                            declaration.name().to_owned(),
                            SymbolContract::Port(contract),
                        );
                    }
                    Err(mut errors) => self.diagnostics.append(&mut errors),
                },
                ComponentItem::PortFamily(declaration) => {
                    let active = match self.scope.boundary_family_scope(declaration.binder()) {
                        Ok(active) => active,
                        Err(error) => {
                            self.diagnostics.push(error);
                            continue;
                        }
                    };
                    match component_port_family_contract(
                        self.scope.elaborator,
                        self.definition,
                        declaration,
                        active.support(),
                    ) {
                        Ok(contract) => {
                            self.scope.symbols.insert(
                                declaration.port().name().to_owned(),
                                SymbolContract::PortFamily(contract),
                            );
                        }
                        Err(mut errors) => self.diagnostics.append(&mut errors),
                    }
                }
                ComponentItem::Field(declaration) => {
                    let support = declaration
                        .domain()
                        .and_then(|domain| self.scope.spatial_support(domain));
                    if let Some(record) = self
                        .scope
                        .elaborator
                        .record_for_type(&self.scope.namespace, declaration.value_type())
                    {
                        for (name, value_type) in record.definition.members() {
                            self.scope.symbols.insert(
                                format!("{}.{name}", declaration.name()),
                                SymbolContract::Field(
                                    eqiora_schema::kernel::typing::ExpressionType::new(
                                        value_type.clone(),
                                        support.clone(),
                                    ),
                                    declaration.role(),
                                    declaration.activation().clone(),
                                ),
                            );
                        }
                        continue;
                    }
                    match field_expression_type(
                        self.definition.file,
                        declaration,
                        support,
                        self.compile_time_values,
                    ) {
                        Ok(inferred) => {
                            self.scope.symbols.insert(
                                declaration.name().to_owned(),
                                SymbolContract::Field(
                                    inferred,
                                    declaration.role(),
                                    declaration.activation().clone(),
                                ),
                            );
                        }
                        Err(error) => self.diagnostics.push(error),
                    }
                }
                ComponentItem::Observable(declaration) => {
                    self.scope
                        .symbols
                        .insert(declaration.name().to_owned(), SymbolContract::Observable);
                }
                ComponentItem::Event(declaration) => {
                    self.scope
                        .symbols
                        .insert(declaration.name().to_owned(), SymbolContract::Event);
                }
                ComponentItem::Clock(declaration) => {
                    self.scope
                        .symbols
                        .insert(declaration.name().to_owned(), SymbolContract::Clock);
                }
                ComponentItem::Relation(declaration) => {
                    self.scope
                        .symbols
                        .insert(declaration.name().to_owned(), SymbolContract::Relation);
                }
                ComponentItem::RelationFamily(declaration) => {
                    self.scope.symbols.insert(
                        declaration.relation().name().to_owned(),
                        SymbolContract::Relation,
                    );
                }
                ComponentItem::IndexSet(_) | ComponentItem::Instance(_) => {}
                ComponentItem::Initial(_)
                | ComponentItem::Connection(_)
                | ComponentItem::BoundaryConnection(_) => {}
                _ => self.diagnostics.push(source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    self.definition.file,
                    self.definition.declaration.range(),
                    "component item is newer than definition-body validation",
                )),
            }
        }
    }

    fn bind_child_instances(&mut self) {
        for item in self.definition.declaration.items() {
            if let ComponentItem::Instance(instance) = item
                && let Ok(child) = self.scope.elaborator.resolve_component(
                    &self.definition.namespace,
                    instance.definition(),
                    self.definition.file,
                    instance.range(),
                )
            {
                let extent = instance.family().and_then(|family| {
                    self.scope
                        .index_sets
                        .get(family.set().as_str())
                        .copied()
                        .flatten()
                });
                let count = extent.map_or(1, |value| value as usize);
                if count
                    > self
                        .scope
                        .elaborator
                        .limits
                        .max_instances
                        .saturating_sub(self.proof.children.len())
                {
                    self.diagnostics.push(source_error(
                        codes::LANGUAGE_LOWERING_ERROR,
                        self.scope.file,
                        instance.range(),
                        "indexed child proof exceeds the instance expansion limit",
                    ));
                    continue;
                }
                let names = (0..count).map(|ordinal| {
                    if extent.is_some() {
                        format!("{}[{ordinal}]", instance.name())
                    } else {
                        instance.name().to_owned()
                    }
                });
                for name in names {
                    self.proof.children.insert(
                        name,
                        ChildInstanceProof {
                            definition: DefinitionKey {
                                namespace: child.namespace.clone(),
                                name: child.declaration.name().to_owned(),
                            },
                            range: instance.range(),
                        },
                    );
                }
                self.scope
                    .children
                    .insert(instance.name().to_owned(), child);
                self.scope
                    .child_instances
                    .insert(instance.name().to_owned(), instance);
            }
        }
    }

    fn validate_declarations(&mut self) {
        for item in self.definition.owned_items() {
            match item {
                ComponentItem::Initial(declaration) => {
                    if let Err(errors) =
                        super::expression::validate_initial_expression(&self.scope, declaration)
                    {
                        self.diagnostics.extend(errors);
                    }
                }
                ComponentItem::IndexSet(_)
                | ComponentItem::Let(_)
                | ComponentItem::Parameter(_)
                | ComponentItem::Port(_)
                | ComponentItem::PortFamily(_) => {}
                ComponentItem::Field(declaration) => {
                    if let eqiora_lang::ActivationSyntax::Named(clock) = declaration.activation()
                        && !matches!(self.scope.symbols.get(clock), Some(SymbolContract::Clock))
                    {
                        self.diagnostics.push(self.scope.wrong_local_kind(
                            declaration.range(),
                            clock,
                            "Field ClockDomain",
                        ));
                    }
                    if let Some(domain) = declaration.domain()
                        && self.scope.spatial_support(domain).is_none()
                    {
                        self.diagnostics.push(self.scope.wrong_local_kind(
                            declaration.range(),
                            domain,
                            "Field support",
                        ));
                    }
                }
                ComponentItem::Observable(declaration) => {
                    if let Err(error) =
                        super::expression::validate_observable(&self.scope, declaration)
                    {
                        self.diagnostics.push(error);
                    }
                }
                ComponentItem::Event(declaration) => {
                    if let Err(error) =
                        super::expression::validate_event_guard(&self.scope, declaration.guard())
                    {
                        self.diagnostics.push(error);
                    }
                }
                ComponentItem::Clock(declaration) => {
                    if let Err(error) = validate_clock(
                        self.definition.file,
                        declaration.range(),
                        declaration.period(),
                        declaration.phase(),
                    ) {
                        self.diagnostics.push(error);
                    }
                }
                ComponentItem::Relation(declaration) => {
                    let support = declaration.domain().and_then(|domain| {
                        let support = self.scope.spatial_support(domain);
                        if support.is_none() {
                            self.diagnostics.push(self.scope.wrong_local_kind(
                                declaration.range(),
                                domain,
                                "Relation support",
                            ));
                        }
                        support
                    });
                    match validate_relation_expression(&self.scope, declaration, support) {
                        Ok(endpoints) => self.proof.relation_endpoints.push(endpoints),
                        Err(errors) => self.diagnostics.extend(errors),
                    }
                }
                ComponentItem::RelationFamily(declaration) => {
                    if self
                        .scope
                        .index_sets
                        .contains_key(declaration.binder().set().as_str())
                    {
                        let extent = match super::indexed::extent(&self.scope, declaration.binder())
                        {
                            Ok(extent) => extent,
                            Err(error) => {
                                self.diagnostics.push(error);
                                continue;
                            }
                        };
                        for ordinal in 0..extent {
                            match super::indexed::relation(self.scope.file, declaration, ordinal) {
                                Ok(relation) => {
                                    let support = relation
                                        .domain()
                                        .and_then(|name| self.scope.spatial_support(name));
                                    match validate_relation_expression(
                                        &self.scope,
                                        &relation,
                                        support,
                                    ) {
                                        Ok(endpoints) => {
                                            self.proof.relation_endpoints.push(endpoints)
                                        }
                                        Err(errors) => self.diagnostics.extend(errors),
                                    }
                                }
                                Err(error) => self.diagnostics.push(error),
                            }
                        }
                        continue;
                    }
                    let active = match self.scope.boundary_family_scope(declaration.binder()) {
                        Ok(active) => active,
                        Err(error) => {
                            self.diagnostics.push(error);
                            continue;
                        }
                    };
                    match validate_relation_family_expression(&self.scope, declaration, &active) {
                        Ok(_) => {}
                        Err(errors) => self.diagnostics.extend(errors),
                    }
                }
                ComponentItem::Connection(declaration) => {
                    match super::indexed::connections(
                        &self.scope,
                        declaration,
                        &mut self.connected_ports,
                        self.proof.connection_limits,
                    ) {
                        Ok(fragments) => self.proof.physical_connection_fragments.extend(fragments),
                        Err(error) => self.diagnostics.push(error),
                    }
                }
                ComponentItem::BoundaryConnection(declaration) => {
                    match validate_boundary_connection(&self.scope, declaration) {
                        Ok(Some(memberships)) if memberships.len() >= 2 => {
                            match crate::connection_sets::ConnectionFragment::try_new(
                                memberships,
                                self.proof.connection_limits,
                            ) {
                                Ok(fragment) => {
                                    self.proof.physical_connection_fragments.push(fragment);
                                }
                                Err(error) => self.diagnostics.push(source_error(
                                    codes::LANGUAGE_LOWERING_ERROR,
                                    self.scope.file,
                                    declaration.range(),
                                    format!(
                                        "cannot retain exact boundary Connection class: {error}"
                                    ),
                                )),
                            }
                        }
                        Ok(Some(memberships)) => self
                            .proof
                            .deferred_connection_memberships
                            .extend(memberships),
                        Ok(None) => {}
                        Err(error) => self.diagnostics.push(error),
                    }
                }
                ComponentItem::Instance(instance) => {
                    if let Err(error) = super::scope::validate_input_bindings(
                        &self.scope,
                        instance,
                        &mut self.connected_ports,
                        self.proof.connection_limits,
                    ) {
                        self.diagnostics.push(error);
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy::field_slots::component_field_interface;
    use crate::hierarchy::parameters::resolve_component_parameters_symbolically;
    use crate::hierarchy::supports::component_support_interface;
    use crate::source_identity::LocalSourceIdentity;

    fn validate_component(
        source: &str,
        name: &str,
    ) -> Result<DefinitionBodyProof, Vec<Diagnostic>> {
        let document = eqiora_lang::parse("family.eqi", source)
            .into_document()
            .expect("test source parses");
        let identity = LocalSourceIdentity::from_document(&document).expect("source identity");
        let elaborator = Elaborator::new(
            "family.eqi",
            source.len(),
            &document,
            identity,
            crate::hierarchy::HierarchyLimits::default(),
        )?;
        let definition = elaborator
            .components()
            .find_map(|(_, definition)| {
                (definition.declaration.name() == name).then(|| definition.clone())
            })
            .expect("selected Component exists");
        let compile_time_values = resolve_component_parameters_symbolically(
            definition.file,
            definition.declaration,
            |name| {
                crate::hierarchy::clocks::component(definition.file, definition.declaration, name)
            },
            &crate::hierarchy::parameters::RecordContext::component(&elaborator, &definition),
        )?;
        let supports = component_support_interface(definition.file, definition.declaration)?;
        let fields = component_field_interface(
            definition.file,
            definition.declaration,
            &supports,
            &compile_time_values,
        )?;
        validate(
            &elaborator,
            &definition,
            &compile_time_values,
            &supports,
            &fields,
        )
    }

    const SCALAR_CONNECTOR: &str = r#"
public connector BoundaryScalar {
  trace value: 1;
  flux flux: 1;
  shape [];
  frame invariant;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}
"#;

    #[test]
    fn complete_exterior_family_is_checked_once_with_a_synthetic_member_identity() {
        let source = format!(
            r#"{SCALAR_CONNECTOR}
component BoundaryLaw(support exterior: complete_exterior(parent = body), support body: volume(ambient_dimension = 2), port natural[boundary in exterior]: BoundaryScalar over boundary, port coupled[boundary in exterior]: BoundaryScalar over boundary) {{
  
  


  relation natural_law[boundary in exterior] on boundary {{
    natural[boundary = boundary].flux = 0;
  }}
  connect [boundary in exterior]
    natural[boundary = boundary], coupled[boundary = boundary];
}}
"#
        );
        let proof = validate_component(&source, "BoundaryLaw")
            .expect("closed family is valid before occurrence expansion");

        assert!(proof.relation_endpoints.is_empty());
        assert!(proof.physical_connection_fragments.is_empty());
        assert!(proof.local_physical_ports.is_empty());
    }

    #[test]
    fn binderless_exact_boundary_connection_retains_its_component_class() {
        let source = format!(
            r#"{SCALAR_CONNECTOR}
component Coupler(support left_body: volume(ambient_dimension = 2), support left_face: boundary(parent = left_body), support right_body: volume(ambient_dimension = 2), support right_face: boundary(parent = right_body), port left: BoundaryScalar over left_face, port right: BoundaryScalar over right_face) {{




  
  
  relation left_law on left_face {{ left.value = 0; left.flux = 0; }}
  relation right_law on right_face {{ right.value = 0; right.flux = 0; }}
  connect left, right;
}}
"#
        );
        let proof = validate_component(&source, "Coupler")
            .expect("binderless exact boundary Connection is typed");

        assert_eq!(proof.physical_connection_fragments.len(), 1);
        assert_eq!(proof.physical_connection_fragments[0].members().len(), 2);
        assert!(proof.deferred_connection_memberships.is_empty());
    }

    #[test]
    fn family_binders_and_selectors_fail_closed_outside_one_exact_scope() {
        let cases = [
            (
                "unknown exterior",
                "missing",
                "complete-exterior support set",
            ),
            (
                "wrong selector target",
                "exterior",
                "does not name active family member",
            ),
        ];
        for (case, relation_set, expected) in cases {
            let target = if case == "wrong selector target" {
                "other"
            } else {
                "boundary"
            };
            let source = format!(
                r#"{SCALAR_CONNECTOR}
component BoundaryLaw(support body: volume(ambient_dimension = 2), support exterior: complete_exterior(parent = body), port natural[boundary in exterior]: BoundaryScalar over boundary) {{


  
  relation law[boundary in {relation_set}] on boundary {{
    natural[boundary = {target}].flux = 0;
  }}
}}
"#
            );
            let diagnostics = validate_component(&source, "BoundaryLaw").expect_err(case);
            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message().contains(expected)),
                "{case}: {diagnostics:?}"
            );
        }
    }

    #[test]
    fn boundary_connection_requires_exactly_selected_matching_port_families() {
        let source = r#"
public connector A {
  trace value: 1;
  flux flux: 1;
  shape [];
  frame invariant;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}
public connector B {
  trace value: 1;
  flux flux: 1;
  shape [];
  frame invariant;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}
component InvalidConnection(support body: volume(ambient_dimension = 2), support exterior: complete_exterior(parent = body), port left[boundary in exterior]: A over boundary, port right[boundary in exterior]: B over boundary) {


  
  
  connect [boundary in exterior]
    left[boundary = boundary], right[boundary = boundary];
}
"#;
        let diagnostics =
            validate_component(source, "InvalidConnection").expect_err("nominal mismatch");
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic
                .message()
                .contains("exact same specialized Connector")
        }));
    }

    #[test]
    fn child_port_family_requires_explicit_complete_exterior_forwarding() {
        let prefix = format!(
            r#"{SCALAR_CONNECTOR}
component Leaf(support body: volume(ambient_dimension = 2), support exterior: complete_exterior(parent = body), port mechanical[side in exterior]: BoundaryScalar over side) {{


  
}}
"#
        );
        let parent = |forwarding: &str| {
            format!(
                r#"{prefix}
component Parent(support body: volume(ambient_dimension = 2), support exterior: complete_exterior(parent = body), port mechanical[boundary in exterior]: BoundaryScalar over boundary) {{


  
  instance child: Leaf(body = body{forwarding});
  connect [boundary in exterior]
    child.mechanical[side = boundary], mechanical[boundary = boundary];
}}
"#
            )
        };

        validate_component(&parent(", exterior = exterior"), "Parent")
            .expect("child family is mapped through one explicit set forwarding");
        let diagnostics = validate_component(&parent(""), "Parent")
            .expect_err("a child family cannot capture an unrelated active binder");
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic
                .message()
                .contains("is not forwarded from active complete exterior `exterior`")
        }));
    }
}
