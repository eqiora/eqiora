use std::collections::BTreeMap;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{ComponentItem, Item, NamePath, PortSyntax, TextRange, VisibilitySyntax};

use crate::diagnostics::source_error;

use super::{
    AnalyzedResolvedHierarchy, AnalyzedSourceUnit, CompilationModuleId, CompilationNamespaceId,
    ResolvedAlias, canonical_declaration_path,
};

impl AnalyzedResolvedHierarchy {
    /// Compiler-canonical declarations in `(namespace, path, kind)` order.
    #[must_use]
    pub fn canonical_declarations(&self) -> &[CanonicalDeclarationIdentity] {
        &self.canonical_declarations
    }

    /// Compiler-resolved declaration references in source-file and range order.
    pub fn resolved_references(
        &self,
    ) -> impl ExactSizeIterator<
        Item = (
            &CanonicalDeclarationIdentity,
            &str,
            TextRange,
            &str,
            TextRange,
        ),
    > {
        self.reference_locations
            .iter()
            .map(|(target, file, range)| {
                let (definition_file, definition_range) = &self.declaration_locations[*target];
                (
                    &self.canonical_declarations[*target],
                    file.as_str(),
                    *range,
                    definition_file.as_str(),
                    *definition_range,
                )
            })
    }
}

/// Top-level declaration families currently understood by package lowering.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum CanonicalDeclarationKind {
    Dimension,
    Enum,
    PropertyContract,
    PropertyRelease,
    MaterialComposition,
    PureOperator,
    Connector,
    Component,
    Model,
}

/// File-layout-independent canonical declaration emitted by the compiler.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalDeclarationIdentity {
    pub(super) namespace: CompilationNamespaceId,
    pub(super) path: String,
    pub(super) kind: CanonicalDeclarationKind,
    pub(super) visibility: VisibilitySyntax,
    pub(super) canonical_form: String,
}

impl CanonicalDeclarationIdentity {
    #[must_use]
    pub const fn namespace(&self) -> &CompilationNamespaceId {
        &self.namespace
    }
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }
    #[must_use]
    pub const fn kind(&self) -> CanonicalDeclarationKind {
        self.kind
    }
    #[must_use]
    pub const fn visibility(&self) -> VisibilitySyntax {
        self.visibility
    }
    #[must_use]
    pub fn canonical_form(&self) -> &str {
        &self.canonical_form
    }
}

pub(super) fn collect_declaration_locations(
    units: &[AnalyzedSourceUnit],
    declarations: &[CanonicalDeclarationIdentity],
) -> Vec<(String, TextRange)> {
    let mut locations = BTreeMap::new();
    for unit in units {
        let namespace = unit.module.owner();
        let mut insert = |name: &str, kind, range| {
            locations.insert(
                (
                    namespace.clone(),
                    canonical_declaration_path(&unit.module, name),
                    kind,
                ),
                (unit.file.clone(), range),
            );
        };
        for value in unit.document.dimensions() {
            insert(
                value.name(),
                CanonicalDeclarationKind::Dimension,
                value.range(),
            );
        }
        for (_, name, _, range) in unit.document.property_contract_syntax() {
            insert(name, CanonicalDeclarationKind::PropertyContract, range);
        }
        for (_, name, _, _, _, _, _, _, range) in unit.document.property_release_syntax() {
            insert(name, CanonicalDeclarationKind::PropertyRelease, range);
        }
        for (_, name, _, range) in unit.document.material_composition_syntax() {
            insert(name, CanonicalDeclarationKind::MaterialComposition, range);
        }
        for declaration in unit.document.enumerations() {
            insert(
                declaration.name(),
                CanonicalDeclarationKind::Enum,
                declaration.range(),
            );
        }
        for declaration in unit.document.connectors() {
            insert(
                declaration.name(),
                CanonicalDeclarationKind::Connector,
                declaration.range(),
            );
        }
        for declaration in unit.document.pure_operators() {
            insert(
                declaration.name(),
                CanonicalDeclarationKind::PureOperator,
                declaration.range(),
            );
        }
        for declaration in unit.document.components() {
            insert(
                declaration.name(),
                CanonicalDeclarationKind::Component,
                declaration.range(),
            );
        }
        for declaration in unit.document.models() {
            insert(
                declaration.name(),
                CanonicalDeclarationKind::Model,
                declaration.range(),
            );
        }
    }
    declarations
        .iter()
        .map(|declaration| {
            locations
                .remove(&(
                    declaration.namespace().clone(),
                    declaration.path().to_owned(),
                    declaration.kind(),
                ))
                .expect("every canonical declaration retains its source location")
        })
        .collect()
}

pub(super) fn collect_reference_locations(
    units: &[AnalyzedSourceUnit],
    aliases: &[ResolvedAlias],
    declarations: &[CanonicalDeclarationIdentity],
) -> Result<Vec<(usize, String, TextRange)>, Vec<Diagnostic>> {
    let declaration_index = declarations
        .iter()
        .enumerate()
        .map(|(index, declaration)| {
            (
                (
                    declaration.namespace().clone(),
                    declaration.path().to_owned(),
                    declaration.kind(),
                ),
                index,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let alias_index = aliases
        .iter()
        .map(|alias| {
            (
                (alias.declaring_module().clone(), alias.alias().to_owned()),
                alias.target_module().clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut references = Vec::new();
    let mut diagnostics = Vec::new();

    for unit in units {
        let resolve = |path: &NamePath, kind| {
            resolve_reference(&unit.module, path, kind, &alias_index, &declaration_index)
        };
        let instance_properties = |instance: &eqiora_lang::InstanceDecl| {
            let Some(index) = resolve(instance.definition(), CanonicalDeclarationKind::Component)
            else {
                return Vec::new();
            };
            let target = &declarations[index];
            units
                .iter()
                .flat_map(|unit| {
                    unit.document
                        .components()
                        .iter()
                        .map(move |component| (unit, component))
                })
                .find(|(unit, component)| {
                    unit.module.owner() == target.namespace()
                        && canonical_declaration_path(&unit.module, component.name())
                            == target.path()
                })
                .map(|(_, component)| {
                    component
                        .signature()
                        .iter()
                        .filter_map(|item| match item {
                            eqiora_lang::SignatureItem::Property(value) => {
                                Some(value.name().to_owned())
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        let is_composition = |path: &NamePath| {
            resolve(path, CanonicalDeclarationKind::MaterialComposition).is_some()
        };
        let mut dimension_document = unit.document.clone();
        eqiora_lang::SourceAstFactory::rewrite_dimension_expressions(
            &mut dimension_document,
            |expression| {
                let mut pending = vec![expression];
                while let Some(value) = pending.pop() {
                    let path = match value.kind() {
                        eqiora_lang::ExprKind::Name(name) => Some(
                            NamePath::from_segments([name.clone()], value.range())
                                .expect("parsed dimension name"),
                        ),
                        eqiora_lang::ExprKind::Path(path) => Some(path.clone()),
                        eqiora_lang::ExprKind::Binary { left, right, .. } => {
                            pending.extend([left.as_ref(), right.as_ref()]);
                            None
                        }
                        eqiora_lang::ExprKind::Unary { value, .. } => {
                            pending.push(value);
                            None
                        }
                        _ => None,
                    };
                    if let Some(path) = path
                        && let Some(target) = resolve(&path, CanonicalDeclarationKind::Dimension)
                    {
                        references.push((target, unit.file.clone(), path.range()));
                    }
                }
                expression.clone()
            },
        );
        let mut enum_document = unit.document.clone();
        eqiora_lang::SourceAstFactory::visit_value_types(&mut enum_document, |_, syntax| {
            if let eqiora_lang::ValueTypeSyntaxKind::Named(path) = syntax.kind()
                && let Some(target) = resolve(path, CanonicalDeclarationKind::Enum)
            {
                references.push((target, unit.file.clone(), path.range()));
            }
        });
        eqiora_lang::SourceAstFactory::visit_expressions(&mut enum_document, |_, expression| {
            let paths: Vec<_> = match expression.kind() {
                eqiora_lang::ExprKind::Path(path) => vec![path],
                eqiora_lang::ExprKind::Case { arms, .. } => {
                    arms.iter().map(|arm| arm.pattern()).collect()
                }
                _ => Vec::new(),
            };
            for path in paths {
                if let Some((prefix, _)) = path.as_str().rsplit_once('.')
                    && let Ok(prefix) = NamePath::from_segments(prefix.split('.'), path.range())
                    && let Some(target) = resolve(&prefix, CanonicalDeclarationKind::Enum)
                {
                    references.push((target, unit.file.clone(), path.range()));
                }
            }
        });
        let mut push = |path: &NamePath, kind| match resolve_reference(
            &unit.module,
            path,
            kind,
            &alias_index,
            &declaration_index,
        ) {
            Some(target) => references.push((target, unit.file.clone(), path.range())),
            None => diagnostics.push(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                &unit.file,
                path.range(),
                format!("resolved declaration reference `{path}` has no canonical target"),
            )),
        };

        for (_, _, contract, _, _, _, _, _, _) in unit.document.property_release_syntax() {
            push(contract, CanonicalDeclarationKind::PropertyContract);
        }
        for (_, _, bindings, _) in unit.document.material_composition_syntax() {
            for (_, release, _) in bindings {
                push(release, CanonicalDeclarationKind::PropertyRelease);
            }
        }
        for component in unit.document.components() {
            collect_signature_references(component.signature(), &mut push);
            for item in component.items() {
                collect_component_item_references(
                    item,
                    &mut push,
                    &instance_properties,
                    &is_composition,
                );
            }
        }
        for model in unit.document.models() {
            collect_signature_references(model.signature(), &mut push);
            for item in model.items() {
                collect_model_item_references(
                    item,
                    &mut push,
                    &instance_properties,
                    &is_composition,
                );
            }
        }
    }
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }
    references.sort_by(|left, right| {
        (&left.1, left.2.start(), left.2.end()).cmp(&(&right.1, right.2.start(), right.2.end()))
    });
    Ok(references)
}

fn collect_component_item_references(
    item: &ComponentItem,
    push: &mut impl FnMut(&NamePath, CanonicalDeclarationKind),
    properties: &impl Fn(&eqiora_lang::InstanceDecl) -> Vec<String>,
    is_composition: &impl Fn(&NamePath) -> bool,
) {
    match item {
        ComponentItem::Port(port) => collect_port_reference(port.syntax(), push),
        ComponentItem::PortFamily(family) => collect_port_reference(family.port().syntax(), push),
        ComponentItem::Instance(instance) => {
            collect_instance_references(instance, push, properties, is_composition)
        }
        _ => {}
    }
}

fn collect_model_item_references(
    item: &Item,
    push: &mut impl FnMut(&NamePath, CanonicalDeclarationKind),
    properties: &impl Fn(&eqiora_lang::InstanceDecl) -> Vec<String>,
    is_composition: &impl Fn(&NamePath) -> bool,
) {
    match item {
        Item::Port(port) => collect_port_reference(port.syntax(), push),
        Item::Instance(instance) => {
            collect_instance_references(instance, push, properties, is_composition)
        }
        _ => {}
    }
}

fn collect_port_reference(
    syntax: &PortSyntax,
    push: &mut impl FnMut(&NamePath, CanonicalDeclarationKind),
) {
    match syntax {
        PortSyntax::ScalarPhysicalConnector { connector }
        | PortSyntax::FieldPhysical { connector, .. } => {
            push(connector, CanonicalDeclarationKind::Connector);
        }
        _ => {}
    }
}

fn collect_signature_references(
    signature: &[eqiora_lang::SignatureItem],
    push: &mut impl FnMut(&NamePath, CanonicalDeclarationKind),
) {
    for item in signature {
        match item {
            eqiora_lang::SignatureItem::Property(value) => {
                push(value.contract(), CanonicalDeclarationKind::PropertyContract)
            }
            eqiora_lang::SignatureItem::Port(port) => collect_port_reference(port.syntax(), push),
            eqiora_lang::SignatureItem::PortFamily(family) => {
                collect_port_reference(family.port().syntax(), push)
            }
            _ => {}
        }
    }
}

fn collect_instance_references(
    instance: &eqiora_lang::InstanceDecl,
    push: &mut impl FnMut(&NamePath, CanonicalDeclarationKind),
    properties: &impl Fn(&eqiora_lang::InstanceDecl) -> Vec<String>,
    is_composition: &impl Fn(&NamePath) -> bool,
) {
    push(instance.definition(), CanonicalDeclarationKind::Component);
    let properties = properties(instance);
    for binding in instance
        .bindings()
        .iter()
        .filter(|binding| properties.iter().any(|name| name == binding.name()))
    {
        let path = match binding.value().kind() {
            eqiora_lang::ExprKind::Name(name) => {
                NamePath::from_segments([name.as_str()], binding.value().range())
                    .expect("parsed name")
            }
            eqiora_lang::ExprKind::Path(path) => path.clone(),
            _ => continue, // Property elaboration already rejects non-reference values.
        };
        let segments = path.segments().collect::<Vec<_>>();
        if segments.len() > 1 {
            let prefix = NamePath::from_segments(
                segments[..segments.len() - 1].iter().copied(),
                path.range(),
            )
            .expect("parsed path");
            if is_composition(&prefix) {
                push(&prefix, CanonicalDeclarationKind::MaterialComposition);
                continue;
            }
        }
        push(&path, CanonicalDeclarationKind::PropertyRelease);
    }
}

fn resolve_reference(
    declaring_module: &CompilationModuleId,
    path: &NamePath,
    kind: CanonicalDeclarationKind,
    aliases: &BTreeMap<(CompilationModuleId, String), CompilationModuleId>,
    declarations: &BTreeMap<(CompilationNamespaceId, String, CanonicalDeclarationKind), usize>,
) -> Option<usize> {
    let segments = path.segments().collect::<Vec<_>>();
    let (target_module, name) = match segments.as_slice() {
        [name] => (declaring_module, *name),
        [alias, name] => (
            aliases.get(&(declaring_module.clone(), (*alias).to_owned()))?,
            *name,
        ),
        _ => return None,
    };
    declarations
        .get(&(
            target_module.owner().clone(),
            canonical_declaration_path(target_module, name),
            kind,
        ))
        .copied()
}
