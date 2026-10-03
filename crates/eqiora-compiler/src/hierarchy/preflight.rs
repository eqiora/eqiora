mod interfaces;
mod notation;
mod property;
pub(crate) use interfaces::model_items as owned_model_items;
use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{
    ComponentDecl, ComponentItem, ConnectorDecl, Document, InstanceDecl, Item, ModelDecl, NamePath,
    PortSyntax, PureOperatorDecl, TextRange,
};
use eqiora_schema::kernel::pure_operator::PureOperatorDefinition;

use crate::diagnostics::source_error;
use crate::identity::IdentityNamespace;
use crate::pure_operator::compile_definitions;
use crate::resolved::{AnalyzedResolvedHierarchy, CompilationModuleId};
#[cfg(test)]
use crate::source_identity::LocalSourceIdentity;

use super::HierarchyLimits;

#[derive(Debug, Clone, Copy)]
pub(super) struct ExpansionSize {
    pub(super) declarations: usize,
    pub(super) connections: usize,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum DefinitionNamespace {
    Local,
    Resolved(CompilationModuleId),
}

impl DefinitionNamespace {
    pub(super) fn declaration_prefix(&self) -> Vec<String> {
        match self {
            Self::Local => Vec::new(),
            Self::Resolved(module) => {
                let mut prefix = core::iter::once("package".to_owned())
                    .chain(module.owner().segments().iter().cloned())
                    .collect::<Vec<_>>();
                prefix.push("module".to_owned());
                prefix.extend(module.name().segments().iter().cloned());
                prefix
            }
        }
    }
}

impl core::fmt::Display for DefinitionNamespace {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Local => formatter.write_str("<local>"),
            Self::Resolved(namespace) => namespace.fmt(formatter),
        }
    }
}

#[derive(Clone)]
pub(super) struct ConnectorDefinition<'a> {
    pub(super) namespace: DefinitionNamespace,
    pub(super) file: &'a str,
    pub(super) declaration: &'a ConnectorDecl,
}

impl core::ops::Deref for ConnectorDefinition<'_> {
    type Target = ConnectorDecl;

    fn deref(&self) -> &Self::Target {
        self.declaration
    }
}

#[derive(Clone)]
pub(super) struct ComponentDefinition<'a> {
    pub(super) namespace: DefinitionNamespace,
    pub(super) file: &'a str,
    pub(super) declaration: &'a ComponentDecl,
    owned_interfaces: std::sync::Arc<[ComponentItem]>,
}

impl core::ops::Deref for ComponentDefinition<'_> {
    type Target = ComponentDecl;

    fn deref(&self) -> &Self::Target {
        self.declaration
    }
}

#[derive(Clone)]
pub(super) struct ModelDefinition<'a> {
    pub(super) namespace: DefinitionNamespace,
    pub(super) file: &'a str,
    pub(super) declaration: &'a ModelDecl,
    pub(super) owned_interfaces: std::sync::Arc<[Item]>,
}

#[derive(Clone)]
pub(super) struct PureOperatorSourceDefinition<'a> {
    pub(super) declaration: &'a PureOperatorDecl,
    pub(super) definition: PureOperatorDefinition,
}

impl core::ops::Deref for PureOperatorSourceDefinition<'_> {
    type Target = PureOperatorDecl;

    fn deref(&self) -> &Self::Target {
        self.declaration
    }
}

impl core::ops::Deref for ModelDefinition<'_> {
    type Target = ModelDecl;

    fn deref(&self) -> &Self::Target {
        self.declaration
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct DefinitionKey {
    pub(super) namespace: DefinitionNamespace,
    pub(super) name: String,
}

impl DefinitionKey {
    pub(super) fn display(&self) -> String {
        match self.namespace {
            DefinitionNamespace::Local => self.name.clone(),
            DefinitionNamespace::Resolved(_) => format!("{}::{}", self.namespace, self.name),
        }
    }
}

pub(super) struct Elaborator<'a> {
    pub(super) property_catalog: std::sync::Arc<crate::property::catalog::Catalog>,
    pub(super) property_aliases: Vec<crate::resolved::ResolvedAlias>,
    pub(super) property_local_module: Option<crate::resolved::CompilationModuleId>,
    pub(super) selected_models: BTreeSet<DefinitionKey>,
    pub(super) selected_boundary_sides:
        BTreeMap<(DefinitionKey, String), (usize, eqiora_schema::kernel::BoundarySide)>,
    pub(super) selected_component: Option<(DefinitionKey, super::parameters::SymbolicParameterMap)>,
    pub(super) notations: BTreeMap<(String, u32, u32), eqiora_lang::Notation>,
    pub(super) native: BTreeMap<DefinitionNamespace, &'a eqiora_lang::Module>,
    pub(super) records: BTreeMap<DefinitionNamespace, BTreeMap<String, crate::record::BoundRecord>>,
    pub(super) enumerations:
        BTreeMap<DefinitionNamespace, BTreeMap<String, crate::enumeration::BoundEnum>>,
    pub(super) finite_spaces:
        BTreeMap<DefinitionNamespace, BTreeMap<String, crate::nominal::BoundFiniteSpace>>,
    root_namespace: DefinitionNamespace,
    pub(super) identity_namespace: IdentityNamespace,
    connectors: BTreeMap<DefinitionKey, ConnectorDefinition<'a>>,
    pure_operators: BTreeMap<DefinitionKey, PureOperatorSourceDefinition<'a>>,
    components: BTreeMap<DefinitionKey, ComponentDefinition<'a>>,
    models: BTreeMap<DefinitionKey, ModelDefinition<'a>>,
    aliases: BTreeMap<(DefinitionNamespace, String), DefinitionNamespace>,
    pub(super) limits: HierarchyLimits,
}

impl<'a> Elaborator<'a> {
    pub(super) fn enum_definition(
        &self,
        id: eqiora_core::RawId,
    ) -> Option<&eqiora_schema::kernel::EnumDef> {
        self.enumerations
            .values()
            .flat_map(|values| values.values())
            .map(|value| &value.definition)
            .find(|definition| definition.id().erase() == id)
    }

    #[cfg(test)]
    pub(super) fn new(
        file: &'a str,
        source_bytes: usize,
        document: &'a Document,
        source_identity: LocalSourceIdentity,
        limits: HierarchyLimits,
    ) -> Result<Self, Vec<Diagnostic>> {
        Self::with_identity(
            file,
            source_bytes,
            document,
            source_identity.namespace().map_err(|error| vec![error])?,
            None,
            limits,
        )
    }

    pub(super) fn with_identity(
        file: &'a str,
        source_bytes: usize,
        document: &'a Document,
        identity_namespace: IdentityNamespace,
        native: Option<&'a eqiora_lang::Module>,
        limits: HierarchyLimits,
    ) -> Result<Self, Vec<Diagnostic>> {
        let mut diagnostics = Vec::new();
        if source_bytes > limits.max_source_bytes {
            diagnostics.push(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                file,
                TextRange::new(0, u32::try_from(source_bytes).unwrap_or(u32::MAX)),
                format!(
                    "source requires {source_bytes} bytes, exceeding the {} byte hierarchy limit",
                    limits.max_source_bytes
                ),
            ));
        }

        let namespace = DefinitionNamespace::Local;
        let mut connectors = BTreeMap::new();
        let mut pure_operators = BTreeMap::new();
        let mut components = BTreeMap::new();
        let mut models = BTreeMap::new();
        index_unit(
            namespace.clone(),
            file,
            document,
            limits,
            &mut connectors,
            &mut pure_operators,
            &mut components,
            &mut models,
            &mut diagnostics,
        );

        let finite_spaces =
            crate::nominal::finite_spaces(file, document, &identity_namespace, |name| {
                native.and_then(|native| native.nominal_identity(name))
            })?;
        let elaborator = Self {
            property_catalog: std::sync::Arc::new(Default::default()),
            property_aliases: Vec::new(),
            property_local_module: None,
            selected_boundary_sides: BTreeMap::new(),
            selected_models: BTreeSet::new(),
            selected_component: None,
            notations: notation::index(file, document),
            native: native
                .map(|module| (DefinitionNamespace::Local, module))
                .into_iter()
                .collect(),
            records: BTreeMap::from([(
                namespace.clone(),
                crate::record::declarations(file, document, &identity_namespace, |name| {
                    native.and_then(|native| native.nominal_identity(name))
                })?,
            )]),
            enumerations: BTreeMap::from([(
                namespace.clone(),
                crate::enumeration::declarations(file, document, &identity_namespace, |name| {
                    native.and_then(|native| native.nominal_identity(name))
                })?,
            )]),
            finite_spaces: BTreeMap::from([(namespace.clone(), finite_spaces)]),
            root_namespace: namespace,
            identity_namespace,
            connectors,
            pure_operators,
            components,
            models,
            aliases: BTreeMap::new(),
            limits,
        };
        elaborator.validate_definition_scopes(&mut diagnostics);
        if diagnostics.is_empty() {
            Ok(elaborator)
        } else {
            Err(diagnostics)
        }
    }

    pub(super) fn new_resolved(
        analysis: &'a AnalyzedResolvedHierarchy,
        limits: HierarchyLimits,
    ) -> Result<Self, Vec<Diagnostic>> {
        let root_namespace = DefinitionNamespace::Resolved(analysis.root.clone());
        let mut identity_segments = core::iter::once("resolved-package-v1".to_owned())
            .chain(analysis.root.owner().segments().iter().cloned())
            .collect::<Vec<_>>();
        identity_segments.push("module".to_owned());
        identity_segments.extend(analysis.root.name().segments().iter().cloned());
        let identity_namespace = IdentityNamespace::with_limits(identity_segments, limits.identity)
            .map_err(|error| vec![error])?;
        let mut connectors = BTreeMap::new();
        let mut pure_operators = BTreeMap::new();
        let mut components = BTreeMap::new();
        let mut models = BTreeMap::new();
        let mut diagnostics = Vec::new();
        for unit in &analysis.units {
            if unit.source_bytes > limits.max_source_bytes {
                diagnostics.push(source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    &unit.file,
                    TextRange::new(0, u32::try_from(unit.source_bytes).unwrap_or(u32::MAX)),
                    format!(
                        "source requires {} bytes, exceeding the {} byte hierarchy limit",
                        unit.source_bytes, limits.max_source_bytes
                    ),
                ));
            }
            index_unit(
                DefinitionNamespace::Resolved(unit.module.clone()),
                &unit.file,
                &unit.document,
                limits,
                &mut connectors,
                &mut pure_operators,
                &mut components,
                &mut models,
                &mut diagnostics,
            );
        }
        let aliases = analysis
            .aliases
            .iter()
            .map(|alias| {
                (
                    (
                        DefinitionNamespace::Resolved(alias.declaring_module().clone()),
                        alias.alias().to_owned(),
                    ),
                    DefinitionNamespace::Resolved(alias.target_module().clone()),
                )
            })
            .collect();
        let elaborator = Self {
            property_catalog: analysis.property_catalog.clone(),
            property_aliases: analysis.aliases.to_vec(),
            property_local_module: None,
            selected_boundary_sides: BTreeMap::new(),
            selected_models: BTreeSet::new(),
            selected_component: None,
            notations: analysis
                .units
                .iter()
                .flat_map(|unit| notation::index(&unit.file, &unit.document))
                .collect(),
            native: analysis
                .units
                .iter()
                .filter_map(|unit| {
                    unit.native
                        .as_deref()
                        .map(|module| (DefinitionNamespace::Resolved(unit.module.clone()), module))
                })
                .collect(),
            records: crate::record::resolved_declarations(&analysis.units, &analysis.aliases)?
                .into_iter()
                .map(|(module, values)| (DefinitionNamespace::Resolved(module), values))
                .collect(),
            enumerations: analysis
                .units
                .iter()
                .map(|unit| {
                    let namespace = crate::enumeration::resolved_namespace(&unit.module)
                        .map_err(|e| vec![e])?;
                    crate::enumeration::declarations(
                        &unit.file,
                        &unit.document,
                        &namespace,
                        |name| {
                            unit.native
                                .as_ref()
                                .and_then(|module| module.nominal_identity(name))
                        },
                    )
                    .map(|values| (DefinitionNamespace::Resolved(unit.module.clone()), values))
                })
                .collect::<Result<_, _>>()?,
            finite_spaces: crate::nominal::resolved_spaces(&analysis.units, &analysis.aliases)?
                .into_iter()
                .map(|(module, values)| (DefinitionNamespace::Resolved(module), values))
                .collect(),
            root_namespace,
            identity_namespace,
            connectors,
            pure_operators,
            components,
            models,
            aliases,
            limits,
        };
        elaborator.validate_definition_scopes(&mut diagnostics);
        if diagnostics.is_empty() {
            Ok(elaborator)
        } else {
            Err(diagnostics)
        }
    }

    /// Check the selected Model against its actual static bindings while retaining
    /// the authored source namespace and every unselected definition.
    pub(super) fn bind_selected_model(
        &mut self,
        model: ModelDefinition<'a>,
        supports: &[crate::external::ExternalGeometrySupportBinding],
    ) {
        let key = DefinitionKey {
            namespace: model.namespace.clone(),
            name: model.name().to_owned(),
        };
        self.selected_models.insert(key.clone());
        for support in supports {
            if let crate::external::ExternalGeometrySupportBinding::Boundary {
                slot,
                embedding: Some(embedding),
                ..
            } = support
            {
                self.selected_boundary_sides.insert(
                    (key.clone(), slot.clone()),
                    (embedding.normal_axis(), embedding.side()),
                );
            }
        }
        self.models.insert(
            DefinitionKey {
                namespace: model.namespace.clone(),
                name: model.name().to_owned(),
            },
            model,
        );
    }

    pub(super) fn entry_model(&self, path: &str) -> Result<ModelDefinition<'a>, String> {
        self.find_entry_model(path)?
            .ok_or_else(|| format!("unresolved entry Model `{path}`"))
    }

    pub(super) fn find_entry_model(
        &self,
        path: &str,
    ) -> Result<Option<ModelDefinition<'a>>, String> {
        let (namespace, name, imported) = match path.split_once('.') {
            None if !path.is_empty() => (self.root_namespace.clone(), path, false),
            Some((alias, name)) if !alias.is_empty() && !name.is_empty() && !name.contains('.') => {
                let target = self
                    .aliases
                    .get(&(self.root_namespace.clone(), alias.to_owned()))
                    .ok_or_else(|| {
                        format!("unknown direct module alias `{alias}` in entry Model `{path}`")
                    })?;
                (target.clone(), name, true)
            }
            _ => {
                return Err(format!(
                    "entry Model `{path}` must be a root-local name or one direct alias-qualified name"
                ));
            }
        };
        let Some(definition) = self
            .models
            .get(&DefinitionKey {
                namespace,
                name: name.to_owned(),
            })
            .cloned()
        else {
            return Ok(None);
        };
        if imported && definition.declaration.visibility() != eqiora_lang::VisibilitySyntax::Public
        {
            return Err(format!("private Model `{path}` cannot be imported"));
        }
        Ok(Some(definition))
    }

    pub(super) fn connectors(
        &self,
    ) -> impl ExactSizeIterator<Item = (&DefinitionKey, &ConnectorDefinition<'a>)> {
        self.connectors.iter()
    }

    pub(super) fn components(
        &self,
    ) -> impl ExactSizeIterator<Item = (&DefinitionKey, &ComponentDefinition<'a>)> {
        self.components.iter()
    }

    pub(super) fn pure_operators(
        &self,
    ) -> impl ExactSizeIterator<Item = (&DefinitionKey, &PureOperatorSourceDefinition<'a>)> {
        self.pure_operators.iter()
    }

    pub(super) fn record_for_type(
        &self,
        owner: &DefinitionNamespace,
        syntax: &eqiora_lang::ValueTypeSyntax,
    ) -> Option<&crate::record::BoundRecord> {
        let eqiora_lang::ValueTypeSyntaxKind::Named(path) = syntax.kind() else {
            return None;
        };
        self.visible_records(owner).get(path.as_str()).copied()
    }

    pub(super) fn visible_records(
        &self,
        owner: &DefinitionNamespace,
    ) -> BTreeMap<String, &crate::record::BoundRecord> {
        let mut visible = self
            .records
            .get(owner)
            .into_iter()
            .flat_map(|values| values.iter())
            .map(|(name, value)| (name.clone(), value))
            .collect::<BTreeMap<_, _>>();
        for ((declaring, alias), target) in &self.aliases {
            if declaring != owner {
                continue;
            }
            for (name, value) in self
                .records
                .get(target)
                .into_iter()
                .flat_map(|values| values.iter())
            {
                if value.visibility == eqiora_lang::VisibilitySyntax::Public {
                    visible.insert(format!("{alias}.{name}"), value);
                }
            }
        }
        visible
    }

    pub(super) fn visible_enumerations(
        &self,
        owner: &DefinitionNamespace,
    ) -> BTreeMap<String, &crate::enumeration::BoundEnum> {
        let mut visible = self
            .enumerations
            .get(owner)
            .into_iter()
            .flat_map(|values| values.iter())
            .map(|(name, value)| (name.clone(), value))
            .collect::<BTreeMap<_, _>>();
        for ((declaring, alias), target) in &self.aliases {
            if declaring != owner {
                continue;
            }
            for (name, value) in self
                .enumerations
                .get(target)
                .into_iter()
                .flat_map(|values| values.iter())
            {
                if value.visibility == eqiora_lang::VisibilitySyntax::Public {
                    visible.insert(format!("{alias}.{name}"), value);
                }
            }
        }
        visible
    }

    /// Pure definitions visible from one declaration namespace, keyed by the
    /// exact source path accepted at that use site.
    pub(super) fn visible_pure_operators(
        &self,
        owner: &DefinitionNamespace,
    ) -> BTreeMap<String, (PureOperatorDefinition, Vec<String>)> {
        let mut visible = self
            .pure_operators
            .iter()
            .filter(|(key, _)| &key.namespace == owner)
            .map(|(key, value)| {
                (
                    key.name.clone(),
                    (
                        value.definition.clone(),
                        value
                            .declaration
                            .formals()
                            .iter()
                            .map(|formal| formal.name().to_owned())
                            .collect(),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for ((declaring, alias), target) in &self.aliases {
            if declaring != owner {
                continue;
            }
            for (key, value) in &self.pure_operators {
                if &key.namespace == target
                    && value.declaration.visibility() == eqiora_lang::VisibilitySyntax::Public
                {
                    visible.insert(
                        format!("{alias}.{}", key.name),
                        (
                            value.definition.clone(),
                            value
                                .declaration
                                .formals()
                                .iter()
                                .map(|formal| formal.name().to_owned())
                                .collect(),
                        ),
                    );
                }
            }
        }
        visible
    }

    pub(super) fn models(
        &self,
    ) -> impl ExactSizeIterator<Item = (&DefinitionKey, &ModelDefinition<'a>)> {
        self.models.iter()
    }

    pub(super) fn resolve_connector(
        &self,
        owner: &DefinitionNamespace,
        path: &NamePath,
        file: &str,
        range: TextRange,
    ) -> Result<ConnectorDefinition<'a>, Diagnostic> {
        let (key, imported) = self.resolve_key(owner, path, "Connector", file, range)?;
        let definition = self.connectors.get(&key).cloned().ok_or_else(|| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                format!("unresolved Connector `{path}`"),
            )
        })?;
        if imported && definition.declaration.visibility() != eqiora_lang::VisibilitySyntax::Public
        {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                format!("private Connector `{path}` cannot be imported"),
            ));
        }
        Ok(definition)
    }

    pub(super) fn resolve_component(
        &self,
        owner: &DefinitionNamespace,
        path: &NamePath,
        file: &str,
        range: TextRange,
    ) -> Result<ComponentDefinition<'a>, Diagnostic> {
        let (key, imported) = self.resolve_key(owner, path, "component", file, range)?;
        let definition = self.components.get(&key).cloned().ok_or_else(|| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                format!("unresolved component `{path}`"),
            )
        })?;
        if imported && definition.declaration.visibility() != eqiora_lang::VisibilitySyntax::Public
        {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                format!("private component `{path}` cannot be imported"),
            ));
        }
        Ok(definition)
    }

    pub(super) fn resolve_pure_operator(
        &self,
        owner: &DefinitionNamespace,
        path: &NamePath,
        file: &str,
        range: TextRange,
    ) -> Result<PureOperatorSourceDefinition<'a>, Diagnostic> {
        let (key, imported) = self.resolve_key(owner, path, "pure operator", file, range)?;
        let definition = self.pure_operators.get(&key).cloned().ok_or_else(|| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                format!("unresolved pure operator `{path}`"),
            )
        })?;
        if imported && definition.declaration.visibility() != eqiora_lang::VisibilitySyntax::Public
        {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                format!("private pure operator `{path}` cannot be imported"),
            ));
        }
        Ok(definition)
    }

    fn resolve_key(
        &self,
        owner: &DefinitionNamespace,
        path: &NamePath,
        label: &str,
        file: &str,
        range: TextRange,
    ) -> Result<(DefinitionKey, bool), Diagnostic> {
        let segments = path.segments().collect::<Vec<_>>();
        match segments.as_slice() {
            [name] => Ok((
                DefinitionKey {
                    namespace: owner.clone(),
                    name: (*name).to_owned(),
                },
                false,
            )),
            [alias, name] => {
                let target = self
                    .aliases
                    .get(&(owner.clone(), (*alias).to_owned()))
                    .ok_or_else(|| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            range,
                            format!("unknown direct package alias `{alias}` in {label} `{path}`"),
                        )
                    })?;
                Ok((
                    DefinitionKey {
                        namespace: target.clone(),
                        name: (*name).to_owned(),
                    },
                    true,
                ))
            }
            _ => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                format!(
                    "{label} `{path}` uses transitive or member qualification; v1 accepts only a package-local name or one direct alias"
                ),
            )),
        }
    }

    fn validate_definition_scopes(&self, diagnostics: &mut Vec<Diagnostic>) {
        for definition in self.components.values() {
            let component = definition.declaration;
            let file = definition.file;
            let mut names = BTreeMap::<&str, TextRange>::new();
            for requirement in component.signature() {
                let (name, range) = (requirement.name(), requirement.range());
                validate_identifier(file, name, range, self.limits, diagnostics);
                if names.insert(name, range).is_some() {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        range,
                        format!("duplicate signature name `{name}`"),
                    ));
                }
            }
            for item in component.items() {
                let named = match item {
                    ComponentItem::Let(value) => Some((value.name(), value.range())),
                    ComponentItem::Parameter(value) => Some((value.name(), value.range())),
                    ComponentItem::Port(value) => Some((value.name(), value.range())),
                    ComponentItem::PortFamily(value) => Some((value.port().name(), value.range())),
                    ComponentItem::Observable(value) => Some((value.name(), value.range())),
                    ComponentItem::Field(value) => Some((value.name(), value.range())),
                    ComponentItem::Clock(value) => Some((value.name(), value.range())),
                    ComponentItem::Event(value) => Some((value.name(), value.range())),
                    ComponentItem::Relation(value) => Some((value.name(), value.range())),
                    ComponentItem::RelationFamily(value) => {
                        Some((value.relation().name(), value.range()))
                    }
                    ComponentItem::Instance(value) => Some((value.name(), value.range())),
                    ComponentItem::Connection(_) | ComponentItem::BoundaryConnection(_) => None,
                    _ => None,
                };
                if let Some((name, range)) = named {
                    validate_identifier(file, name, range, self.limits, diagnostics);
                    if names.insert(name, range).is_some() {
                        diagnostics.push(source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            range,
                            format!(
                                "duplicate member name `{name}` in component `{}`",
                                component.name()
                            ),
                        ));
                    }
                }

                match item {
                    ComponentItem::Port(port) => {
                        if let PortSyntax::ScalarPhysicalConnector { connector }
                        | PortSyntax::FieldPhysical { connector, .. } = port.syntax()
                            && let Err(error) = self.resolve_connector(
                                &definition.namespace,
                                connector,
                                file,
                                port.range(),
                            )
                        {
                            diagnostics.push(error);
                        }
                    }
                    ComponentItem::PortFamily(family) => {
                        let port = family.port();
                        if let PortSyntax::FieldPhysical { connector, .. } = port.syntax()
                            && let Err(error) = self.resolve_connector(
                                &definition.namespace,
                                connector,
                                file,
                                port.range(),
                            )
                        {
                            diagnostics.push(error);
                        }
                    }
                    ComponentItem::Instance(instance) => {
                        if let Err(error) = self.resolve_component(
                            &definition.namespace,
                            instance.definition(),
                            file,
                            instance.range(),
                        ) {
                            diagnostics.push(error);
                        }
                        validate_binding_names(file, instance, diagnostics);
                    }
                    _ => {}
                }
            }
        }

        for definition in self.models.values() {
            let model = definition.declaration;
            let file = definition.file;
            let mut names = BTreeMap::<&str, TextRange>::new();
            for requirement in model.signature() {
                let (name, range) = (requirement.name(), requirement.range());
                validate_identifier(file, name, range, self.limits, diagnostics);
                if names.insert(name, range).is_some() {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        range,
                        format!("duplicate signature name `{name}`"),
                    ));
                }
            }
            for item in model.items() {
                let named = match item {
                    Item::Domain(value) => Some((value.name(), value.range())),
                    Item::Field(value) => Some((value.name(), value.range())),
                    Item::Observable(value) => Some((value.name(), value.range())),
                    Item::Parameter(value) => Some((value.name(), value.range())),
                    Item::Let(value) => Some((value.name(), value.range())),
                    Item::Port(value) => Some((value.name(), value.range())),
                    Item::Clock(value) => Some((value.name(), value.range())),
                    Item::Event(value) => Some((value.name(), value.range())),
                    Item::Relation(value) => Some((value.name(), value.range())),
                    Item::RelationFamily(value) => Some((value.relation().name(), value.range())),
                    Item::Instance(value) => Some((value.name(), value.range())),
                    Item::Connection(_) | Item::BoundaryConnection(_) => None,
                    _ => None,
                };
                if let Some((name, range)) = named {
                    validate_identifier(file, name, range, self.limits, diagnostics);
                    if names.insert(name, range).is_some() {
                        diagnostics.push(source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            range,
                            format!("duplicate name `{name}` in model `{}`", model.name()),
                        ));
                    }
                }
                if let Item::Instance(instance) = item {
                    if let Err(error) = self.resolve_component(
                        &definition.namespace,
                        instance.definition(),
                        file,
                        instance.range(),
                    ) {
                        diagnostics.push(error);
                    }
                    validate_binding_names(file, instance, diagnostics);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
mod index;
use index::{index_unit, validate_binding_names, validate_identifier};

impl ComponentDefinition<'_> {
    pub(super) fn owned_items(&self) -> impl Iterator<Item = &ComponentItem> {
        self.owned_interfaces.iter().chain(self.declaration.items())
    }
}
impl ModelDefinition<'_> {
    pub(super) fn owned_items(&self) -> impl Iterator<Item = &Item> {
        self.owned_interfaces.iter().chain(self.declaration.items())
    }
}
