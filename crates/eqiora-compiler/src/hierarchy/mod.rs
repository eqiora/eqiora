//! Typed component elaboration into the existing flat Relation network.
//!
//! Hierarchy is compiler-owned source structure. This module resolves and
//! validates the complete instance tree, stages every semantic identity, and
//! only then invokes the ordinary flat lowerer once. No Component, Instance,
//! or ConnectorType node enters the Semantic Kernel.

mod reductions;

use crate::resolved::AnalyzedResolvedHierarchy;
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{
    ComponentItem, Document, ExprKind, Item, NamePath, SourceAstFactory, TextRange, parse,
};

use crate::connection_sets::ConnectionSetLimits;
use crate::diagnostics::source_error;
use crate::external::{
    ExternalComponentBinding, ExternalGeometrySupportBinding, ExternalParameterBinding,
};
use crate::identity::ElaborationIdentityLimits;
use crate::lower::CompiledModel;
use crate::provenance::ProvenanceLimits;
use crate::source_identity::LocalSourceIdentity;

mod body_check;
pub(crate) use body_check::CompletionIndex;
mod check;
mod clocks;
mod complete_exterior;
mod definition_graph;
mod expand;
mod exposure_cuts;
mod field_slots;
mod flat;
mod named_bindings;
mod occurrence_connections;
mod parameters;
pub(crate) use parameters::exact_signed_literal;
mod physical_closure;
mod preflight;
mod quantity_member;
mod scope;
pub(crate) mod selected;
mod supports;

pub(crate) use definition_graph::CheckedDefinitionGraph;
use expand::RootExpansion;
use preflight::Elaborator;

/// Independent resource limits for one physical-exposure projection catalog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalExposureProjectionLimits {
    /// Maximum eliminated exposure projections.
    pub max_projections: usize,
    /// Maximum retained Port memberships in one occurrence cut.
    pub max_members_per_cut: usize,
    /// Maximum retained Port memberships summed across every cut.
    pub max_memberships: usize,
    /// Maximum fragment-membership visits summed across cut derivation.
    pub max_traversal_memberships: usize,
}

impl Default for PhysicalExposureProjectionLimits {
    fn default() -> Self {
        Self {
            max_projections: 1_000_000,
            max_members_per_cut: 65_536,
            max_memberships: 4_000_000,
            max_traversal_memberships: 16_000_000,
        }
    }
}

/// Independent resource limits for occurrence-bound complete exteriors.
///
/// These limits constrain memberships, not the declarations produced by
/// boundary families. The latter remain governed by the ordinary hierarchy
/// declaration and connection budgets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompleteExteriorLimits {
    /// Maximum exact Boundary identities admitted by one explicit set.
    pub max_members_per_set: usize,
    /// Maximum explicit Boundary memberships resolved during one elaboration.
    pub max_total_memberships: usize,
}

impl Default for CompleteExteriorLimits {
    fn default() -> Self {
        Self {
            max_members_per_set: 65_536,
            max_total_memberships: 4_000_000,
        }
    }
}

/// Bounded policy for one typed hierarchy elaboration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HierarchyLimits {
    /// Maximum source bytes accepted by this elaboration entry point.
    pub max_source_bytes: usize,
    /// Maximum nested component instance depth, including the root Model.
    pub max_instance_depth: usize,
    /// Maximum component instances below one root Model.
    pub max_instances: usize,
    /// Maximum named Kernel declarations after flattening.
    pub max_declarations: usize,
    /// Maximum anonymous Connections after flattening.
    pub max_connections: usize,
    /// Independent topology-normalization budgets for physical connection
    /// fragments, endpoint memberships, maximal sets, and set width.
    pub connection_sets: ConnectionSetLimits,
    /// Independent budgets for eliminated physical exposure projections.
    pub physical_exposures: PhysicalExposureProjectionLimits,
    /// Independent budgets for occurrence-bound complete exteriors.
    pub complete_exteriors: CompleteExteriorLimits,
    /// Maximum source identifier bytes.
    pub max_identifier_bytes: usize,
    /// Maximum globally indexed Connector, pure-operator, Component, and Model definitions.
    pub max_definitions: usize,
    /// Maximum nested Component instance edges across definition bodies.
    pub max_definition_edges: usize,
    /// Maximum symbolic Parameter terms retained during definition checking.
    pub max_parameter_terms: usize,
    /// Maximum source diagnostics retained by definition validation.
    pub max_definition_diagnostics: usize,
    /// Maximum `(definition, reachable Connector)` memberships retained by
    /// exact reusable-definition summaries.
    pub max_definition_reachability_pairs: usize,
    /// Identity construction and projection limits.
    pub identity: ElaborationIdentityLimits,
    /// Source-provenance sidecar limits.
    pub provenance: ProvenanceLimits,
}

impl Default for HierarchyLimits {
    fn default() -> Self {
        let identity = ElaborationIdentityLimits::default();
        let connection_sets = ConnectionSetLimits {
            max_members_per_fragment: identity.max_anonymous_connection_members,
            max_members_per_set: identity.max_anonymous_connection_members,
            ..ConnectionSetLimits::default()
        };
        Self {
            max_source_bytes: 16 * 1_024 * 1_024,
            max_instance_depth: 64,
            max_instances: 1_000_000,
            max_declarations: 4_000_000,
            max_connections: 1_000_000,
            connection_sets,
            physical_exposures: PhysicalExposureProjectionLimits::default(),
            complete_exteriors: CompleteExteriorLimits::default(),
            max_identifier_bytes: 1_024,
            max_definitions: 1_000_000,
            max_definition_edges: 1_000_000,
            max_parameter_terms: 4_000_000,
            max_definition_diagnostics: 4_096,
            max_definition_reachability_pairs: 4_000_000,
            identity,
            provenance: ProvenanceLimits::default(),
        }
    }
}

fn compile_external_component_from_definition<'a>(
    elaborator: &Elaborator<'a>,
    checked: &CheckedDefinitionGraph,
    component: preflight::ComponentDefinition<'a>,
    binding: &ExternalComponentBinding,
    limits: HierarchyLimits,
) -> Result<CompiledModel, Vec<Diagnostic>> {
    let file = component.file;
    let range = TextRange::default();
    let component_path = NamePath::from_segments([binding.component()], range)
        .map_err(|error| vec![hierarchy_error(error.message())])?;
    if component.visibility() != eqiora_lang::VisibilitySyntax::Public {
        return Err(vec![source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            component.range(),
            format!(
                "external Component `{}` must be declared public",
                component.name()
            ),
        )]);
    }
    let values = parameters::resolve_external_parameters(
        file,
        component.declaration,
        binding.parameters(),
        &parameters::RecordContext::component(elaborator, &component),
    )?;
    let summary = definition_graph::selected_component_summary(
        elaborator,
        checked,
        &component,
        values.clone(),
    )?;
    // The Component summary already reserves one declaration, identity, and
    // provenance entry for every Parameter slot. An explicit external value
    // realizes that reserved slot as one root Model Parameter; only the
    // synthetic Geometry support Domains increase the summarized footprint.
    let declarations = checked_external_footprint(
        "declarations",
        summary.declarations(),
        binding
            .supports()
            .iter()
            .map(crate::external::ExternalGeometrySupportBinding::allocated_support_count)
            .sum::<usize>(),
        limits.max_declarations,
    )?;
    checked_external_footprint(
        "staged identities",
        summary.staged_identities(),
        binding
            .supports()
            .iter()
            .map(crate::external::ExternalGeometrySupportBinding::allocated_support_count)
            .sum::<usize>(),
        limits.identity.max_staged_identities,
    )?;
    checked_external_footprint(
        "provenance entries",
        summary.provenance_entries(),
        binding
            .supports()
            .iter()
            .map(crate::external::ExternalGeometrySupportBinding::allocated_support_count)
            .sum::<usize>(),
        limits.provenance.max_entries,
    )?;

    let bindings_by_name = binding
        .parameters()
        .iter()
        .map(|parameter| (parameter.parameter(), parameter))
        .collect::<std::collections::BTreeMap<_, _>>();
    let frame_context = supports::component_spatial_supports(file, component.declaration)?;
    let mut root_items = Vec::new();
    for item in component.owned_items() {
        let ComponentItem::Parameter(declaration) = item else {
            continue;
        };
        let Some(parameter) = bindings_by_name.get(declaration.name()) else {
            continue;
        };
        let declaration = SourceAstFactory::parameter(
            declaration.name(),
            parameters::specialize_type(file, declaration.value_type(), &values)
                .map_err(|error| vec![error])?,
            SourceAstFactory::value_literal(
                parameter.value(),
                projection_frame(
                    file,
                    parameter.value(),
                    declaration.default(),
                    &frame_context,
                    range,
                )?,
                range,
                |id| {
                    declaration
                        .value_type()
                        .resolved_nominal()
                        .filter(|ty| {
                            ty.enum_definition()
                                .is_some_and(|definition| definition.erase() == id)
                        })
                        .and_then(|_| match declaration.value_type().kind() {
                            eqiora_lang::ValueTypeSyntaxKind::Named(name) => Some(name.clone()),
                            _ => None,
                        })
                },
                |id| elaborator.enum_definition(id),
            )
            .map_err(|error| vec![hierarchy_error(error.message())])?,
            range,
        )
        .map_err(|error| vec![hierarchy_error(error.message())])?;
        root_items.push(Item::Parameter(declaration));
    }
    let parameter_bindings = binding
        .parameters()
        .iter()
        .map(|parameter| {
            let value = SourceAstFactory::expression(
                ExprKind::Name(parameter.parameter().to_owned()),
                range,
            )?;
            SourceAstFactory::named_binding(parameter.parameter(), value, range)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| vec![hierarchy_error(error.message())])?;
    let support_bindings = binding
        .supports()
        .iter()
        .map(|support| {
            SourceAstFactory::named_binding(
                support.slot(),
                SourceAstFactory::expression(ExprKind::Name(support.slot().to_owned()), range)?,
                range,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| vec![hierarchy_error(error.message())])?;
    let clock_bindings = binding
        .clocks
        .iter()
        .map(|(name, _)| {
            SourceAstFactory::named_binding(
                name,
                SourceAstFactory::expression(ExprKind::Name(name.clone()), range)?,
                range,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| vec![hierarchy_error(error.message())])?;
    let property_bindings = binding
        .properties()
        .keys()
        .map(|name| {
            SourceAstFactory::named_binding(
                name,
                SourceAstFactory::expression(ExprKind::Name(name.clone()), range)?,
                range,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| vec![hierarchy_error(error.message())])?;
    let instance = SourceAstFactory::instance(
        "definition",
        component_path,
        None,
        parameter_bindings
            .into_iter()
            .chain(support_bindings)
            .chain(property_bindings)
            .chain(clock_bindings)
            .collect(),
        range,
    )
    .map_err(|error| vec![hierarchy_error(error.message())])?;
    root_items.push(Item::Instance(instance));
    let root = SourceAstFactory::model(
        eqiora_lang::VisibilitySyntax::Private,
        binding.model(),
        Vec::new(),
        root_items,
        range,
    )
    .map_err(|error| vec![hierarchy_error(error.message())])?;
    let model = preflight::ModelDefinition {
        owned_interfaces: std::sync::Arc::from([]),
        namespace: component.namespace.clone(),
        file,
        declaration: &root,
    };
    RootExpansion::new(
        elaborator,
        model,
        preflight::ExpansionSize {
            declarations,
            connections: summary.connections(),
        },
    )
    .map_err(|error| vec![error])?
    .expand_external(
        component,
        binding.supports(),
        &binding.clocks,
        binding.properties(),
    )?
    .compile(limits)
}

fn checked_external_footprint(
    label: &str,
    summarized: usize,
    synthetic_supports: usize,
    limit: usize,
) -> Result<usize, Vec<Diagnostic>> {
    let total = summarized.checked_add(synthetic_supports).ok_or_else(|| {
        vec![hierarchy_error(format!(
            "external Component occurrence {label} overflow usize"
        ))]
    })?;
    if total > limit {
        return Err(vec![hierarchy_error(format!(
            "external Component occurrence has {total} {label}, exceeding the {limit} limit"
        ))]);
    }
    Ok(total)
}

fn projection_frame(
    file: &str,
    value: &eqiora_core::ValueLiteral,
    initializer: Option<&eqiora_lang::Expr>,
    frames: &std::collections::BTreeMap<
        String,
        eqiora_schema::kernel::typing::SpatialSupport<String>,
    >,
    range: TextRange,
) -> Result<Option<NamePath>, Vec<Diagnostic>> {
    if value.value_type().frame() == eqiora_core::ValueFrame::Invariant {
        return Ok(None);
    }
    let mut pending = initializer.into_iter().collect::<Vec<_>>();
    while let Some(expression) = pending.pop() {
        match expression.kind() {
            ExprKind::Array(values) => pending.extend(values.iter().rev()),
            ExprKind::Call { callee, arguments } if callee.as_str() == "tensor_value" => {
                let name = arguments
                    .named()
                    .and_then(|bindings| bindings.iter().find(|binding| binding.name() == "frame"))
                    .and_then(|binding| match binding.value().kind() {
                        ExprKind::Name(name) => Some(name.as_str()),
                        ExprKind::Path(name) => Some(name.as_str()),
                        _ => None,
                    });
                if let Some(name) = name.filter(|name| frames.contains_key(*name)) {
                    return NamePath::from_segments(name.split('.'), range)
                        .map(Some)
                        .map_err(|error| vec![hierarchy_error(error.message())]);
                }
            }
            _ => {}
        }
    }
    let mut unique = Vec::new();
    for (name, support) in frames {
        if !unique.iter().any(|(_, existing)| *existing == support) {
            unique.push((name, support));
        }
    }
    if let [(name, _)] = unique.as_slice() {
        return NamePath::from_segments(name.split('.'), range)
            .map(Some)
            .map_err(|error| vec![hierarchy_error(error.message())]);
    }
    Err(vec![source_error(
        codes::LANGUAGE_TYPE_ERROR,
        file,
        range,
        "external spatial Parameter requires an explicit or unique exact frame context",
    )])
}

#[cfg(test)]
fn compile_hierarchy_with_limits(
    file: &str,
    source_bytes: usize,
    document: &Document,
    limits: HierarchyLimits,
) -> Result<Vec<CompiledModel>, Vec<Diagnostic>> {
    selected::local_document(file, source_bytes, document.clone(), None, &[], limits)
}

pub(crate) fn compile_resolved_hierarchy(
    analysis: &AnalyzedResolvedHierarchy,
    checked: &CheckedDefinitionGraph,
    model: &str,
    limits: HierarchyLimits,
) -> Result<CompiledModel, Vec<Diagnostic>> {
    let elaborator = Elaborator::new_resolved(analysis, limits)?;
    let root = elaborator
        .entry_model(model)
        .map_err(|message| vec![hierarchy_error(message)])?;
    let size = checked_model_expansion_size(checked, &root)?;
    RootExpansion::new(&elaborator, root, size)
        .map_err(|error| vec![error])?
        .expand()?
        .compile(limits)
}

fn checked_model_expansion_size(
    checked: &CheckedDefinitionGraph,
    model: &preflight::ModelDefinition<'_>,
) -> Result<preflight::ExpansionSize, Vec<Diagnostic>> {
    let key = preflight::DefinitionKey {
        namespace: model.namespace.clone(),
        name: model.declaration.name().to_owned(),
    };
    let summary = checked.model_summary(&key).ok_or_else(|| {
        vec![hierarchy_error(format!(
            "validated definition graph has no summary for root Model `{}`",
            key.display()
        ))]
    })?;
    Ok(preflight::ExpansionSize {
        declarations: summary.declarations(),
        connections: summary.connections(),
    })
}

pub(crate) fn validate_resolved_hierarchy(
    analysis: &AnalyzedResolvedHierarchy,
    limits: HierarchyLimits,
) -> Result<(), Vec<Diagnostic>> {
    Elaborator::new_resolved(analysis, limits).map(|_| ())
}

pub(crate) fn validate_resolved_definitions(
    analysis: &AnalyzedResolvedHierarchy,
    limits: HierarchyLimits,
) -> Result<CheckedDefinitionGraph, Vec<Diagnostic>> {
    let elaborator = Elaborator::new_resolved(analysis, limits)?;
    check::validate(&elaborator)
}

fn hierarchy_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::LANGUAGE_LOWERING_ERROR, message)
}

#[cfg(test)]
mod tests;

pub(crate) use parameters::closed_value;

pub(crate) fn closed_index(expression: &eqiora_lang::Expr) -> Result<u32, Diagnostic> {
    parameters::static_index("", expression, &Default::default())
}

#[cfg(test)]
#[path = "tests/native_ast.rs"]
mod native_ast_tests;

#[cfg(test)]
#[path = "tests/observable.rs"]
mod observable_tests;
