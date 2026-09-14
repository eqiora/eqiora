use std::collections::BTreeMap;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity};
#[cfg(test)]
use eqiora_lang::SourceAstFactory;
use eqiora_lang::{
    ActivationSyntax, BoundaryPortReferenceSyntax, BoundaryPortSelectorSyntax, Expr, ExprKind,
    FieldDecl, NamePath, PortSyntax, RelationDecl, TextRange,
};

use crate::diagnostics::source_error;
use crate::identity::FullElaborationIdentity;
use crate::lower::{LoweringEquation, LoweringExpression};
use crate::pure_operator::is_builtin_operator;
use eqiora_schema::kernel::pure_operator::PureOperatorDefinition;
use eqiora_schema::kernel::typing::{ExpressionType, SpatialSupport};

use super::flat::SourceLocation;

use super::supports::ResolvedBoundarySet;

mod activation;
pub(super) use activation::port_activation;
mod external;
mod indexed;
mod lets;
mod operators;
mod property;
mod value_expression;
pub(super) use value_expression::rewrite_expression_with_boundary_member;
mod reductions;

#[derive(Debug, Clone)]
pub(super) struct FlatSymbol {
    pub(super) internal_name: String,
    pub(super) display_name: String,
    pub(super) full_identity: FullElaborationIdentity,
    pub(super) kind: SymbolKind,
}

#[derive(Debug, Clone)]
pub(super) enum SymbolKind {
    Domain,
    Field,
    Parameter,
    Port {
        activation: ActivationSyntax,
        quantities: Option<PhysicalMemberNames>,
    },
    Clock(eqiora_schema::kernel::RationalTime),
    Event,
    Observable,
    Relation,
}

impl FlatSymbol {
    fn is_port(&self) -> bool {
        matches!(self.kind, SymbolKind::Port { .. })
    }
}

/// Presentation names on an already nominally resolved Port contract.
/// These never participate in physical compatibility or junction synthesis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PhysicalMemberNames {
    Scalar { across: String, through: String },
    Boundary { trace: String, flux: String },
}

impl PhysicalMemberNames {
    pub(super) fn from_connector(syntax: &eqiora_lang::ConnectorSyntax) -> Option<Self> {
        match syntax {
            eqiora_lang::ConnectorSyntax::ScalarPhysical {
                across_name,
                through_name,
                ..
            } => Some(Self::Scalar {
                across: across_name.clone(),
                through: through_name.clone(),
            }),
            eqiora_lang::ConnectorSyntax::FieldPhysical { trace, flux, .. } => {
                Some(Self::Boundary {
                    trace: trace.name().to_owned(),
                    flux: flux.name().to_owned(),
                })
            }
            _ => None,
        }
    }

    pub(super) fn role(&self, member: &str) -> Option<&'static str> {
        match self {
            Self::Scalar { across, .. } if member == across => Some("across"),
            Self::Scalar { through, .. } if member == through => Some("through"),
            Self::Boundary { trace, .. } if member == trace => Some("trace"),
            Self::Boundary { flux, .. } if member == flux => Some("flux"),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct InstanceInterface {
    pub(super) index_set: Option<eqiora_core::Id<eqiora_core::entity::kinds::IndexSet>>,
    pub(super) public_ports: BTreeMap<String, FlatSymbol>,
    public_port_families: BTreeMap<String, BoundaryPortFamilyIndex>,
}

impl InstanceInterface {
    pub(super) fn with_public_port_families(
        public_ports: BTreeMap<String, FlatSymbol>,
        public_port_families: BTreeMap<String, BoundaryPortFamilyIndex>,
    ) -> Self {
        Self {
            index_set: None,
            public_ports,
            public_port_families,
        }
    }
}

/// The exact occurrence members of one source-declared boundary Port family.
///
/// This is deliberately not a collection of values in the semantic model. It
/// is a lexical resolution index: the family spelling and selector binder lead
/// to one ordinary Port identified by the exact Boundary occurrence identity.
#[derive(Debug, Clone)]
pub(super) struct BoundaryPortFamilyIndex {
    selector_member: String,
    members: BTreeMap<FullElaborationIdentity, FlatSymbol>,
}

/// The exact Boundary selected by the currently expanded family binder.
#[derive(Debug, Clone, Copy)]
pub(super) struct ActiveBoundaryMember<'a> {
    member: &'a str,
    boundary: FullElaborationIdentity,
}

impl<'a> ActiveBoundaryMember<'a> {
    pub(super) const fn new(member: &'a str, boundary: FullElaborationIdentity) -> Self {
        Self { member, boundary }
    }
}

#[derive(Debug, Default, Clone)]
pub(super) struct Scope {
    pub(in crate::hierarchy) lexical_namespace: Option<super::preflight::DefinitionNamespace>,
    pub(in crate::hierarchy) properties:
        BTreeMap<String, std::sync::Arc<eqiora_schema::kernel::PropertyRelease>>,
    pub(super) record_context: super::parameters::RecordContext,
    pub(super) reduction_terms_limit: usize,
    index_sets: BTreeMap<String, indexed::ScopedIndexSet>,
    symbols: BTreeMap<String, FlatSymbol>,
    port_families: BTreeMap<String, BoundaryPortFamilyIndex>,
    boundary_sets: BTreeMap<String, ResolvedBoundarySet<FullElaborationIdentity>>,
    children: BTreeMap<String, InstanceInterface>,
    spatial_supports: BTreeMap<String, SpatialSupport<FullElaborationIdentity>>,
    pub(super) field_evolution: BTreeMap<String, (eqiora_lang::FieldRoleSyntax, ActivationSyntax)>,
    field_types: BTreeMap<String, ExpressionType<FullElaborationIdentity>>,
    values: BTreeMap<String, lets::ScopedValue>,
    pure_operators: BTreeMap<String, (PureOperatorDefinition, Vec<String>)>,
    occurrence_bindings: Vec<SourceLocation>,
    forwarded_parameter_resolution_bindings: Vec<SourceLocation>,
    forwarded_field_resolution_bindings: Vec<SourceLocation>,
    forwarded_boundary_set_resolution_bindings: Vec<SourceLocation>,
    // Preserve source-occurrence DAG shape only for the external root path.
    detach_parameter_expressions: bool,
}

impl Scope {
    pub(super) fn insert_symbol(&mut self, name: String, symbol: FlatSymbol) -> Option<FlatSymbol> {
        self.symbols.insert(name, symbol)
    }

    pub(super) fn insert_child(
        &mut self,
        name: String,
        child: InstanceInterface,
    ) -> Option<InstanceInterface> {
        self.children.insert(name, child)
    }

    pub(super) fn symbol(&self, name: &str) -> Option<&FlatSymbol> {
        self.symbols.get(name)
    }

    pub(super) fn insert_port_family_member(
        &mut self,
        file: &str,
        range: TextRange,
        family_name: String,
        selector_member: &str,
        boundary: FullElaborationIdentity,
        symbol: FlatSymbol,
    ) -> Result<(), Diagnostic> {
        if !symbol.is_port() {
            return Err(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                file,
                range,
                "boundary Port family member did not lower to an ordinary Port",
            ));
        }
        if self.symbols.contains_key(&family_name) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                format!("Port family `{family_name}` collides with an ordinary symbol"),
            ));
        }

        if let Some(family) = self.port_families.get_mut(&family_name) {
            if family.selector_member != selector_member {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    format!(
                        "Port family `{family_name}` cannot use both `{}` and `{selector_member}` as its selector member",
                        family.selector_member
                    ),
                ));
            }
            if family.members.contains_key(&boundary) {
                return Err(source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    file,
                    range,
                    format!(
                        "Port family `{family_name}` contains the same exact Boundary more than once"
                    ),
                ));
            }
            family.members.insert(boundary, symbol);
        } else {
            self.port_families.insert(
                family_name,
                BoundaryPortFamilyIndex {
                    selector_member: selector_member.to_owned(),
                    members: BTreeMap::from([(boundary, symbol)]),
                },
            );
        }
        Ok(())
    }

    pub(super) fn port_family(&self, name: &str) -> Option<&BoundaryPortFamilyIndex> {
        self.port_families.get(name)
    }

    pub(super) fn insert_boundary_set(
        &mut self,
        name: String,
        set: ResolvedBoundarySet<FullElaborationIdentity>,
    ) -> Option<ResolvedBoundarySet<FullElaborationIdentity>> {
        self.boundary_sets.insert(name, set)
    }

    pub(super) fn boundary_set(
        &self,
        name: &str,
    ) -> Option<&ResolvedBoundarySet<FullElaborationIdentity>> {
        self.boundary_sets.get(name)
    }

    pub(super) fn insert_spatial_support(
        &mut self,
        name: String,
        support: SpatialSupport<FullElaborationIdentity>,
    ) -> Option<SpatialSupport<FullElaborationIdentity>> {
        self.spatial_supports.insert(name, support)
    }

    pub(super) fn spatial_support(
        &self,
        name: &str,
    ) -> Option<&SpatialSupport<FullElaborationIdentity>> {
        self.spatial_supports.get(name)
    }

    pub(super) fn frame_supports(&self) -> BTreeMap<String, SpatialSupport<String>> {
        self.spatial_supports
            .iter()
            .map(|(name, support)| (name.clone(), super::parameters::frames::occurrence(support)))
            .collect()
    }

    pub(super) fn spatial_support_by_identity(
        &self,
        identity: FullElaborationIdentity,
    ) -> Option<&SpatialSupport<FullElaborationIdentity>> {
        self.spatial_supports
            .values()
            .find(|support| match support {
                SpatialSupport::Volume { domain, .. } | SpatialSupport::Boundary { domain, .. } => {
                    *domain == identity
                }
                SpatialSupport::Interface { .. } => false,
            })
    }

    pub(super) fn insert_field_type(
        &mut self,
        name: String,
        field_type: ExpressionType<FullElaborationIdentity>,
    ) -> Option<ExpressionType<FullElaborationIdentity>> {
        self.field_types.insert(name, field_type)
    }

    pub(super) fn field_type(
        &self,
        name: &str,
    ) -> Option<&ExpressionType<FullElaborationIdentity>> {
        self.field_types.get(name)
    }

    pub(super) fn set_occurrence_bindings(&mut self, bindings: Vec<SourceLocation>) {
        self.occurrence_bindings = bindings;
    }

    pub(super) fn occurrence_bindings(&self) -> &[SourceLocation] {
        &self.occurrence_bindings
    }

    pub(super) fn set_forwarded_parameter_resolution_bindings(
        &mut self,
        bindings: Vec<SourceLocation>,
    ) {
        self.forwarded_parameter_resolution_bindings = bindings;
    }

    pub(super) fn forwarded_parameter_resolution_bindings(&self) -> &[SourceLocation] {
        &self.forwarded_parameter_resolution_bindings
    }

    pub(super) fn set_forwarded_field_resolution_bindings(
        &mut self,
        bindings: Vec<SourceLocation>,
    ) {
        self.forwarded_field_resolution_bindings = bindings;
    }

    pub(super) fn forwarded_field_resolution_bindings(&self) -> &[SourceLocation] {
        &self.forwarded_field_resolution_bindings
    }

    pub(super) fn set_forwarded_boundary_set_resolution_bindings(
        &mut self,
        bindings: Vec<SourceLocation>,
    ) {
        self.forwarded_boundary_set_resolution_bindings = bindings;
    }

    pub(super) fn forwarded_boundary_set_resolution_bindings(&self) -> &[SourceLocation] {
        &self.forwarded_boundary_set_resolution_bindings
    }

    pub(super) fn resolve_port(&self, path: &NamePath) -> Option<&FlatSymbol> {
        let segments = path.segments().collect::<Vec<_>>();
        match segments.as_slice() {
            [name] => self.symbols.get(*name).filter(|symbol| symbol.is_port()),
            [instance, member] => self
                .children
                .get(*instance)
                .and_then(|child| child.public_ports.get(*member)),
            _ => None,
        }
    }

    pub(super) fn resolve_symbol(&self, path: &NamePath) -> Option<&FlatSymbol> {
        if let Some(symbol) = self.symbols.get(path.as_str()) {
            return Some(symbol);
        }
        let segments = path.segments().collect::<Vec<_>>();
        match segments.as_slice() {
            [name] => self.symbols.get(*name),
            [instance, member] => self
                .children
                .get(*instance)
                .and_then(|child| child.public_ports.get(*member)),
            _ => None,
        }
    }
}

pub(super) fn rewrite_field_scope(
    file: &str,
    declaration: &FieldDecl,
    scope: &Scope,
) -> Result<(Option<String>, ActivationSyntax), Diagnostic> {
    let domain = declaration
        .domain()
        .map(|name| {
            resolve_local_kind(
                file,
                declaration.range(),
                scope,
                name,
                |kind| matches!(kind, SymbolKind::Domain),
                "Field Domain",
            )
            .map(|symbol| symbol.internal_name.clone())
        })
        .transpose()?;
    let activation =
        rewrite_activation(file, declaration.activation(), declaration.range(), scope)?;
    Ok((domain, activation))
}

pub(super) fn rewrite_activation(
    file: &str,
    activation: &ActivationSyntax,
    range: TextRange,
    scope: &Scope,
) -> Result<ActivationSyntax, Diagnostic> {
    match activation {
        ActivationSyntax::Continuous => Ok(ActivationSyntax::Continuous),
        ActivationSyntax::Named(name) => resolve_local_kind(
            file,
            range,
            scope,
            name,
            |kind| matches!(kind, SymbolKind::Clock(_)),
            "Field ClockDomain",
        )
        .map(|symbol| ActivationSyntax::Named(symbol.internal_name.clone())),
        _ => Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            "unsupported activation",
        )),
    }
}

pub(super) fn rewrite_model_port(
    file: &str,
    syntax: &PortSyntax,
    range: TextRange,
    scope: &Scope,
) -> Result<PortSyntax, Diagnostic> {
    match syntax {
        PortSyntax::Signal {
            direction,
            value_type,
            domain,
            activation,
        } => Ok(PortSyntax::Signal {
            direction: *direction,
            value_type: super::parameters::specialize_type(
                file,
                value_type,
                &scope.symbolic_parameters(),
            )?,
            domain: domain
                .as_deref()
                .map(|name| {
                    resolve_local_kind(
                        file,
                        range,
                        scope,
                        name,
                        |kind| matches!(kind, SymbolKind::Domain),
                        "signal support",
                    )
                    .map(|symbol| symbol.internal_name.clone())
                })
                .transpose()?,
            activation: rewrite_activation(file, activation, range, scope)?,
        }),
        PortSyntax::ScalarPhysical { domain } => {
            let domain = resolve_local_kind(
                file,
                range,
                scope,
                domain,
                |kind| matches!(kind, SymbolKind::Domain),
                "scalar physical Domain",
            )?;
            Ok(PortSyntax::ScalarPhysical {
                domain: domain.internal_name.clone(),
            })
        }
        PortSyntax::ScalarPhysicalConnector { .. } => Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            "model-level Port cannot use a component Connector declaration directly",
        )),
        _ => Err(source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            range,
            "Port syntax is newer than hierarchy elaboration",
        )),
    }
}

pub(super) fn rewrite_relation(
    file: &str,
    declaration: &RelationDecl,
    scope: &Scope,
) -> Result<
    (
        ActivationSyntax,
        Option<String>,
        crate::lower::LoweringRelationBody,
    ),
    Diagnostic,
> {
    let activation = match declaration.activation() {
        ActivationSyntax::Continuous => ActivationSyntax::Continuous,
        ActivationSyntax::Named(clock) => {
            let clock = resolve_local_kind(
                file,
                declaration.range(),
                scope,
                clock,
                |kind| matches!(kind, SymbolKind::Clock(_) | SymbolKind::Event),
                "ClockDomain or Event",
            )?;
            ActivationSyntax::Named(clock.internal_name.clone())
        }
        _ => {
            return Err(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                file,
                declaration.range(),
                "Activation syntax is newer than hierarchy elaboration",
            ));
        }
    };
    let domain = declaration
        .domain()
        .map(|name| {
            resolve_local_kind(
                file,
                declaration.range(),
                scope,
                name,
                |kind| matches!(kind, SymbolKind::Domain),
                "Relation Domain",
            )
            .map(|symbol| symbol.internal_name.clone())
        })
        .transpose()?;
    Ok((
        activation,
        domain,
        rewrite_relation_body(file, declaration, scope, None)?,
    ))
}

pub(super) fn rewrite_relation_body(
    file: &str,
    declaration: &RelationDecl,
    scope: &Scope,
    active: Option<ActiveBoundaryMember<'_>>,
) -> Result<crate::lower::LoweringRelationBody, Diagnostic> {
    Ok(match declaration.body() {
        eqiora_lang::RelationBody::Conditions(conditions) => {
            rewrite_equations(file, conditions, scope, active)?.into()
        }
        eqiora_lang::RelationBody::Conservation(terms) => {
            let rewrite = |value: &eqiora_lang::Expr| {
                if scope.reduction_terms_limit > 0 {
                    super::reductions::preflight(
                        file,
                        value,
                        &mut |name| scope.index_set(name).map(|set| set.extent()),
                        &scope.symbolic_parameters(),
                        scope.reduction_terms_limit,
                    )?;
                }
                rewrite_expression_with_boundary_member(file, value, scope, active)
            };
            crate::lower::LoweringRelationBody::Conservation {
                storage: terms.storage().map(rewrite).transpose()?,
                flux: rewrite(terms.flux())?,
                source: rewrite(terms.source())?,
            }
        }
    })
}

pub(super) fn rewrite_equations(
    file: &str,
    equations: &[eqiora_lang::RelationCondition],
    scope: &Scope,
    active: Option<ActiveBoundaryMember<'_>>,
) -> Result<Vec<LoweringEquation>, Diagnostic> {
    equations
        .iter()
        .map(|equation| {
            if scope.reduction_terms_limit > 0 {
                for value in [equation.left(), equation.right()] {
                    super::reductions::preflight(
                        file,
                        value,
                        &mut |name| scope.index_set(name).map(|set| set.extent()),
                        &scope.symbolic_parameters(),
                        scope.reduction_terms_limit,
                    )?;
                }
            }
            Ok(LoweringEquation::rewritten(
                equation,
                rewrite_expression_with_boundary_member(file, equation.left(), scope, active)?,
                rewrite_expression_with_boundary_member(file, equation.right(), scope, active)?,
            ))
        })
        .collect()
}

pub(super) fn resolve_boundary_port_reference<'a>(
    file: &str,
    reference: &BoundaryPortReferenceSyntax,
    scope: &'a Scope,
    active: Option<ActiveBoundaryMember<'_>>,
) -> Result<&'a FlatSymbol, Diagnostic> {
    match reference.selector() {
        Some(selector) => {
            if scope.resolve_port(reference.port()).is_some() {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    selector.range(),
                    format!(
                        "ordinary Port `{}` cannot carry a boundary-family selector",
                        reference.port()
                    ),
                ));
            }
            resolve_boundary_family_selection(file, reference.port(), selector, scope, active)
        }
        None => {
            if resolve_port_family(scope, reference.port()).is_some() {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    reference.port().range(),
                    format!(
                        "Port family `{}` requires an exact boundary selector",
                        reference.port()
                    ),
                ));
            }
            scope.resolve_port(reference.port()).ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    reference.port().range(),
                    format!(
                        "`{}` does not select a visible Port in this scope",
                        reference.port()
                    ),
                )
            })
        }
    }
}

fn resolve_expression_symbol<'a>(
    file: &str,
    path: &NamePath,
    scope: &'a Scope,
) -> Result<&'a FlatSymbol, Diagnostic> {
    if resolve_port_family(scope, path).is_some() {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            path.range(),
            format!("Port family `{path}` requires an exact boundary selector"),
        ));
    }
    scope.resolve_symbol(path).ok_or_else(|| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            path.range(),
            if path.is_qualified() {
                format!("qualified name `{path}` does not select a public Port in this scope")
            } else {
                format!("unresolved name `{path}` in this component instance")
            },
        )
    })
}

fn resolve_boundary_family_selection<'a>(
    file: &str,
    path: &NamePath,
    selector: &BoundaryPortSelectorSyntax,
    scope: &'a Scope,
    active: Option<ActiveBoundaryMember<'_>>,
) -> Result<&'a FlatSymbol, Diagnostic> {
    let Some(family) = resolve_port_family(scope, path) else {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            path.range(),
            format!("`{path}` does not select a visible boundary Port family"),
        ));
    };
    if selector.member() != family.selector_member {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            selector.range(),
            format!(
                "selector member `{}` does not match `{}` declared by Port family `{path}`",
                selector.member(),
                family.selector_member
            ),
        ));
    }

    let boundary = resolve_exact_boundary(file, selector, scope, active)?;
    family.members.get(&boundary).ok_or_else(|| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            selector.range(),
            format!(
                "Port family `{path}` has no member on exact Boundary `{}`",
                selector.target()
            ),
        )
    })
}

fn resolve_port_family<'a>(
    scope: &'a Scope,
    path: &NamePath,
) -> Option<&'a BoundaryPortFamilyIndex> {
    let segments = path.segments().collect::<Vec<_>>();
    match segments.as_slice() {
        [name] => scope.port_families.get(*name),
        [instance, member] => scope
            .children
            .get(*instance)
            .and_then(|child| child.public_port_families.get(*member)),
        _ => None,
    }
}

fn resolve_exact_boundary(
    file: &str,
    selector: &BoundaryPortSelectorSyntax,
    scope: &Scope,
    active: Option<ActiveBoundaryMember<'_>>,
) -> Result<FullElaborationIdentity, Diagnostic> {
    if let Some(active) = active
        && selector.target() == active.member
    {
        return Ok(active.boundary);
    }

    let Some(support) = scope.spatial_support(selector.target()) else {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            selector.range(),
            format!(
                "unresolved exact Boundary selector target `{}`",
                selector.target()
            ),
        ));
    };
    match support {
        SpatialSupport::Boundary { domain, .. } => Ok(*domain),
        SpatialSupport::Volume { .. } | SpatialSupport::Interface { .. } => Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            selector.range(),
            format!(
                "selector target `{}` is not an exact Boundary support",
                selector.target()
            ),
        )),
    }
}

pub(super) fn resolve_local_kind<'a>(
    file: &str,
    range: TextRange,
    scope: &'a Scope,
    name: &str,
    expected: impl FnOnce(&SymbolKind) -> bool,
    label: &str,
) -> Result<&'a FlatSymbol, Diagnostic> {
    let Some(symbol) = scope.symbols.get(name) else {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            format!("unresolved {label} `{name}`"),
        ));
    };
    if !expected(&symbol.kind) {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            format!("`{name}` is not a {label}"),
        ));
    }
    Ok(symbol)
}

#[cfg(test)]
mod tests;
