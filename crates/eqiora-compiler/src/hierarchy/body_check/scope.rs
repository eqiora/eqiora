mod connections;
mod symbols;
mod value_shape;
use value_shape::resolve_frame;
pub(in crate::hierarchy) use value_shape::resolve_value_shape;
mod indexed;
pub(super) use connections::{validate_connection, validate_resolved_connection};
mod ports;
pub(super) use ports::{component_port_contract, model_port_contract};
mod child_ports;
mod input_bindings;
pub(super) use input_bindings::validate_input_bindings;
mod diagnostics;
pub(super) use diagnostics::unresolved;
mod scalar_connection;
use scalar_connection::{connection_fragment_error, validate_connection_contract};

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::ValueFrame;
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, ValueShape};
use eqiora_lang::{
    BoundaryConnectionDecl, BoundaryPairingSyntax, BoundaryPortReferenceSyntax,
    BoundaryPortSelectorSyntax, ComponentItem, ComponentPortDecl, ComponentPortFamilyDecl,
    ConnectionDecl, ConnectionSyntax, ConnectorSyntax, FamilyBinderSyntax, FieldDecl, FrameSyntax,
    InstanceDecl, NamePath, PortDecl, PortSyntax, SignalDirectionSyntax, SupportSlotSyntax,
    TextRange, ValueShapeSyntax, VisibilitySyntax,
};
use eqiora_schema::kernel::scalar_connection::{
    ScalarConnectionKind, ScalarConnectionViolation, ScalarPortContract, validate_scalar_connection,
};
use eqiora_schema::kernel::typing::{ExpressionType, SpatialSupport};
use eqiora_schema::kernel::{BoundaryPairing, BoundaryPhysicalConnector, SignalDirection};

use crate::connection_sets::{ConnectionFragment, ConnectionSetError, ConnectionSetLimits};
use crate::diagnostics::source_error;
use crate::dimensions::lower_dimension;

use super::super::preflight::{
    ComponentDefinition, DefinitionKey, DefinitionNamespace, Elaborator,
};
use super::super::scope::PhysicalMemberNames;
use super::{PhysicalConnectionFragment, ResolvedPhysicalEndpoint};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PhysicalNominal {
    Connector(DefinitionKey),
    BoundaryConnector {
        definition: DefinitionKey,
        shape: ValueShape,
    },
    ModelDomain(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PortContract {
    Signal {
        direction: SignalDirectionSyntax,
        value_type: eqiora_core::ValueType,
        support: Option<SpatialSupport<String>>,
        activation: eqiora_lang::ActivationSyntax,
    },
    Physical {
        nominal: PhysicalNominal,
        quantities: PhysicalMemberNames,
        across_type: eqiora_core::ValueType,
        through_type: eqiora_core::ValueType,
    },
    BoundaryPhysical {
        nominal: PhysicalNominal,
        quantities: PhysicalMemberNames,
        connector: BoundaryPhysicalConnector,
        support: SpatialSupport<String>,
    },
}

#[derive(Debug, Clone)]
pub(super) struct BoundaryPortFamilyContract {
    binder: FamilyBinderSyntax,
    port: PortContract,
}

impl BoundaryPortFamilyContract {
    fn selected(
        &self,
        file: &str,
        port: &NamePath,
        selector: &BoundaryPortSelectorSyntax,
        active: &BoundaryFamilyScope,
    ) -> Result<PortContract, Diagnostic> {
        if selector.member() != self.binder.member() {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                selector.range(),
                format!(
                    "boundary selector member `{}` does not match Port family member `{}`",
                    selector.member(),
                    self.binder.member()
                ),
            ));
        }
        if selector.target() != active.binder.member() {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                selector.range(),
                format!(
                    "boundary selector target `{}` does not name active family member `{}`",
                    selector.target(),
                    active.binder.member()
                ),
            ));
        }
        self.selected_on_support(file, port, selector, active.support.clone())
    }

    fn selected_on_support(
        &self,
        file: &str,
        port: &NamePath,
        selector: &BoundaryPortSelectorSyntax,
        support: SpatialSupport<String>,
    ) -> Result<PortContract, Diagnostic> {
        if selector.member() != self.binder.member() {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                selector.range(),
                format!(
                    "boundary selector member `{}` does not match Port family member `{}`",
                    selector.member(),
                    self.binder.member()
                ),
            ));
        }
        let PortContract::BoundaryPhysical {
            nominal,
            connector,
            quantities,
            ..
        } = &self.port
        else {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                port.range(),
                "only a field-physical Port can be selected as a boundary family",
            ));
        };
        Ok(PortContract::BoundaryPhysical {
            nominal: nominal.clone(),
            quantities: quantities.clone(),
            connector: connector.clone(),
            support,
        })
    }
}

/// One lexically scoped, identity-parametric member of a complete exterior.
///
/// The synthetic identity is used only while checking a reusable definition;
/// occurrence expansion replaces it with one exact bound boundary identity.
#[derive(Debug, Clone)]
pub(super) struct BoundaryFamilyScope {
    binder: FamilyBinderSyntax,
    support: SpatialSupport<String>,
}

impl BoundaryFamilyScope {
    pub(super) const fn binder(&self) -> &FamilyBinderSyntax {
        &self.binder
    }

    pub(super) fn support(&self) -> SpatialSupport<String> {
        self.support.clone()
    }
}

impl PortContract {
    pub(super) fn for_connection(mut self, syntax: ConnectionSyntax, exposed: bool) -> Self {
        if exposed
            && syntax == ConnectionSyntax::Signal
            && let Self::Signal { direction, .. } = &mut self
        {
            *direction = match direction {
                SignalDirectionSyntax::Input => SignalDirectionSyntax::Output,
                SignalDirectionSyntax::Output => SignalDirectionSyntax::Input,
            };
        }
        self
    }

    pub(super) fn expression_type(&self) -> Option<ExpressionType<String>> {
        match self {
            Self::Signal {
                value_type,
                support,
                ..
            } => Some(ExpressionType::new(value_type.clone(), support.clone())),
            Self::Physical { .. } => None,
            Self::BoundaryPhysical { .. } => None,
        }
    }

    pub(super) const fn is_physical(&self) -> bool {
        matches!(self, Self::Physical { .. } | Self::BoundaryPhysical { .. })
    }
}

#[derive(Debug, Clone)]
pub(super) enum DomainContract {
    Spatial(SpatialSupport<String>),
    Physical {
        quantities: PhysicalMemberNames,
        /// Complete across/through types; indirection keeps spatial entries small.
        types: Box<(eqiora_core::ValueType, eqiora_core::ValueType)>,
    },
}

impl DomainContract {
    pub(super) fn spatial_support(&self) -> Option<SpatialSupport<String>> {
        match self {
            Self::Spatial(support) => Some(support.clone()),
            Self::Physical { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum SymbolContract {
    Domain(DomainContract),
    Support(SpatialSupport<String>),
    Field(
        ExpressionType<String>,
        eqiora_lang::FieldRoleSyntax,
        eqiora_lang::ActivationSyntax,
    ),
    Parameter(ExpressionType<String>),
    Alias(std::sync::Arc<super::expression::AliasContract>),
    Port(PortContract),
    PortFamily(BoundaryPortFamilyContract),
    CompleteExterior {
        parent: SpatialSupport<String>,
    },
    Clock,
    Event,
    Observable,
    Relation,
}

pub(super) struct DefinitionScope<'e, 'd> {
    pub(super) elaborator: &'e Elaborator<'d>,
    pub(super) namespace: DefinitionNamespace,
    pub(super) file: &'d str,
    pub(super) symbols: BTreeMap<String, SymbolContract>,
    pub(super) properties: BTreeMap<String, crate::property::Contract>,
    pub(super) exposed_signals: BTreeSet<String>,
    pub(super) borrowed_clocks: BTreeSet<String>,
    pub(super) index_sets: BTreeMap<String, Option<u32>>,
    pub(super) record_context: crate::hierarchy::parameters::RecordContext,
    pub(super) static_values: crate::hierarchy::parameters::SymbolicParameterMap,
    pub(super) children: BTreeMap<String, ComponentDefinition<'d>>,
    pub(super) child_instances: BTreeMap<String, &'d InstanceDecl>,
}

impl<'e, 'd> DefinitionScope<'e, 'd> {
    pub(super) fn new(
        elaborator: &'e Elaborator<'d>,
        namespace: DefinitionNamespace,
        file: &'d str,
    ) -> Self {
        Self {
            record_context: crate::hierarchy::parameters::RecordContext {
                visible: elaborator
                    .visible_records(&namespace)
                    .into_iter()
                    .map(|(name, record)| (name, record.clone()))
                    .collect(),
                parameters: BTreeMap::new(),
            },
            elaborator,
            namespace,
            file,
            symbols: BTreeMap::new(),
            properties: BTreeMap::new(),
            exposed_signals: BTreeSet::new(),
            borrowed_clocks: BTreeSet::new(),
            static_values: BTreeMap::new(),
            index_sets: BTreeMap::new(),
            children: BTreeMap::new(),
            child_instances: BTreeMap::new(),
        }
    }

    pub(super) fn activation_matches(
        &self,
        left: &eqiora_lang::ActivationSyntax,
        right: &eqiora_lang::ActivationSyntax,
    ) -> bool {
        left == right
            || matches!((left, right), (eqiora_lang::ActivationSyntax::Named(a), eqiora_lang::ActivationSyntax::Named(b)) if self.borrowed_clocks.contains(a) || self.borrowed_clocks.contains(b))
    }

    pub(super) fn spatial_support(&self, name: &str) -> Option<SpatialSupport<String>> {
        match self.symbols.get(name) {
            Some(SymbolContract::Domain(contract)) => contract.spatial_support(),
            Some(SymbolContract::Support(support)) => Some(support.clone()),
            _ => None,
        }
    }

    pub(super) fn boundary_family_scope(
        &self,
        binder: &FamilyBinderSyntax,
    ) -> Result<BoundaryFamilyScope, Diagnostic> {
        let Some(SymbolContract::CompleteExterior { parent }) =
            self.symbols.get(binder.set().as_str())
        else {
            return Err(self.wrong_local_kind(
                binder.range(),
                binder.set().as_str(),
                "complete-exterior support set",
            ));
        };
        let SpatialSupport::Volume { domain, dimensions } = parent else {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                binder.range(),
                format!(
                    "complete-exterior support set `{}` requires a volume parent",
                    binder.set().as_str()
                ),
            ));
        };
        Ok(BoundaryFamilyScope {
            binder: binder.clone(),
            support: SpatialSupport::Boundary {
                domain: synthetic_boundary_member_identity(binder),
                parent: domain.clone(),
                dimensions: *dimensions,
            },
        })
    }

    pub(super) fn resolve_port(&self, path: &NamePath) -> Result<PortContract, Diagnostic> {
        match self.resolve_symbol(path)? {
            SymbolContract::Port(contract) => Ok(contract),
            _ => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                path.range(),
                format!("`{path}` does not select a visible Port in this scope"),
            )),
        }
    }

    pub(super) fn resolve_boundary_port_reference(
        &self,
        reference: &BoundaryPortReferenceSyntax,
        active: &BoundaryFamilyScope,
    ) -> Result<PortContract, Diagnostic> {
        let Some(selector) = reference.selector() else {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                reference.port().range(),
                format!(
                    "boundary-family Port `{}` requires an exact `[member = target]` selector",
                    reference.port()
                ),
            ));
        };
        let family = self.resolve_boundary_port_family(reference.port())?;
        self.validate_boundary_family_mapping(reference.port(), &family, active, selector.range())?;
        family.selected(self.file, reference.port(), selector, active)
    }

    pub(super) fn resolve_boundary_port_selection(
        &self,
        port: &NamePath,
        selector: &BoundaryPortSelectorSyntax,
        active: &BoundaryFamilyScope,
    ) -> Result<PortContract, Diagnostic> {
        let family = self.resolve_boundary_port_family(port)?;
        self.validate_boundary_family_mapping(port, &family, active, selector.range())?;
        family.selected(self.file, port, selector, active)
    }

    fn validate_boundary_family_mapping(
        &self,
        port: &NamePath,
        family: &BoundaryPortFamilyContract,
        active: &BoundaryFamilyScope,
        range: TextRange,
    ) -> Result<(), Diagnostic> {
        let segments = port.segments().collect::<Vec<_>>();
        match segments.as_slice() {
            [_] if family.binder.set().as_str() == active.binder.set().as_str() => Ok(()),
            [_] => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                range,
                format!(
                    "local Port family `{port}` belongs to complete exterior `{}`, not active exterior `{}`",
                    family.binder.set().as_str(),
                    active.binder.set().as_str()
                ),
            )),
            [instance, _] => {
                let Some(occurrence) = self.child_instances.get(*instance) else {
                    return Err(self.invalid_public_port_selection(port));
                };
                if occurrence.bindings().iter().any(|binding| {
                    binding.name() == family.binder.set().as_str() && matches!(binding.value().kind(),eqiora_lang::ExprKind::Name(target) if target==active.binder.set().as_str())
                }) {
                    Ok(())
                } else {
                    Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        range,
                        format!(
                            "child Port family `{port}` is not forwarded from active complete exterior `{}`",
                            active.binder.set().as_str()
                        ),
                    ))
                }
            }
            _ => Err(self.invalid_public_port_selection(port)),
        }
    }

    fn resolve_boundary_port_family(
        &self,
        path: &NamePath,
    ) -> Result<BoundaryPortFamilyContract, Diagnostic> {
        let segments = path.segments().collect::<Vec<_>>();
        match segments.as_slice() {
            [name] => match self.symbols.get(*name) {
                Some(SymbolContract::PortFamily(contract)) => Ok(contract.clone()),
                _ => Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    path.range(),
                    format!("`{path}` does not select a boundary Port family in this scope"),
                )),
            },
            [instance, member] => {
                let child = self
                    .children
                    .get(*instance)
                    .ok_or_else(|| self.invalid_public_port_selection(path))?;
                let family = child
                    .owned_items()
                    .find_map(|item| match item {
                        ComponentItem::PortFamily(family)
                            if family.port().name() == *member
                                && family.port().visibility() == VisibilitySyntax::Public =>
                        {
                            Some(family)
                        }
                        _ => None,
                    })
                    .ok_or_else(|| self.invalid_public_port_selection(path))?;
                let support = synthetic_component_family_support(child, family.binder())?;
                component_port_family_contract(self.elaborator, child, family, support).map_err(
                    |mut errors| {
                        errors.pop().unwrap_or_else(|| {
                            source_error(
                                codes::LANGUAGE_LOWERING_ERROR,
                                self.file,
                                path.range(),
                                "child Port-family contract validation failed without a diagnostic",
                            )
                        })
                    },
                )
            }
            _ => Err(self.invalid_public_port_selection(path)),
        }
    }

    fn invalid_public_port_selection(&self, path: &NamePath) -> Diagnostic {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            self.file,
            path.range(),
            format!("qualified name `{path}` does not select a public Port in this scope"),
        )
    }
}

pub(in crate::hierarchy) use ports::field_expression_type;

pub(super) fn component_port_family_contract(
    elaborator: &Elaborator<'_>,
    owner: &ComponentDefinition<'_>,
    declaration: &ComponentPortFamilyDecl,
    support: SpatialSupport<String>,
) -> Result<BoundaryPortFamilyContract, Vec<Diagnostic>> {
    let port = declaration.port();
    let PortSyntax::FieldPhysical {
        connector,
        support: declared_support,
    } = port.syntax()
    else {
        return Err(vec![source_error(
            codes::LANGUAGE_TYPE_ERROR,
            owner.file,
            declaration.range(),
            "only a field-physical Port can declare a boundary family",
        )]);
    };
    if declared_support != declaration.binder().member() {
        return Err(vec![source_error(
            codes::LANGUAGE_TYPE_ERROR,
            owner.file,
            declaration.range(),
            "field-physical Port-family support must name its binder member",
        )]);
    }
    let connector_definition = elaborator
        .resolve_connector(&owner.namespace, connector, owner.file, declaration.range())
        .map_err(|error| vec![error])?;
    let ConnectorSyntax::FieldPhysical {
        trace,
        flux,
        shape,
        frame,
        pairing,
    } = connector_definition.declaration.syntax()
    else {
        return Err(vec![source_error(
            codes::LANGUAGE_TYPE_ERROR,
            owner.file,
            declaration.range(),
            "field-physical Port family requires a field-physical Connector",
        )]);
    };
    let port = boundary_port_contract(
        connector_definition,
        support,
        owner.file,
        declaration.range(),
        trace,
        flux,
        shape,
        *frame,
        *pairing,
    )?;
    Ok(BoundaryPortFamilyContract {
        binder: declaration.binder().clone(),
        port,
    })
}

fn synthetic_component_family_support(
    owner: &ComponentDefinition<'_>,
    binder: &FamilyBinderSyntax,
) -> Result<SpatialSupport<String>, Diagnostic> {
    let exterior = owner
        .declaration
        .signature()
        .iter()
        .find_map(|item| match item {
            eqiora_lang::SignatureItem::Support(declaration)
                if declaration.name() == binder.set().as_str() =>
            {
                Some(declaration)
            }
            _ => None,
        })
        .ok_or_else(|| {
            unresolved(
                owner.file,
                binder.range(),
                binder.set().as_str(),
                "complete-exterior support set",
            )
        })?;
    let SupportSlotSyntax::CompleteExterior { parent } = exterior.syntax() else {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            owner.file,
            exterior.range(),
            format!(
                "support `{}` is not a complete exterior",
                binder.set().as_str()
            ),
        ));
    };
    let parent_declaration = owner
        .declaration
        .signature()
        .iter()
        .find_map(|item| match item {
            eqiora_lang::SignatureItem::Support(declaration) if declaration.name() == parent => {
                Some(declaration)
            }
            _ => None,
        })
        .ok_or_else(|| {
            unresolved(
                owner.file,
                exterior.range(),
                parent,
                "complete-exterior volume parent",
            )
        })?;
    let SupportSlotSyntax::Volume { ambient_dimension } = parent_declaration.syntax() else {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            owner.file,
            exterior.range(),
            format!(
                "complete-exterior support `{}` requires volume parent `{parent}`",
                binder.set().as_str()
            ),
        ));
    };
    Ok(SpatialSupport::Boundary {
        domain: synthetic_boundary_member_identity(binder),
        parent: parent.clone(),
        dimensions: *ambient_dimension,
    })
}

fn synthetic_boundary_member_identity(binder: &FamilyBinderSyntax) -> String {
    format!(
        "@complete-exterior/{}/{}",
        binder.set().as_str(),
        binder.member()
    )
}

pub(super) fn validate_boundary_connection(
    scope: &DefinitionScope<'_, '_>,
    declaration: &BoundaryConnectionDecl,
) -> Result<Option<super::PhysicalEndpointSelections>, Diagnostic> {
    if declaration.syntax() != ConnectionSyntax::Conserving {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "spatial-periodic Connections are not supported inside reusable Components",
        ));
    }
    let Some(binder) = declaration.binder() else {
        return validate_model_boundary_connection(scope, declaration).map(Some);
    };
    let active = scope.boundary_family_scope(binder)?;
    if declaration.ports().len() < 2 {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "Connection requires at least two Ports",
        ));
    }
    let mut exact_references = BTreeSet::new();
    let mut contracts = Vec::with_capacity(declaration.ports().len());
    for reference in declaration.ports() {
        let Some(selector) = reference.selector() else {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                scope.file,
                reference.port().range(),
                "every Port in a boundary-family Connection requires an exact selector",
            ));
        };
        let key = (
            reference
                .port()
                .segments()
                .map(str::to_owned)
                .collect::<Vec<_>>(),
            selector.member().to_owned(),
            selector.target().to_owned(),
        );
        if !exact_references.insert(key) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                scope.file,
                declaration.range(),
                "Connection repeats the same selected Port",
            ));
        }
        contracts.push(scope.resolve_boundary_port_reference(reference, &active)?);
    }
    let Some(PortContract::BoundaryPhysical { nominal, .. }) = contracts.first() else {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "boundary-family Connection requires field-physical Ports",
        ));
    };
    if contracts.iter().skip(1).any(|contract| {
        !matches!(contract, PortContract::BoundaryPhysical { nominal: candidate, .. } if candidate == nominal)
    }) {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "boundary-family Connection requires the exact same specialized Connector",
        ));
    }
    Ok(None)
}

/// Validate one exact Boundary Connection in a closed Model.
///
/// Family members are specialized to the selected Model Boundary for type
/// checking. Ordinary endpoints are returned separately: their membership is
/// definition-independent, while the equivalence class containing the exact
/// family member is deliberately deferred to occurrence expansion.
pub(super) fn validate_model_boundary_connection(
    scope: &DefinitionScope<'_, '_>,
    declaration: &BoundaryConnectionDecl,
) -> Result<super::PhysicalEndpointSelections, Diagnostic> {
    if declaration.binder().is_some() {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "Model boundary Connection cannot declare a family binder",
        ));
    }
    if declaration.syntax() == ConnectionSyntax::SpatialPeriodic && declaration.ports().len() != 2 {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "spatial-periodic Connection requires exactly two Ports",
        ));
    }
    if declaration.syntax() != ConnectionSyntax::SpatialPeriodic && declaration.ports().len() < 2 {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "Connection requires at least two Ports",
        ));
    }

    let mut exact_references = BTreeSet::new();
    let mut contracts = Vec::with_capacity(declaration.ports().len());
    let mut deferred_memberships = super::PhysicalEndpointSelections::new();
    for reference in declaration.ports() {
        let key = (
            reference
                .port()
                .segments()
                .map(str::to_owned)
                .collect::<Vec<_>>(),
            reference
                .selector()
                .map(|selector| (selector.member().to_owned(), selector.target().to_owned())),
        );
        if !exact_references.insert(key) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                scope.file,
                declaration.range(),
                "Connection repeats the same selected Port",
            ));
        }

        let contract = if let Some(selector) = reference.selector() {
            let support = scope.spatial_support(selector.target()).ok_or_else(|| {
                unresolved(
                    scope.file,
                    selector.range(),
                    selector.target(),
                    "selected boundary Domain",
                )
            })?;
            if !matches!(support, SpatialSupport::Boundary { .. }) {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    scope.file,
                    selector.range(),
                    format!(
                        "boundary selector target `{}` is not a boundary Domain",
                        selector.target()
                    ),
                ));
            }
            scope
                .resolve_boundary_port_family(reference.port())?
                .selected_on_support(scope.file, reference.port(), selector, support)?
        } else {
            let endpoint =
                ResolvedPhysicalEndpoint::from_path(reference.port()).ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        scope.file,
                        reference.port().range(),
                        format!(
                            "`{}` is not a local or child Port selection",
                            reference.port()
                        ),
                    )
                })?;
            deferred_memberships.insert(endpoint);
            scope.resolve_port(reference.port())?
        };
        contracts.push(contract);
    }

    let Some(PortContract::BoundaryPhysical { nominal, .. }) = contracts.first() else {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "exact boundary Connection requires field-physical Ports",
        ));
    };
    if contracts.iter().skip(1).any(|contract| {
        !matches!(contract, PortContract::BoundaryPhysical { nominal: candidate, .. } if candidate == nominal)
    }) {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "exact boundary Connection requires the same specialized Connector",
        ));
    }
    Ok(deferred_memberships)
}

#[allow(clippy::too_many_arguments)]
fn boundary_port_contract(
    connector_definition: super::super::preflight::ConnectorDefinition<'_>,
    support: SpatialSupport<String>,
    file: &str,
    range: TextRange,
    trace: &eqiora_lang::ConnectorQuantitySyntax,
    flux: &eqiora_lang::ConnectorQuantitySyntax,
    shape: &ValueShapeSyntax,
    frame: FrameSyntax,
    pairing: BoundaryPairingSyntax,
) -> Result<PortContract, Vec<Diagnostic>> {
    let SpatialSupport::Boundary { dimensions, .. } = &support else {
        return Err(vec![source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            "field-physical Port `over` support must be an exact boundary",
        )]);
    };
    let dimensions = *dimensions;
    let shape = resolve_value_shape(file, range, shape, dimensions)?;
    let frame = resolve_frame(file, range, frame)?;
    if frame == ValueFrame::SpatialCartesian
        && (shape.is_scalar()
            || shape
                .extents()
                .iter()
                .any(|extent| usize::try_from(extent.get()).ok() != Some(dimensions)))
    {
        return Err(vec![source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            "spatial Connector shape must equal the exact support ambient dimension",
        )]);
    }
    let pairing = match pairing {
        BoundaryPairingSyntax::EuclideanBoundaryDuality => {
            BoundaryPairing::EuclideanBoundaryDuality
        }
        _ => {
            return Err(vec![source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                file,
                range,
                "boundary pairing is newer than this compiler",
            )]);
        }
    };
    let quantity_type = |dimension: &eqiora_lang::Expr| {
        eqiora_core::ValueType::shaped(
            eqiora_core::ScalarDomain::Real,
            lower_dimension(connector_definition.file, dimension).map_err(|e| vec![e])?,
            shape.clone(),
            frame,
        )
        .map_err(|error| {
            vec![source_error(
                codes::LANGUAGE_TYPE_ERROR,
                connector_definition.file,
                dimension.range(),
                error.to_string(),
            )]
        })
    };
    let connector = BoundaryPhysicalConnector::new(
        quantity_type(trace.dimension())?,
        quantity_type(flux.dimension())?,
        pairing,
    )
    .map_err(|violation| {
        vec![source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            format!("invalid field-physical Connector contract: {violation:?}"),
        )]
    })?;
    Ok(PortContract::BoundaryPhysical {
        quantities: PhysicalMemberNames::Boundary {
            trace: trace.name().to_owned(),
            flux: flux.name().to_owned(),
        },
        nominal: PhysicalNominal::BoundaryConnector {
            definition: DefinitionKey {
                namespace: connector_definition.namespace,
                name: connector_definition.declaration.name().to_owned(),
            },
            shape,
        },
        connector,
        support,
    })
}
