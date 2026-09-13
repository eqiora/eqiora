//! Canonical identity of one typed local Eqiora source unit.
//!
//! The identity is derived from parsed semantic syntax, never source bytes or
//! source spans. It seeds deterministic elaboration namespaces independently
//! of whitespace, file location, and declaration traversal order.

mod budget;
use budget::Budget;
mod initial;
use initial::encode_initial;
pub(crate) use initial::initial_declaration_name;

use core::fmt;
use std::collections::BTreeMap;

mod alias;
mod compile_time;
mod component_item;
use component_item::encode_component_item;
mod declarations;
use declarations::{encode_component, encode_connector, encode_pure_operators};
mod dimension;
mod domain;
pub(crate) mod formulation;
mod instance;
mod limits;
mod model;
mod property;
mod record;
mod signature;
mod value_type;
mod visibility;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{
    ActivationSyntax, BinaryOp, BoundaryConnectionDecl, BoundaryPairingSyntax,
    BoundaryPortReferenceSyntax, BoundaryPortSelectorSyntax, BoundarySideSyntax,
    CartesianCoordinateSyntax, ClockDecl, ComponentDecl, ComponentItem, ComponentParameterDecl,
    ComponentPortDecl, ComponentPortFamilyDecl, ConnectionDecl, ConnectionSyntax, ConnectorDecl,
    ConnectorSyntax, Document, DomainDecl, DomainSyntax, Expr, ExprKind, FamilyBinderSyntax,
    FieldDecl, FrameSyntax, Item, NamePath, ParameterDecl, PortDecl, PortSyntax, PureOperatorDecl,
    RelationDecl, RelationFamilyDecl, SignalDirectionSyntax, SupportSlotDecl, SupportSlotSyntax,
    UnaryOp, ValueShapeSyntax, VisibilitySyntax,
};
use sha2::{Digest, Sha256};

use crate::connection_sets::{
    ConnectionFragment, ConnectionSetError, ConnectionSetLimits, normalize_connection_sets,
};
use crate::identity::IdentityNamespace;
pub(crate) use alias::ResolvedAliasTarget;
use alias::encode_type_path;
use compile_time::{encode_let, encode_parameter};
use dimension::encode_dimensions;
use domain::encode_domain;
use instance::encode_instance;
pub use limits::LocalSourceIdentityLimits;
use model::encode_model;
use property::{encode_material_composition, encode_property_contract, encode_property_release};
use visibility::encode_visibility;

const MAGIC: &[u8; 8] = b"EQIORASU";
const CANONICAL_VERSION: u16 = 18;
const COMPONENT_CONNECTION_ITEM_TAG: u16 = 6;
const MODEL_CONNECTION_ITEM_TAG: u16 = 8;
const COMPONENT_PORT_FAMILY_ITEM_TAG: u16 = 11;
const COMPONENT_RELATION_FAMILY_ITEM_TAG: u16 = 12;
const COMPONENT_BOUNDARY_CONNECTION_ITEM_TAG: u16 = 13;
const MODEL_BOUNDARY_CONNECTION_ITEM_TAG: u16 = 11;
const COMPONENT_SPATIAL_PERIODIC_CONNECTION_ITEM_TAG: u16 = 14;
const MODEL_SPATIAL_PERIODIC_CONNECTION_ITEM_TAG: u16 = 12;
const MODEL_LET_ITEM_TAG: u16 = 13;
/// Domain-separated SHA-256 identity of one typed local source unit.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalSourceIdentity([u8; 32]);

impl LocalSourceIdentity {
    /// Compute an identity with the default bounded resource policy.
    pub fn from_document(document: &Document) -> Result<Self, Diagnostic> {
        Self::from_document_with_limits(document, LocalSourceIdentityLimits::default())
    }

    /// Compute an identity with explicit compiler resource limits.
    pub fn from_document_with_limits(
        document: &Document,
        limits: LocalSourceIdentityLimits,
    ) -> Result<Self, Diagnostic> {
        let canonical = canonical_source_bytes(document, limits)?;
        Ok(Self(Sha256::digest(canonical).into()))
    }

    /// Exact SHA-256 bytes.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert this source-unit identity into a reserved deterministic
    /// namespace seed for local elaboration.
    pub fn namespace(&self) -> Result<IdentityNamespace, Diagnostic> {
        let mut digest = String::new();
        digest
            .try_reserve_exact(64)
            .map_err(|_| source_identity_error("cannot reserve source identity namespace"))?;
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for byte in self.0 {
            digest.push(char::from(HEX[usize::from(byte >> 4)]));
            digest.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        IdentityNamespace::new([format!("local-source-v{CANONICAL_VERSION}"), digest])
    }
}

impl fmt::Debug for LocalSourceIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "LocalSourceIdentity({self})")
    }
}

impl fmt::Display for LocalSourceIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

pub(crate) fn module_input_bytes(module: &eqiora_lang::Module) -> Result<usize, Diagnostic> {
    let structural =
        canonical_source_bytes(module.document(), LocalSourceIdentityLimits::default())?.len();
    let docs = module
        .document()
        .doc_comments()
        .map(|(_, doc)| doc.text().len());
    let notations = module
        .document()
        .notations()
        .map(|(_, notation)| notation.canonical().len());
    let total = docs.chain(notations).try_fold(structural, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| source_identity_error("module input byte count overflows"))
    })?;
    Ok(total.max(module.source_bytes().unwrap_or(0)))
}

pub(crate) fn canonical_source_bytes(
    document: &Document,
    limits: LocalSourceIdentityLimits,
) -> Result<Vec<u8>, Diagnostic> {
    canonical_source_bytes_with_aliases(document, limits, BTreeMap::new(), BTreeMap::new())
}

fn canonical_source_bytes_with_aliases(
    document: &Document,
    limits: LocalSourceIdentityLimits,
    resolved_aliases: BTreeMap<String, ResolvedAliasTarget>,
    operator_formals: BTreeMap<String, Vec<String>>,
) -> Result<Vec<u8>, Diagnostic> {
    let top_level_count = document
        .dimensions()
        .len()
        .checked_add(document.property_contract_syntax().len())
        .and_then(|count| count.checked_add(document.property_release_syntax().len()))
        .and_then(|count| count.checked_add(document.material_composition_syntax().len()))
        .and_then(|count| count.checked_add(document.connectors().len()))
        .and_then(|count| count.checked_add(document.pure_operators().len()))
        .and_then(|count| count.checked_add(document.enumerations().len()))
        .and_then(|count| count.checked_add(document.records().len()))
        .and_then(|count| count.checked_add(document.components().len()))
        .and_then(|count| count.checked_add(document.models().len()))
        .ok_or_else(|| source_identity_error("top-level declaration count overflows usize"))?;
    if top_level_count > limits.max_top_level_declarations {
        return Err(source_identity_error(format!(
            "source unit has {top_level_count} top-level declarations, exceeding the {} declaration limit",
            limits.max_top_level_declarations
        )));
    }

    let mut budget = Budget::with_resolved_aliases(limits, resolved_aliases);
    budget.operator_formals = operator_formals;
    budget
        .operator_formals
        .extend(document.pure_operators().iter().map(|operator| {
            (
                operator.name().to_owned(),
                operator
                    .formals()
                    .iter()
                    .map(|formal| formal.name().to_owned())
                    .collect(),
            )
        }));
    let dimensions = encode_dimensions(document.dimensions().iter(), &mut budget)?;
    let connectors = encode_sorted_records(document.connectors(), &mut budget, encode_connector)?;
    let property_contract_syntax = document
        .property_contract_syntax()
        .zip(document.property_contract_profiles())
        .collect::<Vec<_>>();
    let property_release_syntax = document
        .property_release_syntax()
        .zip(document.property_release_profiles())
        .collect::<Vec<_>>();
    let material_composition_syntax = document.material_composition_syntax().collect::<Vec<_>>();
    let property_contracts = encode_sorted_records(
        &property_contract_syntax,
        &mut budget,
        |(syntax, profile), budget| encode_property_contract(syntax, profile, budget),
    )?;
    let property_releases = encode_sorted_records(
        &property_release_syntax,
        &mut budget,
        |(syntax, profile), budget| encode_property_release(syntax, profile, budget),
    )?;
    let material_compositions = encode_sorted_records(
        &material_composition_syntax,
        &mut budget,
        encode_material_composition,
    )?;
    let pure_operators = encode_pure_operators(document, &mut budget)?;
    let components = encode_sorted_records(document.components(), &mut budget, encode_component)?;
    let models = encode_sorted_records(document.models(), &mut budget, encode_model)?;

    let finite_spaces = encode_sorted_records(
        document.finite_spaces(),
        &mut budget,
        |declaration, budget| {
            let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
            encoder.u8(
                if declaration.visibility() == eqiora_lang::VisibilitySyntax::Public {
                    1
                } else {
                    0
                },
            )?;
            encode_let(&mut encoder, declaration, budget)?;
            encoder.finish()
        },
    )?;
    let enumerations = encode_sorted_records(
        document.enumerations(),
        &mut budget,
        declarations::encode_enumeration,
    )?;
    let records = encode_sorted_records(document.records(), &mut budget, record::encode_record)?;
    let mut encoder = Encoder::new(limits.max_canonical_bytes);
    encoder.raw(MAGIC)?;
    encoder.u16(CANONICAL_VERSION)?;
    encoder.field(1, |encoder| encoder.records(&connectors))?;
    encoder.field(2, |encoder| encoder.records(&components))?;
    encoder.field(3, |encoder| encoder.records(&models))?;
    if !pure_operators.is_empty() {
        encoder.field(4, |encoder| encoder.records(&pure_operators))?;
    }
    if !property_contracts.is_empty() {
        encoder.field(5, |encoder| encoder.records(&property_contracts))?;
    }
    if !property_releases.is_empty() {
        encoder.field(6, |encoder| encoder.records(&property_releases))?;
    }
    if !dimensions.is_empty() {
        encoder.field(7, |encoder| encoder.records(&dimensions))?;
    }
    if !material_compositions.is_empty() {
        encoder.field(8, |encoder| encoder.records(&material_compositions))?;
    }
    if !finite_spaces.is_empty() {
        encoder.field(9, |encoder| encoder.records(&finite_spaces))?;
    }
    if !enumerations.is_empty() {
        encoder.field(10, |encoder| encoder.records(&enumerations))?;
    }
    if !records.is_empty() {
        encoder.field(11, |encoder| encoder.records(&records))?;
    }
    encoder.finish()
}

fn component_connection(item: &ComponentItem) -> Option<&ConnectionDecl> {
    match item {
        ComponentItem::Connection(declaration) => Some(declaration),
        _ => None,
    }
}

/// Encode one definition body after replacing its conserving source
/// fragments with their maximal structural connection sets.
///
/// A disjoint fragment encodes exactly as it did before normalization. Signal
/// declarations stay on the ordinary item path because their output is
/// positional and they do not form an undirected equivalence relation.
fn encode_container_records<T>(
    values: &[T],
    budget: &mut Budget,
    connection_of: for<'a> fn(&'a T) -> Option<&'a ConnectionDecl>,
    encode_item: fn(&T, &mut Budget) -> Result<Vec<u8>, Diagnostic>,
    connection_item_tag: u16,
) -> Result<Vec<Vec<u8>>, Diagnostic> {
    let limits = budget.connection_set_limits();
    let mut fragment_count = 0_usize;
    let mut membership_count = 0_usize;
    for value in values {
        let Some(connection) = connection_of(value) else {
            continue;
        };
        if connection.syntax() != ConnectionSyntax::Conserving || connection.binder().is_some() {
            continue;
        }
        let members = connection.port_expressions().len();
        budget.check_connection_members(members, "Connection")?;
        check_connection_set_limit(
            "members in one connection fragment",
            members,
            limits.max_members_per_fragment,
        )?;
        fragment_count = fragment_count.checked_add(1).ok_or_else(|| {
            connection_set_identity_error(
                "preflight",
                ConnectionSetError::CountOverflow {
                    resource: "connection fragments",
                },
            )
        })?;
        membership_count = membership_count.checked_add(members).ok_or_else(|| {
            connection_set_identity_error(
                "preflight",
                ConnectionSetError::CountOverflow {
                    resource: "connection fragment memberships",
                },
            )
        })?;
    }
    check_connection_set_limit("connection fragments", fragment_count, limits.max_fragments)?;
    check_connection_set_limit(
        "connection fragment memberships",
        membership_count,
        limits.max_memberships,
    )?;

    let mut records = Vec::new();
    records
        .try_reserve_exact(values.len())
        .map_err(|_| source_identity_error("cannot reserve canonical source records"))?;
    let mut fragments = Vec::new();
    fragments
        .try_reserve_exact(fragment_count)
        .map_err(|_| source_identity_error("cannot reserve conserving connection fragments"))?;

    for value in values {
        if let Some(connection) = connection_of(value)
            && connection.syntax() == ConnectionSyntax::Conserving
            && connection.binder().is_none()
        {
            fragments.push(encode_conserving_fragment(connection, budget, limits)?);
            continue;
        }
        let record = encode_item(value, budget)?;
        budget.account_materialized_bytes(record.len())?;
        records.push(record);
    }

    let normalized = normalize_connection_sets(&fragments, limits)
        .map_err(|error| connection_set_identity_error("normalize", error))?;
    for set in normalized.sets() {
        let record = encode_conserving_connection_item(
            connection_item_tag,
            set.members(),
            budget.limits.max_canonical_bytes,
        )?;
        budget.account_materialized_bytes(record.len())?;
        records.push(record);
    }
    records.sort_unstable();
    Ok(records)
}

fn check_connection_set_limit(
    resource: &'static str,
    observed: usize,
    limit: usize,
) -> Result<(), Diagnostic> {
    if observed > limit {
        Err(connection_set_identity_error(
            "preflight",
            ConnectionSetError::LimitExceeded {
                resource,
                observed,
                limit,
            },
        ))
    } else {
        Ok(())
    }
}

fn encode_conserving_fragment(
    declaration: &ConnectionDecl,
    budget: &mut Budget,
    limits: ConnectionSetLimits,
) -> Result<ConnectionFragment<Vec<u8>>, Diagnostic> {
    budget.check_connection_members(declaration.port_expressions().len(), "Connection")?;
    let paths = encode_sorted_endpoints(declaration.port_expressions(), budget)?;
    ConnectionFragment::try_new(paths, limits)
        .map_err(|error| connection_set_identity_error("encode", error))
}

fn encode_conserving_connection_item(
    item_tag: u16,
    canonical_paths: &[Vec<u8>],
    max_canonical_bytes: usize,
) -> Result<Vec<u8>, Diagnostic> {
    let mut encoder = Encoder::new(max_canonical_bytes);
    encoder.u16(item_tag)?;
    encoder.field(1, |encoder| encoder.u8(2))?;
    encoder.field(2, |encoder| encoder.records(canonical_paths))?;
    encoder.finish()
}

fn connection_set_identity_error(operation: &'static str, error: ConnectionSetError) -> Diagnostic {
    source_identity_error(format!(
        "cannot {operation} conserving source connection sets: {error}"
    ))
}

fn encode_model_item(item: &Item, budget: &mut Budget) -> Result<Vec<u8>, Diagnostic> {
    let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
    match item {
        Item::Domain(declaration) => {
            encoder.u16(1)?;
            encode_domain(&mut encoder, declaration, budget)?;
        }
        Item::Initial(declaration) => {
            encoder.u16(14)?;
            encode_initial(&mut encoder, declaration, budget)?;
        }
        Item::Field(declaration) => {
            encoder.u16(3)?;
            encode_field(&mut encoder, declaration, budget)?;
        }
        Item::Observable(declaration) => {
            encoder.u16(30)?;
            compile_time::encode_observable(&mut encoder, declaration, budget)?;
        }
        Item::Parameter(declaration) => {
            encoder.u16(4)?;
            encode_parameter(&mut encoder, declaration, budget)?;
        }
        Item::IndexSet(declaration) => {
            encoder.u16(15)?;
            encode_let(&mut encoder, declaration, budget)?;
        }
        Item::Let(declaration) => {
            encoder.u16(MODEL_LET_ITEM_TAG)?;
            encode_let(&mut encoder, declaration, budget)?;
        }
        Item::Port(declaration) => {
            encoder.u16(5)?;
            encode_port(&mut encoder, declaration, budget)?;
        }
        Item::Event(declaration) => {
            encoder.u16(17)?;
            declarations::encode_event(&mut encoder, declaration, budget)?;
        }
        Item::Clock(declaration) => {
            encoder.u16(6)?;
            declarations::encode_clock(&mut encoder, declaration, budget)?;
        }
        Item::Relation(declaration) => {
            encoder.u16(7)?;
            encode_relation(&mut encoder, declaration, budget)?;
        }
        Item::RelationFamily(declaration) => {
            encoder.u16(16)?;
            encode_relation_family(&mut encoder, declaration, budget)?;
        }
        Item::Connection(declaration) => {
            encoder.u16(MODEL_CONNECTION_ITEM_TAG)?;
            encode_connection(&mut encoder, declaration, budget)?;
        }
        Item::BoundaryConnection(declaration) => {
            encoder.u16(match declaration.syntax() {
                ConnectionSyntax::Conserving => MODEL_BOUNDARY_CONNECTION_ITEM_TAG,
                ConnectionSyntax::SpatialPeriodic => MODEL_SPATIAL_PERIODIC_CONNECTION_ITEM_TAG,
                ConnectionSyntax::Signal => {
                    return Err(source_identity_error(
                        "boundary Connection cannot use signal semantics",
                    ));
                }
            })?;
            encode_boundary_connection(&mut encoder, declaration, budget)?;
        }
        Item::Instance(declaration) => {
            encoder.u16(10)?;
            encode_instance(&mut encoder, declaration, budget)?;
        }
        _ => {
            return Err(source_identity_error(
                "model item is newer than source identity v1",
            ));
        }
    }
    encoder.finish()
}

fn encode_component_parameter(
    encoder: &mut Encoder,
    declaration: &ComponentParameterDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| {
        encode_visibility(encoder, declaration.visibility())
    })?;
    encoder.field(2, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(3, |encoder| {
        value_type::encode_value_type(encoder, declaration.value_type(), budget, 1)
    })?;
    encoder.field(4, |encoder| match declaration.default() {
        Some(default) => {
            encoder.u8(1)?;
            encoder.field(1, |encoder| encode_expression(encoder, default, budget, 1))
        }
        None => encoder.u8(0),
    })
}

fn encode_component_port(
    encoder: &mut Encoder,
    declaration: &ComponentPortDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| {
        encode_visibility(encoder, declaration.visibility())
    })?;
    encoder.field(2, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(3, |encoder| {
        encode_port_syntax(encoder, declaration.syntax(), budget)
    })
}

fn encode_component_port_family(
    encoder: &mut Encoder,
    declaration: &ComponentPortFamilyDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| {
        encode_component_port(encoder, declaration.port(), budget)
    })?;
    encoder.field(2, |encoder| {
        encode_boundary_family_binder(encoder, declaration.binder(), budget)
    })
}

fn encode_support_slot(
    encoder: &mut Encoder,
    declaration: &SupportSlotDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| {
        encode_visibility(encoder, declaration.visibility())
    })?;
    encoder.field(2, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(3, |encoder| match declaration.syntax() {
        SupportSlotSyntax::Volume { ambient_dimension } => {
            encoder.u16(1)?;
            encoder.field(1, |encoder| {
                encoder.u64(u64::try_from(*ambient_dimension).map_err(|_| {
                    source_identity_error("support ambient dimension does not fit canonical u64")
                })?)
            })
        }
        SupportSlotSyntax::Boundary { parent } => {
            encoder.u16(2)?;
            encoder.field(1, |encoder| encode_name(encoder, parent, budget))
        }
        SupportSlotSyntax::CompleteExterior { parent } => {
            encoder.u16(3)?;
            encoder.field(1, |encoder| encode_name(encoder, parent, budget))
        }
        _ => Err(source_identity_error(
            "support slot syntax is newer than source identity v1",
        )),
    })
}

fn encode_activation(
    encoder: &mut Encoder,
    activation: &ActivationSyntax,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    match activation {
        ActivationSyntax::Continuous => encoder.u16(1),
        ActivationSyntax::Named(clock) => {
            encoder.u16(2)?;
            encode_name(encoder, clock, budget)
        }
        _ => Err(source_identity_error("unsupported activation")),
    }
}

fn encode_field(
    encoder: &mut Encoder,
    declaration: &FieldDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(2, |encoder| {
        encode_optional_name(encoder, declaration.domain(), budget)
    })?;
    encoder.field(3, |encoder| {
        encoder.u16(match declaration.role() {
            eqiora_lang::FieldRoleSyntax::Variable => 1,
            eqiora_lang::FieldRoleSyntax::State => 2,
        })
    })?;
    encoder.field(4, |encoder| {
        value_type::encode_value_type(encoder, declaration.value_type(), budget, 1)
    })?;
    encoder.field(5, |encoder| {
        encode_activation(encoder, declaration.activation(), budget)
    })?;
    Ok(())
}

fn encode_port(
    encoder: &mut Encoder,
    declaration: &PortDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(2, |encoder| {
        encode_port_syntax(encoder, declaration.syntax(), budget)
    })
}

fn encode_port_syntax(
    encoder: &mut Encoder,
    syntax: &PortSyntax,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    match syntax {
        PortSyntax::Signal {
            direction,
            value_type,
            domain,
            activation,
        } => {
            encoder.u16(1)?;
            encoder.field(1, |encoder| {
                encoder.u8(match direction {
                    SignalDirectionSyntax::Input => 1,
                    SignalDirectionSyntax::Output => 2,
                })
            })?;
            encoder.field(2, |encoder| {
                value_type::encode_value_type(encoder, value_type, budget, 1)
            })?;
            encoder.field(3, |encoder| {
                encode_optional_name(encoder, domain.as_deref(), budget)
            })?;
            encoder.field(4, |encoder| encode_activation(encoder, activation, budget))
        }
        PortSyntax::ScalarPhysical { domain } => {
            encoder.u16(3)?;
            encoder.field(1, |encoder| encode_name(encoder, domain, budget))
        }
        PortSyntax::ScalarPhysicalConnector { connector } => {
            encoder.u16(4)?;
            encoder.field(1, |encoder| encode_type_path(encoder, connector, budget))
        }
        PortSyntax::FieldPhysical { connector, support } => {
            encoder.u16(5)?;
            encoder.field(1, |encoder| encode_type_path(encoder, connector, budget))?;
            encoder.field(2, |encoder| encode_name(encoder, support, budget))
        }
        _ => Err(source_identity_error(
            "Port syntax is newer than source identity v1",
        )),
    }
}

fn encode_connector_quantity(
    encoder: &mut Encoder,
    quantity: &eqiora_lang::ConnectorQuantitySyntax,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| encode_name(encoder, quantity.name(), budget))?;
    encoder.field(2, |encoder| {
        encode_expression(encoder, quantity.dimension(), budget, 1)
    })
}

fn encode_value_shape(encoder: &mut Encoder, shape: &ValueShapeSyntax) -> Result<(), Diagnostic> {
    match shape {
        ValueShapeSyntax::Scalar => encoder.u16(1),
        ValueShapeSyntax::Exact(extents) if extents.is_empty() => encoder.u16(1),
        ValueShapeSyntax::Exact(extents) => {
            encoder.u16(2)?;
            encoder.u32(as_u32(extents.len(), "value-shape rank")?)?;
            for extent in extents {
                encoder.u32(*extent)?;
            }
            Ok(())
        }
        ValueShapeSyntax::SpatialVector => encoder.u16(3),
        _ => Err(source_identity_error(
            "value shape is newer than source identity v1",
        )),
    }
}

fn encode_frame(encoder: &mut Encoder, frame: FrameSyntax) -> Result<(), Diagnostic> {
    match frame {
        FrameSyntax::Invariant => encoder.u8(1),
        FrameSyntax::Spatial => encoder.u8(2),
        _ => Err(source_identity_error(
            "frame syntax is newer than source identity v1",
        )),
    }
}

fn encode_boundary_pairing(
    encoder: &mut Encoder,
    pairing: BoundaryPairingSyntax,
) -> Result<(), Diagnostic> {
    match pairing {
        BoundaryPairingSyntax::EuclideanBoundaryDuality => encoder.u8(1),
        _ => Err(source_identity_error(
            "boundary pairing is newer than source identity v1",
        )),
    }
}

fn encode_boundary_connection(
    encoder: &mut Encoder,
    declaration: &BoundaryConnectionDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    budget.check_connection_members(declaration.ports().len(), "boundary-family Connection")?;
    encoder.field(1, |encoder| match declaration.binder() {
        Some(binder) => {
            encoder.u8(1)?;
            encode_boundary_family_binder(encoder, binder, budget)
        }
        None => encoder.u8(0),
    })?;
    let ports = encode_sorted_records(declaration.ports(), budget, |port, budget| {
        let mut port_encoder = Encoder::new(budget.limits.max_canonical_bytes);
        encode_boundary_port_reference(&mut port_encoder, port, budget)?;
        port_encoder.finish()
    })?;
    encoder.field(2, |encoder| encoder.records(&ports))
}

fn encode_boundary_family_binder(
    encoder: &mut Encoder,
    binder: &FamilyBinderSyntax,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| encode_name(encoder, binder.member(), budget))?;
    encoder.field(2, |encoder| {
        encode_name(encoder, binder.set().as_str(), budget)
    })
}

fn encode_boundary_port_reference(
    encoder: &mut Encoder,
    reference: &BoundaryPortReferenceSyntax,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| encode_path(encoder, reference.port(), budget))?;
    if let Some(selector) = reference.selector() {
        encoder.field(2, |encoder| {
            encode_boundary_port_selector(encoder, selector, budget)
        })?;
    }
    Ok(())
}

fn encode_boundary_port_selector(
    encoder: &mut Encoder,
    selector: &BoundaryPortSelectorSyntax,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| encode_name(encoder, selector.member(), budget))?;
    encoder.field(2, |encoder| encode_name(encoder, selector.target(), budget))
}

fn encode_optional_name(
    encoder: &mut Encoder,
    name: Option<&str>,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    match name {
        Some(name) => {
            encoder.u8(1)?;
            encode_name(encoder, name, budget)
        }
        None => encoder.u8(0),
    }
}

fn encode_name(encoder: &mut Encoder, name: &str, budget: &mut Budget) -> Result<(), Diagnostic> {
    budget.account_name(name)?;
    encoder.string(name)
}

fn encode_path(
    encoder: &mut Encoder,
    path: &NamePath,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    let segment_count = path.segments().len();
    if segment_count > budget.limits.max_path_segments {
        return Err(source_identity_error(format!(
            "source path has {segment_count} segments, exceeding the {} segment limit",
            budget.limits.max_path_segments
        )));
    }
    encoder.u32(as_u32(segment_count, "source path segment count")?)?;
    for segment in path.segments() {
        encode_name(encoder, segment, budget)?;
    }
    Ok(())
}

mod expression;
use expression::{encode_expression, encode_relation, encode_relation_family};
mod connection;
use connection::encode_connection;

fn encode_sorted_endpoints(
    values: &[Expr],
    budget: &mut Budget,
) -> Result<Vec<Vec<u8>>, Diagnostic> {
    encode_sorted_records(values, budget, |value, budget| {
        let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
        encode_expression(&mut encoder, value, budget, 0)?;
        encoder.finish()
    })
}

fn encode_sorted_records<T>(
    values: &[T],
    budget: &mut Budget,
    mut encode: impl FnMut(&T, &mut Budget) -> Result<Vec<u8>, Diagnostic>,
) -> Result<Vec<Vec<u8>>, Diagnostic> {
    let mut records = Vec::new();
    records
        .try_reserve_exact(values.len())
        .map_err(|_| source_identity_error("cannot reserve canonical source records"))?;
    for value in values {
        let record = encode(value, budget)?;
        budget.account_materialized_bytes(record.len())?;
        records.push(record);
    }
    records.sort_unstable();
    Ok(records)
}

struct Encoder {
    bytes: Vec<u8>,
    max_bytes: usize,
}

impl Encoder {
    const fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::new(),
            max_bytes,
        }
    }

    fn finish(self) -> Result<Vec<u8>, Diagnostic> {
        if self.bytes.len() > self.max_bytes {
            return Err(source_identity_error(
                "canonical source encoding exceeded its byte limit",
            ));
        }
        Ok(self.bytes)
    }

    fn field(
        &mut self,
        tag: u8,
        encode: impl FnOnce(&mut Self) -> Result<(), Diagnostic>,
    ) -> Result<(), Diagnostic> {
        self.u8(tag)?;
        let length_offset = self.bytes.len();
        self.raw(&[0; 4])?;
        let payload_offset = self.bytes.len();
        encode(self)?;
        let payload_len = self
            .bytes
            .len()
            .checked_sub(payload_offset)
            .ok_or_else(|| source_identity_error("canonical field length underflow"))?;
        self.bytes[length_offset..payload_offset]
            .copy_from_slice(&as_u32(payload_len, "canonical field length")?.to_be_bytes());
        Ok(())
    }

    fn records(&mut self, records: &[Vec<u8>]) -> Result<(), Diagnostic> {
        self.u32(as_u32(records.len(), "canonical record count")?)?;
        for record in records {
            self.u32(as_u32(record.len(), "canonical record length")?)?;
            self.raw(record)?;
        }
        Ok(())
    }

    fn string(&mut self, value: &str) -> Result<(), Diagnostic> {
        self.u32(as_u32(value.len(), "canonical string length")?)?;
        self.raw(value.as_bytes())
    }

    fn f64(&mut self, value: f64) -> Result<(), Diagnostic> {
        if !value.is_finite() {
            return Err(source_identity_error(
                "source identity accepts only finite f64 values",
            ));
        }
        let canonical = if value == 0.0 { 0.0 } else { value };
        self.u64(canonical.to_bits())
    }

    fn u8(&mut self, value: u8) -> Result<(), Diagnostic> {
        self.raw(&[value])
    }

    fn u16(&mut self, value: u16) -> Result<(), Diagnostic> {
        self.raw(&value.to_be_bytes())
    }

    fn u32(&mut self, value: u32) -> Result<(), Diagnostic> {
        self.raw(&value.to_be_bytes())
    }

    fn u64(&mut self, value: u64) -> Result<(), Diagnostic> {
        self.raw(&value.to_be_bytes())
    }

    fn raw(&mut self, value: &[u8]) -> Result<(), Diagnostic> {
        let next_len = self
            .bytes
            .len()
            .checked_add(value.len())
            .ok_or_else(|| source_identity_error("canonical source bytes overflow usize"))?;
        if next_len > self.max_bytes {
            return Err(source_identity_error(format!(
                "canonical source encoding exceeds the {} byte limit",
                self.max_bytes
            )));
        }
        self.bytes
            .try_reserve_exact(value.len())
            .map_err(|_| source_identity_error("cannot reserve canonical source bytes"))?;
        self.bytes.extend_from_slice(value);
        Ok(())
    }
}

fn next_depth(depth: usize) -> Result<usize, Diagnostic> {
    depth
        .checked_add(1)
        .ok_or_else(|| source_identity_error("expression depth overflows usize"))
}

fn as_u32(value: usize, label: &'static str) -> Result<u32, Diagnostic> {
    u32::try_from(value).map_err(|_| source_identity_error(format!("{label} exceeds u32")))
}

fn source_identity_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::LANGUAGE_LOWERING_ERROR, message)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod exact_clock_tests;
