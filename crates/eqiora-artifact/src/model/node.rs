//! One Kernel node on the current Model wire.

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_schema::kernel::{
    ActivationDef, BoundaryPhysicalConnector, ConnectionDef, DomainDef, DomainKind, EnumDef,
    FieldDef, FiniteSpaceDef, GeometryDigest, IndexSetDef, KernelNode, ObservableDef,
    ObservableMeasure, ObservableReduction, ParameterDef, PortDef, PortPayload, RecordDef,
    RecordInstanceDef, RepresentationDef,
};
use serde::{Deserialize, Serialize};

use crate::{ArtifactDigest, invalid_artifact};

use super::relation::WireRelationMeaning;
use super::value_type::{WireFiniteFactor, WireValueType};
use super::*;
use super::{expression::*, primitive::*, vocabulary::*};

impl WireNode {
    pub(crate) fn encode(node: &KernelNode) -> Result<Self, Diagnostic> {
        let definition = match node {
            KernelNode::Record(value) => WireNodeDefinition::Record {
                members: value
                    .members()
                    .iter()
                    .map(|(name, ty)| Ok((name.clone(), WireValueType::encode(ty)?)))
                    .collect::<Result<_, Diagnostic>>()?,
            },
            KernelNode::RecordInstance(value) => WireNodeDefinition::RecordInstance {
                definition: WireId::from_raw(value.definition().erase()),
                expression: WireExpression::encode(value.expression())?,
            },
            KernelNode::Observable(value) => WireNodeDefinition::Observable {
                value_type: WireValueType::encode(value.value_type())?,
                expression: WireExpression::encode(value.expression())?,
                reduction: WireObservableReduction::encode(value.reduction()),
            },
            KernelNode::Enum(value) => WireNodeDefinition::Enum {
                members: value.members().to_vec(),
            },
            KernelNode::FiniteSpace(value) => match value.factors() {
                Some(factors) => WireNodeDefinition::FiniteProduct {
                    factors: factors.map(|basis| WireFiniteFactor {
                        space: WireId::from_raw(basis.space().expect("atomic factor").erase()),
                        extent: basis.extent(),
                    }),
                },
                None => WireNodeDefinition::FiniteSpace {
                    labels: value.labels().expect("atomic labels").to_vec(),
                },
            },
            KernelNode::IndexSet(value) => WireNodeDefinition::IndexSet {
                extent: value.extent(),
            },
            KernelNode::Domain(value) => WireNodeDefinition::Domain {
                domain: WireDomainKind::encode(value.kind())?,
            },
            KernelNode::Representation(value) => WireNodeDefinition::Representation {
                representation: WireRepresentationKind::encode(value.kind())?,
            },
            KernelNode::Field(value) => WireNodeDefinition::Field {
                value_type: WireValueType::encode(value.value_type())?,
                role: WireFieldRole::encode(value.role()),
            },
            KernelNode::Parameter(value) => WireNodeDefinition::Parameter {
                value: WireValueLiteral::encode(value.value())?,
            },
            KernelNode::Port(value) => match value.payload() {
                PortPayload::ScalarPhysical { domain } => WireNodeDefinition::ScalarPhysicalPort {
                    domain: WireId::from_raw(domain.erase()),
                },
                PortPayload::BoundaryPhysical {
                    connector,
                    boundary,
                } => WireNodeDefinition::BoundaryPhysicalPort {
                    connector: WireId::from_raw(connector.erase()),
                    boundary: WireId::from_raw(boundary.erase()),
                },
                PortPayload::Signal {
                    direction,
                    value_type,
                } => WireNodeDefinition::SignalPort {
                    direction: WireSignalDirection::encode(direction),
                    value_type: WireValueType::encode(&value_type)?,
                },
                _ => return Err(invalid_artifact("unsupported Port payload")),
            },
            KernelNode::Relation(value) => WireNodeDefinition::Relation {
                expression: WireExpression::encode(value.expression())?,
                initial: value.is_initial(),
                meaning: WireRelationMeaning::encode(value.meaning()),
            },
            KernelNode::Activation(value) => WireNodeDefinition::Activation {
                activation: WireActivationKind::encode(value.kind())?,
            },
            KernelNode::Connection(value) => WireNodeDefinition::Connection {
                connection: WireConnectionKind::encode(value.semantics())?,
            },
            KernelNode::ClockDomain(value) => WireNodeDefinition::ClockDomain {
                clock: WireClockKind::encode(value.kind())?,
            },
            _ => {
                return Err(invalid_artifact(
                    "kernel node variant is newer than the current Model wire",
                ));
            }
        };
        Ok(Self {
            id: WireId::from_raw(node.id()),
            definition,
        })
    }

    pub(crate) fn decode(&self) -> Result<KernelNode, Diagnostic> {
        match &self.definition {
            WireNodeDefinition::Record { members } => RecordDef::new(
                self.id.typed::<kinds::Record>()?,
                members
                    .iter()
                    .map(|(name, ty)| Ok((name.clone(), ty.decode()?)))
                    .collect::<Result<_, Diagnostic>>()?,
            )
            .map(Into::into)
            .map_err(|error| invalid_artifact(error.message())),
            WireNodeDefinition::RecordInstance {
                definition,
                expression,
            } => RecordInstanceDef::new(
                self.id.typed::<kinds::RecordInstance>()?,
                definition.typed::<kinds::Record>()?,
                expression.decode()?,
            )
            .map(Into::into)
            .map_err(|error| invalid_artifact(error.message())),
            WireNodeDefinition::Observable {
                value_type,
                expression,
                reduction,
            } => {
                let expression = expression.decode()?;
                let reduction = reduction.decode(&expression)?;
                ObservableDef::new(
                    self.id.typed::<kinds::Observable>()?,
                    value_type.decode()?,
                    expression,
                    reduction,
                )
                .map(Into::into)
            }
            WireNodeDefinition::Enum { members } => {
                EnumDef::new(self.id.typed::<kinds::Enum>()?, members.iter().cloned())
                    .map(Into::into)
                    .map_err(|error| invalid_artifact(error.to_string()))
            }
            WireNodeDefinition::FiniteProduct {
                factors: [left, right],
            } => FiniteSpaceDef::product(
                self.id.typed::<kinds::FiniteSpace>()?,
                eqiora_core::FiniteBasis::new(
                    left.space.typed::<kinds::FiniteSpace>()?,
                    left.extent,
                )
                .map_err(|error| invalid_artifact(error.to_string()))?,
                eqiora_core::FiniteBasis::new(
                    right.space.typed::<kinds::FiniteSpace>()?,
                    right.extent,
                )
                .map_err(|error| invalid_artifact(error.to_string()))?,
            )
            .map(Into::into)
            .map_err(|error| invalid_artifact(error.to_string())),
            WireNodeDefinition::FiniteSpace { labels } => FiniteSpaceDef::new(
                self.id.typed::<kinds::FiniteSpace>()?,
                labels.iter().cloned(),
            )
            .map(Into::into)
            .map_err(|error| invalid_artifact(error.to_string())),
            WireNodeDefinition::IndexSet { extent } => {
                IndexSetDef::new(self.id.typed::<kinds::IndexSet>()?, *extent)
                    .map(Into::into)
                    .map_err(|error| invalid_artifact(error.to_string()))
            }
            WireNodeDefinition::Domain { domain } => {
                let id = self.id.typed::<kinds::Domain>()?;
                Ok(domain.decode(id)?.into())
            }
            WireNodeDefinition::Representation { representation } => {
                let id = self.id.typed::<kinds::Representation>()?;
                Ok(match representation {
                    WireRepresentationKind::Abstract => RepresentationDef::new(id),
                    WireRepresentationKind::Continuum => RepresentationDef::continuum(id),
                }
                .into())
            }
            WireNodeDefinition::Field { value_type, role } => Ok(FieldDef::new(
                self.id.typed::<kinds::Field>()?,
                value_type.decode()?,
                role.decode(),
            )
            .into()),
            WireNodeDefinition::Parameter { value } => {
                Ok(ParameterDef::new(self.id.typed::<kinds::Parameter>()?, value.decode()?).into())
            }
            WireNodeDefinition::SignalPort {
                direction,
                value_type,
            } => Ok(PortDef::signal(
                self.id.typed::<kinds::Port>()?,
                direction.decode(),
                value_type.decode()?,
            )
            .into()),
            WireNodeDefinition::ScalarPhysicalPort { domain } => Ok(PortDef::scalar_physical(
                self.id.typed::<kinds::Port>()?,
                domain.typed::<kinds::Domain>()?,
            )
            .into()),
            WireNodeDefinition::BoundaryPhysicalPort {
                connector,
                boundary,
            } => Ok(PortDef::boundary_physical(
                self.id.typed::<kinds::Port>()?,
                connector.typed::<kinds::Domain>()?,
                boundary.typed::<kinds::Domain>()?,
            )
            .into()),
            WireNodeDefinition::Relation {
                expression,
                initial,
                meaning,
            } => {
                let id = self.id.typed::<kinds::Relation>()?;
                let expression = expression.decode()?;
                Ok(meaning.decode(id, expression, *initial)?.into())
            }
            WireNodeDefinition::Activation { activation } => Ok(ActivationDef::new(
                self.id.typed::<kinds::Activation>()?,
                activation.decode()?,
            )
            .map_err(|error| invalid_artifact(error.message()))?
            .into()),
            WireNodeDefinition::Connection { connection } => Ok(ConnectionDef::new(
                self.id.typed::<kinds::Connection>()?,
                connection.decode()?,
            )
            .into()),
            WireNodeDefinition::ClockDomain { clock } => {
                Ok(clock.decode(self.id.typed::<kinds::ClockDomain>()?)?.into())
            }
        }
    }

    pub(crate) fn expression_node_count(&self) -> usize {
        match &self.definition {
            WireNodeDefinition::Observable { expression, .. }
            | WireNodeDefinition::Relation { expression, .. }
            | WireNodeDefinition::RecordInstance { expression, .. } => expression.nodes.len(),
            WireNodeDefinition::Activation { activation } => activation.expression_node_count(),
            _ => 0,
        }
    }

    pub(crate) fn expression_root_count(&self) -> usize {
        match &self.definition {
            WireNodeDefinition::Observable { expression, .. }
            | WireNodeDefinition::Relation { expression, .. }
            | WireNodeDefinition::RecordInstance { expression, .. } => expression.roots.len(),
            WireNodeDefinition::Activation { activation } => activation.expression_root_count(),
            _ => 0,
        }
    }

    pub(crate) fn pure_operator_counts(&self) -> Result<PureOperatorWireCounts, Diagnostic> {
        match &self.definition {
            WireNodeDefinition::Observable { expression, .. }
            | WireNodeDefinition::Relation { expression, .. }
            | WireNodeDefinition::RecordInstance { expression, .. } => {
                expression.pure_operator_counts()
            }
            WireNodeDefinition::Activation { activation } => activation.pure_operator_counts(),
            _ => Ok(PureOperatorWireCounts::default()),
        }
    }

    pub(crate) fn validate_pure_operator_features(&self) -> Result<(), Diagnostic> {
        match &self.definition {
            WireNodeDefinition::Observable { expression, .. }
            | WireNodeDefinition::Relation { expression, .. }
            | WireNodeDefinition::RecordInstance { expression, .. } => {
                expression.validate_pure_operator_features()
            }
            WireNodeDefinition::Activation { activation } => {
                activation.validate_pure_operator_features()
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn canonicalize_pure_operator_definitions(&mut self) -> Result<(), Diagnostic> {
        match &mut self.definition {
            WireNodeDefinition::Observable { expression, .. }
            | WireNodeDefinition::Relation { expression, .. }
            | WireNodeDefinition::RecordInstance { expression, .. } => {
                expression.canonicalize_pure_operator_definitions()
            }
            WireNodeDefinition::Activation { activation } => {
                activation.canonicalize_pure_operator_definitions()
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn literal_component_count(&self) -> Result<usize, Diagnostic> {
        match &self.definition {
            WireNodeDefinition::Parameter { value } => Ok(value.component_payload_count()),
            WireNodeDefinition::Observable { expression, .. }
            | WireNodeDefinition::Relation { expression, .. }
            | WireNodeDefinition::RecordInstance { expression, .. } => {
                expression.literal_component_count()
            }
            WireNodeDefinition::Activation { activation } => activation.literal_component_count(),
            _ => Ok(0),
        }
    }

    pub(crate) fn ensure_value_shape_limits(
        &self,
        limits: ModelDecoderLimits,
    ) -> Result<(), Diagnostic> {
        match &self.definition {
            WireNodeDefinition::Record { members } => {
                require_decoder_count(
                    "record members",
                    members.len(),
                    limits.max_value_shape_components,
                )?;
                for (_, ty) in members {
                    ty.ensure_limits(limits)?;
                }
                Ok(())
            }
            WireNodeDefinition::Domain {
                domain: WireDomainKind::CoordinateProduct { factors },
            } => require_decoder_count("coordinate factors", factors.len(), limits.max_nodes),
            WireNodeDefinition::Enum { members: labels }
            | WireNodeDefinition::FiniteSpace { labels } => require_decoder_count(
                "nominal declaration members",
                labels.len(),
                limits.max_value_shape_components,
            ),
            WireNodeDefinition::Field { value_type, .. }
            | WireNodeDefinition::SignalPort { value_type, .. } => value_type.ensure_limits(limits),
            WireNodeDefinition::Parameter { value } => value.ensure_limits(limits),
            WireNodeDefinition::Observable {
                value_type,
                expression,
                ..
            } => {
                value_type.ensure_limits(limits)?;
                expression.ensure_value_shape_limits(limits)
            }
            WireNodeDefinition::Relation { expression, .. }
            | WireNodeDefinition::RecordInstance { expression, .. } => {
                expression.ensure_value_shape_limits(limits)
            }
            WireNodeDefinition::Activation { activation } => {
                activation.ensure_value_shape_limits(limits)
            }
            WireNodeDefinition::Domain {
                domain:
                    WireDomainKind::ScalarPhysical {
                        across_type,
                        through_type,
                    },
            } => {
                across_type.ensure_limits(limits)?;
                through_type.ensure_limits(limits)
            }
            WireNodeDefinition::Domain {
                domain:
                    WireDomainKind::BoundaryPhysical {
                        trace_type,
                        flux_type,
                        ..
                    },
            } => {
                trace_type.ensure_limits(limits)?;
                flux_type.ensure_limits(limits)
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn semantic_references(&self) -> Vec<&WireId> {
        match &self.definition {
            WireNodeDefinition::Domain {
                domain: WireDomainKind::PhysicalInterface { boundaries },
            } => boundaries.iter().collect(),
            WireNodeDefinition::Domain {
                domain: WireDomainKind::CoordinateProduct { factors },
            } => factors.iter().collect(),
            WireNodeDefinition::FiniteProduct { factors } => {
                factors.iter().map(|factor| &factor.space).collect()
            }
            WireNodeDefinition::Record { members } => members
                .iter()
                .flat_map(|(_, ty)| ty.nominal_references())
                .collect(),
            WireNodeDefinition::RecordInstance {
                definition,
                expression,
            } => std::iter::once(definition)
                .chain(expression.semantic_references())
                .collect(),
            WireNodeDefinition::Field { value_type, .. }
            | WireNodeDefinition::SignalPort { value_type, .. } => {
                value_type.nominal_references().collect()
            }
            WireNodeDefinition::Parameter { value } => value.nominal_references().collect(),
            WireNodeDefinition::Domain {
                domain:
                    WireDomainKind::ScalarPhysical {
                        across_type,
                        through_type,
                    },
            } => across_type
                .nominal_references()
                .chain(through_type.nominal_references())
                .collect(),
            WireNodeDefinition::Domain {
                domain:
                    WireDomainKind::BoundaryPhysical {
                        trace_type,
                        flux_type,
                        ..
                    },
            } => trace_type
                .nominal_references()
                .chain(flux_type.nominal_references())
                .collect(),
            WireNodeDefinition::ScalarPhysicalPort { domain } => vec![domain],
            WireNodeDefinition::BoundaryPhysicalPort {
                connector,
                boundary,
            } => vec![connector, boundary],
            WireNodeDefinition::Observable {
                value_type,
                expression,
                reduction,
            } => {
                let mut references = expression.semantic_references();
                references.extend(value_type.nominal_references());
                references.extend(reduction.semantic_references());
                references
            }
            WireNodeDefinition::Relation { expression, .. } => expression.semantic_references(),
            WireNodeDefinition::Activation { activation } => activation.semantic_references(),
            WireNodeDefinition::Domain {
                domain: WireDomainKind::CartesianBoxSources { coordinates },
            } => coordinates
                .iter()
                .flat_map(WireCartesianAxisDefinition::semantic_references)
                .collect(),
            _ => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireNodeDefinition {
    Record {
        members: Vec<(String, WireValueType)>,
    },
    RecordInstance {
        definition: WireId,
        expression: WireExpression,
    },
    Observable {
        value_type: WireValueType,
        expression: WireExpression,
        reduction: WireObservableReduction,
    },
    Enum {
        members: Vec<String>,
    },
    FiniteProduct {
        factors: [WireFiniteFactor; 2],
    },
    FiniteSpace {
        labels: Vec<String>,
    },
    IndexSet {
        extent: u32,
    },
    Domain {
        domain: WireDomainKind,
    },
    Representation {
        representation: WireRepresentationKind,
    },
    Field {
        value_type: WireValueType,
        role: WireFieldRole,
    },
    Parameter {
        value: WireValueLiteral,
    },
    SignalPort {
        direction: WireSignalDirection,
        value_type: WireValueType,
    },
    ScalarPhysicalPort {
        domain: WireId,
    },
    BoundaryPhysicalPort {
        connector: WireId,
        boundary: WireId,
    },
    Relation {
        initial: bool,
        meaning: WireRelationMeaning,
        expression: WireExpression,
    },
    Activation {
        activation: WireActivationKind,
    },
    Connection {
        connection: WireConnectionKind,
    },
    ClockDomain {
        clock: WireClockKind,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireDomainKind {
    Abstract,
    CoordinateInterval {
        lower: WireQuantity,
        upper: WireQuantity,
    },
    CoordinateProduct {
        factors: Vec<WireId>,
    },
    CartesianBoxSources {
        coordinates: Vec<WireCartesianAxisDefinition>,
    },
    CartesianBoundary {
        axis: usize,
        side: WireBoundarySide,
    },
    GeometryRegion {
        geometry: String,
        entity_set: String,
    },
    GeometryBoundary {
        entity_set: String,
    },
    PhysicalInterface {
        boundaries: [WireId; 2],
    },
    ScalarPhysical {
        across_type: WireValueType,
        through_type: WireValueType,
    },
    BoundaryPhysical {
        trace_type: WireValueType,
        flux_type: WireValueType,
        pairing: WireBoundaryPairing,
    },
}

impl WireDomainKind {
    pub(crate) fn encode(value: &DomainKind) -> Result<Self, Diagnostic> {
        Ok(match value {
            DomainKind::Abstract => Self::Abstract,
            DomainKind::CoordinateInterval { bounds } => Self::CoordinateInterval {
                lower: WireQuantity::encode(bounds.lower()),
                upper: WireQuantity::encode(bounds.upper()),
            },
            DomainKind::CoordinateProduct { factors } => Self::CoordinateProduct {
                factors: factors
                    .iter()
                    .map(|id| WireId::from_raw(id.erase()))
                    .collect(),
            },
            DomainKind::CartesianBox { coordinates } => Self::CartesianBoxSources {
                coordinates: coordinates
                    .iter()
                    .copied()
                    .map(WireCartesianAxisDefinition::encode)
                    .collect(),
            },
            DomainKind::CartesianBoundary { axis, side } => Self::CartesianBoundary {
                axis: *axis,
                side: WireBoundarySide::encode(*side),
            },
            DomainKind::GeometryRegion {
                geometry,
                entity_set,
            } => Self::GeometryRegion {
                geometry: encode_geometry_digest(*geometry),
                entity_set: entity_set.clone(),
            },
            DomainKind::GeometryBoundary { entity_set } => Self::GeometryBoundary {
                entity_set: entity_set.clone(),
            },
            DomainKind::PhysicalInterface { boundaries } => Self::PhysicalInterface {
                boundaries: boundaries.map(|boundary| WireId::from_raw(boundary.erase())),
            },
            DomainKind::ScalarPhysical {
                across_type,
                through_type,
            } => Self::ScalarPhysical {
                across_type: WireValueType::encode(across_type)?,
                through_type: WireValueType::encode(through_type)?,
            },
            DomainKind::BoundaryPhysical { connector } => Self::BoundaryPhysical {
                trace_type: WireValueType::encode(connector.trace_type())?,
                flux_type: WireValueType::encode(connector.flux_type())?,
                pairing: WireBoundaryPairing::encode(connector.pairing()),
            },
            _ => {
                return Err(invalid_artifact(
                    "the model contains a Domain kind unsupported by this model wire",
                ));
            }
        })
    }

    pub(crate) fn decode(&self, id: Id<kinds::Domain>) -> Result<DomainDef, Diagnostic> {
        match self {
            Self::Abstract => Ok(DomainDef::new(id)),
            Self::CoordinateInterval { lower, upper } => {
                let bounds =
                    eqiora_schema::kernel::AxisBounds::new(lower.decode()?, upper.decode()?)
                        .map_err(|error| invalid_artifact(error.message()))?;
                Ok(DomainDef::coordinate_interval(id, bounds))
            }
            Self::CoordinateProduct { factors } => DomainDef::coordinate_product(
                id,
                factors
                    .iter()
                    .map(WireId::typed)
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .map_err(|error| invalid_artifact(error.message())),
            Self::CartesianBoxSources { coordinates } => DomainDef::cartesian_box_from_sources(
                id,
                coordinates
                    .iter()
                    .map(WireCartesianAxisDefinition::decode)
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .map_err(|error| invalid_artifact(error.message())),
            Self::CartesianBoundary { axis, side } => {
                Ok(DomainDef::cartesian_boundary(id, *axis, side.decode()))
            }
            Self::GeometryRegion {
                geometry,
                entity_set,
            } => DomainDef::geometry_region(id, decode_geometry_digest(geometry)?, entity_set)
                .map_err(|error| invalid_artifact(error.message())),
            Self::GeometryBoundary { entity_set } => DomainDef::geometry_boundary(id, entity_set)
                .map_err(|error| invalid_artifact(error.message())),
            Self::PhysicalInterface { boundaries } => {
                DomainDef::physical_interface(id, [boundaries[0].typed()?, boundaries[1].typed()?])
                    .map_err(|error| invalid_artifact(error.message()))
            }
            Self::ScalarPhysical {
                across_type,
                through_type,
            } => DomainDef::scalar_physical(id, across_type.decode()?, through_type.decode()?)
                .map_err(|error| invalid_artifact(error.message())),
            Self::BoundaryPhysical {
                trace_type,
                flux_type,
                pairing,
            } => Ok(DomainDef::boundary_physical(
                id,
                BoundaryPhysicalConnector::new(
                    trace_type.decode()?,
                    flux_type.decode()?,
                    pairing.decode(),
                )
                .map_err(|_| invalid_artifact("invalid boundary physical connector contract"))?,
            )),
        }
    }
}

/// Lowercase hex, matching how every other digest crosses this wire.
fn encode_geometry_digest(digest: GeometryDigest) -> String {
    ArtifactDigest::from_sha256(digest.bytes()).to_string()
}

/// A malformed digest is refused rather than truncated or padded, because a
/// Domain naming an unreadable geometry names nothing.
///
/// The shared digest type validates first, so a geometry reference is held to
/// the same canonical hex form as every other digest on this wire; the bytes
/// are then re-read because the Kernel holds bytes rather than text.
fn decode_geometry_digest(value: &str) -> Result<GeometryDigest, Diagnostic> {
    let canonical = ArtifactDigest::from_hex(value.to_owned())?.to_string();
    let mut bytes = [0_u8; 32];
    for (slot, pair) in bytes
        .iter_mut()
        .zip(canonical.as_bytes().as_chunks::<2>().0.iter())
    {
        let text = std::str::from_utf8(pair)
            .map_err(|_| invalid_artifact("geometry digest is not hexadecimal"))?;
        *slot = u8::from_str_radix(text, 16)
            .map_err(|_| invalid_artifact("geometry digest is not hexadecimal"))?;
    }
    Ok(GeometryDigest::new(bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireFieldRole {
    Variable,
    State,
}
impl WireFieldRole {
    fn encode(role: eqiora_schema::kernel::FieldRole) -> Self {
        match role {
            eqiora_schema::kernel::FieldRole::Variable => Self::Variable,
            eqiora_schema::kernel::FieldRole::State => Self::State,
        }
    }
    fn decode(self) -> eqiora_schema::kernel::FieldRole {
        match self {
            Self::Variable => eqiora_schema::kernel::FieldRole::Variable,
            Self::State => eqiora_schema::kernel::FieldRole::State,
        }
    }
}

#[cfg(test)]
mod equation_tests {
    use super::*;
    use eqiora_core::{DimExponents, DynQuantity};
    use eqiora_schema::kernel::{ExprDagBuilder, RelationDef};

    #[test]
    fn relation_wire_preserves_side_pairs_and_rejects_unpaired_roots() {
        for initial in [false, true] {
            let id = Id::new();
            let mut builder = ExprDagBuilder::new();
            let left = builder
                .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
                .unwrap();
            let right = builder
                .constant(DynQuantity::new(3.0, DimExponents::DIMENSIONLESS))
                .unwrap();
            let expression = builder.finish([left, right, right, left]).unwrap();
            let relation = if initial {
                RelationDef::initial(id, expression)
            } else {
                RelationDef::new(id, expression)
            }
            .unwrap();
            let node = KernelNode::from(relation);
            let wire = WireNode::encode(&node).unwrap();
            assert_eq!(
                wire.expression_root_count(),
                4,
                "budget counts sides, not equations"
            );
            assert_eq!(wire.decode().unwrap(), node);
            let mut json = serde_json::to_value(&wire).unwrap();
            assert_eq!(
                json["definition"]["expression"]["roots"],
                serde_json::json!([0, 1, 1, 0])
            );
            let mut displaced = json.clone();
            let expression = displaced["definition"]
                .as_object_mut()
                .unwrap()
                .remove("expression")
                .unwrap();
            displaced["definition"]["residuals"] = expression;
            assert!(serde_json::from_value::<WireNode>(displaced).is_err());
            json["definition"]["expression"]["roots"] = serde_json::json!([0, 1, 1]);
            let malformed: WireNode = serde_json::from_value(json.clone()).unwrap();
            assert!(malformed.decode().is_err());
            json["definition"]["expression"]["roots"] = serde_json::json!([]);
            let malformed: WireNode = serde_json::from_value(json).unwrap();
            assert!(malformed.decode().is_err());
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireObservableReduction {
    Value,
    VolumeIntegral {
        input: WireId,
        domain: WireId,
        limits: Option<[u32; 2]>,
    },
    BoundaryIntegral {
        input: WireId,
        domain: WireId,
        limits: Option<[u32; 2]>,
    },
    SphericalVolumeIntegral {
        input: WireId,
        domain: WireId,
        limits: Option<[u32; 2]>,
    },
}

impl WireObservableReduction {
    fn encode(value: ObservableReduction) -> Self {
        match value {
            ObservableReduction::Value => Self::Value,
            ObservableReduction::SpatialIntegral {
                input,
                domain,
                measure: ObservableMeasure::Volume,
                limits,
            } => Self::VolumeIntegral {
                limits: limits.map(|limits| limits.map(|id| id.index())),
                input: WireId::from_raw(input.erase()),
                domain: WireId::from_raw(domain.erase()),
            },
            ObservableReduction::SpatialIntegral {
                input,
                domain,
                measure: ObservableMeasure::Boundary,
                limits,
            } => Self::BoundaryIntegral {
                limits: limits.map(|limits| limits.map(|id| id.index())),
                input: WireId::from_raw(input.erase()),
                domain: WireId::from_raw(domain.erase()),
            },
            ObservableReduction::SpatialIntegral {
                input,
                domain,
                measure: ObservableMeasure::SphericalVolume,
                limits,
            } => Self::SphericalVolumeIntegral {
                limits: limits.map(|limits| limits.map(|id| id.index())),
                input: WireId::from_raw(input.erase()),
                domain: WireId::from_raw(domain.erase()),
            },
        }
    }
    fn decode(
        &self,
        expression: &eqiora_schema::kernel::ExprDag,
    ) -> Result<ObservableReduction, Diagnostic> {
        let decode_limits = |limits: Option<[u32; 2]>| -> Result<_, Diagnostic> {
            limits
                .map(|[lower, upper]| {
                    Ok([
                        expression.node_id(lower).ok_or_else(|| {
                            invalid_artifact("lower integral limit is outside its expression DAG")
                        })?,
                        expression.node_id(upper).ok_or_else(|| {
                            invalid_artifact("upper integral limit is outside its expression DAG")
                        })?,
                    ])
                })
                .transpose()
        };
        Ok(match self {
            Self::Value => ObservableReduction::Value,
            Self::VolumeIntegral {
                input,
                domain,
                limits,
            } => ObservableReduction::SpatialIntegral {
                input: input.typed()?,
                domain: domain.typed()?,
                measure: ObservableMeasure::Volume,
                limits: decode_limits(*limits)?,
            },
            Self::BoundaryIntegral {
                input,
                domain,
                limits,
            } => ObservableReduction::SpatialIntegral {
                input: input.typed()?,
                domain: domain.typed()?,
                measure: ObservableMeasure::Boundary,
                limits: decode_limits(*limits)?,
            },
            Self::SphericalVolumeIntegral {
                input,
                domain,
                limits,
            } => ObservableReduction::SpatialIntegral {
                input: input.typed()?,
                domain: domain.typed()?,
                measure: ObservableMeasure::SphericalVolume,
                limits: decode_limits(*limits)?,
            },
        })
    }
    fn semantic_references(&self) -> Vec<&WireId> {
        match self {
            Self::Value => Vec::new(),
            Self::VolumeIntegral { input, domain, .. }
            | Self::BoundaryIntegral { input, domain, .. }
            | Self::SphericalVolumeIntegral { input, domain, .. } => {
                vec![input, domain]
            }
        }
    }
}
