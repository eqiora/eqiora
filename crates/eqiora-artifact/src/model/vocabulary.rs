//! Wire spellings of the closed Kernel vocabularies.
//!
//! Each is a small closed enum whose only job is to cross the wire without
//! widening what the Kernel means.

use eqiora_core::ValueFrame;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, ValueShape};
use eqiora_schema::kernel::pure_operator::{
    CalculusBuilder, CalculusNode, CalculusNodeId, ExactRational, PureOperatorDefinition,
    PureValueClass, ResultAxis,
};
use eqiora_schema::kernel::{
    ActivationKind, BoundaryPairing, BoundarySide, CartesianAxisDefinition,
    CartesianCoordinateSource, ClockDomainDef, ClockKind, ConnectionSemantics, EventDirection,
    RationalTime, RepresentationKind, SignalDirection,
};
use serde::{Deserialize, Serialize};

use crate::dimension::WireDimension;
use crate::invalid_artifact;

use super::*;
use super::{expression::*, primitive::*, value_type::WireScalarDomain};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct WireValueShape(Vec<u32>);

impl WireValueShape {
    pub(crate) fn encode(value: &ValueShape) -> Self {
        Self(value.extents().iter().map(|extent| extent.get()).collect())
    }

    pub(crate) fn decode(&self) -> Result<ValueShape, Diagnostic> {
        ValueShape::new(self.0.iter().copied()).map_err(|error| {
            invalid_artifact(format!(
                "wire value shape has an invalid extent at axis {}",
                error.axis()
            ))
        })
    }

    pub(crate) fn ensure_limits(&self, limits: ModelDecoderLimits) -> Result<(), Diagnostic> {
        require_decoder_count(
            "value-shape rank",
            self.0.len(),
            limits.max_value_shape_rank,
        )?;
        let shape = self.decode()?;
        let components = shape.component_count().ok_or_else(|| {
            invalid_artifact("wire value-shape component product exceeds local usize")
        })?;
        require_decoder_count(
            "value-shape scalar components",
            components,
            limits.max_value_shape_components,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireValueFrame {
    Invariant,
    SpatialCartesian,
}

impl WireValueFrame {
    pub(crate) const fn encode(value: ValueFrame) -> Self {
        match value {
            ValueFrame::Invariant => Self::Invariant,
            ValueFrame::SpatialCartesian => Self::SpatialCartesian,
        }
    }

    pub(crate) const fn decode(self) -> ValueFrame {
        match self {
            Self::Invariant => ValueFrame::Invariant,
            Self::SpatialCartesian => ValueFrame::SpatialCartesian,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireBoundaryPairing {
    EuclideanBoundaryDuality,
}

impl WireBoundaryPairing {
    pub(crate) const fn encode(value: BoundaryPairing) -> Self {
        match value {
            BoundaryPairing::EuclideanBoundaryDuality => Self::EuclideanBoundaryDuality,
        }
    }

    pub(crate) const fn decode(self) -> BoundaryPairing {
        match self {
            Self::EuclideanBoundaryDuality => BoundaryPairing::EuclideanBoundaryDuality,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireCartesianAxisDefinition {
    lower: WireCartesianCoordinateSource,
    upper: WireCartesianCoordinateSource,
}

impl WireCartesianAxisDefinition {
    pub(crate) fn encode(value: CartesianAxisDefinition) -> Self {
        Self {
            lower: WireCartesianCoordinateSource::encode(value.lower()),
            upper: WireCartesianCoordinateSource::encode(value.upper()),
        }
    }

    pub(crate) fn decode(&self) -> Result<CartesianAxisDefinition, Diagnostic> {
        Ok(CartesianAxisDefinition::new(
            self.lower.decode()?,
            self.upper.decode()?,
        ))
    }

    pub(crate) fn semantic_references(&self) -> impl Iterator<Item = &WireId> {
        self.lower
            .semantic_reference()
            .into_iter()
            .chain(self.upper.semantic_reference())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "kebab-case", deny_unknown_fields)]
enum WireCartesianCoordinateSource {
    Fixed { value: WireQuantity },
    Parameter { parameter: WireId },
}

impl WireCartesianCoordinateSource {
    fn encode(value: CartesianCoordinateSource) -> Self {
        match value {
            CartesianCoordinateSource::Fixed(value) => Self::Fixed {
                value: WireQuantity::encode(value),
            },
            CartesianCoordinateSource::Parameter(parameter) => Self::Parameter {
                parameter: WireId::from_raw(parameter.erase()),
            },
        }
    }

    fn decode(&self) -> Result<CartesianCoordinateSource, Diagnostic> {
        match self {
            Self::Fixed { value } => CartesianCoordinateSource::fixed(value.decode()?)
                .map_err(|error| invalid_artifact(error.message())),
            Self::Parameter { parameter } => Ok(CartesianCoordinateSource::parameter(
                parameter.typed::<kinds::Parameter>()?,
            )),
        }
    }

    const fn semantic_reference(&self) -> Option<&WireId> {
        match self {
            Self::Parameter { parameter } => Some(parameter),
            Self::Fixed { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireBoundarySide {
    Lower,
    Upper,
}

impl WireBoundarySide {
    pub(crate) const fn encode(value: BoundarySide) -> Self {
        match value {
            BoundarySide::Lower => Self::Lower,
            BoundarySide::Upper => Self::Upper,
        }
    }

    pub(crate) const fn decode(self) -> BoundarySide {
        match self {
            Self::Lower => BoundarySide::Lower,
            Self::Upper => BoundarySide::Upper,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireRepresentationKind {
    Abstract,
    Continuum,
}

impl WireRepresentationKind {
    pub(crate) fn encode(value: RepresentationKind) -> Result<Self, Diagnostic> {
        match value {
            RepresentationKind::Abstract => Ok(Self::Abstract),
            RepresentationKind::Continuum => Ok(Self::Continuum),
            _ => Err(invalid_artifact(
                "representation kind is unsupported by the current Model contract",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireSignalDirection {
    Input,
    Output,
}

impl WireSignalDirection {
    pub(crate) const fn encode(value: SignalDirection) -> Self {
        match value {
            SignalDirection::Input => Self::Input,
            SignalDirection::Output => Self::Output,
        }
    }

    pub(crate) const fn decode(self) -> SignalDirection {
        match self {
            Self::Input => SignalDirection::Input,
            Self::Output => SignalDirection::Output,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireActivationKind {
    Continuous,
    Periodic,
    Event {
        guard: WireExpression,
        direction: WireEventDirection,
        priority: i64,
    },
    Guard {
        guard: WireExpression,
    },
}

impl WireActivationKind {
    pub(crate) fn literal_component_count(&self) -> Result<usize, Diagnostic> {
        match self {
            Self::Event { guard, .. } | Self::Guard { guard } => guard.literal_component_count(),
            Self::Continuous | Self::Periodic => Ok(0),
        }
    }

    pub(crate) fn ensure_value_shape_limits(
        &self,
        limits: ModelDecoderLimits,
    ) -> Result<(), Diagnostic> {
        match self {
            Self::Event { guard, .. } | Self::Guard { guard } => {
                guard.ensure_value_shape_limits(limits)
            }
            Self::Continuous | Self::Periodic => Ok(()),
        }
    }

    pub(crate) fn encode(value: &ActivationKind) -> Result<Self, Diagnostic> {
        match value {
            ActivationKind::Continuous => Ok(Self::Continuous),
            ActivationKind::Periodic => Ok(Self::Periodic),
            ActivationKind::Event {
                guard,
                direction,
                priority,
            } => Ok(Self::Event {
                guard: WireExpression::encode(guard)?,
                direction: WireEventDirection::encode(*direction),
                priority: *priority,
            }),
            ActivationKind::Guard { guard } => Ok(Self::Guard {
                guard: WireExpression::encode(guard)?,
            }),
            _ => Err(invalid_artifact(
                "Activation kind is newer than the supported model wire vocabulary",
            )),
        }
    }

    pub(crate) fn decode(&self) -> Result<ActivationKind, Diagnostic> {
        Ok(match self {
            Self::Continuous => ActivationKind::Continuous,
            Self::Periodic => ActivationKind::Periodic,
            Self::Event {
                guard,
                direction,
                priority,
            } => ActivationKind::Event {
                guard: guard.decode()?,
                direction: direction.decode(),
                priority: *priority,
            },
            Self::Guard { guard } => ActivationKind::Guard {
                guard: guard.decode()?,
            },
        })
    }

    pub(crate) fn expression_node_count(&self) -> usize {
        match self {
            Self::Event { guard, .. } | Self::Guard { guard } => guard.nodes.len(),
            Self::Continuous | Self::Periodic => 0,
        }
    }

    pub(crate) fn expression_root_count(&self) -> usize {
        match self {
            Self::Event { guard, .. } | Self::Guard { guard } => guard.roots.len(),
            Self::Continuous | Self::Periodic => 0,
        }
    }

    pub(crate) fn pure_operator_counts(&self) -> Result<PureOperatorWireCounts, Diagnostic> {
        match self {
            Self::Event { guard, .. } | Self::Guard { guard } => guard.pure_operator_counts(),
            Self::Continuous | Self::Periodic => Ok(PureOperatorWireCounts::default()),
        }
    }

    pub(crate) fn validate_pure_operator_features(&self) -> Result<(), Diagnostic> {
        match self {
            Self::Event { guard, .. } | Self::Guard { guard } => {
                guard.validate_pure_operator_features()
            }
            Self::Continuous | Self::Periodic => Ok(()),
        }
    }

    pub(crate) fn canonicalize_pure_operator_definitions(&mut self) -> Result<(), Diagnostic> {
        match self {
            Self::Event { guard, .. } | Self::Guard { guard } => {
                guard.canonicalize_pure_operator_definitions()
            }
            Self::Continuous | Self::Periodic => Ok(()),
        }
    }

    pub(crate) fn semantic_references(&self) -> Vec<&WireId> {
        match self {
            Self::Event { guard, .. } | Self::Guard { guard } => guard.semantic_references(),
            Self::Continuous | Self::Periodic => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireEventDirection {
    Any,
    Rising,
    Falling,
}

impl WireEventDirection {
    pub(crate) const fn encode(value: EventDirection) -> Self {
        match value {
            EventDirection::Any => Self::Any,
            EventDirection::Rising => Self::Rising,
            EventDirection::Falling => Self::Falling,
        }
    }

    pub(crate) const fn decode(self) -> EventDirection {
        match self {
            Self::Any => EventDirection::Any,
            Self::Rising => EventDirection::Rising,
            Self::Falling => EventDirection::Falling,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireConnectionKind {
    Signal { driver: WireId },
    Conserving,
    SpatialPeriodic,
}

impl WireConnectionKind {
    pub(crate) fn encode(value: ConnectionSemantics) -> Result<Self, Diagnostic> {
        match value {
            ConnectionSemantics::Signal { driver } => Ok(Self::Signal {
                driver: WireId::from_raw(driver.erase()),
            }),
            ConnectionSemantics::Conserving => Ok(Self::Conserving),
            ConnectionSemantics::SpatialPeriodic => Ok(Self::SpatialPeriodic),
            _ => Err(invalid_artifact(
                "connection semantics are newer than the current Model wire",
            )),
        }
    }

    pub(crate) fn decode(&self) -> Result<ConnectionSemantics, Diagnostic> {
        Ok(match self {
            Self::Signal { driver } => ConnectionSemantics::Signal {
                driver: driver.typed::<kinds::Port>()?,
            },
            Self::Conserving => ConnectionSemantics::Conserving,
            Self::SpatialPeriodic => ConnectionSemantics::SpatialPeriodic,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireClockKind {
    Continuous,
    Periodic {
        period: WireRationalTime,
        phase: WireRationalTime,
    },
    Aperiodic,
    Inherited,
}

impl WireClockKind {
    pub(crate) fn encode(value: ClockKind) -> Result<Self, Diagnostic> {
        match value {
            ClockKind::Continuous => Ok(Self::Continuous),
            ClockKind::Periodic { period, phase } => Ok(Self::Periodic {
                period: WireRationalTime::encode(period),
                phase: WireRationalTime::encode(phase),
            }),
            ClockKind::Aperiodic => Ok(Self::Aperiodic),
            ClockKind::Inherited => Ok(Self::Inherited),
            _ => Err(invalid_artifact(
                "clock kind is unsupported by the current Model contract",
            )),
        }
    }

    pub(crate) fn decode(&self, id: Id<kinds::ClockDomain>) -> Result<ClockDomainDef, Diagnostic> {
        match self {
            Self::Continuous => Ok(ClockDomainDef::continuous(id)),
            Self::Periodic { period, phase } => {
                ClockDomainDef::periodic(id, period.decode()?, phase.decode()?)
                    .map_err(|error| invalid_artifact(error.message()))
            }
            Self::Aperiodic => Ok(ClockDomainDef::aperiodic(id)),
            Self::Inherited => Ok(ClockDomainDef::inherited(id)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireRationalTime {
    pub(crate) numerator: u64,
    pub(crate) denominator: u64,
}

impl WireRationalTime {
    pub(crate) const fn encode(value: RationalTime) -> Self {
        Self {
            numerator: value.numerator(),
            denominator: value.denominator(),
        }
    }

    pub(crate) fn decode(self) -> Result<RationalTime, Diagnostic> {
        let value = RationalTime::new(self.numerator, self.denominator)
            .map_err(|error| invalid_artifact(error.message()))?;
        if value.numerator() != self.numerator || value.denominator() != self.denominator {
            return Err(invalid_artifact(
                "wire rational time must already be in canonical reduced form",
            ));
        }
        Ok(value)
    }
}

pub(crate) const PURE_COMPONENT_CALCULUS_V2: &str = "eqiora.pure-component-calculus/v2";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WirePureOperatorDefinition {
    pub(crate) digest: String,
    pub(crate) required_features: Vec<String>,
    pub(crate) formals: Vec<WirePureValueClass>,
    pub(crate) result: WirePureValueClass,
    pub(crate) nodes: Vec<WirePureCalculusNode>,
    pub(crate) root: u32,
}

impl WirePureOperatorDefinition {
    pub(crate) fn encode(definition: &PureOperatorDefinition) -> Self {
        Self {
            digest: definition.digest().to_string(),
            required_features: vec![PURE_COMPONENT_CALCULUS_V2.to_owned()],
            formals: definition
                .formals()
                .iter()
                .copied()
                .map(WirePureValueClass::encode)
                .collect(),
            result: WirePureValueClass::encode(definition.result_rule()),
            nodes: definition
                .nodes()
                .iter()
                .map(WirePureCalculusNode::encode)
                .collect(),
            root: definition.root().index(),
        }
    }

    pub(crate) fn validate_features(&self) -> Result<(), Diagnostic> {
        if self.required_features.is_empty() {
            return Err(invalid_artifact(
                "pure-operator definition is missing its required component-calculus feature",
            ));
        }
        if let Some(feature) = self
            .required_features
            .iter()
            .find(|feature| feature.as_str() != PURE_COMPONENT_CALCULUS_V2)
        {
            return Err(invalid_artifact(format!(
                "pure-operator definition requires unknown feature `{feature}`"
            )));
        }
        if self
            .required_features
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid_artifact(
                "pure-operator required features must be sorted and duplicate-free",
            ));
        }
        Ok(())
    }

    pub(crate) fn rebuild_and_validate_digest(&self) -> Result<PureOperatorDefinition, Diagnostic> {
        let formals = self
            .formals
            .iter()
            .copied()
            .map(WirePureValueClass::decode)
            .collect::<Result<Vec<_>, _>>()?;
        let result = self.result.decode()?;
        let mut builder = CalculusBuilder::new(formals, result).map_err(|error| {
            invalid_artifact(format!("invalid pure-operator definition: {error}"))
        })?;
        let mut ids = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let id = node.decode(&mut builder, &ids)?;
            ids.push(id);
        }
        let root = calculus_operand(&ids, self.root)?;
        let definition = builder.finish(root).map_err(|error| {
            invalid_artifact(format!("invalid pure-operator definition: {error}"))
        })?;
        validate_operator_digest(&self.digest)?;
        let actual = definition.digest().to_string();
        if self.digest != actual {
            return Err(invalid_artifact(format!(
                "pure-operator definition digest mismatch: claimed {}, rebuilt {actual}",
                self.digest
            )));
        }
        Ok(definition)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WirePureValueClass {
    InvariantScalar {
        #[serde(deserialize_with = "Deserialize::deserialize")]
        dimension: Option<WireDimension>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        scalar_domain: Option<WireScalarDomain>,
    },
    SpatialTensor {
        rank: u16,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        dimension: Option<WireDimension>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        scalar_domain: Option<WireScalarDomain>,
    },
}

impl WirePureValueClass {
    pub(crate) fn encode(value: PureValueClass) -> Self {
        let dimension = value.dimension().map(WireDimension::encode);
        let scalar_domain = value.scalar_domain().map(WireScalarDomain::encode);
        match value.spatial_rank() {
            None => Self::InvariantScalar {
                dimension,
                scalar_domain,
            },
            Some(rank) => Self::SpatialTensor {
                rank,
                dimension,
                scalar_domain,
            },
        }
    }

    pub(crate) fn decode(self) -> Result<PureValueClass, Diagnostic> {
        let (class, dimension, scalar_domain) = match self {
            Self::InvariantScalar {
                dimension,
                scalar_domain,
            } => (PureValueClass::invariant_scalar(), dimension, scalar_domain),
            Self::SpatialTensor {
                rank,
                dimension,
                scalar_domain,
            } => (
                PureValueClass::spatial_tensor(rank).map_err(|error| {
                    invalid_artifact(format!("invalid pure value class: {error}"))
                })?,
                dimension,
                scalar_domain,
            ),
        };
        let class = match scalar_domain {
            Some(domain) => class
                .with_scalar_domain(domain.decode())
                .map_err(|error| invalid_artifact(format!("invalid pure value class: {error}")))?,
            None => class,
        };
        Ok(match dimension {
            Some(dimension) => class.with_dimension(dimension.decode()),
            None => class,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WirePureCalculusNode {
    BoundInput {
        value: u32,
    },
    Differentiated {
        value: u32,
        source: u32,
        wrt: u32,
    },
    Rational {
        numerator: i64,
        denominator: u64,
        dimension: WireDimension,
    },
    Boolean {
        value: bool,
    },
    Compare {
        comparison: WireComparisonOp,
        left: u32,
        right: u32,
    },
    Not {
        value: u32,
    },
    And {
        left: u32,
        right: u32,
    },
    Or {
        left: u32,
        right: u32,
    },
    Select {
        condition: u32,
        then_value: u32,
        else_value: u32,
    },
    Require {
        condition: u32,
        value: u32,
    },
    UnaryMath {
        function: WireUnaryMath,
        value: u32,
    },
    FormalComponent {
        formal: u16,
        axes: Vec<u16>,
    },
    KroneckerDelta {
        left_axis: u16,
        right_axis: u16,
    },
    Neg {
        value: u32,
    },
    Add {
        left: u32,
        right: u32,
    },
    Mul {
        left: u32,
        right: u32,
    },
}

impl WirePureCalculusNode {
    pub(crate) fn encode(node: &CalculusNode) -> Self {
        match node {
            CalculusNode::Rational { value, dimension } => Self::Rational {
                dimension: WireDimension::encode(*dimension),
                numerator: value.numerator(),
                denominator: value.denominator(),
            },
            CalculusNode::Boolean(value) => Self::Boolean { value: *value },
            CalculusNode::Compare(comparison, left, right) => Self::Compare {
                comparison: WireComparisonOp::encode(*comparison),
                left: left.index(),
                right: right.index(),
            },
            CalculusNode::Not(value) => Self::Not {
                value: value.index(),
            },
            CalculusNode::And(left, right) => Self::And {
                left: left.index(),
                right: right.index(),
            },
            CalculusNode::Or(left, right) => Self::Or {
                left: left.index(),
                right: right.index(),
            },
            CalculusNode::Select {
                condition,
                then_value,
                else_value,
            } => Self::Select {
                condition: condition.index(),
                then_value: then_value.index(),
                else_value: else_value.index(),
            },
            CalculusNode::Require { condition, value } => Self::Require {
                condition: condition.index(),
                value: value.index(),
            },
            CalculusNode::UnaryMath(function, value) => Self::UnaryMath {
                function: WireUnaryMath::encode(*function)
                    .expect("checked pure calculus admits only square root"),
                value: value.index(),
            },
            CalculusNode::FormalComponent { formal, axes } => Self::FormalComponent {
                formal: *formal,
                axes: axes.iter().map(|axis| axis.index()).collect(),
            },
            CalculusNode::KroneckerDelta(left, right) => Self::KroneckerDelta {
                left_axis: left.index(),
                right_axis: right.index(),
            },
            CalculusNode::Differentiated { value, source, wrt } => Self::Differentiated {
                value: value.index(),
                source: source.index(),
                wrt: wrt.index(),
            },
            CalculusNode::BoundInput(value) => Self::BoundInput {
                value: value.index(),
            },
            CalculusNode::Neg(value) => Self::Neg {
                value: value.index(),
            },
            CalculusNode::Add(left, right) => Self::Add {
                left: left.index(),
                right: right.index(),
            },
            CalculusNode::Mul(left, right) => Self::Mul {
                left: left.index(),
                right: right.index(),
            },
        }
    }

    pub(crate) fn decode(
        &self,
        builder: &mut CalculusBuilder,
        ids: &[CalculusNodeId],
    ) -> Result<CalculusNodeId, Diagnostic> {
        let node = match self {
            Self::Rational {
                numerator,
                denominator,
                dimension,
            } => CalculusNode::Rational {
                dimension: dimension.decode(),
                value: ExactRational::from_canonical_parts(*numerator, *denominator).map_err(
                    |error| invalid_artifact(format!("invalid canonical pure rational: {error}")),
                )?,
            },
            Self::Boolean { value } => CalculusNode::Boolean(*value),
            Self::Compare {
                comparison,
                left,
                right,
            } => CalculusNode::Compare(
                comparison.decode(),
                calculus_operand(ids, *left)?,
                calculus_operand(ids, *right)?,
            ),
            Self::Not { value } => CalculusNode::Not(calculus_operand(ids, *value)?),
            Self::And { left, right } => CalculusNode::And(
                calculus_operand(ids, *left)?,
                calculus_operand(ids, *right)?,
            ),
            Self::Or { left, right } => CalculusNode::Or(
                calculus_operand(ids, *left)?,
                calculus_operand(ids, *right)?,
            ),
            Self::Select {
                condition,
                then_value,
                else_value,
            } => CalculusNode::Select {
                condition: calculus_operand(ids, *condition)?,
                then_value: calculus_operand(ids, *then_value)?,
                else_value: calculus_operand(ids, *else_value)?,
            },
            Self::Require { condition, value } => CalculusNode::Require {
                condition: calculus_operand(ids, *condition)?,
                value: calculus_operand(ids, *value)?,
            },
            Self::UnaryMath { function, value } => {
                CalculusNode::UnaryMath(function.decode(), calculus_operand(ids, *value)?)
            }
            Self::FormalComponent { formal, axes } => CalculusNode::FormalComponent {
                formal: *formal,
                axes: axes
                    .iter()
                    .copied()
                    .map(ResultAxis::new)
                    .collect::<Box<[_]>>(),
            },
            Self::KroneckerDelta {
                left_axis,
                right_axis,
            } => CalculusNode::KroneckerDelta(
                ResultAxis::new(*left_axis),
                ResultAxis::new(*right_axis),
            ),
            Self::Differentiated { value, source, wrt } => CalculusNode::Differentiated {
                value: calculus_operand(ids, *value)?,
                source: calculus_operand(ids, *source)?,
                wrt: calculus_operand(ids, *wrt)?,
            },
            Self::BoundInput { value } => CalculusNode::BoundInput(calculus_operand(ids, *value)?),
            Self::Neg { value } => CalculusNode::Neg(calculus_operand(ids, *value)?),
            Self::Add { left, right } => CalculusNode::Add(
                calculus_operand(ids, *left)?,
                calculus_operand(ids, *right)?,
            ),
            Self::Mul { left, right } => CalculusNode::Mul(
                calculus_operand(ids, *left)?,
                calculus_operand(ids, *right)?,
            ),
        };
        builder
            .push(node)
            .map_err(|error| invalid_artifact(format!("invalid pure calculus node: {error}")))
    }
}

pub(crate) fn calculus_operand(
    ids: &[CalculusNodeId],
    index: u32,
) -> Result<CalculusNodeId, Diagnostic> {
    usize::try_from(index)
        .ok()
        .and_then(|index| ids.get(index))
        .copied()
        .ok_or_else(|| {
            invalid_artifact(format!(
                "pure calculus operand {index} is not topologically prior"
            ))
        })
}

pub(crate) fn validate_operator_digest(digest: &str) -> Result<(), Diagnostic> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid_artifact(
            "pure-operator digest must be exactly 64 lowercase hexadecimal characters",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod pure_constraint_tests {
    use super::*;
    use eqiora_core::DimExponents;
    use serde_json::json;

    #[test]
    fn scalar_and_tensor_classes_preserve_exact_optional_dimensions() {
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        for class in [
            PureValueClass::invariant_scalar(),
            PureValueClass::spatial_tensor(1).unwrap(),
            PureValueClass::spatial_tensor(2).unwrap(),
        ] {
            for value in [
                class,
                class.with_dimension(length),
                class
                    .with_scalar_domain(eqiora_core::ScalarDomain::Real)
                    .unwrap(),
                class
                    .with_scalar_domain(eqiora_core::ScalarDomain::Complex)
                    .unwrap(),
            ] {
                let wire = WirePureValueClass::encode(value);
                let bytes = serde_json::to_vec(&wire).unwrap();
                let restored: WirePureValueClass = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(restored.decode().unwrap(), value);
            }
        }
        let rank_two = PureValueClass::spatial_tensor(2)
            .unwrap()
            .with_dimension(length);
        assert_eq!(
            serde_json::to_value(WirePureValueClass::encode(rank_two)).unwrap(),
            json!({"kind":"spatial-tensor","rank":2,"scalar_domain":null,"dimension":[[0,1],[1,1],[0,1],[0,1],[0,1],[0,1],[0,1]]})
        );
        let invalid: WirePureValueClass = serde_json::from_value(
            json!({"kind":"spatial-tensor","rank":0,"scalar_domain":null,"dimension":null}),
        )
        .unwrap();
        assert!(invalid.decode().is_err());
    }

    #[test]
    fn result_constraint_proof_precedes_claimed_digest_acceptance() {
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
        let class = PureValueClass::invariant_scalar().with_dimension(length);
        let mut builder = CalculusBuilder::new([class], class).unwrap();
        let root = builder
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let mut wire = WirePureOperatorDefinition::encode(&builder.finish(root).unwrap());
        assert!(wire.rebuild_and_validate_digest().is_ok());
        wire.result =
            WirePureValueClass::encode(PureValueClass::invariant_scalar().with_dimension(time));
        let error = wire.rebuild_and_validate_digest().unwrap_err().to_string();
        assert!(
            error.contains("invalid pure-operator definition"),
            "{error}"
        );
        assert!(!error.contains("digest mismatch"), "{error}");
    }
}

#[cfg(test)]
#[path = "calculus_wire_tests.rs"]
mod calculus_wire_tests;
