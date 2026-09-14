//! Typed definitions for Semantic Kernel nodes.

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{
    Diagnostic, DimExponents, DynQuantity, EntityKind, GraphPath, Id, RawId, ValueShape,
};

use super::{BoundaryPhysicalConnector, ExprDag, RationalTime};
use eqiora_core::{ValueFrame, ValueLiteral, ValueType};

mod relation;
mod spatial;
pub use relation::{RelationConditionKind, RelationDef, RelationMeaning};

pub use spatial::{
    AxisBounds, BoundarySide, CartesianAxisDefinition, CartesianCoordinateSource, DomainDef,
    DomainKind, GeometryDigest,
};

/// Canonical field representation before a discrete space is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RepresentationKind {
    /// Identity-only representation retained for schema-defined uses.
    Abstract,
    /// A field over a continuous domain.
    Continuum,
}

/// Representation definition. Basis family, mesh, and DOF layout belong to
/// the Realization Graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentationDef {
    id: Id<kinds::Representation>,
    kind: RepresentationKind,
}

impl RepresentationDef {
    /// Construct an abstract representation.
    #[must_use]
    pub const fn new(id: Id<kinds::Representation>) -> Self {
        Self {
            id,
            kind: RepresentationKind::Abstract,
        }
    }

    /// Construct a canonical continuum representation.
    #[must_use]
    pub const fn continuum(id: Id<kinds::Representation>) -> Self {
        Self {
            id,
            kind: RepresentationKind::Continuum,
        }
    }

    /// Typed node ID.
    #[must_use]
    pub const fn id(&self) -> Id<kinds::Representation> {
        self.id
    }

    /// Canonical representation kind.
    #[must_use]
    pub const fn kind(&self) -> RepresentationKind {
        self.kind
    }
}

/// Mathematical evolution ownership, independent of support and scalar type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldRole {
    /// An algebraic unknown; coordinate derivatives do not change its role.
    Variable,
    /// An owned state eligible for evolution at its declared activation.
    State,
}

/// Exact mathematical Field definition before realization.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    id: Id<kinds::Field>,
    value_type: ValueType,
    role: FieldRole,
}

impl FieldDef {
    /// Define a Field with one complete checked mathematical type.
    #[must_use]
    pub fn new(id: Id<kinds::Field>, value_type: ValueType, role: FieldRole) -> Self {
        Self {
            id,
            value_type,
            role,
        }
    }

    /// Typed Field ID.
    #[must_use]
    pub const fn id(&self) -> Id<kinds::Field> {
        self.id
    }

    /// Declared physical dimension.
    #[must_use]
    pub const fn dimension(&self) -> DimExponents {
        self.value_type.dimension()
    }

    /// Exact mathematical value shape.
    #[must_use]
    pub const fn shape(&self) -> &ValueShape {
        self.value_type.shape()
    }

    /// Coordinate-frame meaning of Field components.
    #[must_use]
    pub const fn frame(&self) -> ValueFrame {
        self.value_type.frame()
    }

    /// Complete mathematical type, independent of support and execution choices.
    #[must_use]
    pub const fn value_type(&self) -> &ValueType {
        &self.value_type
    }

    /// Author-declared evolution ownership; solver transformations cannot change it.
    #[must_use]
    pub const fn role(&self) -> FieldRole {
        self.role
    }
}

/// Typed Parameter definition initialized by a complete mathematical value.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterDef {
    id: Id<kinds::Parameter>,
    value: ValueLiteral,
}

impl ParameterDef {
    /// Define a Parameter from an already validated complete value.
    #[must_use]
    pub const fn new(id: Id<kinds::Parameter>, value: ValueLiteral) -> Self {
        Self { id, value }
    }

    /// Typed Parameter ID.
    #[must_use]
    pub const fn id(&self) -> Id<kinds::Parameter> {
        self.id
    }

    /// Complete declared mathematical type.
    #[must_use]
    pub const fn value_type(&self) -> &ValueType {
        self.value.value_type()
    }

    /// Complete value, including every real and imaginary component.
    #[must_use]
    pub const fn value(&self) -> &ValueLiteral {
        &self.value
    }

    /// Extract a value only when its mathematical type is a real scalar.
    #[must_use]
    pub fn real_scalar_value(&self) -> Option<DynQuantity> {
        self.value.real_scalar_value()
    }
}

/// Causal direction of a signal Port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignalDirection {
    /// Value enters the owning relation network.
    Input,
    /// Value leaves the owning relation network.
    Output,
}

/// Closed kernel-level Port payload.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PortPayload {
    /// Causal signal with an explicit direction and complete mathematical type.
    Signal {
        direction: SignalDirection,
        value_type: ValueType,
    },
    /// Scalar conserving connector typed nominally by one physical Domain.
    ScalarPhysical { domain: Id<kinds::Domain> },
    /// Field-valued boundary Port. Parent support and outward orientation are
    /// derived from the boundary's unique `BoundaryOf` edge.
    BoundaryPhysical {
        connector: Id<kinds::Domain>,
        boundary: Id<kinds::Domain>,
    },
}

/// Typed scalar Port definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortDef {
    id: Id<kinds::Port>,
    payload: PortPayload,
}

impl PortDef {
    /// Define a causal signal Port.
    #[must_use]
    pub const fn signal(
        id: Id<kinds::Port>,
        direction: SignalDirection,
        value_type: ValueType,
    ) -> Self {
        Self {
            id,
            payload: PortPayload::Signal {
                direction,
                value_type,
            },
        }
    }

    /// Define a scalar conserving Port with nominal Domain identity.
    #[must_use]
    pub const fn scalar_physical(id: Id<kinds::Port>, domain: Id<kinds::Domain>) -> Self {
        Self {
            id,
            payload: PortPayload::ScalarPhysical { domain },
        }
    }

    /// Define one field-valued physical Port on an exact boundary Domain.
    #[must_use]
    pub const fn boundary_physical(
        id: Id<kinds::Port>,
        connector: Id<kinds::Domain>,
        boundary: Id<kinds::Domain>,
    ) -> Self {
        Self {
            id,
            payload: PortPayload::BoundaryPhysical {
                connector,
                boundary,
            },
        }
    }

    /// Typed Port ID.
    #[must_use]
    pub const fn id(&self) -> Id<kinds::Port> {
        self.id
    }

    /// Closed Port payload.
    #[must_use]
    pub fn payload(&self) -> PortPayload {
        self.payload.clone()
    }

    /// Signal direction and complete mathematical type, if this is a signal Port.
    #[must_use]
    pub const fn signal_contract(&self) -> Option<(SignalDirection, &ValueType)> {
        match &self.payload {
            PortPayload::Signal {
                direction,
                value_type,
            } => Some((*direction, value_type)),
            PortPayload::ScalarPhysical { .. } | PortPayload::BoundaryPhysical { .. } => None,
        }
    }

    /// Nominal physical Domain, if this is a scalar physical Port.
    #[must_use]
    pub const fn physical_domain(&self) -> Option<Id<kinds::Domain>> {
        match self.payload {
            PortPayload::ScalarPhysical { domain } => Some(domain),
            PortPayload::Signal { .. } | PortPayload::BoundaryPhysical { .. } => None,
        }
    }

    /// Exact nominal connector and boundary support for a field-valued Port.
    #[must_use]
    pub const fn boundary_physical_contract(
        &self,
    ) -> Option<(Id<kinds::Domain>, Id<kinds::Domain>)> {
        match self.payload {
            PortPayload::BoundaryPhysical {
                connector,
                boundary,
            } => Some((connector, boundary)),
            PortPayload::Signal { .. } | PortPayload::ScalarPhysical { .. } => None,
        }
    }
}

/// Event zero-crossing direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventDirection {
    /// Either crossing direction.
    Any,
    /// Negative to positive.
    Rising,
    /// Positive to negative.
    Falling,
}

/// Activation semantics independent of execution scheduling.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ActivationKind {
    /// Relation is active throughout continuous model time.
    Continuous,
    /// Relation activates at ticks of its `ClockedBy` ClockDomain edge.
    Periodic,
    /// Relation activates on a zero crossing of the guard expression.
    Event {
        /// Scalar guard expression; exactly one root is required.
        guard: ExprDag,
        /// Crossing direction.
        direction: EventDirection,
    },
    /// Relation is active while a scalar guard is positive.
    Guard {
        /// Scalar guard expression; exactly one root is required.
        guard: ExprDag,
    },
}

/// Activation node definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationDef {
    id: Id<kinds::Activation>,
    kind: ActivationKind,
}

impl ActivationDef {
    /// Define and locally validate an Activation.
    ///
    /// # Errors
    /// Returns `EQ0302` when an event/guard has other than one expression root.
    pub fn new(id: Id<kinds::Activation>, kind: ActivationKind) -> Result<Self, Diagnostic> {
        let guard = match &kind {
            ActivationKind::Event { guard, .. } | ActivationKind::Guard { guard } => Some(guard),
            ActivationKind::Continuous | ActivationKind::Periodic => None,
        };
        if guard.is_some_and(|expression| expression.roots().len() != 1) {
            return Err(Diagnostic::error(
                codes::INVALID_KERNEL_DEFINITION,
                "event and guard Activations require exactly one guard root",
            )
            .with_graph_path(kernel_path(id.erase())));
        }
        Ok(Self { id, kind })
    }

    /// Continuous Activation convenience constructor.
    #[must_use]
    pub fn continuous(id: Id<kinds::Activation>) -> Self {
        Self {
            id,
            kind: ActivationKind::Continuous,
        }
    }

    /// Periodic Activation convenience constructor.
    #[must_use]
    pub fn periodic(id: Id<kinds::Activation>) -> Self {
        Self {
            id,
            kind: ActivationKind::Periodic,
        }
    }

    /// Typed Activation ID.
    #[must_use]
    pub const fn id(&self) -> Id<kinds::Activation> {
        self.id
    }

    /// Activation semantics.
    #[must_use]
    pub const fn kind(&self) -> &ActivationKind {
        &self.kind
    }
}

/// Connection semantics. Runtime transport policy belongs to Realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConnectionSemantics {
    /// Directed causal signal requiring equal exact activation and support.
    Signal {
        /// Exact connected source endpoint, including an interface relay.
        driver: Id<kinds::Port>,
    },
    /// Acausal connection enforcing equality and conservation laws.
    Conserving,
    /// Field-valued conserving pair identified by a derived Cartesian
    /// lower-to-upper translation.
    SpatialPeriodic,
}

/// Connection node definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionDef {
    id: Id<kinds::Connection>,
    semantics: ConnectionSemantics,
}

impl ConnectionDef {
    /// Define a Connection.
    #[must_use]
    pub const fn new(id: Id<kinds::Connection>, semantics: ConnectionSemantics) -> Self {
        Self { id, semantics }
    }

    /// Typed Connection ID.
    #[must_use]
    pub const fn id(&self) -> Id<kinds::Connection> {
        self.id
    }

    /// Closed signal, coincident-conserving, or spatial-periodic semantics.
    #[must_use]
    pub const fn semantics(&self) -> ConnectionSemantics {
        self.semantics
    }
}

/// Exact model-time definition of a ClockDomain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ClockKind {
    /// Continuous model time.
    Continuous,
    /// Exact periodic model time.
    Periodic {
        /// Strictly positive period.
        period: RationalTime,
        /// Non-negative phase relative to model time zero.
        phase: RationalTime,
    },
    /// Tick times arrive explicitly as semantic events.
    Aperiodic,
    /// Clock is inferred from connected signal semantics.
    Inherited,
}

/// ClockDomain node definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClockDomainDef {
    id: Id<kinds::ClockDomain>,
    kind: ClockKind,
}

impl ClockDomainDef {
    /// Continuous model-time ClockDomain.
    #[must_use]
    pub const fn continuous(id: Id<kinds::ClockDomain>) -> Self {
        Self {
            id,
            kind: ClockKind::Continuous,
        }
    }

    /// Periodic model-time ClockDomain.
    ///
    /// # Errors
    /// Returns `EQ0305` when `period` is zero.
    pub fn periodic(
        id: Id<kinds::ClockDomain>,
        period: RationalTime,
        phase: RationalTime,
    ) -> Result<Self, Diagnostic> {
        if period.is_zero() {
            return Err(Diagnostic::error(
                codes::INVALID_CLOCK,
                "periodic ClockDomain requires a strictly positive period",
            )
            .with_graph_path(kernel_path(id.erase())));
        }
        Ok(Self {
            id,
            kind: ClockKind::Periodic { period, phase },
        })
    }

    /// Aperiodic model-time ClockDomain.
    #[must_use]
    pub const fn aperiodic(id: Id<kinds::ClockDomain>) -> Self {
        Self {
            id,
            kind: ClockKind::Aperiodic,
        }
    }

    /// Inherited ClockDomain.
    #[must_use]
    pub const fn inherited(id: Id<kinds::ClockDomain>) -> Self {
        Self {
            id,
            kind: ClockKind::Inherited,
        }
    }

    /// Typed ClockDomain ID.
    #[must_use]
    pub const fn id(&self) -> Id<kinds::ClockDomain> {
        self.id
    }

    /// Exact model-time semantics.
    #[must_use]
    pub const fn kind(&self) -> ClockKind {
        self.kind
    }
}

/// Type-erased storage form of a complete Semantic Kernel node.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum KernelNode {
    /// Domain definition.
    Domain(DomainDef),
    /// Representation definition.
    Representation(RepresentationDef),
    /// Field definition.
    Field(FieldDef),
    /// Derived expression, independent of solve unknowns.
    Observable(super::ObservableDef),
    /// Parameter definition.
    Parameter(ParameterDef),
    /// Port definition.
    Port(PortDef),
    /// Relation definition.
    Relation(RelationDef),
    /// Activation definition.
    Activation(ActivationDef),
    /// Connection definition.
    Connection(ConnectionDef),
    /// Ordered nominal finite mathematical basis.
    FiniteSpace(super::FiniteSpaceDef),
    /// Fixed structural ordinal set.
    IndexSet(super::IndexSetDef),
    /// One closed nominal enum declaration.
    Enum(super::EnumDef),
    /// Closed nominal heterogeneous product declaration.
    Record(super::RecordDef),
    /// Ordered member bindings of one exact record occurrence.
    RecordInstance(super::RecordInstanceDef),
    /// ClockDomain definition.
    ClockDomain(ClockDomainDef),
}

impl KernelNode {
    /// Erased graph ID, derived from the variant's typed ID.
    #[must_use]
    pub fn id(&self) -> RawId {
        match self {
            Self::Domain(value) => value.id().erase(),
            Self::Representation(value) => value.id().erase(),
            Self::Field(value) => value.id().erase(),
            Self::Observable(value) => value.id().erase(),
            Self::Parameter(value) => value.id().erase(),
            Self::Port(value) => value.id().erase(),
            Self::Relation(value) => value.id().erase(),
            Self::Activation(value) => value.id().erase(),
            Self::Connection(value) => value.id().erase(),
            Self::ClockDomain(value) => value.id().erase(),
            Self::FiniteSpace(value) => value.id().erase(),
            Self::Enum(value) => value.id().erase(),
            Self::Record(value) => value.id().erase(),
            Self::RecordInstance(value) => value.id().erase(),
            Self::IndexSet(value) => value.id().erase(),
        }
    }

    /// Closed Semantic Kernel entity kind.
    #[must_use]
    pub const fn kind(&self) -> EntityKind {
        match self {
            Self::Domain(_) => EntityKind::Domain,
            Self::Representation(_) => EntityKind::Representation,
            Self::Field(_) => EntityKind::Field,
            Self::Observable(_) => EntityKind::Observable,
            Self::Parameter(_) => EntityKind::Parameter,
            Self::Port(_) => EntityKind::Port,
            Self::Relation(_) => EntityKind::Relation,
            Self::Activation(_) => EntityKind::Activation,
            Self::Connection(_) => EntityKind::Connection,
            Self::ClockDomain(_) => EntityKind::ClockDomain,
            Self::FiniteSpace(_) => EntityKind::FiniteSpace,
            Self::Enum(_) => EntityKind::Enum,
            Self::Record(_) => EntityKind::Record,
            Self::RecordInstance(_) => EntityKind::RecordInstance,
            Self::IndexSet(_) => EntityKind::IndexSet,
        }
    }

    /// Declared scalar dimension for values addressable by `SetValue`.
    #[must_use]
    pub fn value_dimension(&self) -> Option<DimExponents> {
        match self {
            Self::Parameter(value) => value.real_scalar_value().map(|value| value.dim()),
            _ => None,
        }
    }

    /// Model value installed when the node is first defined.
    #[must_use]
    pub fn initial_value(&self) -> Option<DynQuantity> {
        match self {
            Self::Parameter(value) => value.real_scalar_value(),
            _ => None,
        }
    }
}

macro_rules! kernel_from {
    ($definition:ident, $variant:ident) => {
        impl From<$definition> for KernelNode {
            fn from(value: $definition) -> Self {
                Self::$variant(value)
            }
        }
    };
}

kernel_from!(DomainDef, Domain);
kernel_from!(RepresentationDef, Representation);
kernel_from!(FieldDef, Field);
kernel_from!(ParameterDef, Parameter);
kernel_from!(PortDef, Port);
kernel_from!(RelationDef, Relation);
kernel_from!(ActivationDef, Activation);
kernel_from!(ConnectionDef, Connection);
kernel_from!(ClockDomainDef, ClockDomain);

/// An entity set with no name selects nothing and cannot be validated later.
fn named_entity_set(
    id: Id<kinds::Domain>,
    entity_set: impl Into<String>,
) -> Result<String, Diagnostic> {
    let entity_set = entity_set.into();
    if entity_set.trim().is_empty() {
        return Err(Diagnostic::error(
            codes::INVALID_KERNEL_DEFINITION,
            "geometry Domain requires a named entity set",
        )
        .with_graph_path(kernel_path(id.erase())));
    }
    Ok(entity_set)
}

fn kernel_path(id: RawId) -> GraphPath {
    GraphPath::new([
        "semantic".to_owned(),
        format!("{:?}", id.kind()),
        id.to_string(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::Dimension;
    use eqiora_core::quantity::dim;

    #[test]
    fn parameter_literals_preserve_types_without_real_scalar_narrowing() {
        let real = ValueType::scalar(eqiora_core::ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("checked scalar type");
        let complex = ValueType::scalar(
            eqiora_core::ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .expect("checked scalar type");
        let parameter = ParameterDef::new(
            Id::new(),
            ValueLiteral::from_real(real.clone(), 2.0).unwrap(),
        );
        assert_eq!(parameter.real_scalar_value().unwrap().value(), 2.0);
        let parameter = ParameterDef::new(
            Id::new(),
            ValueLiteral::new(complex.clone(), [(2.0, 3.0)]).unwrap(),
        );
        assert_eq!(parameter.value_type(), &complex);
        assert_eq!(parameter.value().component(0), Some((2.0, 3.0)));
        assert_eq!(parameter.real_scalar_value(), None);
        let array = complex.array(3).unwrap();
        let parameter = ParameterDef::new(
            Id::new(),
            ValueLiteral::from_real(array.clone(), -0.0).unwrap(),
        );
        assert_eq!(
            parameter.value().component(0).unwrap().0.to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(parameter.real_scalar_value(), None);
        assert!(ValueLiteral::from_real(array, 1.0).is_err());
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(ValueLiteral::from_real(real.clone(), invalid).is_err());
        }
    }

    #[test]
    fn field_roles_preserve_complete_types_without_declaration_values() {
        use eqiora_core::ScalarDomain::{Complex, Real};
        for value_type in [
            ValueType::scalar(Real, DimExponents::DIMENSIONLESS).expect("checked scalar type"),
            ValueType::scalar(Complex, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type")
                .array(3)
                .unwrap(),
        ] {
            for role in [FieldRole::Variable, FieldRole::State] {
                let field = FieldDef::new(Id::new(), value_type.clone(), role);
                assert_eq!(field.value_type(), &value_type);
                assert_eq!(field.role(), role);
                assert_eq!(KernelNode::Field(field).initial_value(), None);
            }
        }
    }

    #[test]
    fn initial_equations_validate_both_side_types() {
        use super::super::typing::{ExpressionType, RootContract, TypedResidual};
        use super::super::{ExprDagBuilder, SymbolRef};
        let field = Id::new();
        let temperature = ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            dim::TemperatureDim::EXPONENTS,
        )
        .expect("checked scalar type");
        let mut builder = ExprDagBuilder::new();
        let value = builder.symbol(SymbolRef::Field(field)).unwrap();
        let wrong_dimension = builder
            .constant(DynQuantity::new(2.0, dim::TimeDim::EXPONENTS))
            .unwrap();
        let expression = builder.finish([value, wrong_dimension]).unwrap();
        let initial = RelationDef::initial(Id::new(), expression.clone()).unwrap();
        assert!(initial.is_initial());
        assert!(
            !RelationDef::new(initial.id(), expression.clone())
                .unwrap()
                .is_initial()
        );
        assert!(
            TypedResidual::<()>::infer(
                expression,
                None,
                RootContract::InitialConditions,
                |_| Ok::<_, ()>(ExpressionType::new(temperature.clone(), None))
            )
            .is_err()
        );
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(ValueLiteral::from_real(temperature.clone(), invalid).is_err());
        }
    }

    #[test]
    fn periodic_clock_rejects_zero_period() {
        let clock = Id::<kinds::ClockDomain>::new();
        let diagnostic = ClockDomainDef::periodic(clock, RationalTime::ZERO, RationalTime::ZERO)
            .expect_err("zero period never advances");

        assert_eq!(diagnostic.code(), codes::INVALID_CLOCK);
    }

    #[test]
    fn cartesian_bounds_require_increasing_lengths() {
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
        assert!(
            AxisBounds::new(DynQuantity::new(0.0, length), DynQuantity::new(2.0, length)).is_ok()
        );
        assert_eq!(
            AxisBounds::new(DynQuantity::new(2.0, length), DynQuantity::new(0.0, length))
                .unwrap_err()
                .code(),
            codes::INVALID_KERNEL_DEFINITION
        );
        assert_eq!(
            AxisBounds::new(
                DynQuantity::new(0.0, DimExponents::DIMENSIONLESS),
                DynQuantity::new(2.0, DimExponents::DIMENSIONLESS)
            )
            .unwrap_err()
            .code(),
            codes::INVALID_KERNEL_DEFINITION
        );
    }
}
