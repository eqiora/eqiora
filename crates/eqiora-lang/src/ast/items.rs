//! Source container item ownership.

use super::*;

/// Private Component implementation declarations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ComponentItem {
    /// Private mathematical support composed from bound coordinate factors.
    Domain(DomainDecl),
    /// A private bounded nominal index set.
    IndexSet(NamedDefinitionDecl),
    /// Private immutable static expression, expanded without a Kernel entity.
    Let(NamedDefinitionDecl),
    /// Scalar compile-time Parameter.
    Parameter(ComponentParameterDecl),
    /// Private inspectable derived value.
    Observable(ObservableDecl),
    /// Causal or conserving interface.
    Port(ComponentPortDecl),
    /// Field-physical Port family over one complete exterior.
    PortFamily(ComponentPortFamilyDecl),
    /// Private mutable state.
    Field(FieldDecl),
    /// Simultaneous fresh initialization, owned by this occurrence.
    Initial(InitialDecl),
    /// Private exact periodic clock.
    Clock(ClockDecl),
    /// Private zero-crossing event activation.
    Event(EventDecl),
    /// Private implicit residual group.
    Relation(RelationDecl),
    /// Private Relation family over an exact finite set.
    RelationFamily(RelationFamilyDecl),
    /// Private local connection.
    Connection(ConnectionDecl),
    /// Conserving boundary-family connection, optionally pointwise-bound.
    BoundaryConnection(BoundaryConnectionDecl),
    /// Private nested component instance.
    Instance(InstanceDecl),
}

/// Model-level declaration.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Item {
    /// A bounded nominal index set.
    IndexSet(NamedDefinitionDecl),
    /// Continuous spatial region or one oriented boundary portion.
    Domain(DomainDecl),
    /// Mutable model state.
    Field(FieldDecl),
    /// Simultaneous fresh initialization, separate from numerical guesses.
    Initial(InitialDecl),
    /// Revision-local design value.
    Parameter(ParameterDecl),
    /// Public named derived result.
    Observable(ObservableDecl),
    /// Typed compile-time expression alias expanded before Kernel lowering.
    Let(NamedDefinitionDecl),
    /// Causal or conserving interface.
    Port(PortDecl),
    /// Exact periodic clock.
    Clock(ClockDecl),
    /// Private zero-crossing event activation.
    Event(EventDecl),
    /// Implicit residual group and activation.
    Relation(RelationDecl),
    /// Relation expanded over a finite nominal index set.
    RelationFamily(RelationFamilyDecl),
    /// Signal or conserving connection net.
    Connection(ConnectionDecl),
    /// Conserving connection containing exact boundary-member selectors.
    BoundaryConnection(BoundaryConnectionDecl),
    /// Compile-time component instance.
    Instance(InstanceDecl),
}
