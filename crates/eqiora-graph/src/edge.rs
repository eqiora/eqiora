//! Schema-approved edge vocabulary.

use eqiora_core::{EntityKind, GraphClass, RawId};

/// Stable edge kinds understood by the kernel Graph Federation.
///
/// Standard Ontology may give higher-level names to patterns of these edges,
/// but it does not introduce unchecked strings into the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum EdgeKind {
    /// A Field is defined on a Domain/Representation, or a spatial Port on a Domain.
    DefinedOn,
    /// A spatial relation holds on one domain or boundary domain.
    AppliesOn,
    /// A lower-dimensional Domain is one oriented part of a parent boundary.
    BoundaryOf,
    /// A relation depends on a field, parameter, port, or sampled clock, or a Cartesian
    /// Domain coordinate recipe depends on a Parameter.
    DependsOn,
    /// A declaration's fixed type, membership, or expression shape uses an exact Parameter.
    /// Changing that Parameter requires recompilation rather than a numerical value edit.
    StructurallyDependsOn,
    /// A relation exposes a port.
    HasPort,
    /// An activation controls a relation.
    Activates,
    /// A connection touches a port.
    Connects,
    /// An Activation, State Field, or causal Port belongs to an exact ClockDomain.
    ClockedBy,
    /// A realization entity realizes a semantic entity.
    Realizes,
    /// A space/discretization discretizes a semantic entity.
    Discretizes,
    /// Evidence observes a semantic or realization entity.
    Observes,
    /// Evidence calibrates a semantic or realization entity.
    Calibrates,
    /// Evidence validates a semantic or realization entity.
    Validates,
    /// An evidence/artifact entity was produced by another one.
    ProducedBy,
    /// An evidence/artifact entity derives from another one.
    DerivedFrom,
    /// An action or transaction mutates a graph entity.
    Mutates,
    /// A review or approval approves an action or transaction.
    Approves,
    /// An action or transaction executes against a graph entity.
    Executes,
    /// A realization entity is scheduled by an execution schedule.
    ScheduledBy,
    /// An execution schedule or solver plan targets hardware/deployment.
    Targets,
}

impl EdgeKind {
    /// Whether the erased endpoint kinds satisfy the kernel edge schema.
    #[must_use]
    pub const fn permits(self, from: EntityKind, to: EntityKind) -> bool {
        use EntityKind as K;
        use GraphClass as G;

        match self {
            Self::DefinedOn => {
                (matches!(from, K::Field) && matches!(to, K::Domain | K::Representation))
                    || (matches!(from, K::Port) && matches!(to, K::Domain))
            }
            Self::AppliesOn => {
                matches!(from, K::Relation | K::Observable) && matches!(to, K::Domain)
            }
            Self::BoundaryOf => matches!(from, K::Domain) && matches!(to, K::Domain),
            Self::DependsOn => {
                (matches!(from, K::Relation | K::Observable)
                    && matches!(to, K::Field | K::Parameter | K::Port | K::ClockDomain))
                    || (matches!(from, K::Observable) && matches!(to, K::Observable))
                    || (matches!(from, K::Domain) && matches!(to, K::Parameter))
            }
            Self::StructurallyDependsOn => {
                matches!(
                    from,
                    K::Field | K::Port | K::Parameter | K::Relation | K::IndexSet | K::Connection
                ) && matches!(to, K::Parameter)
            }
            Self::HasPort => matches!(from, K::Relation) && matches!(to, K::Port),
            Self::Activates => matches!(from, K::Activation) && matches!(to, K::Relation),
            Self::Connects => matches!(from, K::Connection) && matches!(to, K::Port),
            Self::ClockedBy => {
                matches!(from, K::Activation | K::Field | K::Port) && matches!(to, K::ClockDomain)
            }
            Self::Realizes => {
                matches!(from.graph(), G::Realization) && matches!(to.graph(), G::Semantic)
            }
            Self::Discretizes => {
                matches!(from, K::Space | K::Discretization)
                    && matches!(to, K::Domain | K::Representation | K::Field)
            }
            Self::Observes => {
                matches!(from, K::Experiment | K::Observation | K::Dataset)
                    && matches!(to.graph(), G::Semantic | G::Realization)
            }
            Self::Calibrates | Self::Validates => {
                matches!(from, K::Experiment | K::Run | K::Evidence)
                    && matches!(to.graph(), G::Semantic | G::Realization)
            }
            Self::ProducedBy | Self::DerivedFrom => {
                matches!(from.graph(), G::EvidenceArtifact)
                    && matches!(to.graph(), G::EvidenceArtifact)
            }
            Self::Mutates | Self::Executes => {
                matches!(from, K::Action | K::Transaction)
            }
            Self::Approves => {
                matches!(from, K::Review | K::Approval) && matches!(to, K::Action | K::Transaction)
            }
            Self::ScheduledBy => {
                matches!(from.graph(), G::Realization) && matches!(to, K::ExecutionSchedule)
            }
            Self::Targets => {
                matches!(from, K::ExecutionSchedule | K::SolverPlan) && matches!(to, K::Target)
            }
        }
    }
}

/// One validated graph edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge {
    from: RawId,
    to: RawId,
    kind: EdgeKind,
}

impl Edge {
    pub(crate) const fn new(from: RawId, to: RawId, kind: EdgeKind) -> Self {
        Self { from, to, kind }
    }

    /// Source node.
    #[must_use]
    pub const fn from(&self) -> RawId {
        self.from
    }

    /// Destination node.
    #[must_use]
    pub const fn to(&self) -> RawId {
        self.to
    }

    /// Stable edge kind.
    #[must_use]
    pub const fn kind(&self) -> EdgeKind {
        self.kind
    }
}

#[cfg(test)]
mod signal_edge_tests {
    use super::*;

    #[test]
    fn signal_activation_and_support_use_existing_exact_edge_kinds() {
        assert!(EdgeKind::ClockedBy.permits(EntityKind::Port, EntityKind::ClockDomain));
        assert!(EdgeKind::DefinedOn.permits(EntityKind::Port, EntityKind::Domain));
        assert!(!EdgeKind::DefinedOn.permits(EntityKind::Port, EntityKind::Representation));
        assert!(!EdgeKind::ClockedBy.permits(EntityKind::Port, EntityKind::Parameter));
    }
}
