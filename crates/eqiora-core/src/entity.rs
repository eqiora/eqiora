//! Entity and graph kinds of the Graph Federation.
//!
//! [`Entity`] is **sealed**: the semantic kernel's entity set is defined here
//! and only here. Standard-ontology concepts — `Model`, `Coupling`,
//! `Scale`, `Objective`, `Solver`, `EvidenceSet` as user-facing notions — are
//! defined at the schema level as compositions of kernel entities and do
//! **not** implement [`Entity`]. Adding a kernel entity requires an RFC
//! proving it cannot be derived from the existing ones.

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// One of the four graphs in the Graph Federation.
pub trait GraphKind: sealed::Sealed + 'static {
    /// Stable, serialization-facing name of the graph.
    const NAME: &'static str;
}

/// Marker trait for entities addressable by a typed [`crate::Id`].
///
/// Sealed — see module docs.
pub trait Entity: sealed::Sealed + 'static {
    /// Runtime discriminant, used by [`crate::RawId`] and serialization.
    const KIND: EntityKind;
    /// The graph this entity lives in. Cross-graph references are only ever
    /// created through schema-approved edge constructors.
    type Graph: GraphKind;
}

/// Runtime discriminant for entity kinds.
///
/// `#[non_exhaustive]`: new kinds may be added by RFC; downstream `match`
/// must carry a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum EntityKind {
    // --- Semantic kernel ---
    /// Region where physics exists.
    Domain,
    /// Continuum, mesh, particles, graph, lattice, modes, distribution, ...
    Representation,
    /// State, observable, or parameter field.
    Field,
    /// Derived typed expression without a solve unknown or equation.
    Observable,
    /// Scalar/tensor parameter.
    Parameter,
    /// Typed input/output contract.
    Port,
    /// Implicit operator relation `r(...) = 0`.
    Relation,
    /// Continuous / periodic / event / guard activation of a relation.
    Activation,
    /// `signal` or `conserving` connection between ports.
    Connection,
    /// Model-time semantics (continuous, periodic, aperiodic, inherited).
    ClockDomain,
    /// Nominal finite ordered mathematical basis.
    FiniteSpace,
    /// Finite ordinal set with fixed structural extent.
    IndexSet,
    /// One nominal declaration of closed finite alternatives.
    Enum,
    /// Nominal closed heterogeneous product declaration.
    Record,
    /// Occurrence of one exact closed product declaration.
    RecordInstance,
    // --- Realization Graph ---
    /// Discrete function space realizing a field.
    Space,
    /// Discretization choice.
    Discretization,
    /// Solver plan artifact.
    SolverPlan,
    /// Distribution partition.
    Partition,
    /// Hardware/deployment target.
    Target,
    /// Task mapping, priorities, deadlines.
    ExecutionSchedule,
    // --- Evidence & Artifact Graph ---
    /// Physical or numerical experiment.
    Experiment,
    /// Observation attached to an experiment.
    Observation,
    /// Dataset specification.
    Dataset,
    /// One execution with its manifest.
    Run,
    /// Content-addressed artifact.
    Artifact,
    /// Validity envelope of a model or surrogate.
    ValidityDomain,
    /// Verification/calibration evidence record.
    Evidence,
    // --- Action & Provenance Graph ---
    /// Committed typed transaction.
    Transaction,
    /// Human, service, or agent actor.
    Actor,
    /// Proposed or executed action.
    Action,
    /// Review record.
    Review,
    /// Approval record.
    Approval,
    /// Organization/project policy.
    Policy,
}

/// Runtime discriminant for the four graphs in the federation.
///
/// Type-level code should prefer [`GraphKind`]. This enum exists for erased
/// storage and schema validation, where an [`EntityKind`] must still retain
/// its graph boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphClass {
    /// Mathematical model truth.
    Semantic,
    /// Discretization, solver, and execution choices.
    Realization,
    /// Experiments, runs, artifacts, and validity.
    EvidenceArtifact,
    /// Transactions, actors, reviews, and decisions.
    ActionProvenance,
}

impl GraphClass {
    /// Stable serialization-facing name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Realization => "realization",
            Self::EvidenceArtifact => "evidence-artifact",
            Self::ActionProvenance => "action-provenance",
        }
    }
}

impl EntityKind {
    /// The graph containing this entity kind.
    #[must_use]
    pub const fn graph(self) -> GraphClass {
        match self {
            Self::Domain
            | Self::Representation
            | Self::Field
            | Self::Parameter
            | Self::Port
            | Self::Relation
            | Self::Activation
            | Self::Connection
            | Self::Record
            | Self::RecordInstance
            | Self::Observable
            | Self::Enum
            | Self::IndexSet
            | Self::FiniteSpace
            | Self::ClockDomain => GraphClass::Semantic,
            Self::Space
            | Self::Discretization
            | Self::SolverPlan
            | Self::Partition
            | Self::Target
            | Self::ExecutionSchedule => GraphClass::Realization,
            Self::Experiment
            | Self::Observation
            | Self::Dataset
            | Self::Run
            | Self::Artifact
            | Self::ValidityDomain
            | Self::Evidence => GraphClass::EvidenceArtifact,
            Self::Transaction
            | Self::Actor
            | Self::Action
            | Self::Review
            | Self::Approval
            | Self::Policy => GraphClass::ActionProvenance,
        }
    }
}

macro_rules! define_graph {
    ($(#[$doc:meta])* $name:ident = $s:literal) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl GraphKind for $name {
            const NAME: &'static str = $s;
        }
    };
}

define_graph!(
    /// Semantic Model Graph — the mathematical truth.
    Semantic = "semantic"
);
define_graph!(
    /// Realization Graph — discretization, solver plans, execution schedules.
    Realization = "realization"
);
define_graph!(
    /// Evidence & Artifact Graph — experiments, runs, artifacts, validity.
    EvidenceArtifact = "evidence-artifact"
);
define_graph!(
    /// Action & Provenance Graph — transactions, actors, reviews, decisions.
    ActionProvenance = "action-provenance"
);

macro_rules! define_entity {
    ($(#[$doc:meta])* $name:ident in $graph:ty) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;
        impl super::sealed::Sealed for $name {}
        impl super::Entity for $name {
            const KIND: super::EntityKind = super::EntityKind::$name;
            type Graph = $graph;
        }
    };
}

/// Zero-sized marker types for use as `Id<kinds::Field>` etc.
pub mod kinds {
    use super::{ActionProvenance, EvidenceArtifact, Realization, Semantic};

    define_entity!(
        /// See [`super::EntityKind::Domain`].
        Domain in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::Representation`].
        Representation in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::Field`].
        Field in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::Observable`].
        Observable in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::Parameter`].
        Parameter in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::Port`].
        Port in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::Relation`].
        Relation in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::Activation`].
        Activation in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::Connection`].
        Connection in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::ClockDomain`].
        ClockDomain in Semantic
    );

    define_entity!(
        /// See [`super::EntityKind::IndexSet`].
        IndexSet in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::Record`].
        Record in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::RecordInstance`].
        RecordInstance in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::Enum`].
        Enum in Semantic
    );
    define_entity!(
        /// See [`super::EntityKind::FiniteSpace`].
        FiniteSpace in Semantic
    );

    define_entity!(
        /// See [`super::EntityKind::Space`].
        Space in Realization
    );
    define_entity!(
        /// See [`super::EntityKind::Discretization`].
        Discretization in Realization
    );
    define_entity!(
        /// See [`super::EntityKind::SolverPlan`].
        SolverPlan in Realization
    );
    define_entity!(
        /// See [`super::EntityKind::Partition`].
        Partition in Realization
    );
    define_entity!(
        /// See [`super::EntityKind::Target`].
        Target in Realization
    );
    define_entity!(
        /// See [`super::EntityKind::ExecutionSchedule`].
        ExecutionSchedule in Realization
    );

    define_entity!(
        /// See [`super::EntityKind::Experiment`].
        Experiment in EvidenceArtifact
    );
    define_entity!(
        /// See [`super::EntityKind::Observation`].
        Observation in EvidenceArtifact
    );
    define_entity!(
        /// See [`super::EntityKind::Dataset`].
        Dataset in EvidenceArtifact
    );
    define_entity!(
        /// See [`super::EntityKind::Run`].
        Run in EvidenceArtifact
    );
    define_entity!(
        /// See [`super::EntityKind::Artifact`].
        Artifact in EvidenceArtifact
    );
    define_entity!(
        /// See [`super::EntityKind::ValidityDomain`].
        ValidityDomain in EvidenceArtifact
    );
    define_entity!(
        /// See [`super::EntityKind::Evidence`].
        Evidence in EvidenceArtifact
    );

    define_entity!(
        /// See [`super::EntityKind::Transaction`].
        Transaction in ActionProvenance
    );
    define_entity!(
        /// See [`super::EntityKind::Actor`].
        Actor in ActionProvenance
    );
    define_entity!(
        /// See [`super::EntityKind::Action`].
        Action in ActionProvenance
    );
    define_entity!(
        /// See [`super::EntityKind::Review`].
        Review in ActionProvenance
    );
    define_entity!(
        /// See [`super::EntityKind::Approval`].
        Approval in ActionProvenance
    );
    define_entity!(
        /// See [`super::EntityKind::Policy`].
        Policy in ActionProvenance
    );
}
