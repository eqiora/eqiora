//! Authored equality authority and exact conforming trace endpoints.
use crate::invalid_realization;
use eqiora_core::{Diagnostic, Id, RawId, entity::kinds};

/// Authored authority for identifying two full-value Field traces.
///
/// A physical Interface declaration alone is never equality authority. The
/// semantic lowerer must prove the selected Relation root is exactly the trace
/// equality, and independently authenticate its interface flux balance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConformingTraceSource {
    /// A conserving Connection with its admitted Port contracts.
    ConservingConnection(Id<kinds::Connection>),
    /// One explicit trace equality root of an authored Relation.
    RelationEquality {
        /// Relation that owns the equality.
        relation: Id<kinds::Relation>,
        /// Zero-based index of an exact root in that Relation's expression DAG.
        root_index: u32,
    },
}

impl ConformingTraceSource {
    fn order_key(self) -> (u8, ulid::Ulid, u32) {
        match self {
            Self::ConservingConnection(connection) => (0, connection.ulid(), 0),
            Self::RelationEquality {
                relation,
                root_index,
            } => (1, relation.ulid(), root_index),
        }
    }
    /// Require the conserving Port/Connection profile used by coupled mechanics.
    ///
    /// # Errors
    /// Rejects an authored Relation equality, which does not establish a Port contract.
    pub fn conserving_connection(self) -> Result<Id<kinds::Connection>, Diagnostic> {
        match self {
            Self::ConservingConnection(connection) => Ok(connection),
            Self::RelationEquality { .. } => Err(invalid_realization(
                "this trace consumer requires a conserving Connection, not a Relation equality",
            )),
        }
    }

    /// Semantic node owning the equality; the source also retains its exact root.
    #[must_use]
    pub fn owner(self) -> RawId {
        match self {
            Self::ConservingConnection(connection) => connection.erase(),
            Self::RelationEquality { relation, .. } => relation.erase(),
        }
    }
}

impl Ord for ConformingTraceSource {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.order_key().cmp(&other.order_key())
    }
}

impl PartialOrd for ConformingTraceSource {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// One Domain/Field endpoint participating in an exact trace quotient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceFieldEndpoint {
    pub(super) domain: Id<kinds::Domain>,
    pub(super) field: Id<kinds::Field>,
}

impl TraceFieldEndpoint {
    /// Select one exact Field trace on one exact Domain.
    #[must_use]
    pub const fn new(domain: Id<kinds::Domain>, field: Id<kinds::Field>) -> Self {
        Self { domain, field }
    }

    /// Selected Domain.
    #[must_use]
    pub const fn domain(self) -> Id<kinds::Domain> {
        self.domain
    }

    /// Selected Field.
    #[must_use]
    pub const fn field(self) -> Id<kinds::Field> {
        self.field
    }
}

/// Equality quotient of two conforming Field traces with exact authored authority.
///
/// This is a numerical identity choice, not a physical interface definition.
/// The semantic lowerer remains responsible for proving conserving Connection
/// semantics or an explicit Relation equality, plus compatible Field shape, support,
/// units, frame, and orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConformingTraceQuotient {
    pub(super) source: ConformingTraceSource,
    pub(super) endpoints: [TraceFieldEndpoint; 2],
}

impl ConformingTraceQuotient {
    /// Construct a canonically ordered cross-Domain trace quotient.
    ///
    /// # Errors
    /// Returns `EQ0807` when both endpoints belong to the same Domain.
    pub fn new(
        source: ConformingTraceSource,
        first: TraceFieldEndpoint,
        second: TraceFieldEndpoint,
    ) -> Result<Self, Diagnostic> {
        if first.domain == second.domain {
            return Err(invalid_realization(
                "a conforming trace quotient must join Fields on distinct Domains",
            ));
        }
        let mut endpoints = [first, second];
        endpoints.sort_by(super::endpoint_order);
        Ok(Self { source, endpoints })
    }

    /// Exact authored equality selected by the semantic lowerer.
    #[must_use]
    pub const fn source(self) -> ConformingTraceSource {
        self.source
    }

    /// Canonically ordered trace endpoints.
    #[must_use]
    pub const fn endpoints(self) -> [TraceFieldEndpoint; 2] {
        self.endpoints
    }
}
