//! Strongly typed identifiers.
//!
//! `Id<kinds::Field>` and `Id<kinds::SolverPlan>` are distinct types: mixing
//! them is a compile error, not a runtime surprise. Type-erased [`RawId`]
//! exists for storage and wire boundaries only, and recovers the typed form
//! exclusively through the checked [`RawId::downcast`].

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

use ulid::Ulid;

use crate::entity::{Entity, EntityKind};

/// Strongly typed identifier for an entity in the Graph Federation.
///
/// The phantom parameter carries both the entity kind and (through
/// [`Entity::Graph`]) the graph it belongs to. `fn() -> E` keeps the type
/// covariant and `Send + Sync` regardless of `E`.
pub struct Id<E: Entity> {
    ulid: Ulid,
    _marker: PhantomData<fn() -> E>,
}

impl<E: Entity> Id<E> {
    /// Mint a fresh, globally unique identifier.
    #[must_use]
    pub fn new() -> Self {
        Self::from_ulid(Ulid::generate())
    }

    /// Rebuild from a raw ULID (deserialization path).
    #[must_use]
    pub const fn from_ulid(ulid: Ulid) -> Self {
        Self {
            ulid,
            _marker: PhantomData,
        }
    }

    /// The underlying ULID.
    #[must_use]
    pub const fn ulid(&self) -> Ulid {
        self.ulid
    }

    /// Erase the type for storage or wire transfer.
    #[must_use]
    pub fn erase(self) -> RawId {
        RawId {
            kind: E::KIND,
            ulid: self.ulid,
        }
    }
}

impl<E: Entity> Default for Id<E> {
    fn default() -> Self {
        Self::new()
    }
}

// Manual impls: derives would add unnecessary bounds on `E`.
impl<E: Entity> Clone for Id<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Entity> Copy for Id<E> {}
impl<E: Entity> PartialEq for Id<E> {
    fn eq(&self, other: &Self) -> bool {
        self.ulid == other.ulid
    }
}
impl<E: Entity> Eq for Id<E> {}
impl<E: Entity> Hash for Id<E> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ulid.hash(state);
    }
}
impl<E: Entity> fmt::Debug for Id<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Id<{:?}>({})", E::KIND, self.ulid)
    }
}
impl<E: Entity> fmt::Display for Id<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ulid)
    }
}

/// Type-erased identifier for storage and wire boundaries.
///
/// Carries the runtime [`EntityKind`] so the typed form can be recovered —
/// but only through the checked [`RawId::downcast`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RawId {
    kind: EntityKind,
    ulid: Ulid,
}

impl RawId {
    /// The runtime entity kind.
    #[must_use]
    pub const fn kind(&self) -> EntityKind {
        self.kind
    }

    /// The underlying ULID.
    #[must_use]
    pub const fn ulid(&self) -> Ulid {
        self.ulid
    }

    /// Recover the typed identifier. Returns `None` on a kind mismatch —
    /// there is no unchecked path back to `Id<E>`.
    #[must_use]
    pub fn downcast<E: Entity>(self) -> Option<Id<E>> {
        (self.kind == E::KIND).then(|| Id::from_ulid(self.ulid))
    }
}

impl<E: Entity> From<Id<E>> for RawId {
    fn from(id: Id<E>) -> Self {
        id.erase()
    }
}

impl fmt::Display for RawId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}:{}", self.kind, self.ulid)
    }
}
