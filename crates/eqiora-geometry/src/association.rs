//! Fail-closed association between two exact correspondence revisions.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::Id;
use eqiora_core::entity::kinds;
use eqiora_schema::kernel::BoundarySide;

use crate::{GeometryMeshCorrespondence, GeometryRevisionReference};

type DomainId = Id<kinds::Domain>;

/// One candidate body identity edge proposed by a geometry regeneration adapter.
///
/// Source and target IDs are revision-scoped. Equal ULID bytes do not prove
/// retention; the candidate is still required and validated as part of the
/// complete relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyAssociationCandidate {
    source: DomainId,
    target: DomainId,
}

impl BodyAssociationCandidate {
    /// Propose one source-to-target body association.
    #[must_use]
    pub const fn new(source: DomainId, target: DomainId) -> Self {
        Self { source, target }
    }

    /// Body Domain in the source correspondence.
    #[must_use]
    pub const fn source(&self) -> DomainId {
        self.source
    }

    /// Body Domain in the target correspondence.
    #[must_use]
    pub const fn target(&self) -> DomainId {
        self.target
    }
}

/// One accepted one-to-one body association.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RetainedBodyPair {
    source: DomainId,
    target: DomainId,
}

impl RetainedBodyPair {
    /// Source-revision body Domain.
    #[must_use]
    pub const fn source(&self) -> DomainId {
        self.source
    }

    /// Target-revision body Domain.
    #[must_use]
    pub const fn target(&self) -> DomainId {
        self.target
    }
}

/// One accepted boundary association derived from a retained parent pair and role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RetainedBoundaryPair {
    source: DomainId,
    target: DomainId,
    source_parent: DomainId,
    target_parent: DomainId,
    axis: usize,
    side: BoundarySide,
}

impl RetainedBoundaryPair {
    /// Source-revision boundary Domain.
    #[must_use]
    pub const fn source(&self) -> DomainId {
        self.source
    }

    /// Target-revision boundary Domain.
    #[must_use]
    pub const fn target(&self) -> DomainId {
        self.target
    }

    /// Source boundary's retained parent body.
    #[must_use]
    pub const fn source_parent(&self) -> DomainId {
        self.source_parent
    }

    /// Target boundary's retained parent body.
    #[must_use]
    pub const fn target_parent(&self) -> DomainId {
        self.target_parent
    }

    /// Shared Cartesian boundary axis.
    #[must_use]
    pub const fn axis(&self) -> usize {
        self.axis
    }

    /// Shared Cartesian boundary side.
    #[must_use]
    pub const fn side(&self) -> BoundarySide {
        self.side
    }
}

/// Total one-to-one association between two exact geometry correspondences.
///
/// Only this accepted type represents retention. Zero-, one-to-many,
/// many-to-one, and non-functional candidate relations remain typed errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainedGeometryAssociation {
    source_revision: GeometryRevisionReference,
    target_revision: GeometryRevisionReference,
    bodies: Box<[RetainedBodyPair]>,
    boundaries: Box<[RetainedBoundaryPair]>,
}

impl RetainedGeometryAssociation {
    /// Close a total body bijection and derive all boundary pairs by
    /// `(retained parent, axis, side)`.
    ///
    /// Duplicate candidate edges are set-equivalent and canonicalized away.
    /// No Domain ULID is implicitly carried from source to target.
    ///
    /// # Errors
    /// Returns a typed [`RetentionRejection`] unless the candidate relation is
    /// a total bijection over both body sets and every paired body has the same
    /// complete Cartesian boundary-role set.
    pub fn validate(
        source: &GeometryMeshCorrespondence,
        target: &GeometryMeshCorrespondence,
        candidates: Vec<BodyAssociationCandidate>,
    ) -> Result<Self, RetentionRejection> {
        let source_ids = source
            .bodies()
            .iter()
            .map(|body| body.domain().erase())
            .collect::<BTreeSet<_>>();
        let target_ids = target
            .bodies()
            .iter()
            .map(|body| body.domain().erase())
            .collect::<BTreeSet<_>>();

        let mut candidates = candidates;
        candidates.sort_by_key(|candidate| (candidate.source.erase(), candidate.target.erase()));
        candidates.dedup();
        for candidate in &candidates {
            if !source_ids.contains(&candidate.source.erase()) {
                return Err(RetentionRejection::UnknownSourceBody {
                    source: candidate.source,
                });
            }
            if !target_ids.contains(&candidate.target.erase()) {
                return Err(RetentionRejection::UnknownTargetBody {
                    target: candidate.target,
                });
            }
        }

        let mut targets_by_source: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut sources_by_target: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for candidate in &candidates {
            targets_by_source
                .entry(candidate.source.erase())
                .or_default()
                .push(candidate.target);
            sources_by_target
                .entry(candidate.target.erase())
                .or_default()
                .push(candidate.source);
        }
        let splits = targets_by_source
            .iter()
            .filter(|(_, targets)| targets.len() > 1)
            .map(|(_, targets)| targets.clone())
            .collect::<Vec<_>>();
        let merges = sources_by_target
            .iter()
            .filter(|(_, sources)| sources.len() > 1)
            .map(|(_, sources)| sources.clone())
            .collect::<Vec<_>>();
        if !splits.is_empty() && !merges.is_empty() {
            return Err(RetentionRejection::Ambiguous {
                candidates: candidates.into_boxed_slice(),
            });
        }
        if let Some((source_id, targets)) = targets_by_source
            .iter()
            .find(|(_, targets)| targets.len() > 1)
        {
            let Some(source_body) = source
                .bodies()
                .iter()
                .find(|body| body.domain().erase() == *source_id)
            else {
                return Err(RetentionRejection::Ambiguous {
                    candidates: candidates.into_boxed_slice(),
                });
            };
            return Err(RetentionRejection::Split {
                source: source_body.domain(),
                targets: targets.clone().into_boxed_slice(),
            });
        }
        if let Some((target_id, sources)) = sources_by_target
            .iter()
            .find(|(_, sources)| sources.len() > 1)
        {
            let Some(target_body) = target
                .bodies()
                .iter()
                .find(|body| body.domain().erase() == *target_id)
            else {
                return Err(RetentionRejection::Ambiguous {
                    candidates: candidates.into_boxed_slice(),
                });
            };
            return Err(RetentionRejection::Merged {
                sources: sources.clone().into_boxed_slice(),
                target: target_body.domain(),
            });
        }

        let missing_sources = source
            .bodies()
            .iter()
            .filter(|body| !targets_by_source.contains_key(&body.domain().erase()))
            .map(|body| body.domain())
            .collect::<Vec<_>>();
        let missing_targets = target
            .bodies()
            .iter()
            .filter(|body| !sources_by_target.contains_key(&body.domain().erase()))
            .map(|body| body.domain())
            .collect::<Vec<_>>();
        if !missing_sources.is_empty() || !missing_targets.is_empty() {
            return Err(RetentionRejection::Missing {
                sources: missing_sources.into_boxed_slice(),
                targets: missing_targets.into_boxed_slice(),
            });
        }

        let mut body_pairs = Vec::with_capacity(candidates.len());
        let mut boundary_pairs = Vec::new();
        for candidate in candidates {
            let Some(source_body) = source.body(candidate.source) else {
                return Err(RetentionRejection::UnknownSourceBody {
                    source: candidate.source,
                });
            };
            let Some(target_body) = target.body(candidate.target) else {
                return Err(RetentionRejection::UnknownTargetBody {
                    target: candidate.target,
                });
            };
            let source_roles = source
                .boundaries_of(source_body.domain())
                .map(|boundary| ((boundary.axis(), boundary.side()), boundary.domain()))
                .collect::<BTreeMap<_, _>>();
            let target_roles = target
                .boundaries_of(target_body.domain())
                .map(|boundary| ((boundary.axis(), boundary.side()), boundary.domain()))
                .collect::<BTreeMap<_, _>>();
            if source_roles.keys().ne(target_roles.keys()) {
                return Err(RetentionRejection::BoundaryRoleMismatch {
                    source: source_body.domain(),
                    target: target_body.domain(),
                });
            }
            body_pairs.push(RetainedBodyPair {
                source: source_body.domain(),
                target: target_body.domain(),
            });
            for ((axis, side), source_boundary) in source_roles {
                let Some(&target_boundary) = target_roles.get(&(axis, side)) else {
                    return Err(RetentionRejection::BoundaryRoleMismatch {
                        source: source_body.domain(),
                        target: target_body.domain(),
                    });
                };
                boundary_pairs.push(RetainedBoundaryPair {
                    source: source_boundary,
                    target: target_boundary,
                    source_parent: source_body.domain(),
                    target_parent: target_body.domain(),
                    axis,
                    side,
                });
            }
        }
        body_pairs.sort_by_key(|pair| pair.source.erase());
        boundary_pairs.sort_by_key(|pair| {
            (
                pair.source_parent.erase(),
                pair.axis,
                side_key(pair.side),
                pair.source.erase(),
            )
        });

        Ok(Self {
            source_revision: source.geometry_revision(),
            target_revision: target.geometry_revision(),
            bodies: body_pairs.into_boxed_slice(),
            boundaries: boundary_pairs.into_boxed_slice(),
        })
    }

    /// Exact source geometry revision.
    #[must_use]
    pub const fn source_revision(&self) -> GeometryRevisionReference {
        self.source_revision
    }

    /// Exact target geometry revision.
    #[must_use]
    pub const fn target_revision(&self) -> GeometryRevisionReference {
        self.target_revision
    }

    /// Canonically ordered total body bijection.
    #[must_use]
    pub const fn bodies(&self) -> &[RetainedBodyPair] {
        &self.bodies
    }

    /// Boundary pairs derived from retained parents and Cartesian roles.
    #[must_use]
    pub const fn boundaries(&self) -> &[RetainedBoundaryPair] {
        &self.boundaries
    }
}

const fn side_key(side: BoundarySide) -> u8 {
    match side {
        BoundarySide::Lower => 0,
        BoundarySide::Upper => 1,
    }
}

/// Typed reason why a cross-revision relation cannot prove retention.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetentionRejection {
    /// One or more bodies have no candidate on the opposite revision.
    Missing {
        sources: Box<[DomainId]>,
        targets: Box<[DomainId]>,
    },
    /// One source body has several target candidates.
    Split {
        source: DomainId,
        targets: Box<[DomainId]>,
    },
    /// Several source bodies have one target candidate.
    Merged {
        sources: Box<[DomainId]>,
        target: DomainId,
    },
    /// The candidate relation contains both split and merge structure.
    Ambiguous {
        candidates: Box<[BodyAssociationCandidate]>,
    },
    /// A candidate names no source body in the source correspondence.
    UnknownSourceBody { source: DomainId },
    /// A candidate names no target body in the target correspondence.
    UnknownTargetBody { target: DomainId },
    /// Retained parents expose different Cartesian boundary-role sets.
    BoundaryRoleMismatch { source: DomainId, target: DomainId },
}

impl fmt::Display for RetentionRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "geometry retention rejected: {self:?}")
    }
}

impl std::error::Error for RetentionRejection {}
