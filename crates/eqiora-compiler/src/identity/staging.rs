//! Stage full and projected identities before transaction allocation.
use super::*;

impl<P: ShortIdProjector> StagingIdAllocator<P> {
    /// Create an allocator with an injected projector and explicit limits.
    /// This supports forced-collision tests without weakening production IDs.
    #[must_use]
    pub fn with_projector_and_limits(projector: P, limits: ElaborationIdentityLimits) -> Self {
        Self {
            projector,
            limits,
            by_identity: Vec::new(),
            by_projection: Vec::new(),
            model_view: None,
        }
    }

    /// Stage one key and return its full identity. Re-staging the same exact
    /// key is idempotent. A full-digest or projected-ID collision fails before
    /// either index is mutated.
    pub fn stage(&mut self, key: &ElaborationKey) -> Result<FullElaborationIdentity, Diagnostic> {
        self.stage_projection(key, None)
    }

    pub(crate) fn stage_bound_clock(
        &mut self,
        key: &ElaborationKey,
        id: eqiora_core::Id<eqiora_core::entity::kinds::ClockDomain>,
    ) -> Result<FullElaborationIdentity, Diagnostic> {
        if key.entity_kind() != EntityKind::ClockDomain {
            return Err(identity_error(
                "external clock identity requires a ClockDomain key",
            ));
        }
        self.stage_projection(key, Some(id.ulid().to_bytes()))
    }

    pub(crate) fn stage_bound_entity(
        &mut self,
        key: &ElaborationKey,
        id: eqiora_core::RawId,
    ) -> Result<FullElaborationIdentity, Diagnostic> {
        if key.entity_kind() != id.kind() {
            return Err(identity_error(
                "supplied nominal identity has a different entity kind",
            ));
        }
        self.stage_projection(key, Some(id.ulid().to_bytes()))
    }

    fn stage_projection(
        &mut self,
        key: &ElaborationKey,
        supplied: Option<[u8; 16]>,
    ) -> Result<FullElaborationIdentity, Diagnostic> {
        let canonical_key = key.canonical_bytes()?;
        let identity = FullElaborationIdentity(Sha256::digest(&canonical_key).into());
        let kind = key.entity_kind();
        let projected = supplied.unwrap_or_else(|| self.projector.project(identity));
        if let Ok(index) = self
            .by_identity
            .binary_search_by_key(&identity, |entry| entry.identity)
        {
            let existing = &self.by_identity[index];
            if existing.canonical_key.as_ref() != canonical_key.as_slice()
                || existing.kind != kind
                || existing.projected != projected
            {
                return Err(identity_error(
                    "distinct canonical elaboration keys share one full SHA-256 identity",
                ));
            }
            return Ok(identity);
        }

        let staged_count = self
            .by_identity
            .len()
            .checked_add(usize::from(self.model_view.is_some()))
            .ok_or_else(|| identity_error("staged identity count overflows usize"))?;
        if staged_count >= self.limits.max_staged_identities {
            return Err(identity_error(format!(
                "elaboration exceeds the {} staged identity limit",
                self.limits.max_staged_identities
            )));
        }

        let projection_search = self
            .by_projection
            .binary_search_by(|entry| (entry.kind, entry.projected).cmp(&(kind, projected)));
        if let Ok(index) = projection_search {
            let existing = self.by_projection[index].identity;
            if existing != identity {
                return Err(identity_error(format!(
                    "projected {:?} identifier collision between full identities {existing} and {identity}",
                    kind
                )));
            }
        }

        self.by_identity
            .try_reserve(1)
            .map_err(|_| identity_error("cannot reserve staged elaboration identity"))?;
        self.by_projection
            .try_reserve(1)
            .map_err(|_| identity_error("cannot reserve projected identity index"))?;

        let identity_index = self
            .by_identity
            .binary_search_by_key(&identity, |entry| entry.identity)
            .unwrap_or_else(|index| index);
        self.by_identity.insert(
            identity_index,
            StagedIdentity {
                identity,
                kind,
                projected,
                canonical_key: canonical_key.into_boxed_slice(),
            },
        );
        let projection_index = projection_search.unwrap_or_else(|index| index);
        self.by_projection.insert(
            projection_index,
            ProjectionIndex {
                kind,
                projected,
                identity,
            },
        );
        Ok(identity)
    }

    /// Stage the one root Model ontology view. Re-staging the exact same key is
    /// idempotent; attempting to stage a second root fails before replacement.
    pub fn stage_model_view(
        &mut self,
        key: &ModelViewKey,
    ) -> Result<FullElaborationIdentity, Diagnostic> {
        let canonical_key = key.canonical_bytes()?;
        let identity = FullElaborationIdentity(Sha256::digest(&canonical_key).into());
        if let Some(existing) = &self.model_view {
            if existing.identity == identity
                && existing.canonical_key.as_ref() == canonical_key.as_slice()
            {
                return Ok(identity);
            }
            if existing.identity == identity {
                return Err(identity_error(
                    "distinct canonical Model view keys share one full SHA-256 identity",
                ));
            }
            return Err(identity_error(
                "one elaboration cannot stage more than one root Model view",
            ));
        }
        if self.by_identity.len() >= self.limits.max_staged_identities {
            return Err(identity_error(format!(
                "elaboration exceeds the {} staged identity limit",
                self.limits.max_staged_identities
            )));
        }

        self.model_view = Some(StagedModelView {
            identity,
            projected: self.projector.project(identity),
            canonical_key: canonical_key.into_boxed_slice(),
        });
        Ok(identity)
    }

    /// Seal all staged identities. This is the first object that can expose a
    /// typed graph ID to transaction construction.
    #[must_use]
    pub fn finish(self) -> StagedIdentities {
        StagedIdentities {
            by_identity: self.by_identity.into_boxed_slice(),
            model_view: self.model_view,
        }
    }
}
