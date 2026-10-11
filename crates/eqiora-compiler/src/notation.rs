//! Full-model declaration labels, independent of physical meaning and graph projection.

use crate::identity::FullElaborationIdentity;
use eqiora_core::{RawId, Span};
use eqiora_lang::{Notation, NotationLabel, NotationProfile};
use std::collections::BTreeMap;

mod kernel;
mod resolve;
#[cfg(test)]
mod tests;

/// The physical role of a declaration-label quantity, not an expression operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuantityRole {
    /// A Field, Parameter, or causal signal value.
    Value,
    /// Scalar conserving across quantity.
    Across,
    /// Scalar conserving through quantity.
    Through,
    /// Boundary connector trace quantity.
    Trace,
    /// Boundary connector outward flux quantity.
    Flux,
}

impl QuantityRole {
    /// Stable human-readable role name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Value => "value",
            Self::Across => "across",
            Self::Through => "through",
            Self::Trace => "trace",
            Self::Flux => "flux",
        }
    }
}

/// Exact quantity in one complete Model rendering scope.
///
/// Occurrence and family identity come from the compiler's existing elaboration
/// keys, never from labels, source names, discovery order, or projected graph IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QuantityIdentity {
    pub(crate) scope: FullElaborationIdentity,
    pub(crate) occurrence: FullElaborationIdentity,
    pub(crate) role: QuantityRole,
    pub(crate) member: Option<FullElaborationIdentity>,
}

impl QuantityIdentity {
    /// Complete Model rendering scope.
    #[must_use]
    pub const fn scope(self) -> FullElaborationIdentity {
        self.scope
    }
    /// Exact declaration occurrence, including its component instance.
    #[must_use]
    pub const fn occurrence(self) -> FullElaborationIdentity {
        self.occurrence
    }
    /// Connector role or ordinary value.
    #[must_use]
    pub const fn role(self) -> QuantityRole {
        self.role
    }
    /// Exact selected boundary of a boundary-family member.
    #[must_use]
    pub const fn member(self) -> Option<FullElaborationIdentity> {
        self.member
    }
}

impl std::fmt::Display for QuantityIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.scope, self.occurrence, self.role.name())?;
        if let Some(member) = self.member {
            write!(f, ":{member}")?;
        }
        Ok(())
    }
}

/// One immutable quantity label with exact occurrence and source traceability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedNotation {
    identity: QuantityIdentity,
    selector: String,
    graph_id: Option<RawId>,
    definition: Option<Span>,
    instance: Option<Span>,
    label: NotationLabel,
}

impl ResolvedNotation {
    /// Exact lookup key; repeated references share this entry.
    #[must_use]
    pub const fn identity(&self) -> QuantityIdentity {
        self.identity
    }
    /// Source-qualified query spelling; never the equality key.
    #[must_use]
    pub fn selector(&self) -> &str {
        &self.selector
    }
    /// Retained Kernel entity, absent for eliminated public exposures.
    #[must_use]
    pub const fn graph_id(&self) -> Option<RawId> {
        self.graph_id
    }
    /// Original declaration location, when source provenance was retained.
    #[must_use]
    pub const fn definition_span(&self) -> Option<&Span> {
        self.definition.as_ref()
    }
    /// Owning instance location, when source provenance was retained.
    #[must_use]
    pub const fn instance_span(&self) -> Option<&Span> {
        self.instance.as_ref()
    }
    /// Checked structural label; qualification never reparses source notation.
    #[must_use]
    pub const fn label(&self) -> &NotationLabel {
        &self.label
    }
    /// The shared declaration-label projection for one output profile.
    #[must_use]
    pub fn render(&self, profile: NotationProfile) -> String {
        self.label.render(profile)
    }
}

/// Labels resolved once against the complete Model, inherited unchanged by views.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelNotation {
    entries: BTreeMap<QuantityIdentity, ResolvedNotation>,
}

impl ModelNotation {
    pub(crate) fn retain_source_files(&mut self, keep: impl Fn(&str) -> bool) {
        for entry in self.entries.values_mut() {
            if entry
                .definition
                .as_ref()
                .is_some_and(|span| !keep(&span.file))
            {
                entry.definition = None;
            }
            if entry
                .instance
                .as_ref()
                .is_some_and(|span| !keep(&span.file))
            {
                entry.instance = None;
            }
        }
    }
    /// Full-scope entries in exact identity order, independent of discovery order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &ResolvedNotation> {
        self.entries.values()
    }
    /// Resolve an exact identity, including its full Model scope.
    #[must_use]
    pub fn get(&self, identity: QuantityIdentity) -> Option<&ResolvedNotation> {
        self.entries.get(&identity)
    }
    /// Select a subview without re-running collision resolution.
    /// Repeated identities retain one label; foreign identities are not captured.
    pub fn view(
        &self,
        identities: impl IntoIterator<Item = QuantityIdentity>,
    ) -> Vec<&ResolvedNotation> {
        let keys = identities
            .into_iter()
            .filter(|key| self.entries.contains_key(key))
            .collect::<std::collections::BTreeSet<_>>();
        keys.into_iter()
            .filter_map(|key| self.entries.get(&key))
            .collect()
    }

    pub(crate) fn resolve(specs: Vec<NotationSpec>) -> Self {
        resolve::resolve(specs)
    }
    pub(crate) fn clear_source_locations(&mut self) {
        for entry in self.entries.values_mut() {
            entry.definition = None;
            entry.instance = None;
        }
    }
}

#[derive(Clone)]
pub(crate) struct NotationSpec {
    pub(crate) graph_identity: Option<FullElaborationIdentity>,
    pub(crate) identity: QuantityIdentity,
    pub(crate) selector: String,
    pub(crate) graph_id: Option<RawId>,
    pub(crate) definition: Option<Span>,
    pub(crate) instance: Option<Span>,
    pub(crate) declared: Option<Notation>,
    pub(crate) qualifiers: Vec<Notation>,
    pub(crate) instance_path: Vec<String>,
    pub(crate) declaration: String,
    pub(crate) role_name: String,
}
