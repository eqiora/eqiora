//! Immutable source provenance for elaborated identities.
//!
//! Source locations are deliberately stored beside, never inside, semantic
//! elaboration identities. Formatting and file relocation may therefore alter
//! this sidecar without changing projected graph identifiers.

use core::cmp::Ordering;
use std::collections::BTreeSet;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, RawId, Span};

use crate::identity::FullElaborationIdentity;

/// Resource limits for one elaboration provenance sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvenanceLimits {
    /// Maximum distinct semantic identities with source provenance.
    pub max_entries: usize,
    /// Maximum distinct complete source origins attached to one identity.
    pub max_origins_per_entry: usize,
    /// Maximum binding spans attached to one identity.
    pub max_binding_spans_per_entry: usize,
    /// Maximum UTF-8 bytes in one workspace-relative source path.
    pub max_source_path_bytes: usize,
    /// Maximum source-path bytes summed across the complete sidecar.
    pub max_total_source_path_bytes: usize,
}

impl Default for ProvenanceLimits {
    fn default() -> Self {
        Self {
            max_entries: 1_000_000,
            max_origins_per_entry: 4_096,
            max_binding_spans_per_entry: 4_096,
            max_source_path_bytes: 4_096,
            max_total_source_path_bytes: 64 * 1_024 * 1_024,
        }
    }
}

/// One complete source origin for an elaborated semantic identity.
///
/// Definition, instance, and binding locations remain an indivisible tuple.
/// This prevents provenance normalization from synthesizing an origin by
/// independently selecting the smallest span for each role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElaborationSourceOrigin {
    definition_span: Span,
    instance_span: Span,
    binding_spans: Box<[Span]>,
}

impl ElaborationSourceOrigin {
    /// Construct one complete origin.
    ///
    /// Binding spans are normalized immediately. Compilation limits and span
    /// validity are checked when the origin is inserted into a builder.
    #[must_use]
    pub fn new(definition_span: Span, instance_span: Span, mut binding_spans: Vec<Span>) -> Self {
        binding_spans.sort_unstable_by(compare_span);
        binding_spans.dedup();
        Self {
            definition_span,
            instance_span,
            binding_spans: binding_spans.into_boxed_slice(),
        }
    }

    /// Source declaration from which the canonical entity was elaborated.
    #[must_use]
    pub const fn definition_span(&self) -> &Span {
        &self.definition_span
    }

    /// Source occurrence of the component instance owning the entity.
    #[must_use]
    pub const fn instance_span(&self) -> &Span {
        &self.instance_span
    }

    /// Binding locations in deterministic `(file, start, end)` order.
    #[must_use]
    pub const fn binding_spans(&self) -> &[Span] {
        &self.binding_spans
    }
}

/// Source locations associated with one elaborated semantic identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElaborationProvenance {
    origins: Box<[ElaborationSourceOrigin]>,
}

impl ElaborationProvenance {
    /// Complete origins in deterministic tuple order.
    #[must_use]
    pub const fn origins(&self) -> &[ElaborationSourceOrigin] {
        &self.origins
    }

    /// Source declaration from the first complete origin.
    ///
    /// This compatibility accessor and the other singular accessors always
    /// select fields from the same origin.
    #[must_use]
    pub const fn definition_span(&self) -> &Span {
        self.origins[0].definition_span()
    }

    /// Source occurrence from the first complete origin.
    #[must_use]
    pub const fn instance_span(&self) -> &Span {
        self.origins[0].instance_span()
    }

    /// Binding locations from the first complete origin.
    #[must_use]
    pub const fn binding_spans(&self) -> &[Span] {
        self.origins[0].binding_spans()
    }
}

/// Bounded builder for one immutable provenance sidecar.
#[derive(Debug)]
pub struct ProvenanceBuilder {
    limits: ProvenanceLimits,
    total_source_path_bytes: usize,
    entries: Vec<ProvenanceEntry>,
    graph_ids: BTreeSet<RawId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProvenanceEntry {
    identity: FullElaborationIdentity,
    graph_id: Option<RawId>,
    provenance: ElaborationProvenance,
}

impl ProvenanceBuilder {
    /// Create an empty builder with default resource limits.
    #[must_use]
    pub fn new() -> Self {
        Self::with_limits(ProvenanceLimits::default())
    }

    /// Create an empty builder with explicit compilation limits.
    #[must_use]
    pub const fn with_limits(limits: ProvenanceLimits) -> Self {
        Self {
            limits,
            total_source_path_bytes: 0,
            entries: Vec::new(),
            graph_ids: BTreeSet::new(),
        }
    }

    /// Attach definition, instance, and binding spans to one exact semantic
    /// identity. Duplicate identities fail closed rather than overwriting
    /// earlier provenance.
    pub fn insert(
        &mut self,
        identity: FullElaborationIdentity,
        definition_span: Span,
        instance_span: Span,
        binding_spans: impl IntoIterator<Item = Span>,
    ) -> Result<(), Diagnostic> {
        self.entry_insertion_index(identity)?;
        let origin = collect_origin(definition_span, instance_span, binding_spans, self.limits)?;
        self.insert_origins_inner(identity, None, [origin])
    }

    /// Attach multiple complete source origins to one exact semantic identity.
    ///
    /// Origins are sorted and deduplicated as complete tuples. Duplicate
    /// identities fail closed rather than extending or overwriting an entry.
    pub fn insert_origins(
        &mut self,
        identity: FullElaborationIdentity,
        origins: impl IntoIterator<Item = ElaborationSourceOrigin>,
    ) -> Result<(), Diagnostic> {
        self.insert_origins_inner(identity, None, origins)
    }

    /// Attach provenance to a projected Semantic Kernel node as well as its
    /// collision-checked full identity.
    pub fn insert_graph(
        &mut self,
        graph_id: RawId,
        identity: FullElaborationIdentity,
        definition_span: Span,
        instance_span: Span,
        binding_spans: impl IntoIterator<Item = Span>,
    ) -> Result<(), Diagnostic> {
        if !self.graph_ids.insert(graph_id) {
            return Err(provenance_error(format!(
                "provenance for graph node {graph_id} already exists"
            )));
        }
        let result = (|| {
            self.entry_insertion_index(identity)?;
            let origin =
                collect_origin(definition_span, instance_span, binding_spans, self.limits)?;
            self.insert_origins_inner(identity, Some(graph_id), [origin])
        })();
        if result.is_err() {
            self.graph_ids.remove(&graph_id);
        }
        result
    }

    /// Attach multiple complete origins to a projected Semantic Kernel node
    /// and its collision-checked full identity.
    pub fn insert_graph_origins(
        &mut self,
        graph_id: RawId,
        identity: FullElaborationIdentity,
        origins: impl IntoIterator<Item = ElaborationSourceOrigin>,
    ) -> Result<(), Diagnostic> {
        if !self.graph_ids.insert(graph_id) {
            return Err(provenance_error(format!(
                "provenance for graph node {graph_id} already exists"
            )));
        }
        let result = self.insert_origins_inner(identity, Some(graph_id), origins);
        if result.is_err() {
            self.graph_ids.remove(&graph_id);
        }
        result
    }

    fn insert_origins_inner(
        &mut self,
        identity: FullElaborationIdentity,
        graph_id: Option<RawId>,
        origins: impl IntoIterator<Item = ElaborationSourceOrigin>,
    ) -> Result<(), Diagnostic> {
        let insertion_index = self.entry_insertion_index(identity)?;

        let mut collected_origins = Vec::new();
        let mut binding_span_count = 0usize;
        for origin in origins {
            validate_origin(&origin, self.limits)?;
            let insertion_index = match collected_origins
                .binary_search_by(|candidate| compare_origin(candidate, &origin))
            {
                Ok(_) => continue,
                Err(index) => index,
            };
            if collected_origins.len() >= self.limits.max_origins_per_entry {
                return Err(provenance_error(format!(
                    "elaboration provenance exceeds the {} origin-per-entry limit",
                    self.limits.max_origins_per_entry
                )));
            }
            binding_span_count = binding_span_count
                .checked_add(origin.binding_spans.len())
                .ok_or_else(|| provenance_error("provenance binding-span count overflows usize"))?;
            if binding_span_count > self.limits.max_binding_spans_per_entry {
                return Err(provenance_error(format!(
                    "elaboration provenance exceeds the {} binding-span-per-entry limit",
                    self.limits.max_binding_spans_per_entry
                )));
            }
            collected_origins
                .try_reserve(1)
                .map_err(|_| provenance_error("cannot reserve provenance source origins"))?;
            collected_origins.insert(insertion_index, origin);
        }
        if collected_origins.is_empty() {
            return Err(provenance_error(
                "elaboration provenance requires at least one source origin",
            ));
        }

        let entry_path_bytes = origins_path_bytes(&collected_origins)?;
        let next_total = self
            .total_source_path_bytes
            .checked_add(entry_path_bytes)
            .ok_or_else(|| provenance_error("provenance source-path bytes overflow usize"))?;
        if next_total > self.limits.max_total_source_path_bytes {
            return Err(provenance_error(format!(
                "elaboration provenance exceeds the {} total source-path byte limit",
                self.limits.max_total_source_path_bytes
            )));
        }

        self.entries
            .try_reserve(1)
            .map_err(|_| provenance_error("cannot reserve elaboration provenance entry"))?;
        self.entries.insert(
            insertion_index,
            ProvenanceEntry {
                identity,
                graph_id,
                provenance: ElaborationProvenance {
                    origins: collected_origins.into_boxed_slice(),
                },
            },
        );
        self.total_source_path_bytes = next_total;
        Ok(())
    }

    fn entry_insertion_index(
        &self,
        identity: FullElaborationIdentity,
    ) -> Result<usize, Diagnostic> {
        let insertion_index = match self
            .entries
            .binary_search_by_key(&identity, |entry| entry.identity)
        {
            Ok(_) => {
                return Err(provenance_error(format!(
                    "provenance for elaboration identity {identity} already exists"
                )));
            }
            Err(index) => index,
        };
        if self.entries.len() >= self.limits.max_entries {
            return Err(provenance_error(format!(
                "elaboration provenance exceeds the {} entry limit",
                self.limits.max_entries
            )));
        }
        Ok(insertion_index)
    }

    /// Seal the sidecar into immutable identity order.
    #[must_use]
    pub fn finish(self) -> ProvenanceMap {
        let mut graph_index = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.graph_id.map(|id| (id, index)))
            .collect::<Vec<_>>();
        graph_index.sort_unstable_by_key(|(id, _)| *id);
        ProvenanceMap {
            entries: self.entries.into_boxed_slice(),
            graph_index: graph_index.into_boxed_slice(),
        }
    }
}

impl Default for ProvenanceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable, deterministically ordered elaboration provenance sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceMap {
    entries: Box<[ProvenanceEntry]>,
    graph_index: Box<[(RawId, usize)]>,
}

impl ProvenanceMap {
    pub(crate) fn retain_source_files(self, keep: impl Fn(&str) -> bool) -> Self {
        let entries = self
            .entries
            .into_vec()
            .into_iter()
            .filter_map(|mut entry| {
                entry.provenance.origins = entry
                    .provenance
                    .origins
                    .into_vec()
                    .into_iter()
                    .filter(|origin| {
                        keep(&origin.definition_span.file)
                            && keep(&origin.instance_span.file)
                            && origin.binding_spans.iter().all(|span| keep(&span.file))
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                (!entry.provenance.origins.is_empty()).then_some(entry)
            })
            .collect::<Vec<_>>();
        let mut graph_index = entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.graph_id.map(|id| (id, index)))
            .collect::<Vec<_>>();
        graph_index.sort_unstable_by_key(|(id, _)| *id);
        Self {
            entries: entries.into_boxed_slice(),
            graph_index: graph_index.into_boxed_slice(),
        }
    }
    /// Number of identities with source provenance.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map contains no provenance entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up source provenance by exact full identity.
    #[must_use]
    pub fn get(&self, identity: FullElaborationIdentity) -> Option<&ElaborationProvenance> {
        self.entries
            .binary_search_by_key(&identity, |entry| entry.identity)
            .ok()
            .map(|index| &self.entries[index].provenance)
    }

    /// Look up source provenance from the projected Semantic Kernel node ID
    /// used by graph snapshots, diagnostics, and Studio selections.
    #[must_use]
    pub fn get_by_graph_id(&self, graph_id: RawId) -> Option<&ElaborationProvenance> {
        self.graph_index
            .binary_search_by_key(&graph_id, |(id, _)| *id)
            .ok()
            .map(|index| &self.entries[self.graph_index[index].1].provenance)
    }

    /// Recover the full collision-resistant identity for a projected graph
    /// node ID.
    #[must_use]
    pub fn identity_for_graph_id(&self, graph_id: RawId) -> Option<FullElaborationIdentity> {
        self.graph_index
            .binary_search_by_key(&graph_id, |(id, _)| *id)
            .ok()
            .map(|index| self.entries[self.graph_index[index].1].identity)
    }

    /// Iterate in ascending full-identity byte order.
    pub fn iter(
        &self,
    ) -> impl ExactSizeIterator<Item = (FullElaborationIdentity, &ElaborationProvenance)> {
        self.entries
            .iter()
            .map(|entry| (entry.identity, &entry.provenance))
    }
}

fn collect_origin(
    definition_span: Span,
    instance_span: Span,
    binding_spans: impl IntoIterator<Item = Span>,
    limits: ProvenanceLimits,
) -> Result<ElaborationSourceOrigin, Diagnostic> {
    validate_span(&definition_span, limits)?;
    validate_span(&instance_span, limits)?;
    let mut collected_binding_spans = Vec::new();
    for span in binding_spans {
        if collected_binding_spans.len() >= limits.max_binding_spans_per_entry {
            return Err(provenance_error(format!(
                "elaboration provenance exceeds the {} binding-span-per-entry limit",
                limits.max_binding_spans_per_entry
            )));
        }
        validate_span(&span, limits)?;
        collected_binding_spans
            .try_reserve(1)
            .map_err(|_| provenance_error("cannot reserve binding source spans"))?;
        collected_binding_spans.push(span);
    }
    Ok(ElaborationSourceOrigin::new(
        definition_span,
        instance_span,
        collected_binding_spans,
    ))
}

fn validate_origin(
    origin: &ElaborationSourceOrigin,
    limits: ProvenanceLimits,
) -> Result<(), Diagnostic> {
    validate_span(origin.definition_span(), limits)?;
    validate_span(origin.instance_span(), limits)?;
    for span in origin.binding_spans() {
        validate_span(span, limits)?;
    }
    Ok(())
}

fn validate_span(span: &Span, limits: ProvenanceLimits) -> Result<(), Diagnostic> {
    if span.file.is_empty() {
        return Err(provenance_error("provenance source path must not be empty"));
    }
    if span.file.len() > limits.max_source_path_bytes {
        return Err(provenance_error(format!(
            "provenance source path requires {} bytes, exceeding the {} byte limit",
            span.file.len(),
            limits.max_source_path_bytes
        )));
    }
    if span.start > span.end {
        return Err(provenance_error(format!(
            "provenance span {}:{}..{} is reversed",
            span.file, span.start, span.end
        )));
    }
    Ok(())
}

fn origins_path_bytes(origins: &[ElaborationSourceOrigin]) -> Result<usize, Diagnostic> {
    let mut total = 0usize;
    for origin in origins {
        total = total
            .checked_add(origin.definition_span.file.len())
            .and_then(|value| value.checked_add(origin.instance_span.file.len()))
            .ok_or_else(|| provenance_error("provenance source-path bytes overflow usize"))?;
        for span in &origin.binding_spans {
            total = total
                .checked_add(span.file.len())
                .ok_or_else(|| provenance_error("provenance source-path bytes overflow usize"))?;
        }
    }
    Ok(total)
}

fn compare_span(left: &Span, right: &Span) -> Ordering {
    (&left.file, left.start, left.end).cmp(&(&right.file, right.start, right.end))
}

fn compare_origin(left: &ElaborationSourceOrigin, right: &ElaborationSourceOrigin) -> Ordering {
    compare_span(left.definition_span(), right.definition_span())
        .then_with(|| compare_span(left.instance_span(), right.instance_span()))
        .then_with(|| compare_span_slices(left.binding_spans(), right.binding_spans()))
}

fn compare_span_slices(left: &[Span], right: &[Span]) -> Ordering {
    for (left, right) in left.iter().zip(right) {
        let ordering = compare_span(left, right);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

fn provenance_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::LANGUAGE_LOWERING_ERROR, message)
}

#[cfg(test)]
mod tests {
    use eqiora_core::entity::kinds;
    use eqiora_core::{EntityKind, Id};

    use super::*;
    use crate::identity::{DeclarationPath, ElaborationKey, IdentityNamespace, InstancePath};

    fn span(file: &str, start: u32, end: u32) -> Span {
        Span {
            file: file.to_owned(),
            start,
            end,
        }
    }

    fn identity(name: &str) -> FullElaborationIdentity {
        ElaborationKey::entity(
            IdentityNamespace::new(["org", "components", "Resistor"]).unwrap(),
            InstancePath::new(["plant", "r1"]).unwrap(),
            DeclarationPath::new([name]).unwrap(),
            EntityKind::Field,
        )
        .unwrap()
        .full_identity()
        .unwrap()
    }

    #[test]
    fn map_and_binding_order_are_deterministic() {
        let a = identity("a");
        let b = identity("b");
        let mut first = ProvenanceBuilder::new();
        first
            .insert(
                b,
                span("definition.eqi", 20, 30),
                span("model.eqi", 50, 60),
                [span("model.eqi", 80, 90), span("model.eqi", 70, 75)],
            )
            .unwrap();
        first
            .insert(
                a,
                span("definition.eqi", 0, 10),
                span("model.eqi", 40, 50),
                [],
            )
            .unwrap();
        let first = first.finish();

        let mut second = ProvenanceBuilder::new();
        second
            .insert(
                a,
                span("definition.eqi", 0, 10),
                span("model.eqi", 40, 50),
                [],
            )
            .unwrap();
        second
            .insert(
                b,
                span("definition.eqi", 20, 30),
                span("model.eqi", 50, 60),
                [span("model.eqi", 70, 75), span("model.eqi", 80, 90)],
            )
            .unwrap();
        let second = second.finish();

        assert_eq!(first, second);
        let identities = first
            .iter()
            .map(|(identity, _)| identity)
            .collect::<Vec<_>>();
        let mut sorted = vec![a, b];
        sorted.sort_unstable();
        assert_eq!(identities, sorted);
    }

    #[test]
    fn graph_nodes_resolve_to_full_identity_and_source_provenance() {
        let semantic_identity = identity("temperature");
        let graph_id = Id::<kinds::Field>::new().erase();
        let mut builder = ProvenanceBuilder::new();
        builder
            .insert_graph(
                graph_id,
                semantic_identity,
                span("definition.eqi", 10, 20),
                span("model.eqi", 30, 40),
                [span("model.eqi", 35, 38)],
            )
            .unwrap();
        let map = builder.finish();

        assert_eq!(map.identity_for_graph_id(graph_id), Some(semantic_identity));
        let provenance = map.get_by_graph_id(graph_id).unwrap();
        assert_eq!(
            provenance.definition_span(),
            &span("definition.eqi", 10, 20)
        );
        assert_eq!(provenance.instance_span(), &span("model.eqi", 30, 40));
        assert_eq!(provenance.binding_spans(), &[span("model.eqi", 35, 38)]);
        assert_eq!(provenance.origins().len(), 1);
        assert_eq!(
            provenance.origins()[0].definition_span(),
            provenance.definition_span()
        );
        assert_eq!(
            provenance.origins()[0].instance_span(),
            provenance.instance_span()
        );
        assert_eq!(
            provenance.origins()[0].binding_spans(),
            provenance.binding_spans()
        );
    }

    #[test]
    fn multiple_origins_are_sorted_and_deduplicated_as_complete_tuples() {
        let semantic_identity = identity("interface");
        let first = ElaborationSourceOrigin::new(
            span("a-definition.eqi", 10, 20),
            span("z-instance.eqi", 30, 40),
            vec![span("z-binding.eqi", 50, 60)],
        );
        let second = ElaborationSourceOrigin::new(
            span("b-definition.eqi", 10, 20),
            span("a-instance.eqi", 30, 40),
            vec![span("a-binding.eqi", 50, 60)],
        );
        let mut builder = ProvenanceBuilder::new();
        builder
            .insert_origins(
                semantic_identity,
                [second.clone(), first.clone(), second.clone()],
            )
            .unwrap();
        let map = builder.finish();
        let provenance = map.get(semantic_identity).unwrap();

        assert_eq!(provenance.origins(), &[first.clone(), second]);
        assert_eq!(provenance.definition_span(), first.definition_span());
        assert_eq!(provenance.instance_span(), first.instance_span());
        assert_eq!(provenance.binding_spans(), first.binding_spans());
        assert_eq!(
            provenance.instance_span(),
            &span("z-instance.eqi", 30, 40),
            "the first definition must not be combined with the other origin's earlier instance"
        );
    }

    #[test]
    fn graph_origin_insertion_rejects_duplicate_identity_and_graph_id() {
        let first_identity = identity("first");
        let second_identity = identity("second");
        let first_graph_id = Id::<kinds::Connection>::new().erase();
        let second_graph_id = Id::<kinds::Connection>::new().erase();
        let origin = || {
            ElaborationSourceOrigin::new(
                span("definition.eqi", 10, 20),
                span("model.eqi", 30, 40),
                Vec::new(),
            )
        };
        let mut builder = ProvenanceBuilder::new();
        builder
            .insert_graph_origins(first_graph_id, first_identity, [origin()])
            .unwrap();

        assert!(
            builder
                .insert_graph_origins(first_graph_id, second_identity, [origin()])
                .is_err()
        );
        assert!(
            builder
                .insert_graph_origins(second_graph_id, first_identity, [origin()])
                .is_err()
        );

        builder
            .insert_graph_origins(second_graph_id, second_identity, [origin()])
            .unwrap();
    }

    #[test]
    fn source_span_changes_do_not_change_semantic_identity() {
        let semantic_identity = identity("voltage");
        let mut before = ProvenanceBuilder::new();
        before
            .insert(
                semantic_identity,
                span("old/location.eqi", 10, 20),
                span("old/model.eqi", 30, 40),
                [span("old/model.eqi", 35, 38)],
            )
            .unwrap();
        let mut after = ProvenanceBuilder::new();
        after
            .insert(
                semantic_identity,
                span("new/location.eqi", 100, 120),
                span("new/model.eqi", 300, 340),
                [span("new/model.eqi", 320, 330)],
            )
            .unwrap();

        assert_eq!(
            before.finish().iter().next().unwrap().0,
            after.finish().iter().next().unwrap().0
        );
    }

    #[test]
    fn duplicate_identity_and_invalid_spans_fail_closed() {
        let semantic_identity = identity("voltage");
        let mut builder = ProvenanceBuilder::new();
        builder
            .insert(
                semantic_identity,
                span("definition.eqi", 0, 10),
                span("model.eqi", 20, 30),
                [],
            )
            .unwrap();
        assert!(
            builder
                .insert(
                    semantic_identity,
                    span("definition.eqi", 0, 10),
                    span("model.eqi", 20, 30),
                    [],
                )
                .is_err()
        );

        let mut reversed = ProvenanceBuilder::new();
        assert!(
            reversed
                .insert(
                    identity("other"),
                    span("definition.eqi", 10, 0),
                    span("model.eqi", 20, 30),
                    [],
                )
                .is_err()
        );
    }

    #[test]
    fn construction_limits_fail_closed() {
        let limits = ProvenanceLimits {
            max_entries: 1,
            max_origins_per_entry: 1,
            max_binding_spans_per_entry: 1,
            max_source_path_bytes: 8,
            max_total_source_path_bytes: 24,
        };
        let mut builder = ProvenanceBuilder::with_limits(limits);
        assert!(
            builder
                .insert(
                    identity("a"),
                    span("def.eqi", 0, 1),
                    span("use.eqi", 0, 1),
                    [span("bind.eqi", 0, 1), span("more.eqi", 0, 1)],
                )
                .is_err()
        );

        let mut too_many_origins = ProvenanceBuilder::with_limits(limits);
        assert!(
            too_many_origins
                .insert_origins(
                    identity("origins"),
                    [
                        ElaborationSourceOrigin::new(
                            span("def.eqi", 0, 1),
                            span("use.eqi", 0, 1),
                            Vec::new(),
                        ),
                        ElaborationSourceOrigin::new(
                            span("def.eqi", 1, 2),
                            span("use.eqi", 0, 1),
                            Vec::new(),
                        ),
                    ],
                )
                .is_err()
        );

        let origins_with_two_bindings = ProvenanceLimits {
            max_entries: 1,
            max_origins_per_entry: 2,
            max_binding_spans_per_entry: 1,
            max_source_path_bytes: 8,
            max_total_source_path_bytes: 64,
        };
        let mut too_many_bindings = ProvenanceBuilder::with_limits(origins_with_two_bindings);
        assert!(
            too_many_bindings
                .insert_origins(
                    identity("bindings"),
                    [
                        ElaborationSourceOrigin::new(
                            span("def.eqi", 0, 1),
                            span("use.eqi", 0, 1),
                            vec![span("a.eqi", 0, 1)],
                        ),
                        ElaborationSourceOrigin::new(
                            span("def.eqi", 1, 2),
                            span("use.eqi", 0, 1),
                            vec![span("b.eqi", 0, 1)],
                        ),
                    ],
                )
                .is_err()
        );
        assert!(
            builder
                .insert(
                    identity("a"),
                    span("too-long-path.eqi", 0, 1),
                    span("use.eqi", 0, 1),
                    [],
                )
                .is_err()
        );

        builder
            .insert(
                identity("a"),
                span("def.eqi", 0, 1),
                span("use.eqi", 0, 1),
                [],
            )
            .unwrap();
        assert!(
            builder
                .insert(
                    identity("b"),
                    span("def.eqi", 0, 1),
                    span("use.eqi", 0, 1),
                    [],
                )
                .is_err()
        );
    }
}
