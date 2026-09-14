//! The single current Model wire owner.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, EntityKind, OntologyId, RawId};
use eqiora_graph::{GraphStore, InMemoryGraphStore, Op, Revision, Transaction};
use eqiora_schema::{Model, ModelView};
use eqiora_sem::KernelProgram;
use serde::{Deserialize, Serialize};

use crate::model::{
    WireEdge, WireId, WireNode, WireValue, WireValueLiteral, checked_count_sum, parse_ulid,
    require_decoder_count,
};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, ModelDecoderLimits, check_json_limits, invalid_artifact,
};

const MODEL_SCHEMA: &str = "eqiora.model-envelope/v26";
const MODEL_LABEL: &str = "current Model";
const ENVELOPE_LABEL: &str = "current Model envelope";
const DECODE_LABEL: &str = "decode eqiora.model-envelope/v26";

/// Canonical serialization of the single current Semantic Model contract.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelEnvelope {
    wire: WireModelEnvelope,
}

impl ModelEnvelope {
    /// Encode one immutable validated Semantic Kernel program.
    ///
    /// # Errors
    /// Returns `EQ0901` for an unsupported kernel value or resource-limit
    /// violation.
    pub fn from_program(program: &KernelProgram) -> Result<Self, Diagnostic> {
        let nodes = program
            .nodes()
            .map(WireNode::encode)
            .collect::<Result<Vec<_>, _>>()?;
        let values = program
            .nodes()
            .filter_map(|node| {
                program.typed_value(node.id()).map(|value| {
                    Ok(WireValue {
                        target: WireId::from_raw(node.id()),
                        value: WireValueLiteral::encode(value)?,
                    })
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let edges = program
            .edges()
            .iter()
            .map(|edge| {
                Ok(WireEdge {
                    from: WireId::from_raw(edge.from()),
                    to: WireId::from_raw(edge.to()),
                    kind: crate::model::WireEdgeKind::encode(edge.kind())?,
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let boundary = program
            .boundary()
            .iter()
            .copied()
            .map(WireId::from_raw)
            .collect();
        let mut envelope = Self {
            wire: WireModelEnvelope {
                schema: MODEL_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                source_revision: program.revision().0,
                model_ulid: program.model().ulid().to_string(),
                nodes,
                values,
                edges,
                boundary,
            },
        };
        envelope.canonicalize_and_validate(ModelDecoderLimits::default())?;
        Ok(envelope)
    }

    /// Decode and validate current bytes without mutating a graph store.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, dangling, duplicated, or
    /// wrong-version data.
    pub fn from_json(bytes: &[u8], limits: ModelDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid {MODEL_SCHEMA} JSON: {error}")))?;
        let mut envelope = Self { wire };
        envelope.canonicalize_and_validate(limits)?;
        Ok(envelope)
    }

    /// Deterministic compact canonical JSON.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!("cannot serialize {ENVELOPE_LABEL}: {error}"))
        })
    }

    /// Domain-separated SHA-256 identity of current semantic content.
    ///
    /// Source revision is provenance and is excluded from content identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        let content = WireModelContent {
            schema: &self.wire.schema,
            encoding: &self.wire.encoding,
            model_ulid: &self.wire.model_ulid,
            nodes: &self.wire.nodes,
            values: &self.wire.values,
            edges: &self.wire.edges,
            boundary: &self.wire.boundary,
        };
        let bytes = serde_json::to_vec(&content).map_err(|error| {
            invalid_artifact(format!("cannot serialize {MODEL_LABEL} content: {error}"))
        })?;
        Ok(ArtifactDigest::compute(MODEL_SCHEMA.as_bytes(), &bytes))
    }

    /// Reconstruct one typed transaction without committing it.
    ///
    /// # Errors
    /// Returns structured diagnostics if validated wire data cannot be
    /// represented by the closed Semantic Model transaction vocabulary.
    pub fn to_transaction(&self) -> Result<(Transaction, OntologyId<Model>), Vec<Diagnostic>> {
        let model = OntologyId::<Model>::from_ulid(
            parse_ulid(&self.wire.model_ulid).map_err(|error| vec![error])?,
        );
        let mut ids = BTreeMap::new();
        let mut definitions = Vec::with_capacity(self.wire.nodes.len());
        for wire_node in &self.wire.nodes {
            let definition = wire_node.decode().map_err(|error| vec![error])?;
            if ids.insert(wire_node.id.clone(), definition.id()).is_some() {
                return Err(vec![invalid_artifact(format!(
                    "{ENVELOPE_LABEL} contains a duplicate kernel node ID"
                ))]);
            }
            definitions.push(definition);
        }

        let mut transaction = Transaction::new(DECODE_LABEL);
        for node in definitions {
            transaction.push(Op::DefineKernelNode { node });
        }
        for value in &self.wire.values {
            transaction.push(Op::SetValue {
                target: lookup_id(&ids, &value.target).map_err(|error| vec![error])?,
                value: value.value.decode().map_err(|error| vec![error])?,
            });
        }
        for edge in &self.wire.edges {
            transaction.push(Op::Connect {
                from: lookup_id(&ids, &edge.from).map_err(|error| vec![error])?,
                to: lookup_id(&ids, &edge.to).map_err(|error| vec![error])?,
                edge: edge.kind.decode(),
            });
        }
        let boundary = self
            .wire
            .boundary
            .iter()
            .map(|id| lookup_id(&ids, id))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| vec![error])?;
        let view =
            ModelView::new(model, ids.values().copied(), boundary).map_err(|error| vec![error])?;
        transaction.push(Op::DefineOntologyView { view: view.into() });
        Ok((transaction, model))
    }

    /// Reconstruct through typed definitions, an atomic commit, and complete
    /// [`KernelProgram`] validation.
    ///
    /// # Errors
    /// Returns diagnostics from reconstruction, commit, or whole-model
    /// validation; no partial state is exposed.
    pub fn to_program(&self) -> Result<KernelProgram, Vec<Diagnostic>> {
        let (transaction, model) = self.to_transaction()?;
        let store =
            InMemoryGraphStore::restore_snapshot(transaction, Revision(self.source_revision()))?;
        KernelProgram::from_snapshot(&store.snapshot(), model)
    }

    /// Source graph revision retained as provenance.
    #[must_use]
    pub const fn source_revision(&self) -> u64 {
        self.wire.source_revision
    }

    /// Typed Semantic Model identity.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state were corrupted.
    pub fn model(&self) -> Result<OntologyId<Model>, Diagnostic> {
        parse_ulid(&self.wire.model_ulid).map(OntologyId::from_ulid)
    }

    /// Whether whole-program admission requires an external geometry bundle.
    ///
    /// This is derived from the typed current Model definitions, not from a
    /// failed semantic replay or its human-readable diagnostics. Callers may
    /// retain such an artifact before the exact geometry closure is available,
    /// but must not treat this predicate as semantic admission.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal wire state cannot be decoded.
    pub fn requires_geometry_admission(&self) -> Result<bool, Diagnostic> {
        for node in &self.wire.nodes {
            let node = node.decode()?;
            if matches!(
                node,
                eqiora_schema::kernel::KernelNode::Domain(domain)
                    if matches!(
                        domain.kind(),
                        eqiora_schema::kernel::DomainKind::GeometryRegion { .. }
                            | eqiora_schema::kernel::DomainKind::GeometryBoundary { .. }
                    )
            ) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn canonicalize_and_validate(&mut self, limits: ModelDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != MODEL_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(format!(
                "unsupported {MODEL_SCHEMA} schema or canonical encoding"
            )));
        }
        if self.wire.source_revision == 0 {
            return Err(invalid_artifact(format!(
                "{ENVELOPE_LABEL} source revision must be nonzero"
            )));
        }
        parse_ulid(&self.wire.model_ulid)?;
        if self.wire.nodes.is_empty() || self.wire.nodes.len() > limits.max_nodes {
            return Err(invalid_artifact(format!(
                "{ENVELOPE_LABEL} requires 1..={} nodes, found {}",
                limits.max_nodes,
                self.wire.nodes.len()
            )));
        }
        require_decoder_count(
            &format!("{MODEL_LABEL} view members"),
            self.wire.nodes.len(),
            limits.max_model_view_members,
        )?;
        require_decoder_count(
            &format!("{MODEL_LABEL} boundary Ports"),
            self.wire.boundary.len(),
            limits.max_model_boundary,
        )?;
        if self.wire.edges.len() > limits.max_edges {
            return Err(invalid_artifact(format!(
                "{ENVELOPE_LABEL} has {} edges, exceeding the {} edge limit",
                self.wire.edges.len(),
                limits.max_edges
            )));
        }
        let expression_nodes = self.wire.nodes.iter().try_fold(0_usize, |count, node| {
            count
                .checked_add(node.expression_node_count())
                .ok_or_else(|| invalid_artifact("expression-node count overflows usize"))
        })?;
        if expression_nodes > limits.max_expression_nodes {
            return Err(invalid_artifact(format!(
                "{ENVELOPE_LABEL} has {expression_nodes} expression nodes, exceeding the {} node limit",
                limits.max_expression_nodes
            )));
        }
        let expression_roots = checked_count_sum(
            self.wire.nodes.iter().map(WireNode::expression_root_count),
            &format!("{MODEL_LABEL} expression-root count"),
        )?;
        require_decoder_count(
            &format!("{MODEL_LABEL} expression roots"),
            expression_roots,
            limits.max_expression_roots,
        )?;
        let pure_operator_counts = self.wire.nodes.iter().try_fold(
            crate::model::PureOperatorWireCounts::default(),
            |counts, node| counts.checked_add(node.pure_operator_counts()?),
        )?;
        pure_operator_counts.ensure_limits(limits, MODEL_LABEL)?;

        self.wire
            .nodes
            .sort_by(|left, right| left.id.cmp(&right.id));
        reject_duplicates(
            self.wire
                .nodes
                .windows(2)
                .any(|pair| pair[0].id == pair[1].id),
            "kernel node ID",
        )?;
        self.wire
            .values
            .sort_by(|left, right| left.target.cmp(&right.target));
        reject_duplicates(
            self.wire
                .values
                .windows(2)
                .any(|pair| pair[0].target == pair[1].target),
            "current value target",
        )?;
        self.wire.edges.sort();
        reject_duplicates(
            self.wire.edges.windows(2).any(|pair| pair[0] == pair[1]),
            "graph edge",
        )?;
        self.wire.boundary.sort();
        reject_duplicates(
            self.wire.boundary.windows(2).any(|pair| pair[0] == pair[1]),
            "boundary ID",
        )?;

        for node in &self.wire.nodes {
            node.validate_pure_operator_features()?;
        }
        for node in &mut self.wire.nodes {
            node.canonicalize_pure_operator_definitions()?;
        }

        let literal_components = checked_count_sum(
            self.wire
                .nodes
                .iter()
                .map(WireNode::literal_component_count)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .chain(
                    self.wire
                        .values
                        .iter()
                        .map(|value| value.value.component_payload_count()),
                ),
            "literal component payload",
        )?;
        require_decoder_count(
            "literal component payload",
            literal_components,
            limits.max_value_literal_components,
        )?;
        for node in &self.wire.nodes {
            node.ensure_value_shape_limits(limits)?;
            node.decode()?;
        }
        for value in &self.wire.values {
            value.value.ensure_limits(limits)?;
            value.value.decode()?;
        }
        let ids = self
            .wire
            .nodes
            .iter()
            .map(|node| node.id.clone())
            .collect::<BTreeSet<_>>();
        for node in &self.wire.nodes {
            for reference in node.semantic_references() {
                require_reference(&ids, reference, "definition reference")?;
            }
        }
        for value in &self.wire.values {
            require_reference(&ids, &value.target, "current value")?;
            if let Some(reference) = value.value.nominal_reference() {
                require_reference(&ids, reference, "current value nominal basis")?;
            }
            if value.target.decode_raw()?.kind() != EntityKind::Parameter {
                return Err(invalid_artifact("only Parameter nodes own current values"));
            }
        }
        for edge in &self.wire.edges {
            require_reference(&ids, &edge.from, "edge source")?;
            require_reference(&ids, &edge.to, "edge target")?;
            let from = edge.from.decode_raw()?;
            let to = edge.to.decode_raw()?;
            if !edge_permitted(edge.kind.decode(), from.kind(), to.kind()) {
                return Err(invalid_artifact(format!(
                    "{MODEL_LABEL} edge endpoints violate the closed graph edge schema"
                )));
            }
        }
        for boundary in &self.wire.boundary {
            require_reference(&ids, boundary, "boundary")?;
            if boundary.decode_raw()?.kind() != EntityKind::Port {
                return Err(invalid_artifact("model boundary may contain only Port IDs"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireModelEnvelope {
    schema: String,
    encoding: String,
    source_revision: u64,
    model_ulid: String,
    nodes: Vec<WireNode>,
    values: Vec<WireValue>,
    edges: Vec<WireEdge>,
    boundary: Vec<WireId>,
}

#[derive(Serialize)]
struct WireModelContent<'a> {
    schema: &'a str,
    encoding: &'a str,
    model_ulid: &'a str,
    nodes: &'a [WireNode],
    values: &'a [WireValue],
    edges: &'a [WireEdge],
    boundary: &'a [WireId],
}

fn lookup_id(ids: &BTreeMap<WireId, RawId>, id: &WireId) -> Result<RawId, Diagnostic> {
    ids.get(id)
        .copied()
        .ok_or_else(|| invalid_artifact("current Model reference is not a Model node"))
}

fn require_reference(ids: &BTreeSet<WireId>, id: &WireId, label: &str) -> Result<(), Diagnostic> {
    if ids.contains(id) {
        Ok(())
    } else {
        Err(invalid_artifact(format!(
            "{MODEL_LABEL} {label} is not a Model node"
        )))
    }
}

fn reject_duplicates(duplicate: bool, label: &str) -> Result<(), Diagnostic> {
    if duplicate {
        Err(invalid_artifact(format!(
            "{ENVELOPE_LABEL} contains a duplicate {label}"
        )))
    } else {
        Ok(())
    }
}

fn edge_permitted(edge: eqiora_graph::EdgeKind, from: EntityKind, to: EntityKind) -> bool {
    edge.permits(from, to)
}
