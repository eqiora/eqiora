//! The single current ordered Semantic Model transaction wire owner.

use eqiora_core::Diagnostic;
use eqiora_graph::Transaction;
use serde::{Deserialize, Serialize};

use crate::model::{checked_count_sum, require_decoder_count};
use crate::model_transaction::{WireModelOp, WireModelPrecondition};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, ModelDecoderLimits, check_json_limits, invalid_artifact,
    validate_text,
};

const TRANSACTION_SCHEMA: &str = "eqiora.model-transaction-envelope/v26";
const TRANSACTION_LABEL: &str = "current Model transaction";
const ENVELOPE_LABEL: &str = "current Model transaction envelope";

/// Canonical serialization of one ordered current Semantic Model transaction.
///
/// This is the identity of a locally typed edit, not a proof that the edit
/// alone forms a complete Semantic Model. References may resolve against the
/// selected store revision or a later operation. After atomic commit, callers
/// must construct a `KernelProgram` before exposing the candidate as valid.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelTransactionEnvelope {
    wire: WireModelTransactionEnvelope,
}

impl ModelTransactionEnvelope {
    /// Encode an ordered transaction without changing operation order.
    ///
    /// # Errors
    /// Returns `EQ0901` for operations outside the Semantic Model vocabulary,
    /// unsupported kernel values, or resource-limit violations.
    pub fn from_transaction(transaction: &Transaction) -> Result<Self, Diagnostic> {
        let ops = transaction
            .ops()
            .iter()
            .map(WireModelOp::encode)
            .collect::<Result<Vec<_>, _>>()?;
        let preconditions = transaction
            .preconditions()
            .iter()
            .map(WireModelPrecondition::encode)
            .collect::<Result<Vec<_>, _>>()?;
        let mut envelope = Self {
            wire: WireModelTransactionEnvelope {
                schema: TRANSACTION_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                label: transaction.label().to_owned(),
                ops,
                preconditions,
            },
        };
        envelope.canonicalize_and_validate(ModelDecoderLimits::default())?;
        Ok(envelope)
    }

    /// Decode and validate the local edit grammar without mutating a graph
    /// store.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, or wrong-version data.
    pub fn from_json(bytes: &[u8], limits: ModelDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid {TRANSACTION_SCHEMA} JSON: {error}"))
        })?;
        let mut envelope = Self { wire };
        envelope.canonicalize_and_validate(limits)?;
        Ok(envelope)
    }

    /// Deterministic compact JSON preserving semantic operation order.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!("cannot serialize {ENVELOPE_LABEL}: {error}"))
        })
    }

    /// Domain-separated SHA-256 identity of the exact ordered transaction.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            TRANSACTION_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Reconstruct the locally typed in-memory transaction without committing
    /// it.
    ///
    /// # Errors
    /// Returns `EQ0901` if locally valid wire data cannot be reconstructed.
    /// Store-dependent references are admitted only when the committed
    /// snapshot is reconstructed as a `KernelProgram`.
    pub fn to_transaction(&self) -> Result<Transaction, Diagnostic> {
        let mut transaction = Transaction::new(&self.wire.label);
        for precondition in &self.wire.preconditions {
            transaction.require(precondition.decode()?);
        }
        for op in &self.wire.ops {
            transaction.push(op.decode()?);
        }
        Ok(transaction)
    }

    fn canonicalize_and_validate(&mut self, limits: ModelDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != TRANSACTION_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(format!(
                "unsupported {TRANSACTION_SCHEMA} schema or canonical encoding"
            )));
        }
        validate_text("transaction label", &self.wire.label)?;
        if self.wire.ops.is_empty() || self.wire.ops.len() > limits.max_transaction_ops {
            return Err(invalid_artifact(format!(
                "{TRANSACTION_LABEL} requires 1..={} operations, found {}",
                limits.max_transaction_ops,
                self.wire.ops.len()
            )));
        }
        if self.wire.preconditions.len() > limits.max_transaction_preconditions {
            return Err(invalid_artifact(format!(
                "{TRANSACTION_LABEL} has {} preconditions, exceeding the {} precondition limit",
                self.wire.preconditions.len(),
                limits.max_transaction_preconditions
            )));
        }
        for op in &mut self.wire.ops {
            op.canonicalize_sets()?;
        }
        let expression_nodes = self.wire.ops.iter().try_fold(0_usize, |count, op| {
            count
                .checked_add(op.expression_node_count())
                .ok_or_else(|| invalid_artifact("expression-node count overflows usize"))
        })?;
        if expression_nodes > limits.max_expression_nodes {
            return Err(invalid_artifact(format!(
                "{TRANSACTION_LABEL} has {expression_nodes} expression nodes, exceeding the {} node limit",
                limits.max_expression_nodes
            )));
        }
        let expression_roots = checked_count_sum(
            self.wire.ops.iter().map(WireModelOp::expression_root_count),
            &format!("{TRANSACTION_LABEL} expression-root count"),
        )?;
        require_decoder_count(
            &format!("{TRANSACTION_LABEL} expression roots"),
            expression_roots,
            limits.max_expression_roots,
        )?;
        let pure_operator_counts = self.wire.ops.iter().try_fold(
            crate::model::PureOperatorWireCounts::default(),
            |counts, op| counts.checked_add(op.pure_operator_counts()?),
        )?;
        pure_operator_counts.ensure_limits(limits, TRANSACTION_LABEL)?;
        let view_members = checked_count_sum(
            self.wire.ops.iter().map(|op| op.model_view_counts().0),
            &format!("{TRANSACTION_LABEL} view-member count"),
        )?;
        require_decoder_count(
            &format!("{TRANSACTION_LABEL} view members"),
            view_members,
            limits.max_model_view_members,
        )?;
        let boundaries = checked_count_sum(
            self.wire.ops.iter().map(|op| op.model_view_counts().1),
            &format!("{TRANSACTION_LABEL} boundary count"),
        )?;
        require_decoder_count(
            &format!("{TRANSACTION_LABEL} boundary Ports"),
            boundaries,
            limits.max_model_boundary,
        )?;
        for op in &self.wire.ops {
            op.validate_pure_operator_features()?;
        }
        for op in &mut self.wire.ops {
            op.canonicalize_pure_operator_definitions()?;
        }
        let literal_components = checked_count_sum(
            self.wire
                .ops
                .iter()
                .map(WireModelOp::literal_component_count)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .chain(
                    self.wire
                        .preconditions
                        .iter()
                        .map(WireModelPrecondition::literal_component_count),
                ),
            "literal component payload",
        )?;
        require_decoder_count(
            "literal component payload",
            literal_components,
            limits.max_value_literal_components,
        )?;
        for precondition in &self.wire.preconditions {
            precondition.ensure_value_shape_limits(limits)?;
            precondition.decode()?;
        }
        for op in &self.wire.ops {
            op.ensure_value_shape_limits(limits)?;
            op.decode()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireModelTransactionEnvelope {
    schema: String,
    encoding: String,
    label: String,
    ops: Vec<WireModelOp>,
    preconditions: Vec<WireModelPrecondition>,
}
