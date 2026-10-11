use eqiora_core::Diagnostic;
use serde::{Deserialize, Serialize};

use crate::{JsonDecoderLimits, invalid_artifact};

/// Semantic work budgets shared by the current Model and Model transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDecoderLimits {
    /// Common JSON syntax admission.
    pub json: JsonDecoderLimits,
    /// Maximum Semantic Kernel nodes in one Model envelope.
    pub max_nodes: usize,
    /// Maximum graph edges in one Model envelope.
    pub max_edges: usize,
    /// Maximum expression nodes summed across one Model or transaction.
    pub max_expression_nodes: usize,
    /// Maximum expression roots summed across one Model or transaction.
    pub max_expression_roots: usize,
    /// Maximum pure-operator definitions summed across expressions.
    pub max_pure_operator_definitions: usize,
    /// Maximum pure-operator formals summed across definitions.
    pub max_pure_operator_formals: usize,
    /// Maximum exact component-calculus nodes summed across definitions.
    pub max_pure_operator_calculus_nodes: usize,
    /// Maximum ordered arguments summed across pure-operator applications.
    pub max_pure_operator_application_arguments: usize,
    /// Maximum Semantic Model members.
    pub max_model_view_members: usize,
    /// Maximum model-root boundary Ports.
    pub max_model_boundary: usize,
    /// Maximum rank of one exact Semantic Model value shape.
    pub max_value_shape_rank: usize,
    /// Maximum checked scalar components in one Semantic Model value shape.
    pub max_value_shape_components: usize,
    /// Maximum scalar payload components actually stored across all literals.
    /// A real/imaginary pair, Integer, or Boolean each consumes one component.
    /// Compact numeric zero payloads consume no component budget.
    pub max_value_literal_components: usize,
    /// Maximum ordered operations in one Model transaction.
    pub max_transaction_ops: usize,
    /// Maximum atomic preconditions in one Model transaction.
    pub max_transaction_preconditions: usize,
}

impl Default for ModelDecoderLimits {
    fn default() -> Self {
        Self {
            json: JsonDecoderLimits::default(),
            max_nodes: 100_000,
            max_edges: 1_000_000,
            max_expression_nodes: 1_000_000,
            max_expression_roots: 1_000_000,
            max_pure_operator_definitions: 100_000,
            max_pure_operator_formals: 1_000_000,
            max_pure_operator_calculus_nodes: 4_000_000,
            max_pure_operator_application_arguments: 4_000_000,
            max_model_view_members: 100_000,
            max_model_boundary: 100_000,
            max_value_shape_rank: 8,
            max_value_shape_components: 4_096,
            max_value_literal_components: 1_000_000,
            max_transaction_ops: 1_000_000,
            max_transaction_preconditions: 100_000,
        }
    }
}

pub(crate) fn checked_count_sum(
    counts: impl IntoIterator<Item = usize>,
    label: &str,
) -> Result<usize, Diagnostic> {
    counts.into_iter().try_fold(0_usize, |total, count| {
        total
            .checked_add(count)
            .ok_or_else(|| invalid_artifact(format!("{label} overflows usize")))
    })
}

pub(crate) fn require_decoder_count(
    label: &str,
    actual: usize,
    limit: usize,
) -> Result<(), Diagnostic> {
    if actual > limit {
        Err(invalid_artifact(format!(
            "{label} count {actual} exceeds decoder limit {limit}",
        )))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireNode {
    pub(crate) id: WireId,
    definition: WireNodeDefinition,
}

mod expression;
mod literal;
mod node;
mod primitive;
mod relation;
mod value_type;
mod vocabulary;

pub(crate) use expression::PureOperatorWireCounts;
pub(crate) use expression::WireValue;
pub(crate) use literal::WireValueLiteral;
pub(crate) use node::WireNodeDefinition;
pub(crate) use primitive::{WireEdge, WireEdgeKind, WireId, parse_ulid};
