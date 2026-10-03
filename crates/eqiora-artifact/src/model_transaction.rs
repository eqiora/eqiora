//! Shared wire vocabulary for the current Semantic Model transaction.

use eqiora_core::{Diagnostic, EntityKind, GraphClass, OntologyId, RawId};
use eqiora_graph::{EdgeKind, Op, Precondition, Revision};
use eqiora_schema::{Model, ModelView};
use serde::{Deserialize, Serialize};

use crate::model::{WireEdgeKind, WireId, WireNode, WireValueLiteral, parse_ulid};
use crate::{ModelDecoderLimits, invalid_artifact};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireModelOp {
    DefineKernelNode {
        node: WireNode,
    },
    SetValue {
        target: WireId,
        value: WireValueLiteral,
    },
    Connect {
        from: WireId,
        to: WireId,
        edge: WireEdgeKind,
    },
    RemoveNode {
        id: WireId,
    },
    DefineModelView {
        view: WireModelView,
    },
    RemoveModelView {
        ulid: String,
    },
}

impl WireModelOp {
    pub(crate) fn encode(op: &Op) -> Result<Self, Diagnostic> {
        match op {
            Op::DefineKernelNode { node } => Ok(Self::DefineKernelNode {
                node: WireNode::encode(node)?,
            }),
            Op::SetValue { target, value } => {
                require_semantic_id(*target, "SetValue target")?;
                if target.kind() != EntityKind::Parameter {
                    return Err(invalid_artifact("SetValue requires a Parameter"));
                }
                Ok(Self::SetValue {
                    target: WireId::from_raw(*target),
                    value: WireValueLiteral::encode(value)?,
                })
            }
            Op::Connect { from, to, edge } => {
                require_semantic_id(*from, "Connect source")?;
                require_semantic_id(*to, "Connect target")?;
                if !operation_edge_permitted(*edge, from.kind(), to.kind()) {
                    return Err(invalid_artifact(
                        "model transaction edge is unsupported by the current Model contract",
                    ));
                }
                Ok(Self::Connect {
                    from: WireId::from_raw(*from),
                    to: WireId::from_raw(*to),
                    edge: WireEdgeKind::encode(*edge)?,
                })
            }
            Op::RemoveNode { id } => {
                require_semantic_id(*id, "RemoveNode target")?;
                Ok(Self::RemoveNode {
                    id: WireId::from_raw(*id),
                })
            }
            Op::DefineOntologyView { view } => {
                let view = view.downcast::<Model>().ok_or_else(|| {
                    invalid_artifact(format!(
                        "model transaction cannot define ontology schema `{}`",
                        view.id().schema()
                    ))
                })?;
                Ok(Self::DefineModelView {
                    view: WireModelView::encode(&view),
                })
            }
            Op::RemoveOntologyView { id } => {
                let id = id.downcast::<Model>().ok_or_else(|| {
                    invalid_artifact(format!(
                        "model transaction cannot remove ontology schema `{}`",
                        id.schema()
                    ))
                })?;
                Ok(Self::RemoveModelView {
                    ulid: id.ulid().to_string(),
                })
            }
            Op::AddNode { kind, .. } => Err(invalid_artifact(format!(
                "infrastructure node {kind:?} cannot enter a Semantic Model transaction"
            ))),
            _ => Err(invalid_artifact(
                "operation is unsupported by the current Model transaction vocabulary",
            )),
        }
    }

    pub(crate) fn decode(&self) -> Result<Op, Diagnostic> {
        match self {
            Self::DefineKernelNode { node } => Ok(Op::DefineKernelNode {
                node: node.decode()?,
            }),
            Self::SetValue { target, value } => {
                let target = target.decode_raw()?;
                if target.kind() != EntityKind::Parameter {
                    return Err(invalid_artifact("SetValue requires a Parameter"));
                }
                Ok(Op::SetValue {
                    target,
                    value: value.decode()?,
                })
            }
            Self::Connect { from, to, edge } => {
                let from = from.decode_raw()?;
                let to = to.decode_raw()?;
                let edge = edge.decode();
                if !edge.permits(from.kind(), to.kind()) {
                    return Err(invalid_artifact(
                        "wire edge endpoints violate the closed Semantic Model edge schema",
                    ));
                }
                Ok(Op::Connect { from, to, edge })
            }
            Self::RemoveNode { id } => Ok(Op::RemoveNode {
                id: id.decode_raw()?,
            }),
            Self::DefineModelView { view } => Ok(Op::DefineOntologyView {
                view: view.decode()?.into(),
            }),
            Self::RemoveModelView { ulid } => Ok(Op::RemoveOntologyView {
                id: OntologyId::<Model>::from_ulid(parse_ulid(ulid)?).erase(),
            }),
        }
    }

    pub(crate) fn expression_node_count(&self) -> usize {
        match self {
            Self::DefineKernelNode { node } => node.expression_node_count(),
            Self::SetValue { .. }
            | Self::Connect { .. }
            | Self::RemoveNode { .. }
            | Self::DefineModelView { .. }
            | Self::RemoveModelView { .. } => 0,
        }
    }

    pub(crate) fn expression_root_count(&self) -> usize {
        match self {
            Self::DefineKernelNode { node } => node.expression_root_count(),
            Self::SetValue { .. }
            | Self::Connect { .. }
            | Self::RemoveNode { .. }
            | Self::DefineModelView { .. }
            | Self::RemoveModelView { .. } => 0,
        }
    }

    pub(crate) fn pure_operator_counts(
        &self,
    ) -> Result<crate::model::PureOperatorWireCounts, Diagnostic> {
        match self {
            Self::DefineKernelNode { node } => node.pure_operator_counts(),
            _ => Ok(crate::model::PureOperatorWireCounts::default()),
        }
    }

    pub(crate) fn validate_pure_operator_features(&self) -> Result<(), Diagnostic> {
        match self {
            Self::DefineKernelNode { node } => node.validate_pure_operator_features(),
            _ => Ok(()),
        }
    }

    pub(crate) fn canonicalize_pure_operator_definitions(&mut self) -> Result<(), Diagnostic> {
        match self {
            Self::DefineKernelNode { node } => node.canonicalize_pure_operator_definitions(),
            _ => Ok(()),
        }
    }

    pub(crate) fn model_view_counts(&self) -> (usize, usize) {
        match self {
            Self::DefineModelView { view } => (view.members.len(), view.boundary.len()),
            _ => (0, 0),
        }
    }

    pub(crate) fn literal_component_count(&self) -> Result<usize, Diagnostic> {
        match self {
            Self::DefineKernelNode { node } => node.literal_component_count(),
            Self::SetValue { value, .. } => Ok(value.component_payload_count()),
            _ => Ok(0),
        }
    }

    pub(crate) fn ensure_value_shape_limits(
        &self,
        limits: ModelDecoderLimits,
    ) -> Result<(), Diagnostic> {
        match self {
            Self::DefineKernelNode { node } => node.ensure_value_shape_limits(limits),
            Self::SetValue { value, .. } => value.ensure_limits(limits),
            _ => Ok(()),
        }
    }

    pub(crate) fn canonicalize_sets(&mut self) -> Result<(), Diagnostic> {
        if let Self::DefineModelView { view } = self {
            view.members.sort();
            reject_duplicates(&view.members, "model-view member")?;
            view.boundary.sort();
            reject_duplicates(&view.boundary, "model-view boundary")?;
        }
        Ok(())
    }
}

fn operation_edge_permitted(edge: EdgeKind, from: EntityKind, to: EntityKind) -> bool {
    edge.permits(from, to)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "condition", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireModelPrecondition {
    ValueEquals {
        target: WireId,
        expected: Box<WireValueLiteral>,
    },
    RevisionIs {
        revision: u64,
    },
}

impl WireModelPrecondition {
    pub(crate) fn literal_component_count(&self) -> usize {
        match self {
            Self::ValueEquals { expected, .. } => expected.component_payload_count(),
            _ => 0,
        }
    }
    pub(crate) fn ensure_value_shape_limits(
        &self,
        limits: crate::ModelDecoderLimits,
    ) -> Result<(), Diagnostic> {
        match self {
            Self::ValueEquals { expected, .. } => expected.ensure_limits(limits),
            _ => Ok(()),
        }
    }

    pub(crate) fn encode(precondition: &Precondition) -> Result<Self, Diagnostic> {
        match precondition {
            Precondition::ValueEquals { target, expected } => {
                require_semantic_id(*target, "ValueEquals target")?;
                Ok(Self::ValueEquals {
                    target: WireId::from_raw(*target),
                    expected: Box::new(WireValueLiteral::encode(expected)?),
                })
            }
            Precondition::RevisionIs(revision) => Ok(Self::RevisionIs {
                revision: revision.0,
            }),
            _ => Err(invalid_artifact(
                "precondition is newer than the supported model transaction wire vocabulary",
            )),
        }
    }

    pub(crate) fn decode(&self) -> Result<Precondition, Diagnostic> {
        match self {
            Self::ValueEquals { target, expected } => Ok(Precondition::ValueEquals {
                target: target.decode_raw()?,
                expected: expected.decode()?,
            }),
            Self::RevisionIs { revision } => Ok(Precondition::RevisionIs(Revision(*revision))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireModelView {
    ulid: String,
    members: Vec<WireId>,
    boundary: Vec<WireId>,
}

impl WireModelView {
    fn encode(view: &ModelView) -> Self {
        Self {
            ulid: view.id().ulid().to_string(),
            members: view
                .members()
                .iter()
                .copied()
                .map(WireId::from_raw)
                .collect(),
            boundary: view
                .boundary()
                .iter()
                .copied()
                .map(WireId::from_raw)
                .collect(),
        }
    }

    fn decode(&self) -> Result<ModelView, Diagnostic> {
        ensure_sorted_unique(&self.members, "model-view member")?;
        ensure_sorted_unique(&self.boundary, "model-view boundary")?;
        let members = self
            .members
            .iter()
            .map(WireId::decode_raw)
            .collect::<Result<Vec<_>, _>>()?;
        let boundary = self
            .boundary
            .iter()
            .map(WireId::decode_raw)
            .collect::<Result<Vec<_>, _>>()?;
        ModelView::new(
            OntologyId::from_ulid(parse_ulid(&self.ulid)?),
            members,
            boundary,
        )
        .map_err(|error| invalid_artifact(error.message()))
    }
}

fn require_semantic_id(id: RawId, label: &str) -> Result<(), Diagnostic> {
    if id.kind().graph() == GraphClass::Semantic && is_wire_kernel_kind(id.kind()) {
        Ok(())
    } else {
        Err(invalid_artifact(format!(
            "{label} {id} is outside the Semantic Model wire scope"
        )))
    }
}

const fn is_wire_kernel_kind(kind: EntityKind) -> bool {
    matches!(
        kind,
        EntityKind::Observable
            | EntityKind::Record
            | EntityKind::RecordInstance
            | EntityKind::Enum
            | EntityKind::FiniteSpace
            | EntityKind::IndexSet
            | EntityKind::Domain
            | EntityKind::Representation
            | EntityKind::Field
            | EntityKind::Parameter
            | EntityKind::Port
            | EntityKind::Relation
            | EntityKind::Activation
            | EntityKind::Connection
            | EntityKind::ClockDomain
    )
}

fn ensure_sorted_unique<T: Ord>(values: &[T], label: &str) -> Result<(), Diagnostic> {
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        Err(invalid_artifact(format!(
            "{label} IDs must be sorted and unique in canonical wire data"
        )))
    } else {
        Ok(())
    }
}

fn reject_duplicates<T: PartialEq>(values: &[T], label: &str) -> Result<(), Diagnostic> {
    if values.windows(2).any(|pair| pair[0] == pair[1]) {
        Err(invalid_artifact(format!(
            "model transaction contains a duplicate {label} ID"
        )))
    } else {
        Ok(())
    }
}
