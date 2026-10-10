//! Coupled artifacts retain the exact authored trace-equality authority.
use super::*;
use eqiora_realization::ConformingTraceSource;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireTraceSource {
    ConservingConnection {
        connection_ulid: String,
    },
    RelationEquality {
        relation_ulid: String,
        root_index: u32,
    },
}

impl WireTraceSource {
    pub(super) fn encode(source: ConformingTraceSource) -> Self {
        match source {
            ConformingTraceSource::ConservingConnection(connection) => Self::ConservingConnection {
                connection_ulid: connection.ulid().to_string(),
            },
            ConformingTraceSource::RelationEquality {
                relation,
                root_index,
            } => Self::RelationEquality {
                relation_ulid: relation.ulid().to_string(),
                root_index,
            },
        }
    }

    pub(super) fn decode(self) -> Result<ConformingTraceSource, Diagnostic> {
        Ok(match self {
            Self::ConservingConnection { connection_ulid } => {
                ConformingTraceSource::ConservingConnection(parse_id(
                    &connection_ulid,
                    "Connection",
                )?)
            }
            Self::RelationEquality {
                relation_ulid,
                root_index,
            } => ConformingTraceSource::RelationEquality {
                relation: parse_id(&relation_ulid, "Relation")?,
                root_index,
            },
        })
    }
}
