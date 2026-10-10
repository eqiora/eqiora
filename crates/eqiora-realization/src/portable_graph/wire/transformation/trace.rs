//! The portable graph retains the complete authored equality authority.
use super::*;
use crate::ConformingTraceSource;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(in crate::portable_graph::wire) enum WireTraceSource {
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
                ConformingTraceSource::ConservingConnection(parse_id(&connection_ulid)?)
            }
            Self::RelationEquality {
                relation_ulid,
                root_index,
            } => ConformingTraceSource::RelationEquality {
                relation: parse_id(&relation_ulid)?,
                root_index,
            },
        })
    }
}
