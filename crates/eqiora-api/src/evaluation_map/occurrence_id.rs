//! Execution identity is independent of requested points and accepted content.
use std::{fmt, str::FromStr};

use eqiora_core::Diagnostic;
use ulid::Ulid;

use super::invalid;

/// One request position in one execution, including failed or unstarted positions.
/// Equal points and repeated sample IDs retain distinct occurrence identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EvaluationMapOccurrenceId {
    execution: Ulid,
    index: usize,
}

impl EvaluationMapOccurrenceId {
    pub(super) const fn new(execution: Ulid, index: usize) -> Self {
        Self { execution, index }
    }

    /// Position in the original complete requested inventory.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }

    /// Execution identity shared by all positions of this execution.
    #[must_use]
    pub fn execution_id(self) -> String {
        self.execution.to_string()
    }
}

impl fmt::Display for EvaluationMapOccurrenceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.execution, self.index)
    }
}

impl FromStr for EvaluationMapOccurrenceId {
    type Err = Diagnostic;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let invalid = || invalid("invalid canonical evaluation occurrence identity");
        let (execution, index) = value.split_once(':').ok_or_else(invalid)?;
        let id = Self {
            execution: execution.parse().map_err(|_| invalid())?,
            index: index.parse().map_err(|_| invalid())?,
        };
        if id.to_string() != value {
            return Err(invalid());
        }
        Ok(id)
    }
}
