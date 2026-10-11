//! **eqiora-schema** — executable kernel definitions and the open Standard
//! Ontology vocabulary (Layer L0).
//!
//! These concepts are typed named subgraphs over Semantic Kernel nodes, not
//! new node kinds. Their stable schema keys make them first-class in APIs,
//! transactions, serialization, and provenance while preserving a closed
//! kernel. Third-party crates may implement [`OntologySchema`] for their own
//! markers without modifying this crate.

use std::collections::BTreeSet;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, EntityKind, NamedSubgraph, OntologySchema, RawId};

pub mod kernel;
pub mod property_table;
pub mod resolved_array;

fn require_member_kind(
    schema: &str,
    members: &BTreeSet<RawId>,
    expected: EntityKind,
) -> Result<(), Diagnostic> {
    if members.iter().any(|member| member.kind() == expected) {
        Ok(())
    } else {
        Err(Diagnostic::error(
            codes::INVALID_ONTOLOGY_VIEW,
            format!("{schema} requires at least one {expected:?} member"),
        ))
    }
}

macro_rules! define_schema {
    ($(#[$doc:meta])* $marker:ident, $view:ident, $key:literal, $required:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $marker;

        impl OntologySchema for $marker {
            const KEY: &'static str = $key;

            fn validate(
                members: &BTreeSet<RawId>,
                _boundary: &BTreeSet<RawId>,
            ) -> Result<(), Diagnostic> {
                require_member_kind(Self::KEY, members, EntityKind::$required)
            }
        }

        $(#[$doc])*
        pub type $view = NamedSubgraph<$marker>;
    };
}

define_schema!(
    /// A reusable relation network with a typed Port boundary.
    Model,
    ModelView,
    "eqiora.model/v1",
    Relation
);
define_schema!(
    /// A relation that joins model boundaries.
    Coupling,
    CouplingView,
    "eqiora.coupling/v1",
    Relation
);
define_schema!(
    /// A scale scope anchored in a semantic Domain.
    Scale,
    ScaleView,
    "eqiora.scale/v1",
    Domain
);
define_schema!(
    /// An optimization objective expressed by one or more Relations.
    Objective,
    ObjectiveView,
    "eqiora.objective/v1",
    Relation
);
define_schema!(
    /// The semantic relation scope to which a realization SolverPlan applies.
    Solver,
    SolverView,
    "eqiora.solver/v1",
    Relation
);
/// A semantic scope supported by records in the Evidence & Artifact Graph.
///
/// Unlike the concrete `kinds::Evidence` graph node, this is an ontology
/// marker. Any non-empty semantic scope is valid, so the common structural
/// validator is sufficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvidenceSet;

impl OntologySchema for EvidenceSet {
    const KEY: &'static str = "eqiora.evidence-set/v1";
}

/// A typed named subgraph for [`EvidenceSet`].
pub type EvidenceSetView = NamedSubgraph<EvidenceSet>;

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::Id;
    use eqiora_core::entity::kinds;

    #[test]
    fn standard_schema_keys_are_stable_and_distinct() {
        let keys = [
            Model::KEY,
            Coupling::KEY,
            Scale::KEY,
            Objective::KEY,
            Solver::KEY,
            EvidenceSet::KEY,
        ];
        let unique = keys.into_iter().collect::<BTreeSet<_>>();

        assert_eq!(unique.len(), 6);
        assert!(unique.iter().all(|key| key.starts_with("eqiora.")));
    }

    #[test]
    fn model_requires_a_relation() {
        let field = Id::<kinds::Field>::new().erase();
        let diagnostic = ModelView::new(Default::default(), [field], [])
            .expect_err("a field alone is not a Model relation network");

        assert_eq!(diagnostic.code(), codes::INVALID_ONTOLOGY_VIEW);
    }
}
