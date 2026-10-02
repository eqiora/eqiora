//! Source-aware access to the shared scalar equation-analysis owner.
use crate::ModelDocument;
use eqiora_core::{Diagnostic, RawId, Span};
use eqiora_sem::{EquationAnalysis, Interpreter};

impl ModelDocument {
    /// Inspect continuous scalar equation incidence without requiring balance or
    /// solving initial values. Row owners can be resolved with [`Self::definition_span`].
    /// The declared-rate partition is a structural candidate, not a numerical certificate.
    ///
    /// # Errors
    /// Returns unsupported-profile or connection/activation admission diagnostics.
    pub fn equation_analysis(&self) -> Result<EquationAnalysis, Diagnostic> {
        Interpreter::new()
            .analyze_equations(&self.program)
            .map_err(|error| self.with_source_origin(error))
    }

    /// Retained source definition origin for an exact node of this Model.
    /// Bare artifact replay and reconstructed transaction documents have no
    /// source sidecar. Multiple origins use the first retained definition;
    /// this is not an individual equation span or a synthesized location.
    #[must_use]
    pub fn definition_span(&self, node: RawId) -> Option<&Span> {
        self.program.node(node)?;
        self.source_provenance
            .as_ref()?
            .get_by_graph_id(node)
            .map(|origin| origin.definition_span())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbalanced_component_report_retains_exact_occurrence_and_source_origin() {
        let relation = "relation r { x=0; x=0; }";
        let source = format!(
            "component C() {{ variable x: 1; variable y: 1; {relation} }} model M() {{ instance c: C(); instance d: C(); }}"
        );
        let model = ModelDocument::compile("analysis-origin.eqi", &source).unwrap();
        let report = model.equation_analysis().unwrap();
        assert_eq!(report.balance().rank(), 2);
        assert_eq!(report.equations().len(), 4);
        let owners = report
            .equations()
            .iter()
            .map(|row| row.owner())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            owners,
            [model.aliases()["c.r"], model.aliases()["d.r"]]
                .into_iter()
                .collect()
        );
        for equation in report.equations() {
            let span = model.definition_span(equation.owner()).unwrap();
            assert_eq!(span.start as usize, source.find(relation).unwrap());
            assert_eq!(
                span.end as usize,
                source.find(relation).unwrap() + relation.len()
            );
        }
        let replay = ModelDocument::replay(&model.canonical_json().unwrap()).unwrap();
        assert_eq!(replay.equation_analysis().unwrap(), report);
        assert!(
            report
                .equations()
                .iter()
                .all(|row| replay.definition_span(row.owner()).is_none())
        );
        let foreign = eqiora_core::Id::<eqiora_core::entity::kinds::Relation>::new().erase();
        assert!(model.definition_span(foreign).is_none());
    }
}
