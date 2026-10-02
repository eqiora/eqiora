//! Thin application entry to the shared reference execution owner.

use eqiora_core::{Diagnostic, RawId, ValueLiteral};
use eqiora_sem::{ExecutionSession, Interpreter, ReferenceConfig};

use crate::ModelDocument;

impl ModelDocument {
    /// Start bounded reference execution on this exact immutable Model.
    ///
    /// Samples are runtime input values, separate from static signature bindings.
    /// Each table identifies the exact input Port and its exact ClockDomain.
    ///
    /// # Errors
    /// Returns reference-profile, input-table, initialization, or clock diagnostics.
    pub fn execution_session(
        &self,
        config: ReferenceConfig,
        inputs: impl IntoIterator<Item = (RawId, RawId, Vec<ValueLiteral>)>,
    ) -> Result<ExecutionSession, Vec<Diagnostic>> {
        Interpreter::default()
            .execution_session(&self.program, config, inputs)
            .map_err(|errors| {
                errors
                    .into_iter()
                    .map(|error| self.with_source_origin(error))
                    .collect()
            })
    }

    fn with_source_origin(&self, diagnostic: Diagnostic) -> Diagnostic {
        if diagnostic.source_span().is_some() {
            return diagnostic;
        }
        let span = (|| {
            let [scope, kind, id] = diagnostic.graph_path()?.segments() else {
                return None;
            };
            if scope != "semantic" {
                return None;
            }
            let node = self.program.nodes().find(|node| {
                node.id().to_string() == *id && format!("{:?}", node.id().kind()) == *kind
            })?;
            self.source_provenance
                .as_ref()?
                .get_by_graph_id(node.id())
                .map(|origin| origin.definition_span().clone())
        })();
        match span {
            Some(span) => diagnostic.with_span(span),
            None => diagnostic,
        }
    }

    /// Resume an in-memory accepted execution checkpoint on this exact Model.
    ///
    /// # Errors
    /// Rejects a checkpoint whose complete immutable Model differs.
    pub fn resume_execution(
        &self,
        checkpoint: &ExecutionSession,
    ) -> Result<ExecutionSession, Vec<Diagnostic>> {
        Interpreter::default().resume_execution(&self.program, checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::diagnostic::codes;

    #[test]
    fn structural_failure_retains_relation_location_and_exact_occurrence() {
        let relation = "relation r { x = 0; x = 0; y + z = 0; }";
        let source = format!(
            "component Bad() {{ variable x: 1; variable y: 1; variable z: 1; {relation} }} model M() {{ instance bad: Bad(); }}"
        );
        let model = ModelDocument::compile("balance-location.eqi", &source).unwrap();
        let config = ReferenceConfig::new(0.0, 0.1).unwrap();
        let errors = model.execution_session(config, []).unwrap_err();
        let error = &errors[0];
        assert_eq!(error.code(), codes::NONLINEAR_SOLVE_FAILED);
        assert!(
            error
                .graph_path()
                .unwrap()
                .to_string()
                .contains(&model.aliases()["bad.r"].to_string())
        );
        let span = error
            .source_span()
            .expect("authored Relation has source provenance");
        assert!(span.file.ends_with("balance-location.eqi"));
        assert_eq!(span.start as usize, source.find(relation).unwrap());
        assert_eq!(
            span.end as usize,
            source.find(relation).unwrap() + relation.len()
        );

        // Bare semantic artifacts carry no source provenance. Replay must not
        // fabricate a source location from declaration IDs or source labels.
        let replayed = ModelDocument::replay(&model.canonical_json().unwrap()).unwrap();
        let replay_errors = replayed.execution_session(config, []).unwrap_err();
        assert_eq!(replay_errors[0].code(), error.code());
        assert_eq!(replay_errors[0].graph_path(), error.graph_path());
        assert!(replay_errors[0].source_span().is_none());
    }

    #[test]
    fn an_existing_location_is_not_replaced_by_a_secondary_origin() {
        let model =
            ModelDocument::compile("ok.eqi", "model M() { variable x: 1; relation r { x=1; } }")
                .unwrap();
        let span = eqiora_core::Span {
            file: "specific.eqi".into(),
            start: 3,
            end: 8,
        };
        let diagnostic = Diagnostic::error(codes::NONLINEAR_SOLVE_FAILED, "specific failure")
            .with_graph_path(eqiora_core::GraphPath::new([
                "semantic".to_owned(),
                "Relation".to_owned(),
                model.aliases()["r"].to_string(),
            ]))
            .with_span(span.clone());
        assert_eq!(
            model.with_source_origin(diagnostic).source_span(),
            Some(&span)
        );
        model
            .execution_session(ReferenceConfig::new(0.0, 0.1).unwrap(), [])
            .unwrap();
    }
}
