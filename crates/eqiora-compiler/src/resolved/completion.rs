use super::AnalyzedResolvedHierarchy;

impl AnalyzedResolvedHierarchy {
    /// Resolve a value Name or Path in a Model or Component to its declaration:
    /// an owned Field, Parameter, or Port, or a public Port of a Model's direct
    /// child. The returned spelling covers the whole expression, so callers must
    /// check that the cursor is on its final identifier. Deeper members and
    /// nested binder scopes are not resolved. Uses the prepared index; no
    /// elaboration runs.
    #[must_use]
    pub fn value_definition(&self, file: &str, offset: u32) -> Option<(&str, eqiora_core::Span)> {
        self.completion.value_definition(file, offset)
    }

    /// Find the value Name and Path expressions that refer to a declaration,
    /// given by its whole Span: an owned Field, Parameter, or Port, or a direct
    /// child's public Port, as indexed by the prepared declaration scopes.
    /// Results are sorted by source file and range and exclude nested binders
    /// and deeper or private child members. Several instance spellings can refer
    /// to one declaration, so the results are declaration references, not
    /// instance identities, and do not support rename. An unused declaration
    /// returns `Some([])`; an unknown span returns `None`. Callers map each
    /// expression to its final identifier token.
    #[must_use]
    pub fn value_references(
        &self,
        declaration: &eqiora_core::Span,
    ) -> Option<Vec<eqiora_core::Span>> {
        self.completion.value_references(declaration)
    }

    /// Whether the position lies in a value Name or Path expression with this
    /// spelling inside a prepared declaration scope, outside unsupported binder
    /// scopes. The check is syntactic and does not resolve the declaration:
    /// callers must still check the cursor token and compare source identities
    /// with `symbol_description`, because expression ranges can include
    /// parentheses. Declaration names, units, and named activation clauses are
    /// not value expressions. No parsing or elaboration runs.
    #[must_use]
    pub fn is_value_reference(&self, file: &str, offset: u32, name: &str) -> bool {
        self.completion.is_value_reference(file, offset, name)
    }

    /// Describe a declaration's known type, role, activation, spatial support,
    /// and Model-local periodic schedule, provided the resolved declaration's
    /// source identity matches. Reads the immutable prepared index; no
    /// elaboration runs.
    #[must_use]
    pub fn symbol_description(
        &self,
        file: &str,
        offset: u32,
        name: &str,
        declaration: (&str, eqiora_lang::TextRange),
    ) -> Option<String> {
        self.completion.describe(file, offset, name, declaration)
    }

    /// Prepare the Model and unspecialized Component contracts used for editor
    /// assistance, once per analysis. Cancelling discards the partly built
    /// index. Existing definition scopes are reused; nothing is executed,
    /// solved, or read from packages.
    pub fn prepare_completion(&mut self, is_cancelled: impl FnMut() -> bool) -> bool {
        let Some(index) = crate::hierarchy::CompletionIndex::build(self, is_cancelled) else {
            return false;
        };
        self.completion = std::sync::Arc::new(index);
        true
    }

    /// Classify visible names using the prepared immutable contract index.
    ///
    /// Entries `(compatible, explanation)` follow candidate order; a `None`
    /// entry means compatibility is unknown. Unsupported positions return
    /// `None`. Explanations are display text and may be truncated, which does
    /// not affect the comparison. No parsing or elaboration runs.
    #[must_use]
    pub fn completion_compatibility(
        &self,
        file: &str,
        offset: u32,
        candidates: &[&str],
    ) -> Option<Vec<Option<(bool, String)>>> {
        self.completion.classify(file, offset, candidates)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit,
        analyze_resolved_hierarchy,
    };

    #[test]
    fn static_preparation_never_invents_free_extents_or_keeps_a_failed_partial_map() {
        for source in [
            "model M(parameter n:integer){variable values:array<1,n>;}",
            "model M(){parameter n:integer=3;let broken=1[m]+1[s];variable values:array<1,n>;}",
        ] {
            let owner = CompilationNamespaceId::new(["static_editor"]).unwrap();
            let unit = ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", source).unwrap();
            let file = unit.diagnostic_file();
            // Direct advisory preparation is also callable before body validation.
            let mut analysis =
                analyze_resolved_hierarchy(ResolvedHierarchyInput::new(owner, vec![unit], vec![]))
                    .unwrap();
            assert!(analysis.prepare_completion(|| false));
            let start = source.find("variable values").unwrap() as u32;
            let end = source[start as usize..].find(';').unwrap() as u32 + start + 1;
            assert!(
                analysis
                    .symbol_description(
                        &file,
                        start,
                        "values",
                        (&file, eqiora_lang::TextRange::new(start, end)),
                    )
                    .is_none()
            );
        }
    }

    #[test]
    fn value_occurrence_proof_keeps_units_declarations_and_wrong_names_out() {
        let owner = CompilationNamespaceId::new(["test"]).unwrap();
        let source = "component C(output value:1){} model M(){parameter m:1=1;parameter copy:1=(m);variable x:m;instance child:C();relation r{x=1[m];child.value=child.value;}}";
        let unit = ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", source).unwrap();
        let file = unit.diagnostic_file();
        let mut analysis =
            analyze_resolved_hierarchy(ResolvedHierarchyInput::new(owner, vec![unit], vec![]))
                .unwrap();
        assert!(analysis.prepare_completion(|| false));
        for (needle, shift, name, expected) in [
            ("m:1", 0, "m", false),
            ("x:m", 2, "m", false),
            ("[m]", 1, "m", false),
            ("(m)", 1, "m", true),
            ("(m)", 1, "copy", false),
            ("child.value", 6, "child.value", true),
            ("child.value", 6, "value", false),
        ] {
            assert_eq!(
                analysis.is_value_reference(
                    &file,
                    (source.find(needle).unwrap() + shift) as u32,
                    name
                ),
                expected,
                "{needle} / {name}"
            );
        }
        let offset = source.find("child.value").unwrap() as u32 + 6;
        let start = source.find("output value:1").unwrap() as u32;
        let declaration = eqiora_core::Span {
            file: file.clone(),
            start,
            end: start + "output value:1".len() as u32,
        };
        assert_eq!(
            analysis.value_definition(&file, offset),
            Some(("child.value", declaration.clone()))
        );
        let expected = source
            .match_indices("child.value")
            .map(|(start, name)| eqiora_core::Span {
                file: file.clone(),
                start: start as u32,
                end: (start + name.len()) as u32,
            })
            .collect::<Vec<_>>();
        assert_eq!(analysis.value_references(&declaration), Some(expected));
        for wrong in [
            eqiora_core::Span {
                file: "other".into(),
                ..declaration.clone()
            },
            eqiora_core::Span {
                start: declaration.start + 1,
                ..declaration
            },
        ] {
            assert!(analysis.value_references(&wrong).is_none());
        }
    }

    #[test]
    fn hover_requires_the_exact_declaration_within_the_current_model() {
        let owner = CompilationNamespaceId::new(["test"]).unwrap();
        let first = "variable value:m;";
        let second = "variable value:s;";
        let source = format!("model A(){{{first}}} model B(){{{second}}}");
        let declaration = |text: &str| {
            let start = source.find(text).unwrap() as u32;
            eqiora_lang::TextRange::new(start, start + text.len() as u32)
        };
        let first = declaration(first);
        let second = declaration(second);
        let offset = second.start() + 10;
        let unit = ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", source).unwrap();
        let file = unit.diagnostic_file();
        let mut analysis =
            analyze_resolved_hierarchy(ResolvedHierarchyInput::new(owner, vec![unit], vec![]))
                .unwrap();
        assert!(!analysis.prepare_completion(|| true));
        assert!(
            analysis
                .symbol_description(&file, offset, "value", (&file, second))
                .is_none()
        );
        assert!(analysis.prepare_completion(|| false));
        assert!(
            analysis
                .symbol_description(&file, offset, "value", (&file, second))
                .unwrap()
                .contains("dimension T")
        );
        assert!(
            analysis
                .symbol_description(&file, offset, "value", (&file, first))
                .is_none()
        );
        assert!(
            analysis
                .symbol_description(&file, offset, "value", ("other", second))
                .is_none()
        );
    }

    #[test]
    fn long_unicode_nominal_labels_are_bounded_without_changing_compatibility() {
        let owner = CompilationNamespaceId::new([
            "test".to_owned(),
            "測".repeat(300),
            "語".repeat(300),
            "型".repeat(300),
            "名".repeat(300),
        ])
        .unwrap();
        let source = "connector Pin {across potential:V; through flow:A;} component Ports(port source:Pin,port target:Pin) {} model M(){instance child:Ports();connect child.source,child.target;}";
        let offset = source.rfind("child.target").unwrap() as u32;
        let unit = ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", source).unwrap();
        let file = unit.diagnostic_file();
        let mut analysis =
            analyze_resolved_hierarchy(ResolvedHierarchyInput::new(owner, vec![unit], vec![]))
                .unwrap();
        assert!(analysis.prepare_completion(|| false));
        let classifications = analysis
            .completion_compatibility(&file, offset, &["child.target"])
            .unwrap();
        let (compatible, description) = classifications[0].as_ref().unwrap();
        assert!(*compatible);
        assert!(description.len() <= 2048);
        assert!(description.contains("nominal") && description.ends_with('…'));
    }

    #[test]
    fn cancelled_preparation_never_publishes_a_partial_index() {
        let owner = CompilationNamespaceId::new(["test"]).unwrap();
        let mut source = String::from("model M(){");
        for i in 0..128 {
            source.push_str(&format!("parameter value{i}:m=1[m];"));
        }
        source.push_str("parameter result:m=value0;}");
        let offset = source.rfind("value0").unwrap() as u32;
        let unit = ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", source).unwrap();
        let file = unit.diagnostic_file();
        let mut analysis =
            analyze_resolved_hierarchy(ResolvedHierarchyInput::new(owner, vec![unit], vec![]))
                .unwrap();
        let mut checkpoints = 0;
        assert!(!analysis.prepare_completion(|| {
            checkpoints += 1;
            checkpoints > 16
        }));
        assert!(
            analysis
                .completion_compatibility(&file, offset, &["value0"])
                .is_none()
        );
        assert!(analysis.prepare_completion(|| false));
        let classifications = analysis
            .completion_compatibility(&file, offset, &["value0", "missing"])
            .unwrap();
        assert_eq!(classifications[0].as_ref().map(|c| c.0), Some(true));
        assert!(classifications[1].is_none());
    }
}
