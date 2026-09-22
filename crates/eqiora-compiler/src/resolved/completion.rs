use super::AnalyzedResolvedHierarchy;

impl AnalyzedResolvedHierarchy {
    /// Resolve a simple value-name occurrence to a same-file Model field or
    /// parameter declaration. Nested binder scopes and qualified members are
    /// unavailable. Uses the prepared immutable scope without elaboration.
    #[must_use]
    pub fn local_definition(
        &self,
        file: &str,
        offset: u32,
    ) -> Option<(&str, eqiora_lang::TextRange)> {
        self.completion.local_definition(file, offset)
    }

    /// Query a named Field or Parameter in the Model scope containing `offset`.
    /// Returns its same-file declaration and value-name expression ranges in
    /// source order, excluding nested binder scopes. The caller projects exact
    /// identifier tokens and validates whether its cursor names this target.
    #[must_use]
    pub fn local_references(
        &self,
        file: &str,
        offset: u32,
        name: &str,
    ) -> Option<(eqiora_lang::TextRange, Vec<eqiora_lang::TextRange>)> {
        self.completion.local_references(file, offset, name)
    }

    /// Whether the position lies in a prepared Model value Name or Path
    /// expression with this full spelling, outside unsupported binder scopes.
    /// This syntactic occurrence check does not resolve the target declaration.
    /// Callers must also validate the exact cursor token and match the target's
    /// source identity with `symbol_description`; expression ranges can include
    /// parentheses. Declaration names, units and named activation clauses are
    /// not value expressions. No parsing or elaboration occurs during this query.
    #[must_use]
    pub fn is_value_reference(&self, file: &str, offset: u32, name: &str) -> bool {
        self.completion.is_value_reference(file, offset, name)
    }

    /// Describe known Model-scope type, role, activation, spatial support and
    /// exact local periodic schedule facts only when the resolved declaration's
    /// source identity matches.
    /// The prepared index is immutable; this query performs no elaboration.
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

    /// Prepare advisory Model contracts for completion and hover once per
    /// immutable analysis. Cancellation discards the whole new index. Existing
    /// bounded definition scopes are reused; no execution graph, package I/O,
    /// or solve is performed.
    pub fn prepare_completion(&mut self, is_cancelled: impl FnMut() -> bool) -> bool {
        let Some(index) = crate::hierarchy::CompletionIndex::build(self, is_cancelled) else {
            return false;
        };
        self.completion = std::sync::Arc::new(index);
        true
    }

    /// Classify visible names using the prepared immutable contract index.
    ///
    /// Entries `(compatible, explanation)` follow candidate order. `None`
    /// preserves an unknown candidate without claiming compatibility. Unsupported
    /// positions return `None`. Descriptions are bounded presentation text;
    /// truncation never affects the exact contract comparison. This query
    /// performs no parsing or elaboration.
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
        // Extending the shared syntax ranges to paths must not broaden local
        // definition/reference navigation beyond simple Fields and Parameters.
        let offset = source.find("child.value").unwrap() as u32 + 6;
        assert!(analysis.local_definition(&file, offset).is_none());
        assert!(
            analysis
                .local_references(&file, offset, "child.value")
                .is_none()
        );
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
