use super::AnalyzedResolvedHierarchy;

impl AnalyzedResolvedHierarchy {
    /// Resolve a Model value Name or Path occurrence to its stored declaration:
    /// a same-file Field/Parameter/Port or a direct child's public Port. The returned
    /// spelling covers the whole expression; callers must prove the cursor is
    /// on its terminal identifier. Deeper members and nested binder scopes are
    /// unavailable. Uses the prepared index without elaboration.
    #[must_use]
    pub fn value_definition(&self, file: &str, offset: u32) -> Option<(&str, eqiora_core::Span)> {
        self.completion.value_definition(file, offset)
    }

    /// Find value Name/Path expressions referring to an exact whole declaration
    /// Span already admitted by prepared Model scopes: an owned Field, Parameter
    /// or Port, or a direct child's public Port. Results are sorted by source file
    /// and range, excluding nested binders and unsupported deeper/private members.
    /// Multiple instance spellings may refer to the same source declaration;
    /// this is declaration provenance, not occurrence identity or rename support.
    /// An admitted unused declaration returns `Some([])`; an unknown key returns
    /// `None`. Callers project each expression to its terminal identifier token.
    #[must_use]
    pub fn value_references(
        &self,
        declaration: &eqiora_core::Span,
    ) -> Option<Vec<eqiora_core::Span>> {
        self.completion.value_references(declaration)
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
