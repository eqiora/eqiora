use super::AnalyzedResolvedHierarchy;

impl AnalyzedResolvedHierarchy {
    /// Describe known Model-scope type, role, activation and spatial support
    /// facts only when the resolved declaration's source identity matches.
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
