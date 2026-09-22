//! Prepared outline details share hover's compiler authority and invalidation.
use super::{EditorSnapshot, EditorSymbol, EditorSymbolKind};
use eqiora_compiler::AnalyzedResolvedHierarchy;

impl EditorSnapshot {
    pub(super) fn prepare_symbol_details(
        &mut self,
        file: &str,
        analysis: &AnalyzedResolvedHierarchy,
        is_cancelled: &mut impl FnMut() -> bool,
    ) -> bool {
        visit(&mut self.symbols, &mut |symbol| {
            if is_cancelled() {
                return false;
            }
            if matches!(
                symbol.kind,
                EditorSymbolKind::Field | EditorSymbolKind::Parameter | EditorSymbolKind::Port
            ) {
                symbol.detail = analysis.symbol_description(
                    file,
                    symbol.range.start(),
                    &symbol.name,
                    (file, symbol.range),
                );
            }
            true
        })
    }

    #[cfg(feature = "project-filesystem")]
    pub(super) fn clear_semantics(&mut self) {
        self.semantics = None;
        // Outline details are prepared facts only. Assistance candidates derive
        // their authored source heads separately and never mutate this outline.
        visit(&mut self.symbols, &mut |symbol| {
            symbol.detail = None;
            true
        });
    }
}

fn visit(symbols: &mut [EditorSymbol], action: &mut impl FnMut(&mut EditorSymbol) -> bool) -> bool {
    for symbol in symbols {
        if !action(symbol) || !visit(&mut symbol.children, action) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use crate::editor::EditorWorkspaceSnapshot;

    #[test]
    fn cancellation_during_outline_preparation_stops_before_the_next_symbol() {
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(
            1,
            "model M(){variable first:m;variable second:s;}",
        );
        let file = workspace.files().next().unwrap();
        let mut snapshot = workspace.document(file).unwrap().clone();
        let analysis = snapshot.semantics.as_ref().unwrap().analysis.clone();
        let mut calls = 0;
        assert!(!snapshot.prepare_symbol_details(file, &analysis, &mut || {
            calls += 1;
            calls == 2
        }));
        assert_eq!(calls, 2);
    }

    #[cfg(feature = "project-filesystem")]
    #[test]
    fn recovery_clears_cached_facts_even_in_documents_without_overrides() {
        use eqiora_compiler::{CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit};
        let owner = CompilationNamespaceId::new(["symbols"]).unwrap();
        let workspace = EditorWorkspaceSnapshot::analyze_modules(
            1,
            ResolvedHierarchyInput::new(
                owner.clone(),
                vec![
                    ResolvedSourceUnit::new(
                        owner.clone(),
                        "src/main.eqi",
                        "model Main(){variable value:m;}",
                    )
                    .unwrap(),
                    ResolvedSourceUnit::new(
                        owner,
                        "src/other.eqi",
                        "model Other(){variable value:s;}",
                    )
                    .unwrap(),
                ],
                vec![],
            ),
        );
        assert!(workspace.diagnostics().is_empty());
        for file in workspace.files() {
            assert!(
                workspace.document(file).unwrap().symbols()[0].children()[0]
                    .detail()
                    .is_some()
            );
        }
        let file = workspace
            .files()
            .find(|file| file.ends_with("src/main.eqi"))
            .unwrap()
            .to_owned();
        let recovered = workspace.recover_overrides(
            &std::collections::BTreeMap::from([(file, "model Main(){variable revised:K;".into())]),
            eqiora_core::Diagnostic::error(
                eqiora_core::diagnostic::codes::INVALID_TOKEN,
                "incomplete overlay",
            ),
        );
        for file in recovered.files() {
            let snapshot = recovered.document(file).unwrap();
            assert!(snapshot.semantics.is_none());
            let model = &snapshot.symbols()[0];
            let child = &model.children()[0];
            assert!(child.detail().is_none());
            assert_eq!(
                child.name(),
                if model.name() == "Main" {
                    "revised"
                } else {
                    "value"
                }
            );
        }
    }
}
