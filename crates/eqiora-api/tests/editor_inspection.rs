use eqiora_api::editor::EditorWorkspaceSnapshot;
use eqiora_compiler::{CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit};

#[test]
fn inspection_reuses_resolved_module_sources_and_rejects_unknown_files() {
    let owner = CompilationNamespaceId::new(["editor_test"]).unwrap();
    let root =
        "import editor_test.library as library; model Demo() { instance part: library.Part(); }";
    let library = "public component Part() { variable x: 1; relation balance { x = 2; } }";
    let root_unit = ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", root).unwrap();
    let file = root_unit.diagnostic_file();
    let input = ResolvedHierarchyInput::new(
        owner.clone(),
        vec![
            root_unit,
            ResolvedSourceUnit::new(owner, "src/library.eqi", library).unwrap(),
        ],
        vec![],
    );
    let snapshot = EditorWorkspaceSnapshot::analyze_modules(1, input);
    assert!(
        snapshot.diagnostics().is_empty(),
        "{:?}",
        snapshot.diagnostics()
    );
    let model = snapshot.compile_model(&file, "Demo").unwrap();
    assert!(model.aliases().keys().any(|name| name.contains("part")));
    assert!(snapshot.compile_model("not-in-graph.eqi", "Demo").is_err());
    let invalid = EditorWorkspaceSnapshot::analyze_standalone(2, "model Broken() { nonsense; }");
    assert!(invalid.compile_model("src/main.eqi", "Broken").is_err());
}
