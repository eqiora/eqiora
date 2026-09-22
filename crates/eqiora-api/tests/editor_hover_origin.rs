use eqiora_api::editor::{EditorWorkspaceService, EditorWorkspaceSnapshot};
use eqiora_compiler::{
    CompilationNamespaceId, ResolvedDependency, ResolvedHierarchyInput, ResolvedSourceUnit,
};

const LIBRARY: &str = "public component Part(output value:1){}";

fn input(main: &str, right: &str) -> ResolvedHierarchyInput {
    let root = CompilationNamespaceId::new(["app"]).unwrap();
    let left = CompilationNamespaceId::new(["left", "exact-a", "digest-a"]).unwrap();
    let right_owner = CompilationNamespaceId::new(["right", "exact-b", "digest-b"]).unwrap();
    ResolvedHierarchyInput::new(
        root.clone(),
        vec![
            ResolvedSourceUnit::new(root.clone(), "src/main.eqi", main).unwrap(),
            ResolvedSourceUnit::new(left.clone(), "src/main.eqi", LIBRARY).unwrap(),
            ResolvedSourceUnit::new(right_owner.clone(), "src/main.eqi", right).unwrap(),
        ],
        vec![
            ResolvedDependency::new(root.clone(), left),
            ResolvedDependency::new(root, right_owner),
        ],
    )
}

fn detail(workspace: &EditorWorkspaceSnapshot, source: &str, name: &str) -> String {
    let file = workspace
        .files()
        .find(|file| {
            workspace
                .document(file)
                .unwrap()
                .symbols()
                .iter()
                .any(|symbol| symbol.name() == "M")
        })
        .unwrap();
    let offset = (source.find(name).unwrap() + 2) as u32;
    workspace
        .assistance(file, offset, name)
        .unwrap()
        .detail()
        .unwrap()
        .to_owned()
}

#[test]
fn identical_exports_keep_exact_target_origins_across_alias_changes_and_versions() {
    let main = "import left.main as l;import right.main as r;model M(){instance a:l.Part();instance b:r.Part();relation law{a.value=b.value;}}";
    let old = EditorWorkspaceSnapshot::analyze_modules(1, input(main, LIBRARY));
    assert!(old.diagnostics().is_empty(), "{:?}", old.diagnostics());
    let left = detail(&old, main, "a.value");
    let right = detail(&old, main, "b.value");
    assert!(left.contains("Origin namespace: [\"left\", \"exact-a\", \"digest-a\"]"));
    assert!(right.contains("Origin namespace: [\"right\", \"exact-b\", \"digest-b\"]"));
    for (text, module) in [(&left, "left.main"), (&right, "right.main")] {
        assert!(text.contains(&format!("Module: {module:?}")));
        assert!(text.contains("Source file: \"src/main.eqi\""));
    }
    let file = old
        .files()
        .find(|file| {
            old.document(file)
                .unwrap()
                .symbols()
                .iter()
                .any(|symbol| symbol.name() == "M")
        })
        .unwrap();
    for (reference, namespace) in [
        ("l.Part", ["left", "exact-a", "digest-a"]),
        ("r.Part", ["right", "exact-b", "digest-b"]),
    ] {
        let (definition, source) = old
            .hover(file, main.find(reference).unwrap() as u32 + 2)
            .unwrap();
        assert_eq!(source, LIBRARY);
        assert_eq!(definition.namespace(), &namespace);
        assert_eq!(definition.path(), "main.Part");
    }
    let renamed = main
        .replace("as l;", "as renamed;")
        .replace("l.Part", "renamed.Part");
    let mut service = EditorWorkspaceService::new(old.clone());
    service.begin(2).unwrap();
    assert!(service.current().is_none());
    assert!(service.replace(old).is_err());
    let current = service
        .replace(EditorWorkspaceSnapshot::analyze_modules(
            2,
            input(&renamed, LIBRARY),
        ))
        .unwrap();
    assert_eq!(detail(current, &renamed, "a.value"), left);
    assert_eq!(detail(current, &renamed, "b.value"), right);
}

#[test]
fn contextless_or_invalid_workspaces_do_not_claim_an_exact_origin() {
    let main = "import left.main as l;import right.main as r;model M(){instance a:l.Part();instance b:r.Part();relation law{a.value=b.value;}}";
    let invalid_library = format!("{LIBRARY} model Broken(){{variable x:m;relation r{{x=1[s];}}}}");
    let invalid = EditorWorkspaceSnapshot::analyze_modules(1, input(main, &invalid_library));
    assert!(!invalid.diagnostics().is_empty());
    assert!(!detail(&invalid, main, "a.value").contains("Origin namespace:"));
    let source = "model M(){parameter value:1=1;relation r{value=1;}}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    let file = workspace.files().next().unwrap();
    let offset = source.rfind("value").unwrap() as u32;
    assert!(
        workspace
            .assistance(file, offset, "value")
            .unwrap()
            .detail()
            .unwrap()
            .contains("Origin namespace:")
    );
    assert!(
        !workspace
            .document(file)
            .unwrap()
            .assistance(offset, "value")
            .unwrap()
            .detail()
            .unwrap()
            .contains("Origin namespace:")
    );
    let recovering = EditorWorkspaceSnapshot::analyze_standalone(
        2,
        "model M(){parameter value:1=1;relation r{value",
    );
    let file = recovering.files().next().unwrap();
    let symbol = recovering.assistance(file, 20, "value");
    assert!(symbol.is_none_or(|symbol| {
        !symbol
            .detail()
            .unwrap_or_default()
            .contains("Origin namespace:")
    }));
}

#[test]
fn local_modules_and_opaque_namespace_segments_remain_distinct_literals() {
    let namespace =
        CompilationNamespaceId::new(["local", "a::b", "opaque````\n[run](command:run)"]).unwrap();
    let main = "import local.left as l;import local.right as r;model M(){instance a:l.Part();instance b:r.Part();relation law{a.value=b.value;}}";
    let root = ResolvedSourceUnit::new(namespace.clone(), "src/main.eqi", main).unwrap();
    let file = root.diagnostic_file();
    let workspace = EditorWorkspaceSnapshot::analyze_modules(
        1,
        ResolvedHierarchyInput::new(
            namespace.clone(),
            vec![
                root,
                ResolvedSourceUnit::new(namespace.clone(), "src/left.eqi", LIBRARY).unwrap(),
                ResolvedSourceUnit::new(namespace, "src/right.eqi", LIBRARY).unwrap(),
            ],
            vec![],
        ),
    );
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    for (name, module, path) in [
        ("a.value", "local.left", "src/left.eqi"),
        ("b.value", "local.right", "src/right.eqi"),
    ] {
        let detail = workspace
            .assistance(&file, main.find(name).unwrap() as u32 + 2, name)
            .unwrap()
            .detail()
            .unwrap()
            .to_owned();
        assert!(detail.contains("[\"local\", \"a::b\", \"opaque````\\n[run](command:run)\"]"));
        assert!(detail.contains(&format!("Module: {module:?}")));
        assert!(detail.contains(&format!("Source file: {path:?}")));
        assert!(!detail.contains("\n[run]"));
    }
}
