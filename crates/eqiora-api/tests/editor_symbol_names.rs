use eqiora_api::editor::{
    EditorService, EditorSnapshot, EditorSymbol, EditorSymbolKind, EditorWorkspaceSnapshot,
};
use eqiora_lang::TextRange;

fn symbol<'a>(snapshot: &'a EditorSnapshot, owner: &str, child: Option<&str>) -> &'a EditorSymbol {
    let parent = snapshot
        .symbols()
        .iter()
        .find(|symbol| symbol.name() == owner)
        .unwrap();
    child.map_or(parent, |name| {
        parent
            .children()
            .iter()
            .find(|symbol| symbol.name() == name)
            .unwrap()
    })
}

fn exact_name(source: &str, symbol: &EditorSymbol, head: &str, prefix: &str, name: &str) {
    let start = source.find(head).unwrap() + prefix.len();
    let expected = TextRange::new(start as u32, (start + name.len()) as u32);
    assert_eq!(symbol.name_range(), Some(expected), "{head}");
    assert_eq!(
        &source[expected.start() as usize..expected.end() as usize],
        name
    );
    assert!(symbol.range().start() <= expected.start());
    assert!(expected.end() <= symbol.range().end());
    assert_ne!(
        symbol.range(),
        expected,
        "full declaration range remains separate"
    );
}

#[test]
fn outline_names_share_exact_header_tokens_with_definitions() {
    let source = "// 🧪\r\ncomponent component(parameter parameter @{p}:1=2){let scale:1=parameter;}\r\nmodel model(){parameter gain:1=3;clock tick=periodic(1[s]);state x:m;event hit=crossing(x,direction=falling);indexset Rows=range(2);\r\n/// variable variable.\r\nvariable variable @{v}:1;let held:m at hit=pre(x);relation reset at hit{next(x)=1[m];}observable total:1=variable;relation balance[row in Rows]{variable=gain;}}";
    let service = EditorService::new("names.eqi", 1, source);
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        service.current().diagnostics().is_empty(),
        "{:?}",
        service.current().diagnostics()
    );
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    for snapshot in [service.current(), workspace.document(file).unwrap()] {
        for (owner, child, head, prefix, name) in [
            (
                "component",
                None,
                "component component",
                "component ",
                "component",
            ),
            (
                "component",
                Some("parameter"),
                "parameter parameter @{p}",
                "parameter ",
                "parameter",
            ),
            ("component", Some("scale"), "let scale", "let ", "scale"),
            ("model", None, "model model", "model ", "model"),
            (
                "model",
                Some("gain"),
                "parameter gain",
                "parameter ",
                "gain",
            ),
            ("model", Some("tick"), "clock tick", "clock ", "tick"),
            ("model", Some("hit"), "event hit", "event ", "hit"),
            ("model", Some("Rows"), "indexset Rows", "indexset ", "Rows"),
            (
                "model",
                Some("variable"),
                "variable variable @{v}",
                "variable ",
                "variable",
            ),
            ("model", Some("held"), "let held", "let ", "held"),
            (
                "model",
                Some("total"),
                "observable total",
                "observable ",
                "total",
            ),
            (
                "model",
                Some("balance"),
                "relation balance",
                "relation ",
                "balance",
            ),
        ] {
            exact_name(source, symbol(snapshot, owner, child), head, prefix, name);
        }
    }
    for definition in workspace.definitions() {
        let name = definition.path().rsplit('.').next().unwrap();
        if ["component", "model"].contains(&name) {
            assert_eq!(
                definition.name_range(),
                symbol(workspace.document(file).unwrap(), name, None).name_range()
            );
        }
    }
}

#[test]
fn recovered_names_track_current_text_without_guessing_import_aliases() {
    let original = "// 🧪\r\nmodel Good(){variable variable @{v}:1;relation r{variable=1;}}";
    let mut service = EditorService::new("names.eqi", 1, original);
    let changed = format!("// moved\r\n{original}\r\nmodel Broken(){{");
    let current = service.replace(2, &changed).unwrap();
    assert!(!current.diagnostics().is_empty());
    exact_name(
        &changed,
        symbol(current, "Good", None),
        "model Good",
        "model ",
        "Good",
    );
    exact_name(
        &changed,
        symbol(current, "Good", Some("variable")),
        "variable variable @{v}",
        "variable ",
        "variable",
    );
    assert!(service.snapshot(1).is_err());
    let imported = "import editor.workspace.library as editor; model M(){relation r{1=1;}}";
    let current = service.replace(3, imported).unwrap();
    let alias = current
        .symbols()
        .iter()
        .find(|s| s.kind() == EditorSymbolKind::Import)
        .unwrap();
    assert_eq!(alias.name(), "editor");
    assert_eq!(
        alias.name_range(),
        None,
        "the matching module prefix is not the alias token"
    );
    assert_eq!(
        &imported[alias.range().start() as usize..alias.range().end() as usize],
        "import editor.workspace.library as editor;"
    );
}
