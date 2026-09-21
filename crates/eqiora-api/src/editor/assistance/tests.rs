use super::*;
use crate::editor::EditorService;
use eqiora_compiler::{
    CompilationNamespaceId as Namespace, ResolvedDependency, ResolvedHierarchyInput,
    ResolvedSourceUnit,
};

fn complete(marked: &str) -> EditorCompletion {
    let offset = marked.find('|').unwrap() as u32;
    EditorService::new("test", 1, marked.replacen('|', "", 1))
        .current()
        .completion(offset)
        .unwrap()
}

fn names(completion: &EditorCompletion) -> Vec<&str> {
    completion.items.iter().map(|c| c.name.as_str()).collect()
}

#[test]
fn incomplete_owners_recover_current_scope_without_leaking_closed_siblings() {
    let c = complete(
        "model Other() { parameter secret: 1 = 1; } model M(parameter gain: 1) { relation r { ga|",
    );
    assert_eq!(names(&c), ["gain"]);
    let c = complete("model Other() { parameter secret: 1 = 1; } |");
    assert!(!names(&c).contains(&"secret"));
    for (declaration, prefix, expected) in [
        ("state position: m;", "pos", "position"),
        ("parameter gain: 1 = 2;", "ga", "gain"),
        ("instance child: C();", "chi", "child"),
    ] {
        let c = complete(&format!(
            "component C() {{}} model M() {{ {declaration} relation r {{ {prefix}|"
        ));
        assert_eq!(names(&c), [expected]);
    }
    assert!(
        complete("model M() { // parameter fake: 1;\n relation r { fa| ")
            .items
            .is_empty()
    );
}

#[test]
fn context_separates_declarations_types_and_values_and_respects_shadowing() {
    assert_eq!(
        complete("mod|").context,
        EditorCompletionContext::Declaration
    );
    let c = complete("dimension Length = m; model M() { parameter value: Len| }");
    assert_eq!(c.context, EditorCompletionContext::Type);
    assert_eq!(names(&c), ["Length"]);
    let c =
        complete("operator gain(input x: 1): 1 = x; model M(parameter gain: 1) { relation r { ga|");
    assert_eq!(c.items[0].kind, EditorSymbolKind::Parameter);
    assert!(c.items[0].parameters.is_none());
    let c = complete("record Sample { value: 1 } model M() { variable x: Sam|");
    assert_eq!(names(&c), ["Sample"]);
    assert_eq!(c.items[0].kind, EditorSymbolKind::Record);
}

#[test]
fn same_named_declarations_follow_aliases_and_ambiguous_exact_packages_are_omitted() {
    let owner = Namespace::new(["app", "root"]).unwrap();
    let left = Namespace::new(["left", "exact-a"]).unwrap();
    let right = Namespace::new(["right", "exact-b"]).unwrap();
    let source = "import left.main as a; import right.main as b; model M() { instance c: b.Part(";
    let root = ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", source).unwrap();
    let file = root.diagnostic_file();
    let units = vec![
        root,
        ResolvedSourceUnit::new(
            left.clone(),
            "src/main.eqi",
            "public component Part(parameter first: 1) {}",
        )
        .unwrap(),
        ResolvedSourceUnit::new(
            right.clone(),
            "src/main.eqi",
            "public component Part(parameter second: 1) {}",
        )
        .unwrap(),
    ];
    let edges = vec![
        ResolvedDependency::new(owner.clone(), left),
        ResolvedDependency::new(owner.clone(), right),
    ];
    let snapshot = EditorWorkspaceSnapshot::analyze_modules(
        1,
        ResolvedHierarchyInput::new(owner.clone(), units, edges),
    );
    assert_eq!(
        names(&snapshot.completion(&file, source.len() as u32).unwrap()),
        ["second"]
    );

    let one = Namespace::new(["library", "exact-one"]).unwrap();
    let two = Namespace::new(["library", "exact-two"]).unwrap();
    let root = ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", "import lib").unwrap();
    let file = root.diagnostic_file();
    let snapshot = EditorWorkspaceSnapshot::analyze_modules(
        2,
        ResolvedHierarchyInput::new(
            owner.clone(),
            vec![
                root,
                ResolvedSourceUnit::new(one.clone(), "src/main.eqi", "public component A() {}")
                    .unwrap(),
                ResolvedSourceUnit::new(two.clone(), "src/main.eqi", "public component B() {}")
                    .unwrap(),
            ],
            vec![
                ResolvedDependency::new(owner.clone(), one),
                ResolvedDependency::new(owner, two),
            ],
        ),
    );
    assert!(snapshot.completion(&file, 10).unwrap().items.is_empty());
}

const COMPONENT: &str = "public connector Pin { across voltage: V; through current: A; }\n/// Adjustable part.\npublic component Part(\n/// Required coefficient.\nparameter gain: 1, parameter offset: 1 = 0, output result: 1, port pin: Pin) { parameter secret: 1 = 2; }\nprivate component Hidden() {}";

fn graph(marked: &str) -> (EditorWorkspaceSnapshot, String, u32) {
    let owner = Namespace::new(["app", "exact-root"]).unwrap();
    let dep = Namespace::new(["library", "exact-dependency"]).unwrap();
    let indirect = Namespace::new(["indirect", "not-direct"]).unwrap();
    let file = ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", marked.replacen('|', "", 1))
        .unwrap();
    let label = file.diagnostic_file();
    let input = ResolvedHierarchyInput::new(
        owner.clone(),
        vec![
            file,
            ResolvedSourceUnit::new(dep.clone(), "src/devices/electrical.eqi", COMPONENT).unwrap(),
            ResolvedSourceUnit::new(
                indirect.clone(),
                "src/internal.eqi",
                "public component No() {}",
            )
            .unwrap(),
        ],
        vec![
            ResolvedDependency::new(owner, dep.clone()),
            ResolvedDependency::new(dep, indirect),
        ],
    );
    (
        EditorWorkspaceSnapshot::analyze_modules(1, input),
        label,
        marked.find('|').unwrap() as u32,
    )
}

#[test]
fn incomplete_imports_use_only_current_graph_and_direct_dependencies() {
    for (source, expected) in [
        ("import lib|", "library"),
        ("import library.|", "library.devices"),
        ("import library.devices.ele|", "library.devices.electrical"),
    ] {
        let (workspace, file, offset) = graph(source);
        assert_eq!(
            names(&workspace.completion(&file, offset).unwrap()),
            [expected]
        );
        assert!(workspace.compile_model(&file, "M").is_err());
    }
    let (workspace, file, offset) = graph("import ind|");
    assert!(
        workspace
            .completion(&file, offset)
            .unwrap()
            .items
            .is_empty()
    );
}

#[test]
fn qualified_exports_and_exposed_members_recover_on_invalid_source() {
    let import = "import library.devices.electrical as electrical; ";
    let (w, f, p) = graph(&format!("{import} model M() {{ instance c: electrical.|"));
    let c = w.completion(&f, p).unwrap();
    assert!(names(&c).contains(&"electrical.Part"));
    assert!(!names(&c).contains(&"electrical.Hidden"));
    let (w, f, p) = graph(&format!(
        "{import} model M() {{ instance child: electrical.Part(gain = 2); relation r {{ child.|"
    ));
    assert_eq!(
        names(&w.completion(&f, p).unwrap()),
        ["child.pin", "child.result"]
    );
    let (w, f, p) = graph(&format!(
        "{import} model M() {{ instance child: electrical.Part(gain = 2); relation r {{ child.pin.|"
    ));
    assert_eq!(
        names(&w.completion(&f, p).unwrap()),
        ["child.pin.current", "child.pin.voltage"]
    );
}

#[test]
fn remaining_bindings_include_required_defaulted_docs_and_avoid_duplicate_equals() {
    for (suffix, expected) in [("ga|", "gain = "), ("ga|in = 2", "gain")] {
        let c = complete(&format!(
            "{COMPONENT} model M() {{ instance c: Part({suffix}"
        ));
        assert_eq!(names(&c), ["gain"]);
        assert_eq!(c.items[0].insert_text, expected);
        assert_eq!(c.items[0].required, Some(true));
        assert!(
            c.items[0]
                .documentation
                .as_ref()
                .unwrap()
                .markdown()
                .contains("Required coefficient")
        );
    }
    let c = complete(&format!(
        "{COMPONENT} model M() {{ instance c: Part(gain = math.max(1, 2), |"
    ));
    assert_eq!(names(&c), ["offset"]);
    assert_eq!(c.items[0].required, Some(false));
    let c = complete(&format!(
        "{COMPONENT} model M() {{ instance c: Part(|, offset = 3)"
    ));
    assert_eq!(names(&c), ["gain"]);
    let c = complete(&format!(
        "{COMPONENT} model M(parameter value: 1) {{ instance c: Part(gain = va|"
    ));
    assert_eq!(names(&c), ["value"]);
    let (w, f, p) =
        graph("import library.devices.electrical as e; model M() { instance c: e.Part(off|");
    assert_eq!(names(&w.completion(&f, p).unwrap()), ["offset"]);
    assert!(w.assistance(&f, p, "e.Part").unwrap().parameters.is_some());

    let operator = "operator blend(input left: 1, input right: 1): 1 = left + right;";
    assert_eq!(
        names(&complete(&format!(
            "{operator} model M() {{ relation r {{ blend(left = 2, ri|"
        ))),
        ["right"]
    );
    assert!(
        !names(&complete(&format!(
            "{operator} model M() {{ relation r {{ blend(2, ri|"
        )))
        .contains(&"right"),
        "positional calls cannot mix named bindings"
    );
    assert!(
        complete("model M() { instance c: Unknown(ga|")
            .items
            .is_empty()
    );
}
