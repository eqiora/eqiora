use super::*;
use crate::editor::EditorService;
use eqiora_compiler::{
    CompilationNamespaceId as Namespace, ResolvedDependency, ResolvedHierarchyInput,
    ResolvedSourceUnit,
};

fn complete(marked: &str) -> Vec<EditorSymbol> {
    let offset = marked.find('|').unwrap() as u32;
    EditorService::new("test", 1, marked.replacen('|', "", 1))
        .current()
        .completion(offset)
        .unwrap()
        .1
}

fn names(completion: &[EditorSymbol]) -> Vec<&str> {
    completion.iter().map(|c| c.name.as_str()).collect()
}

fn semantic_complete(marked: &str) -> Vec<EditorSymbol> {
    let offset = marked.find('|').unwrap() as u32;
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, marked.replacen('|', "", 1));
    let file = workspace.files().next().unwrap();
    workspace.completion(file, offset).unwrap().1
}

#[test]
fn semantic_completion_prioritizes_exact_parameter_dimensions() {
    let c = semantic_complete(
        "dimension Length = m; model M() { parameter value_bad: s = 1[s]; parameter value_good: Length = 2[m]; parameter result: m = value_|; }",
    );
    assert_eq!(names(&c), ["value_good", "value_bad"]);
    assert!(c[0].detail().unwrap().contains("expected"));
    let c = semantic_complete(
        "component C(parameter length: m) {} model M() { parameter value_bad: s = 1[s]; parameter value_good: m = 2[m]; instance child: C(length = value_|); }",
    );
    assert_eq!(names(&c), ["value_good", "value_bad"]);
}

#[test]
fn semantic_completion_uses_imported_formal_aliases_and_nominal_types() {
    for (library, declarations, formal) in [
        (
            "public dimension Length = m; public component Receiver(parameter value: Length) {}",
            "parameter value_bad:s=1[s]; parameter value_good:m=2[m];",
            "dimension L",
        ),
        (
            "public enum Mode {First,Second} public enum Other {First,Second} public component Receiver(parameter value: Mode) {}",
            "parameter value_bad:lib.Other=lib.Other.First; parameter value_good:lib.Mode=lib.Mode.First;",
            "enum",
        ),
    ] {
        let owner = Namespace::new(["app", "exact-root"]).unwrap();
        let dep = Namespace::new(["library", "exact-dependency"]).unwrap();
        let marked = format!(
            "import library.types as lib; model M() {{ {declarations} instance child:lib.Receiver(value=value_|good); }}"
        );
        let root =
            ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", marked.replacen('|', "", 1))
                .unwrap();
        let file = root.diagnostic_file();
        let workspace = EditorWorkspaceSnapshot::analyze_modules(
            1,
            ResolvedHierarchyInput::new(
                owner.clone(),
                vec![
                    root,
                    ResolvedSourceUnit::new(dep.clone(), "src/types.eqi", library).unwrap(),
                ],
                vec![ResolvedDependency::new(owner, dep)],
            ),
        );
        let c = workspace
            .completion(&file, marked.find('|').unwrap() as u32)
            .unwrap()
            .1;
        assert_eq!(names(&c), ["value_good", "value_bad"]);
        assert!(c[0].detail().unwrap().contains(formal));
        assert!(c[1].detail().unwrap().contains("expected"));
    }
}

#[test]
fn semantic_completion_preserves_unknown_incomplete_and_expression_contexts() {
    for ending in ["value_|", "2 * value_|; }"] {
        let c = semantic_complete(&format!(
            "model M() {{ parameter value_bad: s = 1[s]; parameter value_good: m = 2[m]; parameter result: m = {ending}"
        ));
        assert_eq!(names(&c), ["value_bad", "value_good"]);
        assert!(
            c.iter()
                .all(|item| !item.detail().unwrap().contains("expected"))
        );
    }
}

#[test]
fn semantic_connection_completion_checks_direction_dimension_and_nominal_identity() {
    let c = semantic_complete(
        "model M() { port source: signal output m; port target_bad_dimension: signal input s; port target_bad_role: signal output m; port target_good: signal input m; connect source -> target_|; }",
    );
    assert_eq!(
        names(&c),
        ["target_good", "target_bad_dimension", "target_bad_role"]
    );
    let c = semantic_complete(
        "model M() { domain first = scalar_physical(across x: m, through y: s); domain second = scalar_physical(across x: m, through y: s); port source: first; port target_bad: second; port target_good: first; connect source, target_|; }",
    );
    assert_eq!(names(&c), ["target_good", "target_bad"]);
    assert!(c[1].detail().unwrap().contains("nominal"));
    let c = semantic_complete(
        "model M() { port target_bad:signal input m; port target_good:signal output m; port sink:signal input m; connect target_| -> sink; }",
    );
    assert_eq!(names(&c), ["target_good", "target_bad"]);
}

#[test]
fn incomplete_owners_recover_current_scope_without_leaking_closed_siblings() {
    let c = complete(
        "model Other() { parameter secret: 1 = 1; } model M(parameter gain: 1) { relation r { ga|",
    );
    assert_eq!(names(&c), ["gain"]);
    assert_eq!(
        names(&complete(
            "component C(parameter gain: 1) { form f for r { ga|"
        )),
        ["gain"]
    );
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
    assert_eq!(
        names(&complete(
            "model M() { // parameter fake: 1;\n relation r { fa| "
        )),
        ["false"],
        "the commented declaration is absent; the Boolean keyword remains"
    );
}

#[test]
fn context_separates_declarations_types_and_values_and_respects_shadowing() {
    assert_eq!(names(&complete("mod|")), ["model"]);
    let c = complete("dimension Length = m; model M() { parameter value: Len| }");
    assert_eq!(names(&c), ["Length"]);
    let c =
        complete("operator gain(input x: 1): 1 = x; model M(parameter gain: 1) { relation r { ga|");
    assert_eq!(c[0].kind, EditorSymbolKind::Parameter);
    assert!(c[0].parameters().is_none());
    let c = complete("record Sample { value: 1 } model M() { variable x: Sam|");
    assert_eq!(names(&c), ["Sample"]);
    assert_eq!(c[0].kind, EditorSymbolKind::Record);
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
        names(&snapshot.completion(&file, source.len() as u32).unwrap().1),
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
    assert!(snapshot.completion(&file, 10).unwrap().1.is_empty());
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
            names(&workspace.completion(&file, offset).unwrap().1),
            [expected]
        );
        assert!(workspace.compile_model(&file, "M").is_err());
    }
    let (workspace, file, offset) = graph("import ind|");
    assert!(workspace.completion(&file, offset).unwrap().1.is_empty());
}

#[test]
fn qualified_exports_and_exposed_members_recover_on_invalid_source() {
    assert!(
        complete(&format!("{COMPONENT} model M() {{ relation r {{ Part.|")).is_empty(),
        "a definition name is not an instance"
    );
    assert!(
        complete("operator f(input x: 1): 1 = x; model M() { relation r { f.|").is_empty(),
        "operator formals are private to the operator body"
    );
    let import = "import library.devices.electrical as electrical; ";
    let (w, f, p) = graph(&format!("{import} model M() {{ instance c: electrical.|"));
    let c = w.completion(&f, p).unwrap().1;
    assert!(names(&c).contains(&"electrical.Part"));
    assert!(!names(&c).contains(&"electrical.Hidden"));
    let (w, f, p) = graph(&format!(
        "{import} model M() {{ instance child: electrical.Part(gain = 2); relation r {{ child.|"
    ));
    assert_eq!(
        names(&w.completion(&f, p).unwrap().1),
        ["child.pin", "child.result"]
    );
    let (w, f, p) = graph(&format!(
        "{import} model M() {{ instance child: electrical.Part(gain = 2); relation r {{ child.pin.|"
    ));
    assert_eq!(
        names(&w.completion(&f, p).unwrap().1),
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
        assert_eq!(c[0].insert_text(), expected);
        assert_eq!(c[0].required(), Some(true));
        assert!(
            c[0].documentation()
                .unwrap()
                .contains("Required coefficient")
        );
    }
    let c = complete(&format!(
        "{COMPONENT} model M() {{ instance c: Part(gain = math.max(1, 2), |"
    ));
    assert_eq!(names(&c), ["offset"]);
    assert_eq!(c[0].required(), Some(false));
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
    assert_eq!(names(&w.completion(&f, p).unwrap().1), ["offset"]);
    assert!(
        w.assistance(&f, p, "e.Part")
            .unwrap()
            .parameters()
            .is_some()
    );

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
    assert!(complete("model M() { instance c: Unknown(ga|").is_empty());
}

#[test]
fn semantic_completion_keeps_distinct_nominal_values_and_connectors_distinct() {
    let c = semantic_complete(
        "enum Mode {First,Second} enum Other {First,Second} model M() { parameter value_bad:Other=Other.First; parameter value_good:Mode=Mode.First; parameter result:Mode=value_|good; }",
    );
    assert_eq!(names(&c), ["value_good", "value_bad"]);
    let c = semantic_complete(
        "connector A {across potential:V; through flow:A;} connector B {across potential:V; through flow:A;} component Ports(port source:A,port target_bad:B,port target_good:A) {} model M() {instance child:Ports();connect child.source,child.target_|good;}",
    );
    assert_eq!(names(&c), ["child.target_good", "child.target_bad"]);
}

#[test]
fn semantic_completion_leaves_clock_and_support_identity_unknown() {
    for declarations in [
        "clock tick=periodic(1[s]); port target_unknown:signal input m at tick;",
        "domain body=box(0,1); port target_unknown:signal input m on body;",
    ] {
        let c = semantic_complete(&format!(
            "model M() {{port source:signal output m; port target_bad:signal input s; port target_good:signal input m; {declarations} connect source -> target_|;}}"
        ));
        assert_eq!(names(&c), ["target_good", "target_unknown", "target_bad"]);
        assert!(!c[1].detail().unwrap().contains("compatible"));
    }
}

#[test]
fn semantic_completion_checks_shape_and_rebuilds_current_version() {
    let marked = "model M() {parameter value_bad:array<m,3>=[1[m],2[m],3[m]]; parameter value_good:array<m,2>=[1[m],2[m]]; parameter result:array<m,2>=value_|; }";
    assert_eq!(
        names(&semantic_complete(marked)),
        ["value_good", "value_bad"]
    );
    let changed = marked.replace("result:array<m,2>", "result:array<m,3>");
    assert_eq!(
        names(&semantic_complete(&changed)),
        ["value_bad", "value_good"]
    );
}
