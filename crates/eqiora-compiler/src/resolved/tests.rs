mod additional_tests;
use std::cell::Cell;

use eqiora_graph::{GraphStore, InMemoryGraphStore};

use super::*;

fn namespace(name: &str) -> CompilationNamespaceId {
    CompilationNamespaceId::new([name, "1.0.0", "semantic-digest"]).expect("namespace")
}

fn unit(namespace: &CompilationNamespaceId, _file: &str, source: &str) -> ResolvedSourceUnit {
    ResolvedSourceUnit::new(namespace.clone(), "src/main.eqi", source).expect("main source path")
}

fn module_unit(
    namespace: &CompilationNamespaceId,
    module_name: &str,
    _file: &str,
    source: &str,
) -> ResolvedSourceUnit {
    let file = format!("src/{}.eqi", module_name.replace('.', "/"));
    ResolvedSourceUnit::new(namespace.clone(), file, source).expect("module source path")
}

fn dependency_edge(
    declaring: &CompilationNamespaceId,
    target: &CompilationNamespaceId,
) -> ResolvedDependency {
    ResolvedDependency::new(declaring.clone(), target.clone())
}

const LIBRARY: &str = r#"
public component Resistor(parameter resistance: 1) {

  relation law { resistance - 2 = 0; }
}
"#;

#[test]
fn resolved_analysis_cancellation_publishes_no_partial_result() {
    let owner = namespace("org.example.cancelled");
    let immediately = ResolvedHierarchyInput::new(
        owner.clone(),
        vec![unit(&owner, "src/main.eqi", "model Main() {}")],
        vec![],
    )
    .analyze_with_cancellation(|| true)
    .expect("cancellation is not a diagnostic");
    assert!(immediately.is_none());

    let polls = Cell::new(0_u8);
    let between_sources = ResolvedHierarchyInput::new(
        owner.clone(),
        vec![
            unit(&owner, "src/main.eqi", "model Main() {}"),
            module_unit(&owner, "broken", "src/broken.eqi", "not valid source"),
        ],
        vec![],
    )
    .analyze_with_cancellation(|| {
        let next = polls.get() + 1;
        polls.set(next);
        next == 3
    })
    .expect("cancellation suppresses incomplete diagnostics");
    assert!(between_sources.is_none());
    assert_eq!(polls.get(), 3);
}

#[test]
fn module_labels_use_only_path_assigned_identity() {
    let owner = namespace("org.example.project");
    assert_eq!(
        CompilationModuleId::main(owner.clone()).to_string(),
        format!("{owner}::main")
    );
    assert_eq!(
        CompilationModuleId::new(
            owner.clone(),
            ModuleName::new(["library", "primitives"]).expect("module name"),
        )
        .to_string(),
        format!("{owner}::library.primitives")
    );
    let assigned = ResolvedSourceUnit::new(
        owner.clone(),
        "src/library/primitives.eqi",
        "public component Part() {}",
    )
    .expect("path-assigned module");
    assert!(
        assigned
            .diagnostic_file()
            .contains("module:2:7:library:10:primitives:src/library/primitives.eqi")
    );
    let recovering = ResolvedSourceUnit::new(
        owner,
        "src/library/primitives.eqi",
        "public component Part() {",
    )
    .expect("path-assigned broken module");
    assert!(recovering.diagnostic_file().contains("module:"));
}

#[test]
fn hierarchy_footprint_fails_before_source_input_allocation() {
    let limits = ResolvedHierarchyResourceLimits {
        source_units: 2,
        aliases: 1,
        source_unit_bytes: 8,
        total_source_bytes: 10,
    };
    assert!(
        preflight_resolved_hierarchy_with_limits([1], 2, limits)
            .expect_err("alias overflow")
            .message()
            .contains("module-link limit")
    );
    assert!(
        preflight_resolved_hierarchy_with_limits([1, 1, 1], 0, limits)
            .expect_err("source-unit overflow")
            .message()
            .contains("source-unit limit")
    );
    assert!(
        preflight_resolved_hierarchy_with_limits([9], 0, limits)
            .expect_err("per-source overflow")
            .message()
            .contains("byte hierarchy limit")
    );
    assert!(
        preflight_resolved_hierarchy_with_limits([6, 5], 0, limits)
            .expect_err("aggregate overflow")
            .message()
            .contains("total source-byte limit")
    );
    preflight_resolved_hierarchy_with_limits([4, 6], 1, limits).expect("exact footprint limit");
}

#[test]
fn parser_diagnostics_are_independent_of_source_unit_input_order() {
    let root = namespace("root");
    let units = vec![
        unit(&root, "z.eqi", "model Z() { relation broken"),
        unit(&root, "a.eqi", "model A() { parameter p:"),
    ];
    let forward = analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
        root.clone(),
        units.clone(),
        vec![],
    ))
    .expect_err("both source units are invalid");
    let reverse = analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
        root,
        units.into_iter().rev().collect(),
        vec![],
    ))
    .expect_err("input permutation remains invalid");

    assert_eq!(forward, reverse);
}

#[test]
fn exact_direct_alias_elaborates_with_cross_file_provenance() {
    let root = namespace("org.example.root");
    let electrical = namespace("org.eqiora.electrical");
    let input = ResolvedHierarchyInput::new(
        root.clone(),
        vec![
            unit(
                &root,
                "root/main.eqi",
                "import org.eqiora.electrical.main as electrical; model Main() { instance load: electrical.Resistor(resistance = 2); }",
            ),
            unit(&electrical, "electrical/resistor.eqi", LIBRARY),
        ],
        vec![dependency_edge(&root, &electrical)],
    );

    let analysis = analyze_resolved_hierarchy(input).expect("resolved graph analyzes");
    assert_eq!(analysis.canonical_declarations().len(), 2);
    let compiled = analysis
        .validate_definitions()
        .expect("definitions validate")
        .compile_root("Main")
        .expect("root elaborates");
    assert!(
        compiled.symbols().get("load.resistance").is_none(),
        "literal component arguments do not fabricate Kernel Parameters"
    );
    let law = compiled
        .symbols()
        .get("load.law")
        .expect("imported relation symbol");
    let provenance = compiled.provenance().expect("hierarchy provenance");
    let source = provenance
        .get_by_graph_id(law)
        .expect("relation provenance");
    assert!(source.definition_span().file.ends_with("src/main.eqi"));
    assert!(source.instance_span().file.ends_with("src/main.eqi"));
    assert!(source.binding_spans()[0].file.ends_with("src/main.eqi"));
    assert_ne!(
        source.definition_span().file,
        source.instance_span().file,
        "package-qualified source labels remain unambiguous"
    );

    let (transaction, _, _) = compiled.into_parts();
    InMemoryGraphStore::new()
        .commit(transaction)
        .expect("complete transaction commits");
}

#[test]
fn explicit_local_module_import_elaborates_component_and_operator() {
    let owner = namespace("org.example.project");
    let root_source = r#"
import org.example.project.library.primitives as lib;
model Main() {
  domain d = box(0, 1, 0, 1);
  variable a: vector<1, 2> on d;
  variable b: vector<1, 2> on d;
  instance load: lib.Resistor(resistance = 2);
  relation doubled on d { div(div(lib.outer(left=a, right=b))) = 0; }
}
"#;
    let library_source = r#"
public operator outer(input left: spatial[1], input right: spatial[1]): spatial[2]
  = component(left, 0) * component(right, 1);
public component Resistor(parameter resistance: 1) {

  relation law { resistance - 2 = 0; }
}
"#;
    let input = ResolvedHierarchyInput::with_root_module(
        owner.clone(),
        "models.main".split('.'),
        vec![
            ResolvedSourceUnit::new(owner.clone(), "src/library/primitives.eqi", library_source)
                .expect("library module path"),
            ResolvedSourceUnit::new(owner.clone(), "src/models/main.eqi", root_source)
                .expect("root module path"),
        ],
        vec![],
    )
    .expect("root module identity");

    let analysis = analyze_resolved_hierarchy(input).expect("local modules analyze");
    let compiled = analysis
        .validate_definitions()
        .expect("local module definitions validate")
        .compile_root("Main")
        .expect("qualified component and operator elaborate");
    assert!(compiled.symbols().get("load.law").is_some());
    assert!(compiled.symbols().get("doubled").is_some());
}

#[test]
fn directly_imported_public_model_is_an_executable_entry() {
    let owner = namespace("org.example.project");
    let input = ResolvedHierarchyInput::with_root_module(
        owner.clone(),
        ["models", "main"],
        vec![
            module_unit(
                &owner,
                "library.entries",
                "src/library/entries.eqi",
                "public model Shared() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }",
            ),
            module_unit(
                &owner,
                "models.main",
                "src/models/main.eqi",
                "import org.example.project.library.entries as lib; model Local() {}",
            ),
        ],
        vec![],
    )
    .expect("root module identity");

    let analysis = analyze_resolved_hierarchy(input).expect("public entry graph analyzes");
    let exported = analysis
        .canonical_declarations()
        .iter()
        .find(|declaration| declaration.path() == "library.entries.Shared")
        .expect("public Model is canonicalized");
    assert_eq!(exported.kind(), CanonicalDeclarationKind::Model);
    assert_eq!(exported.visibility(), VisibilitySyntax::Public);
    let compiled = analysis
        .validate_definitions()
        .expect("public entry graph validates")
        .compile_root("lib.Shared")
        .expect("direct imported public Model compiles");
    assert!(compiled.symbols().get("law").is_some());
}

#[test]
fn imported_entry_model_rejects_private_and_non_direct_targets() {
    let owner = namespace("org.example.project");
    let input = ResolvedHierarchyInput::with_root_module(
        owner.clone(),
        ["models", "main"],
        vec![
            module_unit(
                &owner,
                "library.entries",
                "src/library/entries.eqi",
                "model Hidden() {}",
            ),
            module_unit(
                &owner,
                "models.main",
                "src/models/main.eqi",
                "import org.example.project.library.entries as lib; model Local() {}",
            ),
        ],
        vec![],
    )
    .expect("root module identity");
    let validated = analyze_resolved_hierarchy(input)
        .expect("private entry graph analyzes")
        .validate_definitions()
        .expect("private entry graph validates");

    let private = validated
        .compile_root("lib.Hidden")
        .expect_err("private imported Model");
    assert!(
        private
            .iter()
            .any(|diagnostic| diagnostic.message().contains("private Model"))
    );
    let transitive = validated
        .compile_root("lib.nested.Hidden")
        .expect_err("transitive entry path");
    assert!(transitive.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("one direct alias-qualified name")
    }));
}

#[test]
fn one_logical_module_rejects_multiple_source_units() {
    let owner = namespace("org.example.project");
    let input = ResolvedHierarchyInput::with_root_module(
        owner.clone(),
        ["models"],
        vec![
            module_unit(&owner, "models", "src/models/components.eqi", LIBRARY),
            module_unit(
                &owner,
                "models",
                "src/models/main.eqi",
                "model Main() { instance load: Resistor(resistance = 2); }",
            ),
        ],
        vec![],
    )
    .expect("root module identity");

    let diagnostics = analyze_resolved_hierarchy(input).expect_err("duplicate module path");
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("canonical module identity"))
    );
}

#[test]
fn host_assigned_module_identity_requires_no_source_header() {
    let owner = namespace("org.example.project");
    let analysis = analyze_resolved_hierarchy(
        ResolvedHierarchyInput::with_root_module(
            owner.clone(),
            ["models", "main"],
            vec![
                module_unit(
                    &owner,
                    "models.main",
                    "src/models/main.eqi",
                    "import org.example.project.library.parts as lib; model Main() { instance load: lib.Resistor(resistance = 2); }",
                ),
                module_unit(
                    &owner,
                    "library.parts",
                    "src/library/parts.eqi",
                    "public component Resistor(parameter resistance: 1) {  relation law { resistance - 2 = 0; } }",
                ),
            ],
            vec![],
        )
        .expect("root module identity"),
    )
    .expect("host-assigned graph analyzes");

    let compiled = analysis
        .validate_definitions()
        .expect("module definitions validate")
        .compile_root("Main")
        .expect("host-owned module compiles");
    assert!(compiled.symbols().get("load.law").is_some());
}

#[test]
fn removed_source_module_declaration_is_rejected() {
    let owner = namespace("org.example.project");
    let source = "module declared.name; public component Value() {}";
    let input = ResolvedHierarchyInput::with_root_module(
        owner.clone(),
        ["assigned", "name"],
        vec![module_unit(&owner, "assigned.name", "value.eqi", source)],
        vec![],
    )
    .expect("host module identity");

    let diagnostics = analyze_resolved_hierarchy(input).expect_err("removed grammar");
    let diagnostic = diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic
                .message()
                .contains("expected `import`, `dimension`")
        })
        .expect("removed module diagnostic");
    let span = diagnostic.source_span().expect("module source span");
    assert_eq!(&source[span.start as usize..span.end as usize], "module");
}

#[test]
fn oversized_provenance_path_fails_before_malformed_source_is_parsed() {
    let owner = namespace("org.example.project");
    let file = format!("src/{}/main.eqi", vec!["x".repeat(250); 20].join("/"));
    let input = ResolvedHierarchyInput::new(
        owner.clone(),
        vec![
            ResolvedSourceUnit::new(owner, file, "not valid Eqiora")
                .expect("bounded module segments"),
        ],
        vec![],
    );

    let diagnostics = analyze_resolved_hierarchy(input).expect_err("oversized provenance path");
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("provenance-path limit")),
        "the bounded path failure must precede parser diagnostics"
    );
    assert!(
        diagnostics
            .iter()
            .all(|diagnostic| diagnostic.source_span().is_none()),
        "the malformed source must not reach the parser"
    );
}

#[test]
fn source_local_alias_identity_cannot_collide_with_an_external_namespace() {
    let root = namespace("root");
    let local = analyze_resolved_hierarchy(
        ResolvedHierarchyInput::with_root_module(
            root.clone(),
            ["main"],
            vec![
                module_unit(
                    &root,
                    "main",
                    "root.eqi",
                    "import root.parts as value; model Main() { instance x: value.Resistor(); }",
                ),
                module_unit(
                    &root,
                    "parts",
                    "parts.eqi",
                    "public component Resistor() {}",
                ),
            ],
            vec![],
        )
        .expect("root module"),
    )
    .expect("local module graph");
    let external_target = CompilationNamespaceId::new(["local_module_v1", "parts"])
        .expect("adversarial external namespace");
    let external = analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
        root.clone(),
        vec![
            unit(
                &root,
                "root.eqi",
                "import local_module_v1.main as value; model Main() { instance x: value.Resistor(); }",
            ),
            unit(
                &external_target,
                "parts.eqi",
                "public component Resistor() {}",
            ),
        ],
        vec![dependency_edge(&root, &external_target)],
    ))
    .expect("external package graph");
    let root_form = |analysis: &AnalyzedResolvedHierarchy| {
        analysis
            .canonical_declarations()
            .iter()
            .find(|declaration| {
                declaration.namespace() == &root && declaration.path() == "main.Main"
            })
            .expect("root model declaration")
            .canonical_form()
            .to_owned()
    };
    assert_ne!(root_form(&local), root_form(&external));
}

#[test]
fn explicit_local_import_graph_rejects_missing_duplicate_reserved_and_cycle() {
    let owner = namespace("org.example.project");
    let analyze = |root_name: &str, sources: &[(&str, &str)]| {
        analyze_resolved_hierarchy(
            ResolvedHierarchyInput::with_root_module(
                owner.clone(),
                root_name.split('.'),
                sources
                    .iter()
                    .map(|(name, source)| {
                        ResolvedSourceUnit::new(
                            owner.clone(),
                            format!("src/{}.eqi", name.replace('.', "/")),
                            *source,
                        )
                        .expect("module source path")
                    })
                    .collect(),
                vec![],
            )
            .expect("root module identity"),
        )
    };

    let missing = analyze(
        "models.main",
        &[(
            "models.main",
            "import org.example.project.library.missing as lib; model Main() {}",
        )],
    )
    .expect_err("missing local module");
    assert!(missing.iter().any(|diagnostic| {
        diagnostic.message().contains("unknown canonical module")
            && diagnostic.source_span().is_some()
    }));

    let missing_export = analyze(
        "models.main",
        &[
            (
                "models.main",
                "import org.example.project.library.one as lib; model Main() { instance value: lib.Missing(); }",
            ),
            ("library.one", "public component One() {}"),
        ],
    )
    .expect_err("missing exported declaration");
    assert!(missing_export.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("unresolved component `lib.Missing`")
            && diagnostic.source_span().is_some()
    }));

    let duplicate = analyze(
        "models.main",
        &[
            (
                "models.main",
                "import org.example.project.library.one as lib; import org.example.project.library.two as lib; model Main() {}",
            ),
            ("library.one", "public component One() {}"),
            ("library.two", "public component Two() {}"),
        ],
    )
    .expect_err("duplicate alias");
    assert!(duplicate.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("duplicate direct alias `lib`")
    }));

    let reserved = analyze(
        "models.main",
        &[
            (
                "models.main",
                "import org.example.project.library.one as math; model Main() {}",
            ),
            ("library.one", "public component One() {}"),
        ],
    )
    .expect_err("reserved root");
    assert!(
        reserved
            .iter()
            .any(|diagnostic| diagnostic.message().contains("alias `math` is reserved"))
    );

    let cycle = analyze(
        "modules.a",
        &[
            (
                "modules.a",
                "import org.example.project.modules.b as b; model Main() {}",
            ),
            (
                "modules.b",
                "import org.example.project.modules.a as a; public component B() {}",
            ),
        ],
    )
    .expect_err("import cycle");
    assert!(cycle.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("semantic module import cycle")
            && diagnostic.source_span().is_some()
    }));
}

#[test]
fn canonical_declarations_normalize_aliases_to_exact_targets() {
    let root = namespace("root");
    let target = namespace("target");
    let renamed = |alias_name: &str| {
        ResolvedHierarchyInput::new(
            root.clone(),
            vec![
                unit(
                    &root,
                    "root.eqi",
                    &format!(
                        "import target.main as {alias_name}; model Main() {{ instance c: {alias_name}.Resistor(); }}"
                    ),
                ),
                unit(&target, "target.eqi", LIBRARY),
            ],
            vec![dependency_edge(&root, &target)],
        )
    };
    let first = analyze_resolved_hierarchy(renamed("electrical")).expect("first alias");
    let second = analyze_resolved_hierarchy(renamed("components")).expect("renamed alias");
    assert_eq!(
        first.canonical_declarations(),
        second.canonical_declarations(),
        "import aliases are not package semantics"
    );

    let other_target = namespace("other_target");
    let changed = analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
        root.clone(),
        vec![
            unit(
                &root,
                "root.eqi",
                "import other_target.main as electrical; model Main() { instance c: electrical.Resistor(); }",
            ),
            unit(&other_target, "target.eqi", LIBRARY),
        ],
        vec![dependency_edge(&root, &other_target)],
    ))
    .expect("changed exact target");
    let root_form = |analysis: &AnalyzedResolvedHierarchy| {
        analysis
            .canonical_declarations()
            .iter()
            .find(|declaration| {
                declaration.namespace() == &root && declaration.path() == "main.Main"
            })
            .expect("root declaration")
            .canonical_form()
            .to_owned()
    };
    assert_ne!(root_form(&first), root_form(&changed));
}

#[test]
fn pure_operator_declarations_and_calls_are_file_and_alias_invariant() {
    let root = namespace("root");
    let operators = namespace("operators");
    let dependency = r#"
public operator outer(input left: spatial[1], input right: spatial[1]): spatial[2]
  = component(left, 0) * component(right, 1);
"#;
    let analyzed = |alias_name: &str, operator_file: &str| {
        analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
            root.clone(),
            vec![
                unit(
                    &root,
                    "root.eqi",
                    &format!(
                        "import operators.main as {alias_name}; model Main() {{ domain d = box(0,1,0,1); variable a: vector<1, 2> on d; variable b: vector<1, 2> on d; relation r on d {{ div(div({alias_name}.outer(left=a,right=b))) = 0; }} }}"
                    ),
                ),
                unit(&operators, operator_file, dependency),
            ],
            vec![dependency_edge(&root, &operators)],
        ))
        .expect("resolved pure operator")
    };

    let first = analyzed("ops", "a/operator.eqi");
    let renamed = analyzed("tensor_ops", "relocated/definition.eqi");
    assert_eq!(
        first.canonical_declarations(),
        renamed.canonical_declarations()
    );
    let operator = first
        .canonical_declarations()
        .iter()
        .find(|declaration| declaration.kind() == CanonicalDeclarationKind::PureOperator)
        .expect("pure declaration");
    assert!(
        operator
            .canonical_form()
            .starts_with("eqiora.source-declaration.v1:sha256:")
    );
}

#[test]
fn private_pure_operator_cannot_cross_an_exact_package_boundary() {
    let root = namespace("root");
    let dependency = namespace("operators");
    let input = ResolvedHierarchyInput::new(
        root.clone(),
        vec![
            unit(
                &root,
                "root.eqi",
                "import operators.main as ops; model Main() { domain d = box(0,1); variable a: vector<1, 1> on d; variable b: vector<1, 1> on d; relation r on d { div(ops.outer(a=a,b=b)) = 0; } }",
            ),
            unit(
                &dependency,
                "operator.eqi",
                "private operator outer(input a: spatial[1], input b: spatial[1]): spatial[2] = component(a,0) * component(b,1);",
            ),
        ],
        vec![dependency_edge(&root, &dependency)],
    );
    let diagnostics = analyze_resolved_hierarchy(input)
        .expect("global package shape")
        .validate_definitions()
        .expect_err("private exact definitions are not importable");
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("private pure operator `ops.outer` cannot be imported")
    }));
}

#[test]
fn private_unknown_and_transitive_imports_fail_during_analysis() {
    let root = namespace("root");
    let dependency = namespace("dependency");
    let cases = [
        (
            "import dependency.main as dep; model Main() { instance c: dep.Private(); }",
            "component Private() {}",
            "private component `dep.Private` cannot be imported",
        ),
        (
            "model Main() { instance c: missing.C(); }",
            "public component C() {}",
            "unknown direct package alias `missing`",
        ),
        (
            "import dependency.main as dep; model Main() { instance c: dep.nested.C(); }",
            "public component C() {}",
            "uses transitive or member qualification",
        ),
    ];
    for (root_source, dependency_source, expected) in cases {
        let input = ResolvedHierarchyInput::new(
            root.clone(),
            vec![
                unit(&root, "root.eqi", root_source),
                unit(&dependency, "dependency.eqi", dependency_source),
            ],
            vec![dependency_edge(&root, &dependency)],
        );
        let diagnostics = analyze_resolved_hierarchy(input).unwrap_err();
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message().contains(expected)),
            "expected `{expected}`, got {diagnostics:#?}"
        );
    }
}

#[test]
fn package_local_names_do_not_collide_but_duplicates_and_aliases_do() {
    let root = namespace("root");
    let first = namespace("first");
    let second = namespace("second");
    let valid = ResolvedHierarchyInput::new(
        root.clone(),
        vec![
            unit(
                &root,
                "root.eqi",
                "import first.main as one; import second.main as two; model Main() { instance a: one.C(); instance b: two.C(); }",
            ),
            unit(
                &first,
                "first.eqi",
                "public component C() { parameter p: 1 = 1; relation law { p - 1 = 0; } }",
            ),
            unit(
                &second,
                "second.eqi",
                "public component C() { parameter p: 1 = 2; relation law { p - 2 = 0; } }",
            ),
        ],
        vec![
            dependency_edge(&root, &first),
            dependency_edge(&root, &second),
        ],
    );
    let analysis = analyze_resolved_hierarchy(valid).expect("names are package-local");
    let compiled = analysis
        .validate_definitions()
        .expect("definitions validate")
        .compile_root("Main")
        .expect("both definitions resolve");
    let (transaction, _, _) = compiled.into_parts();
    InMemoryGraphStore::new()
        .commit(transaction)
        .expect("both package-local definitions elaborate atomically");

    let duplicate = ResolvedHierarchyInput::new(
        first.clone(),
        vec![unit(
            &first,
            "a.eqi",
            "public component C() {} public component C() {}",
        )],
        vec![],
    );
    let diagnostics = analyze_resolved_hierarchy(duplicate).unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("duplicate component declaration `C`")
    }));

    let duplicate_alias = ResolvedHierarchyInput::new(
        root.clone(),
        vec![
            unit(
                &root,
                "root.eqi",
                "import first.main as lib; import second.main as lib; model Main() {}",
            ),
            unit(&first, "first.eqi", "public component C() {}"),
            unit(&second, "second.eqi", "public component D() {}"),
        ],
        vec![
            dependency_edge(&root, &first),
            dependency_edge(&root, &second),
        ],
    );
    let diagnostics = analyze_resolved_hierarchy(duplicate_alias).unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("duplicate direct alias `lib`")
    }));
}

#[test]
fn cross_package_import_cycle_fails_before_a_transaction_exists() {
    let root = namespace("root");
    let dependency = namespace("dependency");
    let input = ResolvedHierarchyInput::new(
        root.clone(),
        vec![
            unit(
                &root,
                "root.eqi",
                "import dependency.main as dep; public component A() { instance b: dep.B(); } model Main() {}",
            ),
            unit(
                &dependency,
                "dependency.eqi",
                "import root.main as app; public component B() { instance a: app.A(); }",
            ),
        ],
        vec![
            dependency_edge(&root, &dependency),
            dependency_edge(&dependency, &root),
        ],
    );
    let diagnostics = analyze_resolved_hierarchy(input).unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("semantic module import cycle")
    }));
    assert!(
        diagnostics
            .iter()
            .all(|diagnostic| diagnostic.graph_path().is_none())
    );
}

#[test]
fn unused_connector_contract_fails_definition_validation() {
    let root = namespace("root");
    let input = ResolvedHierarchyInput::new(
        root.clone(),
        vec![unit(
            &root,
            "root.eqi",
            "public connector Broken {\n  across voltage: mystery;\n  through current: A;\n} model Main() {}",
        )],
        vec![],
    );
    let analysis = analyze_resolved_hierarchy(input).expect("declarations index");
    let diagnostics = analysis.validate_definitions().unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("unknown SI base-dimension symbol `mystery`")
    }));
    assert!(
        diagnostics
            .iter()
            .all(|diagnostic| diagnostic.graph_path().is_none())
    );
}

#[test]
fn symbolic_component_interfaces_validate_without_occurrence_values() {
    let root = namespace("root");
    let input = ResolvedHierarchyInput::new(
        root.clone(),
        vec![unit(
            &root,
            "root.eqi",
            r#"
public component Leaf(parameter period: s, parameter offset: s = period) {


  relation invariant { offset - period = 0; }
}
public component Wrapper(parameter period: s) {

  instance leaf: Leaf(period = period);
}
model Empty() {}
"#,
        )],
        vec![],
    );
    analyze_resolved_hierarchy(input)
        .expect("declarations analyze")
        .validate_definitions()
        .expect("required public Parameters remain typed free variables");
}

#[test]
fn unused_nested_parameter_contracts_fail_before_root_selection() {
    let root = namespace("root");
    let input = ResolvedHierarchyInput::new(
        root.clone(),
        vec![unit(
            &root,
            "root.eqi",
            r#"
public component Leaf(parameter period: s) {  }
public component Missing() { instance leaf: Leaf(); }
public component WrongDimension(parameter length: m) {

  instance leaf: Leaf(period = length);
}
public component InvalidPrivate() { parameter hidden: s; }
model Empty() {}
"#,
        )],
        vec![],
    );
    let diagnostics = analyze_resolved_hierarchy(input)
        .expect("declarations analyze")
        .validate_definitions()
        .unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("required Parameter `period` has no instance binding")
    }));
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("Parameter binding has dimension")
    }));
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("required private Parameter `hidden` has no default")
    }));
    assert!(
        diagnostics
            .iter()
            .all(|diagnostic| diagnostic.graph_path().is_none())
    );
}
