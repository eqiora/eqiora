use super::*;

#[test]
fn direct_module_metadata_consumes_the_shared_input_byte_budget() {
    let owner = namespace("metadata");
    let document = eqiora_lang::parse("authored.eqi", "model Main() {}")
        .into_document()
        .unwrap();
    let range = document.models()[0].range();
    let bare = ResolvedSourceUnit::from_module(
        owner.clone(),
        "src/main.eqi",
        eqiora_lang::Module::from_document(document.clone()),
    )
    .unwrap();
    let documented = document
        .with_declaration_metadata(range, Some("d".repeat(64)), None)
        .unwrap();
    let documented = ResolvedSourceUnit::from_module(
        owner.clone(),
        "src/docs.eqi",
        eqiora_lang::Module::from_document(documented),
    )
    .unwrap();
    assert_eq!(documented.input_bytes(), bare.input_bytes() + 64);
    let total = bare.input_bytes() + documented.input_bytes();
    let limits = ResolvedHierarchyResourceLimits {
        source_units: 2,
        aliases: 0,
        source_unit_bytes: documented.input_bytes(),
        total_source_bytes: total,
    };
    let counts = [bare.input_bytes(), documented.input_bytes()];
    preflight_resolved_hierarchy_with_limits(counts, 0, limits).unwrap();
    let error = preflight_resolved_hierarchy_with_limits(
        counts,
        0,
        ResolvedHierarchyResourceLimits {
            total_source_bytes: total - 1,
            ..limits
        },
    )
    .unwrap_err();
    assert!(error.message().contains("total source-byte limit"));

    let text = format!("// {}\nmodel Main() {{}}", "x".repeat(512));
    let parsed = ResolvedSourceUnit::from_module(
        owner,
        "src/parsed.eqi",
        eqiora_lang::Module::parse("parsed.eqi", &text).unwrap(),
    )
    .unwrap();
    assert!(
        parsed.input_bytes() >= text.len(),
        "ordinary parsed comments remain budgeted"
    );
}

#[test]
fn canonical_declarations_ignore_files_formatting_and_input_order() {
    let package = namespace("org.example.library");
    let split = ResolvedHierarchyInput::with_root_module(
        package.clone(),
        "parts.component".split('.'),
        vec![
            module_unit(
                &package,
                "parts.component",
                "z/component.eqi",
                "public component C(parameter p: 1 = 2, parameter q: 1 = 3) {   }",
            ),
            module_unit(
                &package,
                "parts.connector",
                "a/connector.eqi",
                "public connector Pin {\n  across potential: 1;\n  through flow: 1;\n}",
            ),
        ],
        vec![],
    )
    .expect("split root module");
    let moved = ResolvedHierarchyInput::with_root_module(
        package.clone(),
        "parts.component".split('.'),
        vec![
            module_unit(
                &package,
                "parts.connector",
                "elsewhere/pin.eqi",
                "// relocated\npublic connector Pin {\n  across potential: 1;\n  through flow: 1;\n}",
            ),
            module_unit(
                &package,
                "parts.component",
                "elsewhere/c.eqi",
                "public component C(parameter q: 1=3, parameter p: 1=2) {\n \n \n}",
            ),
        ],
        vec![],
    )
    .expect("moved root module");

    let first = analyze_resolved_hierarchy(split).expect("first analysis");
    let second = analyze_resolved_hierarchy(moved).expect("second analysis");
    assert_eq!(
        first.canonical_declarations(),
        second.canonical_declarations()
    );
    assert_eq!(
        first.canonical_declarations()[0].path(),
        "parts.component.C",
        "canonical declarations sort by path"
    );
    assert!(
        first.canonical_declarations()[0]
            .canonical_form()
            .starts_with("eqiora.source-declaration.v1:sha256:")
    );
}

#[test]
fn compiler_owned_math_root_cannot_be_a_package_alias() {
    let root = namespace("root");
    let dependency = namespace("dependency");
    let input = ResolvedHierarchyInput::new(
        root.clone(),
        vec![
            unit(
                &root,
                "root.eqi",
                "import dependency.main as math; model Main() {}",
            ),
            unit(&dependency, "dependency.eqi", "public component C() {}"),
        ],
        vec![dependency_edge(&root, &dependency)],
    );
    let diagnostics = analyze_resolved_hierarchy(input).expect_err("math alias must reject");
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("direct alias `math` is reserved for compiler-owned scalar mathematics")
    }));
}

#[test]
fn compiler_owned_math_root_cannot_be_a_property_declaration() {
    for source in [
        "public property contract math(): 1 { derivatives value_only; } model Main() {}",
        "public property contract C(): 1 { derivatives value_only; } public property release math: C { analytic { value = 1; source_unit: 1 = 1; } validity unconditional; outside reject; branch single; citation org.example; license spdx.CC0_1_0; } model Main() {}",
    ] {
        let root = namespace("root");
        let input = ResolvedHierarchyInput::new(
            root.clone(),
            vec![unit(&root, "root.eqi", source)],
            vec![],
        );
        let diagnostics =
            analyze_resolved_hierarchy(input).expect_err("math declaration must reject");
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic
                .message()
                .contains("identifier `math` is reserved for compiler-owned scalar mathematics")
        }));
    }
}

#[test]
fn hierarchy_does_not_resolve_bare_sin_as_a_user_operator() {
    let root = namespace("root");
    let input = ResolvedHierarchyInput::new(
        root.clone(),
        vec![unit(
            &root,
            "root.eqi",
            "public operator sin(input value: scalar): scalar = rational(1, 1); component C(support body: volume(ambient_dimension = 1)) {  variable value: 1 on body; initial { value = 0; } relation law on body { value - sin(0) = 0; } }",
        )],
        vec![],
    );
    let diagnostics = analyze_resolved_hierarchy(input)
        .expect("resolved graph shape")
        .validate_definitions()
        .expect_err("bare sin call must remain invalid");
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic
                .message()
                .contains("bare `sin` is not language vocabulary")
        }),
        "{diagnostics:#?}"
    );
}

#[test]
fn nominal_operator_signature_aliases_preserve_exact_declaration_targets() {
    let owner = namespace("root");
    let analyze = |alias: &str, target: &str| {
        analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
            owner.clone(),
            vec![
                module_unit(&owner, "types", "types.eqi", "public space A=orthonormal(a,b); public space B=orthonormal(c,d);"),
                unit(&owner, "main.eqi", &format!(
                    "import root.types as {alias}; public operator retain(input x:coordinates<complex<1>,{alias}.{target}>):coordinates<complex<1>,{alias}.{target}>=component(x,0); model Main(){{}}"
                )),
            ],
            vec![],
        )).unwrap()
    };
    let first = analyze("basis", "A");
    let renamed = analyze("other", "A");
    let rewired = analyze("basis", "B");
    assert_eq!(
        first.canonical_declarations(),
        renamed.canonical_declarations()
    );
    assert_ne!(
        first.canonical_declarations(),
        rewired.canonical_declarations()
    );
}
