use eqiora_api::editor::{EditorService, EditorSymbol, EditorWorkspaceSnapshot};
use eqiora_compiler::{CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit};
use eqiora_lang::{NotationLabel, NotationProfile};

fn label(symbol: &EditorSymbol) -> Option<String> {
    symbol
        .notation()
        .map(|notation| NotationLabel::from_notation(notation).render(NotationProfile::Plain))
}

fn assistance(marked: &str) -> Option<EditorSymbol> {
    let offset = marked.find('|').unwrap() as u32;
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, marked.replacen('|', "", 1));
    let file = workspace.files().next().unwrap();
    let (name, _) = workspace.document(file)?.name_at(offset)?;
    workspace.assistance(file, offset, &name)
}

fn assert_value_positions(source: &str, positions: &[(&str, usize, bool)]) {
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    for (needle, shift, is_value) in positions {
        let offset = (source.find(needle).unwrap() + shift) as u32;
        let (name, _) = workspace.document(file).unwrap().name_at(offset).unwrap();
        let symbol = workspace.assistance(file, offset, &name);
        if *is_value {
            let symbol = symbol.unwrap();
            assert!(symbol.notation().is_some(), "{needle}");
            assert!(symbol.detail().unwrap().contains("//"), "{needle}");
        } else {
            assert!(
                symbol.is_none_or(|symbol| {
                    symbol.notation().is_none()
                        && !symbol.detail().unwrap_or_default().contains("//")
                        && !symbol.detail().unwrap_or_default().contains("@{")
                }),
                "{needle}"
            );
        }
    }
}

#[test]
fn valid_keyword_spelling_is_not_a_declaration_or_value_reference() {
    assert_value_positions(
        "model M(){variable variable @{v}:1;relation r{variable=1;}}",
        &[
            ("variable variable", 0, false),
            ("variable @{", 0, true),
            ("variable=1", 0, true),
        ],
    );
}

#[test]
fn valid_unit_spelling_is_not_a_parameter_reference() {
    assert_value_positions(
        "model M(){parameter m @{p}:1=1;parameter copy:1=m;variable x:m;relation r{x=1[m];}}",
        &[
            ("x:m", 2, false),
            ("[m]", 1, false),
            ("m @{", 0, true),
            ("1=m", 2, true),
        ],
    );
}

#[test]
fn authored_symbols_keep_their_own_notation_across_lexical_scopes() {
    for (marked, expected) in [
        (
            "model Other(){variable value @{q}:1;} model M(){variable value @{\\mathbf{x_i}}:1;relation r{va|lue=1;}}",
            "x_{i}",
        ),
        (
            "model Other(){parameter value @{q}:1=1;} model M(){parameter value @{\\alpha_2}:1=1;relation r{1=va|lue;}}",
            "alpha_{2}",
        ),
        (
            "component C(parameter va|lue @{\\hat{q}^{2}}:1){}",
            "hat(q)^{2}",
        ),
        ("model M(){variable va|lue @{v}:unknown_type;}", "v"),
    ] {
        let symbol = assistance(marked).unwrap();
        assert_eq!(label(&symbol).as_deref(), Some(expected), "{marked}");
        let source = marked.replace('|', "");
        let notation = symbol.notation().unwrap();
        assert!(
            source[notation.range().start() as usize..notation.range().end() as usize]
                .starts_with("@{")
        );
    }
    assert!(
        assistance("model Other(){variable value @{q}:1;} model M(){variable va|lue:1;}")
            .unwrap()
            .notation()
            .is_none()
    );
    assert!(
        assistance("model Other(){variable value @{q}:1;} model M(){relation r{va|lue=1;}}")
            .is_none()
    );
}

#[test]
fn recovery_keeps_admitted_current_notation_without_granting_compiler_facts() {
    let source =
        "model M(){\n/// Current value.\nvariable value @{\\mathbf{x_i}}:1; relation r { value";
    let mut service = EditorService::new("main.eqi", 1, source);
    let offset = source.find("value @").unwrap() as u32;
    let symbol = service.current().assistance(offset, "value").unwrap();
    assert_eq!(label(&symbol).as_deref(), Some("x_{i}"));
    assert_eq!(symbol.doc_comment().unwrap().summary(), "Current value.");
    assert!(!symbol.detail().unwrap().contains("dimension"));
    assert!(
        service
            .current()
            .assistance(source.rfind("value").unwrap() as u32, "value")
            .unwrap()
            .notation()
            .is_none()
    );
    let changed = source.replace(r"@{\mathbf{x_i}}", r"@{\alpha_2}");
    service.replace(2, &changed).unwrap();
    assert!(service.replace(1, source).is_err());
    assert_eq!(
        label(
            &service
                .current()
                .assistance(changed.find("value @").unwrap() as u32, "value")
                .unwrap()
        )
        .as_deref(),
        Some("alpha_{2}")
    );
    for (version, decoration) in [(3, ""), (4, r"@{\input{secret}}"), (5, "@{")] {
        let changed = source.replace(r"@{\mathbf{x_i}}", decoration);
        service.replace(version, &changed).unwrap();
        assert!(
            service
                .current()
                .assistance(changed.find("value ").unwrap() as u32, "value")
                .and_then(|symbol| label(&symbol))
                .is_none()
        );
    }
}

#[test]
fn imported_public_members_keep_target_notation_and_hide_private_fields() {
    let owner = CompilationNamespaceId::new(["notation"]).unwrap();
    let main = "import notation.left as left;import notation.right as right;model M(){instance a:left.Part();instance b:right.Part();relation r{a.value=b.value;}}";
    let left = "public component Part(output value @{\\mathbf{x_i}}:1){variable hidden @{h}:1;}";
    let right = "public component Part(output value @{\\hat{q}^{2}}:1){}";
    let workspace = EditorWorkspaceSnapshot::analyze_modules(
        1,
        ResolvedHierarchyInput::new(
            owner.clone(),
            vec![
                ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", main).unwrap(),
                ResolvedSourceUnit::new(owner.clone(), "src/left.eqi", left).unwrap(),
                ResolvedSourceUnit::new(owner, "src/right.eqi", right).unwrap(),
            ],
            vec![],
        ),
    );
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace
        .files()
        .find(|file| file.ends_with(":src/main.eqi"))
        .unwrap();
    for (name, expected) in [("a.value", "x_{i}"), ("b.value", "hat(q)^{2}")] {
        let symbol = workspace
            .assistance(file, (main.find(name).unwrap() + 2) as u32, name)
            .unwrap();
        assert_eq!(label(&symbol).as_deref(), Some(expected));
        assert!(
            workspace
                .assistance(file, main.find(name).unwrap() as u32, name)
                .is_none()
        );
        assert!(
            workspace
                .assistance(file, (main.find(name).unwrap() + 1) as u32, name)
                .is_none()
        );
    }
    assert!(
        workspace
            .assistance(file, main.find("a.value").unwrap() as u32, "a.hidden")
            .is_none()
    );
}

#[test]
fn nested_binders_and_initializers_do_not_inherit_outer_notation() {
    for marked in [
        "model M(){parameter value @{q}:1=1;indexset Rows=range(2);relation r[value in Rows]{ordinal(va|lue)=0;}}",
        "model M(){indexset Rows=range(2);parameter value @{q}:integer=sum(ordinal(va|lue),over=(value in Rows));}",
    ] {
        assert!(assistance(marked).unwrap().notation().is_none(), "{marked}");
    }
}

#[test]
fn admitted_binder_scope_suppresses_an_outer_reference_without_rejecting_the_source() {
    let marked = "model M(){parameter value @{q}:integer=1;indexset Rows=range(2);relation r[row in Rows]{va|lue=ordinal(row);}}";
    let source = marked.replace('|', "");
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let symbol = workspace
        .assistance(file, marked.find('|').unwrap() as u32, "value")
        .unwrap();
    assert!(symbol.notation().is_none());

    let recovering = "model M(){indexset Rows=range(2);parameter value @{q}:integer=sum(ordinal(va|lue),over=(value in Rows));relation r{";
    assert!(assistance(recovering).unwrap().notation().is_none());
}

#[test]
fn recovered_declaration_name_can_equal_its_keyword_without_labelling_the_keyword() {
    for (declaration, name, expected) in [
        ("variable variable @{v}:1;", "variable", "v"),
        ("parameter parameter @{p}:1=1;", "parameter", "p"),
    ] {
        let valid = format!("model M(){{{declaration}}}");
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, valid);
        assert!(
            workspace.diagnostics().is_empty(),
            "{:?}",
            workspace.diagnostics()
        );
        let source = format!("model M(){{{declaration}relation r{{");
        let service = EditorService::new("main.eqi", 1, &source);
        let name_offset = source.find(&format!("{name} @")).unwrap() as u32;
        let symbol = service.current().assistance(name_offset, name).unwrap();
        assert_eq!(label(&symbol).as_deref(), Some(expected));
        let keyword_offset = source.find(&format!("{name} {name}")).unwrap() as u32;
        assert!(
            service
                .current()
                .assistance(keyword_offset, name)
                .unwrap()
                .notation()
                .is_none()
        );
    }
}
