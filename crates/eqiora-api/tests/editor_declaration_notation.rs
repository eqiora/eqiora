use eqiora_api::editor::{EditorWorkspaceService, EditorWorkspaceSnapshot};
use eqiora_compiler::{CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit};
use eqiora_lang::{NotationLabel, NotationProfile};

const MAIN: &str = "import notation.left as left; import notation.right as right; model Main @{M}(){instance a @{a_9}:left.Part();instance b:right.Part();}";

fn workspace(version: u64, left_notation: &str) -> EditorWorkspaceSnapshot {
    let owner = CompilationNamespaceId::new(["notation"]).unwrap();
    let left =
        format!("// 🧪\n/// Left declaration.\npublic component Part {left_notation}() {{}}");
    let right = "/// Right declaration.\npublic component Part @{\\hat{q}^{2}}() {}";
    EditorWorkspaceSnapshot::analyze_modules(
        version,
        ResolvedHierarchyInput::new(
            owner.clone(),
            vec![
                ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", MAIN).unwrap(),
                ResolvedSourceUnit::new(owner.clone(), "src/left.eqi", left).unwrap(),
                ResolvedSourceUnit::new(owner, "src/right.eqi", right).unwrap(),
            ],
            vec![],
        ),
    )
}

fn label(workspace: &EditorWorkspaceSnapshot, reference: &str) -> Option<String> {
    let file = workspace
        .files()
        .find(|file| file.ends_with(":src/main.eqi"))?;
    let (definition, _) = workspace.hover(file, MAIN.find(reference)? as u32)?;
    definition
        .notation()
        .map(|notation| NotationLabel::from_notation(notation).render(NotationProfile::Plain))
}

#[test]
fn notation_uses_exact_canonical_target_and_retains_source_provenance() {
    let workspace = workspace(1, r"@{\mathbf{x_i}}");
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    // Plain rendering deliberately drops the bold style and keeps intrinsic scripts.
    // The instance's a_9 decoration does not qualify a declaration hover.
    assert_eq!(label(&workspace, "left.Part"), Some("x_{i}".into()));
    assert_eq!(label(&workspace, "right.Part"), Some("hat(q)^{2}".into()));
    assert_eq!(label(&workspace, "Main"), Some("M".into()));
    for reference in workspace.references() {
        let target = reference.definition();
        let canonical = workspace
            .definitions()
            .iter()
            .find(|definition| {
                definition.file() == target.file() && definition.range() == target.range()
            })
            .unwrap();
        assert_eq!(target, canonical);
        let notation = target.notation().unwrap();
        let (hovered, source) = workspace
            .hover(target.file(), target.name_range().unwrap().start())
            .unwrap();
        let start = (notation.range().start() - target.range().start()) as usize;
        let end = (notation.range().end() - target.range().start()) as usize;
        let expected = if target.file().ends_with("left.eqi") {
            r"@{\mathbf{x_i}}"
        } else {
            r"@{\hat{q}^{2}}"
        };
        assert_eq!(&source[start..end], expected);
        assert_eq!(hovered.notation(), target.notation());
        assert_eq!(hovered.doc_comment(), target.doc_comment());
    }
}

#[test]
fn changed_absent_and_rejected_notation_never_reuse_an_old_label() {
    let old = workspace(1, r"@{\mathbf{x_i}}");
    let mut service = EditorWorkspaceService::new(old.clone());
    service.begin(2).unwrap();
    assert!(service.current().is_none());
    assert!(service.replace(old).is_err());
    let current = service.replace(workspace(2, r"@{\alpha_2}")).unwrap();
    assert_eq!(label(current, "left.Part"), Some("alpha_{2}".into()));
    assert_eq!(label(current, "right.Part"), Some("hat(q)^{2}".into()));

    let absent = workspace(3, "");
    assert!(absent.diagnostics().is_empty());
    assert_eq!(label(&absent, "left.Part"), None);
    assert_eq!(label(&absent, "right.Part"), Some("hat(q)^{2}".into()));

    let rejected = workspace(4, r"@{\input{secret}}");
    assert!(!rejected.diagnostics().is_empty());
    assert_eq!(label(&rejected, "left.Part"), None);
}

#[test]
fn canonical_name_ranges_exclude_equal_spelled_declaration_keywords() {
    for source in [
        "// 🧪\r\npublic component component(){} model model(){instance a:component();}",
        "// 🧪\r\npublic component component @{c}(){} model model @{m}(){instance a:component();}",
    ] {
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
        assert!(
            workspace.diagnostics().is_empty(),
            "{:?}",
            workspace.diagnostics()
        );
        for name in ["component", "model"] {
            let definition = workspace
                .definitions()
                .iter()
                .find(|value| value.path().rsplit('.').next() == Some(name))
                .unwrap();
            let prefix = format!("{name} {name}");
            let start = (source.find(&prefix).unwrap() + name.len() + 1) as u32;
            let expected = eqiora_lang::TextRange::new(start, start + name.len() as u32);
            assert_eq!(definition.name_range(), Some(expected));
            for reference in workspace
                .references()
                .iter()
                .filter(|value| value.definition().path().rsplit('.').next() == Some(name))
            {
                assert_eq!(reference.definition().name_range(), Some(expected));
            }
        }
        assert!(!workspace.references().is_empty());
    }
}
