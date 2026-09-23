use eqiora_api::editor::{EditorWorkspaceService, EditorWorkspaceSnapshot};
use eqiora_compiler::{
    CompilationNamespaceId, ResolvedDependency, ResolvedHierarchyInput, ResolvedSourceUnit,
};
use eqiora_core::Span;

fn token(file: &str, source: &str, occurrence: &str, shift: usize, name: &str) -> Span {
    let start = (source.find(occurrence).unwrap() + shift) as u32;
    Span {
        file: file.to_owned(),
        start,
        end: start + name.len() as u32,
    }
}

fn query(workspace: &EditorWorkspaceSnapshot, cursor: &Span, include: bool) -> Option<Vec<Span>> {
    workspace.value_references_at_position(
        &cursor.file,
        workspace.document(&cursor.file)?.position(cursor.start)?,
        include,
    )
}

fn sorted(mut spans: Vec<Span>) -> Vec<Span> {
    spans.sort_by(|a, b| (&a.file, a.start, a.end).cmp(&(&b.file, b.start, b.end)));
    spans
}

#[test]
fn port_references_follow_declaration_provenance_across_models_modules_and_packages() {
    let app = CompilationNamespaceId::new(["app"]).unwrap();
    let left = CompilationNamespaceId::new(["left", "exact-a"]).unwrap();
    let right = CompilationNamespaceId::new(["right", "exact-b"]).unwrap();
    let main = "// 🧪\r\nimport left.lib as left;import right.lib as right;import app.local as local;model First(){instance a:left.Part();instance b:left.Part();instance foreign:right.Part();instance own:local.Part();indexset Rows=range(2);relation values{(a . p)=b.p;foreign.p=own.p;}relation hidden[row in Rows]{a.p=1;}}model Second(){instance c:left.Part();relation values{c.p=1;}}model Shadow(){instance a:right.Part();relation values{a.p=1;}}";
    let other =
        "import left.lib as left;model Third(){instance d:left.Part();relation values{d.p=1;}}";
    // Identical declaration spellings and offsets in separate exact sources
    // must not merge; a.p/b.p do share one declaration, not an occurrence.
    let library = "// 🧪\r\npublic component Part(output p @{q}:1){port hidden:signal output 1;}public component Never(output p:1){}";
    let units = [
        ResolvedSourceUnit::new(app.clone(), "src/main.eqi", main).unwrap(),
        ResolvedSourceUnit::new(app.clone(), "src/other.eqi", other).unwrap(),
        ResolvedSourceUnit::new(left.clone(), "src/lib.eqi", library).unwrap(),
        ResolvedSourceUnit::new(right.clone(), "src/lib.eqi", library).unwrap(),
        ResolvedSourceUnit::new(app.clone(), "src/local.eqi", library).unwrap(),
    ];
    let files = units.each_ref().map(|unit| unit.diagnostic_file());
    let workspace = EditorWorkspaceSnapshot::analyze_modules(
        1,
        ResolvedHierarchyInput::new(
            app.clone(),
            units.to_vec(),
            vec![
                ResolvedDependency::new(app.clone(), left),
                ResolvedDependency::new(app, right),
            ],
        ),
    );
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let a = token(&files[0], main, "a . p", 4, "p");
    let expected = sorted(vec![
        a.clone(),
        token(&files[0], main, "b.p", 2, "p"),
        token(&files[0], main, "c.p=", 2, "p"),
        token(&files[1], other, "d.p", 2, "p"),
    ]);
    let declaration = token(&files[2], library, "p @{", 0, "p");
    for cursor in [&a, &declaration, &expected[3]] {
        assert_eq!(query(&workspace, cursor, false), Some(expected.clone()));
        let mut included = expected.clone();
        included.push(declaration.clone());
        assert_eq!(query(&workspace, cursor, true), Some(sorted(included)));
    }
    let right_expected = sorted(vec![
        token(&files[0], main, "foreign.p", 8, "p"),
        token(
            &files[0],
            main,
            "Shadow(){instance a:right.Part();relation values{a.p",
            "Shadow(){instance a:right.Part();relation values{a.".len(),
            "p",
        ),
    ]);
    assert_eq!(
        query(
            &workspace,
            &token(&files[3], library, "p @{", 0, "p"),
            false
        ),
        Some(right_expected)
    );
    assert_eq!(
        query(
            &workspace,
            &token(&files[4], library, "p @{", 0, "p"),
            false
        ),
        Some(vec![token(&files[0], main, "own.p", 4, "p")])
    );
    for cursor in [
        token(&files[0], main, "a . p", 0, "a"),
        token(&files[0], main, "a . p", 2, "."),
        token(
            &files[0],
            main,
            "hidden[row in Rows]{a.p",
            "hidden[row in Rows]{a.".len(),
            "p",
        ),
        token(&files[2], library, "hidden:", 0, "hidden"),
        token(
            &files[2],
            library,
            "Never(output p",
            "Never(output ".len(),
            "p",
        ),
        token(&files[2], library, "@{q}", 2, "q"),
    ] {
        assert_eq!(query(&workspace, &cursor, true), None, "{cursor:?}");
    }
}

#[test]
fn unused_owned_and_instantiated_ports_are_distinct_from_unadmitted_targets() {
    let source = "component C(output p:1){}component Never(output p:1){}model M(){port unused:signal input 1;instance p:C();relation r{p.p=1;}}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let unused = token(file, source, "unused:", 0, "unused");
    assert_eq!(query(&workspace, &unused, false), Some(vec![]));
    assert_eq!(query(&workspace, &unused, true), Some(vec![unused.clone()]));
    assert_eq!(
        query(&workspace, &token(file, source, "p.p", 0, "p"), true),
        None
    );
    let declaration = token(file, source, "C(output p", "C(output ".len(), "p");
    let reference = token(file, source, "p.p", 2, "p");
    assert_eq!(
        query(&workspace, &declaration, false),
        Some(vec![reference])
    );
    for marked in [
        "component C(output p:1){}model M(){instance a:C();variable x:a.|p;}",
        "component C(output p:1){}model M(){instance a:C();relation r{a.p.|deeper=1;}}",
        "component C(output p:1){}model M(){instance a:C();relation r{a.|p",
    ] {
        let current = EditorWorkspaceSnapshot::analyze_standalone(2, marked.replace('|', ""));
        let file = current.files().next().unwrap();
        assert!(
            current
                .value_references_at_position(
                    file,
                    current
                        .document(file)
                        .unwrap()
                        .position(marked.find('|').unwrap() as u32)
                        .unwrap(),
                    true
                )
                .is_none()
        );
    }
}

#[test]
fn port_reference_sets_follow_unsaved_versions_without_publishing_stale_or_recovered_results() {
    let source = "component C(output p @{q}:1){}model M(){instance a:C();relation r{a.p=1;}}";
    let old = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    let mut service = EditorWorkspaceService::new(old.clone());
    service.begin(2).unwrap();
    assert!(service.current().is_none());
    assert!(service.replace(old).is_err());
    let changed = source.replace("relation r{a.p=1;}", "\r\nrelation r{a.p=a.p;}");
    let current = service
        .replace(EditorWorkspaceSnapshot::analyze_standalone(2, &changed))
        .unwrap();
    let file = current.files().next().unwrap();
    let expected = changed
        .match_indices("a.p")
        .map(|(start, _)| Span {
            file: file.to_owned(),
            start: start as u32 + 2,
            end: start as u32 + 3,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        query(current, &token(file, &changed, "p @{", 0, "p"), false),
        Some(expected)
    );
    for (version, text) in [
        (3, source.replace("a.p=1;", "a.p=1[m];")),
        (4, source.replace("a.p=1;}}", "a.p=")),
    ] {
        service.begin(version).unwrap();
        let current = service
            .replace(EditorWorkspaceSnapshot::analyze_standalone(version, &text))
            .unwrap();
        assert!(!current.diagnostics().is_empty());
        let file = current.files().next().unwrap();
        assert_eq!(
            query(current, &token(file, &text, "p @{", 0, "p"), true),
            None
        );
    }
    service.begin(5).unwrap();
    let current = service
        .replace(EditorWorkspaceSnapshot::analyze_standalone(5, source))
        .unwrap();
    let file = current.files().next().unwrap();
    assert_eq!(
        query(current, &token(file, source, "p @{", 0, "p"), false),
        Some(vec![token(file, source, "a.p=", 2, "p")])
    );
}
