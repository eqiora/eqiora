use eqiora_api::editor::{EditorWorkspaceService, EditorWorkspaceSnapshot};
use eqiora_compiler::{CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit};
use eqiora_core::Span;

fn span(file: &str, source: &str, needle: &str, name: &str) -> Span {
    let start = source.find(needle).unwrap() as u32;
    Span {
        file: file.into(),
        start,
        end: start + name.len() as u32,
    }
}

fn assert_clock(
    workspace: &EditorWorkspaceSnapshot,
    file: &str,
    source: &str,
    declaration: &str,
    uses: &[&str],
) {
    let snapshot = workspace.document(file).unwrap();
    let target = span(file, source, declaration, "tick");
    let references = uses
        .iter()
        .map(|needle| span(file, source, needle, "tick"))
        .collect::<Vec<_>>();
    for reference in &references {
        assert_eq!(
            workspace
                .value_definition_at_position(file, snapshot.position(reference.start).unwrap()),
            Some(target.clone())
        );
    }
    for cursor in std::iter::once(&target).chain(&references) {
        let position = snapshot.position(cursor.start).unwrap();
        assert_eq!(
            workspace.value_references_at_position(file, position, false),
            Some(references.clone())
        );
        let expected = std::iter::once(target.clone())
            .chain(references.clone())
            .collect();
        assert_eq!(
            workspace.value_references_at_position(file, position, true),
            Some(expected)
        );
    }
}

#[test]
fn owned_clock_navigation_preserves_declaration_identity_and_exact_source_tokens() {
    let source = "// 🧪\r\ncomponent C(){clock tick @{t_c}=periodic(100[ms]);clock peer=periodic(100[ms]);state memory:1 at tick;relation c{period(tick)=period(tick)+0[s];}} model Other(){clock tick=periodic(1[s]);relation o{period(tick)=1[s];}} model M(){clock tick @{t_m}=periodic(100[ms]);instance first:C();instance second:C();relation m{period(tick)=period(tick);}}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    assert_clock(
        &workspace,
        file,
        source,
        "tick @{t_c}",
        &["tick;", "tick)=period(tick)+", "tick)+"],
    );
    assert_clock(
        &workspace,
        file,
        source,
        "tick=periodic(1[s])",
        &["tick)=1[s]"],
    );
    assert_clock(
        &workspace,
        file,
        source,
        "tick @{t_m}",
        &["tick)=period(tick);", "tick);"],
    );
    let snapshot = workspace.document(file).unwrap();
    let unused = snapshot
        .position(source.find("peer=").unwrap() as u32)
        .unwrap();
    assert_eq!(
        workspace.value_references_at_position(file, unused, false),
        Some(vec![])
    );
    assert_eq!(
        workspace.value_references_at_position(file, unused, true),
        Some(vec![span(file, source, "peer=", "peer")])
    );
    // Notation contents and unit tokens are not declaration references.
    for needle in ["t_c}", "ms]", "period(tick)"] {
        let position = snapshot
            .position(source.find(needle).unwrap() as u32)
            .unwrap();
        assert!(
            workspace
                .value_definition_at_position(file, position)
                .is_none(),
            "{needle}"
        );
        assert!(
            workspace
                .value_references_at_position(file, position, true)
                .is_none(),
            "{needle}"
        );
    }
}

#[test]
fn clock_navigation_never_recovers_invalid_or_nested_bindings() {
    for (marked, valid_source) in [
        (
            "model M(){clock tick=periodic(1[s]);indexset Rows=range(2);relation r[i in Rows] at |tick{period(tick)=1[s];}}",
            true,
        ),
        (
            "model M(){state x:m;event hit=crossing(x,direction=falling);relation reset at |hit{next(x)=1[m];}}",
            true,
        ),
        ("model M(){clock |tick=periodic(0[s]);}", false),
        ("model M(){clock |tick=periodic(1[m]);}", false),
        (
            "model M(){clock tick=periodic(1[s]);relation r{period(|tick)",
            false,
        ),
        (
            "model M(){clock tick=periodic(1[s]);indexset Rows=range(2);relation r[i in Rows]{period(|tick)=1[s];}}",
            true,
        ),
        (
            "component C(parameter n:integer){clock |tick=periodic(1[s]);variable unknown:array<1,n>;}",
            false,
        ),
    ] {
        let source = marked.replace('|', "");
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
        if valid_source {
            assert!(
                workspace.diagnostics().is_empty(),
                "{marked}: {:?}",
                workspace.diagnostics()
            );
        }
        let file = workspace.files().next().unwrap();
        let position = workspace
            .document(file)
            .unwrap()
            .position(marked.find('|').unwrap() as u32)
            .unwrap();
        assert!(
            workspace
                .value_definition_at_position(file, position)
                .is_none(),
            "{marked}"
        );
        assert!(
            workspace
                .value_references_at_position(file, position, true)
                .is_none(),
            "{marked}"
        );
    }
}

#[test]
fn same_named_clock_files_and_new_unsaved_versions_do_not_share_references() {
    let namespace = CompilationNamespaceId::new(["clock_navigation"]).unwrap();
    let source = "public component C(){clock tick=periodic(1[s]);relation r{period(tick)=1[s];}}";
    let workspace = EditorWorkspaceSnapshot::analyze_modules(
        1,
        ResolvedHierarchyInput::new(
            namespace.clone(),
            vec![
                ResolvedSourceUnit::new(namespace.clone(), "src/main.eqi", source).unwrap(),
                ResolvedSourceUnit::new(namespace, "src/second.eqi", source).unwrap(),
            ],
            vec![],
        ),
    );
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    for file in workspace.files() {
        assert_clock(&workspace, file, source, "tick=", &["tick)=1"]);
    }
    let mut service = EditorWorkspaceService::new(workspace.clone());
    service.begin(2).unwrap();
    assert!(service.current().is_none());
    assert!(service.replace(workspace).is_err());
    let changed = source.replace("relation r{", "relation r{\r\n");
    let current = service
        .replace(EditorWorkspaceSnapshot::analyze_standalone(2, &changed))
        .unwrap();
    let file = current.files().next().unwrap();
    assert_clock(current, file, &changed, "tick=", &["tick)=1"]);
    service.begin(3).unwrap();
    let current = service
        .replace(EditorWorkspaceSnapshot::analyze_standalone(
            3,
            "component C(){clock tick=periodic(0[s]);}",
        ))
        .unwrap();
    let file = current.files().next().unwrap();
    let position = current.document(file).unwrap().position(20).unwrap();
    assert!(
        current
            .value_references_at_position(file, position, true)
            .is_none()
    );
}

#[test]
fn activation_clauses_share_owned_clock_declarations_and_follow_current_source() {
    for owner in ["model M", "component C"] {
        let source = format!(
            "{owner}(){{clock tick=periodic(100[ms]);state memory:1 at tick;port out:signal output 1 at tick;let previous at tick=pre(memory);relation step at tick{{next(memory)=previous;out=memory;}}}}"
        );
        let old = EditorWorkspaceSnapshot::analyze_standalone(1, &source);
        assert!(
            old.diagnostics().is_empty(),
            "{source}: {:?}",
            old.diagnostics()
        );
        let mut service = EditorWorkspaceService::new(old.clone());
        let uses = ["tick;port", "tick;let", "tick=pre", "tick{"];
        {
            let current = &old;
            let file = current.files().next().unwrap();
            assert_clock(current, file, &source, "tick=periodic", &uses);
            for needle in uses {
                let hover = current
                    .assistance(file, source.find(needle).unwrap() as u32, "tick")
                    .unwrap();
                assert!(
                    hover
                        .detail()
                        .unwrap()
                        .contains("periodic clock; period 1/10 s; phase 0/1 s;")
                );
            }
        }
        service.begin(2).unwrap();
        assert!(service.current().is_none());
        assert!(service.replace(old).is_err());
        let changed = format!("// 🧪\r\n{source}");
        let current = service
            .replace(EditorWorkspaceSnapshot::analyze_standalone(2, &changed))
            .unwrap();
        let file = current.files().next().unwrap();
        assert_clock(current, file, &changed, "tick=periodic", &uses);
    }
}

#[test]
fn clock_requirements_have_declaration_references_without_borrowing_instance_schedules() {
    for owner in ["model", "component"] {
        let mut source = format!(
            "{owner} Definition(clock tick @{{t}}:periodic,clock peer:periodic){{state memory:1 at tick;relation update at tick{{next(memory)=pre(memory);}}relation duration{{period(tick)=period(tick);}}}}"
        );
        if owner == "component" {
            source.push_str(" model Main(){clock first=periodic(100[ms]);clock second=periodic(200[ms]);instance one:Definition(tick=first,peer=second);instance two:Definition(tick=second,peer=first);}");
        }
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, &source);
        assert!(
            workspace.diagnostics().is_empty(),
            "{source}: {:?}",
            workspace.diagnostics()
        );
        let file = workspace.files().next().unwrap();
        let uses = ["tick;relation", "tick{", "tick)=period", "tick);"];
        assert_clock(&workspace, file, &source, "tick @{", &uses);
        let snapshot = workspace.document(file).unwrap();
        let peer = snapshot
            .position(source.find("peer:periodic").unwrap() as u32)
            .unwrap();
        assert_eq!(
            workspace.value_references_at_position(file, peer, false),
            Some(vec![])
        );
        assert_eq!(
            workspace.value_references_at_position(file, peer, true),
            Some(vec![span(file, &source, "peer:periodic", "peer")])
        );
        let definition = snapshot
            .symbols()
            .iter()
            .find(|symbol| symbol.name() == "Definition")
            .unwrap();
        for clock in definition
            .children()
            .iter()
            .filter(|symbol| ["tick", "peer"].contains(&symbol.name()))
        {
            assert!(
                clock.detail().is_none(),
                "no concrete period or phase for a requirement"
            );
        }
        for needle in uses {
            let hover = workspace
                .assistance(file, source.find(needle).unwrap() as u32, "tick")
                .unwrap();
            assert!(!hover.detail().unwrap().contains("periodic clock;"));
        }
        if owner == "component" {
            let position = snapshot
                .position(source.find("tick=first").unwrap() as u32)
                .unwrap();
            assert!(
                workspace
                    .value_definition_at_position(file, position)
                    .is_none()
            );
            assert!(
                workspace
                    .value_references_at_position(file, position, true)
                    .is_none()
            );
        }
    }
}

#[test]
fn clock_names_equal_to_unit_symbols_do_not_capture_quantity_units() {
    for source in [
        "model M(clock s:periodic){relation r{period(s)=1[s];}}",
        "model M(){clock s=periodic(1[s]);relation r{period(s)=1[s];}}",
    ] {
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
        assert!(
            workspace.diagnostics().is_empty(),
            "{:?}",
            workspace.diagnostics()
        );
        let file = workspace.files().next().unwrap();
        let snapshot = workspace.document(file).unwrap();
        let declaration = if source.contains("s:periodic") {
            "s:periodic"
        } else {
            "s=periodic"
        };
        let target = span(file, source, declaration, "s");
        let reference = span(file, source, "s)=", "s");
        assert_eq!(
            workspace
                .value_definition_at_position(file, snapshot.position(reference.start).unwrap()),
            Some(target.clone())
        );
        assert_eq!(
            workspace.value_references_at_position(
                file,
                snapshot.position(target.start).unwrap(),
                false
            ),
            Some(vec![reference])
        );
        for (offset, _) in source.match_indices("s]") {
            let position = snapshot.position(offset as u32).unwrap();
            assert!(
                workspace
                    .value_definition_at_position(file, position)
                    .is_none()
            );
            assert!(
                workspace
                    .value_references_at_position(file, position, true)
                    .is_none()
            );
        }
    }
}
