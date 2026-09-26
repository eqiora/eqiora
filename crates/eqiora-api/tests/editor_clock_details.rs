use eqiora_api::editor::{
    EditorSnapshot, EditorSymbol, EditorWorkspaceService, EditorWorkspaceSnapshot,
};
use eqiora_compiler::{CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit};

const EXACT: &str = "periodic clock; period 1/10 s; phase 1/20 s; Model-local declaration; occurrence identity unknown";

fn member<'a>(snapshot: &'a EditorSnapshot, owner: &str, name: &str) -> &'a EditorSymbol {
    snapshot
        .symbols()
        .iter()
        .find(|symbol| symbol.name() == owner)
        .unwrap()
        .children()
        .iter()
        .find(|symbol| symbol.name() == name)
        .unwrap()
}

#[test]
fn exact_clock_details_preserve_declarations_without_claiming_occurrence_identity() {
    let source = "model Other(){clock tick=periodic(2[s]);} model M(){clock tick=periodic(100[ms],phase=50[ms]);clock peer=periodic(1[s]/10,phase=1[s]/20);state memory:1 at tick;relation schedule{period(tick)=period(tick);}}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let snapshot = workspace.document(file).unwrap();
    let tick = member(snapshot, "M", "tick");
    let peer = member(snapshot, "M", "peer");
    // 100/1000 = 1/10 and 50/1000 = 1/20 seconds, independently of display output.
    assert_eq!(tick.detail(), Some(EXACT));
    assert_eq!(peer.detail(), Some(EXACT));
    assert_ne!(tick.range(), peer.range());
    for symbol in [tick, peer] {
        let hover = workspace
            .assistance(file, symbol.range().start() + 6, symbol.name())
            .unwrap();
        assert_eq!(hover.range(), symbol.range());
        assert_eq!(hover.detail().unwrap().matches(EXACT).count(), 1);
    }
    let reference = workspace
        .assistance(file, source.rfind("tick").unwrap() as u32, "tick")
        .unwrap();
    assert_eq!(reference.range(), tick.range());
    assert!(reference.detail().unwrap().contains(EXACT));
    let activation = workspace
        .assistance(file, source.find("tick;").unwrap() as u32, "tick")
        .unwrap();
    assert_eq!(activation.range(), tick.range());
    assert!(activation.detail().unwrap().contains(EXACT));
    assert!(
        member(snapshot, "Other", "tick")
            .detail()
            .unwrap()
            .contains("period 2/1 s; phase 0/1 s")
    );
    assert!(
        member(snapshot, "M", "memory")
            .detail()
            .unwrap()
            .contains("activation tick (occurrence identity unknown)")
    );
}

#[test]
fn clock_details_select_the_current_file_and_definition() {
    let namespace = CompilationNamespaceId::new(["clocks"]).unwrap();
    let workspace = EditorWorkspaceSnapshot::analyze_modules(
        1,
        ResolvedHierarchyInput::new(
            namespace.clone(),
            vec![
                ResolvedSourceUnit::new(
                    namespace.clone(),
                    "src/main.eqi",
                    "model Main(){clock tick=periodic(100[ms],phase=50[ms]);}",
                )
                .unwrap(),
                ResolvedSourceUnit::new(
                    namespace,
                    "src/other.eqi",
                    "public component Other(){clock tick=periodic(3[s]);}",
                )
                .unwrap(),
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
        let (owner, expected) = if file.ends_with("main.eqi") {
            ("Main", EXACT)
        } else {
            (
                "Other",
                "periodic clock; period 3/1 s; phase 0/1 s; Component-local declaration; occurrence identity unknown",
            )
        };
        let symbol = member(workspace.document(file).unwrap(), owner, "tick");
        assert_eq!(symbol.detail(), Some(expected));
        let hover = workspace
            .assistance(file, symbol.range().start() + 6, "tick")
            .unwrap();
        assert!(hover.detail().unwrap().contains(expected));
    }
}

#[test]
fn clock_updates_reject_stale_facts_and_recover_after_invalid_source() {
    let valid = "model M(){clock tick=periodic(100[ms],phase=50[ms]);}";
    let old = EditorWorkspaceSnapshot::analyze_standalone(1, valid);
    let mut service = EditorWorkspaceService::new(old.clone());
    service.begin(2).unwrap();
    assert!(service.current().is_none());
    assert!(service.replace(old).is_err());
    for (index, invalid) in [
        "model M(){clock tick=periodic(0[s]);}",
        "model M(){clock tick=periodic(1[m]);}",
        "model M(){clock tick=periodic(1[s],phase=-1[s]);}",
        "model M(){clock tick=periodic(100[ms]);relation r{",
    ]
    .into_iter()
    .enumerate()
    {
        let version = index as u64 + 2;
        if index > 0 {
            service.begin(version).unwrap();
        }
        let current = service
            .replace(EditorWorkspaceSnapshot::analyze_standalone(
                version, invalid,
            ))
            .unwrap();
        assert!(!current.diagnostics().is_empty(), "{invalid}");
        let file = current.files().next().unwrap();
        let symbol = member(current.document(file).unwrap(), "M", "tick");
        assert!(symbol.detail().is_none(), "{invalid}");
        let hover = current
            .assistance(file, symbol.range().start() + 6, "tick")
            .unwrap();
        assert!(!hover.detail().unwrap().contains("periodic clock;"));
    }
    service.begin(6).unwrap();
    let current = service
        .replace(EditorWorkspaceSnapshot::analyze_standalone(
            6,
            valid.replace("100[ms]", "200[ms]"),
        ))
        .unwrap();
    let file = current.files().next().unwrap();
    assert_eq!(
        member(current.document(file).unwrap(), "M", "tick").detail(),
        Some(
            "periodic clock; period 1/5 s; phase 1/20 s; Model-local declaration; occurrence identity unknown"
        )
    );
}

#[test]
fn borrowed_component_and_event_activations_have_no_concrete_clock_facts() {
    let source = "component C(clock tick:periodic){} model Borrowed(clock tick:periodic){state memory:1 at tick;} model M(){state x:m;event hit=crossing(x,direction=falling);}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let snapshot = workspace.document(file).unwrap();
    for (owner, name) in [("C", "tick"), ("Borrowed", "tick")] {
        let symbol = member(snapshot, owner, name);
        assert!(symbol.detail().is_none());
        let hover = workspace
            .assistance(file, symbol.range().start() + 6, name)
            .unwrap();
        assert!(!hover.detail().unwrap().contains("periodic clock;"));
    }
    let event = member(snapshot, "M", "hit");
    assert_eq!(event.kind(), eqiora_api::editor::EditorSymbolKind::Event);
    assert!(event.detail().is_none());
    assert!(
        !workspace
            .assistance(file, source.find("hit").unwrap() as u32, "hit")
            .unwrap()
            .detail()
            .unwrap()
            .contains("periodic clock;")
    );
    assert!(
        member(snapshot, "Borrowed", "memory")
            .detail()
            .unwrap()
            .contains("activation tick (occurrence identity unknown)")
    );
}

#[test]
fn component_clock_details_are_owned_unspecialized_and_use_exact_current_source() {
    let source = "component C(){clock tick=periodic(100[ms],phase=50[ms]);clock peer=periodic(1[s]/10,phase=1[s]/20);relation r{period(tick)=period(tick);}} model M(){clock tick=periodic(2[s]);instance first:C();instance second:C();}";
    let expected = "periodic clock; period 1/10 s; phase 1/20 s; Component-local declaration; occurrence identity unknown";
    let mut service =
        EditorWorkspaceService::new(EditorWorkspaceSnapshot::analyze_standalone(1, source));
    let current = service.current().unwrap();
    assert!(
        current.diagnostics().is_empty(),
        "{:?}",
        current.diagnostics()
    );
    let file = current.files().next().unwrap();
    let snapshot = current.document(file).unwrap();
    let tick = member(snapshot, "C", "tick");
    let peer = member(snapshot, "C", "peer");
    // 100/1000 and 50/1000 seconds reduce to 1/10 and 1/20 independently.
    assert_eq!(tick.detail(), Some(expected));
    assert_eq!(peer.detail(), Some(expected));
    assert_ne!(tick.range(), peer.range());
    let hover = current
        .assistance(file, source.find("tick);").unwrap() as u32, "tick")
        .unwrap();
    assert_eq!(hover.range(), tick.range());
    assert!(hover.detail().unwrap().contains(expected));
    assert!(
        member(snapshot, "M", "tick")
            .detail()
            .unwrap()
            .contains("period 2/1 s")
    );
    service.begin(2).unwrap();
    assert!(service.current().is_none());
    let changed = source.replace("100[ms]", "200[ms]");
    let current = service
        .replace(EditorWorkspaceSnapshot::analyze_standalone(2, changed))
        .unwrap();
    let file = current.files().next().unwrap();
    assert!(
        member(current.document(file).unwrap(), "C", "tick")
            .detail()
            .unwrap()
            .contains("period 1/5 s; phase 1/20 s; Component-local")
    );
    for (index, invalid) in [
        "component C(){clock tick=periodic(0[s]);}",
        "component C(){clock tick=periodic(1[m]);}",
        "component C(){clock tick=periodic(1[s],phase=-1[s]);}",
        "component C(){clock tick=periodic(100[ms]);relation r{",
        "component C(parameter n:integer){clock tick=periodic(100[ms]);variable unknown:array<1,n>;}",
    ].into_iter().enumerate() {
        let version = index as u64 + 3;
        service.begin(version).unwrap();
        let current = service.replace(EditorWorkspaceSnapshot::analyze_standalone(version, invalid)).unwrap();
        let file = current.files().next().unwrap();
        assert!(member(current.document(file).unwrap(), "C", "tick").detail().is_none(), "{invalid}");
    }
}
