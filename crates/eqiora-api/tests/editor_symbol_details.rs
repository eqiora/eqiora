use eqiora_api::editor::{
    EditorSnapshot, EditorSymbol, EditorWorkspaceService, EditorWorkspaceSnapshot,
};
use eqiora_compiler::{CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit};

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
fn outline_details_share_exact_model_types_without_duplicating_hover_details() {
    let source = "model Other(){variable value:s;} model M(){variable value:m;parameter duration:s=1[s];port input:signal input K;clock tick=periodic(1[s]);state clocked:m at tick;}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let snapshot = workspace.document(file).unwrap();
    for (owner, name, expected) in [
        ("Other", "value", "dimension T"),
        ("M", "value", "dimension L"),
        ("M", "duration", "parameter; Real; dimension T"),
        ("M", "input", "signal Input; Real; dimension Θ"),
        (
            "M",
            "clocked",
            "activation tick (occurrence identity unknown)",
        ),
    ] {
        let symbol = member(snapshot, owner, name);
        assert!(
            symbol.detail().unwrap().contains(expected),
            "{:?}",
            symbol.detail()
        );
        let hover = workspace
            .assistance(
                file,
                symbol.range().start()
                    + source[symbol.range().start() as usize..]
                        .find(name)
                        .unwrap() as u32,
                name,
            )
            .unwrap();
        assert_eq!(
            hover
                .detail()
                .unwrap()
                .matches(symbol.detail().unwrap())
                .count(),
            1
        );
    }
    let repeated = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert_eq!(
        snapshot.symbols(),
        repeated.document(file).unwrap().symbols()
    );
}

#[test]
fn outline_distinguishes_channel_axes_and_keeps_binder_boundaries() {
    let source = "component C(){variable value:s;} model M(){domain body=box(0,1,0,1);variable channels:array<vector<m,2>,2> on body;variable matrix:tensor<m,2,2> on body;parameter value:m=1[m];indexset Rows=range(2);relation family[member in Rows]{value=1[m];}}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let snapshot = workspace.document(file).unwrap();
    for (name, rank) in [("channels", 1), ("matrix", 0)] {
        let detail = member(snapshot, "M", name).detail().unwrap();
        for expected in [
            "shape [2, 2]",
            "frame SpatialCartesian",
            "support volume body",
        ] {
            assert!(detail.contains(expected), "{detail}");
        }
        assert!(detail.contains(&format!("array rank {rank}")), "{detail}");
    }
    assert!(
        member(snapshot, "C", "value")
            .detail()
            .unwrap()
            .contains("dimension T")
    );
    let binder_start = source.find("relation family").unwrap() as u32;
    let model = snapshot
        .symbols()
        .iter()
        .find(|symbol| symbol.name() == "M")
        .unwrap();
    assert!(
        model
            .children()
            .iter()
            .filter(|symbol| symbol.detail().is_some())
            .all(|symbol| symbol.range().end() <= binder_start)
    );
    assert!(
        !model
            .children()
            .iter()
            .any(|symbol| symbol.name() == "member")
    );
}

#[test]
fn invalid_and_pending_versions_preserve_outline_without_old_typed_facts() {
    let old = EditorWorkspaceSnapshot::analyze_standalone(1, "model M(){variable value:m;}");
    let mut service = EditorWorkspaceService::new(old.clone());
    service.begin(2).unwrap();
    assert!(service.current().is_none());
    assert!(service.replace(old).is_err());
    for (version, source) in [
        (2, "model M(){variable value:s;relation r{value=1[m];}}"),
        (3, "model M(){variable value:K;relation r{value"),
    ] {
        if version > 2 {
            service.begin(version).unwrap();
        }
        let workspace = service
            .replace(EditorWorkspaceSnapshot::analyze_standalone(version, source))
            .unwrap();
        assert!(!workspace.diagnostics().is_empty());
        let file = workspace.files().next().unwrap();
        assert!(
            member(workspace.document(file).unwrap(), "M", "value")
                .detail()
                .is_none()
        );
    }
    service.begin(4).unwrap();
    let workspace = service
        .replace(EditorWorkspaceSnapshot::analyze_standalone(
            4,
            "model M(){variable value:K;}",
        ))
        .unwrap();
    let file = workspace.files().next().unwrap();
    let detail = member(workspace.document(file).unwrap(), "M", "value")
        .detail()
        .unwrap();
    assert!(detail.contains("dimension Θ") && !detail.contains("dimension L"));
}

#[test]
fn late_cancellation_never_publishes_a_partially_prepared_outline() {
    let owner = CompilationNamespaceId::new(["symbols"]).unwrap();
    let input = ResolvedHierarchyInput::new(
        owner.clone(),
        vec![
            ResolvedSourceUnit::new(
                owner,
                "src/main.eqi",
                "model M(){variable first:m;variable second:s;}",
            )
            .unwrap(),
        ],
        vec![],
    );
    let mut polls = 0;
    let complete =
        EditorWorkspaceSnapshot::analyze_modules_with_cancellation(1, input.clone(), || {
            polls += 1;
            false
        });
    assert!(complete.is_some());
    let mut calls = 0;
    let cancelled = EditorWorkspaceSnapshot::analyze_modules_with_cancellation(2, input, || {
        calls += 1;
        calls == polls - 1
    });
    assert!(cancelled.is_none());
}
