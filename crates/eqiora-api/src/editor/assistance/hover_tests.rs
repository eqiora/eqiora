use super::*;
use eqiora_compiler::{CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit};

fn hover(marked: &str) -> EditorSymbol {
    let offset = marked.find('|').unwrap() as u32;
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, marked.replacen('|', "", 1));
    let file = workspace.files().next().unwrap();
    let (name, _) = workspace.document(file).unwrap().name_at(offset).unwrap();
    let symbol = workspace.assistance(file, offset, &name).unwrap();
    let local = workspace
        .document(file)
        .unwrap()
        .assistance(offset, &name)
        .unwrap();
    // A contextless document retains the same declaration/type facts; exact
    // workspace origin is added only by the workspace query.
    assert!(
        symbol
            .detail()
            .unwrap()
            .starts_with(local.detail().unwrap())
    );
    let mut same_facts = symbol.clone();
    same_facts.detail = local.detail.clone();
    assert_eq!(same_facts, local);
    symbol
}

#[test]
fn model_hover_projects_known_fields_parameters_ports_and_nominal_types() {
    for (marked, facts) in [
        (
            "model M(){variable va|lue:array<m,2>;}",
            vec![
                "Variable",
                "dimension L",
                "shape [2]",
                "continuous",
                "no spatial support",
            ],
        ),
        (
            "model M(){domain body=box(0,1);variable value:K on body; relation r{va|lue=1[K];}}",
            vec!["dimension Θ", "support volume body", "axes 1"],
        ),
        (
            "model M(){clock tick=periodic(1[s]);state va|lue:m at tick;}",
            vec!["State", "activation tick", "occurrence identity unknown"],
        ),
        (
            "model M(){parameter va|lue:s=1[s];}",
            vec!["dimension T", "static"],
        ),
        (
            "enum Mode {First,Second} model M(){parameter va|lue:Mode=Mode.First;}",
            vec!["nominal enum"],
        ),
        (
            "model M(){port va|lue:signal input m;}",
            vec!["signal Input", "dimension L"],
        ),
        (
            "model M(){domain electrical=scalar_physical(across potential:V,through flow:A);port va|lue:electrical;}",
            vec!["physical; across", "through", "nominal"],
        ),
        (
            "connector Mechanical {trace velocity:m/s;flux traction:Pa;shape spatial_vector;frame spatial;pairing euclidean_boundary_duality;orientation parent_outward;} component Wall(support body:volume(ambient_dimension=2),support face:boundary(parent=body),port value:Mechanical over face){} model M(){domain body=box(0,1,0,1);domain face=boundary(body,axis=0,side=upper);instance child:Wall(body=body,face=face);relation r{child.va|lue=child.value;}}",
            vec![
                "field-physical; trace",
                "flux",
                "shape [2]",
                "support boundary face",
                "parent body",
            ],
        ),
    ] {
        let symbol = hover(marked);
        let detail = symbol.detail().unwrap();
        for fact in facts {
            assert!(detail.contains(fact), "missing {fact:?} in {detail:?}");
        }
    }
}

#[test]
fn hover_distinguishes_channel_array_axes_from_equal_shaped_spatial_axes() {
    let source = "model M(){domain body=box(0,1,0,1);variable channels:array<vector<m,2>,2> on body;variable matrix:tensor<m,2,2> on body;}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    for (name, rank) in [("channels", 1), ("matrix", 0)] {
        let symbol = workspace
            .assistance(file, source.find(name).unwrap() as u32, name)
            .unwrap();
        let detail = symbol.detail().unwrap();
        assert!(detail.contains("shape [2, 2]"), "{detail}");
        assert!(detail.contains("frame SpatialCartesian"), "{detail}");
        assert!(detail.contains(&format!("array rank {rank}")), "{detail}");
    }
}

#[test]
fn hover_uses_current_model_scope_and_source_version() {
    let marked =
        "model Other(){variable value:s;} model M(){variable value:m; relation r{va|lue=1[m];}}";
    assert!(hover(marked).detail().unwrap().contains("dimension L"));
    let changed = marked.replace("value:m", "value:K").replace("1[m]", "1[K]");
    let detail = hover(&changed).detail().unwrap().to_owned();
    assert!(detail.contains("dimension Θ") && !detail.contains("dimension L"));
    let incomplete = "model M(){variable value:m; relation r{va|lue";
    assert!(!hover(incomplete).detail().unwrap().contains("//"));
    let component = "component C(){variable va|lue:m;} model M(){}";
    assert!(!hover(component).detail().unwrap().contains("//"));

    let old = EditorWorkspaceSnapshot::analyze_standalone(1, marked.replace('|', ""));
    let mut service = crate::editor::EditorWorkspaceService::new(old.clone());
    service.begin(2).unwrap();
    assert!(service.current().is_none());
    assert!(service.replace(old).is_err());
    let current = service
        .replace(EditorWorkspaceSnapshot::analyze_standalone(
            2,
            changed.replace('|', ""),
        ))
        .unwrap();
    let file = current.files().next().unwrap();
    assert!(
        current
            .assistance(file, changed.find('|').unwrap() as u32, "value")
            .unwrap()
            .detail()
            .unwrap()
            .contains("dimension Θ")
    );
}

#[test]
fn nested_binders_never_receive_outer_declaration_metadata() {
    for marked in [
        "model M(){parameter value:m=1[m];indexset Rows=range(2);relation r{sum(ordinal(va|lue),over=(value in Rows))=1;}}",
        "model M(){parameter value:m=1[m];indexset Rows=range(2);relation r[value in Rows]{ordinal(va|lue)=0;}}",
        "component C(parameter length:integer){} model M(){parameter value:m=1[m];indexset Rows=range(2);instance child[value in Rows]:C(length=ordinal(va|lue));}",
    ] {
        assert!(!hover(marked).detail().unwrap().contains("//"));
        let declaration =
            marked
                .replace('|', "")
                .replacen("parameter value", "parameter va|lue", 1);
        assert!(
            hover(&declaration)
                .detail()
                .unwrap()
                .contains("dimension L")
        );
    }
}

#[test]
fn imported_child_hover_keeps_actual_support_and_activation_with_declaration_identity() {
    let owner = CompilationNamespaceId::new(["sample"]).unwrap();
    let marked = "import sample.lib as lib; model M(){domain body=box(0,1);clock tick=periodic(1[s]);instance child:lib.Sensor(region=body,source_clock=tick);relation r{child.va|lue=1[K];}}";
    let root =
        ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", marked.replace('|', "")).unwrap();
    let file = root.diagnostic_file();
    let library = "public component Sensor(support region:volume(ambient_dimension=1),clock source_clock:periodic,output value:K on region at source_clock) {}";
    let workspace = EditorWorkspaceSnapshot::analyze_modules(
        1,
        ResolvedHierarchyInput::new(
            owner.clone(),
            vec![
                root,
                ResolvedSourceUnit::new(owner, "src/lib.eqi", library).unwrap(),
            ],
            vec![],
        ),
    );
    let offset = marked.find('|').unwrap() as u32;
    let symbol = workspace.assistance(&file, offset, "child.value").unwrap();
    let detail = symbol.detail().unwrap();
    for fact in [
        "signal Output",
        "dimension Θ",
        "support volume body",
        "activation tick",
    ] {
        assert!(detail.contains(fact), "missing {fact:?} in {detail:?}");
    }
}
