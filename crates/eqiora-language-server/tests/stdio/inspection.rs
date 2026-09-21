use super::*;

#[test]
fn model_inspection_tracks_selected_unsaved_models_and_rejects_invalid_sources() {
    let source = "model A() { variable x: 1; relation balance { x = 1; } }\nmodel B() { variable y: 1; relation balance { y = 2; } }\n";
    let uri = "file:///workspace/decay.eqi";
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let messages = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{},"workspaceFolders":[{"uri":"file:///workspace","name":"workspace"}]}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":source}}}),
        json!({"jsonrpc":"2.0","id":2,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"model":"B","fingerprint":true}}),
        json!({"jsonrpc":"2.0","id":3,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"model":"Missing"}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":2},"contentChanges":[{"text":source.replace("y = 2", "y = 3")}]}}),
        json!({"jsonrpc":"2.0","id":4,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"model":"B","fingerprint":true}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":3},"contentChanges":[{"text":"model Broken() { nonsense; }"}]}}),
        json!({"jsonrpc":"2.0","id":5,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri}}}),
        json!({"jsonrpc":"2.0","id":6,"method":"shutdown","params":null}),
        json!({"jsonrpc":"2.0","method":"exit","params":null}),
    ];
    let mut stdin = child.stdin.take().unwrap();
    for message in messages {
        write_packet(&mut stdin, &message);
    }
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let messages = parse_packets(&output.stdout);
    assert_eq!(
        response(&messages, 1)["result"]["capabilities"]["experimental"]["eqioraInspection"],
        1
    );
    let before = &response(&messages, 2)["result"];
    assert_eq!(before["model"], "B");
    assert_eq!(before["errors"], json!([]));
    assert_eq!(before["version"], 1);
    assert!(
        before["equations"][0]["plain"]
            .as_str()
            .unwrap()
            .contains("2")
    );
    assert!(before["fingerprint"].is_string());
    assert!(response(&messages, 3)["error"].is_object());
    let after = &response(&messages, 4)["result"];
    assert_eq!(after["version"], 2);
    assert!(
        after["equations"][0]["plain"]
            .as_str()
            .unwrap()
            .contains("3")
    );
    assert_ne!(before["fingerprint"], after["fingerprint"]);
    let invalid = response(&messages, 5);
    assert!(invalid["error"].is_object() || invalid["result"]["equations"] == json!([]));
}

#[test]
fn plan_inspection_validates_exact_artifacts_and_tracks_selected_model_edits() {
    use eqiora::{
        api::ModelDocument,
        artifact::{ModelDecoderLimits, ModelEnvelope},
        compiler::{CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit},
        kernel::KernelNode,
    };
    use eqiora_numerics::{CommonTsitouras45, CommonTsitourasTolerance, resolve_common_ode_plan};
    let source = "model Decay() { state x: 1; initial { x = 1; } parameter rate: 1 / s = 1; relation flow { derivative(x) + rate * x = 0; } }";
    let owner = CompilationNamespaceId::new(["editor.workspace"]).unwrap();
    let unit = ResolvedSourceUnit::new(owner.clone(), "src/decay.eqi", source).unwrap();
    let input =
        ResolvedHierarchyInput::with_root_module(owner, ["decay"], vec![unit], vec![]).unwrap();
    let model = ModelDocument::compile_modules(input, "Decay", &[]).unwrap();
    let envelope = ModelEnvelope::from_json(
        &model.canonical_json().unwrap(),
        ModelDecoderLimits::default(),
    )
    .unwrap();
    let field = model
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Field(field) => Some(field.id()),
            _ => None,
        })
        .unwrap();
    let plan = resolve_common_ode_plan(
        &envelope,
        model.program(),
        CommonTsitouras45::new(
            0.01,
            1e-6,
            vec![CommonTsitourasTolerance::new(field, 1e-9).unwrap()],
        )
        .unwrap(),
        eqiora::backends::diffsol::DIFFSOL_TIME_BACKEND,
    )
    .unwrap();
    let bytes = String::from_utf8(plan.to_bytes().unwrap()).unwrap();
    let uri = "file:///workspace/decay.eqi";
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let messages = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{},"workspaceFolders":[{"uri":"file:///workspace","name":"workspace"}]}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":source}}}),
        json!({"jsonrpc":"2.0","id":2,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"model":"Decay","plan":bytes}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":2},"contentChanges":[{"text":source.replace("s = 1", "s = 2")}]}}),
        json!({"jsonrpc":"2.0","id":3,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"model":"Decay","plan":bytes}}),
        json!({"jsonrpc":"2.0","id":4,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"plan":"{}"}}),
        json!({"jsonrpc":"2.0","id":5,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"plan":" ".repeat(2 * 1024 * 1024 + 1)}}),
        json!({"jsonrpc":"2.0","id":6,"method":"shutdown","params":null}),
        json!({"jsonrpc":"2.0","method":"exit","params":null}),
    ];
    let mut stdin = child.stdin.take().unwrap();
    // Read concurrently because the bounded artifact request can exceed a pipe buffer.
    let writer = std::thread::spawn(move || {
        for message in messages {
            write_packet(&mut stdin, &message);
        }
    });
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let messages = parse_packets(&output.stdout);
    assert_eq!(
        response(&messages, 1)["result"]["capabilities"]["experimental"]["eqioraPlanInspection"],
        1
    );
    let accepted = &response(&messages, 2)["result"];
    assert_eq!(accepted["version"], 1);
    assert_eq!(accepted["plan"]["identity"], plan.identity());
    assert_eq!(accepted["plan"]["matchesSelectedModel"], true);
    let edited = &response(&messages, 3)["result"];
    assert_eq!(edited["version"], 2);
    assert_eq!(edited["plan"]["identity"], plan.identity());
    assert_eq!(edited["plan"]["matchesSelectedModel"], false);
    assert_ne!(
        accepted["plan"]["selectedModelDigest"],
        edited["plan"]["selectedModelDigest"]
    );
    assert!(
        response(&messages, 4)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Cannot validate")
    );
    assert!(
        response(&messages, 5)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("2 MiB")
    );
}
