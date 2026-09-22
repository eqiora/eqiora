use super::*;

#[test]
fn stdio_authored_notation_respects_current_declarations_and_proven_references() {
    let uri = "file:///workspace/main.eqi";
    let source = "model Other(){variable value @{q}:1;}model M(){variable value @{\\mathbf{x_i}}:1;relation r{value=1;}}";
    let changed = source.replace(r"@{\mathbf{x_i}}", r"@{\alpha_2}");
    let cases = [
        (
            "model M(){variable variable @{v}:1;relation r{",
            "variable @{",
            Some("v"),
        ),
        (
            "model M(){variable variable @{v}:1;relation r{",
            "variable variable",
            None,
        ),
        (
            "model M(){parameter parameter @{p}:1=1;relation r{",
            "parameter @{",
            Some("p"),
        ),
        (source, "value=1", Some("x_{i}")),
        (changed.as_str(), "value=1", Some("alpha_{2}")),
        (
            "model M(){variable value @{\\hat{q}^{2}}:unknown_type;}",
            "value @",
            Some("hat(q)^{2}"),
        ),
        (
            "model M(){variable value @{x_i}:1;relation r{value =",
            "value @",
            Some("x_{i}"),
        ),
        (
            "model M(){variable value @{x_i}:1;relation r{value =",
            "value =",
            None,
        ),
        (
            "model M(){parameter value @{q}:integer=1;indexset Rows=range(2);relation r[row in Rows]{value=ordinal(row);}}",
            "value=ordinal",
            None,
        ),
        (
            "model Other(){variable value @{q}:1;}model M(){variable value:1;relation r{value=1;}}",
            "value=1",
            None,
        ),
        (
            "model M(){variable value @{\\input{secret}}:1;}",
            "value @",
            None,
        ),
    ];
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for message in [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":""}}}),
    ] {
        write_packet(&mut stdin, &message);
    }
    for (index, (text, occurrence, _)) in cases.iter().enumerate() {
        write_packet(
            &mut stdin,
            &json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":index+2},"contentChanges":[{"text":text}]}}),
        );
        if index == 4 {
            write_packet(
                &mut stdin,
                &json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":1},"contentChanges":[{"text":source}]}}),
            );
        }
        write_packet(
            &mut stdin,
            &json!({"jsonrpc":"2.0","id":index+2,"method":"textDocument/hover","params":{"textDocument":{"uri":uri},"position":source_position(text,occurrence)}}),
        );
    }
    write_packet(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":30,"method":"shutdown","params":null}),
    );
    write_packet(
        &mut stdin,
        &json!({"jsonrpc":"2.0","method":"exit","params":null}),
    );
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let messages = parse_packets(&output.stdout);
    for (index, (_, _, expected)) in cases.iter().enumerate() {
        let result = &response(&messages, (index + 2) as i64)["result"];
        if let Some(label) = expected {
            assert_eq!(result["contents"]["kind"], "markdown");
            let rendered = result["contents"]["value"].as_str().unwrap();
            assert!(
                rendered.contains(&format!("Notation: `{label}`")),
                "{result}"
            );
        } else {
            assert!(!result.to_string().contains("Notation:"), "{result}");
        }
    }
}
