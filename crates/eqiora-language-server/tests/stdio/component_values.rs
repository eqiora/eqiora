use super::*;

#[test]
fn stdio_component_values_follow_current_types_and_source_locations() {
    let uri = "file:///workspace/main.eqi";
    let source = "// 🧪\r\ncomponent Other(){variable value:K;} component C(){variable value @{v}:m;relation r{value=1[m];}}";
    let changed = format!(
        "// moved\r\n{}",
        source.replace(":m", ":s").replace("[m]", "[s]")
    );
    let broken = "component C(){variable value:m;relation r{value";
    let query = |id, method, text: &str, needle| json!({"jsonrpc":"2.0","id":id,"method":method,"params":{"textDocument":{"uri":uri},"position":source_position(text,needle),"context":{"includeDeclaration":true}}});
    let change = |version, text: &str| json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":version},"contentChanges":[{"text":text}]}});
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
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":source}}}),
        query(2, "textDocument/hover", source, "value=1"),
        query(3, "textDocument/definition", source, "value=1"),
        query(4, "textDocument/references", source, "value @{v}"),
        change(2, &changed),
        change(1, source),
        query(5, "textDocument/hover", &changed, "value=1"),
        query(6, "textDocument/definition", &changed, "value=1"),
        query(7, "textDocument/references", &changed, "value @{v}"),
        change(3, broken),
        query(8, "textDocument/definition", broken, "value:m"),
        query(9, "textDocument/references", broken, "value:m"),
        json!({"jsonrpc":"2.0","id":10,"method":"shutdown","params":null}),
        json!({"jsonrpc":"2.0","method":"exit","params":null}),
    ] {
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
    for (id, dimension) in [(2, "dimension L"), (5, "dimension T")] {
        assert!(
            response(&messages, id)["result"]["contents"]["value"]
                .as_str()
                .unwrap()
                .contains(dimension)
        );
    }
    let location = |text, needle| {
        let start = source_position(text, needle);
        json!({"uri":uri,"range":{"start":start,"end":{"line":start["line"],"character":start["character"].as_u64().unwrap()+5}}})
    };
    for (definition, references, text) in [(3, 4, source), (6, 7, changed.as_str())] {
        assert_eq!(
            response(&messages, definition)["result"],
            location(text, "value @{v}")
        );
        assert_eq!(
            response(&messages, references)["result"],
            json!([location(text, "value @{v}"), location(text, "value=1")])
        );
    }
    assert!(response(&messages, 8)["result"].is_null());
    assert_eq!(response(&messages, 9)["result"], json!([]));
}
