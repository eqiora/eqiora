use super::*;

fn highlight(source: &str, needle: &str, shift: usize, width: u64) -> Value {
    let offset = source.find(needle).unwrap() + shift;
    let before = &source[..offset];
    let line = before.bytes().filter(|byte| *byte == b'\n').count();
    let character = before.rsplit('\n').next().unwrap().encode_utf16().count() as u64;
    json!({"range":{"start":{"line":line,"character":character},"end":{"line":line,"character":character+width}},"kind":1})
}

#[test]
fn stdio_highlights_exact_references_only_in_the_current_document() {
    let main_uri = "file:///workspace/main.eqi";
    let library_uri = "file:///workspace/library.eqi";
    let other_uri = "file:///workspace/other.eqi";
    let main = "// 🧪 value\r\nimport editor.workspace.library as lib;\r\nmodel First(){parameter value:1=2;variable x:1;instance a:lib.Part();instance b:lib.Part();relation r{x=value;a.p=b.p;}}\r\nmodel Second(){parameter value:1=3;variable x:1;relation s{x=value;}}\r\n";
    let library = "// 🧪\r\npublic component Part(output p:1){}";
    let other = "import editor.workspace.library as lib;model Other(){instance c:lib.Part();relation r{c.p=1;}}";
    let moved = format!("// moved\n{main}");
    let open = |uri, text| json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":text}}});
    let change = |uri, version, text| json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":version},"contentChanges":[{"text":text}]}});
    let query = |id, uri, source, needle, shift| json!({"jsonrpc":"2.0","id":id,"method":"textDocument/documentHighlight","params":{"textDocument":{"uri":uri},"position":highlight(source,needle,shift,1)["range"]["start"]}});
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for message in [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{},"workspaceFolders":[{"uri":"file:///workspace","name":"workspace"}]}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        open(library_uri, library),
        open(other_uri, other),
        open(main_uri, main),
        query(2, main_uri, main, "x=value;", 2),
        query(3, main_uri, main, "value:1=3", 0),
        query(4, main_uri, main, "a.p", 2),
        query(5, library_uri, library, "p:1", 0),
        query(6, main_uri, main, "lib.Part", 4),
        query(7, library_uri, library, "Part(output", 0),
        query(8, main_uri, main, "// 🧪 value", "// 🧪 ".len()),
        query(9, main_uri, main, "value:1=2", "value:".len()),
        query(10, main_uri, main, "a.p", 0),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":main_uri,"version":2},"contentChanges":[{"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":0}},"text":"// moved\n"}]}}),
        change(main_uri, 1, "model Stale() {}"),
        query(11, main_uri, &moved, "a.p", 2),
        change(library_uri, 2, "public component Part(output p:1){"),
        query(12, main_uri, &moved, "a.p", 2),
        change(library_uri, 3, library),
        query(13, main_uri, &moved, "a.p", 2),
        json!({"jsonrpc":"2.0","id":14,"method":"textDocument/documentHighlight","params":{"textDocument":42}}),
        json!({"jsonrpc":"2.0","id":15,"method":"shutdown","params":null}),
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
    assert_eq!(
        response(&messages, 1)["result"]["capabilities"]["documentHighlightProvider"],
        true
    );
    assert!(
        messages
            .iter()
            .any(|m| m["method"] == "textDocument/publishDiagnostics"
                && m["params"]["uri"] == main_uri
                && m["params"]["version"] == 1
                && m["params"]["diagnostics"] == json!([]))
    );
    assert_eq!(
        response(&messages, 2)["result"],
        json!([
            highlight(main, "value:1=2", 0, 5),
            highlight(main, "x=value;", 2, 5),
        ])
    );
    assert_eq!(
        response(&messages, 3)["result"],
        json!([
            highlight(main, "value:1=3", 0, 5),
            highlight(main, "relation s{x=value;", "relation s{x=".len(), 5),
        ])
    );
    assert_eq!(
        response(&messages, 4)["result"],
        json!([highlight(main, "a.p", 2, 1), highlight(main, "b.p", 2, 1),])
    );
    assert_eq!(
        response(&messages, 5)["result"],
        json!([highlight(library, "p:1", 0, 1)])
    );
    assert_eq!(
        response(&messages, 6)["result"],
        json!([
            highlight(main, "a:lib.Part", "a:".len(), 8),
            highlight(main, "b:lib.Part", "b:".len(), 8),
        ])
    );
    assert_eq!(
        response(&messages, 7)["result"],
        json!([highlight(library, "Part(output", 0, 4)])
    );
    for id in [8, 9, 10, 12] {
        assert_eq!(response(&messages, id)["result"], json!([]), "request {id}");
    }
    for id in [11, 13] {
        assert_eq!(
            response(&messages, id)["result"],
            json!([
                highlight(&moved, "a.p", 2, 1),
                highlight(&moved, "b.p", 2, 1),
            ])
        );
    }
    assert_eq!(response(&messages, 14)["error"]["code"], -32602);
}
