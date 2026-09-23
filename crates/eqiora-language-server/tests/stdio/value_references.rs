use super::*;

fn location(uri: &str, source: &str, needle: &str, shift: usize, width: u64) -> Value {
    let offset = source.find(needle).unwrap() + shift;
    let before = &source[..offset];
    let start = json!({
        "line": before.bytes().filter(|byte| *byte == b'\n').count(),
        "character": before.rsplit('\n').next().unwrap().encode_utf16().count(),
    });
    json!({"uri":uri,"range":{"start":start,"end":{"line":start["line"],"character":start["character"].as_u64().unwrap()+width}}})
}

fn ordered(mut values: Vec<Value>) -> Value {
    values.sort_by_key(|value| {
        (
            value["uri"].as_str().unwrap().to_owned(),
            value["range"]["start"]["line"].as_u64().unwrap(),
            value["range"]["start"]["character"].as_u64().unwrap(),
        )
    });
    json!(values)
}

#[test]
fn stdio_port_declaration_references_keep_source_identity_and_current_overlays() {
    let main_uri = "file:///workspace/main.eqi";
    let library_uri = "file:///workspace/library.eqi";
    let right_uri = "file:///workspace/right.eqi";
    let other_uri = "file:///workspace/other.eqi";
    let main = "// 🧪\r\nimport editor.workspace.library as lib;import editor.workspace.right as other;model First(){port unused:signal input 1;instance a:lib.Part();instance b:lib.Part();instance foreign:other.Part();relation r{(a . p)=b.p;foreign.p=1;}}model Second(){instance c:lib.Part();relation r{c.p=1;}}";
    let library =
        "// 🧪\r\npublic component Part(output p @{q}:1){}public component Never(output p:1){}";
    let other = "import editor.workspace.library as lib;model Third(){instance d:lib.Part();relation r{d.p=1;}}";
    let changed = main.replace("relation r{(a . p)=b.p;", "relation r{\r\n(a . p)=a.p;");
    let moved = format!("// moved\r\n{library}");
    let open = |uri, source| json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":source}}});
    let change = |uri, version, text| json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":version},"contentChanges":[{"text":text}]}});
    let query = |id, uri: &str, source: &str, needle: &str, shift, include| json!({"jsonrpc":"2.0","id":id,"method":"textDocument/references","params":{"textDocument":{"uri":uri},"position":location(uri, source, needle, shift, 1)["range"]["start"],"context":{"includeDeclaration":include}}});
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for message in [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{"workspace":{"workspaceFolders":true}},"workspaceFolders":[{"uri":"file:///workspace","name":"workspace"}]}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        open(library_uri, library),
        open(right_uri, library),
        open(other_uri, other),
        open(main_uri, main),
        query(2, main_uri, main, "a . p", 4, false),
        query(3, library_uri, library, "p @{", 0, true),
        query(4, right_uri, library, "p @{", 0, false),
        query(5, main_uri, main, "unused:", 0, false),
        query(6, main_uri, main, "unused:", 0, true),
        query(7, main_uri, main, "a . p", 0, true),
        query(
            8,
            library_uri,
            library,
            "Never(output p",
            "Never(output ".len(),
            true,
        ),
        change(main_uri, 2, changed.as_str()),
        change(main_uri, 1, main),
        change(library_uri, 2, moved.as_str()),
        query(9, library_uri, &moved, "p @{", 0, true),
        change(library_uri, 3, "public component Part(output p:1){"),
        query(10, main_uri, &changed, "a . p", 4, false),
        change(library_uri, 4, moved.as_str()),
        query(11, main_uri, &changed, "a . p", 4, false),
        json!({"jsonrpc":"2.0","id":12,"method":"shutdown","params":null}),
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
    // LSP does not promise URI sorting; compare exact location multisets.
    // API tests separately bind the deterministic compiler source-label order.
    let locations = |id| {
        ordered(
            response(&messages, id)["result"]
                .as_array()
                .unwrap()
                .clone(),
        )
    };
    assert!(messages.iter().any(
        |message| message["method"] == "textDocument/publishDiagnostics"
            && message["params"]["uri"] == main_uri
            && message["params"]["version"] == 1
            && message["params"]["diagnostics"] == json!([])
    ));
    // a.p/b.p/c.p/d.p are four source uses of one Port declaration. This
    // deliberately makes no assertion about their physical occurrence IDs.
    let initial = vec![
        location(main_uri, main, "a . p", 4, 1),
        location(main_uri, main, "b.p", 2, 1),
        location(main_uri, main, "c.p", 2, 1),
        location(other_uri, other, "d.p", 2, 1),
    ];
    assert_eq!(locations(2), ordered(initial.clone()));
    let mut included = initial;
    included.push(location(library_uri, library, "p @{", 0, 1));
    assert_eq!(locations(3), ordered(included));
    assert_eq!(
        response(&messages, 4)["result"],
        json!([location(main_uri, main, "foreign.p", 8, 1)])
    );
    assert_eq!(
        response(&messages, 6)["result"],
        json!([location(main_uri, main, "unused:", 0, 6)])
    );
    for id in [5, 7, 10] {
        assert_eq!(response(&messages, id)["result"], json!([]));
    }
    assert_eq!(
        response(&messages, 8)["result"],
        json!([location(
            library_uri,
            library,
            "Never(output p",
            "Never(output ".len(),
            1
        )])
    );
    let revised = vec![
        location(main_uri, &changed, "a . p", 4, 1),
        location(main_uri, &changed, "a.p;", 2, 1),
        location(main_uri, &changed, "c.p", 2, 1),
        location(other_uri, other, "d.p", 2, 1),
    ];
    assert_eq!(locations(11), ordered(revised.clone()));
    let mut included = revised;
    included.push(location(library_uri, &moved, "p @{", 0, 1));
    assert_eq!(locations(9), ordered(included));
}
