use super::*;

#[test]
fn stdio_clock_assistance_uses_exact_current_declarations() {
    let uri = "file:///workspace/main.eqi";
    let source = "component C(){clock beat=periodic(100[ms],phase=50[ms]);relation r{period(beat)=period(beat);}} model Other(){clock tick=periodic(2[s]);} model M(){clock tick=periodic(100[ms],phase=50[ms]);clock peer=periodic(1[s]/10,phase=1[s]/20);state memory:1 at tick;relation schedule{period(tick)=period(tick);}}";
    let changed = source.replace("100[ms]", "200[ms]");
    let invalid = "model M(){clock tick=periodic(0[s]);}";
    let incomplete = "model M(){clock tick=periodic(100[ms]);relation r{";
    let unsupported = "component C(clock tick:periodic){} model Borrowed(clock tick:periodic){} model M(){state x:m;event hit=crossing(x,direction=falling);}";
    let outline = |id| json!({"jsonrpc":"2.0","id":id,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":uri}}});
    let hover = |id, text: &str, needle: &str| json!({"jsonrpc":"2.0","id":id,"method":"textDocument/hover","params":{"textDocument":{"uri":uri},"position":source_position(text, needle)}});
    let definition = |id, text: &str, needle: &str| json!({"jsonrpc":"2.0","id":id,"method":"textDocument/definition","params":{"textDocument":{"uri":uri},"position":source_position(text, needle)}});
    let references = |id, text: &str, needle: &str, include| json!({"jsonrpc":"2.0","id":id,"method":"textDocument/references","params":{"textDocument":{"uri":uri},"position":source_position(text, needle),"context":{"includeDeclaration":include}}});
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
        outline(2),
        hover(3, source, "tick=periodic(100"),
        hover(4, source, "tick);"),
        hover(17, source, "tick;"),
        hover(18, source, "beat=periodic"),
        hover(19, source, "beat);"),
        definition(21, source, "beat);"),
        references(22, source, "beat);", true),
        references(23, source, "peer=", false),
        references(31, source, "tick;", true),
        definition(32, source, "tick;"),
        change(2, &changed),
        change(1, source),
        outline(5),
        hover(6, &changed, "tick);"),
        hover(20, &changed, "beat);"),
        definition(24, &changed, "beat);"),
        references(25, &changed, "beat=", false),
        change(3, invalid),
        outline(7),
        hover(8, invalid, "tick"),
        definition(26, invalid, "tick"),
        references(27, invalid, "tick", true),
        change(4, incomplete),
        outline(9),
        hover(10, incomplete, "tick"),
        change(5, unsupported),
        outline(11),
        hover(12, unsupported, "tick:periodic"),
        definition(28, unsupported, "tick:periodic"),
        references(29, unsupported, "tick:periodic", true),
        hover(16, unsupported, "hit"),
        change(6, source),
        outline(13),
        hover(14, source, "tick);"),
        definition(30, source, "beat);"),
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
    let initial_diagnostics = messages
        .iter()
        .find(|message| {
            message["method"] == "textDocument/publishDiagnostics"
                && message["params"]["version"] == 1
        })
        .unwrap();
    assert_eq!(initial_diagnostics["params"]["diagnostics"], json!([]));
    let detail = |id, owner: &str, name: &str| {
        response(&messages, id)["result"]
            .as_array()
            .unwrap()
            .iter()
            .find(|value| value["name"] == owner)
            .unwrap()["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|value| value["name"] == name)
            .unwrap()["detail"]
            .as_str()
            .unwrap()
    };
    let location = |text: &str, needle: &str| {
        let start = source_position(text, needle);
        let end = json!({"line":start["line"],"character":start["character"].as_u64().unwrap()+4});
        json!({"uri":uri,"range":{"start":start,"end":end}})
    };
    for id in [21, 30] {
        assert_eq!(response(&messages, id)["result"], location(source, "beat="));
    }
    assert_eq!(
        response(&messages, 24)["result"],
        location(&changed, "beat=")
    );
    assert_eq!(
        response(&messages, 22)["result"],
        json!([
            location(source, "beat="),
            location(source, "beat)="),
            location(source, "beat);")
        ])
    );
    assert_eq!(
        response(&messages, 25)["result"],
        json!([location(&changed, "beat)="), location(&changed, "beat);")])
    );
    for id in [23, 27, 29, 31] {
        assert_eq!(response(&messages, id)["result"], json!([]));
    }
    for id in [26, 28, 32] {
        assert!(response(&messages, id)["result"].is_null());
    }
    let exact = "periodic clock; period 1/10 s; phase 1/20 s; Model-local declaration; occurrence identity unknown";
    assert_eq!(detail(2, "M", "tick"), exact);
    assert_eq!(detail(2, "M", "peer"), exact);
    assert_eq!(detail(13, "M", "tick"), exact);
    assert!(detail(2, "Other", "tick").contains("period 2/1 s; phase 0/1 s"));
    assert!(detail(2, "M", "memory").contains("activation tick (occurrence identity unknown)"));
    for id in [3, 4, 14] {
        let text = response(&messages, id)["result"]["contents"]["value"]
            .as_str()
            .unwrap();
        assert_eq!(text.matches(exact).count(), 1, "{text}");
    }
    let component_exact = "periodic clock; period 1/10 s; phase 1/20 s; Component-local declaration; occurrence identity unknown";
    assert_eq!(detail(2, "C", "beat"), component_exact);
    assert_eq!(detail(13, "C", "beat"), component_exact);
    for id in [18, 19] {
        assert_eq!(
            response(&messages, id)["result"]["contents"]["value"]
                .as_str()
                .unwrap()
                .matches(component_exact)
                .count(),
            1
        );
    }
    let component_revised = "periodic clock; period 1/5 s; phase 1/20 s; Component-local declaration; occurrence identity unknown";
    assert_eq!(detail(5, "C", "beat"), component_revised);
    assert!(
        response(&messages, 20)["result"]["contents"]["value"]
            .as_str()
            .unwrap()
            .contains(component_revised)
    );
    let revised = "periodic clock; period 1/5 s; phase 1/20 s; Model-local declaration; occurrence identity unknown";
    assert_eq!(detail(5, "M", "tick"), revised);
    assert!(
        response(&messages, 6)["result"]["contents"]["value"]
            .as_str()
            .unwrap()
            .contains(revised)
    );
    for id in [7, 9] {
        assert_eq!(detail(id, "M", "tick"), "Clock");
    }
    for (owner, name, kind) in [("C", "tick", "Clock"), ("Borrowed", "tick", "Clock")] {
        assert_eq!(detail(11, owner, name), kind);
    }
    assert!(
        response(&messages, 11)["result"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|owner| owner["children"].as_array().unwrap())
            .all(|symbol| symbol["name"] != "hit")
    );
    assert!(response(&messages, 16)["result"].is_null());
    assert!(response(&messages, 17)["result"].is_null());
    for id in [8, 10, 12] {
        assert!(
            !response(&messages, id)["result"]["contents"]["value"]
                .as_str()
                .unwrap()
                .contains("periodic clock;")
        );
    }
}
