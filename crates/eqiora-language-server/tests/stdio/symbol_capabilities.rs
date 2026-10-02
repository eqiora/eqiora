use super::*;

#[test]
fn stdio_document_symbols_follow_client_shape_and_kind_capabilities() {
    let uri = "file:///workspace/main.eqi";
    let source = "// 🧪\r\ndimension Length=m;record Record{x:1}enum Mode{Off,On}model M(){state x:m;clock tick=periodic(1[s]);event hit=crossing(x,direction=falling);relation reset at hit{next(x)=1[m];}}";
    for (capabilities, hierarchical, extended) in [
        (json!({}), false, false),
        (json!({"textDocument":{"documentSymbol":{}}}), false, false),
        (
            json!({"textDocument":{"documentSymbol":{"hierarchicalDocumentSymbolSupport":false,"symbolKind":{}}}}),
            false,
            false,
        ),
        (
            json!({"textDocument":{"documentSymbol":{"hierarchicalDocumentSymbolSupport":true}}}),
            true,
            false,
        ),
        (
            json!({"textDocument":{"documentSymbol":{"hierarchicalDocumentSymbolSupport":true,"symbolKind":{"valueSet":[5]}}}}),
            true,
            true,
        ),
        (
            json!({"textDocument":{"documentSymbol":{"symbolKind":{"valueSet":[]}}}}),
            false,
            true,
        ),
    ] {
        let mut child = Command::new(SERVER)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        for message in [
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":capabilities}}),
            json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
            json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":source}}}),
            json!({"jsonrpc":"2.0","id":2,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":uri}}}),
            json!({"jsonrpc":"2.0","id":3,"method":"shutdown","params":null}),
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
        let diagnostics = messages
            .iter()
            .find(|m| m["method"] == "textDocument/publishDiagnostics")
            .unwrap();
        assert_eq!(diagnostics["params"]["diagnostics"], json!([]));
        let entries = response(&messages, 2)["result"].as_array().unwrap();
        assert_eq!(entries.len(), if hierarchical { 4 } else { 11 });
        for (owner, name, head, legacy_kind, extended_kind) in [
            (None, "Length", "dimension Length", 5, 26),
            (None, "Record", "record Record", 5, 23),
            (Some("Mode"), "Off", "Off,On", 14, 22),
            (Some("M"), "tick", "clock tick", 12, 24),
            (Some("M"), "hit", "event hit", 12, 24),
            (Some("M"), "reset", "relation reset", 12, 25),
        ] {
            let candidates = if hierarchical && owner.is_some() {
                entries
                    .iter()
                    .find(|s| s["name"] == owner.unwrap())
                    .unwrap()["children"]
                    .as_array()
                    .unwrap()
            } else {
                entries
            };
            let entry = candidates.iter().find(|s| s["name"] == name).unwrap();
            assert_eq!(
                entry["kind"],
                if extended { extended_kind } else { legacy_kind }
            );
            if hierarchical {
                assert!(entry.get("location").is_none());
                assert!(entry.get("selectionRange").is_some());
                assert_eq!(entry["range"]["start"], source_position(source, head));
            } else {
                assert_eq!(entry["location"]["uri"], uri);
                assert_eq!(
                    entry["location"]["range"]["start"],
                    source_position(source, head)
                );
                assert_eq!(entry["containerName"], json!(owner));
                assert!(entry.get("children").is_none());
                assert!(entry.get("range").is_none());
                assert!(entry.get("selectionRange").is_none());
                assert!(entry.get("detail").is_none());
            }
        }
    }
}
