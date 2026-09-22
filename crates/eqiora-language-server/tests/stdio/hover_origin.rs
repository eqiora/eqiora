use super::*;

#[test]
fn exact_package_hover_origins_distinguish_identical_exports_and_follow_valid_versions() {
    let fixture = TestDirectory::create("hover origins");
    let library = "public component Part(output value:1){}";
    let main = "import org.example.Left.main as left;import org.example.Right.main as right;model M(){instance a:left.Part();instance b:right.Part();relation r{a.value=b.value;}}";
    let mut dependencies = Vec::new();
    let mut expected = Vec::new();
    let mut manifest = "[package]\nname = \"org.example.Root\"\nversion = \"1.0.0\"\nsource = \"root/src\"\nentry = \"main\"\n".to_owned();
    for (name, folder) in [("org.example.Left", "left"), ("org.example.Right", "right")] {
        let sources = author_sources(name, library, vec![]);
        let release = prepare_package_release_v1(sources.clone(), &[]).unwrap();
        let identity = release.package_identity().unwrap();
        expected.push(format!(
            "Origin namespace: {:?}",
            [
                identity.name.as_str(),
                identity.version.as_str(),
                &identity.semantic_digest.to_hex()
            ]
        ));
        dependencies.push(PackageDependencyV1::new(identity));
        let directory = fixture.0.join(folder);
        write_package(&directory, &sources);
        fs::write(
            directory.join("eqiora.toml"),
            format!("[package]\nname = {name:?}\nversion = \"1.0.0\"\nentry = \"main\"\n"),
        )
        .unwrap();
        manifest.push_str(&format!(
            "\n[dependencies.{name:?}]\nversion = \"1.0.0\"\nsources = [{{ path = {folder:?} }}]\n"
        ));
    }
    write_package(
        &fixture.0.join("root"),
        &author_sources("org.example.Root", main, dependencies),
    );
    fs::write(fixture.0.join("eqiora.toml"), manifest).unwrap();
    let uri = file_uri(&fixture.0.join("root").join(SOURCE_PATH));
    let workspace_uri = file_uri(&fixture.0);
    let renamed = main
        .replace("as left;", "as renamed;")
        .replace("left.Part", "renamed.Part");
    let invalid = renamed.replace("a.value=b.value", "a.value=1[m]");
    let request = |id, source: &str, needle: &str, shift: u64| {
        let mut position = source_position(source, needle);
        position["character"] = json!(position["character"].as_u64().unwrap() + shift);
        json!({"jsonrpc":"2.0","id":id,"method":"textDocument/hover","params":{"textDocument":{"uri":uri},"position":position}})
    };
    let change = |version, text: &str| json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":version},"contentChanges":[{"text":text}]}});
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for message in [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{},"workspaceFolders":[{"uri":workspace_uri,"name":"origins"}]}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":main}}}),
        request(2, main, "left.Part", 0),
        request(3, main, "right.Part", 0),
        request(4, main, "a.value", 2),
        request(5, main, "b.value", 2),
        change(2, &renamed),
        change(1, main),
        request(6, &renamed, "renamed.Part", 0),
        request(7, &renamed, "a.value", 2),
        change(3, &invalid),
        request(8, &invalid, "renamed.Part", 0),
        request(9, &invalid, "a.value", 2),
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
    let text = |id| {
        response(&messages, id)["result"]["contents"]["value"]
            .as_str()
            .unwrap_or_default()
    };
    for (id, index) in [(2, 0), (3, 1), (4, 0), (5, 1), (6, 0), (7, 0)] {
        assert!(text(id).contains(&expected[index]), "{}", text(id));
        assert!(!text(id).contains(&expected[1 - index]));
    }
    assert_eq!(text(2), text(6));
    assert_eq!(text(4), text(7));
    assert!(text(4).contains("Module: \"org.example.Left.main\""));
    assert!(text(5).contains("Module: \"org.example.Right.main\""));
    for id in [8, 9] {
        assert!(!text(id).contains("Origin namespace:"), "{}", text(id));
    }
    assert!(!fixture.0.join("eqiora.lock").exists());
}
