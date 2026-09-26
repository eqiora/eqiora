use super::*;
use std::{io::BufReader, process::Child, sync::mpsc, time::Duration};

// Keep a live ordinary stdio session so disk mutations happen after accepted analysis.
struct Session {
    child: Child,
    input: std::process::ChildStdin,
    output: mpsc::Receiver<Value>,
}
impl Session {
    fn new() -> Self {
        let mut child = Command::new(SERVER)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (sender, output) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            while let Some(message) = lsp_server::Message::read(&mut reader).unwrap() {
                if sender.send(serde_json::to_value(message).unwrap()).is_err() {
                    break;
                }
            }
        });
        Self {
            child,
            input,
            output,
        }
    }
    fn send(&mut self, message: Value) {
        write_packet(&mut self.input, &message);
    }
    fn next(&self) -> Value {
        self.output
            .recv_timeout(Duration::from_secs(15))
            .expect("server response")
    }
    fn response(&self, id: i64) -> Value {
        loop {
            let message = self.next();
            if message["id"].as_i64() == Some(id) {
                assert!(message.get("error").is_none(), "{message}");
                return message["result"].clone();
            }
            assert_eq!(
                message["method"], "textDocument/publishDiagnostics",
                "{message}"
            );
            assert_eq!(message["params"]["version"], 7);
        }
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn watched_project_files_refresh_unopened_sources_and_preserve_unsaved_text() {
    let fixture = TestDirectory::create("watched package");
    let library_path = fixture.0.join("library");
    let root_path = fixture.0.join("root");
    let library = "/// Before disk edit.\npublic component Part() {}\n";
    let root = "import org.example.Library.main as library;\nmodel Main() { instance load: library.Part(); }\n";
    let library_sources = author_sources("org.example.Library", library, vec![]);
    let release = prepare_package_release_v1(library_sources.clone(), &[]).unwrap();
    write_package(&library_path, &library_sources);
    write_package(
        &root_path,
        &author_sources(
            "org.example.Root",
            root,
            vec![PackageDependencyV1::new(
                release.package_identity().unwrap(),
            )],
        ),
    );
    let manifest = "[package]\nname = \"org.example.Root\"\nversion = \"1.0.0\"\nsource = \"root/src\"\nentry = \"main\"\n\n[dependencies.\"org.example.Library\"]\nversion = \"1.0.0\"\nsources = [{ path = \"library\" }]\n";
    fs::write(fixture.0.join("eqiora.toml"), manifest).unwrap();
    fs::write(
        library_path.join("eqiora.toml"),
        "[package]\nname = \"org.example.Library\"\nversion = \"1.0.0\"\nentry = \"main\"\n",
    )
    .unwrap();
    let uri = file_uri(&root_path.join(SOURCE_PATH));
    let library_uri = file_uri(&library_path.join(SOURCE_PATH));
    let unsaved = format!("// unsaved 🦀\n{root}");
    let mut session = Session::new();
    session.send(
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
            "capabilities":{"workspace":{"didChangeWatchedFiles":{"dynamicRegistration":true}}},
            "workspaceFolders":[{"uri":file_uri(&fixture.0),"name":"project"}]
        }}),
    );
    assert_eq!(
        session.response(1)["capabilities"]["textDocumentSync"]["save"],
        true
    );
    session.send(json!({"jsonrpc":"2.0","method":"initialized","params":{}}));
    let registration = session.next();
    assert_eq!(registration["method"], "client/registerCapability");
    assert_eq!(
        registration["params"]["registrations"][0]["method"],
        "workspace/didChangeWatchedFiles"
    );
    session.send(json!({"jsonrpc":"2.0","id":registration["id"],"result":null}));
    session.send(
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{
            "uri":uri,"languageId":"eqiora","version":7,"text":unsaved
        }}}),
    );
    let hover = |id| {
        json!({"jsonrpc":"2.0","id":id,"method":"textDocument/hover","params":{
            "textDocument":{"uri":uri},"position":source_position(&unsaved,"Part()")
        }})
    };
    let event = |changed_uri: &str, kind| {
        json!({"jsonrpc":"2.0","method":"workspace/didChangeWatchedFiles","params":{
            "changes":[{"uri":changed_uri,"type":kind}]
        }})
    };
    session.send(hover(2));
    assert!(
        session.response(2)["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("Before disk edit")
    );
    // A newly discovered package file must also use its current open buffer.
    let extra_path = library_path.join("src/extra.eqi");
    let extra_uri = file_uri(&extra_path);
    fs::write(
        &extra_path,
        "/// Disk declaration.\npublic component Extra() {}\n",
    )
    .unwrap();
    let extra_source = "/// Current unsaved declaration.\npublic component Extra() {}\n";
    session.send(
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{
            "uri":extra_uri,"languageId":"eqiora","version":7,"text":extra_source
        }}}),
    );
    session.send(
        json!({"jsonrpc":"2.0","id":30,"method":"textDocument/hover","params":{
            "textDocument":{"uri":extra_uri},"position":source_position(extra_source,"Extra()")
        }}),
    );
    let extra_hover = session.response(30)["contents"]["value"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(
        extra_hover.contains("Current unsaved declaration"),
        "{extra_hover}"
    );
    assert!(!extra_hover.contains("Disk declaration"));
    // Disk changes must not replace the current unsaved root or its LSP version.
    fs::write(root_path.join(SOURCE_PATH), "invalid disk text").unwrap();
    fs::write(
        library_path.join(SOURCE_PATH),
        "// moved declaration\n/// After disk edit.\npublic component Part() {}\n",
    )
    .unwrap();
    session.send(event(&library_uri, 2));
    session.send(hover(3));
    let text = session.response(3)["contents"]["value"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(text.contains("After disk edit"), "{text}");
    assert!(!text.contains("Before disk edit"));
    session.send(
        json!({"jsonrpc":"2.0","id":4,"method":"textDocument/definition","params":{
            "textDocument":{"uri":uri},"position":source_position(&unsaved,"Part()")
        }}),
    );
    let definition = session.response(4);
    assert_eq!(definition["uri"], library_uri);
    assert_eq!(definition["range"]["start"]["line"], 2);
    // Deletion drops previous package facts; recreation is eligible for fresh analysis.
    fs::remove_file(library_path.join(SOURCE_PATH)).unwrap();
    session.send(event(&library_uri, 3));
    session.send(hover(5));
    assert!(session.response(5).is_null());
    fs::write(library_path.join(SOURCE_PATH), library).unwrap();
    session.send(event(&library_uri, 1));
    session.send(hover(6));
    assert!(
        session.response(6)["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("Before disk edit")
    );
    // Clients without active file watching can save an open file to refresh the graph.
    fs::write(
        library_path.join(SOURCE_PATH),
        "/// Save refresh.\npublic component Part() {}\n",
    )
    .unwrap();
    session.send(
        json!({"jsonrpc":"2.0","method":"textDocument/didSave","params":{
            "textDocument":{"uri":uri},"text":"ignored; didChange owns the current text"
        }}),
    );
    session.send(hover(7));
    assert!(
        session.response(7)["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("Save refresh")
    );
    for (index, name) in ["eqiora.toml", "eqiora.lock"].into_iter().enumerate() {
        let path = fixture.0.join(name);
        let before = fs::read(&path).ok();
        fs::write(&path, "invalid project input").unwrap();
        let id = 8 + index as i64 * 2;
        session.send(event(&file_uri(&path), 2));
        session.send(hover(id));
        assert!(
            session.response(id).is_null(),
            "stale facts after invalid {name}"
        );
        assert_eq!(fs::read_to_string(&path).unwrap(), "invalid project input");
        let kind = if let Some(bytes) = before {
            fs::write(&path, bytes).unwrap();
            2
        } else {
            fs::remove_file(&path).unwrap();
            3
        };
        session.send(event(&file_uri(&path), kind));
        session.send(hover(id + 1));
        assert!(
            session.response(id + 1)["contents"]["value"]
                .as_str()
                .unwrap()
                .contains("Save refresh")
        );
    }
    assert!(!fixture.0.join("eqiora.lock").exists());
    assert_eq!(
        fs::read_to_string(root_path.join(SOURCE_PATH)).unwrap(),
        "invalid disk text"
    );
    session.send(json!({"jsonrpc":"2.0","id":12,"method":"shutdown","params":null}));
    assert!(session.response(12).is_null());
    session.send(json!({"jsonrpc":"2.0","method":"exit","params":null}));
    assert!(session.child.wait().unwrap().success());
}
