#[path = "stdio/assistance.rs"]
mod assistance;
#[path = "stdio/authored_notation.rs"]
mod authored_notation;
#[path = "stdio/clock_details.rs"]
mod clock_details;
#[path = "stdio/component_values.rs"]
mod component_values;
#[path = "stdio/hover_origin.rs"]
mod hover_origin;
#[path = "stdio/inspection.rs"]
mod inspection;
#[path = "stdio/lexical_outline.rs"]
mod lexical_outline;
#[path = "stdio/local_navigation.rs"]
mod local_navigation;
#[path = "stdio/notation.rs"]
mod notation;
#[path = "stdio/project_refresh.rs"]
mod project_refresh;
#[path = "stdio/symbol_details.rs"]
mod symbol_details;
#[path = "stdio/value_references.rs"]
mod value_references;

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};

use eqiora::api::package::prepare_package_release_v1;
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, NormalizedRelativePath, PackageDependencyV1,
    PackageManifestV1, PackageSourcesV1, QualifiedName, SourceFileV1,
};
use serde_json::{Value, json};

const SERVER: &str = env!("CARGO_BIN_EXE_eqiora-language-server");
const SOURCE_PATH: &str = "src/main.eqi";
static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn create(name: &str) -> Self {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "eqiora-lsp-{name}-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create test directory");
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove test directory");
    }
}

fn source_position(source: &str, occurrence: &str) -> Value {
    let offset = source.find(occurrence).expect("source occurrence");
    let before = &source[..offset];
    json!({
        "line": before.bytes().filter(|byte| *byte == b'\n').count(),
        "character": before.rsplit('\n').next().unwrap().encode_utf16().count(),
    })
}

fn author_sources(
    name: &str,
    source: &str,
    dependencies: Vec<PackageDependencyV1>,
) -> PackageSourcesV1 {
    let path = NormalizedRelativePath::parse(SOURCE_PATH).expect("source path");
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse(name).expect("package name"),
        ExactVersion::parse("1.0.0").expect("package version"),
        dependencies,
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .expect("package manifest");
    PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            source.as_bytes().to_vec(),
        )],
    )
    .expect("author sources")
}

fn write_package(path: &Path, sources: &PackageSourcesV1) {
    fs::create_dir_all(path.join("src")).expect("create package source directory");
    fs::write(path.join(SOURCE_PATH), sources.files()[0].bytes()).expect("write package source");
}

fn file_uri(path: &Path) -> String {
    let mut encoded = String::new();
    for byte in path.to_str().expect("UTF-8 test path").bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/') {
            encoded.push(char::from(byte));
        } else {
            use std::fmt::Write as _;
            write!(&mut encoded, "%{byte:02X}").expect("write URI");
        }
    }
    format!("file://{encoded}")
}

#[test]
fn version_command_reports_the_release() {
    let output = Command::new(SERVER)
        .arg("--version")
        .output()
        .expect("run language-server version command");
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).expect("UTF-8 version output"),
        format!("eqiora-language-server {}\n", env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn stdio_session_syncs_diagnostics_and_serves_editor_requests() {
    let source = "dimension Scalar = 1;\npublic component Part(parameter gain: Scalar) {\n  relation law { gain = 0; }\n}\nmodel Demo(parameter input: Scalar = 1) {\n  variable state: Scalar;\n  instance part: Part(gain = input);\n  relation balance { state = 0; }\n}\n";
    let uri = "file:///workspace/main.eqi";
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn language server");

    let messages = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":-3,"text":source}}}),
        json!({"jsonrpc":"2.0","id":2,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":uri}}}),
        json!({"jsonrpc":"2.0","id":3,"method":"textDocument/foldingRange","params":{"textDocument":{"uri":uri}}}),
        json!({"jsonrpc":"2.0","id":4,"method":"textDocument/hover","params":{"textDocument":{"uri":uri},"position":source_position(source, "Part(gain")}}),
        json!({"jsonrpc":"2.0","id":5,"method":"textDocument/definition","params":{"textDocument":{"uri":uri},"position":source_position(source, "Part(gain")}}),
        json!({"jsonrpc":"2.0","id":6,"method":"textDocument/formatting","params":{"textDocument":{"uri":uri},"options":{"tabSize":2,"insertSpaces":true}}}),
        json!({"jsonrpc":"2.0","id":8,"method":"textDocument/formatting","params":{"textDocument":42}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":42}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":-1},"contentChanges":[{"text":"model Broken() { nonsense; }\n"}]}}),
        json!({"jsonrpc":"2.0","id":9,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":uri}}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":-2},"contentChanges":[{"text":source}]}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":0},"contentChanges":[{"text":source}]}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didClose","params":{"textDocument":{"uri":uri}}}),
        json!({"jsonrpc":"2.0","id":7,"method":"shutdown","params":null}),
        json!({"jsonrpc":"2.0","method":"exit","params":null}),
    ];
    let mut stdin = child.stdin.take().expect("language server stdin");
    for message in messages {
        write_packet(&mut stdin, &message);
    }
    drop(stdin);

    let output = child.wait_with_output().expect("language server output");
    assert!(
        output.status.success(),
        "language server failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lifecycle = String::from_utf8(output.stderr)
        .expect("UTF-8 lifecycle log")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("structured lifecycle log"))
        .collect::<Vec<_>>();
    assert_eq!(lifecycle[0]["event"], "server_started");
    assert_eq!(lifecycle[1]["event"], "server_stopped");
    let messages = parse_packets(&output.stdout);

    let initialize = response(&messages, 1);
    assert_eq!(
        initialize["result"]["serverInfo"]["name"],
        "eqiora-language-server"
    );
    assert_eq!(
        initialize["result"]["capabilities"]["positionEncoding"],
        "utf-16"
    );
    assert_eq!(
        initialize["result"]["capabilities"]["referencesProvider"],
        true
    );

    let symbols = response(&messages, 2)["result"]
        .as_array()
        .expect("nested document symbols");
    assert_eq!(symbols[0]["name"], "Scalar");
    assert_eq!(symbols[1]["name"], "Part");
    assert_eq!(symbols[2]["name"], "Demo");
    assert_eq!(symbols[2]["children"][2]["name"], "part");

    let folds = response(&messages, 3)["result"]
        .as_array()
        .expect("folding ranges");
    assert_eq!(folds.len(), 2);
    assert_eq!(folds[0]["collapsedText"], "Part");

    let hover = response(&messages, 4);
    assert!(
        hover["result"]["contents"]["value"]
            .as_str()
            .expect("Markdown hover")
            .contains("**Component** `main.Part`")
    );

    let definition = response(&messages, 5);
    assert_eq!(definition["result"]["uri"], uri);
    assert_eq!(definition["result"]["range"]["start"]["line"], 1);

    let edits = response(&messages, 6)["result"]
        .as_array()
        .expect("format edits");
    assert_eq!(edits.len(), 1);
    assert!(
        edits[0]["newText"]
            .as_str()
            .expect("formatted source")
            .contains("public component Part(\n  parameter gain: Scalar,\n) {")
    );

    let diagnostics = messages
        .iter()
        .filter(|message| message["method"] == "textDocument/publishDiagnostics")
        .map(|message| &message["params"])
        .collect::<Vec<_>>();
    assert_eq!(diagnostics.len(), 3);
    assert_eq!(diagnostics[0]["version"], -3);
    assert_eq!(diagnostics[0]["diagnostics"], json!([]));
    assert_eq!(diagnostics[1]["version"], -1);
    assert!(
        !diagnostics[1]["diagnostics"]
            .as_array()
            .expect("malformed diagnostics")
            .is_empty()
    );
    assert!(diagnostics[2]["version"].is_null());
    assert_eq!(diagnostics[2]["diagnostics"], json!([]));
    assert!(
        messages
            .iter()
            .all(|message| !matches!(message["params"]["version"].as_i64(), Some(-2 | 0)))
    );
    assert!(response(&messages, 7)["result"].is_null());
    assert_eq!(response(&messages, 8)["error"]["code"], -32602);
}

#[test]
fn stdio_workspace_resolves_open_modules_and_tracks_unsaved_changes() {
    let main = "// 🧪\nimport editor.workspace.library as lib;\nmodel Main() { instance load: lib.Resistor(); }\n";
    let library = "public component Resistor() {}\n";
    let changed_library = "/// **Updated summary**\n///\n/// [run](command:delete) <script> ```\npublic component Resistor() {\n  // unsaved workspace edit\n}\n";
    let main_uri = "file:///workspace/main.eqi";
    let library_uri = "file:///workspace/library.eqi";
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn language server");

    let messages = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{"workspace":{"workspaceFolders":true}},"workspaceFolders":[{"uri":"file:///workspace","name":"workspace"}]}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":library_uri,"languageId":"eqiora","version":1,"text":library}}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":main_uri,"languageId":"eqiora","version":1,"text":main}}}),
        json!({"jsonrpc":"2.0","id":2,"method":"textDocument/definition","params":{"textDocument":{"uri":main_uri},"position":source_position(main, "Resistor()")}}),
        json!({"jsonrpc":"2.0","id":6,"method":"textDocument/references","params":{"textDocument":{"uri":main_uri},"position":source_position(main, "Resistor()"),"context":{"includeDeclaration":true}}}),
        json!({"jsonrpc":"2.0","id":7,"method":"textDocument/references","params":{"textDocument":{"uri":library_uri},"position":{"line":0,"character":20},"context":{"includeDeclaration":false}}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":library_uri,"version":2},"contentChanges":[{"text":changed_library}]}}),
        json!({"jsonrpc":"2.0","id":3,"method":"textDocument/hover","params":{"textDocument":{"uri":main_uri},"position":source_position(main, "Resistor()")}}),
        json!({"jsonrpc":"2.0","id":4,"method":"textDocument/definition","params":{"textDocument":{"uri":main_uri},"position":source_position(main, "Resistor()")}}),
        json!({"jsonrpc":"2.0","id":8,"method":"textDocument/signatureHelp","params":{"textDocument":{"uri":main_uri},"position":source_position(main, ");")}}),
        json!({"jsonrpc":"2.0","id":5,"method":"shutdown","params":null}),
        json!({"jsonrpc":"2.0","method":"exit","params":null}),
    ];
    let mut stdin = child.stdin.take().expect("language server stdin");
    for message in messages {
        write_packet(&mut stdin, &message);
    }
    drop(stdin);

    let output = child.wait_with_output().expect("language server output");
    assert!(
        output.status.success(),
        "language server failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let messages = parse_packets(&output.stdout);

    assert_eq!(
        response(&messages, 1)["result"]["capabilities"]["workspace"]["workspaceFolders"]["supported"],
        true
    );
    let definition = response(&messages, 2);
    assert_eq!(definition["result"]["uri"], library_uri);
    assert_eq!(definition["result"]["range"]["start"]["line"], 0);
    assert_eq!(definition["result"]["range"]["start"]["character"], 17);
    assert_eq!(definition["result"]["range"]["end"]["character"], 25);

    let references_with_declaration = response(&messages, 6)["result"]
        .as_array()
        .expect("references with declaration");
    assert_eq!(references_with_declaration.len(), 2);
    assert_eq!(references_with_declaration[0]["uri"], library_uri);
    assert_eq!(references_with_declaration[1]["uri"], main_uri);
    assert_eq!(references_with_declaration[1]["range"]["start"]["line"], 2);
    let references = response(&messages, 7)["result"]
        .as_array()
        .expect("references without declaration");
    assert_eq!(references.len(), 1);
    assert_eq!(references[0]["uri"], main_uri);

    let hover = response(&messages, 3)["result"]["contents"]["value"]
        .as_str()
        .expect("workspace Markdown hover");
    assert!(hover.contains("**Component** `library.Resistor`"));
    assert!(hover.contains("// unsaved workspace edit"));
    assert!(hover.starts_with(
        "**Updated summary**\n\n\\[run\\](command&#58;delete) \\<script\\> \\`\\`\\`"
    ));
    assert!(!hover.contains("[run](command:delete)"));
    assert_eq!(response(&messages, 4)["result"]["uri"], library_uri);
    let signature = &response(&messages, 8)["result"]["signatures"][0];
    assert_eq!(signature["label"], "public component Resistor()");
    assert!(
        signature["documentation"]["value"]
            .as_str()
            .unwrap()
            .starts_with("**Updated summary**")
    );
    assert_eq!(signature["parameters"], json!([]));
    assert!(response(&messages, 5)["result"].is_null());
}

#[test]
fn stdio_publishes_current_analysis_after_successive_changes() {
    let uri = "file:///workspace/cancelled.eqi";
    let valid = "model Current() {}\n";
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn language server");
    let messages = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":valid}}}),
        json!({"jsonrpc":"2.0","id":2,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":uri}}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":2},"contentChanges":[{"text":"model Superseded() { nonsense; }\n"}]}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":3},"contentChanges":[{"text":valid}]}}),
        json!({"jsonrpc":"2.0","id":3,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":uri}}}),
        json!({"jsonrpc":"2.0","id":4,"method":"shutdown","params":null}),
        json!({"jsonrpc":"2.0","method":"exit","params":null}),
    ];
    let mut stdin = child.stdin.take().expect("language server stdin");
    for message in messages {
        write_packet(&mut stdin, &message);
    }
    drop(stdin);

    let output = child.wait_with_output().expect("language server output");
    assert!(
        output.status.success(),
        "language server failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let messages = parse_packets(&output.stdout);
    let diagnostics = messages
        .iter()
        .filter(|message| message["method"] == "textDocument/publishDiagnostics")
        .map(|message| message["params"]["version"].as_i64())
        .collect::<Vec<_>>();
    // Revision 2 may finish before the server receives revision 3. Supersession
    // is tested deterministically at the in-flight revision boundary in protocol tests.
    assert!(
        diagnostics == [Some(1), Some(3)] || diagnostics == [Some(1), Some(2), Some(3)],
        "unexpected diagnostic revisions: {diagnostics:?}"
    );
    let latest = messages
        .iter()
        .rev()
        .find(|message| message["method"] == "textDocument/publishDiagnostics")
        .unwrap();
    assert_eq!(latest["params"]["diagnostics"], json!([]));
    assert_eq!(response(&messages, 3)["result"][0]["name"], "Current");
    assert!(response(&messages, 4)["result"].is_null());
}

#[test]
fn stdio_answers_once_when_cancellation_races_with_analysis() {
    let uri = "file:///workspace/request-cancellation.eqi";
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn language server");
    let messages = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":"model Cancelled() {}\n"}}}),
        json!({"jsonrpc":"2.0","id":2,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":uri}}}),
        json!({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":2}}),
        json!({"jsonrpc":"2.0","id":3,"method":"shutdown","params":null}),
        json!({"jsonrpc":"2.0","method":"exit","params":null}),
    ];
    let mut stdin = child.stdin.take().expect("language server stdin");
    for message in messages {
        write_packet(&mut stdin, &message);
    }
    drop(stdin);

    let output = child.wait_with_output().expect("language server output");
    assert!(
        output.status.success(),
        "language server failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let messages = parse_packets(&output.stdout);
    // Stdio delivery and analysis run independently: a completed response may win.
    // The pending-analysis cancellation path is exercised deterministically in protocol tests.
    let answer = response(&messages, 2);
    if answer.get("error").is_some() {
        assert_eq!(answer["error"]["code"], -32800);
        assert_eq!(answer["error"]["message"], "canceled by client");
        assert!(answer.get("result").is_none());
    } else {
        let symbols = answer["result"].as_array().expect("completed symbols");
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0]["name"], "Cancelled");
    }
    assert_eq!(
        messages
            .iter()
            .filter(|message| message["id"].as_i64() == Some(2))
            .count(),
        1
    );
    assert!(response(&messages, 3)["result"].is_null());
}

#[test]
fn stdio_workspace_loads_unopened_exact_package_sources_without_writing_a_lock() {
    let fixture = TestDirectory::create("package project");
    let library_path = fixture.0.join("library");
    let root_path = fixture.0.join("root");
    let library = "public component Resistor() {}\n";
    let root = "import org.example.EditorLibrary.main as library;\nmodel Main() { instance load: library.Resistor(); }\n";
    let library_sources = author_sources("org.example.EditorLibrary", library, vec![]);
    let library_release =
        prepare_package_release_v1(library_sources.clone(), &[]).expect("library release");
    let dependency = PackageDependencyV1::new(
        library_release
            .package_identity()
            .expect("library package identity"),
    );
    let root_sources = author_sources("org.example.EditorRoot", root, vec![dependency]);
    write_package(&library_path, &library_sources);
    write_package(&root_path, &root_sources);
    fs::write(
        fixture.0.join("eqiora.toml"),
        "[package]\nname = \"org.example.EditorRoot\"\nversion = \"1.0.0\"\nsource = \"root/src\"\nentry = \"main\"\n\n[dependencies.\"org.example.EditorLibrary\"]\nversion = \"1.0.0\"\nsources = [{ path = \"library\" }]\n",
    )
    .expect("write project manifest");
    fs::write(
        library_path.join("eqiora.toml"),
        "[package]\nname = \"org.example.EditorLibrary\"\nversion = \"1.0.0\"\nentry = \"main\"\n",
    )
    .expect("write library manifest");
    let workspace_uri = file_uri(&fixture.0);
    let root_uri = file_uri(&root_path.join(SOURCE_PATH));
    let library_uri = file_uri(&library_path.join(SOURCE_PATH));

    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn language server");
    let changed_root = format!("// unsaved\n{root}");
    let messages = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{"workspace":{"workspaceFolders":true}},"workspaceFolders":[{"uri":workspace_uri,"name":"package project"}]}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":root_uri,"languageId":"eqiora","version":1,"text":root}}}),
        json!({"jsonrpc":"2.0","id":2,"method":"textDocument/hover","params":{"textDocument":{"uri":root_uri},"position":source_position(root, "Resistor()")}}),
        json!({"jsonrpc":"2.0","id":3,"method":"textDocument/definition","params":{"textDocument":{"uri":root_uri},"position":source_position(root, "Resistor()")}}),
        json!({"jsonrpc":"2.0","id":6,"method":"textDocument/references","params":{"textDocument":{"uri":root_uri},"position":source_position(root, "Resistor()"),"context":{"includeDeclaration":true}}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":root_uri,"version":2},"contentChanges":[{"text":changed_root}]}}),
        json!({"jsonrpc":"2.0","id":4,"method":"textDocument/definition","params":{"textDocument":{"uri":root_uri},"position":source_position(&changed_root, "Resistor()")}}),
        json!({"jsonrpc":"2.0","id":5,"method":"shutdown","params":null}),
        json!({"jsonrpc":"2.0","method":"exit","params":null}),
    ];
    let mut stdin = child.stdin.take().expect("language server stdin");
    for message in messages {
        write_packet(&mut stdin, &message);
    }
    drop(stdin);

    let output = child.wait_with_output().expect("language server output");
    assert!(
        output.status.success(),
        "language server failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let messages = parse_packets(&output.stdout);
    let hover = response(&messages, 2)["result"]["contents"]["value"]
        .as_str()
        .expect("package Markdown hover");
    assert!(hover.contains("**Component** `main.Resistor`"));
    assert_eq!(response(&messages, 3)["result"]["uri"], library_uri);
    let references = response(&messages, 6)["result"]
        .as_array()
        .expect("exact-package references");
    assert_eq!(references.len(), 2);
    assert_eq!(references[0]["uri"], library_uri);
    assert_eq!(references[1]["uri"], root_uri);
    assert_eq!(response(&messages, 4)["result"]["uri"], library_uri);
    assert!(response(&messages, 5)["result"].is_null());
    assert!(!fixture.0.join("eqiora.lock").exists());
}

fn write_packet(writer: &mut impl Write, message: &Value) {
    let body = serde_json::to_vec(message).expect("serialize LSP message");
    write!(writer, "Content-Length: {}\r\n\r\n", body.len()).expect("write LSP header");
    writer.write_all(&body).expect("write LSP body");
    writer.flush().expect("flush LSP message");
}

fn parse_packets(mut bytes: &[u8]) -> Vec<Value> {
    let mut messages = Vec::new();
    while !bytes.is_empty() {
        let header_end = bytes
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("complete LSP header");
        let header = std::str::from_utf8(&bytes[..header_end]).expect("UTF-8 LSP header");
        let length = header
            .lines()
            .find_map(|line| line.strip_prefix("Content-Length: "))
            .expect("Content-Length header")
            .parse::<usize>()
            .expect("decimal Content-Length");
        let body_start = header_end + 4;
        let body_end = body_start + length;
        messages
            .push(serde_json::from_slice(&bytes[body_start..body_end]).expect("JSON-RPC response"));
        bytes = &bytes[body_end..];
    }
    messages
}

fn response(messages: &[Value], id: i64) -> &Value {
    messages
        .iter()
        .find(|message| message["id"].as_i64() == Some(id))
        .unwrap_or_else(|| panic!("missing response {id}: {messages:#?}"))
}
