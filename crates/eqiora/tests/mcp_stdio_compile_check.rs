use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use eqiora::api::{ModelDocument, SemanticFingerprintGeneration};
use eqiora::{Diagnostic, Severity};
use serde_json::{Map, Value, json};

#[path = "mcp_stdio_compile_check/release_identity.rs"]
mod release_identity;
use release_identity::{assert_frozen_release_identity, expected, frozen_expected};

const CONTRACT_SOURCE: &str =
    include_str!("../../../verify/interfaces/mcp-stdio-compile-check/expected/contract.json");
const MODELS_README_SOURCE: &str =
    include_str!("../../../verify/interfaces/mcp-stdio-compile-check/models/README.md");
const REFERENCES_README_SOURCE: &str =
    include_str!("../../../verify/interfaces/mcp-stdio-compile-check/references/README.md");
const ACCEPTED_CONTROL_REQUEST: &[u8] = include_bytes!(
    "../../../verify/interfaces/control-plane-compile-check/models/accepted-v2.json"
);

const MAIN_SOURCE: &str = include_str!("../src/bin/eqiora-mcp/main.rs");
const FRAMING_SOURCE: &str = include_str!("../src/bin/eqiora-mcp/framing.rs");
const PROTOCOL_SOURCE: &str = include_str!("../src/bin/eqiora-mcp/protocol.rs");
const TOOL_SOURCE: &str = include_str!("../src/bin/eqiora-mcp/tool.rs");
const ORACLE_SOURCE: &str = include_str!("../src/bin/eqiora-mcp/oracle.rs");
const WORKER_OUTCOMES_SOURCE: &str =
    include_str!("../src/bin/eqiora-mcp/oracle/worker_outcomes.rs");
fn timeout(name: &str) -> Duration {
    Duration::from_millis(
        expected()["oracleTimeouts"][name]
            .as_u64()
            .unwrap_or_else(|| panic!("missing frozen timeout `{name}`")),
    )
}

fn request_meta() -> Value {
    json!({
        "io.modelcontextprotocol/protocolVersion": expected()["protocolVersion"],
        "io.modelcontextprotocol/clientCapabilities": {}
    })
}

fn request(id: Value, method: &str, mut params: Map<String, Value>) -> Value {
    params.insert("_meta".to_owned(), request_meta());
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
}

fn discover_request(id: Value) -> Value {
    request(id, "server/discover", Map::new())
}

fn list_request(id: Value) -> Value {
    request(id, "tools/list", Map::new())
}

fn call_request(id: Value, arguments: Value) -> Value {
    let mut params = Map::new();
    params.insert("name".to_owned(), expected()["toolName"].clone());
    params.insert("arguments".to_owned(), arguments);
    request(id, "tools/call", params)
}

struct Client {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<Vec<u8>>,
    stdout_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<Vec<u8>>>,
}

impl Client {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_eqiora-mcp"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("launch eqiora-mcp");
        let stdin = child.stdin.take().expect("child stdin");
        let stdout = child.stdout.take().expect("child stdout");
        let stderr = child.stderr.take().expect("child stderr");
        let (sender, lines) = mpsc::channel();
        let stdout_thread = thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = Vec::new();
                match reader.read_until(b'\n', &mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if sender.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        let stderr_thread = thread::spawn(move || {
            let mut bytes = Vec::new();
            let mut stderr = stderr;
            stderr.read_to_end(&mut bytes).expect("read child stderr");
            bytes
        });
        Self {
            child,
            stdin: Some(stdin),
            lines,
            stdout_thread: Some(stdout_thread),
            stderr_thread: Some(stderr_thread),
        }
    }

    fn send_value(&mut self, value: &Value) {
        let mut bytes = serde_json::to_vec(value).expect("serialize request");
        bytes.push(b'\n');
        self.send_raw(&bytes);
    }

    fn send_value_crlf(&mut self, value: &Value) {
        let mut bytes = serde_json::to_vec(value).expect("serialize request");
        bytes.extend_from_slice(b"\r\n");
        self.send_raw(&bytes);
    }

    fn send_raw(&mut self, bytes: &[u8]) {
        let stdin = self.stdin.as_mut().expect("open child stdin");
        stdin.write_all(bytes).expect("write child stdin");
        stdin.flush().expect("flush child stdin");
    }

    fn recv_raw(&self) -> Vec<u8> {
        self.recv_raw_with(timeout("ordinaryResponseMs"))
    }

    fn recv_raw_with(&self, duration: Duration) -> Vec<u8> {
        let line = self
            .lines
            .recv_timeout(duration)
            .unwrap_or_else(|error| panic!("timed out waiting for MCP response: {error}"));
        assert_eq!(line.last(), Some(&b'\n'), "response must end in one LF");
        assert_ne!(line.get(line.len().saturating_sub(2)), Some(&b'\r'));
        assert_eq!(line.iter().filter(|byte| **byte == b'\n').count(), 1);
        assert_wire_response(&line);
        line
    }

    fn recv(&self) -> Value {
        let bytes = self.recv_raw();
        serde_json::from_slice(&bytes).expect("response is one JSON object")
    }

    fn recv_with(&self, duration: Duration) -> Value {
        let bytes = self.recv_raw_with(duration);
        serde_json::from_slice(&bytes).expect("response is one JSON object")
    }

    fn assert_silent(&self) {
        assert!(
            self.lines.recv_timeout(timeout("silenceProbeMs")).is_err(),
            "notification or suppressed request unexpectedly produced a response"
        );
    }

    fn shutdown(mut self) -> Vec<u8> {
        self.stdin.take();
        let deadline = Instant::now() + timeout("shutdownMs");
        let status = loop {
            if let Some(status) = self.child.try_wait().expect("poll child") {
                break status;
            }
            assert!(Instant::now() < deadline, "eqiora-mcp did not shut down");
            thread::sleep(Duration::from_millis(10));
        };
        assert!(status.success(), "eqiora-mcp exited with {status}");
        self.stdout_thread.take().unwrap().join().unwrap();
        let queued = self.lines.try_iter().collect::<Vec<_>>();
        assert!(
            queued.is_empty(),
            "server emitted unsolicited or EOF-triggered stdout: {queued:?}"
        );
        self.stderr_thread.take().unwrap().join().unwrap()
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn response_id(response: &Value) -> Option<&Value> {
    response.as_object().unwrap().get("id")
}

fn assert_exact_members(value: &Value, members: &[&str]) {
    let object = value.as_object().expect("closed JSON object");
    let mut actual = object.keys().map(String::as_str).collect::<Vec<_>>();
    let mut expected = members.to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    assert_eq!(actual, expected);
}

fn parse_raw_value(source: &str, start: usize) -> (Value, usize) {
    let mut values = serde_json::Deserializer::from_str(&source[start..]).into_iter::<Value>();
    let value = values
        .next()
        .expect("raw JSON value")
        .expect("valid raw JSON value");
    (value, start + values.byte_offset())
}

fn skip_raw_whitespace(source: &str, mut cursor: usize) -> usize {
    while source
        .as_bytes()
        .get(cursor)
        .is_some_and(u8::is_ascii_whitespace)
    {
        cursor += 1;
    }
    cursor
}

fn raw_object_members(source: &str) -> Vec<(String, &str)> {
    let mut cursor = skip_raw_whitespace(source, 0);
    assert_eq!(source.as_bytes().get(cursor), Some(&b'{'));
    cursor += 1;
    let mut members = Vec::new();
    loop {
        cursor = skip_raw_whitespace(source, cursor);
        if source.as_bytes().get(cursor) == Some(&b'}') {
            cursor = skip_raw_whitespace(source, cursor + 1);
            assert_eq!(cursor, source.len());
            return members;
        }
        let (key, key_end) = parse_raw_value(source, cursor);
        let key = key.as_str().expect("JSON object member").to_owned();
        cursor = skip_raw_whitespace(source, key_end);
        assert_eq!(source.as_bytes().get(cursor), Some(&b':'));
        let value_start = skip_raw_whitespace(source, cursor + 1);
        let (_, value_end) = parse_raw_value(source, value_start);
        members.push((key, &source[value_start..value_end]));
        cursor = skip_raw_whitespace(source, value_end);
        match source.as_bytes().get(cursor) {
            Some(b',') => cursor += 1,
            Some(b'}') => {}
            other => panic!("invalid raw object separator {other:?}"),
        }
    }
}

fn raw_array_values(source: &str) -> Vec<&str> {
    let mut cursor = skip_raw_whitespace(source, 0);
    assert_eq!(source.as_bytes().get(cursor), Some(&b'['));
    cursor += 1;
    let mut values = Vec::new();
    loop {
        cursor = skip_raw_whitespace(source, cursor);
        if source.as_bytes().get(cursor) == Some(&b']') {
            cursor = skip_raw_whitespace(source, cursor + 1);
            assert_eq!(cursor, source.len());
            return values;
        }
        let value_start = cursor;
        let (_, value_end) = parse_raw_value(source, value_start);
        values.push(&source[value_start..value_end]);
        cursor = skip_raw_whitespace(source, value_end);
        match source.as_bytes().get(cursor) {
            Some(b',') => cursor += 1,
            Some(b']') => {}
            other => panic!("invalid raw array separator {other:?}"),
        }
    }
}

fn canonical_json(source: &str) -> String {
    match serde_json::from_str::<Value>(source).unwrap() {
        Value::Object(_) => {
            let members = raw_object_members(source);
            let body = members
                .iter()
                .map(|(name, raw)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(name).unwrap(),
                        canonical_json(raw)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
        Value::Array(_) => {
            let body = raw_array_values(source)
                .iter()
                .map(|raw| canonical_json(raw))
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        value => serde_json::to_string(&value).unwrap(),
    }
}

fn assert_raw_order(members: &[(String, &str)], expected: &[&str]) {
    assert_eq!(
        members
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        expected
    );
}

fn raw_member<'a>(members: &[(String, &'a str)], name: &str) -> &'a str {
    members
        .iter()
        .find(|(candidate, _)| candidate == name)
        .unwrap_or_else(|| panic!("missing raw member `{name}`"))
        .1
}

fn frozen_order<'a>(contract: &'a Value, name: &str) -> Vec<&'a str> {
    contract["resultProjection"][name]
        .as_array()
        .unwrap()
        .iter()
        .map(|member| member.as_str().unwrap())
        .collect()
}

fn assert_wire_response(bytes: &[u8]) {
    let response: Value = serde_json::from_slice(bytes).expect("response is one JSON object");
    let text = std::str::from_utf8(bytes)
        .unwrap()
        .strip_suffix('\n')
        .expect("one response LF");
    assert_eq!(
        canonical_json(text),
        text,
        "response JSON must be canonical compact JSON"
    );
    let contract = expected();
    let outer_members = raw_object_members(text);
    if let Some(error) = response.get("error") {
        let outer = if response.get("id").is_some() {
            &["jsonrpc", "id", "error"][..]
        } else {
            &["jsonrpc", "error"][..]
        };
        assert_exact_members(&response, outer);
        assert_raw_order(&outer_members, outer);
        assert_exact_members(error, &["code", "message", "data"]);
        let error_members = raw_object_members(raw_member(&outer_members, "error"));
        let error_order = contract["responseMemberOrder"]["errorObject"]
            .as_array()
            .unwrap()
            .iter()
            .map(|member| member.as_str().unwrap())
            .collect::<Vec<_>>();
        assert_raw_order(&error_members, &error_order);
    } else {
        assert_exact_members(&response, &["jsonrpc", "id", "result"]);
        assert_raw_order(&outer_members, &["jsonrpc", "id", "result"]);
        if response["result"].get("content").is_some() {
            let result_members = raw_object_members(raw_member(&outer_members, "result"));
            let result_order = frozen_order(&contract, "resultMemberOrder");
            assert_raw_order(&result_members, &result_order);
            let content = raw_array_values(raw_member(&result_members, "content"));
            assert_eq!(content.len(), 1);
            let content_members = raw_object_members(content[0]);
            assert_raw_order(&content_members, &["type", "text"]);
            let text_content: String =
                serde_json::from_str(raw_member(&content_members, "text")).unwrap();
            let structured_raw = raw_member(&result_members, "structuredContent");
            let canonical = canonical_json(structured_raw);
            assert_eq!(structured_raw.as_bytes(), canonical.as_bytes());
            assert_eq!(text_content.as_bytes(), canonical.as_bytes());
            let structured = &response["result"]["structuredContent"];
            let structured_members = raw_object_members(structured_raw);
            if structured["status"] == "accepted" {
                let order = frozen_order(&contract, "acceptedStructuredMemberOrder");
                assert_raw_order(&structured_members, &order);
                let model_members = raw_object_members(raw_member(&structured_members, "model"));
                let order = frozen_order(&contract, "acceptedModelMemberOrder");
                assert_raw_order(&model_members, &order);
                let fingerprint_members =
                    raw_object_members(raw_member(&model_members, "structuralFingerprint"));
                let order = frozen_order(&contract, "fingerprintMemberOrder");
                assert_raw_order(&fingerprint_members, &order);
            } else {
                let order = frozen_order(&contract, "rejectedStructuredMemberOrder");
                assert_raw_order(&structured_members, &order);
                for diagnostic in raw_array_values(raw_member(&structured_members, "diagnostics")) {
                    let diagnostic_members = raw_object_members(diagnostic);
                    let order = frozen_order(&contract, "diagnosticMemberOrder");
                    assert_raw_order(&diagnostic_members, &order);
                    let span = raw_member(&diagnostic_members, "span");
                    if span != "null" {
                        let span_members = raw_object_members(span);
                        let order = frozen_order(&contract, "spanMemberOrder");
                        assert_raw_order(&span_members, &order);
                    }
                    let patch = raw_member(&diagnostic_members, "patch");
                    if patch != "null" {
                        let patch_members = raw_object_members(patch);
                        let order = frozen_order(&contract, "patchMemberOrder");
                        assert_raw_order(&patch_members, &order);
                    }
                }
            }
        }
    }
}

fn assert_error(response: &Value, code: i64, id: Option<&Value>) {
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["error"]["code"], code);
    assert!(response["error"]["message"].as_str().is_some());
    assert_eq!(response_id(response), id);
    assert!(response["error"].get("data").is_some());
    if matches!(code, -32700 | -32600 | -32601 | -32602 | -32603) {
        assert_eq!(
            response["error"]["data"],
            expected()["responseMemberOrder"]["genericErrorData"]
        );
    }
    assert!(!response.as_object().unwrap().contains_key("result"));
    assert_ne!(response_id(response), Some(&Value::Null));
}

fn result<'a>(response: &'a Value, id: &Value) -> &'a Value {
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response_id(response), Some(id));
    assert!(!response.as_object().unwrap().contains_key("error"));
    &response["result"]
}

fn rust_identifiers(source: &str) -> Vec<&str> {
    source
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .filter(|token| !token.is_empty())
        .collect()
}

fn rust_code_tokens(source: &str) -> Vec<&str> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at].is_ascii_whitespace() {
            at += 1;
        } else if bytes[at..].starts_with(b"//") {
            at += bytes[at..]
                .iter()
                .position(|byte| *byte == b'\n')
                .unwrap_or(bytes.len() - at);
        } else if bytes[at..].starts_with(b"/*") {
            let mut depth = 1;
            at += 2;
            while at < bytes.len() && depth != 0 {
                if bytes[at..].starts_with(b"/*") {
                    depth += 1;
                    at += 2;
                } else if bytes[at..].starts_with(b"*/") {
                    depth -= 1;
                    at += 2;
                } else {
                    at += 1;
                }
            }
        } else {
            let raw_at = at + usize::from(matches!(bytes[at], b'b' | b'c'));
            if bytes.get(raw_at) == Some(&b'r') {
                let mut quote = raw_at + 1;
                while bytes.get(quote) == Some(&b'#') {
                    quote += 1;
                }
                if bytes.get(quote) == Some(&b'"') {
                    let hashes = quote - raw_at - 1;
                    at = quote + 1;
                    while at < bytes.len() {
                        if bytes[at] == b'"'
                            && bytes.get(at + 1..at + 1 + hashes) == Some(&bytes[raw_at + 1..quote])
                        {
                            at += hashes + 1;
                            break;
                        }
                        at += 1;
                    }
                    continue;
                }
            }
            if bytes[at] == b'"' {
                at += 1;
                while at < bytes.len() {
                    match bytes[at] {
                        b'\\' => at += usize::min(2, bytes.len() - at),
                        b'"' => {
                            at += 1;
                            break;
                        }
                        _ => at += 1,
                    }
                }
            } else if bytes[at] == b'\'' {
                let mut end = at + 1;
                if bytes.get(end) == Some(&b'\\') {
                    end += 2;
                    if bytes.get(end - 1) == Some(&b'x') {
                        end += 2;
                    } else if bytes.get(end - 1) == Some(&b'u') && bytes.get(end) == Some(&b'{') {
                        end += bytes[end..]
                            .iter()
                            .position(|byte| *byte == b'}')
                            .unwrap_or(0)
                            + 1;
                    }
                } else if end < bytes.len() {
                    end += source[end..].chars().next().unwrap().len_utf8();
                }
                if bytes.get(end) == Some(&b'\'') {
                    at = end + 1;
                } else {
                    tokens.push(&source[at..at + 1]);
                    at += 1;
                }
            } else if bytes[at].is_ascii_alphanumeric() || bytes[at] == b'_' {
                let start = at;
                at += 1;
                while at < bytes.len() && (bytes[at].is_ascii_alphanumeric() || bytes[at] == b'_') {
                    at += 1;
                }
                tokens.push(&source[start..at]);
            } else {
                let width = source[at..].chars().next().unwrap().len_utf8();
                tokens.push(&source[at..at + width]);
                at += width;
            }
        }
    }
    tokens
}

fn raw_identifier_end(tokens: &[&str], at: usize, expected: &str) -> Option<usize> {
    if tokens.get(at) == Some(&expected) {
        Some(at + 1)
    } else if tokens.get(at..at + 3) == Some(&["r", "#", expected]) {
        Some(at + 3)
    } else {
        None
    }
}

fn closed_route_violation(sources: &[&str]) -> Option<String> {
    const FORBIDDEN_IDENTIFIERS: &[&str] = &[
        "File",
        "OpenOptions",
        "Path",
        "PathBuf",
        "path",
        "exists",
        "try_exists",
        "TcpListener",
        "TcpStream",
        "UdpSocket",
        "UnixListener",
        "UnixStream",
        "Command",
        "Stdio",
        "Child",
        "Library",
        "libloading",
        "dlopen",
        "fs",
        "net",
        "process",
        "read_to_string",
        "read_dir",
        "write_all_at",
        "var",
        "var_os",
        "vars",
        "vars_os",
        "env",
        "option_env",
        "credential",
        "credentials",
        "api_key",
        "unsafe",
        "extern",
        "include",
        "include_bytes",
        "include_str",
        "concat_idents",
        "paste",
        "eqiora_api",
        "eqiora_compiler",
        "ToolRegistry",
        "registry",
        "reqwest",
        "hyper",
        "ureq",
    ];
    for source in sources {
        let source = source.replace("env!(\"CARGO_PKG_VERSION\")", "");
        let tokens = rust_code_tokens(&source);
        if tokens.windows(2).any(|pair| pair == ["macro_rules", "!"]) {
            return Some("production macro definition".to_owned());
        }
        for at in 0..tokens.len() {
            let after_colons = (tokens.get(at..at + 2) == Some(&[":", ":"]))
                .then(|| at + 2 + usize::from(tokens.get(at + 2) == Some(&"{")));
            if after_colons
                .and_then(|start| raw_identifier_end(&tokens, start, "compiler"))
                .is_some()
            {
                return Some("lower compiler namespace".to_owned());
            }
            if let Some(after_compiler) = raw_identifier_end(&tokens, at, "compiler") {
                if tokens.get(after_compiler..after_compiler + 2) == Some(&[":", ":"])
                    && raw_identifier_end(&tokens, after_compiler + 2, "compile").is_some()
                {
                    return Some("lower compiler invocation".to_owned());
                }
                if tokens[..at]
                    .iter()
                    .rev()
                    .take_while(|token| **token != ";")
                    .any(|token| *token == "use")
                {
                    return Some("lower compiler import".to_owned());
                }
            }
        }
        for identifier in tokens {
            if FORBIDDEN_IDENTIFIERS.contains(&identifier) {
                return Some(identifier.to_owned());
            }
        }
    }
    None
}

fn generic_map_registry(source: &str) -> Option<&str> {
    const MAP_TYPES: &[&str] = &["BTreeMap", "HashMap", "IndexMap", "DashMap"];
    rust_identifiers(source)
        .into_iter()
        .find(|identifier| MAP_TYPES.contains(identifier))
}

fn cfg_test_immediately_gates(source: &str, item: &str) -> bool {
    let compact = source
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect::<String>();
    compact.contains(&format!("#[cfg(test)]{item}"))
}

#[test]
fn frozen_snapshots_and_static_route_are_closed() {
    let expected = expected();
    let frozen = frozen_expected();
    assert_eq!(
        expected["schema"],
        "eqiora.verify.mcp-stdio-compile-check/v1"
    );
    assert_frozen_release_identity(&frozen);
    assert_eq!(env!("CARGO_PKG_VERSION"), expected["serverInfo"]["version"]);
    assert_eq!(
        MODELS_README_SOURCE,
        "# Models\n\nStructural placeholder required by the generic verification case layout.\nThis transport-only case owns no Model fixture in this directory. Its raw\nprogress-token, protocol-reflection, and sensitive-payload witnesses exercise\nthe ordinary stdio path directly and are not Model data.\n"
    );
    assert_eq!(
        REFERENCES_README_SOURCE,
        "# References\n\nThe final metadata schema authority is upstream commit\n`5f5440bb26a62e2cf3440b92da5a667efa03b267`, tag `2026-07-28`, file\n`schema/2026-07-28/schema.json`, SHA-256\n`ef70b61f99b6d2e5e3b46863822eab08dff6a45bedc7a08914e0e5b133f40203`.\nThe exact raw-number witnesses live in `expected/contract.json`; this\ntransport-only case owns no copied reference fixture in this directory.\n"
    );
    let _operation: fn(&str, &str) -> Result<ModelDocument, Vec<Diagnostic>> =
        ModelDocument::compile;

    let production = [MAIN_SOURCE, FRAMING_SOURCE, PROTOCOL_SOURCE, TOOL_SOURCE];
    assert_eq!(
        production
            .iter()
            .map(|source| source.matches("ModelDocument::compile").count())
            .sum::<usize>(),
        expected["staticRoute"]["productionCompileCallCount"]
            .as_u64()
            .unwrap() as usize
    );
    assert_eq!(TOOL_SOURCE.matches("ModelDocument::compile").count(), 1);
    let accepted_facade =
        rust_identifiers(expected["staticRoute"]["acceptedFacade"].as_str().unwrap());
    assert_eq!(
        accepted_facade,
        ["eqiora", "api", "ModelDocument", "compile"]
    );
    let tool_identifiers = rust_identifiers(TOOL_SOURCE);
    assert!(
        tool_identifiers
            .windows(3)
            .any(|window| window == &accepted_facade[..3])
    );
    assert_ne!(
        canonical_json("{ \"mutant\": \"\\u0061\" }"),
        "{ \"mutant\": \"\\u0061\" }"
    );
    assert_eq!(closed_route_violation(&production), None);
    assert!(
        production
            .iter()
            .all(|source| generic_map_registry(source).is_none())
    );
    for forbidden in ["execute_compile_v2", "CompileRequestV2", "pyo3", "tauri"] {
        assert!(production.iter().all(|source| !source.contains(forbidden)));
    }
    assert_eq!(
        production
            .iter()
            .map(|source| source.matches("env!").count())
            .sum::<usize>(),
        1
    );
    assert!(MAIN_SOURCE.contains("env!(\"CARGO_PKG_VERSION\")"));
    assert!(cfg_test_immediately_gates(MAIN_SOURCE, "modoracle;"));
    assert!(cfg_test_immediately_gates(MAIN_SOURCE, "fnrun_for_oracle"));
    assert_eq!(MAIN_SOURCE.matches("mod oracle;").count(), 1);
    assert_eq!(MAIN_SOURCE.matches("run_for_oracle").count(), 1);
    assert_eq!(MAIN_SOURCE.matches("OracleControl").count(), 1);
    assert!(ORACLE_SOURCE.contains("pub(super) struct OracleControl"));
    assert_eq!(ORACLE_SOURCE.matches("mod worker_outcomes;").count(), 1);
    for mutant in [
        "use std::{fs::File}; fn f() { let _ = File::open(\"x\"); }",
        "use std::net as transport; fn f() { let _ = transport::TcpStream::connect(\"x\"); }",
        "use std::process as launch; fn f() { let _ = launch::Command::new(\"x\"); }",
        "fn f() { let _ = std::env::var_os(\"TOKEN\"); }",
        "fn f() { for secret in std::env::vars() { drop(secret); } }",
        "fn f(filename: &str) { let _ = std::path::Path::new(filename).exists(); }",
        "fn f(filename: &str) { let _ = std::path::Path::new(filename).try_exists(); }",
        "fn f() { let _ = eqiora::compiler::compile(\"x\", \"y\"); }",
        "fn f() { let _ = eqiora_api::ModelDocument::compile(\"x\", \"y\"); }",
        "fn f() { let _ = eqiora_compiler::compile(\"x\", \"y\"); }",
        "use eqiora::compiler as lower; fn f() { let _ = lower::compile(\"x\", \"y\"); let _ = eqiora::api::ModelDocument::compile(\"x\", \"y\"); }",
        "use eqiora::{compiler as lower}; fn f() { let _ = lower::compile(\"x\", \"y\"); }",
        "use eqiora::*; fn f() { let _ = compiler::compile(\"x\", \"y\"); let _ = eqiora::api::ModelDocument::compile(\"x\", \"y\"); }",
        "use eqiora::*; use compiler as lower; fn f() { let _ = lower::compile(\"x\", \"y\"); }",
        "fn f() { let _ = eqiora::/* route */compiler::compile(\"x\", \"y\"); }",
        "fn f() { let _ = eqiora::/* outer /* inner */ route */compiler::compile(\"x\", \"y\"); }",
        "fn f() { let _ = eqiora::// route\ncompiler::compile(\"x\", \"y\"); }",
        "use eqiora::{/* route */ compiler as lower}; fn f() { let _ = lower::compile(\"x\", \"y\"); }",
        r##"const S: &str = r#"/* not a comment */"#; const Q: char = '"'; fn f() { let _ = eqiora::compiler::compile("x", "y"); }"##,
        r##"macro_rules! lower { ($module:ident, $function:ident) => { eqiora::$module::$function("x", "y") }; } fn f() { let _ = lower!(compiler, compile); let _ = eqiora::api::ModelDocument::compile("x", "y"); }"##,
        r##"macro_rules /* comment */ ! r#lower { ($module:ident, $function:ident) => { eqiora::$module::$function("x", "y") }; } fn f() { let _ = r#lower!(compiler, compile); }"##,
        "use eqiora_api as lower; fn f() { let _ = lower::ModelDocument::compile(\"x\", \"y\"); }",
        "unsafe extern \"C\" { fn dlopen(); }",
    ] {
        assert!(closed_route_violation(&[mutant]).is_some());
    }
    assert_eq!(
        closed_route_violation(&[
            r##"const URL: &str = "https://host/::compiler/macro_rules!"; const RAW: &str = r#"/* compiler::compile */"#;"##
        ]),
        None
    );
    for mutant in [
        "let tools = std::collections::BTreeMap::<String, fn()>::new();",
        "let tools = std::collections::HashMap::<String, fn()>::new();",
    ] {
        assert!(generic_map_registry(mutant).is_some());
    }
    assert!(!cfg_test_immediately_gates(
        "fn run_for_oracle() {}",
        "fnrun_for_oracle"
    ));
    for (name, source) in [
        ("main.rs", MAIN_SOURCE),
        ("framing.rs", FRAMING_SOURCE),
        ("protocol.rs", PROTOCOL_SOURCE),
        ("tool.rs", TOOL_SOURCE),
    ] {
        assert!(source.lines().count() < 1000, "{name} crossed 1,000 lines");
    }
    assert!(ORACLE_SOURCE.lines().count() < 1000);
    assert!(WORKER_OUTCOMES_SOURCE.lines().count() < 1000);
    assert!(include_str!("mcp_stdio_compile_check.rs").lines().count() < 2000);
}

#[test]
fn discovery_and_listing_advertise_the_current_public_model_schema() {
    let expected = expected();
    let mut client = Client::spawn();

    client.send_value(&discover_request(json!("discover-lf")));
    let discover_raw = client.recv_raw();
    let discover: Value = serde_json::from_slice(&discover_raw).unwrap();
    assert_eq!(
        result(&discover, &json!("discover-lf")),
        &expected["serverDiscover"]
    );
    let discover_members = raw_object_members(std::str::from_utf8(&discover_raw).unwrap().trim());
    assert_raw_order(&discover_members, &["jsonrpc", "id", "result"]);
    let discover_result = raw_object_members(raw_member(&discover_members, "result"));
    assert_raw_order(
        &discover_result,
        &[
            "resultType",
            "supportedVersions",
            "capabilities",
            "_meta",
            "instructions",
            "ttlMs",
            "cacheScope",
        ],
    );

    client.send_value_crlf(&list_request(json!("list-crlf")));
    let list_raw = client.recv_raw();
    let list: Value = serde_json::from_slice(&list_raw).unwrap();
    let listed = result(&list, &json!("list-crlf"));
    assert_eq!(listed["resultType"], expected["toolsList"]["resultType"]);
    let tools = listed["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_current_tool_schema(&tools[0]);
    assert_eq!(listed["ttlMs"], expected["toolsList"]["ttlMs"]);
    assert_eq!(listed["cacheScope"], expected["toolsList"]["cacheScope"]);
    assert_eq!(listed["_meta"], expected["serverDiscover"]["_meta"]);
    assert!(!listed.as_object().unwrap().contains_key("nextCursor"));
    assert_eq!(listed.as_object().unwrap().len(), 5);
    let list_members = raw_object_members(std::str::from_utf8(&list_raw).unwrap().trim());
    assert_raw_order(&list_members, &["jsonrpc", "id", "result"]);
    let list_result = raw_object_members(raw_member(&list_members, "result"));
    assert_raw_order(
        &list_result,
        &["resultType", "tools", "ttlMs", "cacheScope", "_meta"],
    );

    let stderr = client.shutdown();
    assert!(
        stderr.is_empty(),
        "ordinary server wrote stderr: {stderr:?}"
    );
}

fn assert_closed_object(value: &Value, required: &Value) {
    let object = value.as_object().expect("closed object");
    let mut actual = object.keys().map(String::as_str).collect::<Vec<_>>();
    let mut expected = required
        .as_array()
        .unwrap()
        .iter()
        .map(|member| member.as_str().unwrap())
        .collect::<Vec<_>>();
    actual.sort_unstable();
    expected.sort_unstable();
    assert_eq!(actual, expected);
}

fn assert_result_envelope(result: &Value, accepted: bool) {
    let expected = expected();
    assert_exact_members(
        result,
        &[
            "resultType",
            "content",
            "structuredContent",
            "isError",
            "_meta",
        ],
    );
    assert_eq!(result["resultType"], "complete");
    assert_eq!(result["isError"], !accepted);
    assert_eq!(result["_meta"], expected["serverDiscover"]["_meta"]);
    let content = result["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_exact_members(&content[0], &["type", "text"]);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0].as_object().unwrap().len(), 2);
    let text = content[0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).expect("TextContent compact JSON");
    assert_eq!(parsed, result["structuredContent"]);
}

fn assert_current_tool_schema(tool: &Value) {
    assert_eq!(tool["name"], "eqiora.model.compile_check");
    assert_eq!(tool["inputSchema"]["additionalProperties"], false);
    assert_eq!(tool["inputSchema"]["required"], json!(["source"]));
    let public_schema: Value =
        serde_json::from_str(eqiora::control::COMPILE_V2_SCHEMA_JSON).unwrap();
    let advertised = &tool["outputSchema"]["oneOf"][0]["properties"]["model"];
    for field in ["schema", "transactionSchema"] {
        assert_eq!(
            advertised["properties"][field],
            public_schema["$defs"]["model"]["properties"][field]
        );
    }
    assert_eq!(
        advertised["properties"]["structuralFingerprint"]["properties"]["generation"]["enum"],
        json!([SemanticFingerprintGeneration::V22.as_str()])
    );
}

fn assert_accepted_output_schema(structured: &Value, tool: &Value) {
    let schema = &tool["outputSchema"]["oneOf"][0];
    assert_closed_object(structured, &schema["required"]);
    assert_eq!(
        structured["schema"],
        schema["properties"]["schema"]["const"]
    );
    assert_eq!(structured["status"], "accepted");
    let model_schema = &schema["properties"]["model"];
    let model = &structured["model"];
    assert_closed_object(model, &model_schema["required"]);
    assert_eq!(
        model["schema"],
        model_schema["properties"]["schema"]["const"]
    );
    assert_eq!(
        model["transactionSchema"],
        model_schema["properties"]["transactionSchema"]["const"]
    );
    assert_hex_digest(model["digest"].as_str().unwrap());
    let model_id = model["modelId"].as_str().unwrap();
    assert!(!model_id.is_empty() && model_id.chars().count() <= 128);
    assert!(model["semanticRevision"].as_u64().is_some());
    let fingerprint = &model["structuralFingerprint"];
    assert_closed_object(
        fingerprint,
        &model_schema["properties"]["structuralFingerprint"]["required"],
    );
    assert!(
        model_schema["properties"]["structuralFingerprint"]["properties"]["generation"]["enum"]
            .as_array()
            .unwrap()
            .contains(&fingerprint["generation"])
    );
    assert_hex_digest(fingerprint["digest"].as_str().unwrap());
}

fn assert_hex_digest(value: &str) {
    assert_eq!(value.len(), 64);
    assert!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    );
}

fn normalized_diagnostic(diagnostic: &Diagnostic) -> Value {
    let severity = match diagnostic.severity() {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Note => "note",
    };
    json!({
        "source": "kernel",
        "severity": severity,
        "code": diagnostic.code().to_string(),
        "message": diagnostic.message(),
        "graphPath": diagnostic.graph_path().map(|path| path.segments()),
        "span": diagnostic.source_span().map(|span| json!({
            "file": span.file,
            "start": span.start,
            "end": span.end
        })),
        "patch": diagnostic.suggestion().map(|patch| json!({"summary": patch.summary}))
    })
}

fn assert_rejected_output_schema(structured: &Value, tool: &Value) {
    let schema = &tool["outputSchema"]["oneOf"][1];
    assert_closed_object(structured, &schema["required"]);
    assert_eq!(
        structured["schema"],
        schema["properties"]["schema"]["const"]
    );
    assert_eq!(structured["status"], "rejected");
    let diagnostics = structured["diagnostics"].as_array().unwrap();
    assert!(!diagnostics.is_empty() && diagnostics.len() <= 1024);
    let item = &schema["properties"]["diagnostics"]["items"];
    for diagnostic in diagnostics {
        assert_closed_object(diagnostic, &item["required"]);
        assert!(
            item["properties"]["source"]["enum"]
                .as_array()
                .unwrap()
                .contains(&diagnostic["source"])
        );
        assert!(
            item["properties"]["severity"]["enum"]
                .as_array()
                .unwrap()
                .contains(&diagnostic["severity"])
        );
        let code = diagnostic["code"].as_str().unwrap();
        assert_eq!(code.len(), 6);
        assert!(code[..2].bytes().all(|byte| byte.is_ascii_uppercase()));
        assert!(code[2..].bytes().all(|byte| byte.is_ascii_digit()));
        let message = diagnostic["message"].as_str().unwrap();
        assert!(!message.is_empty() && message.chars().count() <= 1_048_576);
        assert!(message.len() <= 1_048_576);
        if let Some(path) = diagnostic["graphPath"].as_array() {
            assert!(path.len() <= 256);
            for segment in path {
                let segment = segment.as_str().unwrap();
                assert!(!segment.is_empty());
                assert!(segment.chars().count() <= 4096 && segment.len() <= 4096);
            }
        }
        if !diagnostic["span"].is_null() {
            assert_exact_members(&diagnostic["span"], &["file", "start", "end"]);
        }
        if !diagnostic["patch"].is_null() {
            assert_exact_members(&diagnostic["patch"], &["summary"]);
        }
    }
}

#[test]
fn accepted_and_rejected_calls_preserve_direct_operation_meaning() {
    let fixture: Value = serde_json::from_slice(ACCEPTED_CONTROL_REQUEST).unwrap();
    let filename = fixture["filename"].as_str().unwrap();
    let source = fixture["source"].as_str().unwrap();
    let direct = ModelDocument::compile(filename, source).expect("direct accepted witness");
    let direct_reference = direct.artifact_reference().unwrap();
    let direct_fingerprint = direct.structural_fingerprint().unwrap();
    assert_eq!(
        direct_fingerprint.generation(),
        SemanticFingerprintGeneration::V22
    );
    let mut client = Client::spawn();
    client.send_value(&list_request(json!("parity-tool")));
    let listing = client.recv();
    let tools = result(&listing, &json!("parity-tool"))["tools"]
        .as_array()
        .unwrap();
    assert_eq!(tools.len(), 1);
    let tool = &tools[0];
    assert_current_tool_schema(tool);
    client.send_value(&call_request(
        json!("accepted-parity"),
        json!({"filename": filename, "source": source}),
    ));
    let accepted_raw = client.recv_raw();
    let accepted: Value = serde_json::from_slice(&accepted_raw).unwrap();
    let accepted_result = result(&accepted, &json!("accepted-parity"));
    assert_result_envelope(accepted_result, true);
    let structured = &accepted_result["structuredContent"];
    assert_accepted_output_schema(structured, tool);
    let model = &structured["model"];
    assert_eq!(
        model["semanticRevision"],
        direct_reference.semantic_revision().get()
    );
    assert_eq!(
        model["structuralFingerprint"]["generation"],
        direct_fingerprint.generation().as_str()
    );
    assert_eq!(
        model["structuralFingerprint"]["digest"],
        direct_fingerprint.digest()
    );
    assert_eq!(model["modelId"], direct_reference.model().to_string());
    assert_eq!(model["digest"], direct_reference.artifact().as_str());
    let direct_rejected = ModelDocument::compile("empty.eqi", "").unwrap_err();
    client.send_value(&call_request(
        json!("rejected-parity"),
        json!({"filename": "empty.eqi", "source": ""}),
    ));
    let rejected = client.recv();
    let rejected_result = result(&rejected, &json!("rejected-parity"));
    assert_result_envelope(rejected_result, false);
    let rejected_structured = &rejected_result["structuredContent"];
    assert_rejected_output_schema(rejected_structured, tool);
    let expected_diagnostics = direct_rejected
        .iter()
        .map(normalized_diagnostic)
        .collect::<Vec<_>>();
    assert_eq!(
        rejected_structured["diagnostics"].as_array().unwrap(),
        &expected_diagnostics
    );
    assert_eq!(rejected_structured["diagnostics"][0]["code"], "EQ0602");
    assert!(
        !rejected_structured
            .as_object()
            .unwrap()
            .contains_key("model")
    );
    let stderr = client.shutdown();
    assert!(
        stderr.is_empty(),
        "ordinary server wrote stderr: {stderr:?}"
    );
}

#[test]
fn framing_and_json_rpc_envelopes_fail_closed_with_safe_id_echo() {
    let mut client = Client::spawn();
    for bytes in [b"{\xff}\n".as_slice(), b"{\n".as_slice()] {
        client.send_raw(bytes);
        let response = client.recv();
        assert_error(&response, -32700, None);
    }
    for value in [json!([]), json!(1), json!(null)] {
        client.send_value(&value);
        let response = client.recv();
        assert_error(&response, -32600, None);
    }
    client.send_raw(b"{\"jsonrpc\":\"2.0\",\"id\":\"raw-cr\",\r\"method\":\"server/discover\"}\n");
    let raw_cr = client.recv();
    assert!(matches!(
        raw_cr["error"]["code"].as_i64(),
        Some(-32700) | Some(-32600)
    ));
    assert_ne!(raw_cr.get("id"), Some(&Value::Null));
    let duplicate_nested = format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":\"duplicate-nested\",\"method\":\"server/discover\",\"params\":{{\"_meta\":{{\"io.modelcontextprotocol/protocolVersion\":\"{}\",\"io.modelcontextprotocol/clientCapabilities\":{{}},\"com.example/marker\":1,\"com.example/marker\":2}}}}}}\n",
        expected()["protocolVersion"].as_str().unwrap()
    );
    client.send_raw(duplicate_nested.as_bytes());
    let duplicate = client.recv();
    assert_error(&duplicate, -32600, Some(&json!("duplicate-nested")));
    client.send_raw(
        b"{\"jsonrpc\":\"2.0\",\"id\":\"first\",\"id\":\"second\",\"method\":\"server/discover\",\"params\":{}}\n",
    );
    let duplicate_id = client.recv();
    assert_error(&duplicate_id, -32600, None);
    for (id, array_levels, accepted) in [("depth-64", 61, true), ("depth-65", 62, false)] {
        let mut nested = json!(0);
        for _ in 0..array_levels {
            nested = Value::Array(vec![nested]);
        }
        client.send_value(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "server/discover",
            "params": {
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": expected()["protocolVersion"],
                    "io.modelcontextprotocol/clientCapabilities": {},
                    "com.example/depth": nested,
                }
            }
        }));
        let response = client.recv();
        if accepted {
            assert_eq!(result(&response, &json!(id)), &expected()["serverDiscover"]);
        } else {
            assert_error(&response, -32600, Some(&json!(id)));
        }
    }
    for (id, value) in [
        (
            Some(json!("bad-version")),
            json!({"jsonrpc":"1.0","id":"bad-version","method":"server/discover","params":{"_meta":request_meta()}}),
        ),
        (
            Some(json!("unknown-top")),
            json!({"jsonrpc":"2.0","id":"unknown-top","method":"server/discover","params":{"_meta":request_meta()},"extra":true}),
        ),
        (
            None,
            json!({"jsonrpc":"2.0","method":"server/discover","params":{"_meta":request_meta()}}),
        ),
        (
            None,
            json!({"jsonrpc":"2.0","id":null,"method":"server/discover","params":{"_meta":request_meta()}}),
        ),
        (
            None,
            json!({"jsonrpc":"2.0","id":1.5,"method":"server/discover","params":{"_meta":request_meta()}}),
        ),
        (
            None,
            json!({"jsonrpc":"2.0","id":true,"method":"server/discover","params":{"_meta":request_meta()}}),
        ),
    ] {
        client.send_value(&value);
        let response = client.recv();
        assert_error(&response, -32600, id.as_ref());
    }
    for id in [
        json!(""),
        json!("a".repeat(128)),
        json!("é".repeat(64)),
        json!(i64::MIN),
        json!(i64::MAX),
    ] {
        client.send_value(&discover_request(id.clone()));
        let response = client.recv();
        assert_eq!(result(&response, &id), &expected()["serverDiscover"]);
    }
    client.send_value(&discover_request(json!("é".repeat(65))));
    let overlong_id = client.recv();
    assert_error(&overlong_id, -32600, None);
    client.send_value(&discover_request(json!("a".repeat(129))));
    assert_error(&client.recv(), -32600, None);
    client.send_value(&json!({
        "jsonrpc":"2.0",
        "id":9223372036854775808_u64,
        "method":"server/discover",
        "params":{"_meta":request_meta()}
    }));
    let unsigned_id = client.recv();
    assert_error(&unsigned_id, -32600, None);
    client.send_raw(b"{\"jsonrpc\":\"2.0\",\"id\":-9223372036854775809,\"method\":\"server/discover\",\"params\":{}}\n");
    assert_error(&client.recv(), -32600, None);
    let stderr = client.shutdown();
    assert!(
        stderr.is_empty(),
        "malformed inputs leaked to stderr: {stderr:?}"
    );
}

#[test]
fn overlong_line_is_drained_and_the_next_frame_still_works() {
    let mut client = Client::spawn();
    let maximum = expected()["framing"]["maximumEncodedLineBytes"]
        .as_u64()
        .unwrap() as usize;
    let mut line = serde_json::to_vec(&discover_request(json!("exact-line-limit"))).unwrap();
    line.resize(maximum, b' ');
    line.push(b'\n');
    client.send_raw(&line);
    let exact = client.recv_with(timeout("largeResponseMs"));
    assert_eq!(
        result(&exact, &json!("exact-line-limit")),
        &expected()["serverDiscover"]
    );
    let mut line = vec![b' '; maximum + 1];
    line.push(b'\n');
    client.send_raw(&line);
    let response = client.recv_with(timeout("largeResponseMs"));
    assert_error(&response, -32600, None);
    assert_eq!(
        response["error"]["message"],
        expected()["framing"]["overlongMessage"]
    );
    client.send_value(&discover_request(json!("after-overlong")));
    let after = client.recv();
    assert_eq!(
        result(&after, &json!("after-overlong")),
        &expected()["serverDiscover"]
    );
    assert!(client.shutdown().is_empty());
}

#[test]
fn method_metadata_and_call_errors_obey_the_frozen_stage_precedence() {
    let expected = expected();
    let mut client = Client::spawn();
    client
        .send_value(&json!({"jsonrpc":"2.0","id":"initialize","method":"initialize","params":{}}));
    let initialize = client.recv();
    assert_error(&initialize, -32601, Some(&json!("initialize")));
    let message = initialize["error"]["message"].as_str().unwrap();
    assert!(message.contains(expected["protocolVersion"].as_str().unwrap()));
    assert!(message.to_ascii_lowercase().contains("only"));
    client.send_value(
        &json!({"jsonrpc":"2.0","id":"unknown-missing-meta","method":"unknown","params":{}}),
    );
    let metadata_first = client.recv();
    assert_error(
        &metadata_first,
        -32602,
        Some(&json!("unknown-missing-meta")),
    );
    let mut unknown_params = Map::new();
    unknown_params.insert("extra".to_owned(), json!(true));
    client.send_value(&request(json!("unknown"), "unknown", unknown_params));
    let unknown = client.recv();
    assert_error(&unknown, -32601, Some(&json!("unknown")));
    client.send_value(
        &json!({"jsonrpc":"2.0","id":"missing-meta","method":"server/discover","params":{}}),
    );
    let missing_meta = client.recv();
    assert_error(&missing_meta, -32602, Some(&json!("missing-meta")));
    let mut unsupported_meta = request_meta();
    unsupported_meta["io.modelcontextprotocol/protocolVersion"] =
        expected["unsupportedVersion"]["witness"].clone();
    client.send_value(&json!({
        "jsonrpc":"2.0",
        "id":"unsupported",
        "method":"server/discover",
        "params":{"_meta":unsupported_meta}
    }));
    let unsupported = client.recv();
    assert_error(&unsupported, -32022, Some(&json!("unsupported")));
    assert_eq!(
        unsupported["error"],
        json!({
            "code": expected["unsupportedVersion"]["code"],
            "message": expected["unsupportedVersion"]["message"],
            "data": expected["unsupportedVersion"]["data"]
        })
    );
    let mut discover_extra = Map::new();
    discover_extra.insert("extra".to_owned(), Value::Null);
    client.send_value(&request(
        json!("discover-extra"),
        "server/discover",
        discover_extra,
    ));
    assert_error(&client.recv(), -32602, Some(&json!("discover-extra")));
    let mut list_cursor = Map::new();
    list_cursor.insert("cursor".to_owned(), Value::Null);
    client.send_value(&request(json!("cursor"), "tools/list", list_cursor));
    assert_error(&client.recv(), -32602, Some(&json!("cursor")));
    let mut unknown_tool = Map::new();
    unknown_tool.insert("name".to_owned(), json!("eqiora.unknown"));
    unknown_tool.insert("arguments".to_owned(), json!({}));
    client.send_value(&request(json!("unknown-tool"), "tools/call", unknown_tool));
    let unknown_tool_response = client.recv();
    assert_error(&unknown_tool_response, -32602, Some(&json!("unknown-tool")));
    assert_eq!(
        unknown_tool_response["error"]["message"],
        format!(
            "{}{}",
            expected["callAdmission"]["unknownNameMessagePrefix"]
                .as_str()
                .unwrap(),
            "eqiora.unknown"
        )
    );
    for (id, name) in [
        ("empty-tool", "".to_owned()),
        ("long-tool", "a".repeat(129)),
        ("slash-tool", "eqiora/tool".to_owned()),
        ("unicode-tool", "eqiora.é".to_owned()),
    ] {
        let mut params = Map::new();
        params.insert("name".to_owned(), json!(name.clone()));
        params.insert("arguments".to_owned(), json!({}));
        client.send_value(&request(json!(id), "tools/call", params));
        let response = client.recv();
        assert_error(&response, -32602, Some(&json!(id)));
        assert_eq!(
            response["error"],
            expected["callAdmission"]["malformedNameError"]
        );
        if !name.is_empty() {
            assert!(!serde_json::to_string(&response).unwrap().contains(&name));
        }
    }
    for (id, name) in [
        ("one-character-tool", "a".to_owned()),
        ("max-tool", "a".repeat(128)),
    ] {
        let mut params = Map::new();
        params.insert("name".to_owned(), json!(name.clone()));
        params.insert("arguments".to_owned(), json!({}));
        client.send_value(&request(json!(id), "tools/call", params));
        let response = client.recv();
        assert_error(&response, -32602, Some(&json!(id)));
        assert_eq!(
            response["error"]["message"],
            format!(
                "{}{}",
                expected["callAdmission"]["unknownNameMessagePrefix"]
                    .as_str()
                    .unwrap(),
                name
            )
        );
    }
    for (id, params) in [
        (
            "non-object-arguments",
            json!({"name": expected["toolName"], "arguments": []}),
        ),
        (
            "input-responses",
            json!({"name": expected["toolName"], "arguments": {}, "inputResponses": []}),
        ),
        (
            "request-state",
            json!({"name": expected["toolName"], "arguments": {}, "requestState": {}}),
        ),
    ] {
        client.send_value(&request(
            json!(id),
            "tools/call",
            params.as_object().unwrap().clone(),
        ));
        assert_error(&client.recv(), -32602, Some(&json!(id)));
    }
    let mut absent_arguments = Map::new();
    absent_arguments.insert("name".to_owned(), expected["toolName"].clone());
    client.send_value(&request(
        json!("absent-arguments"),
        "tools/call",
        absent_arguments,
    ));
    let absent = client.recv();
    assert_input_error(&absent, &json!("absent-arguments"));
    client.send_value(
        &json!({"jsonrpc":"2.0","method":"notifications/unknown","params":{"marker":"ignored"}}),
    );
    client.send_value(&discover_request(json!("after-notification")));
    let after = client.recv();
    assert_eq!(
        result(&after, &json!("after-notification")),
        &expected["serverDiscover"]
    );
    client.assert_silent();
    assert!(client.shutdown().is_empty());
}

fn assert_input_error(response: &Value, id: &Value) {
    let result = result(response, id);
    assert_result_envelope(result, false);
    assert_eq!(result["structuredContent"]["status"], "rejected");
    assert_eq!(
        result["structuredContent"]["schema"],
        "eqiora.mcp.compile-check-result/v1"
    );
    assert_eq!(
        result["structuredContent"]["diagnostics"],
        json!([expected()["inputDiagnostic"]])
    );
    assert!(
        !result["structuredContent"]
            .as_object()
            .unwrap()
            .contains_key("model")
    );
}

#[test]
fn tool_input_schema_and_utf8_bounds_fail_as_complete_results() {
    let expected = expected();
    let mut client = Client::spawn();
    let violations = [
        ("missing-source", json!({})),
        ("empty-filename", json!({"filename":"", "source":"x"})),
        (
            "control-filename",
            json!({"filename":"bad\nname", "source":"x"}),
        ),
        (
            "filename-4097-ascii",
            json!({"filename":"a".repeat(4097), "source":"x"}),
        ),
        (
            "filename-2049-two-byte",
            json!({"filename":"é".repeat(2049), "source":"x"}),
        ),
        (
            "source-8388609-ascii",
            json!({"filename":"x.eqi", "source":" ".repeat(8_388_609)}),
        ),
        (
            "source-4194305-two-byte",
            json!({"filename":"x.eqi", "source":"é".repeat(4_194_305)}),
        ),
        (
            "additional-argument",
            json!({"filename":"x.eqi", "source":"x", "extra":true}),
        ),
        ("wrong-filename-type", json!({"filename":1, "source":"x"})),
        ("wrong-source-type", json!({"filename":"x.eqi", "source":1})),
    ];
    for (id, arguments) in violations {
        client.send_value(&call_request(json!(id), arguments));
        let response = client.recv_with(timeout("largeResponseMs"));
        assert_input_error(&response, &json!(id));
    }
    let fixture: Value = serde_json::from_slice(ACCEPTED_CONTROL_REQUEST).unwrap();
    let source = fixture["source"].clone();
    for (id, filename) in [
        ("filename-4096-ascii", "a".repeat(4096)),
        ("filename-2048-two-byte", "é".repeat(2048)),
    ] {
        client.send_value(&call_request(
            json!(id),
            json!({"filename":filename, "source":source}),
        ));
        let response = client.recv_with(timeout("largeResponseMs"));
        let result = result(&response, &json!(id));
        assert_result_envelope(result, true);
    }
    for (id, source) in [
        ("source-8388608-ascii", " ".repeat(8_388_608)),
        ("source-4194304-two-byte", "é".repeat(4_194_304)),
    ] {
        client.send_value(&call_request(
            json!(id),
            json!({"filename":"large.eqi", "source":source}),
        ));
        let response = client.recv_with(timeout("largeResponseMs"));
        let result = result(&response, &json!(id));
        assert_eq!(result["resultType"], "complete");
        assert_eq!(result["structuredContent"]["status"], "rejected");
        assert_ne!(
            result["structuredContent"]["diagnostics"],
            json!([expected["inputDiagnostic"]])
        );
    }
    assert!(client.shutdown().is_empty());
}

fn discover_with_meta(client: &mut Client, id: &str, meta: Value) -> Value {
    client.send_value(&json!({
        "jsonrpc":"2.0",
        "id":id,
        "method":"server/discover",
        "params":{"_meta":meta}
    }));
    client.recv()
}

fn discover_with_raw_progress(client: &mut Client, id: &str, progress_token: &str) -> Value {
    let mut request = discover_request(json!(id));
    request["params"]["_meta"]["progressToken"] = json!(0);
    let encoded = serde_json::to_string(&request).unwrap();
    let marker = "\"progressToken\":0";
    assert_eq!(encoded.matches(marker).count(), 1);
    let line = format!(
        "{}\n",
        encoded.replacen(marker, &format!("\"progressToken\":{progress_token}"), 1)
    );
    client.send_raw(line.as_bytes());
    client.recv()
}

#[test]
fn final_request_metadata_is_validated_then_ignored_without_changing_results_or_logs() {
    let expected = expected();
    let mut client = Client::spawn();
    let trace_id = "0af7651916cd43dd8448eb211c80319c";
    let parent_id = "00f067aa0ba902b7";
    let max_tracestate = (0..32)
        .map(|index| format!("k{index}=v"))
        .collect::<Vec<_>>()
        .join(",");
    let max_baggage = (0..180)
        .map(|index| format!("k{index}=v;property=p%25"))
        .collect::<Vec<_>>()
        .join(",");
    let max_simple_key = "a".repeat(256);
    let max_tracestate_value = "v".repeat(256);
    let full = expected["metadata"]["compilePathWitnesses"]["valid"].clone();
    let response = discover_with_meta(&mut client, "full-meta", full);
    assert_eq!(
        result(&response, &json!("full-meta")),
        &expected["serverDiscover"]
    );
    let encoded = serde_json::to_string(&response).unwrap();
    for marker in ["metadata-compile-marker", "metadata-vendor-marker"] {
        assert!(!encoded.contains(marker));
    }

    let valid_variants = [
        (
            "future-trace",
            "traceparent",
            json!(format!("01-{trace_id}-{parent_id}-01-extra-opaque")),
        ),
        ("empty-tracestate", "tracestate", json!("")),
        ("empty-tracestate-member", "tracestate", json!("a=1,,b=2")),
        ("ows-tracestate-member", "tracestate", json!("a=1, \t ,b=2")),
        ("trailing-ows-tracestate", "tracestate", json!("key=value ")),
        ("max-tracestate", "tracestate", json!(max_tracestate)),
        (
            "max-simple-tracestate-key",
            "tracestate",
            json!(format!("{max_simple_key}=v")),
        ),
        (
            "max-tracestate-value",
            "tracestate",
            json!(format!("key={max_tracestate_value}")),
        ),
        ("max-baggage", "baggage", json!(max_baggage)),
        ("literal-percent", "baggage", json!("key=%25")),
        ("empty-baggage-value", "baggage", json!("key=")),
        (
            "duplicate-baggage",
            "baggage",
            json!("key=first,key=second"),
        ),
        ("integer-progress", "progressToken", json!(1)),
    ];
    for (id, key, value) in valid_variants {
        let mut meta = request_meta();
        meta[key] = value;
        let response = discover_with_meta(&mut client, id, meta);
        assert_eq!(result(&response, &json!(id)), &expected["serverDiscover"]);
    }
    let raw_progress = &expected["metadata"]["requestProgressTokenRawNumberWitnesses"];
    for (group, code, echoes_id) in [
        ("validIntegers", None, true),
        ("decoderRejected", Some(-32700), false),
        ("invalidNonIntegers", Some(-32602), true),
    ] {
        for (index, token) in raw_progress[group].as_array().unwrap().iter().enumerate() {
            let id = format!("{group}-{index}");
            let response = discover_with_raw_progress(&mut client, &id, token.as_str().unwrap());
            if let Some(code) = code {
                assert_error(&response, code, echoes_id.then(|| json!(id)).as_ref());
            } else {
                assert_eq!(result(&response, &json!(id)), &expected["serverDiscover"]);
            }
        }
    }
    for level in expected["metadata"]["logLevels"].as_array().unwrap() {
        let id = format!("log-{}", level.as_str().unwrap());
        let mut meta = request_meta();
        meta["io.modelcontextprotocol/logLevel"] = level.clone();
        let response = discover_with_meta(&mut client, &id, meta);
        assert_eq!(result(&response, &json!(id)), &expected["serverDiscover"]);
    }

    let stderr = client.shutdown();
    let stderr = String::from_utf8_lossy(&stderr);
    for marker in [
        "metadata-compile-marker",
        "metadata-vendor-marker",
        trace_id,
    ] {
        assert!(!stderr.contains(marker), "metadata leaked to stderr");
    }
}

#[test]
fn invalid_final_request_metadata_rejects_before_dispatch() {
    let mut client = Client::spawn();
    let trace_id = "0af7651916cd43dd8448eb211c80319c";
    let parent_id = "00f067aa0ba902b7";
    let tenant_too_long = format!("1{}", "a".repeat(241));
    let system_too_long = format!("a{}", "b".repeat(14));
    let too_many_tracestate = (0..33)
        .map(|index| format!("k{index}=v"))
        .collect::<Vec<_>>()
        .join(",");
    for witness in expected()["metadata"]["compilePathWitnesses"]["invalid"]
        .as_array()
        .unwrap()
    {
        let id = witness[0].as_str().unwrap();
        let key = witness[1].as_str().unwrap();
        let mut meta = request_meta();
        if witness.get(3).and_then(Value::as_bool) == Some(true) {
            meta.as_object_mut().unwrap().remove(key);
        } else {
            meta[key] = witness[2].clone();
        }
        let response = discover_with_meta(&mut client, id, meta);
        assert_error(&response, -32602, Some(&json!(id)));
    }

    for (id, key, value) in [
        (
            "trace-zero",
            "traceparent",
            json!("00-00000000000000000000000000000000-00f067aa0ba902b7-01"),
        ),
        (
            "parent-zero",
            "traceparent",
            json!("00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01"),
        ),
        (
            "trace-ff",
            "traceparent",
            json!(format!("ff-{trace_id}-{parent_id}-01")),
        ),
        (
            "future-flags-tail",
            "traceparent",
            json!(format!("01-{trace_id}-{parent_id}-01opaque")),
        ),
        (
            "too-many-tracestate",
            "tracestate",
            json!(too_many_tracestate),
        ),
        (
            "tenant-too-long",
            "tracestate",
            json!(format!("{tenant_too_long}@system=v")),
        ),
        (
            "system-too-long",
            "tracestate",
            json!(format!("tenant@{system_too_long}=v")),
        ),
        ("system-start", "tracestate", json!("tenant@1system=v")),
        (
            "simple-key-too-long",
            "tracestate",
            json!(format!("{}=v", "a".repeat(257))),
        ),
        (
            "tracestate-value-too-long",
            "tracestate",
            json!(format!("key={}", "v".repeat(257))),
        ),
        (
            "tracestate-value-vt",
            "tracestate",
            json!("key=value\u{000b}"),
        ),
        (
            "reserved-long-vendor",
            "org.modelcontextprotocol.api/value",
            json!("marker"),
        ),
        ("bad-vendor-label", "com.-example/value", json!("marker")),
        ("bad-vendor-name", "com.example/-value", json!("marker")),
    ] {
        let mut meta = request_meta();
        meta[key] = value;
        let response = discover_with_meta(&mut client, id, meta);
        assert_error(&response, -32602, Some(&json!(id)));
    }

    for (id, version) in [
        ("version-128-ascii", "x".repeat(128)),
        ("version-64-two-byte", "é".repeat(64)),
    ] {
        let mut meta = request_meta();
        meta["io.modelcontextprotocol/protocolVersion"] = json!(version);
        let response = discover_with_meta(&mut client, id, meta);
        assert_error(&response, -32022, Some(&json!(id)));
    }
    for (id, version) in [
        ("version-129-ascii", "x".repeat(129)),
        ("version-65-two-byte", "é".repeat(65)),
    ] {
        let mut meta = request_meta();
        meta["io.modelcontextprotocol/protocolVersion"] = json!(version);
        let response = discover_with_meta(&mut client, id, meta);
        assert_error(&response, -32602, Some(&json!(id)));
    }
    assert!(client.shutdown().is_empty());
}

#[test]
fn sensitive_application_payloads_do_not_enter_protocol_errors_or_stderr() {
    let mut client = Client::spawn();
    let source_marker = "source-secret-leak-marker";
    let filename_marker = "filename-secret-leak-marker\n";
    client.send_value(&call_request(
        json!("payload-leak"),
        json!({"filename":filename_marker,"source":source_marker,"extra":true}),
    ));
    let response = client.recv();
    assert_input_error(&response, &json!("payload-leak"));
    let encoded = serde_json::to_string(&response).unwrap();
    assert!(!encoded.contains(source_marker));
    assert!(!encoded.contains("filename-secret-leak-marker"));

    client.send_value(&json!({
        "jsonrpc":"2.0",
        "method":"notifications/cancelled",
        "params":{"requestId":"unknown","reason":"cancellation-secret-leak-marker"}
    }));
    client.send_value(&discover_request(json!("after-secret")));
    let after = client.recv();
    assert_eq!(
        result(&after, &json!("after-secret")),
        &expected()["serverDiscover"]
    );
    client.assert_silent();
    let stderr = String::from_utf8(client.shutdown()).unwrap();
    for marker in [
        source_marker,
        "filename-secret-leak-marker",
        "cancellation-secret-leak-marker",
    ] {
        assert!(!stderr.contains(marker), "caller content leaked to stderr");
    }
}

#[test]
fn nonempty_bytes_without_a_final_delimiter_are_discarded_on_shutdown() {
    let mut client = Client::spawn();
    client.send_raw(b"{\"jsonrpc\":\"2.0\",\"id\":\"incomplete\"");
    client.assert_silent();
    assert!(client.shutdown().is_empty());
}

#[test]
fn broken_stdout_is_a_nonzero_payload_safe_exit() {
    let marker = "broken-stdout-caller-marker";
    let read_only_stdout = std::fs::File::open(env!("CARGO_BIN_EXE_eqiora-mcp"))
        .expect("open read-only stdout witness");
    let mut child = Command::new(env!("CARGO_BIN_EXE_eqiora-mcp"))
        .stdin(Stdio::piped())
        .stdout(Stdio::from(read_only_stdout))
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch broken-stdout witness");
    let mut stdin = child.stdin.take().expect("child stdin");
    let mut frame = serde_json::to_vec(&discover_request(json!(marker))).unwrap();
    frame.push(b'\n');
    stdin.write_all(&frame).unwrap();
    stdin.flush().unwrap();
    drop(stdin);

    let deadline = Instant::now() + timeout("shutdownMs");
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll broken-stdout child") {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "broken-stdout child did not exit"
        );
        thread::sleep(Duration::from_millis(10));
    };
    assert!(!status.success(), "broken stdout must cause nonzero exit");
    let mut stderr = Vec::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_end(&mut stderr)
        .unwrap();
    assert!(!String::from_utf8_lossy(&stderr).contains(marker));
}
