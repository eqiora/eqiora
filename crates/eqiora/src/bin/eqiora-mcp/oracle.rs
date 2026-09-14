//! Independent crate-private oracle for the bounded stdio adapter.
//!
//! The parent binary exposes `run_for_oracle` only under `cfg(test)`. The
//! injected compiler, pre-compile gate, and cancellation observer are absent
//! from release builds and do not create a product mode or public hook.

use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use eqiora::api::{ModelDocument, SemanticFingerprintGeneration};
use eqiora::{Code, Diagnostic, GraphPath, Patch, Span};
use serde_json::{Map, Value, json};

use super::run_for_oracle;

mod worker_outcomes;
use worker_outcomes::{
    accepted_source, assert_current_model_schema, assert_protocol_error, call, counting_harness,
    current_server_discover, deterministic_harness, response_result, tool_definition,
};

const CONTRACT_SOURCE: &str =
    include_str!("../../../../../verify/interfaces/mcp-stdio-compile-check/expected/contract.json");
const ACCEPTED_CONTROL_REQUEST: &[u8] = include_bytes!(
    "../../../../../verify/interfaces/control-plane-compile-check/models/accepted-v2.json"
);

fn expected() -> Value {
    serde_json::from_str(CONTRACT_SOURCE).expect("frozen MCP contract JSON")
}

fn timeout(name: &str) -> Duration {
    Duration::from_millis(
        expected()["oracleTimeouts"][name]
            .as_u64()
            .unwrap_or_else(|| panic!("missing frozen timeout `{name}`")),
    )
}

struct Gate {
    reached_tx: mpsc::Sender<()>,
    reached_rx: Mutex<mpsc::Receiver<()>>,
    release_tx: mpsc::Sender<()>,
    release_rx: Mutex<mpsc::Receiver<()>>,
}

/// Test-only deterministic observation shared with the parent server.
pub(super) struct OracleControl {
    gate: Option<Gate>,
    cancellation_tx: mpsc::Sender<Value>,
    cancellation_rx: Mutex<mpsc::Receiver<Value>>,
    cancellations: AtomicUsize,
    decision_tx: mpsc::Sender<(Value, bool)>,
    decision_rx: Mutex<mpsc::Receiver<(Value, bool)>>,
}

impl OracleControl {
    fn new() -> Self {
        let (cancellation_tx, cancellation_rx) = mpsc::channel();
        let (decision_tx, decision_rx) = mpsc::channel();
        Self {
            gate: None,
            cancellation_tx,
            cancellation_rx: Mutex::new(cancellation_rx),
            cancellations: AtomicUsize::new(0),
            decision_tx,
            decision_rx: Mutex::new(decision_rx),
        }
    }

    fn armed() -> Self {
        let mut control = Self::new();
        let (reached_tx, reached_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        control.gate = Some(Gate {
            reached_tx,
            reached_rx: Mutex::new(reached_rx),
            release_tx,
            release_rx: Mutex::new(release_rx),
        });
        control
    }

    /// Called by the worker immediately before its final cancellation check
    /// and before invoking the injected compiler.
    pub(super) fn before_compile(&self) {
        if let Some(gate) = &self.gate {
            gate.reached_tx.send(()).unwrap();
            gate.release_rx.lock().unwrap().recv().unwrap();
        }
    }

    /// Called after a well-formed notification has marked a live request and
    /// before any response for that request can be committed.
    pub(super) fn cancellation_processed(&self, request_id: &Value) {
        self.cancellations.fetch_add(1, Ordering::SeqCst);
        self.cancellation_tx.send(request_id.clone()).unwrap();
    }

    /// Called by the server event loop after a completed worker's final
    /// cancellation decision and before any response can be committed.
    pub(super) fn response_commit_decided(&self, request_id: &Value, commit: bool) {
        self.decision_tx.send((request_id.clone(), commit)).unwrap();
    }

    fn wait_gate(&self) {
        self.gate
            .as_ref()
            .unwrap()
            .reached_rx
            .lock()
            .unwrap()
            .recv_timeout(timeout("ordinaryResponseMs"))
            .expect("pre-compile gate was not reached");
    }

    fn release_gate(&self) {
        self.gate.as_ref().unwrap().release_tx.send(()).unwrap();
    }

    fn wait_cancellation(&self, request_id: &Value) {
        assert_eq!(
            self.cancellation_rx
                .lock()
                .unwrap()
                .recv_timeout(timeout("ordinaryResponseMs"))
                .expect("cancellation mark was not observed"),
            *request_id
        );
    }

    fn cancellation_count(&self) -> usize {
        self.cancellations.load(Ordering::SeqCst)
    }

    fn wait_commit_decision(&self, request_id: &Value, commit: bool) {
        assert_eq!(
            self.decision_rx
                .lock()
                .unwrap()
                .recv_timeout(timeout("ordinaryResponseMs"))
                .expect("response commit decision was not observed"),
            (request_id.clone(), commit)
        );
    }
}

struct ChannelReader {
    receiver: mpsc::Receiver<Vec<u8>>,
    current: Vec<u8>,
    offset: usize,
}

impl Read for ChannelReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        while self.offset == self.current.len() {
            match self.receiver.recv() {
                Ok(chunk) => {
                    self.current = chunk;
                    self.offset = 0;
                }
                Err(_) => return Ok(0),
            }
        }
        let count = output
            .len()
            .min(self.current.len().saturating_sub(self.offset));
        output[..count].copy_from_slice(&self.current[self.offset..self.offset + count]);
        self.offset += count;
        Ok(count)
    }
}

struct LineWriter {
    sender: mpsc::Sender<Vec<u8>>,
    buffered: Vec<u8>,
}

impl Write for LineWriter {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        for byte in input {
            self.buffered.push(*byte);
            if *byte == b'\n' {
                self.sender
                    .send(std::mem::take(&mut self.buffered))
                    .map_err(|_| {
                        io::Error::new(io::ErrorKind::BrokenPipe, "oracle receiver closed")
                    })?;
            }
        }
        Ok(input.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct Harness {
    input: Option<mpsc::Sender<Vec<u8>>>,
    output: mpsc::Receiver<Vec<u8>>,
    done: mpsc::Receiver<Result<(), String>>,
    thread: Option<JoinHandle<()>>,
}

impl Harness {
    fn start<C>(compiler: C, control: Arc<OracleControl>) -> Self
    where
        C: Fn(&str, &str) -> Result<ModelDocument, Vec<Diagnostic>> + Send + Sync + 'static,
    {
        let (input, receiver) = mpsc::channel();
        let (sender, output) = mpsc::channel();
        let (done_sender, done) = mpsc::channel();
        let thread = thread::spawn(move || {
            let outcome = run_for_oracle(
                ChannelReader {
                    receiver,
                    current: Vec::new(),
                    offset: 0,
                },
                LineWriter {
                    sender,
                    buffered: Vec::new(),
                },
                compiler,
                control,
            )
            .map_err(|error| error.to_string());
            let _ = done_sender.send(outcome);
        });
        Self {
            input: Some(input),
            output,
            done,
            thread: Some(thread),
        }
    }

    fn send(&self, value: Value) {
        let mut bytes = serde_json::to_vec(&value).unwrap();
        bytes.push(b'\n');
        self.input.as_ref().unwrap().send(bytes).unwrap();
    }

    fn send_raw(&self, bytes: impl Into<Vec<u8>>) {
        self.input.as_ref().unwrap().send(bytes.into()).unwrap();
    }

    fn recv(&self) -> Value {
        self.recv_with(timeout("ordinaryResponseMs"))
    }

    fn recv_with(&self, duration: Duration) -> Value {
        let bytes = self
            .output
            .recv_timeout(duration)
            .expect("oracle server response");
        assert_eq!(bytes.last(), Some(&b'\n'));
        assert_eq!(bytes.iter().filter(|byte| **byte == b'\n').count(), 1);
        worker_outcomes::assert_exact_result_wire(&bytes);
        serde_json::from_slice(&bytes).expect("oracle server JSON response")
    }

    fn assert_empty_now(&self) {
        assert!(
            matches!(self.output.try_recv(), Err(mpsc::TryRecvError::Empty)),
            "suppressed request produced output or closed the server"
        );
    }

    fn finish(mut self) {
        self.input.take();
        let outcome = self
            .done
            .recv_timeout(timeout("shutdownMs"))
            .expect("oracle server shutdown");
        outcome.expect("oracle server returned I/O error");
        self.thread.take().unwrap().join().unwrap();
    }
}

fn request_meta() -> Value {
    json!({
        "io.modelcontextprotocol/protocolVersion": expected()["protocolVersion"],
        "io.modelcontextprotocol/clientCapabilities": {}
    })
}

fn request(id: Value, method: &str, mut params: Map<String, Value>) -> Value {
    params.insert("_meta".to_owned(), request_meta());
    json!({"jsonrpc":"2.0","id":id,"method":method,"params":params})
}

fn call_with_meta(id: Value, arguments: Value, meta: Value) -> Value {
    json!({
        "jsonrpc":"2.0",
        "id":id,
        "method":"tools/call",
        "params":{"name":expected()["toolName"],"arguments":arguments,"_meta":meta}
    })
}

fn discover(id: Value) -> Value {
    request(id, "server/discover", Map::new())
}

fn list(id: Value) -> Value {
    request(id, "tools/list", Map::new())
}

fn cancel(request_id: Value, reason: Option<Value>, meta: Option<Value>) -> Value {
    let mut params = Map::new();
    params.insert("requestId".to_owned(), request_id);
    if let Some(reason) = reason {
        params.insert("reason".to_owned(), reason);
    }
    if let Some(meta) = meta {
        params.insert("_meta".to_owned(), meta);
    }
    json!({"jsonrpc":"2.0","method":"notifications/cancelled","params":params})
}

fn recv_without_compile(harness: &Harness, calls: &AtomicUsize, witness: &str) -> Value {
    let before = calls.load(Ordering::SeqCst);
    let response = harness.recv_with(timeout("largeResponseMs"));
    assert_eq!(calls.load(Ordering::SeqCst), before, "{witness} compiled");
    response
}

#[test]
fn admission_counts_once_and_links_the_descriptor_to_the_same_document() {
    let calls = Arc::new(AtomicUsize::new(0));
    let captured = Arc::new(Mutex::new(None::<ModelDocument>));
    let captured_input = Arc::new(Mutex::new(None::<(String, String)>));
    let compiler_calls = Arc::clone(&calls);
    let compiler_capture = Arc::clone(&captured);
    let compiler_input = Arc::clone(&captured_input);
    let control = Arc::new(OracleControl::new());
    let harness = Harness::start(
        move |filename, source| {
            compiler_calls.fetch_add(1, Ordering::SeqCst);
            *compiler_input.lock().unwrap() = Some((filename.to_owned(), source.to_owned()));
            let document = ModelDocument::compile(filename, source)?;
            *compiler_capture.lock().unwrap() = Some(document.clone());
            Ok(document)
        },
        control,
    );

    harness.send(call(json!("missing-source"), json!({})));
    let missing = harness.recv();
    let missing_result = response_result(&missing, &json!("missing-source"));
    assert_eq!(missing_result["isError"], true);
    assert_eq!(
        missing_result["structuredContent"]["diagnostics"],
        json!([expected()["inputDiagnostic"]])
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut unknown = Map::new();
    unknown.insert("name".to_owned(), json!("eqiora.unknown"));
    unknown.insert("arguments".to_owned(), json!({"source":"x"}));
    harness.send(request(json!("unknown"), "tools/call", unknown));
    let unknown = harness.recv();
    assert_eq!(unknown["error"]["code"], -32602);
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let (_, source) = accepted_source();
    harness.send(call(json!("accepted"), json!({"source":source})));
    let response = harness.recv();
    let result = response_result(&response, &json!("accepted"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        captured_input.lock().unwrap().as_ref().unwrap().0,
        expected()["inputBounds"]["filenameDefault"]
    );
    assert_eq!(result["isError"], false);
    assert_eq!(result["resultType"], "complete");
    let document = captured.lock().unwrap().clone().expect("captured document");
    let reference = document.artifact_reference().unwrap();
    let fingerprint = document.structural_fingerprint().unwrap();
    let model = &result["structuredContent"]["model"];
    assert_eq!(model["digest"], reference.artifact().as_str());
    assert_eq!(model["modelId"], reference.model().to_string());
    assert_eq!(
        model["semanticRevision"],
        reference.semantic_revision().get()
    );
    assert_eq!(
        model["structuralFingerprint"]["generation"],
        fingerprint.generation().as_str()
    );
    assert_eq!(
        model["structuralFingerprint"]["digest"],
        fingerprint.digest()
    );
    assert_eq!(fingerprint.generation(), SemanticFingerprintGeneration::V21);
    assert_current_model_schema(model);
    let text: Value = serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(text, result["structuredContent"]);

    harness.finish();
}

#[test]
fn an_admitted_rejection_invokes_the_operation_once() {
    let calls = Arc::new(AtomicUsize::new(0));
    let compiler_calls = Arc::clone(&calls);
    let harness = Harness::start(
        move |filename, source| {
            compiler_calls.fetch_add(1, Ordering::SeqCst);
            ModelDocument::compile(filename, source)
        },
        Arc::new(OracleControl::new()),
    );
    harness.send(call(
        json!("rejected"),
        json!({"filename":"empty.eqi","source":""}),
    ));
    let response = harness.recv();
    let result = response_result(&response, &json!("rejected"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(result["isError"], true);
    assert_eq!(result["structuredContent"]["status"], "rejected");
    assert_eq!(
        result["structuredContent"]["diagnostics"][0]["source"],
        "kernel"
    );
    assert!(
        !result["structuredContent"]
            .as_object()
            .unwrap()
            .contains_key("model")
    );
    harness.finish();
}

fn input_boundary(name: &str) -> Value {
    let (_, source) = accepted_source();
    match name {
        "missing-source" => json!({}),
        "empty-filename" => json!({"filename":"","source":"x"}),
        "control-filename" => json!({"filename":"bad\nname","source":"x"}),
        "filename-4097-ascii-bytes" => json!({"filename":"a".repeat(4097),"source":"x"}),
        "filename-2049-two-byte-characters" => json!({"filename":"é".repeat(2049),"source":"x"}),
        "source-8388609-ascii-bytes" => json!({"filename":"x.eqi","source":" ".repeat(8_388_609)}),
        "source-4194305-two-byte-characters" => {
            json!({"filename":"x.eqi","source":"é".repeat(4_194_305)})
        }
        "additional-argument" => json!({"filename":"x.eqi","source":"x","extra":true}),
        "wrong-filename-type" => json!({"filename":1,"source":"x"}),
        "wrong-source-type" => json!({"filename":"x.eqi","source":1}),
        "filename-1-character" => json!({"filename":"x","source":source}),
        "filename-4096-ascii-bytes" => json!({"filename":"a".repeat(4096),"source":source}),
        "filename-2048-two-byte-characters" => json!({"filename":"é".repeat(2048),"source":source}),
        "source-8388608-ascii-bytes" => json!({"filename":"x.eqi","source":" ".repeat(8_388_608)}),
        "source-4194304-two-byte-characters" => {
            json!({"filename":"x.eqi","source":"é".repeat(4_194_304)})
        }
        _ => panic!("unknown frozen input boundary `{name}`"),
    }
}

#[test]
fn every_frozen_input_boundary_has_exact_compile_cardinality() {
    let (harness, calls) = counting_harness(Arc::new(OracleControl::new()));
    for id in expected()["inputBounds"]["zeroCompilationWitnesses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
    {
        harness.send(call(json!(id), input_boundary(id)));
        let response = harness.recv_with(timeout("largeResponseMs"));
        let result = response_result(&response, &json!(id));
        assert_eq!(result["isError"], true);
        assert_eq!(
            result["structuredContent"]["diagnostics"],
            json!([expected()["inputDiagnostic"]])
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0, "{id} compiled");
    }

    for id in expected()["inputBounds"]["admittedWitnesses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
    {
        let before = calls.load(Ordering::SeqCst);
        harness.send(call(json!(id), input_boundary(id)));
        let response = harness.recv_with(timeout("largeResponseMs"));
        assert_ne!(
            response_result(&response, &json!(id))["structuredContent"]["diagnostics"],
            json!([expected()["inputDiagnostic"]])
        );
        assert_eq!(calls.load(Ordering::SeqCst), before + 1, "{id}");
    }
    harness.finish();
}

#[test]
fn every_pre_admission_error_stage_has_zero_compile_cardinality() {
    let (harness, calls) = counting_harness(Arc::new(OracleControl::new()));

    for (witness, raw) in [
        ("invalid-utf8", vec![b'{', 0xff, b'}', b'\n']),
        ("invalid-json", b"{\"jsonrpc\":\n".to_vec()),
        ("non-object", b"[]\n".to_vec()),
        (
            "duplicate-key",
            b"{\"jsonrpc\":\"2.0\",\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"server/discover\",\"params\":{}}\n".to_vec(),
        ),
    ] {
        harness.send_raw(raw);
        recv_without_compile(&harness, &calls, witness);
    }

    let maximum = expected()["framing"]["maximumEncodedLineBytes"]
        .as_u64()
        .unwrap() as usize;
    let mut exact_line = serde_json::to_vec(&discover(json!("exact-line"))).unwrap();
    exact_line.resize(maximum, b' ');
    exact_line.push(b'\n');
    harness.send_raw(exact_line);
    recv_without_compile(&harness, &calls, "exact framing");
    let mut overlong = vec![b' '; maximum + 1];
    overlong.push(b'\n');
    harness.send_raw(overlong);
    recv_without_compile(&harness, &calls, "overlong framing");

    for (id, array_levels) in [("depth-64", 61), ("depth-65", 62)] {
        let mut nested = json!(0);
        for _ in 0..array_levels {
            nested = Value::Array(vec![nested]);
        }
        let mut meta = request_meta();
        meta["com.example/depth"] = nested;
        harness.send(
            json!({"jsonrpc":"2.0","id":id,"method":"server/discover","params":{"_meta":meta}}),
        );
        recv_without_compile(&harness, &calls, id);
    }

    for (witness, value) in [
        (
            "bad-jsonrpc",
            json!({"jsonrpc":"1.0","id":"bad-jsonrpc","method":"server/discover","params":{"_meta":request_meta()}}),
        ),
        (
            "top-level-extra",
            json!({"jsonrpc":"2.0","id":"extra","method":"server/discover","params":{"_meta":request_meta()},"extra":true}),
        ),
        (
            "null-id",
            json!({"jsonrpc":"2.0","id":null,"method":"server/discover","params":{"_meta":request_meta()}}),
        ),
        (
            "boolean-id",
            json!({"jsonrpc":"2.0","id":true,"method":"server/discover","params":{"_meta":request_meta()}}),
        ),
        (
            "float-id",
            json!({"jsonrpc":"2.0","id":1.5,"method":"server/discover","params":{"_meta":request_meta()}}),
        ),
        (
            "out-of-range-id",
            json!({"jsonrpc":"2.0","id":u64::MAX,"method":"server/discover","params":{"_meta":request_meta()}}),
        ),
        (
            "non-object-params",
            json!({"jsonrpc":"2.0","id":"params","method":"server/discover","params":[]}),
        ),
    ] {
        harness.send(value);
        recv_without_compile(&harness, &calls, witness);
    }
    let before_notification = calls.load(Ordering::SeqCst);
    harness.send(
        json!({"jsonrpc":"2.0","method":"server/discover","params":{"_meta":request_meta()}}),
    );
    harness.send(discover(json!("notification-barrier")));
    recv_without_compile(&harness, &calls, "missing-id notification");
    assert_eq!(calls.load(Ordering::SeqCst), before_notification);

    harness.send(discover(json!("x".repeat(129))));
    recv_without_compile(&harness, &calls, "overlong id");
    for (id, method, params) in [
        ("unknown", "unknown", Map::new()),
        ("initialize", "initialize", Map::new()),
        (
            "discover-extra",
            "server/discover",
            Map::from_iter([("extra".to_owned(), Value::Null)]),
        ),
        (
            "list-cursor",
            "tools/list",
            Map::from_iter([("cursor".to_owned(), Value::Null)]),
        ),
    ] {
        harness.send(request(json!(id), method, params));
        recv_without_compile(&harness, &calls, id);
    }
    harness.send(json!({
        "jsonrpc":"2.0","id":"missing-meta","method":"server/discover","params":{}
    }));
    recv_without_compile(&harness, &calls, "missing metadata");
    let mut unsupported = request_meta();
    unsupported["io.modelcontextprotocol/protocolVersion"] =
        expected()["unsupportedVersion"]["witness"].clone();
    harness.send(json!({
        "jsonrpc":"2.0","id":"unsupported","method":"server/discover",
        "params":{"_meta":unsupported}
    }));
    recv_without_compile(&harness, &calls, "unsupported version");
    let mut unknown = Map::new();
    unknown.insert("name".to_owned(), json!("eqiora.unknown"));
    unknown.insert("arguments".to_owned(), json!({}));
    harness.send(request(json!("unknown-tool"), "tools/call", unknown));
    recv_without_compile(&harness, &calls, "unknown tool");
    for (id, params) in [
        (
            "malformed-name",
            json!({"name":"eqiora/tool","arguments":{}}),
        ),
        (
            "arguments-type",
            json!({"name":expected()["toolName"],"arguments":[]}),
        ),
        (
            "input-responses",
            json!({"name":expected()["toolName"],"arguments":{},"inputResponses":[]}),
        ),
        (
            "request-state",
            json!({"name":expected()["toolName"],"arguments":{},"requestState":{}}),
        ),
    ] {
        harness.send(request(
            json!(id),
            "tools/call",
            params.as_object().unwrap().clone(),
        ));
        recv_without_compile(&harness, &calls, id);
    }
    harness.send(call(json!("invalid-input"), json!({})));
    recv_without_compile(&harness, &calls, "invalid input");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    harness.finish();
}

#[test]
fn request_metadata_is_validated_on_the_compile_path_before_admission() {
    let (filename, source) = accepted_source();
    let (harness, inputs) = deterministic_harness(&filename, &source);
    let contract = expected();
    let witnesses = &contract["metadata"]["compilePathWitnesses"];
    harness.send(call_with_meta(
        json!("valid-minimal-meta"),
        json!({"filename":filename,"source":source}),
        witnesses["minimal"].clone(),
    ));
    let minimal = response_result(&harness.recv(), &json!("valid-minimal-meta")).clone();
    let mut valid_metadata = witnesses["valid"].clone();
    valid_metadata["tracestate"] = json!("a=1, \t ,b=2 ");
    harness.send(call_with_meta(
        json!("valid-full-meta"),
        json!({"filename":filename,"source":source}),
        valid_metadata,
    ));
    let accepted = harness.recv();
    let result = response_result(&accepted, &json!("valid-full-meta"));
    assert_eq!(result, &minimal);
    assert_eq!(result["isError"], false);
    let encoded = serde_json::to_string(&accepted).unwrap();
    assert!(!encoded.contains("metadata-compile-marker"));
    assert!(!encoded.contains("metadata-vendor-marker"));
    assert_eq!(
        inputs.lock().unwrap().as_slice(),
        [
            (filename.clone(), source.clone()),
            (filename.clone(), source.clone()),
        ]
    );
    for witness in witnesses["invalid"].as_array().unwrap() {
        let id = witness[0].as_str().unwrap();
        let key = witness[1].as_str().unwrap();
        let mut meta = request_meta();
        if witness.get(3).and_then(Value::as_bool).unwrap_or(false) {
            meta.as_object_mut().unwrap().remove(key);
        } else {
            meta[key] = witness[2].clone();
        }
        harness.send(call_with_meta(
            json!(id),
            json!({"filename":filename,"source":source}),
            meta,
        ));
        let response = harness.recv();
        assert_protocol_error(&response, Some(&json!(id)), -32602);
        assert_eq!(inputs.lock().unwrap().len(), 2, "invalid {id} compiled");
        assert!(
            !serde_json::to_string(&response)
                .unwrap()
                .contains("metadata-invalid-marker")
        );
    }
    harness.finish();
}

#[test]
fn one_active_call_refuses_busy_work_but_keeps_discovery_responsive() {
    let calls = Arc::new(AtomicUsize::new(0));
    let compiler_calls = Arc::clone(&calls);
    let (started_sender, started_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = mpsc::channel();
    let release_receiver = Arc::new(Mutex::new(release_receiver));
    let compiler_release = Arc::clone(&release_receiver);
    let harness = Harness::start(
        move |filename, source| {
            compiler_calls.fetch_add(1, Ordering::SeqCst);
            started_sender.send(()).unwrap();
            compiler_release.lock().unwrap().recv().unwrap();
            ModelDocument::compile(filename, source)
        },
        Arc::new(OracleControl::new()),
    );
    let (filename, source) = accepted_source();
    harness.send(call(
        json!("active"),
        json!({"filename":filename,"source":source}),
    ));
    worker_outcomes::wait_flag(&started_receiver, "active compiler");
    harness.send(call(
        json!("active"),
        json!({"filename":"duplicate.eqi","source":""}),
    ));
    harness.send(call(
        json!("busy"),
        json!({"filename":"second.eqi","source":""}),
    ));
    harness.send(discover(json!("discover-during")));
    harness.send(list(json!("list-during")));
    let duplicate = harness.recv();
    assert_eq!(duplicate["error"]["code"], -32600);
    assert_eq!(duplicate["id"], "active");
    let busy = harness.recv();
    assert_eq!(busy["id"], "busy");
    assert_eq!(busy["error"]["code"], expected()["lifecycle"]["busyCode"]);
    assert_eq!(
        busy["error"]["message"],
        expected()["lifecycle"]["busyMessage"]
    );
    assert_eq!(busy["error"]["data"], expected()["lifecycle"]["busyData"]);
    let discover_response = harness.recv();
    assert_eq!(
        response_result(&discover_response, &json!("discover-during")),
        &current_server_discover()
    );
    let list_response = harness.recv();
    assert_eq!(
        response_result(&list_response, &json!("list-during"))["tools"]
            .as_array()
            .unwrap(),
        &[tool_definition()]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    release_sender.send(()).unwrap();
    let active = harness.recv();
    assert_eq!(active["id"], "active");
    assert_eq!(active["result"]["isError"], false);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    harness.finish();
}

#[test]
fn cancellation_before_start_is_deterministic_and_malformed_metadata_is_silent() {
    let calls = Arc::new(AtomicUsize::new(0));
    let compiler_calls = Arc::clone(&calls);
    let control = Arc::new(OracleControl::armed());
    let harness = Harness::start(
        move |filename, source| {
            compiler_calls.fetch_add(1, Ordering::SeqCst);
            ModelDocument::compile(filename, source)
        },
        Arc::clone(&control),
    );
    let (filename, source) = accepted_source();
    harness.send(call(json!(1), json!({"filename":filename,"source":source})));
    control.wait_gate();

    harness.send(cancel(json!("1"), None, None));
    harness.send(cancel(
        json!(1),
        None,
        Some(json!({"traceparent":"invalid"})),
    ));
    harness.send(cancel(json!(1), None, Some(json!({"bad key":"invalid"}))));
    harness.send(cancel(
        json!(1),
        None,
        Some(json!({"io.modelcontextprotocol/subscriptionId":true})),
    ));
    harness.send(cancel(json!(1), None, Some(json!({"baggage":"key=%"}))));
    harness.send(cancel(json!(1), Some(json!("x".repeat(4097))), None));
    harness.send(json!({
        "jsonrpc":"2.0",
        "method":"notifications/cancelled",
        "params":{"requestId":1,"unknown":true}
    }));
    harness.send(discover(json!("malformed-cancellations-drained")));
    let drained = harness.recv();
    assert_eq!(drained["id"], "malformed-cancellations-drained");
    assert_eq!(control.cancellation_count(), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let trace_id = "0af7651916cd43dd8448eb211c80319c";
    let parent_id = "00f067aa0ba902b7";
    harness.send(cancel(
        json!(1),
        Some(json!("bounded cancellation reason")),
        Some(json!({
            "io.modelcontextprotocol/subscriptionId":1,
            "traceparent":format!("00-{trace_id}-{parent_id}-01"),
            "tracestate":"tenant@system=value",
            "baggage":"key=value;property=p%25",
            "com.example/vendor":{"ignored":true}
        })),
    ));
    control.wait_cancellation(&json!(1));
    control.release_gate();
    control.wait_commit_decision(&json!(1), false);
    harness.send(list(json!("after-cancel")));
    let after = harness.recv();
    assert_eq!(after["id"], "after-cancel");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    harness.assert_empty_now();
    harness.finish();
}

#[test]
fn cancellation_during_compile_suppresses_only_the_cancelled_response() {
    let calls = Arc::new(AtomicUsize::new(0));
    let compiler_calls = Arc::clone(&calls);
    let control = Arc::new(OracleControl::new());
    let (started_sender, started_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = mpsc::channel();
    let release_receiver = Arc::new(Mutex::new(release_receiver));
    let compiler_release = Arc::clone(&release_receiver);
    let harness = Harness::start(
        move |filename, source| {
            compiler_calls.fetch_add(1, Ordering::SeqCst);
            started_sender.send(()).unwrap();
            compiler_release.lock().unwrap().recv().unwrap();
            ModelDocument::compile(filename, source)
        },
        Arc::clone(&control),
    );
    let (filename, source) = accepted_source();
    harness.send(call(
        json!("cancel-running"),
        json!({"filename":filename,"source":source}),
    ));
    worker_outcomes::wait_flag(&started_receiver, "running compiler");
    harness.send(cancel(
        json!("cancel-running"),
        Some(json!("do-not-log-this-reason")),
        Some(json!({"io.modelcontextprotocol/subscriptionId":"sub"})),
    ));
    control.wait_cancellation(&json!("cancel-running"));
    harness.send(discover(json!("responsive")));
    let responsive = harness.recv();
    assert_eq!(responsive["id"], "responsive");
    release_sender.send(()).unwrap();
    control.wait_commit_decision(&json!("cancel-running"), false);
    harness.send(list(json!("still-responsive")));
    let list_response = harness.recv();
    assert_eq!(list_response["id"], "still-responsive");
    harness.assert_empty_now();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    harness.finish();
}

#[test]
fn cancellation_after_commit_cannot_retract_a_response() {
    let control = Arc::new(OracleControl::new());
    let harness = Harness::start(ModelDocument::compile, Arc::clone(&control));
    let (filename, source) = accepted_source();
    harness.send(call(
        json!("completed"),
        json!({"filename":filename,"source":source}),
    ));
    let completed = harness.recv();
    assert_eq!(completed["id"], "completed");
    harness.send(cancel(json!("completed"), None, None));
    harness.send(discover(json!("after-completed-cancel")));
    let after = harness.recv();
    assert_eq!(after["id"], "after-completed-cancel");
    assert_eq!(control.cancellation_count(), 0);
    harness.assert_empty_now();
    harness.finish();
}

#[test]
fn cancellation_metadata_and_reason_boundaries_are_silent_and_exact() {
    const CHILD_ENV: &str = "EQIORA_MCP_CANCELLATION_ORACLE_CHILD";
    const TEST_NAME: &str =
        "oracle::cancellation_metadata_and_reason_boundaries_are_silent_and_exact";
    if worker_outcomes::rerun_capturing_stderr(
        CHILD_ENV,
        TEST_NAME,
        &["cancel-secret-marker", "cancel-metadata-marker"],
    ) {
        return;
    }

    let contract = expected();
    let witnesses = &contract["cancellation"]["boundaryWitnesses"];
    for witness in witnesses.as_array().unwrap() {
        let name = witness[0].as_str().unwrap();
        let reason = match witness[1].as_str().unwrap() {
            "max-ascii" => Some(json!(format!("cancel-secret-marker{}", "x".repeat(4076)))),
            "max-two-byte" => Some(json!("é".repeat(2048))),
            "over-ascii" => Some(json!(format!("cancel-secret-marker{}", "x".repeat(4077)))),
            "over-two-byte" => Some(json!("é".repeat(2049))),
            "empty" => Some(json!("")),
            "boolean" => Some(json!(true)),
            "absent" => None,
            _ => unreachable!(),
        };
        let meta = match witness[2].as_str().unwrap() {
            "full" => contract["cancellation"]["validMetadata"].clone(),
            "integer-subscription" => json!({"io.modelcontextprotocol/subscriptionId":1}),
            "bad-subscription" => json!({"io.modelcontextprotocol/subscriptionId":true}),
            "bad-trace" => json!({"traceparent":"cancel-metadata-marker"}),
            "bad-baggage" => json!({"baggage":"cancel-metadata-marker=%"}),
            "non-object" => Value::Null,
            "empty" => json!({}),
            _ => unreachable!(),
        };
        let valid = witness[3].as_bool().unwrap();
        let control = Arc::new(OracleControl::armed());
        let (harness, calls) = counting_harness(Arc::clone(&control));
        let (filename, source) = accepted_source();
        let id = witness.get(5).cloned().unwrap_or_else(|| json!(1));
        harness.send(call(
            id.clone(),
            json!({"filename":filename,"source":source}),
        ));
        control.wait_gate();
        let notification = match witness.get(4).filter(|value| !value.is_null()) {
            Some(notification) => notification.clone(),
            None => cancel(id.clone(), reason, Some(meta)),
        };
        if name == "request-id-negative-overflow" {
            harness.send_raw(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/cancelled\",\"params\":{\"requestId\":-9223372036854775809}}\n".to_vec());
        } else {
            harness.send(notification);
        }
        harness.send(discover(json!(format!("notification-barrier-{name}"))));
        let barrier = harness.recv();
        assert_eq!(barrier["id"], format!("notification-barrier-{name}"));
        if valid {
            control.wait_cancellation(&id);
        } else {
            assert_eq!(control.cancellation_count(), 0);
        }
        control.release_gate();
        control.wait_commit_decision(&id, !valid);
        if !valid {
            let completed = harness.recv();
            assert_eq!(completed["id"], id);
            assert_eq!(completed["result"]["isError"], false);
        }
        harness.send(list(json!(format!("completion-barrier-{name}"))));
        let completion = harness.recv();
        assert_eq!(completion["id"], format!("completion-barrier-{name}"));
        harness.assert_empty_now();
        assert_eq!(calls.load(Ordering::SeqCst), usize::from(!valid));
        harness.finish();
    }
}
