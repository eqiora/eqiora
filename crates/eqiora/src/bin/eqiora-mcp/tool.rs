use eqiora::api::ModelDocument;
use eqiora::{Diagnostic, Severity};
use serde_json::{Map, Value};

pub(super) const NAME: &str = "eqiora.model.compile_check";
pub(super) const DEFINITION: &str = r#"{"name":"eqiora.model.compile_check","title":"Compile/check an Eqiora Model","description":"Compile one in-memory Eqiora source string into the current Model descriptor, or return structured compiler diagnostics. filename is a diagnostic label and is never read from the filesystem.","inputSchema":{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":false,"properties":{"filename":{"type":"string","minLength":1,"maxLength":4096,"default":"<memory>","description":"Diagnostic source label only; never a filesystem path."},"source":{"type":"string","maxLength":8388608,"description":"Complete in-memory Eqiora source text."}},"required":["source"]},"outputSchema":{"$schema":"https://json-schema.org/draft/2020-12/schema","oneOf":[{"type":"object","additionalProperties":false,"properties":{"schema":{"const":"eqiora.mcp.compile-check-result/v1"},"status":{"const":"accepted"},"model":{"type":"object","additionalProperties":false,"properties":{"schema":{"const":"eqiora.model-envelope/v27"},"transactionSchema":{"const":"eqiora.model-transaction-envelope/v27"},"digest":{"type":"string","pattern":"^[0-9a-f]{64}$"},"modelId":{"type":"string","minLength":1,"maxLength":128},"semanticRevision":{"type":"integer","minimum":0},"structuralFingerprint":{"type":"object","additionalProperties":false,"properties":{"generation":{"enum":["eqiora.structural-semantic-fingerprint/v22"]},"digest":{"type":"string","pattern":"^[0-9a-f]{64}$"}},"required":["generation","digest"]}},"required":["schema","transactionSchema","digest","modelId","semanticRevision","structuralFingerprint"]}},"required":["schema","status","model"]},{"type":"object","additionalProperties":false,"properties":{"schema":{"const":"eqiora.mcp.compile-check-result/v1"},"status":{"const":"rejected"},"diagnostics":{"type":"array","minItems":1,"maxItems":1024,"items":{"type":"object","additionalProperties":false,"properties":{"source":{"enum":["mcp","kernel"]},"severity":{"enum":["error","warning","note"]},"code":{"type":"string","pattern":"^[A-Z]{2}[0-9]{4}$"},"message":{"type":"string","minLength":1,"maxLength":1048576},"graphPath":{"oneOf":[{"type":"array","maxItems":256,"items":{"type":"string","minLength":1,"maxLength":4096}},{"type":"null"}]},"span":{"oneOf":[{"type":"object","additionalProperties":false,"properties":{"file":{"type":"string","maxLength":4096},"start":{"type":"integer","minimum":0,"maximum":4294967295},"end":{"type":"integer","minimum":0,"maximum":4294967295}},"required":["file","start","end"]},{"type":"null"}]},"patch":{"oneOf":[{"type":"object","additionalProperties":false,"properties":{"summary":{"type":"string","minLength":1,"maxLength":4096}},"required":["summary"]},{"type":"null"}]}},"required":["source","severity","code","message","graphPath","span","patch"]}}},"required":["schema","status","diagnostics"]}]}}"#;

const RESULT_SCHEMA: &str = "eqiora.mcp.compile-check-result/v1";
const STRUCTURED_LIMIT: usize = 8_404_992;

pub(super) struct Arguments {
    pub(super) filename: String,
    pub(super) source: String,
}

pub(super) fn admit(arguments: &Map<String, Value>) -> Result<Arguments, String> {
    if arguments
        .keys()
        .any(|key| key != "filename" && key != "source")
    {
        return Err(input_failure());
    }
    let Some(source) = arguments.get("source").and_then(Value::as_str) else {
        return Err(input_failure());
    };
    let filename = match arguments.get("filename") {
        Some(Value::String(text)) => text.as_str(),
        None => "<memory>",
        _ => return Err(input_failure()),
    };
    if filename.is_empty()
        || filename.chars().count() > 4096
        || filename.len() > 4096
        || filename.chars().any(char::is_control)
        || source.chars().count() > 8_388_608
        || source.len() > 8_388_608
    {
        return Err(input_failure());
    }
    Ok(Arguments {
        filename: filename.to_owned(),
        source: source.to_owned(),
    })
}

pub(super) fn compile_document(
    filename: &str,
    source: &str,
) -> Result<ModelDocument, Vec<Diagnostic>> {
    eqiora::api::ModelDocument::compile(filename, source)
}

pub(super) fn project(outcome: Result<ModelDocument, Vec<Diagnostic>>) -> Result<String, ()> {
    match outcome {
        Ok(document) => accepted(&document),
        Err(diagnostics) => Ok(rejected(&diagnostics)),
    }
}

pub(super) fn input_failure() -> String {
    rejected_raw(&[diagnostic_raw(
        "mcp",
        "error",
        "EQ0901",
        "tool arguments do not satisfy eqiora.model.compile_check input schema",
        "null",
        "null",
        "null",
    )])
}

fn overflow_failure() -> String {
    rejected_raw(&[diagnostic_raw(
        "mcp",
        "error",
        "EQ0901",
        "compile/check diagnostics exceed the MCP stdio response limits",
        "null",
        "null",
        "null",
    )])
}

fn accepted(document: &ModelDocument) -> Result<String, ()> {
    let reference = document.artifact_reference().map_err(|_| ())?;
    let fingerprint = document.structural_fingerprint().map_err(|_| ())?;
    let fingerprint_raw = object(&[
        ("generation", quoted(fingerprint.generation().as_str())),
        ("digest", quoted(fingerprint.digest())),
    ]);
    let model = object(&[
        ("schema", quoted("eqiora.model-envelope/v27")),
        (
            "transactionSchema",
            quoted("eqiora.model-transaction-envelope/v27"),
        ),
        ("digest", quoted(reference.artifact().as_str())),
        ("modelId", quoted(&reference.model().to_string())),
        (
            "semanticRevision",
            reference.semantic_revision().get().to_string(),
        ),
        ("structuralFingerprint", fingerprint_raw),
    ]);
    Ok(object(&[
        ("schema", quoted(RESULT_SCHEMA)),
        ("status", quoted("accepted")),
        ("model", model),
    ]))
}

fn rejected(diagnostics: &[Diagnostic]) -> String {
    if diagnostics.is_empty() || diagnostics.len() > 1024 {
        return overflow_failure();
    }
    let mut projected = Vec::with_capacity(diagnostics.len());
    for diagnostic in diagnostics {
        let Some(raw) = project_diagnostic(diagnostic) else {
            return overflow_failure();
        };
        projected.push(raw);
    }
    let structured = rejected_raw(&projected);
    if structured.len() > STRUCTURED_LIMIT {
        overflow_failure()
    } else {
        structured
    }
}

fn project_diagnostic(diagnostic: &Diagnostic) -> Option<String> {
    let code = diagnostic.code().0;
    let bytes = code.as_bytes();
    if bytes.len() != 6
        || !bytes[..2].iter().all(u8::is_ascii_uppercase)
        || !bytes[2..].iter().all(u8::is_ascii_digit)
    {
        return None;
    }
    let message = diagnostic.message();
    if message.is_empty() || message.chars().count() > 1_048_576 || message.len() > 1_048_576 {
        return None;
    }
    let graph_raw = match diagnostic.graph_path() {
        Some(graph) => {
            let segments = graph.segments();
            if segments.len() > 256
                || segments.iter().any(|segment| {
                    segment.is_empty() || segment.chars().count() > 4096 || segment.len() > 4096
                })
            {
                return None;
            }
            array(
                &segments
                    .iter()
                    .map(|segment| quoted(segment))
                    .collect::<Vec<_>>(),
            )
        }
        None => "null".to_owned(),
    };
    let span_raw = match diagnostic.source_span() {
        Some(span) => {
            if span.file.chars().count() > 4096 || span.file.len() > 4096 || span.end < span.start {
                return None;
            }
            object(&[
                ("file", quoted(&span.file)),
                ("start", span.start.to_string()),
                ("end", span.end.to_string()),
            ])
        }
        None => "null".to_owned(),
    };
    let patch_raw = match diagnostic.suggestion() {
        Some(suggestion) => {
            if suggestion.summary.is_empty()
                || suggestion.summary.chars().count() > 4096
                || suggestion.summary.len() > 4096
            {
                return None;
            }
            object(&[("summary", quoted(&suggestion.summary))])
        }
        None => "null".to_owned(),
    };
    let severity = match diagnostic.severity() {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Note => "note",
    };
    Some(diagnostic_raw(
        "kernel", severity, code, message, &graph_raw, &span_raw, &patch_raw,
    ))
}

fn rejected_raw(diagnostics: &[String]) -> String {
    object(&[
        ("schema", quoted(RESULT_SCHEMA)),
        ("status", quoted("rejected")),
        ("diagnostics", array(diagnostics)),
    ])
}

fn diagnostic_raw(
    source: &str,
    severity: &str,
    code: &str,
    message: &str,
    graph: &str,
    span: &str,
    patch: &str,
) -> String {
    object(&[
        ("source", quoted(source)),
        ("severity", quoted(severity)),
        ("code", quoted(code)),
        ("message", quoted(message)),
        ("graphPath", graph.to_owned()),
        ("span", span.to_owned()),
        ("patch", patch.to_owned()),
    ])
}

fn quoted(text: &str) -> String {
    serde_json::to_string(text).expect("string JSON encoding")
}

fn array(items: &[String]) -> String {
    let mut output = String::from("[");
    output.push_str(&items.join(","));
    output.push(']');
    output
}

fn object(fields: &[(&str, String)]) -> String {
    let mut output = String::from("{");
    for (index, (name, raw)) in fields.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        output.push_str(&quoted(name));
        output.push(':');
        output.push_str(raw);
    }
    output.push('}');
    output
}
