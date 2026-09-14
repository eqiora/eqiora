use eqiora::api::ModelDocument;
use eqiora::control::{
    CompileOutcomeV2, CompileRequestV2, ControlDiagnosticSourceV2, ControlDiagnosticV2,
    ControlSeverityV2, execute_compile_v2,
};
use eqiora::{Diagnostic, Severity};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyDictMethods, PyModule, PyModuleMethods};
use std::fs;
use std::path::Path;
use std::process::Command;

const SOURCE: &str = r#"
model decay() {
  state x: 1;
  initial { x = 1; }
  parameter rate: 1 / s = 1;
  relation flow {
    derivative(x) + rate * x = 0;
  }
}
"#;

const PANIC_PAYLOAD: &str = "private panic payload must not cross Python boundary";
const PANIC_PROBE_CHILD: &str = "EQIORA_PYTHON_PANIC_PROBE_CHILD";
const MAX_FILENAME_BYTES: usize = 4_096;
const MAX_SOURCE_BYTES: usize = 8_388_608;
const FILENAME_MESSAGE: &str = "source filename must contain 1 to 4096 non-control UTF-8 bytes";
const SOURCE_MESSAGE: &str = "source exceeds the 8388608-byte compile/check v2 limit";
const PUBLIC_STUB: &str = include_str!("../../../bindings/python/python/eqiora/__init__.pyi");

#[derive(Debug, PartialEq, Eq)]
struct NormalizedDiagnostic {
    source: String,
    severity: String,
    code: String,
    message: String,
    graph_path: Option<Vec<String>>,
    source_span: Option<(String, u32, u32)>,
    suggestion: Option<String>,
}

#[pyfunction]
fn boundary_panic(py: Python<'_>) -> PyResult<()> {
    _eqiora::panic_boundary(py, || -> PyResult<()> { panic!("{PANIC_PAYLOAD}") })
}

#[test]
fn python_compile_contract_is_claim_local_and_transport_neutral() -> PyResult<()> {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut rust_sources = Vec::new();
    collect_rust_sources(&crate_root, &mut rust_sources);
    for path in rust_sources {
        let source = fs::read_to_string(&path).unwrap();
        for forbidden in [
            "eqiora::control",
            "CompileRequest",
            "CompileOutcome",
            "CompileResponse",
            "CompileControlExecution",
            "ControlDiagnostic",
            "execute_compile",
        ] {
            assert!(
                !source.contains(forbidden),
                "{} retains forbidden control dependency {forbidden:?}",
                path.display()
            );
        }
    }

    Python::initialize();
    Python::attach(|py| {
        assert_stub_compile_contract(py)?;

        let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
        let module = native.bind(py);
        let signature = py
            .import("inspect")?
            .getattr("signature")?
            .call1((module.getattr("compile")?,))?;
        assert_eq!(
            signature.str()?.to_str()?,
            "(*, path=None, source=None, filename=None, geometry=None, bindings=None, entry=None)"
        );
        Ok(())
    })
}

fn assert_stub_compile_contract(py: Python<'_>) -> PyResult<()> {
    let ast = py.import("ast")?;
    let syntax = ast.call_method1("parse", (PUBLIC_STUB,))?;
    let function_def = ast.getattr("FunctionDef")?;
    let declarations = syntax
        .getattr("body")?
        .try_iter()?
        .filter_map(|item| {
            let item = item.expect("stub syntax tree must be iterable");
            if !item
                .is_instance(&function_def)
                .expect("FunctionDef identity check must succeed")
            {
                return None;
            }
            let name = item
                .getattr("name")
                .expect("function name must exist")
                .extract::<String>()
                .expect("function name must be text");
            (name == "compile").then_some(item)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        declarations.len(),
        1,
        "stub must declare compile exactly once"
    );

    let declaration = &declarations[0];
    let arguments = declaration.getattr("args")?;
    assert!(argument_names(&arguments.getattr("posonlyargs")?)?.is_empty());
    assert!(argument_names(&arguments.getattr("args")?)?.is_empty());
    assert_eq!(
        argument_names(&arguments.getattr("kwonlyargs")?)?,
        [
            "path".to_owned(),
            "source".to_owned(),
            "filename".to_owned(),
            "geometry".to_owned(),
            "bindings".to_owned(),
            "entry".to_owned(),
        ]
    );
    assert!(arguments.getattr("vararg")?.is_none());
    assert!(arguments.getattr("kwarg")?.is_none());

    let defaults = arguments.getattr("kw_defaults")?;
    assert_eq!(defaults.len()?, 6);
    for index in 0..6 {
        assert!(
            ast.call_method1("literal_eval", (defaults.get_item(index)?,))?
                .is_none()
        );
    }
    assert_eq!(
        ast.call_method1("unparse", (declaration.getattr("returns")?,))?
            .extract::<String>()?,
        "Model"
    );
    Ok(())
}

fn argument_names(arguments: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    arguments
        .try_iter()?
        .map(|argument| argument?.getattr("arg")?.extract::<String>())
        .collect()
}

#[test]
fn independent_python_control_and_direct_compilations_share_canonical_identity() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
        let module = native.bind(py);
        let filename = "three-canonical-routes.eqi";

        let kwargs = PyDict::new(py);
        kwargs.set_item("filename", filename)?;
        kwargs.set_item("source", SOURCE)?;
        let python = module.getattr("compile")?.call((), Some(&kwargs))?;

        let request = CompileRequestV2::new("three-path-control", filename, SOURCE).unwrap();
        let control = execute_compile_v2(&request);
        let control_document = control.document().expect("accepted control document");
        let CompileOutcomeV2::Accepted { model } = control.response().outcome() else {
            panic!("control-v2 rejected the accepted frozen source")
        };
        let control_reference = control_document.artifact_reference().unwrap();
        assert_eq!(model.schema(), "eqiora.model-envelope/v27");
        assert_eq!(
            model.transaction_schema(),
            "eqiora.model-transaction-envelope/v27"
        );
        assert_eq!(model.model_id(), control_reference.model().to_string());
        assert_eq!(model.digest(), control_reference.artifact().as_str());
        assert_eq!(
            model.semantic_revision(),
            control_reference.semantic_revision().get()
        );

        let direct = ModelDocument::compile(filename, SOURCE).unwrap();
        let direct_reference = direct.artifact_reference().unwrap();
        let python_id = python.getattr("model_id")?.extract::<String>()?;
        let python_digest = python.getattr("digest")?.extract::<String>()?;
        let ids = [
            python_id,
            control_reference.model().ulid().to_string(),
            direct_reference.model().ulid().to_string(),
        ];
        let digests = [
            python_digest,
            control_reference.artifact().to_string(),
            direct_reference.artifact().to_string(),
        ];
        assert_eq!(ids[0], ids[1]);
        assert_eq!(ids[1], ids[2]);
        assert_eq!(digests[0], digests[1]);
        assert_eq!(digests[1], digests[2]);
        let changed_source =
            SOURCE.replace("parameter rate: 1 / s = 1;", "parameter rate: 1 / s = 2;");
        assert_ne!(changed_source, SOURCE);
        let changed = ModelDocument::compile(filename, &changed_source).unwrap();
        assert_ne!(changed.digest().unwrap(), direct.digest().unwrap());

        let python_fingerprint = python.getattr("structural_fingerprint")?;
        let python_fingerprint = (
            python_fingerprint
                .getattr("generation")?
                .extract::<String>()?,
            python_fingerprint.getattr("digest")?.extract::<String>()?,
        );
        let control_fingerprint = control_document.structural_fingerprint().unwrap();
        let direct_fingerprint = direct.structural_fingerprint().unwrap();
        assert_eq!(
            python_fingerprint,
            (
                control_fingerprint.generation().as_str().to_owned(),
                control_fingerprint.digest().to_owned(),
            )
        );
        assert_eq!(control_fingerprint, direct_fingerprint);
        Ok(())
    })
}

#[test]
fn rejected_python_control_and_direct_compilations_preserve_ordinary_diagnostics() -> PyResult<()> {
    const REJECTED: &str = "model broken() { field ; }";
    const FILENAME: &str = "three-path-rejection.eqi";

    let direct = ModelDocument::compile(FILENAME, REJECTED).unwrap_err();
    let direct = direct.iter().map(normalize_kernel).collect::<Vec<_>>();

    let request = CompileRequestV2::new("three-path-rejection", FILENAME, REJECTED).unwrap();
    let control = execute_compile_v2(&request);
    assert!(control.document().is_none());
    let CompileOutcomeV2::Rejected { diagnostics } = control.response().outcome() else {
        panic!("control-v2 accepted rejected source")
    };
    let control = diagnostics
        .iter()
        .map(normalize_control)
        .collect::<Vec<_>>();

    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
        let module = native.bind(py);
        let kwargs = PyDict::new(py);
        kwargs.set_item("filename", FILENAME)?;
        kwargs.set_item("source", REJECTED)?;
        let error = module
            .getattr("compile")?
            .call((), Some(&kwargs))
            .expect_err("Python accepted rejected source");
        let (category, python) = normalize_python_error(py, &error)?;
        assert_eq!(category, "validation");
        assert_eq!(python, direct);
        assert_eq!(python, control);
        Ok(())
    })
}

#[test]
fn python_admission_boundaries_are_local_exact_and_fail_closed() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
        let module = native.bind(py);

        for filename in ["a".repeat(MAX_FILENAME_BYTES), "é".repeat(2_048)] {
            assert_eq!(filename.len(), MAX_FILENAME_BYTES);
            assert_python_compile_reaches_operation(module, SOURCE, &filename)?;
        }

        let mut source = SOURCE.to_owned();
        source.push_str(&" ".repeat(MAX_SOURCE_BYTES - source.len()));
        assert_eq!(source.chars().count(), MAX_SOURCE_BYTES);
        assert_eq!(source.len(), MAX_SOURCE_BYTES);
        assert_python_compile_reaches_operation(module, &source, "source-at-bound.eqi")?;

        for filename in [
            String::new(),
            "bad\nname.eqi".to_owned(),
            "a".repeat(MAX_FILENAME_BYTES + 1),
            "é".repeat(2_049),
        ] {
            assert_python_admission_error(module, SOURCE, &filename, FILENAME_MESSAGE)?;
        }

        let source = "a".repeat(MAX_SOURCE_BYTES + 1);
        assert_python_admission_error(module, &source, "source-over-bound.eqi", SOURCE_MESSAGE)?;
        Ok(())
    })
}

#[test]
fn python_control_plane_preserves_identity_and_fails_closed() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let loaded = py
            .import("sys")?
            .getattr("modules")?
            .cast_into::<PyDict>()?;
        for optional in ["numpy", "torch", "jax"] {
            assert!(
                !loaded.contains(optional)?,
                "{optional} was loaded before the native control module initialized"
            );
        }

        let native_module = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
        let module = native_module.bind(py);
        assert_eq!(
            module.getattr("__version__")?.extract::<String>()?,
            "0.1.0a11"
        );
        for optional in ["numpy", "torch", "jax"] {
            assert!(
                !loaded.contains(optional)?,
                "initializing the control module imported optional module {optional}"
            );
        }
        let exception_base = module.getattr("EqioraError")?;
        let is_subclass = py.import("builtins")?.getattr("issubclass")?;
        for family in [
            "ValidationError",
            "CompatibilityError",
            "CapabilityError",
            "ExecutionError",
            "CancellationError",
            "InternalError",
        ] {
            assert!(
                is_subclass
                    .call1((module.getattr(family)?, &exception_base))?
                    .extract::<bool>()?,
                "{family} must remain a distinct EqioraError family"
            );
        }
        let manually_constructed = module.getattr("ValidationError")?.call1(("manual",))?;
        assert_eq!(
            manually_constructed
                .getattr("category")?
                .extract::<String>()?,
            "validation"
        );
        assert_eq!(manually_constructed.getattr("diagnostics")?.len()?, 0);

        let compile_kwargs = PyDict::new(py);
        compile_kwargs.set_item("filename", "python-control-plane.eqi")?;
        compile_kwargs.set_item("source", SOURCE)?;
        let base = module.getattr("compile")?.call((), Some(&compile_kwargs))?;
        let base_bytes = model_bytes(&base)?;
        let base_digest = base.getattr("digest")?.extract::<String>()?;
        let base_model_id = base.getattr("model_id")?.extract::<String>()?;
        let base_revision = revision_number(&base)?;

        let replayed = eqiora::api::ModelDocument::replay(&base_bytes)
            .expect("Python-produced current Model must replay through the Rust facade");
        let replayed_reference = replayed
            .artifact_reference()
            .expect("the validated replay must retain a typed Model reference");
        assert_eq!(
            replayed.canonical_json().unwrap(),
            base_bytes,
            "Python must expose the exact canonical artifact bytes"
        );
        assert_eq!(replayed.digest().unwrap(), base_digest);
        assert_eq!(replayed_reference.model().ulid().to_string(), base_model_id);
        assert_eq!(replayed_reference.semantic_revision().get(), base_revision);

        let edit = base.call_method1("preview_value_edit", ("rate", 2.0))?;
        assert_eq!(
            edit.getattr("base_digest")?.extract::<String>()?,
            base_digest
        );
        assert_eq!(
            edit.getattr("base_revision")?.extract::<u64>()?,
            base_revision
        );
        let child = base.call_method1("commit", (&edit,))?;
        assert_eq!(
            model_bytes(&base)?,
            base_bytes,
            "the base Model was mutated"
        );
        assert_eq!(base.getattr("digest")?.extract::<String>()?, base_digest);
        assert_eq!(
            base.getattr("model_id")?.extract::<String>()?,
            base_model_id
        );
        assert_eq!(revision_number(&base)?, base_revision);

        assert_eq!(
            child.getattr("model_id")?.extract::<String>()?,
            base_model_id,
            "an edit must advance one Model rather than minting another"
        );
        assert_eq!(revision_number(&child)?, base_revision + 1);
        assert_ne!(child.getattr("digest")?.extract::<String>()?, base_digest);
        let child_bytes = model_bytes(&child)?;
        let child_replay = eqiora::api::ModelDocument::replay(&child_bytes).unwrap();
        assert_eq!(child_replay.canonical_json().unwrap(), child_bytes);
        assert_eq!(
            child_replay.digest().unwrap(),
            child.getattr("digest")?.extract::<String>()?
        );

        let replayed_child = module
            .getattr("Model")?
            .getattr("from_bytes")?
            .call1((PyBytes::new(py, &child_bytes),))?;
        let parameter_id = replayed_child
            .getattr("parameter_ids")?
            .get_item(0)?
            .extract::<String>()?;
        let grandchild_edit =
            replayed_child.call_method1("preview_value_edit", (parameter_id, 3.0))?;
        let grandchild = replayed_child.call_method1("commit", (&grandchild_edit,))?;
        assert_eq!(revision_number(&grandchild)?, base_revision + 2);

        let sibling_edit = base.call_method1("preview_value_edit", ("rate", 3.0))?;
        let sibling = base.call_method1("commit", (&sibling_edit,))?;
        let child_parameter_edit = child.call_method1("preview_value_edit", ("rate", 4.0))?;
        let sibling_parameter_edit = sibling.call_method1("preview_value_edit", ("rate", 4.0))?;
        assert_ne!(
            child_parameter_edit.getattr("key")?.extract::<String>()?,
            sibling_parameter_edit.getattr("key")?.extract::<String>()?
        );
        assert!(
            !child_parameter_edit.eq(&sibling_parameter_edit)?,
            "edits over divergent base artifacts must retain distinct identity"
        );

        let stale = child
            .call_method1("commit", (&edit,))
            .expect_err("an edit prepared for the base must be stale on its child");
        assert_exception(
            module,
            py,
            stale,
            "ValidationError",
            "validation",
            Some("EQ0106"),
        )?;

        let malformed = module
            .getattr("Model")?
            .getattr("from_bytes")?
            .call1((PyBytes::new(py, b"{}"),))
            .expect_err("malformed current Model wire must fail closed");
        assert_exception(
            module,
            py,
            malformed,
            "CompatibilityError",
            "compatibility",
            Some("EQ0901"),
        )?;

        let invalid_kwargs = PyDict::new(py);
        invalid_kwargs.set_item("source", "model broken() { field ; }")?;
        let invalid_source = module
            .getattr("compile")?
            .call((), Some(&invalid_kwargs))
            .expect_err("invalid source must be rejected by the shared compiler");
        assert_exception(
            module,
            py,
            invalid_source,
            "ValidationError",
            "validation",
            None,
        )?;

        let probe = PyModule::new(py, "eqiora_boundary_probe")?;
        probe.add_function(wrap_pyfunction!(boundary_panic, &probe)?)?;
        let contained = probe
            .getattr("boundary_panic")?
            .call0()
            .expect_err("the boundary probe must panic before conversion");
        assert!(
            !contained.to_string().contains(PANIC_PAYLOAD),
            "the Rust panic payload crossed into the Python exception"
        );
        assert_exception(
            module,
            py,
            contained,
            "InternalError",
            "internal",
            Some("EQ0002"),
        )?;

        assert_panic_hook_is_sanitized();

        Ok(())
    })
}

#[test]
fn panic_probe_child() {
    if std::env::var_os(PANIC_PROBE_CHILD).is_none() {
        return;
    }
    Python::initialize();
    Python::attach(|py| {
        let error = _eqiora::panic_boundary(py, || -> PyResult<()> { panic!("{PANIC_PAYLOAD}") })
            .expect_err("probe panic must become one Python error");
        assert!(!error.to_string().contains(PANIC_PAYLOAD));
    });
}

fn assert_panic_hook_is_sanitized() {
    let output = Command::new(std::env::current_exe().expect("current integration-test binary"))
        .args(["--exact", "panic_probe_child", "--nocapture"])
        .env(PANIC_PROBE_CHILD, "1")
        .env("RUST_BACKTRACE", "1")
        .output()
        .expect("panic containment subprocess must start");
    assert!(
        output.status.success(),
        "panic containment subprocess failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains(PANIC_PAYLOAD) && !stderr.contains("python_control_plane.rs"),
        "panic hook disclosed its payload or Rust location: {stderr}"
    );
}

fn assert_python_compile_reaches_operation(
    module: &Bound<'_, PyModule>,
    source: &str,
    filename: &str,
) -> PyResult<()> {
    let kwargs = PyDict::new(module.py());
    kwargs.set_item("filename", filename)?;
    kwargs.set_item("source", source)?;
    match module.getattr("compile")?.call((), Some(&kwargs)) {
        Ok(_) => Ok(()),
        Err(error) => {
            let (_, diagnostics) = normalize_python_error(module.py(), &error)?;
            assert!(!diagnostics.is_empty());
            assert!(
                diagnostics
                    .iter()
                    .all(|diagnostic| diagnostic.source != "control"),
                "an exact admitted boundary witness failed before ModelDocument::compile: {diagnostics:?}"
            );
            Ok(())
        }
    }
}

fn assert_python_admission_error(
    module: &Bound<'_, PyModule>,
    source: &str,
    filename: &str,
    message: &str,
) -> PyResult<()> {
    let kwargs = PyDict::new(module.py());
    kwargs.set_item("filename", filename)?;
    kwargs.set_item("source", source)?;
    let error = module
        .getattr("compile")?
        .call((), Some(&kwargs))
        .expect_err("an over-bound Python input reached compilation");
    assert!(error.is_instance(module.py(), &module.getattr("ValidationError")?));
    let (category, diagnostics) = normalize_python_error(module.py(), &error)?;
    assert_eq!(category, "validation");
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0],
        NormalizedDiagnostic {
            source: "control".to_owned(),
            severity: "error".to_owned(),
            code: "EQ0901".to_owned(),
            message: message.to_owned(),
            graph_path: None,
            source_span: None,
            suggestion: None,
        }
    );
    Ok(())
}

fn normalize_kernel(diagnostic: &Diagnostic) -> NormalizedDiagnostic {
    NormalizedDiagnostic {
        source: "kernel".to_owned(),
        severity: match diagnostic.severity() {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
        .to_owned(),
        code: diagnostic.code().to_string(),
        message: diagnostic.message().to_owned(),
        graph_path: diagnostic.graph_path().map(|path| path.segments().to_vec()),
        source_span: diagnostic
            .source_span()
            .map(|span| (span.file.clone(), span.start, span.end)),
        suggestion: diagnostic.suggestion().map(|patch| patch.summary.clone()),
    }
}

fn normalize_control(diagnostic: &ControlDiagnosticV2) -> NormalizedDiagnostic {
    NormalizedDiagnostic {
        source: match diagnostic.source() {
            ControlDiagnosticSourceV2::Control => "control",
            ControlDiagnosticSourceV2::Kernel => "kernel",
        }
        .to_owned(),
        severity: match diagnostic.severity() {
            ControlSeverityV2::Error => "error",
            ControlSeverityV2::Warning => "warning",
            ControlSeverityV2::Note => "note",
        }
        .to_owned(),
        code: diagnostic.code().to_owned(),
        message: diagnostic.message().to_owned(),
        graph_path: diagnostic.graph_path().map(<[String]>::to_vec),
        source_span: diagnostic
            .span()
            .map(|span| (span.file().to_owned(), span.start(), span.end())),
        suggestion: diagnostic.patch().map(|patch| patch.summary().to_owned()),
    }
}

fn normalize_python_error(
    py: Python<'_>,
    error: &PyErr,
) -> PyResult<(String, Vec<NormalizedDiagnostic>)> {
    let value = error.value(py);
    let category = value.getattr("category")?.extract::<String>()?;
    let diagnostics = value.getattr("diagnostics")?;
    let mut normalized = Vec::with_capacity(diagnostics.len()?);
    for index in 0..diagnostics.len()? {
        let diagnostic = diagnostics.get_item(index)?;
        normalized.push(NormalizedDiagnostic {
            source: diagnostic.getattr("source")?.extract::<String>()?,
            severity: diagnostic.getattr("severity")?.extract::<String>()?,
            code: diagnostic.getattr("code")?.extract::<String>()?,
            message: diagnostic.getattr("message")?.extract::<String>()?,
            graph_path: diagnostic
                .getattr("graph_path")?
                .extract::<Option<Vec<String>>>()?,
            source_span: diagnostic
                .getattr("source_span")?
                .extract::<Option<(String, u32, u32)>>()?,
            suggestion: diagnostic
                .getattr("suggestion")?
                .extract::<Option<String>>()?,
        });
    }
    Ok((category, normalized))
}

fn collect_rust_sources(directory: &Path, output: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_rust_sources(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
}

fn model_bytes(model: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    model.call_method0("to_bytes")?.extract()
}

fn revision_number(model: &Bound<'_, PyAny>) -> PyResult<u64> {
    model.getattr("revision")?.getattr("number")?.extract()
}

fn assert_exception(
    module: &Bound<'_, PyModule>,
    py: Python<'_>,
    error: PyErr,
    class: &str,
    category: &str,
    code: Option<&str>,
) -> PyResult<()> {
    assert!(
        error.is_instance(py, &module.getattr(class)?),
        "expected {class}, received {error}"
    );
    let value = error.value(py);
    assert_eq!(value.getattr("category")?.extract::<String>()?, category);
    let diagnostics = value.getattr("diagnostics")?;
    assert!(diagnostics.len()? > 0, "structured diagnostics were lost");
    if let Some(code) = code {
        assert_eq!(
            diagnostics
                .get_item(0)?
                .getattr("code")?
                .extract::<String>()?,
            code
        );
    }
    Ok(())
}
