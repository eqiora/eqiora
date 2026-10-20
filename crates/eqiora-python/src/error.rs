//! Stable, structured failures at the Python language boundary.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Once;

use eqiora::{Diagnostic, Severity};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

create_exception!(
    eqiora._eqiora,
    EqioraError,
    PyException,
    "Base class for an Eqiora operation rejected with structured diagnostics."
);

thread_local! {
    static SANITIZE_BOUNDARY_PANIC: Cell<bool> = const { Cell::new(false) };
}

static INSTALL_BOUNDARY_PANIC_HOOK: Once = Once::new();

struct BoundaryPanicGuard {
    previous: bool,
}

impl BoundaryPanicGuard {
    fn enter() -> Self {
        let previous = SANITIZE_BOUNDARY_PANIC.replace(true);
        Self { previous }
    }
}

impl Drop for BoundaryPanicGuard {
    fn drop(&mut self) {
        SANITIZE_BOUNDARY_PANIC.set(self.previous);
    }
}
create_exception!(
    eqiora._eqiora,
    ValidationError,
    EqioraError,
    "The submitted model or request violates a typed Eqiora contract."
);
create_exception!(
    eqiora._eqiora,
    CompatibilityError,
    EqioraError,
    "A versioned or persisted representation is incompatible with the selected contract."
);
create_exception!(
    eqiora._eqiora,
    CapabilityError,
    EqioraError,
    "The selected adapter does not provide a required capability."
);
create_exception!(
    eqiora._eqiora,
    ExecutionError,
    EqioraError,
    "An admitted execution failed."
);
create_exception!(
    eqiora._eqiora,
    CancellationError,
    EqioraError,
    "An Eqiora operation was cancelled."
);
create_exception!(
    eqiora._eqiora,
    InternalError,
    EqioraError,
    "Eqiora failed internally without exposing implementation details."
);

/// Stable failure families exposed by the Python SDK.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorCategory {
    Validation,
    Compatibility,
    Capability,
    Execution,
    Cancellation,
    Internal,
}

impl ErrorCategory {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::Compatibility => "compatibility",
            Self::Capability => "capability",
            Self::Execution => "execution",
            Self::Cancellation => "cancellation",
            Self::Internal => "internal",
        }
    }

    fn from_code(code: &str) -> Self {
        match code {
            "EQ0001" => Self::Capability,
            "EQ0002" => Self::Internal,
            "EQ0901" => Self::Compatibility,
            "EQ0506" => Self::Cancellation,
            "EQ0501" | "EQ0504" | "EQ0505" | "EQ0507" | "EQ0802" => Self::Execution,
            _ => Self::Validation,
        }
    }

    fn exception(self, message: String) -> PyErr {
        match self {
            Self::Validation => ValidationError::new_err(message),
            Self::Compatibility => CompatibilityError::new_err(message),
            Self::Capability => CapabilityError::new_err(message),
            Self::Execution => ExecutionError::new_err(message),
            Self::Cancellation => CancellationError::new_err(message),
            Self::Internal => InternalError::new_err(message),
        }
    }
}

/// Immutable, lossless projection of the fields in a current Rust diagnostic.
#[pyclass(
    name = "Diagnostic",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyDiagnostic {
    source: String,
    code: String,
    severity: String,
    message: String,
    graph_path: Option<Vec<String>>,
    source_span: Option<(String, u32, u32)>,
    suggestion: Option<String>,
}

#[pymethods]
impl PyDiagnostic {
    #[getter]
    fn source(&self) -> &str {
        &self.source
    }

    #[getter]
    fn code(&self) -> &str {
        &self.code
    }

    #[getter]
    fn severity(&self) -> &str {
        &self.severity
    }

    #[getter]
    fn message(&self) -> &str {
        &self.message
    }

    #[getter]
    fn graph_path(&self) -> Option<Vec<String>> {
        self.graph_path.clone()
    }

    #[getter]
    fn source_span(&self) -> Option<(String, u32, u32)> {
        self.source_span.clone()
    }

    #[getter]
    fn suggestion(&self) -> Option<String> {
        self.suggestion.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Diagnostic(source={:?}, code={:?}, severity={:?}, message={:?})",
            self.source, self.code, self.severity, self.message
        )
    }
}

impl From<&Diagnostic> for PyDiagnostic {
    fn from(diagnostic: &Diagnostic) -> Self {
        let severity = match diagnostic.severity() {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        };
        Self {
            source: "kernel".to_owned(),
            code: diagnostic.code().to_string(),
            severity: severity.to_owned(),
            message: diagnostic.message().to_owned(),
            graph_path: diagnostic.graph_path().map(|path| path.segments().to_vec()),
            source_span: diagnostic
                .source_span()
                .map(|span| (span.file.clone(), span.start, span.end)),
            suggestion: diagnostic.suggestion().map(|patch| patch.summary.clone()),
        }
    }
}

pub(crate) fn diagnostic_error(py: Python<'_>, diagnostics: &[Diagnostic]) -> PyErr {
    structured_diagnostic_error(py, diagnostics.iter().map(PyDiagnostic::from).collect())
}

pub(crate) fn validation_error(py: Python<'_>, diagnostics: &[Diagnostic]) -> PyErr {
    structured_diagnostic_error_as(
        py,
        ErrorCategory::Validation,
        diagnostics.iter().map(PyDiagnostic::from).collect(),
    )
}

pub(crate) fn compatibility_error(py: Python<'_>, diagnostics: &[Diagnostic]) -> PyErr {
    structured_diagnostic_error_as(
        py,
        ErrorCategory::Compatibility,
        diagnostics.iter().map(PyDiagnostic::from).collect(),
    )
}

pub(crate) fn execution_error(py: Python<'_>, diagnostics: &[Diagnostic]) -> PyErr {
    structured_diagnostic_error_as(
        py,
        ErrorCategory::Execution,
        diagnostics.iter().map(PyDiagnostic::from).collect(),
    )
}

pub(crate) fn cancellation_error(py: Python<'_>, diagnostics: &[Diagnostic]) -> PyErr {
    structured_diagnostic_error_as(
        py,
        ErrorCategory::Cancellation,
        diagnostics.iter().map(PyDiagnostic::from).collect(),
    )
}

pub(crate) fn internal_diagnostic_error(py: Python<'_>, diagnostics: &[Diagnostic]) -> PyErr {
    structured_diagnostic_error_as(
        py,
        ErrorCategory::Internal,
        diagnostics.iter().map(PyDiagnostic::from).collect(),
    )
}

pub(crate) fn python_compile_admission_error(py: Python<'_>, message: &str) -> PyErr {
    structured_diagnostic_error_as(
        py,
        ErrorCategory::Validation,
        vec![PyDiagnostic {
            source: "control".to_owned(),
            code: "EQ0901".to_owned(),
            severity: "error".to_owned(),
            message: message.to_owned(),
            graph_path: None,
            source_span: None,
            suggestion: None,
        }],
    )
}

pub(crate) fn internal_error(py: Python<'_>, message: &str) -> PyErr {
    structured_diagnostic_error(py, vec![internal_failure_diagnostic(message)])
}

/// Contain a panic on a native worker that never owns Python objects.
pub(crate) fn catch_native_panic<T>(operation: impl FnOnce() -> T) -> Result<T, Diagnostic> {
    install_boundary_panic_hook();
    let _guard = BoundaryPanicGuard::enter();
    catch_unwind(AssertUnwindSafe(operation)).map_err(|_| {
        Diagnostic::error(
            eqiora::diagnostic::codes::INTERNAL_FAILURE,
            "the native Python worker failed internally",
        )
    })
}

/// Catch a Rust panic at one Python entry point without disclosing its payload.
///
/// One process hook wrapper delegates every panic outside the guarded thread
/// to the hook that preceded Eqiora. While this thread is inside a boundary it
/// suppresses the upstream hook, so payload and Rust location cannot escape on
/// stderr before `catch_unwind` constructs the sanitized Python exception.
#[doc(hidden)]
pub fn panic_boundary<T>(py: Python<'_>, operation: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    install_boundary_panic_hook();
    let _guard = BoundaryPanicGuard::enter();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(_) => Err(structured_diagnostic_error(
            py,
            vec![internal_failure_diagnostic(
                "the native Python boundary failed internally",
            )],
        )),
    }
}

fn install_boundary_panic_hook() {
    INSTALL_BOUNDARY_PANIC_HOOK.call_once(|| {
        let upstream = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |information| {
            let sanitize = SANITIZE_BOUNDARY_PANIC.try_with(Cell::get).unwrap_or(false);
            if !sanitize {
                upstream(information);
            }
        }));
    });
}

fn structured_diagnostic_error(py: Python<'_>, mut diagnostics: Vec<PyDiagnostic>) -> PyErr {
    if diagnostics.is_empty() {
        diagnostics.push(internal_failure_diagnostic(
            "the native Python boundary failed internally",
        ));
    }
    let category = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.severity == "error")
        .or_else(|| diagnostics.first())
        .map_or(ErrorCategory::Internal, |diagnostic| {
            ErrorCategory::from_code(&diagnostic.code)
        });
    structured_diagnostic_error_as(py, category, diagnostics)
}

fn structured_diagnostic_error_as(
    py: Python<'_>,
    category: ErrorCategory,
    mut diagnostics: Vec<PyDiagnostic>,
) -> PyErr {
    if diagnostics.is_empty() {
        diagnostics.push(internal_failure_diagnostic(
            "the native Python boundary failed internally",
        ));
    }
    let message = diagnostics
        .iter()
        .map(|diagnostic| format!("{}: {}", diagnostic.code(), diagnostic.message()))
        .collect::<Vec<_>>()
        .join("\n");
    let error = category.exception(message);
    let records = diagnostics
        .into_iter()
        .map(|diagnostic| Py::new(py, diagnostic))
        .collect::<PyResult<Vec<_>>>();
    match records.and_then(|records| PyTuple::new(py, records)) {
        Ok(records) => {
            if let Err(attribute_error) = error.value(py).setattr("diagnostics", records) {
                return attribute_error;
            }
            if let Err(attribute_error) = error.value(py).setattr("category", category.as_str()) {
                return attribute_error;
            }
            error
        }
        Err(allocation_error) => allocation_error,
    }
}

fn internal_failure_diagnostic(message: &str) -> PyDiagnostic {
    PyDiagnostic {
        source: "python".to_owned(),
        code: eqiora::diagnostic::codes::INTERNAL_FAILURE.to_string(),
        severity: "error".to_owned(),
        message: message.to_owned(),
        graph_path: None,
        source_span: None,
        suggestion: None,
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let base = py.get_type::<EqioraError>();
    base.setattr("category", "unknown")?;
    base.setattr("diagnostics", PyTuple::empty(py))?;
    module.add("EqioraError", &base)?;
    for (name, exception, category) in [
        (
            "ValidationError",
            py.get_type::<ValidationError>(),
            ErrorCategory::Validation,
        ),
        (
            "CompatibilityError",
            py.get_type::<CompatibilityError>(),
            ErrorCategory::Compatibility,
        ),
        (
            "CapabilityError",
            py.get_type::<CapabilityError>(),
            ErrorCategory::Capability,
        ),
        (
            "ExecutionError",
            py.get_type::<ExecutionError>(),
            ErrorCategory::Execution,
        ),
        (
            "CancellationError",
            py.get_type::<CancellationError>(),
            ErrorCategory::Cancellation,
        ),
        (
            "InternalError",
            py.get_type::<InternalError>(),
            ErrorCategory::Internal,
        ),
    ] {
        exception.setattr("category", category.as_str())?;
        exception.setattr("diagnostics", PyTuple::empty(py))?;
        module.add(name, exception)?;
    }
    module.add_class::<PyDiagnostic>()?;
    Ok(())
}

impl From<&eqiora::api::DiagnosticRecord> for PyDiagnostic {
    fn from(value: &eqiora::api::DiagnosticRecord) -> Self {
        use eqiora::Severity;
        use eqiora::api::DiagnosticSource;
        Self {
            source: match value.source() {
                DiagnosticSource::Control => "control",
                DiagnosticSource::Kernel => "kernel",
            }
            .to_owned(),
            severity: match value.severity() {
                Severity::Error => "error",
                Severity::Warning => "warning",
                Severity::Note => "note",
            }
            .to_owned(),
            code: value.code().to_owned(),
            message: value.message().to_owned(),
            graph_path: value.graph_path().map(<[String]>::to_vec),
            source_span: value
                .span()
                .map(|span| (span.file.clone(), span.start, span.end)),
            suggestion: value.patch().map(|patch| patch.summary.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ErrorCategory, catch_native_panic};

    #[test]
    fn stable_codes_select_specific_python_failure_families() {
        assert_eq!(
            ErrorCategory::from_code("EQ0602"),
            ErrorCategory::Validation
        );
        assert_eq!(
            ErrorCategory::from_code("EQ0901"),
            ErrorCategory::Compatibility
        );
        assert_eq!(
            ErrorCategory::from_code("EQ0001"),
            ErrorCategory::Capability
        );
        assert_eq!(ErrorCategory::from_code("EQ0504"), ErrorCategory::Execution);
        assert_eq!(ErrorCategory::from_code("EQ0507"), ErrorCategory::Execution);
        assert_eq!(
            ErrorCategory::from_code("EQ0506"),
            ErrorCategory::Cancellation
        );
        assert_eq!(ErrorCategory::from_code("EQ0002"), ErrorCategory::Internal);
    }

    #[test]
    fn every_failure_family_has_a_stable_machine_name() {
        for (category, expected) in [
            (ErrorCategory::Validation, "validation"),
            (ErrorCategory::Compatibility, "compatibility"),
            (ErrorCategory::Capability, "capability"),
            (ErrorCategory::Execution, "execution"),
            (ErrorCategory::Cancellation, "cancellation"),
            (ErrorCategory::Internal, "internal"),
        ] {
            assert_eq!(category.as_str(), expected);
        }
    }

    #[test]
    fn native_worker_panics_become_sanitized_internal_diagnostics() {
        let diagnostic = catch_native_panic(|| panic!("private worker payload"))
            .expect_err("the worker probe must panic");
        assert_eq!(diagnostic.code().to_string(), "EQ0002");
        assert!(!diagnostic.message().contains("private worker payload"));
    }
}
