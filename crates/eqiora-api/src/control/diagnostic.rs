//! Control-specific admission failures over the shared diagnostic record.
use crate::{DiagnosticRecord, DiagnosticSource};
use eqiora_core::Severity;

impl DiagnosticRecord {
    pub(crate) fn invalid_request(message: impl Into<String>) -> Self {
        Self::from_message(
            DiagnosticSource::Control,
            Severity::Error,
            "EQ0901",
            message,
        )
    }

    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self::from_message(
            DiagnosticSource::Control,
            Severity::Error,
            "EQ0001",
            message,
        )
    }

    pub(crate) fn diagnostics_overflow() -> Self {
        Self::invalid_request("compile/check diagnostics exceed the control v2 response limits")
    }
}
