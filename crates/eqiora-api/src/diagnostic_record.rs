//! Bounded diagnostic records shared by persisted evaluations and client protocols.
use core::fmt;

use eqiora_core::diagnostic::Patch;
use eqiora_core::{Diagnostic, Severity, Span};
use serde::{Deserialize, Serialize};

/// Largest admitted diagnostic message.
pub(crate) const MAX_DIAGNOSTIC_MESSAGE_BYTES: usize = 1_024 * 1_024;
/// Largest admitted graph path.
pub(crate) const MAX_DIAGNOSTIC_GRAPH_PATH_SEGMENTS: usize = 256;
/// Largest admitted filename, graph segment, or suggestion summary.
pub(crate) const MAX_DIAGNOSTIC_TEXT_MEMBER_BYTES: usize = 4_096;

/// Origin retained by a structured diagnostic record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DiagnosticSource {
    /// The request failed the control contract.
    Control,
    /// The ordinary compiler, semantic validator, or artifact boundary
    /// rejected an admitted request.
    Kernel,
}

/// Transport-neutral projection of one structured Eqiora diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiagnosticRecord {
    source: DiagnosticSource,
    #[serde(with = "wire::severity")]
    severity: Severity,
    code: String,
    message: String,
    graph_path: Option<Box<[String]>>,
    #[serde(default, with = "wire::span")]
    span: Option<Box<Span>>,
    #[serde(default, with = "wire::patch")]
    patch: Option<Box<Patch>>,
}

impl DiagnosticRecord {
    /// Subsystem that admitted the diagnostic.
    #[must_use]
    pub const fn source(&self) -> DiagnosticSource {
        self.source
    }

    /// Stable severity.
    #[must_use]
    pub const fn severity(&self) -> Severity {
        self.severity
    }

    /// Stable append-only diagnostic identity.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Human-readable explanation.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Structured graph path, when one is available.
    #[must_use]
    pub fn graph_path(&self) -> Option<&[String]> {
        self.graph_path.as_deref()
    }

    /// Source byte range, when one is available.
    #[must_use]
    pub fn span(&self) -> Option<&Span> {
        self.span.as_deref()
    }

    /// Machine-applicable suggestion, when one is available.
    #[must_use]
    pub fn patch(&self) -> Option<&Patch> {
        self.patch.as_deref()
    }

    pub(crate) fn from_kernel(diagnostic: Diagnostic) -> Result<Self, ()> {
        let value = Self {
            source: DiagnosticSource::Kernel,
            severity: diagnostic.severity(),
            code: diagnostic.code().to_string(),
            message: diagnostic.message().to_owned(),
            graph_path: diagnostic
                .graph_path()
                .map(|path| path.segments().to_vec().into_boxed_slice()),
            span: diagnostic.source_span().cloned().map(Box::new),
            patch: diagnostic.suggestion().cloned().map(Box::new),
        };
        value.validate()?;
        Ok(value)
    }

    pub(crate) fn validate(&self) -> Result<(), ()> {
        let valid_code = self.code.len() == 6
            && self.code.as_bytes()[..2].iter().all(u8::is_ascii_uppercase)
            && self.code.as_bytes()[2..].iter().all(u8::is_ascii_digit);
        let valid_message = bounded_nonempty(&self.message, MAX_DIAGNOSTIC_MESSAGE_BYTES);
        let valid_path = self.graph_path.as_deref().is_none_or(|path| {
            path.len() <= MAX_DIAGNOSTIC_GRAPH_PATH_SEGMENTS
                && path
                    .iter()
                    .all(|segment| bounded_nonempty(segment, MAX_DIAGNOSTIC_TEXT_MEMBER_BYTES))
        });
        let valid_span = self.span.as_deref().is_none_or(|span| {
            span.file.chars().count() <= MAX_DIAGNOSTIC_TEXT_MEMBER_BYTES
                && span.file.len() <= MAX_DIAGNOSTIC_TEXT_MEMBER_BYTES
                && span.end >= span.start
        });
        let valid_patch = self
            .patch
            .as_deref()
            .is_none_or(|patch| bounded_nonempty(&patch.summary, MAX_DIAGNOSTIC_TEXT_MEMBER_BYTES));
        if valid_code && valid_message && valid_path && valid_span && valid_patch {
            Ok(())
        } else {
            Err(())
        }
    }

    pub(crate) fn from_message(
        source: DiagnosticSource,
        severity: Severity,
        code: &str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            source,
            severity,
            code: code.to_owned(),
            message: message.into(),
            graph_path: None,
            span: None,
            patch: None,
        }
    }
}

impl fmt::Display for DiagnosticRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

fn bounded_nonempty(value: &str, maximum: usize) -> bool {
    !value.is_empty() && value.chars().count() <= maximum && value.len() <= maximum
}

mod wire;

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{GraphPath, diagnostic::codes};

    #[test]
    fn shared_kernel_details_preserve_the_closed_diagnostic_wire() {
        let source = Diagnostic::warning(codes::INVALID_KERNEL_DEFINITION, "explain")
            .with_graph_path(GraphPath::new(["model", "field"]))
            .with_span(Span {
                file: "model.eqi".into(),
                start: 2,
                end: 7,
            })
            .with_suggestion(Patch::new("repair"));
        let record = DiagnosticRecord::from_kernel(source.clone()).unwrap();
        assert_eq!(record.severity(), source.severity());
        assert_eq!(record.span(), source.source_span());
        assert_eq!(record.patch(), source.suggestion());
        let wire = serde_json::json!({
            "source": "kernel", "severity": "warning", "code": "EQ0302",
            "message": "explain", "graphPath": ["model", "field"],
            "span": {"file": "model.eqi", "start": 2, "end": 7},
            "patch": {"summary": "repair"}
        });
        assert_eq!(serde_json::to_value(&record).unwrap(), wire);
        let replay: DiagnosticRecord = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(replay, record);
        replay.validate().unwrap();
        for member in ["span", "patch"] {
            let mut extra = wire.clone();
            extra[member]["extra"] = serde_json::json!(true);
            assert!(serde_json::from_value::<DiagnosticRecord>(extra).is_err());
        }
        let without_details = serde_json::json!({
            "source": "kernel", "severity": "note", "code": "EQ0302", "message": "explain"
        });
        let replay: DiagnosticRecord = serde_json::from_value(without_details).unwrap();
        replay.validate().unwrap();
        assert!(replay.span().is_none());
        assert!(replay.patch().is_none());
    }
}
