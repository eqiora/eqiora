//! Bounded diagnostic records shared by persisted evaluations and client protocols.
use core::fmt;

use eqiora_core::{Diagnostic, Severity};
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

/// Stable severity spelling in persisted diagnostic records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DiagnosticSeverity {
    /// The operation failed.
    Error,
    /// The operation produced a result with an attached warning.
    Warning,
    /// Supporting information for another diagnostic.
    Note,
}

/// Source byte range retained by a diagnostic record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiagnosticSpan {
    file: String,
    start: u32,
    end: u32,
}

impl DiagnosticSpan {
    /// Workspace-relative source filename.
    #[must_use]
    pub fn file(&self) -> &str {
        &self.file
    }

    /// Inclusive starting byte offset.
    #[must_use]
    pub const fn start(&self) -> u32 {
        self.start
    }

    /// Exclusive ending byte offset.
    #[must_use]
    pub const fn end(&self) -> u32 {
        self.end
    }
}

/// Bounded summary-only diagnostic suggestion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiagnosticSuggestion {
    summary: String,
}

impl DiagnosticSuggestion {
    /// Human-readable patch summary.
    #[must_use]
    pub fn summary(&self) -> &str {
        &self.summary
    }
}

/// Transport-neutral projection of one structured Eqiora diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiagnosticRecord {
    source: DiagnosticSource,
    severity: DiagnosticSeverity,
    code: String,
    message: String,
    graph_path: Option<Box<[String]>>,
    span: Option<Box<DiagnosticSpan>>,
    patch: Option<Box<DiagnosticSuggestion>>,
}

impl DiagnosticRecord {
    /// Subsystem that admitted the diagnostic.
    #[must_use]
    pub const fn source(&self) -> DiagnosticSource {
        self.source
    }

    /// Stable severity.
    #[must_use]
    pub const fn severity(&self) -> DiagnosticSeverity {
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
    pub fn span(&self) -> Option<&DiagnosticSpan> {
        self.span.as_deref()
    }

    /// Machine-applicable suggestion, when one is available.
    #[must_use]
    pub fn patch(&self) -> Option<&DiagnosticSuggestion> {
        self.patch.as_deref()
    }

    pub(crate) fn from_kernel(diagnostic: Diagnostic) -> Result<Self, ()> {
        let severity = match diagnostic.severity() {
            Severity::Error => DiagnosticSeverity::Error,
            Severity::Warning => DiagnosticSeverity::Warning,
            Severity::Note => DiagnosticSeverity::Note,
        };
        let value = Self {
            source: DiagnosticSource::Kernel,
            severity,
            code: diagnostic.code().to_string(),
            message: diagnostic.message().to_owned(),
            graph_path: diagnostic
                .graph_path()
                .map(|path| path.segments().to_vec().into_boxed_slice()),
            span: diagnostic.source_span().map(|span| {
                Box::new(DiagnosticSpan {
                    file: span.file.clone(),
                    start: span.start,
                    end: span.end,
                })
            }),
            patch: diagnostic.suggestion().map(|patch| {
                Box::new(DiagnosticSuggestion {
                    summary: patch.summary.clone(),
                })
            }),
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
        severity: DiagnosticSeverity,
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
