//! Engineering-aware diagnostics.
//!
//! Every crate keeps its own internal `Error` type and converts to
//! [`Diagnostic`] at its public boundary — a bare `String` error never
//! crosses a public API. No C ABI is implemented. A future foreign ABI must
//! use a versioned serialized buffer behind an opaque handle rather than
//! treating the struct layout below as ABI.

use core::fmt;

/// Diagnostic severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    /// The operation cannot proceed.
    Error,
    /// The operation proceeds, but the result deserves attention.
    Warning,
    /// Additional context attached to another diagnostic.
    Note,
}

/// Stable diagnostic code (e.g. `EQ0401`).
///
/// Codes are append-only and never reused; the registry lives in
/// `docs/diagnostics.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Code(pub &'static str);

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// Well-known diagnostic codes. `EQ00xx` = infrastructure, `EQ01xx` = graph,
/// `EQ02xx` = ontology views, `EQ03xx` = semantic definitions, and `EQ04xx` =
/// units and dimensions, `EQ05xx` = reference execution, and `EQ08xx` =
/// numerical realization. Ranges are assigned in `docs/diagnostics.md`.
pub mod codes {
    use super::Code;

    /// Feature is specified but not yet implemented.
    pub const NOT_IMPLEMENTED: Code = Code("EQ0001");
    /// An internal invariant failed at a public language or process boundary.
    pub const INTERNAL_FAILURE: Code = Code("EQ0002");
    /// A graph operation references a node that does not exist.
    pub const NODE_NOT_FOUND: Code = Code("EQ0101");
    /// A graph operation attempts to reuse an existing identifier.
    pub const NODE_ALREADY_EXISTS: Code = Code("EQ0102");
    /// The runtime kind attached to an erased ID does not match the request.
    pub const ID_KIND_MISMATCH: Code = Code("EQ0103");
    /// An edge is not permitted by the kernel edge schema.
    pub const INVALID_EDGE: Code = Code("EQ0104");
    /// An operation is not valid for the target entity kind.
    pub const INVALID_OPERATION: Code = Code("EQ0105");
    /// An optimistic-concurrency precondition failed.
    pub const PRECONDITION_FAILED: Code = Code("EQ0106");
    /// An immutable provenance record was targeted for mutation.
    pub const IMMUTABLE_PROVENANCE: Code = Code("EQ0107");
    /// A named subgraph violates a schema-independent structural invariant.
    pub const INVALID_ONTOLOGY_VIEW: Code = Code("EQ0201");
    /// An ontology-view identifier is already present in the registry.
    pub const ONTOLOGY_VIEW_ALREADY_EXISTS: Code = Code("EQ0202");
    /// An ontology-view identifier is not present in the registry.
    pub const ONTOLOGY_VIEW_NOT_FOUND: Code = Code("EQ0203");
    /// A kernel node cannot be removed while an ontology view references it.
    pub const NODE_REFERENCED_BY_ONTOLOGY_VIEW: Code = Code("EQ0204");
    /// An expression arena is empty, cyclic, or references an invalid index.
    pub const INVALID_EXPRESSION_DAG: Code = Code("EQ0301");
    /// A typed Semantic Kernel node definition violates its local invariant.
    pub const INVALID_KERNEL_DEFINITION: Code = Code("EQ0302");
    /// A semantic expression references a symbol outside the selected model.
    pub const UNRESOLVED_SYMBOL: Code = Code("EQ0303");
    /// A residual expression combines incompatible physical dimensions.
    pub const INVALID_RELATION_DIMENSION: Code = Code("EQ0304");
    /// A ClockDomain has an invalid exact-time definition.
    pub const INVALID_CLOCK: Code = Code("EQ0305");
    /// Runtime dimension does not match the statically expected dimension.
    pub const DIMENSION_MISMATCH: Code = Code("EQ0401");
    /// A reference-execution request contains an invalid time or limit.
    pub const INVALID_EXECUTION_CONFIG: Code = Code("EQ0501");
    /// A required initial value or external execution input is absent.
    pub const MISSING_EXECUTION_INPUT: Code = Code("EQ0502");
    /// The active implicit system is not square in executable-kernel v0.
    pub const NONSQUARE_SYSTEM: Code = Code("EQ0503");
    /// The reference nonlinear solve failed to converge or was singular.
    pub const NONLINEAR_SOLVE_FAILED: Code = Code("EQ0504");
    /// Evaluation produced NaN or infinity.
    pub const NONFINITE_EVALUATION: Code = Code("EQ0505");
    /// Cooperative execution cancellation was accepted at a safe boundary.
    pub const EXECUTION_CANCELLED: Code = Code("EQ0506");
    /// Eqiora Language input contains an invalid token.
    pub const INVALID_TOKEN: Code = Code("EQ0601");
    /// Eqiora Language input does not satisfy the source grammar.
    pub const SYNTAX_ERROR: Code = Code("EQ0602");
    /// Eqiora Language name or static type cannot be resolved.
    pub const LANGUAGE_TYPE_ERROR: Code = Code("EQ0603");
    /// Typed source could not be lowered to a Graph Federation transaction.
    pub const LANGUAGE_LOWERING_ERROR: Code = Code("EQ0604");
    /// Operator IR is structurally invalid or inconsistent with its source.
    pub const INVALID_OPERATOR_IR: Code = Code("EQ0701");
    /// Operator IR received the wrong number of scalar symbol inputs.
    pub const OPERATOR_INPUT_MISMATCH: Code = Code("EQ0702");
    /// A canonical spatial Relation cannot be lowered by the selected realization.
    pub const INVALID_SPATIAL_LOWERING: Code = Code("EQ0703");
    /// A linearization point, variable binding, tangent, or cotangent is invalid.
    pub const INVALID_LINEARIZATION: Code = Code("EQ0704");
    /// A continuous subsystem cannot be lowered to the selected time-equation class.
    pub const INVALID_TIME_LOWERING: Code = Code("EQ0705");
    /// A numerical grid, coefficient, time step, or state is invalid.
    pub const INVALID_DISCRETIZATION: Code = Code("EQ0801");
    /// A numerical linear solve failed or produced a non-finite result.
    pub const NUMERICAL_SOLVE_FAILED: Code = Code("EQ0802");
    /// Mesh topology or geometry violates a realization invariant.
    pub const INVALID_MESH: Code = Code("EQ0803");
    /// A quadrature rule is incompatible with its reference cell or invalid.
    pub const INVALID_QUADRATURE: Code = Code("EQ0804");
    /// A local operator contribution has invalid shape or values.
    pub const INVALID_LOCAL_CONTRIBUTION: Code = Code("EQ0805");
    /// A local-to-global assembly map or sparse accumulation is invalid.
    pub const ASSEMBLY_FAILED: Code = Code("EQ0806");
    /// A realization policy is invalid, contradictory, or unsupported.
    pub const INVALID_REALIZATION: Code = Code("EQ0807");
    /// External mesh input violates an admitted importer boundary.
    pub const INVALID_MESH_IMPORT: Code = Code("EQ0808");
    /// A mesh-associated discrete field violates its shape or value contract.
    pub const INVALID_DISCRETE_FIELD: Code = Code("EQ0809");
    /// External scientific data violates an admitted adapter or resolver boundary.
    pub const INVALID_EXTERNAL_DATA_IMPORT: Code = Code("EQ0810");
    /// Accepted scientific data cannot be projected through an export adapter.
    pub const INVALID_EXTERNAL_DATA_EXPORT: Code = Code("EQ0811");
    /// A serialized artifact, digest, or run manifest is invalid.
    pub const INVALID_ARTIFACT: Code = Code("EQ0901");
}

/// Dot-separated path to the graph node that caused a diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct GraphPath {
    segments: Vec<String>,
}

impl GraphPath {
    /// Build from path segments, e.g. `["fluid", "inlet", "velocity"]`.
    #[must_use]
    pub fn new<I, S>(segments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            segments: segments.into_iter().map(Into::into).collect(),
        }
    }

    /// The path segments.
    #[must_use]
    pub fn segments(&self) -> &[String] {
        &self.segments
    }
}

impl fmt::Display for GraphPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.segments.join("."))
    }
}

/// Source location in an Eqiora Language file (byte offsets).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Span {
    /// Source file, workspace-relative.
    pub file: String,
    /// Start byte offset.
    pub start: u32,
    /// End byte offset (exclusive).
    pub end: u32,
}

/// Machine-applicable fix shared by agents and Studio clients.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Patch {
    /// Human-readable one-line summary of the fix.
    pub summary: String,
}

impl Patch {
    /// Create a patch suggestion with a summary.
    #[must_use]
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
        }
    }
}

/// An engineering diagnostic.
///
/// `#[non_exhaustive]` — construct through [`Diagnostic::error`] /
/// [`Diagnostic::warning`] and the `with_*` builders.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Diagnostic {
    severity: Severity,
    code: Code,
    message: String,
    details: Option<Box<Details>>,
}

#[derive(Debug, Clone, PartialEq, Default)]
struct Details {
    graph_path: Option<GraphPath>,
    source_span: Option<Span>,
    suggestion: Option<Patch>,
}

impl Diagnostic {
    /// Create an error diagnostic.
    #[must_use]
    pub fn error(code: Code, message: impl Into<String>) -> Self {
        Self::new(Severity::Error, code, message)
    }

    /// Create a warning diagnostic.
    #[must_use]
    pub fn warning(code: Code, message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, code, message)
    }

    fn new(severity: Severity, code: Code, message: impl Into<String>) -> Self {
        Self {
            severity,
            code,
            message: message.into(),
            details: None,
        }
    }

    /// Diagnostic severity.
    #[must_use]
    pub const fn severity(&self) -> Severity {
        self.severity
    }

    /// Stable append-only diagnostic code.
    #[must_use]
    pub const fn code(&self) -> Code {
        self.code
    }

    /// Engineering explanation of the cause.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Prefix explanatory context, preserving severity, code and every detail.
    #[must_use]
    pub fn with_context(mut self, context: impl AsRef<str>) -> Self {
        self.message = format!("{}: {}", context.as_ref(), self.message);
        self
    }

    /// Path to the responsible graph node, when known.
    #[must_use]
    pub fn graph_path(&self) -> Option<&GraphPath> {
        self.details
            .as_deref()
            .and_then(|details| details.graph_path.as_ref())
    }

    /// Source location, when the cause maps to source text.
    #[must_use]
    pub fn source_span(&self) -> Option<&Span> {
        self.details
            .as_deref()
            .and_then(|details| details.source_span.as_ref())
    }

    /// Machine-applicable fix, when one exists.
    #[must_use]
    pub fn suggestion(&self) -> Option<&Patch> {
        self.details
            .as_deref()
            .and_then(|details| details.suggestion.as_ref())
    }

    /// Attach the responsible graph path.
    #[must_use]
    pub fn with_graph_path(mut self, path: GraphPath) -> Self {
        self.details_mut().graph_path = Some(path);
        self
    }

    /// Attach a source span.
    #[must_use]
    pub fn with_span(mut self, span: Span) -> Self {
        self.details_mut().source_span = Some(span);
        self
    }

    /// Attach a machine-applicable suggestion.
    #[must_use]
    pub fn with_suggestion(mut self, patch: Patch) -> Self {
        self.details_mut().suggestion = Some(patch);
        self
    }

    fn details_mut(&mut self) -> &mut Details {
        self.details
            .get_or_insert_with(|| Box::new(Details::default()))
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sev = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        };
        write!(f, "{sev}[{}]: {}", self.code, self.message)?;
        if let Some(path) = self.graph_path() {
            write!(f, " (at {path})")?;
        }
        Ok(())
    }
}

impl std::error::Error for Diagnostic {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_preserves_every_severity_and_existing_detail() {
        for severity in [Severity::Error, Severity::Warning, Severity::Note] {
            let original = Diagnostic::new(severity, codes::NONFINITE_EVALUATION, "overflow")
                .with_graph_path(GraphPath::new(["model", "field"]))
                .with_span(Span {
                    file: "model.eqi".into(),
                    start: 3,
                    end: 12,
                })
                .with_suggestion(Patch::new("check the input scale"));
            let contextual = original.clone().with_context("mapped JVP point 1, seed 2");
            assert_eq!(contextual.message(), "mapped JVP point 1, seed 2: overflow");
            assert_eq!(contextual.severity(), original.severity());
            assert_eq!(contextual.code(), original.code());
            assert_eq!(contextual.graph_path(), original.graph_path());
            assert_eq!(contextual.source_span(), original.source_span());
            assert_eq!(contextual.suggestion(), original.suggestion());
            assert_eq!(original.message(), "overflow");
        }
    }
}
