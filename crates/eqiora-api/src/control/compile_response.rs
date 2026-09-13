use serde::{Deserialize, Serialize};

use super::compile::{COMPILE_COMMAND_V1, CompileRequestV2, validate_request_id};
use super::{
    CONTROL_PROTOCOL_V2, ControlDiagnosticV2, MAX_COMPILE_RESPONSE_BYTES_V2,
    MAX_CONTROL_DIAGNOSTICS_V2,
};
use crate::ModelDocument;

const MODEL_SCHEMA: &str = "eqiora.model-envelope/v26";
const MODEL_TRANSACTION_SCHEMA: &str = "eqiora.model-transaction-envelope/v26";

/// Exact canonical Model identity returned by successful compile/check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompileModelDescriptorV2 {
    schema: String,
    transaction_schema: String,
    digest: String,
    model_id: String,
    semantic_revision: u64,
}

impl CompileModelDescriptorV2 {
    /// Immutable current Model artifact schema.
    #[must_use]
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Immutable current Model transaction schema.
    #[must_use]
    pub fn transaction_schema(&self) -> &str {
        &self.transaction_schema
    }

    /// Domain-separated canonical content digest.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// Typed Semantic Model identity.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Semantic Graph Federation revision serialized by the Model artifact.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.semantic_revision
    }
}

/// Terminal compile/check outcome. Scientific values never enter this DTO.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "kebab-case", deny_unknown_fields)]
pub enum CompileOutcomeV2 {
    /// One canonical Model was compiled by the current operation and described on this wire.
    Accepted {
        /// Exact typed Model identity.
        model: CompileModelDescriptorV2,
    },
    /// No Model document is admitted.
    Rejected {
        /// One or more structured Eqiora diagnostics.
        diagnostics: Vec<ControlDiagnosticV2>,
    },
}

/// Closed v2 response echoing the admitted request identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompileResponseV2 {
    protocol: String,
    command: String,
    request_id: String,
    outcome: CompileOutcomeV2,
}

impl CompileResponseV2 {
    /// Exact protocol generation.
    #[must_use]
    pub fn protocol(&self) -> &str {
        &self.protocol
    }

    /// Exact command identity.
    #[must_use]
    pub fn command(&self) -> &str {
        &self.command
    }

    /// Request identity echoed without rewriting.
    #[must_use]
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Terminal typed outcome.
    #[must_use]
    pub const fn outcome(&self) -> &CompileOutcomeV2 {
        &self.outcome
    }

    /// Deterministic compact JSON in frozen member order.
    ///
    /// # Errors
    /// Returns a diagnostic if serialization unexpectedly fails or the encoded
    /// response exceeds the v2 transport bound.
    pub fn canonical_json(&self) -> Result<Vec<u8>, ControlDiagnosticV2> {
        let bytes = serde_json::to_vec(self).map_err(|error| {
            ControlDiagnosticV2::invalid_request(format!(
                "cannot encode compile/check v2 response: {error}"
            ))
        })?;
        if bytes.len() > MAX_COMPILE_RESPONSE_BYTES_V2 {
            return Err(ControlDiagnosticV2::diagnostics_overflow());
        }
        Ok(bytes)
    }

    /// Decode and validate one exact compile/check v2 response.
    ///
    /// # Errors
    /// Returns a structured diagnostic for oversized, malformed, unknown, or
    /// contradictory response data.
    pub fn from_json(bytes: &[u8]) -> Result<Self, ControlDiagnosticV2> {
        if bytes.len() > MAX_COMPILE_RESPONSE_BYTES_V2 {
            return Err(ControlDiagnosticV2::invalid_request(format!(
                "compile/check response exceeds {MAX_COMPILE_RESPONSE_BYTES_V2} encoded bytes"
            )));
        }
        let response: Self = serde_json::from_slice(bytes).map_err(|error| {
            ControlDiagnosticV2::invalid_request(format!(
                "invalid compile/check v2 response JSON: {error}"
            ))
        })?;
        response.validate()?;
        Ok(response)
    }

    fn accepted(request: &CompileRequestV2, model: CompileModelDescriptorV2) -> Self {
        Self {
            protocol: CONTROL_PROTOCOL_V2.to_owned(),
            command: COMPILE_COMMAND_V1.to_owned(),
            request_id: request.request_id().to_owned(),
            outcome: CompileOutcomeV2::Accepted { model },
        }
    }

    fn rejected(request: &CompileRequestV2, diagnostics: Vec<ControlDiagnosticV2>) -> Self {
        debug_assert!(!diagnostics.is_empty());
        Self {
            protocol: CONTROL_PROTOCOL_V2.to_owned(),
            command: COMPILE_COMMAND_V1.to_owned(),
            request_id: request.request_id().to_owned(),
            outcome: CompileOutcomeV2::Rejected { diagnostics },
        }
    }

    fn overflow(request: &CompileRequestV2) -> Self {
        Self::rejected(request, vec![ControlDiagnosticV2::diagnostics_overflow()])
    }

    fn validate(&self) -> Result<(), ControlDiagnosticV2> {
        if self.protocol != CONTROL_PROTOCOL_V2 || self.command != COMPILE_COMMAND_V1 {
            return Err(ControlDiagnosticV2::unsupported(
                "unsupported compile/check response protocol or command",
            ));
        }
        validate_request_id(&self.request_id)?;
        match &self.outcome {
            CompileOutcomeV2::Accepted { model } => validate_model(model),
            CompileOutcomeV2::Rejected { diagnostics } => {
                if diagnostics.is_empty() || diagnostics.len() > MAX_CONTROL_DIAGNOSTICS_V2 {
                    return Err(ControlDiagnosticV2::invalid_request(
                        "rejected compile/check response has an invalid diagnostic count",
                    ));
                }
                for diagnostic in diagnostics {
                    diagnostic.validate()?;
                }
                Ok(())
            }
        }
    }
}

/// Control-adapter result: wire response plus the optional Model from that invocation.
#[derive(Debug, Clone)]
pub struct CompileControlExecutionV2 {
    response: CompileResponseV2,
    document: Option<ModelDocument>,
}

impl CompileControlExecutionV2 {
    /// Transport-neutral command response.
    #[must_use]
    pub const fn response(&self) -> &CompileResponseV2 {
        &self.response
    }

    /// Accepted immutable Model, absent for every rejected response.
    #[must_use]
    pub const fn document(&self) -> Option<&ModelDocument> {
        self.document.as_ref()
    }

    /// Transfer the response and optional opaque Model to an adapter.
    #[must_use]
    pub fn into_parts(self) -> (CompileResponseV2, Option<ModelDocument>) {
        (self.response, self.document)
    }
}

/// Adapt one fully admitted request to exactly one current-Model compile operation.
///
/// The response descriptor and optional document are projected from that same
/// invocation; protocol and request identity do not enter compilation meaning.
#[must_use]
pub fn execute_compile_v2(request: &CompileRequestV2) -> CompileControlExecutionV2 {
    let document = match ModelDocument::compile(request.filename(), request.source()) {
        Ok(document) => document,
        Err(diagnostics) => {
            if diagnostics.len() > MAX_CONTROL_DIAGNOSTICS_V2 {
                return overflow_execution(request);
            }
            let diagnostics = diagnostics
                .into_iter()
                .map(ControlDiagnosticV2::from_kernel)
                .collect::<Result<Vec<_>, _>>();
            let Ok(diagnostics) = diagnostics else {
                return overflow_execution(request);
            };
            let response = CompileResponseV2::rejected(request, diagnostics);
            if response.canonical_json().is_err() {
                return overflow_execution(request);
            }
            return CompileControlExecutionV2 {
                response,
                document: None,
            };
        }
    };

    let reference = match document.artifact_reference() {
        Ok(reference) => reference,
        Err(diagnostic) => {
            let Ok(projected) = ControlDiagnosticV2::from_kernel(diagnostic) else {
                return overflow_execution(request);
            };
            return CompileControlExecutionV2 {
                response: CompileResponseV2::rejected(request, vec![projected]),
                document: None,
            };
        }
    };
    let model = CompileModelDescriptorV2 {
        schema: MODEL_SCHEMA.to_owned(),
        transaction_schema: MODEL_TRANSACTION_SCHEMA.to_owned(),
        digest: reference.artifact().to_string(),
        model_id: reference.model().to_string(),
        semantic_revision: reference.semantic_revision().get(),
    };
    CompileControlExecutionV2 {
        response: CompileResponseV2::accepted(request, model),
        document: Some(document),
    }
}

fn overflow_execution(request: &CompileRequestV2) -> CompileControlExecutionV2 {
    CompileControlExecutionV2 {
        response: CompileResponseV2::overflow(request),
        document: None,
    }
}

fn validate_model(model: &CompileModelDescriptorV2) -> Result<(), ControlDiagnosticV2> {
    let valid_digest = model.digest.len() == 64
        && model
            .digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'));
    if model.schema != MODEL_SCHEMA
        || model.transaction_schema != MODEL_TRANSACTION_SCHEMA
        || !valid_digest
        || model.model_id.is_empty()
        || model.model_id.chars().count() > 128
    {
        return Err(ControlDiagnosticV2::invalid_request(
            "accepted compile/check response has invalid Model identity facts",
        ));
    }
    Ok(())
}
