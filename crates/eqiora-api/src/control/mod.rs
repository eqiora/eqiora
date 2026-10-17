//! Versioned, client-neutral control-plane contracts.
//!
//! [`crate::ModelDocument::compile`] owns transport-neutral compilation. This
//! module owns the bounded command, admission, response, and diagnostic policy
//! that adapts that operation for control-v2 clients. It deliberately excludes
//! scientific arrays, meshes, Fields, and trajectories; those values belong to
//! the data plane.

mod compile;
mod compile_response;
mod diagnostic;
mod schema;
#[cfg(test)]
mod tests;

use crate::DiagnosticRecord;
pub use compile::{COMPILE_COMMAND_V1, CompileRequestV2};
pub use compile_response::{
    CompileControlExecutionV2, CompileModelDescriptorV2, CompileOutcomeV2, CompileResponseV2,
    execute_compile_v2,
};
pub use schema::COMPILE_V2_SCHEMA_JSON;

/// Exact protocol identity for the current bounded control-plane slice.
pub const CONTROL_PROTOCOL_V2: &str = "eqiora.control/v2";

/// Largest admitted encoded compile request or response.
pub const MAX_COMPILE_REQUEST_BYTES_V2: usize = 8 * 1_024 * 1_024 + 16 * 1_024;
/// Largest admitted encoded compile response.
pub const MAX_COMPILE_RESPONSE_BYTES_V2: usize = MAX_COMPILE_REQUEST_BYTES_V2;
/// Largest admitted Eqiora Language source in one compile request.
pub const MAX_COMPILE_SOURCE_BYTES_V2: usize = 8 * 1_024 * 1_024;
/// Largest admitted UTF-8 filename in one compile request.
pub const MAX_COMPILE_FILENAME_BYTES_V2: usize = 4_096;
/// Largest admitted request identifier.
pub const MAX_CONTROL_REQUEST_ID_BYTES_V2: usize = 128;
/// Largest prelude protocol or command identity.
pub const MAX_CONTROL_DISPATCH_IDENTITY_BYTES_V2: usize = 128;
/// Largest admitted diagnostic list.
pub const MAX_CONTROL_DIAGNOSTICS_V2: usize = 1_024;
