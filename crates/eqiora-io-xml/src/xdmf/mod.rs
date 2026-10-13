//! Pure, bounded XDMF 3 metadata planning, rendering, and caller-owned replay.
//!
//! This L3 adapter never opens a path or URL. Untrusted XML is reduced to an
//! immutable list of typed HDF array requests; a caller with explicit I/O
//! authority supplies complete source bytes and values in a second phase.
//! Accepted values are reconstructed through Eqiora's shared mesh and field
//! invariants. The sibling export path renders a complete Temporal Collection
//! from already validated typed frames without opening its HDF display
//! locator. Artifact and provenance composition remain at L4.

mod export;
mod parse;
mod plan;
mod resolve;

pub use export::{
    XdmfTemporalExportLimits, XdmfTemporalExportPlan, XdmfTemporalField, XdmfTemporalFrame,
};

pub use plan::{
    XDMF_ADAPTER_ID, XDMF_ADAPTER_VERSION, XdmfArrayRequest, XdmfArrayResponse, XdmfArrayRole,
    XdmfArrayValues, XdmfImportLimits, XdmfImportPlan, XdmfScalarType, XdmfSelection,
};
pub use resolve::{XdmfImport, XdmfImportedField};
