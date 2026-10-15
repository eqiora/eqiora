//! Pure, bounded VTU and XDMF import and XDMF temporal rendering.
//!
//! A private XML reader owns syntax, structural paths and resource accounting.
//! Format adapters own their admissible elements and typed payloads. Neither
//! parsing nor rendering grants filesystem or network authority; artifact
//! identity and persisted replay belong to the caller's L4 workflow.

mod allocation;
mod vtu;
mod xdmf;
mod xml;

pub use vtu::{
    VTU_ADAPTER_ID, VTU_ADAPTER_VERSION, VtuCellKind, VtuImportLimits, VtuImportPlan, VtuSelection,
};
pub use vtu::{VtuImport, VtuImportedField};

pub use xdmf::{
    XdmfTemporalExportLimits, XdmfTemporalExportPlan, XdmfTemporalField, XdmfTemporalFrame,
};

pub use xdmf::{
    XDMF_ADAPTER_ID, XDMF_ADAPTER_VERSION, XdmfArrayRequest, XdmfArrayResponse, XdmfArrayRole,
    XdmfArrayValues, XdmfImportLimits, XdmfImportPlan, XdmfScalarType, XdmfSelection,
};
pub use xdmf::{XdmfImport, XdmfImportedField};
