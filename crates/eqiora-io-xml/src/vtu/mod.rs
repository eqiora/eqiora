//! Pure, bounded VTK XML UnstructuredGrid import.
//!
//! This L3 adapter has no filesystem or network authority. It accepts one
//! deliberately narrow serial ASCII `.vtu` subset, retains exact structural
//! selectors and normalized arrays for an L4 provenance workflow, and passes
//! accepted topology, geometry, and fields through Eqiora's shared invariant
//! constructors. Artifact identity and persisted replay remain L4 concerns.

mod parse;
mod plan;
mod resolve;

pub use plan::{
    VTU_ADAPTER_ID, VTU_ADAPTER_VERSION, VtuCellKind, VtuImportLimits, VtuImportPlan, VtuSelection,
};
pub use resolve::{VtuImport, VtuImportedField};
