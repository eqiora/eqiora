//! Conservative cell-centered transport realization on Cartesian meshes.

mod admission;
mod api;
mod assembly;
mod faces;
mod initial;
mod periodic;
mod reconstruction;
mod replay;

pub use api::{
    FinalizedScalarTransportFvmStep2d, ScalarTransportBoundaryRole, ScalarTransportCellState2d,
    ScalarTransportFvmStep2d, ScalarTransportFvmStepEvidence2d,
};
pub use assembly::{
    finalize_resolved_scalar_transport_fvm_step_2d,
    finalize_resolved_scalar_transport_fvm_step_2d_with_assembly,
    initialize_resolved_scalar_transport_fvm_2d, solve_resolved_scalar_transport_fvm_step_2d,
};
