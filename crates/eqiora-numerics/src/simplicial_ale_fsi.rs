//! Fixed-topology ALE fluid--structure interaction on affine simplices.

mod acceptance;
mod api;
mod assembly;
mod boundary_step;
mod contract;
mod element;
mod motion;
mod newton;
#[cfg(test)]
mod test_support;
mod verification;

#[allow(unused_imports)]
pub(crate) use boundary_step::{
    AleFsiBoundaryEndpointIdentity, AleFsiExteriorFacetDisposition, PreparedAleFsiBoundaryStep,
    advance_simplicial_ale_fsi_prepared_step,
};

pub use api::{AleFsiInterfaceAction, AleFsiStepEvidence, AleFsiTrajectory};
pub use contract::{AleFsiBoundary, AleFsiState, AleFsiStepPlan};
pub use motion::P1HarmonicMeshMotionAction;
pub(crate) use newton::{
    advance_simplicial_ale_fsi_2d_with_assembly, advance_simplicial_ale_fsi_3d_with_assembly,
};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}
