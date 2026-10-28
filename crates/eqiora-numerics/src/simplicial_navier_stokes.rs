//! Fixed-domain transient incompressible Navier--Stokes on the 2D MINI pair.
//!
//! The Semantic Model owns the conservative momentum relation. This numerical
//! realization selects backward Euler and the skew-symmetric convective form,
//! whose discrete self-action is exactly zero even though MINI velocity is
//! only weakly divergence free. Mesh motion, stabilization, turbulence, and
//! backend-specific algebra remain outside this module.

mod acceptance;
mod api;
mod assembly;
pub(crate) mod element;
mod form;
mod layout;
mod newton;
mod verification;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

pub use api::{
    MiniNavierStokesStepPlan2d, SimplicialMiniNavierStokesState2d,
    SimplicialMiniNavierStokesStepEvidence2d, SimplicialMiniNavierStokesTrajectory2d,
};
pub(crate) use assembly::{
    PreparedStepStructure, boundary_reaction_prepared, prepare_step_structure,
};
pub use newton::advance_simplicial_mini_navier_stokes_2d;
pub(crate) use newton::{
    advance_simplicial_mini_navier_stokes_2d_with_prepared_structure,
    advance_simplicial_mini_navier_stokes_2d_with_prepared_structure_and_linear,
};

const DIMENSION: usize = 2;
const COMPONENTS: usize = 2;
const REQUIRED_CONVECTIVE_QUADRATURE_EXACTNESS: usize = 8;
const REQUIRED_CONVECTIVE_FACET_QUADRATURE_EXACTNESS: usize = 3;

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

fn norm(values: &[f64]) -> Result<f64, Diagnostic> {
    eqiora_solver::FixedOrderInnerProduct::norm(&eqiora_solver::SERIAL_LINEAR_EXECUTION, values)
        .map_err(|error| invalid(error.message()))
}
