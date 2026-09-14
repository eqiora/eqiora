//! Fixed-reference monolithic fluid--structure interaction on affine simplices.
//!
//! This module owns one deliberately narrow numerical realization. An exact
//! conforming cell partition is lowered to one backward-Euler saddle system:
//! fluid MINI velocity/P1 pressure and solid P1 velocity share mesh-vertex
//! velocity unknowns, while the solid displacement update is eliminated as
//! `d_next = d_previous + dt * v_next`. The shared trace is therefore a
//! quotient in the algebraic layout rather than a penalty or copied array.
//!
//! The v1 load contract is explicitly zero. This keeps the first reference
//! evidence focused on inertial exchange, incompressibility, elastic storage,
//! viscous loss, and interface cancellation. Body and boundary loads can be
//! added later as typed laws without changing the state or operator contracts.

mod acceptance;
mod api;
mod contract;
pub(crate) mod element;
pub(crate) mod layout;
pub(crate) mod partition;
mod solve;
mod state;

#[cfg(test)]
pub(crate) mod test_model;
#[cfg(test)]
mod tests;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

pub use api::{
    FixedReferenceFsiEnergyBalance, FixedReferenceFsiInterfaceAction, FixedReferenceFsiSolution,
};
pub(crate) use contract::validate_problem;
pub use contract::{
    FixedReferenceFsiBoundary, FixedReferenceFsiLoad, FixedReferenceFsiMaterial,
    FixedReferenceFsiScale, FixedReferenceFsiState, FixedReferenceFsiStepConfig,
};
pub use partition::FixedReferenceFsiPartition;
pub use solve::FinalizedFixedReferenceFsiStep;
#[cfg(test)]
pub(crate) use solve::solve_fixed_reference_fsi_step_3d;
pub(crate) use solve::{FixedReferenceFsiAssemblyTargetRoles, PreparedFixedReferenceFsiAssembly};

const fn p1_count<const D: usize>() -> usize {
    D + 1
}

const fn mini_count<const D: usize>() -> usize {
    p1_count::<D>() + 1
}

const fn fluid_local_size<const D: usize>() -> usize {
    mini_count::<D>() * D + p1_count::<D>()
}

const fn required_quadrature_exactness<const D: usize>() -> usize {
    2 * (D + 1)
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}
