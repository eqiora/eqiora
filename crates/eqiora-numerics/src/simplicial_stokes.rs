//! Stable two-dimensional MINI finite elements for steady incompressible Stokes flow.
//!
//! This module owns one numerical realization: continuous P1 velocity enriched
//! by one cell bubble per component and continuous P1 pressure. A complete
//! essential trace adds one independent global mean-pressure constraint;
//! admitted mixed essential/traction data instead fixes pressure through the
//! boundary weak form and adds no gauge. Fluid meaning and package identity
//! remain outside this layer.

pub(crate) mod acceptance;
mod api;
pub(crate) mod boundary;
pub(crate) mod element;
pub(crate) mod facet;
mod finalized;
pub(crate) mod integrals;
pub(crate) mod layout;
mod solve;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

pub(crate) const DIMENSION: usize = 2;
pub(crate) const COMPONENTS: usize = 2;
pub(crate) const P1_BASIS_COUNT: usize = 3;
pub(crate) const VELOCITY_BASIS_COUNT: usize = 4;
pub(crate) const LOCAL_VELOCITY_DOF_COUNT: usize = VELOCITY_BASIS_COUNT * COMPONENTS;
pub(crate) const LOCAL_PRESSURE_OFFSET: usize = LOCAL_VELOCITY_DOF_COUNT;
pub(crate) const CELL_LOCAL_DOF_COUNT: usize = LOCAL_PRESSURE_OFFSET + P1_BASIS_COUNT;
const CONSTRAINT_LOCAL_GAUGE: usize = P1_BASIS_COUNT;
pub(crate) const CONSTRAINT_LOCAL_DOF_COUNT: usize = CONSTRAINT_LOCAL_GAUGE + 1;
const FACET_BASIS_COUNT: usize = 2;
pub(crate) const FACET_LOCAL_DOF_COUNT: usize = FACET_BASIS_COUNT * COMPONENTS;
const REQUIRED_QUADRATURE_EXACTNESS: usize = 4;
const REQUIRED_FACET_QUADRATURE_EXACTNESS: usize = 1;
const REQUIRED_ERROR_QUADRATURE_EXACTNESS: usize = 6;

pub use api::{
    SimplicialMiniStokesErrorNorms2d, SimplicialMiniStokesPressureReference2d,
    SimplicialMiniStokesSolution2d, SimplicialMiniVelocityField2d,
};
pub use boundary::{
    SimplicialMiniStokesBoundary2d, SimplicialMiniStokesBoundaryCondition2d,
    SimplicialMiniStokesBoundaryFacet2d,
};
pub use solve::{finalize_simplicial_mini_stokes_2d, solve_simplicial_mini_stokes_2d};

pub(crate) use finalized::{FinalizedMiniStokesAssembly, FinalizedMiniStokesState};

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}
