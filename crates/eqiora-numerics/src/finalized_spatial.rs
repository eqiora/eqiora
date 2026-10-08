mod core;
mod elasticity;
mod scalar;
mod stokes;

pub(crate) use core::{FinalizedLinearCore, ResidualScalar};
pub use elasticity::{
    FinalizedConformingIsotropicElasticityCartesianPair2dProblem,
    FinalizedIsotropicElasticityCartesian2dProblem,
};
pub use scalar::FinalizedScalarEllipticCartesianProblem;
pub use stokes::FinalizedSimplicialMiniStokes2dProblem;
