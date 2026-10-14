//! **eqiora-numerics** — numerical realizations and verified solver kernels.
//!
//! This crate owns approximation choices, not model meaning. Its public
//! surface is organized by scientific responsibility. Implementation modules
//! and lowering bridges remain private; each public item has one owner path.

mod additive_residual;
mod affine_fem;
mod assembled_linearization;
mod canonical;
mod canonical_boundary;
mod canonical_elasticity;
#[cfg(test)]
mod canonical_euler;
mod canonical_fsi;
mod canonical_stokes;
mod canonical_transport;
mod cartesian_elasticity;
mod cartesian_elliptic;
mod cartesian_fvm_geometry;
mod cartesian_incompressible;
mod cartesian_periodic_3d;
mod cartesian_transport;
mod common_ode;
mod common_result;
mod common_trajectory;
#[cfg(test)]
mod conservative_system;
mod constrained_dofs;
mod continuum_kinematics;
mod diffusion;
mod discrete_block;
mod discrete_space;
mod elliptic;
mod finalized_spatial;
mod form_compiler;
pub use form_compiler::check_authored_spatial_formulation;
pub use form_compiler::region::balance::{SpatialBalanceRole, SpatialBalanceTerm};
pub use region_assembly::mapping::{SpatialBoundarySource, SpatialBoundaryWork};
mod interleaved_dofs;
mod jacobian_audit;
mod linearized_output;
mod nonlinear_execution;
mod nullspace;
mod numerical_admission;
mod region_assembly;
pub use canonical_stokes::{
    IncompressibleScalingReceipt2d, IncompressibleScalingRequest2d, ScalingAuthority2d,
    ScalingComponent2d, ScalingComponentRecord2d, ScalingMode2d, ScalingRule2d,
};
pub use common_ode::{
    CommonOdePlan, CommonOdePolicy, CommonOdeRunRequest, CommonOdeState, CommonTimeTolerance,
};
pub use common_result::{CommonObservableStateTangent, CommonResult};
pub use common_trajectory::{
    CommonTrajectory, FiniteSpectrum, TimeFunctionalQuadrature, UniformDft,
};
pub use numerical_admission::{
    AuthenticatedCommonMesh, CommonAcceptedSpatialStep, CommonAlgebraicPlan, CommonAlgebraicState,
    CommonBackwardEuler, CommonEigenPlan, CommonEigenRequest, CommonFormulationDescription,
    CommonInitialField, CommonInitialValues, CommonLinearRequest, CommonMethodRequest,
    CommonPressureGauge2d, CommonScalarDifferentiationPoint, CommonScalingRequest,
    CommonScopedSpatialPolicy, CommonSolvePolicy, CommonSpatialPlan, CommonSpatialPolicy,
    CommonSpatialStepBalance, CommonState, CommonSteadyStokesPlan, CommonTransientFlowPlan,
    CommonTransientRunRequest, FormulationKind, FormulationSelectionMode, ResolvedCommonPlan,
};
mod factor_measure;
pub mod finite_constraints;
mod linear_elasticity;
mod operator;
mod physical_network;
mod poisson;
mod prepared_execution;
pub(crate) mod scalar_conservation;
mod simplicial_ale_fsi;
mod simplicial_ale_remesh;
mod simplicial_boundary;
mod simplicial_elasticity;
mod simplicial_elliptic;
mod simplicial_fsi;
#[cfg(test)]
mod simplicial_mini_transient;
mod simplicial_motion;
mod simplicial_navier_stokes;
#[cfg(test)]
mod simplicial_solid_element;
mod simplicial_stokes;
mod spatial_design;
mod spatial_expression;
mod step_count;

/// Moving-domain ALE and remeshing realizations.
pub mod ale;
/// Numerical contracts shared by more than one scientific family.
pub mod common;
/// Incompressible-flow numerical realizations.
pub mod fluid;
/// Fixed-reference fluid-structure interaction realizations.
pub mod fsi;
/// Scalar elliptic, transport, diffusion, and affine-network realizations.
pub mod scalar;
/// Solid-mechanics numerical realizations.
pub mod solid;
