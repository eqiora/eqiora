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
#[allow(dead_code)]
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
mod constrained_dofs;
// Private face-action seam consumed by the forthcoming common Run integration (#686).
#[allow(dead_code)]
mod conservative_system;
mod continuum_kinematics;
mod diffusion;
mod discrete_block;
mod discrete_space;
mod elliptic;
mod finalized_spatial;
mod form_compiler;
mod interleaved_dofs;
mod jacobian_audit;
mod linearized_output;
mod numerical_admission;
mod region_assembly;
pub use canonical_stokes::{
    IncompressibleScalingReceipt2d, IncompressibleScalingRequest2d, ScalingAuthority2d,
    ScalingComponent2d, ScalingComponentRecord2d, ScalingMode2d, ScalingRule2d,
    check_authored_mixed_formulation,
};
pub use common_ode::{
    CommonOdePlan, CommonOdeRunRequest, CommonOdeState, CommonTsitouras45, CommonTsitourasTolerance,
};
pub use common_result::{CommonObservableStateTangent, CommonResult};
pub use common_trajectory::{CommonTrajectory, TimeFunctionalQuadrature};
pub use numerical_admission::{
    AuthenticatedCommonMesh, CommonAlgebraicPlan, CommonAlgebraicState, CommonBackwardEuler,
    CommonElasticityPlan, CommonFormulationDescription, CommonFsiConnectionInventory,
    CommonFsiDomainInventory, CommonFsiPlan, CommonFsiRunRequest, CommonInitialField,
    CommonInitialValues, CommonLinearRequest, CommonMethodRequest, CommonPressureGauge2d,
    CommonScalarPlan, CommonScopedSpatialPolicy, CommonSolvePolicy, CommonSpatialPolicy,
    CommonState, CommonSteadyStokesPlan, CommonTransientFlowPlan, CommonTransientRunRequest,
    FormulationKind, FormulationSelectionMode, ResolvedCommonPlan, resolve_common_ode_plan,
    resolve_common_plan,
};
pub mod finite_constraints;
mod linear_elasticity;
mod operator;
mod physical_network;
mod poisson;
mod prepared_execution;
// Private provider seam for the #698 realization consumer; remove this scoped
// allowance when that sibling lands and makes the provider live in non-test builds.
#[allow(dead_code)]
pub(crate) mod scalar_conservation;
mod simplicial_ale_fsi;
mod simplicial_ale_remesh;
mod simplicial_boundary;
mod simplicial_elasticity;
mod simplicial_elliptic;
mod simplicial_fsi;
mod simplicial_mini_transient;
mod simplicial_motion;
mod simplicial_navier_stokes;
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
