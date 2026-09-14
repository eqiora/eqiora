//! Method-neutral lowering of exact incompressible Newtonian fluid subsets.

mod additive_boundary;
mod api;
mod block;
mod boundary;
mod dissipation_profile;
mod expression;
mod geometry_realization;
mod inertial;
mod mixed_certificate;
pub use mixed_certificate::check_authored_mixed_formulation;
mod navier_stokes;
mod navier_stokes_fvm_acceptance;
mod navier_stokes_fvm_realization;
mod navier_stokes_integral_formulation;
mod navier_stokes_mixed_formulation;
mod navier_stokes_realization;
mod physical;
mod prescribed_velocity;
mod realization;
mod recognize;
mod support;
mod transient_geometry_realization;

pub use api::{SteadyIncompressibleStokesCartesianModel2d, SteadyStokesNormalPressure2d};
pub(crate) use boundary::LoweredStokesBoundary;
pub(crate) use geometry_realization::scaling::ResolvedIncompressibleScaling2d;
pub use geometry_realization::scaling::{
    IncompressibleScalingReceipt2d, IncompressibleScalingRequest2d, ScalingAuthority2d,
    ScalingComponent2d, ScalingComponentRecord2d, ScalingMode2d, ScalingRule2d,
};
pub(crate) use geometry_realization::scaling::{
    resolve_complete_manual_incompressible_scaling_2d, resolve_fixed_reference_fsi_scaling_2d,
};
pub use geometry_realization::{
    SteadyStokesGeometryBinding2d, solve_resolved_steady_stokes_geometry_mini_2d,
};
pub use inertial::InertialIncompressibleNewtonianCartesianModel2d;
pub(crate) use inertial::lower_inertial_incompressible_newtonian_subdomain_2d_with_boundaries;
pub(crate) use navier_stokes::lower_transient_incompressible_navier_stokes_subdomain;
pub(crate) use navier_stokes::recognize_transient_incompressible_navier_stokes_geometry_mathematics;
pub use navier_stokes::{
    TransientIncompressibleNavierStokesCartesianModel,
    TransientIncompressibleNavierStokesCartesianModel2d,
    TransientIncompressibleNavierStokesCartesianModel3d,
    lower_transient_incompressible_navier_stokes_cartesian_2d,
    lower_transient_incompressible_navier_stokes_cartesian_3d,
};
pub use navier_stokes_fvm_realization::{
    CellCenteredNavierStokesInitialState2d, CellCenteredNavierStokesStepEvidence2d,
    ResolvedCellCenteredNavierStokesState2d, ResolvedCellCenteredNavierStokesTrajectory2d,
    advance_resolved_transient_navier_stokes_cell_centered_2d,
    transient_navier_stokes_cell_centered_plan_2d,
    transient_navier_stokes_cell_centered_requirements_2d,
};
pub(crate) use navier_stokes_fvm_realization::{
    PreparedResolvedTransientCellCenteredRun2d,
    prepare_resolved_transient_navier_stokes_cell_centered_run_2d,
    transient_cell_centered_algebraic_structure,
};
pub(crate) use navier_stokes_integral_formulation::integral_conservative_correspondence;
pub(crate) use navier_stokes_realization::{
    PreparedResolvedTransientMiniRun2d, prepare_resolved_transient_navier_stokes_mini_run_2d,
    require_complete_zero_trace, transient_mini_algebraic_structure,
};
pub use navier_stokes_realization::{
    ResolvedTransientNavierStokesState2d, ResolvedTransientNavierStokesTrajectory2d,
    TransientNavierStokesInitialState2d, TransientNavierStokesRun2d,
    advance_resolved_transient_navier_stokes_mini_2d,
    advance_resolved_transient_navier_stokes_mini_2d_with_assembly,
    transient_navier_stokes_fieldwise_requirements_2d, transient_navier_stokes_mini_plan_2d,
};
pub use physical::{
    FinalizedSteadyStokesMini2dProblem, SteadyStokesMiniSolution2d, SteadyStokesPressureReference2d,
};
pub use realization::{
    IncompressibleFlowScaleProfile2d, SteadyStokesScaleProfile2d,
    finalize_resolved_steady_stokes_mini_2d, solve_resolved_steady_stokes_mini_2d,
    steady_stokes_fieldwise_requirements_2d, steady_stokes_mini_plan_2d,
};
pub use recognize::lower_steady_incompressible_stokes_cartesian_2d;
pub(crate) use recognize::recognize_steady_incompressible_stokes_geometry_mathematics;
pub(crate) use transient_geometry_realization::{
    PreparedResolvedTransientGeometryMiniRun2d, TransientNavierStokesGeometryBinding2d,
    prepare_resolved_transient_navier_stokes_geometry_mini_run_2d,
};

#[cfg(test)]
mod tests;
