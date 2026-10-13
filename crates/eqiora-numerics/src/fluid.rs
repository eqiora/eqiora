//! Incompressible-flow numerical realizations.

pub use crate::canonical_stokes::{
    CellCenteredNavierStokesInitialState2d, CellCenteredNavierStokesStepEvidence2d,
    FinalizedSteadyStokesMini2dProblem, IncompressibleFlowScaleProfile2d,
    ResolvedCellCenteredNavierStokesState2d, ResolvedCellCenteredNavierStokesTrajectory2d,
    ResolvedTransientNavierStokesState2d, ResolvedTransientNavierStokesTrajectory2d,
    SteadyIncompressibleStokesCartesianModel2d, SteadyStokesGeometryBinding2d,
    SteadyStokesMiniSolution2d, SteadyStokesNormalPressure2d, SteadyStokesPressureReference2d,
    TransientIncompressibleNavierStokesCartesianModel2d,
    TransientIncompressibleNavierStokesCartesianModel3d, TransientNavierStokesInitialState2d,
    TransientNavierStokesRun2d, advance_resolved_transient_navier_stokes_cell_centered_2d,
    advance_resolved_transient_navier_stokes_mini_2d,
    advance_resolved_transient_navier_stokes_mini_2d_with_assembly,
    finalize_resolved_steady_stokes_mini_2d, lower_steady_incompressible_stokes_cartesian_2d,
    lower_transient_incompressible_navier_stokes_cartesian_2d,
    lower_transient_incompressible_navier_stokes_cartesian_3d,
    solve_resolved_steady_stokes_geometry_mini_2d, solve_resolved_steady_stokes_mini_2d,
    steady_stokes_fieldwise_requirements_2d, steady_stokes_mini_plan_2d,
    transient_navier_stokes_cell_centered_plan_2d,
    transient_navier_stokes_cell_centered_requirements_2d,
    transient_navier_stokes_fieldwise_requirements_2d, transient_navier_stokes_mini_plan_2d,
};
pub use crate::cartesian_incompressible::{
    CellCenteredPressureField2d, CellCenteredVelocityField2d,
};
pub use crate::simplicial_navier_stokes::{
    MiniNavierStokesStepPlan2d, SimplicialMiniNavierStokesState2d,
    SimplicialMiniNavierStokesStepEvidence2d, SimplicialMiniNavierStokesTrajectory2d,
    advance_simplicial_mini_navier_stokes_2d,
    advance_simplicial_mini_navier_stokes_2d_with_assembly,
};
pub use crate::simplicial_stokes::{
    SimplicialMiniStokesBoundary2d, SimplicialMiniStokesBoundaryCondition2d,
    SimplicialMiniStokesBoundaryFacet2d, SimplicialMiniStokesErrorNorms2d,
    SimplicialMiniStokesPressureReference2d, SimplicialMiniStokesSolution2d,
    SimplicialMiniVelocityField2d, finalize_simplicial_mini_stokes_2d,
    solve_simplicial_mini_stokes_2d,
};
