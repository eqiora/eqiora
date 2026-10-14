//! Moving-domain ALE and remeshing realizations.

pub use crate::canonical_fsi::{
    AcceptedResolvedAleFsiRemesh2d, AleFsiCartesianModel, AleFsiFieldIdentities,
    AleFsiInitialPhysicalState, FinalizedResolvedFixedTopologyAleFsi,
    finalize_resolved_fixed_topology_ale_fsi_2d, finalize_resolved_fixed_topology_ale_fsi_3d,
    fixed_topology_ale_fsi_requirements, lower_ale_fsi_cartesian,
    remesh_resolved_fixed_topology_ale_fsi_2d, solve_resolved_fixed_topology_ale_fsi_2d,
    solve_resolved_fixed_topology_ale_fsi_2d_with_assembly,
    solve_resolved_fixed_topology_ale_fsi_3d,
    solve_resolved_fixed_topology_ale_fsi_3d_with_assembly,
};
pub use crate::simplicial_ale_fsi::{
    AleFsiBoundary, AleFsiInterfaceAction, AleFsiState, AleFsiStepEvidence, AleFsiStepPlan,
    AleFsiTrajectory, P1HarmonicMeshMotionAction,
};
pub use crate::simplicial_ale_remesh::{
    AcceptedAleFsiRemeshProjection2d, AleFsiRemeshProjectionEvidence2d,
    project_simplicial_ale_fsi_remesh_2d,
};
pub use crate::simplicial_motion::SimplicialMeshVelocity;
