//! Scalar elliptic, transport, diffusion, and affine-network realizations.

pub use crate::canonical::{
    DefaultScalarElliptic1dConfig, FinalizedScalarEllipticParameterPoint,
    ResolvedScalarEllipticCartesianSolution, ResolvedScalarEllipticSolution1d,
    ScalarEllipticCartesianModel, ScalarEllipticModel1d, compare_canonical_scalar_elliptic_1d,
    finalize_lowered_scalar_elliptic_cartesian_with_assembly,
    finalize_resolved_scalar_elliptic_cartesian, finalize_scalar_elliptic_parameter_point,
    lower_scalar_elliptic_1d, lower_scalar_elliptic_cartesian,
    solve_and_linearize_resolved_scalar_elliptic_cartesian, solve_default_scalar_elliptic_1d,
    solve_resolved_scalar_elliptic_1d, solve_resolved_scalar_elliptic_1d_with_assembly,
    solve_resolved_scalar_elliptic_cartesian,
    solve_resolved_scalar_elliptic_cartesian_with_assembly,
    solve_resolved_scalar_elliptic_simplicial,
    solve_resolved_scalar_elliptic_simplicial_with_assembly,
};
pub use crate::canonical_transport::{
    ScalarTransportCartesianBoundary, ScalarTransportCartesianModel2d,
    lower_scalar_transport_cartesian_2d,
};
pub use crate::cartesian_elliptic::{
    linearize_scalar_elliptic_cartesian_fem, lower_cartesian_q1_diffusion_local_action,
    solve_scalar_elliptic_cartesian_fem_with_assembly,
    solve_scalar_elliptic_cartesian_fvm_with_assembly,
};
pub use crate::cartesian_transport::{
    FinalizedScalarTransportFvmStep2d, ScalarTransportBoundaryRole, ScalarTransportCellState2d,
    ScalarTransportFvmStep2d, ScalarTransportFvmStepEvidence2d,
    finalize_resolved_scalar_transport_fvm_step_2d,
    finalize_resolved_scalar_transport_fvm_step_2d_with_assembly,
    initialize_resolved_scalar_transport_fvm_2d, solve_resolved_scalar_transport_fvm_step_2d,
    solve_resolved_scalar_transport_fvm_step_2d_with_assembly,
};
pub use crate::diffusion::Diffusion1d;
pub use crate::elliptic::{
    ScalarBoundaryCondition1d, ScalarBoundaryPair1d, ScalarEllipticSolution1d,
    solve_scalar_elliptic_linear_fem, solve_scalar_elliptic_linear_fem_with_assembly,
};
pub use crate::finalized_spatial::FinalizedScalarEllipticCartesianProblem;
pub(crate) use crate::linearized_output::CartesianScalarFieldLinearization;
pub use crate::physical_network::{
    ScalarPhysicalAffineProblem, ScalarPhysicalAffineSolution, lower_scalar_physical_affine,
    solve_scalar_physical_affine,
};
pub use crate::poisson::{
    DirichletBoundary1d, PiecewiseLinearField1d, PoissonComparisonRow, PoissonSolution1d,
    ScalarEllipticComparisonRow1d, ScalarEllipticFvmSolution1d, compare_sine_poisson_1d,
    solve_scalar_elliptic_cell_fvm,
};
pub use crate::simplicial_elliptic::{
    ScalarEllipticSimplicialFemSolution, linearize_scalar_elliptic_simplicial_compliance,
    linearize_scalar_elliptic_simplicial_fem, solve_scalar_elliptic_simplicial_fem,
    solve_scalar_elliptic_simplicial_fem_with_assembly,
};
