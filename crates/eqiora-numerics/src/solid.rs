//! Solid-mechanics numerical realizations.

pub use crate::canonical_elasticity::{
    ConformingElasticityInterface2d, ConformingElasticityInterfaceSide2d,
    ElasticityIntegrationMeasure, IsotropicElasticityContinuum, IsotropicElasticityReduction,
    IsotropicElastodynamicsCartesianModel,
    finalize_resolved_conforming_isotropic_elasticity_cartesian_pair_2d_with_assembly,
    finalize_resolved_isotropic_elasticity_cartesian_2d,
    finalize_resolved_isotropic_elasticity_cartesian_2d_with_assembly,
    lower_conforming_isotropic_elasticity_cartesian_pair_2d,
    lower_isotropic_elasticity_cartesian_2d, lower_isotropic_elastodynamics_cartesian_2d,
    solve_resolved_isotropic_elasticity_cartesian_2d,
    solve_resolved_isotropic_elasticity_cartesian_2d_with_assembly,
};
pub use crate::cartesian_elasticity::{
    CartesianLinearElasticity2dSolution, CartesianQ1VectorField2d,
    ConformingCartesianInterfaceMap2d, ConformingElasticityInterfaceAction2d,
    lower_cartesian_q1_linear_elasticity_local_action_2d, solve_cartesian_q1_linear_elasticity_2d,
    solve_cartesian_q1_linear_elasticity_2d_with_assembly,
};
pub use crate::finalized_spatial::FinalizedIsotropicElasticityCartesian2dProblem;
