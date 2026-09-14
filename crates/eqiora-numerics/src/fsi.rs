//! Fixed-reference fluid-structure interaction realizations.

pub use crate::canonical_fsi::{
    AcceptedDistributedFixedReferenceFsiStep2d, FinalizedResolvedFixedReferenceFsiStep2d,
    FixedReferenceFsiCartesianModel2d, FixedReferenceFsiScaleProfile2d, FsiInterface,
    FsiInterfaceSide, PreparedDistributedFixedReferenceFsiStep2d,
    ResolvedFixedReferenceFsiSolution2d, finalize_resolved_fixed_reference_fsi_step_2d,
    finalize_resolved_fixed_reference_fsi_step_2d_with_assembly, fixed_reference_fsi_cuda_plan_2d,
    fixed_reference_fsi_distributed_cuda_plan_2d, fixed_reference_fsi_plan_2d,
    fixed_reference_fsi_requirements_2d, fixed_reference_fsi_requirements_2d_for_layout,
    lower_fixed_reference_fsi_cartesian_2d,
};
pub use crate::linear_elasticity::IsotropicElasticityMaterial;
pub use crate::simplicial_fsi::{
    FinalizedFixedReferenceFsiStep, FixedReferenceFsiBoundary, FixedReferenceFsiEnergyBalance,
    FixedReferenceFsiInterfaceAction, FixedReferenceFsiLoad, FixedReferenceFsiMaterial,
    FixedReferenceFsiPartition, FixedReferenceFsiScale, FixedReferenceFsiSolution,
    FixedReferenceFsiState, FixedReferenceFsiStepConfig,
};
