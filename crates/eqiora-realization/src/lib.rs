//! Pure Realization Graph v0 payloads and deterministic policy selection.
//!
//! A realization refers to a Semantic Model revision but cannot contain an
//! equation, unit, physical boundary condition, or model-time clock. It owns
//! only choices for discrete space, discretization, solver, deployment target,
//! and deployment schedule. Graph storage, wire encoding, and numerical
//! lowering are deliberately outside this crate.

mod capability;
mod coupled_fieldwise;
mod coupled_fieldwise_resolution;
mod diagnostic;
mod discretization;
mod execution;
mod fieldwise;
mod fieldwise_resolution;
mod fixed_topology_ale;
mod identity;
mod plan;
mod portable_graph;
mod remesh_transfer;
mod resolution;
mod transient_cell_centered_incompressible;
mod transient_cell_centered_transport;
mod transient_fieldwise;

pub use capability::{
    RealizationCapabilities, RealizationCapability, RealizationCapabilityContext,
    RealizationRequirements, ScheduleCapability, SpatialCapability, SpatialDimensionSupport,
    TargetCapabilities, TargetCapability, VectorLayoutKind,
};
pub use coupled_fieldwise::{
    BackwardEulerStateBinding, BackwardEulerStatePair, BackwardEulerStep, ConformingTraceQuotient,
    CoupledFieldwiseRealizationPlan, CoupledFieldwiseSpatialDiscretization,
    DomainFieldDiscretization, DomainFieldInventory, RepresentedPhysicalField, TraceFieldEndpoint,
};
pub use coupled_fieldwise_resolution::{
    CoupledFieldwiseRealizationRequest, CoupledFieldwiseRealizationRequirements,
    ResolvedCoupledFieldwiseRealization, resolve_coupled_fieldwise,
};
pub use discretization::{
    Discretization, DiscretizationMethod, MeshArtifactReference, MeshKind, MeshPolicy,
    QuadraturePolicy, Space, SpaceFamily,
};
use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};
pub use execution::{ExecutionSchedule, Target};
pub use fieldwise::{
    AlgebraicBlockScale, FieldSpaceBinding, FieldwiseRealizationPlan,
    FieldwiseSpatialDiscretization, PositivePhysicalScale, SymmetricCongruenceScaling,
};
pub use fieldwise_resolution::{
    FieldwiseRealizationRequest, FieldwiseRealizationRequirements, ResolvedFieldwiseRealization,
    resolve_fieldwise,
};
pub use fixed_topology_ale::{
    AleGeometryQualityGate, FixedTopologyAleCoupledRealizationPlan,
    FixedTopologyAleCoupledRealizationRequest, FixedTopologyAleCoupledRealizationRequirements,
    GclCompatibleAlePullback, P1HarmonicMeshMotionPolicy,
    ResolvedFixedTopologyAleCoupledRealization, resolve_fixed_topology_ale_coupled,
};
pub use identity::{
    DefaultPolicyVersion, RealizationRequest, RealizationRevision, SemanticRevision,
};
pub use plan::{RealizationPlan, default_plan_v0};
pub use portable_graph::{
    AlgebraicSystemId, AlgebraicSystemNode, CoordinateTreatment, DomainConfiguration,
    DomainDiscretizationId, DomainDiscretizationNode, FieldRepresentationId,
    FieldRepresentationNode, GeometryActionId, GeometryActionNode, LinearSolveId, LinearSolveNode,
    NonlinearSolveId, NonlinearSolveNode, PlacementRequirementId, PlacementRequirementNode,
    PortableRealizationGraph, SolveRoot, SystemBlock, SystemScaling, TransformationId,
    TransformationNode,
};
pub use remesh_transfer::{AleFsiRemeshScaleProfile2d, AleFsiRemeshTransferPlan2d};
pub use resolution::{RealizationLineage, ResolutionSource, ResolvedRealization, resolve};
pub use transient_cell_centered_incompressible::{
    CartesianCentralNewtonianTraction, ImplicitCenteredMomentumConvection,
    MomentumWeightedLinearExactCoupling, PositiveMomentumDiagonal,
    ResolvedTransientCellCenteredIncompressibleFlowRealization,
    TransientCellCenteredIncompressibleFlowCapabilities,
    TransientCellCenteredIncompressibleFlowRealizationPlan,
    TransientCellCenteredIncompressibleFlowRealizationRequest,
    TransientCellCenteredIncompressibleFlowRealizationRequirements, TransientFaceFluxHistory,
    resolve_transient_cell_centered_incompressible_flow,
};
pub use transient_cell_centered_transport::{
    CellCenteredConvection, CellCenteredConvectionScheme, OrthogonalTwoPointDiffusion,
    ResolvedTransientCellCenteredTransportRealization, TransientCellCenteredTransportCapabilities,
    TransientCellCenteredTransportRealizationPlan,
    TransientCellCenteredTransportRealizationRequest,
    TransientCellCenteredTransportRealizationRequirements,
    resolve_transient_cell_centered_transport,
};
pub use transient_fieldwise::{
    BackwardEulerRelationStep, EnergySkewConvection, NonlinearSolvePlan,
    ResolvedTransientFieldwiseRealization, TransientFieldwiseRealizationPlan,
    TransientFieldwiseRealizationRequest, TransientFieldwiseRealizationRequirements,
    resolve_transient_fieldwise,
};

use diagnostic::invalid_realization;
use identity::Selection;

#[cfg(test)]
mod tests;
