//! Equation-aware admission of the fixed-reference FSI numerical realization.

use eqiora_solver::AlgebraicBlock;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_assembly::{AssemblyBackend, AssemblyPacketSetIdentityV1, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, ScalarType};
use eqiora_meshing::{QuadratureRule, SimplicialMesh, triangle_duffy_gauss_legendre};
use eqiora_realization::{
    AlgebraicBlockScale, BackwardEulerStateBinding, BackwardEulerStep,
    CoupledFieldwiseRealizationPlan, CoupledFieldwiseRealizationRequirements,
    CoupledFieldwiseSpatialDiscretization, Discretization, DiscretizationMethod,
    DomainFieldDiscretization, DomainFieldInventory, ExecutionSchedule, FieldSpaceBinding,
    MeshArtifactReference, MeshPolicy, PlacementRequirementNode, PositivePhysicalScale,
    QuadraturePolicy, RealizationRequirements, ResolvedCoupledFieldwiseRealization, Space,
    SymmetricCongruenceScaling, Target, VectorLayoutKind,
};
use eqiora_solver::{LinearOperatorProperties, ReductionPolicy, SolverPlan};

use super::FixedReferenceFsiCartesianModel2d;
use crate::discrete_block::DiscreteBlockSystem;
use crate::simplicial_fsi::PreparedFixedReferenceFsiAssembly;
use crate::simplicial_fsi::{
    FixedReferenceFsiBoundary, FixedReferenceFsiLoad, FixedReferenceFsiMaterial,
    FixedReferenceFsiPartition, FixedReferenceFsiScale, FixedReferenceFsiState,
    FixedReferenceFsiStepConfig,
};

mod block;
mod regions;
mod result;
mod validate;

pub use result::{
    AcceptedDistributedFixedReferenceFsiStep2d, FinalizedResolvedFixedReferenceFsiStep2d,
    FixedReferenceFsiFieldIdentities2d, PreparedDistributedFixedReferenceFsiStep2d,
    ResolvedFixedReferenceFsiSolution2d,
};
use validate::{
    field_identities, fluid_domain, fluid_pressure, fluid_velocity, invalid_realization,
    realization_error, require_boundary_meaning, require_dimension, require_exact_plan,
    require_mesh_partition, require_solver, require_zero_load, solid_displacement, solid_domain,
    solid_kinematic_relation, solid_velocity, state_pair, trace_quotient,
};

const DIMENSION: usize = 2;
const DUFFY_POINTS_PER_AXIS: usize = 4;

/// Run-local fixed-reference FSI structure authenticated independently of action State.
pub(crate) struct PreparedResolvedFixedReferenceFsiRun2d<'a> {
    model: &'a FixedReferenceFsiCartesianModel2d,
    resolved: &'a ResolvedCoupledFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    mesh: &'a SimplicialMesh,
    partition: &'a FixedReferenceFsiPartition<2>,
    assembly: &'a dyn AssemblyBackend,
    config: FixedReferenceFsiStepConfig<2>,
    quadrature: QuadratureRule,
    realization_graph: eqiora_realization::PortableRealizationGraph,
    block_system: DiscreteBlockSystem,
    layout: crate::simplicial_fsi::layout::FsiLayout<2>,
    regions: std::collections::BTreeMap<
        eqiora_core::RawId,
        crate::form_compiler::region::BoundRegionForm,
    >,
}

impl PreparedResolvedFixedReferenceFsiRun2d<'_> {
    #[cfg(test)]
    pub(crate) fn layout(&self) -> &crate::simplicial_fsi::layout::FsiLayout<2> {
        &self.layout
    }

    /// Assemble and finalize one action against the immutable prepared structure.
    pub(crate) fn finalize(
        &self,
        previous: &FixedReferenceFsiState<2>,
    ) -> Result<FinalizedResolvedFixedReferenceFsiStep2d, Diagnostic> {
        let checked_assembly = self.block_system.checked_backend(self.assembly);
        let prepared = PreparedFixedReferenceFsiAssembly::new(
            self.mesh,
            self.partition,
            previous,
            self.config,
            &self.quadrature,
            AssemblyPacketSetIdentityV1::from_sha256(self.mesh_artifact.sha256()),
            self.layout.clone(),
        )?;
        let work = regions::prepare_cells(
            self.model,
            &self.regions,
            self.mesh,
            self.partition,
            previous,
            &self.quadrature,
            &prepared,
            AssemblyPacketSetIdentityV1::from_sha256(self.mesh_artifact.sha256()),
        )?;
        let assembled = checked_assembly.assemble(prepared.plan(), &work)?;
        let reactions = prepared.reactions(&work)?;
        let inner = prepared.finish(assembled, reactions)?;
        FinalizedResolvedFixedReferenceFsiStep2d::new(
            self.resolved.model(),
            self.resolved.semantic_revision(),
            self.resolved.realization_revision(),
            self.mesh_artifact,
            field_identities(self.model),
            self.partition.clone(),
            self.resolved.plan().clone(),
            self.realization_graph.clone(),
            self.block_system.clone(),
            inner,
        )
    }
}

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");

/// Closed execution tuples admitted by the fixed-reference FSI finalizer.
///
/// The legacy target still carries a local CUDA ordinal for compatibility,
/// while the portable placement retains only the one-device requirement.
/// Keeping those facts together prevents target and reduction validation from
/// becoming two independently widened match statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FixedReferenceFsiExecutionProfile {
    HostReproducible,
    CudaFast { device: u16 },
    DistributedCudaReproducible { device: u16 },
}

impl FixedReferenceFsiExecutionProfile {
    const fn target(self) -> Target {
        match self {
            Self::HostReproducible => Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            Self::CudaFast { device } => Target::CudaGpu { device },
            Self::DistributedCudaReproducible { device } => Target::CudaGpu { device },
        }
    }

    const fn placement(self) -> PlacementRequirementNode {
        match self {
            Self::HostReproducible => PlacementRequirementNode::HostWorkers {
                workers_per_partition: NonZeroUsize::MIN,
            },
            Self::CudaFast { .. } | Self::DistributedCudaReproducible { .. } => {
                PlacementRequirementNode::CudaDevices {
                    devices_per_partition: NonZeroUsize::MIN,
                }
            }
        }
    }

    const fn reduction(self) -> ReductionPolicy {
        match self {
            Self::HostReproducible => ReductionPolicy::Reproducible,
            Self::CudaFast { .. } => ReductionPolicy::Fast,
            Self::DistributedCudaReproducible { .. } => ReductionPolicy::Reproducible,
        }
    }
}

/// Native coherent-SI scales for the monolithic fixed-reference FSI step.
///
/// The displacement scale is exactly `L`. Both velocity traces share `U` and
/// pressure uses `P`. The intrinsic-2D weak-functional scale is derived as
/// `Theta = P U L`, so no caller can independently alter the congruence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedReferenceFsiScaleProfile2d {
    length: PositivePhysicalScale,
    velocity: PositivePhysicalScale,
    pressure: PositivePhysicalScale,
    weak_functional: PositivePhysicalScale,
}

impl FixedReferenceFsiScaleProfile2d {
    /// Validate positive finite characteristic `L`, `U`, and `P` quantities.
    ///
    /// # Errors
    /// Returns `EQ0807` for an incompatible dimension, non-positive value, or
    /// overflow while deriving `Theta = P U L`.
    pub fn new(
        length: DynQuantity,
        velocity: DynQuantity,
        pressure: DynQuantity,
    ) -> Result<Self, Diagnostic> {
        require_dimension(length, LENGTH, "fixed-reference FSI length scale L")?;
        require_dimension(velocity, VELOCITY, "fixed-reference FSI velocity scale U")?;
        require_dimension(pressure, PRESSURE, "fixed-reference FSI pressure scale P")?;
        let length = PositivePhysicalScale::new(length).map_err(realization_error)?;
        let velocity = PositivePhysicalScale::new(velocity).map_err(realization_error)?;
        let pressure = PositivePhysicalScale::new(pressure).map_err(realization_error)?;
        let weak_functional = PositivePhysicalScale::new(
            pressure
                .quantity()
                .try_mul(velocity.quantity())?
                .try_mul(length.quantity())?,
        )
        .map_err(realization_error)?;
        Ok(Self {
            length,
            velocity,
            pressure,
            weak_functional,
        })
    }

    /// Characteristic length `L` and eliminated-displacement scale.
    #[must_use]
    pub const fn length(self) -> DynQuantity {
        self.length.quantity()
    }

    /// Common fluid/solid velocity scale `U`.
    #[must_use]
    pub const fn velocity(self) -> DynQuantity {
        self.velocity.quantity()
    }

    /// Fluid pressure scale `P`.
    #[must_use]
    pub const fn pressure(self) -> DynQuantity {
        self.pressure.quantity()
    }

    /// Derived intrinsic-2D weak-functional scale `Theta = P U L`.
    #[must_use]
    pub const fn weak_functional(self) -> DynQuantity {
        self.weak_functional.quantity()
    }
}

/// Exact lowerer facts for the admitted fixed-reference FSI realization.
#[must_use]
pub fn fixed_reference_fsi_requirements_2d(
    model: &FixedReferenceFsiCartesianModel2d,
) -> CoupledFieldwiseRealizationRequirements {
    fixed_reference_fsi_requirements_2d_for_layout(model, VectorLayoutKind::Replicated)
}

/// Exact lowerer facts with an explicit execution-vector layout.
///
/// The layout changes only how the one finalized algebraic system is owned at
/// execution. Domain, Field, trace-quotient, and eliminated-state meaning are
/// identical to [`fixed_reference_fsi_requirements_2d`].
#[must_use]
pub fn fixed_reference_fsi_requirements_2d_for_layout(
    model: &FixedReferenceFsiCartesianModel2d,
    vector_layout: VectorLayoutKind,
) -> CoupledFieldwiseRealizationRequirements {
    CoupledFieldwiseRealizationRequirements::new(
        [
            DomainFieldInventory::new(
                fluid_domain(model),
                [fluid_velocity(model), fluid_pressure(model)],
            )
            .expect("lowered fluid owns distinct velocity and pressure Fields"),
            DomainFieldInventory::new(
                solid_domain(model),
                [solid_displacement(model), solid_velocity(model)],
            )
            .expect("lowered solid owns distinct displacement and velocity Fields"),
        ],
        [trace_quotient(model)],
        state_pair(model),
        RealizationRequirements::new(
            NonZeroUsize::new(DIMENSION).expect("two is non-zero"),
            ScalarType::F64,
            vector_layout,
        ),
    )
    .expect("lowered FSI model owns one exact cross-Domain state and trace inventory")
}

/// Build the sole admitted CPU-reference plan from exact canonical roles.
///
/// Fluid velocity uses MINI, fluid pressure uses P1, solid velocity uses P1,
/// and solid displacement is an eliminated P1 Backward-Euler state. The two
/// velocity traces are one exact algebraic quotient. The complete coupled
/// operator, rather than an inherited fluid-only policy, closes pressure; no
/// pressure gauge is selected.
///
/// # Errors
/// Returns `EQ0807` for an invalid time quantity, solver tuple, scale, or
/// cross-field plan.
pub fn fixed_reference_fsi_plan_2d(
    model: &FixedReferenceFsiCartesianModel2d,
    mesh: MeshArtifactReference,
    time_step: DynQuantity,
    scales: FixedReferenceFsiScaleProfile2d,
    solver: SolverPlan,
) -> Result<CoupledFieldwiseRealizationPlan, Diagnostic> {
    fixed_reference_fsi_plan_2d_for_profile(
        model,
        mesh,
        time_step,
        scales,
        solver,
        FixedReferenceFsiExecutionProfile::HostReproducible,
    )
}

/// Build the sole admitted one-device CUDA plan from the same canonical roles.
///
/// This changes only execution placement and reduction policy. It retains the
/// exact symmetric-indefinite operator, identity-preconditioned MINRES,
/// coherent-SI scaling, trace quotient, pressure closure, and Backward-Euler
/// elimination selected by [`fixed_reference_fsi_plan_2d`]. The device ordinal
/// is a compatibility binding; the portable graph records one device per
/// partition without embedding an environment-local ordinal.
///
/// # Errors
/// Returns `EQ0807` unless the solver selects identity-preconditioned MINRES
/// with the backend-native fast reduction policy, or if another plan component
/// is invalid.
pub fn fixed_reference_fsi_cuda_plan_2d(
    model: &FixedReferenceFsiCartesianModel2d,
    mesh: MeshArtifactReference,
    time_step: DynQuantity,
    scales: FixedReferenceFsiScaleProfile2d,
    solver: SolverPlan,
    device: u16,
) -> Result<CoupledFieldwiseRealizationPlan, Diagnostic> {
    fixed_reference_fsi_plan_2d_for_profile(
        model,
        mesh,
        time_step,
        scales,
        solver,
        FixedReferenceFsiExecutionProfile::CudaFast { device },
    )
}

/// Build the sole admitted distributed-CUDA plan from the same canonical
/// roles and the MPI parent's reproducible host reduction policy.
///
/// CUDA owns only the deterministic partition-local CSR action. Distributed
/// MINRES vectors and reductions remain host/MPI owned, so this plan does not
/// inherit the complete-device solver's `Fast` reduction policy. Every rank
/// resolves the same deployment-local ordinal; physical device distinctness
/// is proven later by adapter UUID evidence.
///
/// # Errors
/// Returns `EQ0807` unless `device` is zero and the solver selects
/// reproducible identity-preconditioned MINRES, or if another plan component
/// is invalid.
pub fn fixed_reference_fsi_distributed_cuda_plan_2d(
    model: &FixedReferenceFsiCartesianModel2d,
    mesh: MeshArtifactReference,
    time_step: DynQuantity,
    scales: FixedReferenceFsiScaleProfile2d,
    solver: SolverPlan,
    device: u16,
) -> Result<CoupledFieldwiseRealizationPlan, Diagnostic> {
    if device != 0 {
        return Err(invalid_realization(
            "distributed CUDA FSI requires deployment-local device ordinal zero on every rank",
        ));
    }
    fixed_reference_fsi_plan_2d_for_profile(
        model,
        mesh,
        time_step,
        scales,
        solver,
        FixedReferenceFsiExecutionProfile::DistributedCudaReproducible { device },
    )
}

pub(super) fn fixed_reference_fsi_plan_2d_for_profile(
    model: &FixedReferenceFsiCartesianModel2d,
    mesh: MeshArtifactReference,
    time_step: DynQuantity,
    scales: FixedReferenceFsiScaleProfile2d,
    solver: SolverPlan,
    execution: FixedReferenceFsiExecutionProfile,
) -> Result<CoupledFieldwiseRealizationPlan, Diagnostic> {
    require_dimension(time_step, TIME, "fixed-reference FSI time step")?;
    require_solver(solver, execution)?;
    let p1 = Space::continuous_lagrange(NonZeroU16::MIN);
    let spatial = CoupledFieldwiseSpatialDiscretization::new(
        scales.length,
        [
            DomainFieldDiscretization::new(
                fluid_domain(model),
                [
                    FieldSpaceBinding::new(fluid_velocity(model), Space::simplex_p1_bubble()),
                    FieldSpaceBinding::new(fluid_pressure(model), p1),
                ],
                [],
            )
            .map_err(realization_error)?,
            DomainFieldDiscretization::new(
                solid_domain(model),
                [FieldSpaceBinding::new(solid_velocity(model), p1)],
                [],
            )
            .map_err(realization_error)?,
        ],
        [trace_quotient(model)],
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::ImportedSimplicial { artifact: mesh },
            QuadraturePolicy::TriangleDuffyGaussLegendre {
                points_per_axis: NonZeroUsize::new(DUFFY_POINTS_PER_AXIS)
                    .expect("four is non-zero"),
            },
        ),
    )
    .map_err(realization_error)?;
    let time_step = BackwardEulerStep::new(
        time_step,
        BackwardEulerStateBinding::new(state_pair(model), p1, scales.length),
    )
    .map_err(realization_error)?;
    let scaling = SymmetricCongruenceScaling::new(
        [
            AlgebraicBlockScale::new(
                AlgebraicBlock::Field(fluid_velocity(model)),
                scales.velocity,
            ),
            AlgebraicBlockScale::new(
                AlgebraicBlock::Field(fluid_pressure(model)),
                scales.pressure,
            ),
            AlgebraicBlockScale::new(
                AlgebraicBlock::Field(solid_velocity(model)),
                scales.velocity,
            ),
        ],
        scales.weak_functional,
    )
    .map_err(realization_error)?;
    CoupledFieldwiseRealizationPlan::new(
        spatial,
        time_step,
        scaling,
        LinearOperatorProperties::SymmetricIndefinite,
        solver,
        execution.target(),
        ExecutionSchedule::Offline,
    )
    .map_err(realization_error)
}

/// Finalize one exact resolved fixed-reference FSI step through the pure core.
///
/// The artifact layer must authenticate `mesh` and construct `partition` from
/// the exact geometry/correspondence witness before this function is called.
/// This layer replays all equation-aware Realization choices and geometric
/// Domain facts without depending on an L3 artifact representation.
///
/// # Errors
/// Rejects any drift in Domain/Field/Connection roles, mesh reference, spaces,
/// pressure policy, scaling, time elimination, solver, target, schedule,
/// boundary closure, zero-load policy, or physical mesh partition before the
/// pure simplicial operator is finalized.
pub fn finalize_resolved_fixed_reference_fsi_step_2d(
    model: &FixedReferenceFsiCartesianModel2d,
    resolved: &ResolvedCoupledFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    previous: &FixedReferenceFsiState<2>,
) -> Result<FinalizedResolvedFixedReferenceFsiStep2d, Diagnostic> {
    finalize_resolved_fixed_reference_fsi_step_2d_with_assembly(
        model,
        resolved,
        mesh_artifact,
        mesh,
        partition,
        previous,
        &REFERENCE_ASSEMBLY_BACKEND,
    )
}

/// Authenticate and retain fixed structure for one ephemeral common Run.
pub(crate) fn prepare_resolved_fixed_reference_fsi_run_2d<'a>(
    model: &'a FixedReferenceFsiCartesianModel2d,
    resolved: &'a ResolvedCoupledFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    mesh: &'a SimplicialMesh,
    partition: &'a FixedReferenceFsiPartition<2>,
) -> Result<PreparedResolvedFixedReferenceFsiRun2d<'a>, Diagnostic> {
    prepare_resolved_fixed_reference_fsi_run_2d_with_assembly(
        model,
        resolved,
        mesh_artifact,
        mesh,
        partition,
        &REFERENCE_ASSEMBLY_BACKEND,
    )
}

fn prepare_resolved_fixed_reference_fsi_run_2d_with_assembly<'a>(
    model: &'a FixedReferenceFsiCartesianModel2d,
    resolved: &'a ResolvedCoupledFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    mesh: &'a SimplicialMesh,
    partition: &'a FixedReferenceFsiPartition<2>,
    assembly: &'a dyn AssemblyBackend,
) -> Result<PreparedResolvedFixedReferenceFsiRun2d<'a>, Diagnostic> {
    require_zero_load(model)?;
    require_boundary_meaning(model)?;
    require_mesh_partition(model, mesh, partition)?;
    let realization_graph = resolved.portable_graph(solid_kinematic_relation(model))?;
    let scales = require_exact_plan(model, resolved, &realization_graph, mesh_artifact)?;
    let material = FixedReferenceFsiMaterial::<2>::from_admitted_solid(
        model.fluid().mass_density(),
        model.fluid().dynamic_viscosity(),
        model.solid().mass_density(),
        model.solid().continuum().material(),
    )
    .map_err(realization_error)?;
    let scale = FixedReferenceFsiScale::<2>::new(
        scales.length().value(),
        scales.velocity().value(),
        scales.pressure().value(),
    )
    .map_err(realization_error)?;
    let config = FixedReferenceFsiStepConfig::<2>::new(
        resolved.plan().time_step().duration().value(),
        material,
        scale,
        FixedReferenceFsiLoad::Zero,
    )
    .map_err(realization_error)?;
    let boundary =
        FixedReferenceFsiBoundary::<2>::homogeneous_exterior(mesh).map_err(realization_error)?;
    let quadrature =
        triangle_duffy_gauss_legendre(DUFFY_POINTS_PER_AXIS).map_err(realization_error)?;
    let block_system =
        block::fixed_reference_fsi_block_system(model, resolved, mesh_artifact, mesh, partition)?;
    let regions = regions::bind(model, resolved.plan())?;
    let layout = regions::layout(model, &regions, resolved.plan(), mesh, partition, &boundary)?;
    Ok(PreparedResolvedFixedReferenceFsiRun2d {
        model,
        resolved,
        mesh_artifact,
        mesh,
        partition,
        assembly,
        config,
        quadrature,
        realization_graph,
        block_system,
        regions,
        layout,
    })
}

/// Finalize the exact resolved FSI step through an explicit assembly backend.
///
/// This workspace-level conformance seam is intentionally absent from the
/// curated `eqiora` facade. It exists so transport and execution adapters can
/// prove that they consume the same equation-aware block system and local
/// packets as reference assembly, without receiving a public physics IR.
///
/// # Errors
/// Preserves every reference admission rule and the selected backend's
/// assembly diagnostic before constructing a resolved finalized step.
#[doc(hidden)]
pub fn finalize_resolved_fixed_reference_fsi_step_2d_with_assembly(
    model: &FixedReferenceFsiCartesianModel2d,
    resolved: &ResolvedCoupledFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    previous: &FixedReferenceFsiState<2>,
    assembly: &dyn AssemblyBackend,
) -> Result<FinalizedResolvedFixedReferenceFsiStep2d, Diagnostic> {
    prepare_resolved_fixed_reference_fsi_run_2d_with_assembly(
        model,
        resolved,
        mesh_artifact,
        mesh,
        partition,
        assembly,
    )?
    .finalize(previous)
}

#[cfg(test)]
#[path = "realization/tests.rs"]
mod tests;
