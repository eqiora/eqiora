use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};
use std::num::{NonZeroU16, NonZeroU64, NonZeroUsize};

use crate::dimension::WireDimension;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DynQuantity, Id, ScalarType};
use eqiora_realization::{
    AlgebraicBlockScale, Discretization, DiscretizationMethod, ExecutionSchedule,
    FieldSpaceBinding, FieldwiseRealizationPlan, FieldwiseRealizationRequirements,
    FieldwiseSpatialDiscretization, MeshArtifactReference, MeshPolicy, PositivePhysicalScale,
    QuadraturePolicy, RealizationRequirements, Space, SpaceFamily, SymmetricCongruenceScaling,
    Target, VectorLayoutKind,
};
use eqiora_solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverPlan,
};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{ArtifactDigest, LayoutArtifacts, RealizationDecoderLimits, invalid_artifact};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireFieldwiseRequirements {
    domain_ulid: String,
    unknown_field_ulids: Vec<String>,
    execution: WireExecutionRequirements,
}

impl WireFieldwiseRequirements {
    pub(super) fn encode(value: &FieldwiseRealizationRequirements) -> Result<Self, Diagnostic> {
        Ok(Self {
            domain_ulid: value.domain().ulid().to_string(),
            unknown_field_ulids: value
                .unknown_fields()
                .iter()
                .map(|field| field.ulid().to_string())
                .collect(),
            execution: WireExecutionRequirements::encode(value.execution())?,
        })
    }

    pub(super) fn decode(self) -> Result<FieldwiseRealizationRequirements, Diagnostic> {
        let domain = parse_id::<kinds::Domain>(&self.domain_ulid, "Domain")?;
        let fields = self
            .unknown_field_ulids
            .iter()
            .map(|value| parse_id::<kinds::Field>(value, "Field"))
            .collect::<Result<Vec<_>, _>>()?;
        FieldwiseRealizationRequirements::new(domain, fields, self.execution.decode()?)
            .map_err(realization_error)
    }

    pub(super) fn validate_limits(
        &self,
        limits: RealizationDecoderLimits,
    ) -> Result<(), Diagnostic> {
        if self.unknown_field_ulids.len() > limits.max_realization_fields {
            return Err(invalid_artifact(
                "field-wise realization unknown-Field count exceeds the decoder limit",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireExecutionRequirements {
    spatial_dimension: u64,
    scalar_type: WireScalarType,
    vector_layout: WireVectorLayout,
}

impl WireExecutionRequirements {
    pub(crate) fn encode(value: RealizationRequirements) -> Result<Self, Diagnostic> {
        Ok(Self {
            spatial_dimension: encode_usize(value.spatial_dimension().get(), "spatial dimension")?,
            scalar_type: WireScalarType::encode(value.scalar_type()),
            vector_layout: WireVectorLayout::encode(value.vector_layout()),
        })
    }

    pub(crate) fn decode(self) -> Result<RealizationRequirements, Diagnostic> {
        Ok(RealizationRequirements::new(
            decode_nonzero_usize(self.spatial_dimension, "spatial dimension")?,
            self.scalar_type.decode(),
            self.vector_layout.decode(),
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireFieldwisePlan {
    spatial: WireFieldwiseSpatial,
    scaling: WireCongruenceScaling,
    operator_properties: WireOperatorProperties,
    solver: WireSolverPlan,
    target: WireTarget,
    schedule: WireSchedule,
}

impl WireFieldwisePlan {
    pub(super) fn encode(value: &FieldwiseRealizationPlan) -> Result<Self, Diagnostic> {
        Ok(Self {
            spatial: WireFieldwiseSpatial::encode(value.spatial())?,
            scaling: WireCongruenceScaling::encode(value.scaling()),
            operator_properties: WireOperatorProperties::encode(value.operator_properties()),
            solver: WireSolverPlan::encode(value.solver())?,
            target: WireTarget::encode(value.target())?,
            schedule: WireSchedule::encode(value.schedule()),
        })
    }

    pub(super) fn decode(self) -> Result<FieldwiseRealizationPlan, Diagnostic> {
        FieldwiseRealizationPlan::new(
            self.spatial.decode()?,
            self.scaling.decode()?,
            self.operator_properties.decode(),
            self.solver.decode()?,
            self.target.decode()?,
            self.schedule.decode()?,
        )
        .map_err(realization_error)
    }

    pub(super) fn validate_limits(
        &self,
        limits: RealizationDecoderLimits,
    ) -> Result<(), Diagnostic> {
        if self.spatial.field_spaces.len() > limits.max_realization_fields {
            return Err(invalid_artifact(
                "field-wise realization Field-space count exceeds the decoder limit",
            ));
        }
        if self.spatial.constraints.len() > limits.max_realization_constraints {
            return Err(invalid_artifact(
                "field-wise realization constraint count exceeds the decoder limit",
            ));
        }
        if self.scaling.block_scales.len() > limits.max_realization_blocks {
            return Err(invalid_artifact(
                "field-wise realization block-scale count exceeds the decoder limit",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFieldwiseSpatial {
    domain_ulid: String,
    coordinate_length_scale: WirePhysicalScale,
    field_spaces: Vec<WireFieldSpaceBinding>,
    constraints: Vec<WireAlgebraicConstraint>,
    discretization: WireDiscretization,
}

impl WireFieldwiseSpatial {
    fn encode(value: &FieldwiseSpatialDiscretization) -> Result<Self, Diagnostic> {
        Ok(Self {
            domain_ulid: value.domain().ulid().to_string(),
            coordinate_length_scale: WirePhysicalScale::encode(value.coordinate_length_scale()),
            field_spaces: value
                .field_spaces()
                .iter()
                .copied()
                .map(WireFieldSpaceBinding::encode)
                .collect(),
            constraints: value
                .constraints()
                .iter()
                .copied()
                .map(WireAlgebraicConstraint::encode)
                .collect(),
            discretization: WireDiscretization::encode(value.discretization())?,
        })
    }

    fn decode(self) -> Result<FieldwiseSpatialDiscretization, Diagnostic> {
        FieldwiseSpatialDiscretization::new(
            parse_id::<kinds::Domain>(&self.domain_ulid, "Domain")?,
            self.coordinate_length_scale.decode()?,
            self.field_spaces
                .into_iter()
                .map(WireFieldSpaceBinding::decode)
                .collect::<Result<Vec<_>, _>>()?,
            self.constraints
                .into_iter()
                .map(WireAlgebraicConstraint::decode)
                .collect::<Result<Vec<_>, _>>()?,
            self.discretization.decode()?,
        )
        .map_err(realization_error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireFieldSpaceBinding {
    field_ulid: String,
    space: WireSpace,
}

impl WireFieldSpaceBinding {
    pub(crate) fn encode(value: FieldSpaceBinding) -> Self {
        Self {
            field_ulid: value.field().ulid().to_string(),
            space: WireSpace::encode(value.space()),
        }
    }

    pub(crate) fn decode(self) -> Result<FieldSpaceBinding, Diagnostic> {
        Ok(FieldSpaceBinding::new(
            parse_id::<kinds::Field>(&self.field_ulid, "Field")?,
            self.space.decode()?,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireSpace {
    ContinuousLagrange { order: u16 },
    SimplexP1Bubble,
    CellConstant,
}

impl WireSpace {
    pub(crate) const fn encode(value: Space) -> Self {
        match value.family() {
            SpaceFamily::ContinuousLagrange { order } => {
                Self::ContinuousLagrange { order: order.get() }
            }
            SpaceFamily::SimplexP1Bubble => Self::SimplexP1Bubble,
            SpaceFamily::CellConstant => Self::CellConstant,
        }
    }

    pub(crate) fn decode(self) -> Result<Space, Diagnostic> {
        match self {
            Self::ContinuousLagrange { order } => NonZeroU16::new(order)
                .map(Space::continuous_lagrange)
                .ok_or_else(|| invalid_artifact("Lagrange order must be non-zero")),
            Self::SimplexP1Bubble => Ok(Space::simplex_p1_bubble()),
            Self::CellConstant => Ok(Space::cell_constant()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireAlgebraicConstraint {
    ZeroIntegral { field_ulid: String },
}

impl WireAlgebraicConstraint {
    pub(crate) fn encode(value: AlgebraicConstraint) -> Self {
        match value {
            AlgebraicConstraint::ZeroIntegral { field } => Self::ZeroIntegral {
                field_ulid: field.ulid().to_string(),
            },
        }
    }

    pub(crate) fn decode(self) -> Result<AlgebraicConstraint, Diagnostic> {
        match self {
            Self::ZeroIntegral { field_ulid } => Ok(AlgebraicConstraint::ZeroIntegral {
                field: parse_id::<kinds::Field>(&field_ulid, "constraint Field")?,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireCongruenceScaling {
    block_scales: Vec<WireAlgebraicBlockScale>,
    weak_functional_scale: WirePhysicalScale,
}

impl WireCongruenceScaling {
    pub(crate) fn encode(value: &SymmetricCongruenceScaling) -> Self {
        Self {
            block_scales: value
                .block_scales()
                .iter()
                .copied()
                .map(WireAlgebraicBlockScale::encode)
                .collect(),
            weak_functional_scale: WirePhysicalScale::encode(value.weak_functional_scale()),
        }
    }

    pub(crate) fn decode(self) -> Result<SymmetricCongruenceScaling, Diagnostic> {
        SymmetricCongruenceScaling::new(
            self.block_scales
                .into_iter()
                .map(WireAlgebraicBlockScale::decode)
                .collect::<Result<Vec<_>, _>>()?,
            self.weak_functional_scale.decode()?,
        )
        .map_err(realization_error)
    }

    pub(crate) fn block_count(&self) -> usize {
        self.block_scales.len()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAlgebraicBlockScale {
    block: WireAlgebraicBlock,
    scale: WirePhysicalScale,
}

impl WireAlgebraicBlockScale {
    fn encode(value: AlgebraicBlockScale) -> Self {
        Self {
            block: WireAlgebraicBlock::encode(value.block()),
            scale: WirePhysicalScale::encode(value.scale()),
        }
    }

    fn decode(self) -> Result<AlgebraicBlockScale, Diagnostic> {
        Ok(AlgebraicBlockScale::new(
            self.block.decode()?,
            self.scale.decode()?,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireAlgebraicBlock {
    Field { field_ulid: String },
    ConstraintMultiplier { field_ulid: String },
}

impl WireAlgebraicBlock {
    fn encode(value: AlgebraicBlock) -> Self {
        match value {
            AlgebraicBlock::Field(field) => Self::Field {
                field_ulid: field.ulid().to_string(),
            },
            AlgebraicBlock::ConstraintMultiplier { field } => Self::ConstraintMultiplier {
                field_ulid: field.ulid().to_string(),
            },
        }
    }

    fn decode(self) -> Result<AlgebraicBlock, Diagnostic> {
        match self {
            Self::Field { field_ulid } => Ok(AlgebraicBlock::Field(parse_id::<kinds::Field>(
                &field_ulid,
                "algebraic-block Field",
            )?)),
            Self::ConstraintMultiplier { field_ulid } => Ok(AlgebraicBlock::ConstraintMultiplier {
                field: parse_id::<kinds::Field>(&field_ulid, "constraint-multiplier Field")?,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WirePhysicalScale {
    coherent_si_value: f64,
    dimension: WireDimension,
}

impl WirePhysicalScale {
    pub(crate) const fn encode(value: PositivePhysicalScale) -> Self {
        let quantity = value.quantity();
        Self {
            coherent_si_value: quantity.value(),
            dimension: WireDimension::encode(quantity.dim()),
        }
    }

    pub(crate) fn decode(self) -> Result<PositivePhysicalScale, Diagnostic> {
        PositivePhysicalScale::new(DynQuantity::new(
            self.coherent_si_value,
            self.dimension.decode(),
        ))
        .map_err(realization_error)
    }
}

pub(crate) type WireDiscretization = WireDiscretizationWith<WireQuadrature>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireDiscretizationWith<Q> {
    method: WireDiscretizationMethod,
    mesh: WireMesh,
    quadrature: Q,
}

impl<Q: WireQuadratureCodec> WireDiscretizationWith<Q> {
    pub(crate) fn encode(value: Discretization) -> Result<Self, Diagnostic> {
        Ok(Self {
            method: WireDiscretizationMethod::encode(value.method()),
            mesh: WireMesh::encode(value.mesh())?,
            quadrature: Q::encode(value.quadrature())?,
        })
    }

    pub(crate) fn decode(self) -> Result<Discretization, Diagnostic> {
        Ok(Discretization::new(
            self.method.decode(),
            self.mesh.decode()?,
            self.quadrature.decode()?,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireDiscretizationMethod {
    ContinuousGalerkin,
    CellCenteredFiniteVolume,
}

impl WireDiscretizationMethod {
    const fn encode(value: DiscretizationMethod) -> Self {
        match value {
            DiscretizationMethod::ContinuousGalerkin => Self::ContinuousGalerkin,
            DiscretizationMethod::CellCenteredFiniteVolume => Self::CellCenteredFiniteVolume,
        }
    }

    const fn decode(self) -> DiscretizationMethod {
        match self {
            Self::ContinuousGalerkin => DiscretizationMethod::ContinuousGalerkin,
            Self::CellCenteredFiniteVolume => DiscretizationMethod::CellCenteredFiniteVolume,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireMesh {
    GeneratedUniform { cells_per_axis: u64 },
    ImportedSimplicial { artifact_sha256: String },
}

impl WireMesh {
    fn encode(value: MeshPolicy) -> Result<Self, Diagnostic> {
        match value {
            MeshPolicy::GeneratedUniform { cells_per_axis } => Ok(Self::GeneratedUniform {
                cells_per_axis: encode_usize(cells_per_axis.get(), "cells per axis")?,
            }),
            MeshPolicy::ImportedSimplicial { artifact } => Ok(Self::ImportedSimplicial {
                artifact_sha256: ArtifactDigest::from_sha256(artifact.sha256()).to_string(),
            }),
            MeshPolicy::SuppliedCartesian { .. }
            | MeshPolicy::SuppliedCartesian1d { .. }
            | MeshPolicy::SuppliedCartesian3d { .. } => Err(invalid_artifact(
                "this realization family cannot encode supplied Cartesian mesh policy",
            )),
        }
    }

    fn decode(self) -> Result<MeshPolicy, Diagnostic> {
        match self {
            Self::GeneratedUniform { cells_per_axis } => Ok(MeshPolicy::GeneratedUniform {
                cells_per_axis: decode_nonzero_usize(cells_per_axis, "cells per axis")?,
            }),
            Self::ImportedSimplicial { artifact_sha256 } => {
                let digest = ArtifactDigest::from_hex(artifact_sha256)?;
                Ok(MeshPolicy::ImportedSimplicial {
                    artifact: MeshArtifactReference::from_sha256(digest.sha256_bytes()),
                })
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireQuadrature {
    GaussLegendre { points_per_axis: u64 },
    CellCentroid,
    SimplexCentroid,
    TriangleDuffyGaussLegendre { points_per_axis: u64 },
}

pub(crate) trait WireQuadratureCodec: Sized {
    fn encode(value: QuadraturePolicy) -> Result<Self, Diagnostic>;

    fn decode(self) -> Result<QuadraturePolicy, Diagnostic>;
}

impl WireQuadratureCodec for WireQuadrature {
    fn encode(value: QuadraturePolicy) -> Result<Self, Diagnostic> {
        Ok(match value {
            QuadraturePolicy::GaussLegendre { points_per_axis } => Self::GaussLegendre {
                points_per_axis: encode_usize(points_per_axis.get(), "quadrature points per axis")?,
            },
            QuadraturePolicy::CellCentroid => Self::CellCentroid,
            QuadraturePolicy::SimplexCentroid => Self::SimplexCentroid,
            QuadraturePolicy::TriangleDuffyGaussLegendre { points_per_axis } => {
                Self::TriangleDuffyGaussLegendre {
                    points_per_axis: encode_usize(
                        points_per_axis.get(),
                        "Duffy quadrature points per axis",
                    )?,
                }
            }
            QuadraturePolicy::SimplexDuffyGaussLegendre { .. } => {
                return Err(invalid_artifact(
                    "this realization family cannot encode dimension-explicit simplex Duffy quadrature; the ALE envelope supports it",
                ));
            }
        })
    }

    fn decode(self) -> Result<QuadraturePolicy, Diagnostic> {
        match self {
            Self::GaussLegendre { points_per_axis } => Ok(QuadraturePolicy::GaussLegendre {
                points_per_axis: decode_nonzero_usize(
                    points_per_axis,
                    "quadrature points per axis",
                )?,
            }),
            Self::CellCentroid => Ok(QuadraturePolicy::CellCentroid),
            Self::SimplexCentroid => Ok(QuadraturePolicy::SimplexCentroid),
            Self::TriangleDuffyGaussLegendre { points_per_axis } => {
                Ok(QuadraturePolicy::TriangleDuffyGaussLegendre {
                    points_per_axis: decode_nonzero_usize(
                        points_per_axis,
                        "Duffy quadrature points per axis",
                    )?,
                })
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireOperatorProperties {
    General,
    SymmetricPositiveDefinite,
    SymmetricIndefinite,
}

impl WireOperatorProperties {
    pub(crate) const fn encode(value: LinearOperatorProperties) -> Self {
        match value {
            LinearOperatorProperties::General => Self::General,
            LinearOperatorProperties::SymmetricPositiveDefinite => Self::SymmetricPositiveDefinite,
            LinearOperatorProperties::SymmetricIndefinite => Self::SymmetricIndefinite,
        }
    }

    pub(crate) const fn decode(self) -> LinearOperatorProperties {
        match self {
            Self::General => LinearOperatorProperties::General,
            Self::SymmetricPositiveDefinite => LinearOperatorProperties::SymmetricPositiveDefinite,
            Self::SymmetricIndefinite => LinearOperatorProperties::SymmetricIndefinite,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireSolverPlan {
    algorithm: WireLinearSolver,
    preconditioner: WirePreconditioner,
    reduction: WireReduction,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: u64,
}

impl WireSolverPlan {
    pub(crate) fn encode(value: SolverPlan) -> Result<Self, Diagnostic> {
        Ok(Self {
            algorithm: WireLinearSolver::encode(value.algorithm()),
            preconditioner: WirePreconditioner::encode(value.preconditioner()),
            reduction: WireReduction::encode(value.reduction()),
            relative_tolerance: value.relative_tolerance(),
            absolute_tolerance: value.absolute_tolerance(),
            maximum_iterations: encode_usize(
                value.maximum_iterations().get(),
                "maximum iterations",
            )?,
        })
    }

    pub(crate) fn decode(self) -> Result<SolverPlan, Diagnostic> {
        SolverPlan::new(
            self.algorithm.decode(),
            self.relative_tolerance,
            self.absolute_tolerance,
            decode_nonzero_usize(self.maximum_iterations, "maximum iterations")?,
        )
        .map(|plan| {
            plan.with_preconditioner(self.preconditioner.decode())
                .with_reduction(self.reduction.decode())
        })
        .map_err(realization_error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireLinearSolver {
    ConjugateGradient,
    BiConjugateGradientStabilized,
    MinimumResidual,
    SparseLu,
}

impl WireLinearSolver {
    const fn encode(value: LinearSolver) -> Self {
        match value {
            LinearSolver::ConjugateGradient => Self::ConjugateGradient,
            LinearSolver::BiConjugateGradientStabilized => Self::BiConjugateGradientStabilized,
            LinearSolver::MinimumResidual => Self::MinimumResidual,
            LinearSolver::SparseLu => Self::SparseLu,
        }
    }

    const fn decode(self) -> LinearSolver {
        match self {
            Self::ConjugateGradient => LinearSolver::ConjugateGradient,
            Self::BiConjugateGradientStabilized => LinearSolver::BiConjugateGradientStabilized,
            Self::MinimumResidual => LinearSolver::MinimumResidual,
            Self::SparseLu => LinearSolver::SparseLu,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WirePreconditioner {
    Identity,
    Jacobi,
}

impl WirePreconditioner {
    const fn encode(value: PreconditionerPolicy) -> Self {
        match value {
            PreconditionerPolicy::Identity => Self::Identity,
            PreconditionerPolicy::Jacobi => Self::Jacobi,
        }
    }

    const fn decode(self) -> PreconditionerPolicy {
        match self {
            Self::Identity => PreconditionerPolicy::Identity,
            Self::Jacobi => PreconditionerPolicy::Jacobi,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireReduction {
    Reproducible,
    Fast,
}

impl WireReduction {
    const fn encode(value: ReductionPolicy) -> Self {
        match value {
            ReductionPolicy::Reproducible => Self::Reproducible,
            ReductionPolicy::Fast => Self::Fast,
        }
    }

    const fn decode(self) -> ReductionPolicy {
        match self {
            Self::Reproducible => ReductionPolicy::Reproducible,
            Self::Fast => ReductionPolicy::Fast,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireScalarType {
    F32,
    F64,
}

impl WireScalarType {
    const fn encode(value: ScalarType) -> Self {
        match value {
            ScalarType::F32 => Self::F32,
            ScalarType::F64 => Self::F64,
        }
    }

    const fn decode(self) -> ScalarType {
        match self {
            Self::F32 => ScalarType::F32,
            Self::F64 => ScalarType::F64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireVectorLayout {
    Replicated,
    Distributed,
}

impl WireVectorLayout {
    const fn encode(value: VectorLayoutKind) -> Self {
        match value {
            VectorLayoutKind::Replicated => Self::Replicated,
            VectorLayoutKind::Distributed => Self::Distributed,
        }
    }

    const fn decode(self) -> VectorLayoutKind {
        match self {
            Self::Replicated => VectorLayoutKind::Replicated,
            Self::Distributed => VectorLayoutKind::Distributed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireTarget {
    HostCpu { threads: u64 },
    CudaGpu { device: u16 },
}

impl WireTarget {
    pub(crate) fn encode(value: Target) -> Result<Self, Diagnostic> {
        match value {
            Target::HostCpu { threads } => Ok(Self::HostCpu {
                threads: encode_usize(threads.get(), "host threads")?,
            }),
            Target::CudaGpu { device } => Ok(Self::CudaGpu { device }),
        }
    }

    pub(crate) fn decode(self) -> Result<Target, Diagnostic> {
        match self {
            Self::HostCpu { threads } => Ok(Target::HostCpu {
                threads: decode_nonzero_usize(threads, "host threads")?,
            }),
            Self::CudaGpu { device } => Ok(Target::CudaGpu { device }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireSchedule {
    Offline,
    RealTime { priority: u16, deadline_ns: u64 },
}

impl WireSchedule {
    pub(crate) const fn encode(value: ExecutionSchedule) -> Self {
        match value {
            ExecutionSchedule::Offline => Self::Offline,
            ExecutionSchedule::RealTime {
                priority,
                deadline_ns,
            } => Self::RealTime {
                priority,
                deadline_ns: deadline_ns.get(),
            },
        }
    }

    pub(crate) fn decode(self) -> Result<ExecutionSchedule, Diagnostic> {
        match self {
            Self::Offline => Ok(ExecutionSchedule::Offline),
            Self::RealTime {
                priority,
                deadline_ns,
            } => Ok(ExecutionSchedule::RealTime {
                priority,
                deadline_ns: NonZeroU64::new(deadline_ns)
                    .ok_or_else(|| invalid_artifact("real-time deadline must be non-zero"))?,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireLayoutArtifacts {
    Replicated,
    Distributed {
        layout_sha256: String,
        partition_sha256: String,
    },
}

impl WireLayoutArtifacts {
    pub(crate) fn encode(value: LayoutArtifacts) -> Self {
        match value {
            LayoutArtifacts::Replicated => Self::Replicated,
            LayoutArtifacts::Distributed { layout, partition } => Self::Distributed {
                layout_sha256: layout.to_string(),
                partition_sha256: partition.to_string(),
            },
        }
    }

    pub(crate) fn decode(&self) -> LayoutArtifacts {
        match self {
            Self::Replicated => LayoutArtifacts::Replicated,
            Self::Distributed {
                layout_sha256,
                partition_sha256,
            } => LayoutArtifacts::Distributed {
                layout: ArtifactDigest::from_hex(layout_sha256.clone())
                    .expect("validated V2 layout digest remains valid"),
                partition: ArtifactDigest::from_hex(partition_sha256.clone())
                    .expect("validated V2 partition digest remains valid"),
            },
        }
    }

    pub(crate) fn decode_validated(&self) -> Result<LayoutArtifacts, Diagnostic> {
        match self {
            Self::Replicated => Ok(LayoutArtifacts::Replicated),
            Self::Distributed {
                layout_sha256,
                partition_sha256,
            } => Ok(LayoutArtifacts::Distributed {
                layout: ArtifactDigest::from_hex(layout_sha256.clone())?,
                partition: ArtifactDigest::from_hex(partition_sha256.clone())?,
            }),
        }
    }
}

fn parse_id<E: eqiora_core::Entity>(value: &str, label: &str) -> Result<Id<E>, Diagnostic> {
    value
        .parse::<Ulid>()
        .map(Id::from_ulid)
        .map_err(|_| invalid_artifact(format!("{label} ULID is malformed")))
}

pub(crate) fn encode_usize(value: usize, label: &str) -> Result<u64, Diagnostic> {
    u64::try_from(value).map_err(|_| invalid_artifact(format!("{label} exceeds wire u64")))
}

pub(crate) fn decode_nonzero_usize(value: u64, label: &str) -> Result<NonZeroUsize, Diagnostic> {
    let value = usize::try_from(value)
        .map_err(|_| invalid_artifact(format!("{label} exceeds local usize")))?;
    NonZeroUsize::new(value).ok_or_else(|| invalid_artifact(format!("{label} must be non-zero")))
}

fn realization_error(error: Diagnostic) -> Diagnostic {
    invalid_artifact(error.message())
}
