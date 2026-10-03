use std::num::{NonZeroU16, NonZeroU64, NonZeroUsize};

use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Entity, Id, ScalarType};
use eqiora_solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverPlan,
};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    AlgebraicBlock, AlgebraicBlockScale, AlgebraicConstraint, CellCenteredConvectionScheme,
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshArtifactReference, MeshPolicy,
    NonlinearSolvePlan, PositiveMomentumDiagonal, PositivePhysicalScale, QuadraturePolicy, Space,
    SpaceFamily, SymmetricCongruenceScaling, TransientFaceFluxHistory, VectorLayoutKind,
    invalid_realization,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "[(i32, i32); 7]", into = "[(i32, i32); 7]")]
pub(super) struct WireDimension(DimExponents);

impl WireDimension {
    pub(super) const fn encode(value: DimExponents) -> Self {
        Self(value)
    }

    const fn decode(self) -> DimExponents {
        self.0
    }
}

impl From<WireDimension> for [(i32, i32); 7] {
    fn from(value: WireDimension) -> Self {
        value.0.exponents()
    }
}

impl TryFrom<[(i32, i32); 7]> for WireDimension {
    type Error = Diagnostic;

    fn try_from(value: [(i32, i32); 7]) -> Result<Self, Self::Error> {
        let dimension = DimExponents::from_rationals(value)
            .ok_or_else(|| invalid_realization("invalid rational dimension exponents"))?;
        if dimension.exponents() != value {
            return Err(invalid_realization(
                "dimension exponents must be canonical reduced fractions",
            ));
        }
        Ok(Self(dimension))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireQuantity {
    coherent_si_value: f64,
    dimension: WireDimension,
}

impl WireQuantity {
    pub(super) const fn encode(value: DynQuantity) -> Self {
        Self {
            coherent_si_value: normalize_zero(value.value()),
            dimension: WireDimension::encode(value.dim()),
        }
    }

    pub(super) const fn decode(self) -> DynQuantity {
        DynQuantity::new(self.coherent_si_value, self.dimension.decode())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WirePositiveScale {
    quantity: WireQuantity,
}

impl WirePositiveScale {
    pub(super) const fn encode(value: PositivePhysicalScale) -> Self {
        Self {
            quantity: WireQuantity::encode(value.quantity()),
        }
    }

    pub(super) fn decode(self) -> Result<PositivePhysicalScale, Diagnostic> {
        PositivePhysicalScale::new(self.quantity.decode())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireSpace {
    ContinuousLagrange { order: u16 },
    SimplexP1Bubble,
    CellConstant,
}

impl WireSpace {
    pub(super) fn encode(value: Space) -> Self {
        match value.family() {
            SpaceFamily::ContinuousLagrange { order } => {
                Self::ContinuousLagrange { order: order.get() }
            }
            SpaceFamily::SimplexP1Bubble => Self::SimplexP1Bubble,
            SpaceFamily::CellConstant => Self::CellConstant,
        }
    }

    pub(super) fn decode(self) -> Result<Space, Diagnostic> {
        match self {
            Self::ContinuousLagrange { order } => Ok(Space::continuous_lagrange(
                NonZeroU16::new(order)
                    .ok_or_else(|| invalid_realization("portable graph space order is zero"))?,
            )),
            Self::SimplexP1Bubble => Ok(Space::simplex_p1_bubble()),
            Self::CellConstant => Ok(Space::cell_constant()),
        }
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
    GeneratedUniform {
        cells_per_axis: u64,
    },
    SuppliedCartesian {
        artifact_sha256: String,
        cells: Vec<u64>,
    },
    ImportedSimplicial {
        artifact_sha256: String,
    },
}

impl WireMesh {
    fn encode(value: MeshPolicy) -> Result<Self, Diagnostic> {
        match value {
            MeshPolicy::GeneratedUniform { cells_per_axis } => Ok(Self::GeneratedUniform {
                cells_per_axis: encode_usize(cells_per_axis.get(), "cells per axis")?,
            }),
            MeshPolicy::SuppliedCartesian { artifact, cells } => {
                Self::encode_supplied(artifact, &cells)
            }
            MeshPolicy::SuppliedCartesian1d { artifact, cells } => {
                Self::encode_supplied(artifact, &cells)
            }
            MeshPolicy::SuppliedCartesian3d { artifact, cells } => {
                Self::encode_supplied(artifact, &cells)
            }
            MeshPolicy::ImportedSimplicial { artifact } => Ok(Self::ImportedSimplicial {
                artifact_sha256: hex_bytes(&artifact.sha256()),
            }),
        }
    }

    fn decode(self) -> Result<MeshPolicy, Diagnostic> {
        match self {
            Self::GeneratedUniform { cells_per_axis } => Ok(MeshPolicy::GeneratedUniform {
                cells_per_axis: decode_nonzero_usize(cells_per_axis, "cells per axis")?,
            }),
            Self::SuppliedCartesian {
                artifact_sha256,
                cells,
            } => {
                let artifact = MeshArtifactReference::from_sha256(parse_sha256(&artifact_sha256)?);
                let cells = cells
                    .into_iter()
                    .enumerate()
                    .map(|(axis, count)| {
                        decode_nonzero_usize(count, &format!("Cartesian axis {axis} cells"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                match cells.as_slice() {
                    [x] => Ok(MeshPolicy::SuppliedCartesian1d {
                        artifact,
                        cells: [*x],
                    }),
                    [x, y] => Ok(MeshPolicy::SuppliedCartesian {
                        artifact,
                        cells: [*x, *y],
                    }),
                    [x, y, z] => Ok(MeshPolicy::SuppliedCartesian3d {
                        artifact,
                        cells: [*x, *y, *z],
                    }),
                    _ => Err(invalid_realization(
                        "supplied Cartesian mesh requires one to three cell counts",
                    )),
                }
            }
            Self::ImportedSimplicial { artifact_sha256 } => Ok(MeshPolicy::ImportedSimplicial {
                artifact: MeshArtifactReference::from_sha256(parse_sha256(&artifact_sha256)?),
            }),
        }
    }

    fn encode_supplied(
        artifact: MeshArtifactReference,
        cells: &[NonZeroUsize],
    ) -> Result<Self, Diagnostic> {
        Ok(Self::SuppliedCartesian {
            artifact_sha256: hex_bytes(&artifact.sha256()),
            cells: cells
                .iter()
                .enumerate()
                .map(|(axis, count)| {
                    encode_usize(count.get(), &format!("Cartesian axis {axis} cells"))
                })
                .collect::<Result<_, _>>()?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireQuadrature {
    GaussLegendre {
        points_per_axis: u64,
    },
    CellCentroid,
    SimplexCentroid,
    TriangleDuffyGaussLegendre {
        points_per_axis: u64,
    },
    SimplexDuffyGaussLegendre {
        spatial_dimension: u64,
        points_per_axis: u64,
    },
}

impl WireQuadrature {
    fn encode(value: QuadraturePolicy) -> Result<Self, Diagnostic> {
        match value {
            QuadraturePolicy::GaussLegendre { points_per_axis } => Ok(Self::GaussLegendre {
                points_per_axis: encode_usize(points_per_axis.get(), "quadrature points")?,
            }),
            QuadraturePolicy::CellCentroid => Ok(Self::CellCentroid),
            QuadraturePolicy::SimplexCentroid => Ok(Self::SimplexCentroid),
            QuadraturePolicy::TriangleDuffyGaussLegendre { points_per_axis } => {
                Ok(Self::TriangleDuffyGaussLegendre {
                    points_per_axis: encode_usize(points_per_axis.get(), "quadrature points")?,
                })
            }
            QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension,
                points_per_axis,
            } => Ok(Self::SimplexDuffyGaussLegendre {
                spatial_dimension: encode_usize(
                    spatial_dimension.get(),
                    "quadrature spatial dimension",
                )?,
                points_per_axis: encode_usize(points_per_axis.get(), "quadrature points")?,
            }),
        }
    }

    fn decode(self) -> Result<QuadraturePolicy, Diagnostic> {
        match self {
            Self::GaussLegendre { points_per_axis } => Ok(QuadraturePolicy::GaussLegendre {
                points_per_axis: decode_nonzero_usize(points_per_axis, "quadrature points")?,
            }),
            Self::CellCentroid => Ok(QuadraturePolicy::CellCentroid),
            Self::SimplexCentroid => Ok(QuadraturePolicy::SimplexCentroid),
            Self::TriangleDuffyGaussLegendre { points_per_axis } => {
                Ok(QuadraturePolicy::TriangleDuffyGaussLegendre {
                    points_per_axis: decode_nonzero_usize(points_per_axis, "quadrature points")?,
                })
            }
            Self::SimplexDuffyGaussLegendre {
                spatial_dimension,
                points_per_axis,
            } => Ok(QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension: decode_nonzero_usize(
                    spatial_dimension,
                    "quadrature spatial dimension",
                )?,
                points_per_axis: decode_nonzero_usize(points_per_axis, "quadrature points")?,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireDiscretization {
    method: WireDiscretizationMethod,
    mesh: WireMesh,
    quadrature: WireQuadrature,
}

impl WireDiscretization {
    pub(super) fn encode(value: Discretization) -> Result<Self, Diagnostic> {
        Ok(Self {
            method: WireDiscretizationMethod::encode(value.method()),
            mesh: WireMesh::encode(value.mesh())?,
            quadrature: WireQuadrature::encode(value.quadrature())?,
        })
    }

    pub(super) fn decode(self) -> Result<Discretization, Diagnostic> {
        Ok(Discretization::new(
            self.method.decode(),
            self.mesh.decode()?,
            self.quadrature.decode()?,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireLinearSolver {
    ConjugateGradient,
    MinimumResidual,
    BiConjugateGradientStabilized,
    SparseLu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WirePreconditioner {
    Identity,
    Jacobi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireReduction {
    Reproducible,
    Fast,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireSolverPlan {
    algorithm: WireLinearSolver,
    preconditioner: WirePreconditioner,
    reduction: WireReduction,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: u64,
}

impl WireSolverPlan {
    pub(super) fn encode(value: SolverPlan) -> Result<Self, Diagnostic> {
        Ok(Self {
            algorithm: match value.algorithm() {
                LinearSolver::ConjugateGradient => WireLinearSolver::ConjugateGradient,
                LinearSolver::MinimumResidual => WireLinearSolver::MinimumResidual,
                LinearSolver::BiConjugateGradientStabilized => {
                    WireLinearSolver::BiConjugateGradientStabilized
                }
                LinearSolver::SparseLu => WireLinearSolver::SparseLu,
            },
            preconditioner: match value.preconditioner() {
                PreconditionerPolicy::Identity => WirePreconditioner::Identity,
                PreconditionerPolicy::Jacobi => WirePreconditioner::Jacobi,
            },
            reduction: match value.reduction() {
                ReductionPolicy::Reproducible => WireReduction::Reproducible,
                ReductionPolicy::Fast => WireReduction::Fast,
            },
            relative_tolerance: normalize_zero(value.relative_tolerance()),
            absolute_tolerance: normalize_zero(value.absolute_tolerance()),
            maximum_iterations: encode_usize(
                value.maximum_iterations().get(),
                "linear maximum iterations",
            )?,
        })
    }

    pub(super) fn decode(self) -> Result<SolverPlan, Diagnostic> {
        let algorithm = match self.algorithm {
            WireLinearSolver::ConjugateGradient => LinearSolver::ConjugateGradient,
            WireLinearSolver::MinimumResidual => LinearSolver::MinimumResidual,
            WireLinearSolver::BiConjugateGradientStabilized => {
                LinearSolver::BiConjugateGradientStabilized
            }
            WireLinearSolver::SparseLu => LinearSolver::SparseLu,
        };
        let preconditioner = match self.preconditioner {
            WirePreconditioner::Identity => PreconditionerPolicy::Identity,
            WirePreconditioner::Jacobi => PreconditionerPolicy::Jacobi,
        };
        let reduction = match self.reduction {
            WireReduction::Reproducible => ReductionPolicy::Reproducible,
            WireReduction::Fast => ReductionPolicy::Fast,
        };
        Ok(SolverPlan::new(
            algorithm,
            self.relative_tolerance,
            self.absolute_tolerance,
            decode_nonzero_usize(self.maximum_iterations, "linear maximum iterations")?,
        )?
        .with_preconditioner(preconditioner)
        .with_reduction(reduction))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireOperatorProperties {
    General,
    SymmetricPositiveDefinite,
    SymmetricIndefinite,
    Symmetric,
    ComplexSymmetric,
    Hermitian,
    HermitianPositiveDefinite,
}

impl WireOperatorProperties {
    pub(super) const fn encode(value: LinearOperatorProperties) -> Self {
        match value {
            LinearOperatorProperties::General => Self::General,
            LinearOperatorProperties::SymmetricPositiveDefinite => Self::SymmetricPositiveDefinite,
            LinearOperatorProperties::SymmetricIndefinite => Self::SymmetricIndefinite,
            LinearOperatorProperties::Symmetric => Self::Symmetric,
            LinearOperatorProperties::ComplexSymmetric => Self::ComplexSymmetric,
            LinearOperatorProperties::Hermitian => Self::Hermitian,
            LinearOperatorProperties::HermitianPositiveDefinite => Self::HermitianPositiveDefinite,
        }
    }

    pub(super) const fn decode(self) -> LinearOperatorProperties {
        match self {
            Self::General => LinearOperatorProperties::General,
            Self::SymmetricPositiveDefinite => LinearOperatorProperties::SymmetricPositiveDefinite,
            Self::SymmetricIndefinite => LinearOperatorProperties::SymmetricIndefinite,
            Self::Symmetric => LinearOperatorProperties::Symmetric,
            Self::ComplexSymmetric => LinearOperatorProperties::ComplexSymmetric,
            Self::Hermitian => LinearOperatorProperties::Hermitian,
            Self::HermitianPositiveDefinite => LinearOperatorProperties::HermitianPositiveDefinite,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireScalarType {
    F32,
    F64,
}

impl WireScalarType {
    pub(super) const fn encode(value: ScalarType) -> Self {
        match value {
            ScalarType::F32 => Self::F32,
            ScalarType::F64 => Self::F64,
        }
    }

    pub(super) const fn decode(self) -> ScalarType {
        match self {
            Self::F32 => ScalarType::F32,
            Self::F64 => ScalarType::F64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireVectorLayout {
    Replicated,
    Distributed,
}

impl WireVectorLayout {
    pub(super) const fn encode(value: VectorLayoutKind) -> Self {
        match value {
            VectorLayoutKind::Replicated => Self::Replicated,
            VectorLayoutKind::Distributed => Self::Distributed,
        }
    }

    pub(super) const fn decode(self) -> VectorLayoutKind {
        match self {
            Self::Replicated => VectorLayoutKind::Replicated,
            Self::Distributed => VectorLayoutKind::Distributed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireSchedule {
    Offline,
    RealTime { priority: u16, deadline_ns: u64 },
}

impl WireSchedule {
    pub(super) const fn encode(value: ExecutionSchedule) -> Self {
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

    pub(super) fn decode(self) -> Result<ExecutionSchedule, Diagnostic> {
        match self {
            Self::Offline => Ok(ExecutionSchedule::Offline),
            Self::RealTime {
                priority,
                deadline_ns,
            } => Ok(ExecutionSchedule::RealTime {
                priority,
                deadline_ns: NonZeroU64::new(deadline_ns)
                    .ok_or_else(|| invalid_realization("real-time deadline is zero"))?,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireNonlinearPlan {
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: u64,
    maximum_line_search_steps: u64,
}

impl WireNonlinearPlan {
    pub(super) fn encode(value: NonlinearSolvePlan) -> Result<Self, Diagnostic> {
        Ok(Self {
            relative_tolerance: normalize_zero(value.relative_tolerance()),
            absolute_tolerance: normalize_zero(value.absolute_tolerance()),
            maximum_iterations: encode_usize(
                value.maximum_iterations().get(),
                "nonlinear maximum iterations",
            )?,
            maximum_line_search_steps: encode_usize(
                value.maximum_line_search_steps(),
                "maximum line-search steps",
            )?,
        })
    }

    pub(super) fn decode(self) -> Result<NonlinearSolvePlan, Diagnostic> {
        NonlinearSolvePlan::new(
            self.relative_tolerance,
            self.absolute_tolerance,
            decode_nonzero_usize(self.maximum_iterations, "nonlinear maximum iterations")?,
            decode_usize(self.maximum_line_search_steps, "maximum line-search steps")?,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireScaling {
    Dimensional,
    SymmetricCongruence {
        block_scales: Vec<WireBlockScale>,
        weak_functional_scale: WirePositiveScale,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireBlockScale {
    block: WireAlgebraicBlock,
    scale: WirePositiveScale,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireAlgebraicBlock {
    Field { field_ulid: String },
    ConstraintMultiplier { field_ulid: String },
}

impl WireAlgebraicBlock {
    pub(super) fn encode(value: AlgebraicBlock) -> Self {
        match value {
            AlgebraicBlock::Field(field) => Self::Field {
                field_ulid: field.ulid().to_string(),
            },
            AlgebraicBlock::ConstraintMultiplier { field } => Self::ConstraintMultiplier {
                field_ulid: field.ulid().to_string(),
            },
        }
    }

    pub(super) fn decode(self) -> Result<AlgebraicBlock, Diagnostic> {
        match self {
            Self::Field { field_ulid } => Ok(AlgebraicBlock::Field(parse_id(&field_ulid)?)),
            Self::ConstraintMultiplier { field_ulid } => Ok(AlgebraicBlock::ConstraintMultiplier {
                field: parse_id(&field_ulid)?,
            }),
        }
    }
}

impl WireScaling {
    pub(super) fn encode(value: &super::super::SystemScaling) -> Self {
        match value {
            super::super::SystemScaling::Dimensional => Self::Dimensional,
            super::super::SystemScaling::SymmetricCongruence(scaling) => {
                Self::SymmetricCongruence {
                    block_scales: scaling
                        .block_scales()
                        .iter()
                        .map(|entry| WireBlockScale {
                            block: WireAlgebraicBlock::encode(entry.block()),
                            scale: WirePositiveScale::encode(entry.scale()),
                        })
                        .collect(),
                    weak_functional_scale: WirePositiveScale::encode(
                        scaling.weak_functional_scale(),
                    ),
                }
            }
        }
    }

    pub(super) fn decode(self) -> Result<super::super::SystemScaling, Diagnostic> {
        match self {
            Self::Dimensional => Ok(super::super::SystemScaling::Dimensional),
            Self::SymmetricCongruence {
                block_scales,
                weak_functional_scale,
            } => Ok(super::super::SystemScaling::SymmetricCongruence(
                SymmetricCongruenceScaling::new(
                    block_scales
                        .into_iter()
                        .map(|entry| {
                            Ok(AlgebraicBlockScale::new(
                                entry.block.decode()?,
                                entry.scale.decode()?,
                            ))
                        })
                        .collect::<Result<Vec<_>, Diagnostic>>()?,
                    weak_functional_scale.decode()?,
                )?,
            )),
        }
    }
}

pub(super) fn encode_constraint(value: AlgebraicConstraint) -> WireAlgebraicBlock {
    match value {
        AlgebraicConstraint::ZeroIntegral { field } => WireAlgebraicBlock::ConstraintMultiplier {
            field_ulid: field.ulid().to_string(),
        },
    }
}

pub(super) fn decode_constraint(
    value: WireAlgebraicBlock,
) -> Result<AlgebraicConstraint, Diagnostic> {
    match value {
        WireAlgebraicBlock::ConstraintMultiplier { field_ulid } => {
            Ok(AlgebraicConstraint::ZeroIntegral {
                field: parse_id(&field_ulid)?,
            })
        }
        WireAlgebraicBlock::Field { .. } => Err(invalid_realization(
            "portable graph constraint wire contains a Field block",
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireConvectionScheme {
    ImplicitFirstOrderUpwind,
    ExplicitPreviousStateCartesianMinmod,
}

impl WireConvectionScheme {
    pub(super) const fn encode(value: CellCenteredConvectionScheme) -> Self {
        match value {
            CellCenteredConvectionScheme::ImplicitFirstOrderUpwind => {
                Self::ImplicitFirstOrderUpwind
            }
            CellCenteredConvectionScheme::ExplicitPreviousStateCartesianMinmod => {
                Self::ExplicitPreviousStateCartesianMinmod
            }
        }
    }

    pub(super) const fn decode(self) -> CellCenteredConvectionScheme {
        match self {
            Self::ImplicitFirstOrderUpwind => {
                CellCenteredConvectionScheme::ImplicitFirstOrderUpwind
            }
            Self::ExplicitPreviousStateCartesianMinmod => {
                CellCenteredConvectionScheme::ExplicitPreviousStateCartesianMinmod
            }
        }
    }
}

pub(super) fn encode_momentum_diagonal(value: PositiveMomentumDiagonal) -> &'static str {
    match value {
        PositiveMomentumDiagonal::BackwardEulerMassAndLocalNewtonian => {
            "backward-euler-mass-and-local-newtonian"
        }
    }
}

pub(super) fn decode_momentum_diagonal(
    value: &str,
) -> Result<PositiveMomentumDiagonal, Diagnostic> {
    match value {
        "backward-euler-mass-and-local-newtonian" => {
            Ok(PositiveMomentumDiagonal::BackwardEulerMassAndLocalNewtonian)
        }
        _ => Err(invalid_realization(
            "portable graph has an unknown momentum-diagonal policy",
        )),
    }
}

pub(super) fn encode_face_history(value: TransientFaceFluxHistory) -> &'static str {
    match value {
        TransientFaceFluxHistory::Bdf1PreviousAccepted => "bdf1-previous-accepted",
    }
}

pub(super) fn decode_face_history(value: &str) -> Result<TransientFaceFluxHistory, Diagnostic> {
    match value {
        "bdf1-previous-accepted" => Ok(TransientFaceFluxHistory::Bdf1PreviousAccepted),
        _ => Err(invalid_realization(
            "portable graph has an unknown transient face-history policy",
        )),
    }
}

pub(super) fn parse_id<E: Entity>(value: &str) -> Result<Id<E>, Diagnostic> {
    value
        .parse::<Ulid>()
        .map(Id::from_ulid)
        .map_err(|_| invalid_realization("portable graph contains an invalid ULID"))
}

pub(super) fn encode_usize(value: usize, label: &str) -> Result<u64, Diagnostic> {
    u64::try_from(value)
        .map_err(|_| invalid_realization(format!("portable graph {label} exceeds u64")))
}

pub(super) fn decode_usize(value: u64, label: &str) -> Result<usize, Diagnostic> {
    usize::try_from(value)
        .map_err(|_| invalid_realization(format!("portable graph {label} exceeds usize")))
}

pub(super) fn decode_nonzero_usize(value: u64, label: &str) -> Result<NonZeroUsize, Diagnostic> {
    NonZeroUsize::new(decode_usize(value, label)?)
        .ok_or_else(|| invalid_realization(format!("portable graph {label} is zero")))
}

fn hex_bytes(value: &[u8; 32]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn parse_sha256(value: &str) -> Result<[u8; 32], Diagnostic> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid_realization(
            "portable graph contains an invalid SHA-256 identity",
        ));
    }
    let mut result = [0_u8; 32];
    for (index, byte) in result.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).map_err(|_| {
            invalid_realization("portable graph contains an invalid SHA-256 identity")
        })?;
    }
    Ok(result)
}

pub(super) const fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> MeshArtifactReference {
        MeshArtifactReference::from_sha256([0xab; 32])
    }

    #[test]
    fn supplied_cartesian_wire_round_trips_one_to_three_dimensions() {
        let policies = [
            MeshPolicy::SuppliedCartesian1d {
                artifact: artifact(),
                cells: [NonZeroUsize::new(7).unwrap()],
            },
            MeshPolicy::SuppliedCartesian {
                artifact: artifact(),
                cells: [
                    NonZeroUsize::new(7).unwrap(),
                    NonZeroUsize::new(11).unwrap(),
                ],
            },
            MeshPolicy::SuppliedCartesian3d {
                artifact: artifact(),
                cells: [
                    NonZeroUsize::new(7).unwrap(),
                    NonZeroUsize::new(11).unwrap(),
                    NonZeroUsize::new(13).unwrap(),
                ],
            },
        ];

        for policy in policies {
            assert_eq!(WireMesh::encode(policy).unwrap().decode().unwrap(), policy);
        }
    }

    #[test]
    fn supplied_cartesian_two_dimensional_wire_remains_byte_identical() {
        let wire = WireMesh::encode(MeshPolicy::SuppliedCartesian {
            artifact: artifact(),
            cells: [
                NonZeroUsize::new(7).unwrap(),
                NonZeroUsize::new(11).unwrap(),
            ],
        })
        .unwrap();

        assert_eq!(
            serde_json::to_vec(&wire).unwrap(),
            format!(
                "{{\"kind\":\"supplied-cartesian\",\"artifact_sha256\":\"{}\",\"cells\":[7,11]}}",
                "ab".repeat(32)
            )
            .into_bytes()
        );
    }

    #[test]
    fn quantity_wire_requires_canonical_rational_dimensions() {
        let dimension =
            DimExponents::from_rationals([(0, 1), (-1, 2), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)])
                .unwrap();
        let quantity = DynQuantity::new(2.0, dimension);
        let wire = serde_json::to_value(WireQuantity::encode(quantity)).unwrap();
        assert_eq!(
            serde_json::from_value::<WireQuantity>(wire.clone())
                .unwrap()
                .decode(),
            quantity
        );
        for pair in [(1, 0), (2, 4), (1, -2), (0, 2), (i32::MIN, 1)] {
            let mut invalid = wire.clone();
            invalid["dimension"][1] = serde_json::json!(pair);
            assert!(serde_json::from_value::<WireQuantity>(invalid).is_err());
        }
        let mut old = wire;
        old["dimension"] = serde_json::json!([0, -1, 0, 0, 0, 0, 0]);
        assert!(serde_json::from_value::<WireQuantity>(old).is_err());
    }

    #[test]
    fn supplied_cartesian_wire_rejects_invalid_dimensions_and_zero_cells() {
        for cells in [vec![], vec![1, 2, 3, 4], vec![1, 0]] {
            let wire = WireMesh::SuppliedCartesian {
                artifact_sha256: "ab".repeat(32),
                cells,
            };
            assert_eq!(
                wire.decode().unwrap_err().code(),
                eqiora_core::diagnostic::codes::INVALID_REALIZATION
            );
        }
    }
}
