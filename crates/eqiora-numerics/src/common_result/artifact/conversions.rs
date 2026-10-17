//! Exact wire conversions for common Result artifacts.

use eqiora_core::Diagnostic;
use eqiora_solver::{
    ConvergenceReason, LinearOperatorOrientation, LinearSolver, PreconditionerPolicy,
    ReductionPolicy,
};

use super::{
    CommonFieldAssociation, CommonResultFamily, WireAlgorithm, WireAssociation,
    WireConvergenceReason, WireOrientation, WirePreconditioner, WireReduction, WireResultFamily,
    invalid,
};

pub(super) fn encode_shape(shape: &[usize]) -> Result<Vec<u64>, Diagnostic> {
    shape
        .iter()
        .map(|extent| to_u64(*extent, "Result shape extent"))
        .collect()
}

pub(super) fn decode_shape(shape: &[u64]) -> Result<Vec<usize>, Diagnostic> {
    shape
        .iter()
        .map(|extent| positive_usize(*extent, "Result shape extent"))
        .collect()
}

pub(super) fn to_u64(value: usize, label: &str) -> Result<u64, Diagnostic> {
    u64::try_from(value).map_err(|_| invalid(format!("{label} exceeds u64")))
}

pub(super) fn to_usize(value: u64, label: &str) -> Result<usize, Diagnostic> {
    usize::try_from(value).map_err(|_| invalid(format!("{label} exceeds usize")))
}

pub(super) fn positive_usize(value: u64, label: &str) -> Result<usize, Diagnostic> {
    let value = to_usize(value, label)?;
    if value == 0 {
        return Err(invalid(format!("{label} must be positive")));
    }
    Ok(value)
}

impl From<CommonResultFamily> for WireResultFamily {
    fn from(value: CommonResultFamily) -> Self {
        match value {
            CommonResultFamily::Eigen => Self::Eigen,
            CommonResultFamily::Algebraic => Self::Algebraic,
            CommonResultFamily::Spatial => Self::Spatial,
            CommonResultFamily::SteadyStokes => Self::SteadyStokes,
            CommonResultFamily::Time => Self::Time,
            CommonResultFamily::TransientFlow => Self::TransientFlow,
        }
    }
}

impl From<WireResultFamily> for CommonResultFamily {
    fn from(value: WireResultFamily) -> Self {
        match value {
            WireResultFamily::Eigen => Self::Eigen,
            WireResultFamily::Algebraic => Self::Algebraic,
            WireResultFamily::Spatial => Self::Spatial,
            WireResultFamily::SteadyStokes => Self::SteadyStokes,
            WireResultFamily::Time => Self::Time,
            WireResultFamily::TransientFlow => Self::TransientFlow,
        }
    }
}

impl From<CommonFieldAssociation> for WireAssociation {
    fn from(value: CommonFieldAssociation) -> Self {
        match value {
            CommonFieldAssociation::Edge => Self::Edge,
            CommonFieldAssociation::Face => Self::Face,
            CommonFieldAssociation::Vertex => Self::Vertex,
            CommonFieldAssociation::Cell => Self::Cell,
            CommonFieldAssociation::CellBubble => Self::CellBubble,
        }
    }
}

impl From<WireAssociation> for CommonFieldAssociation {
    fn from(value: WireAssociation) -> Self {
        match value {
            WireAssociation::Edge => Self::Edge,
            WireAssociation::Face => Self::Face,
            WireAssociation::Vertex => Self::Vertex,
            WireAssociation::Cell => Self::Cell,
            WireAssociation::CellBubble => Self::CellBubble,
        }
    }
}

impl From<LinearOperatorOrientation> for WireOrientation {
    fn from(value: LinearOperatorOrientation) -> Self {
        match value {
            LinearOperatorOrientation::Normal => Self::Normal,
            LinearOperatorOrientation::Transposed => Self::Transposed,
            LinearOperatorOrientation::ConjugateTransposed => Self::ConjugateTransposed,
        }
    }
}
impl From<WireOrientation> for LinearOperatorOrientation {
    fn from(value: WireOrientation) -> Self {
        match value {
            WireOrientation::Normal => Self::Normal,
            WireOrientation::Transposed => Self::Transposed,
            WireOrientation::ConjugateTransposed => Self::ConjugateTransposed,
        }
    }
}
impl From<LinearSolver> for WireAlgorithm {
    fn from(value: LinearSolver) -> Self {
        match value {
            LinearSolver::ConjugateGradient => Self::ConjugateGradient,
            LinearSolver::MinimumResidual => Self::MinimumResidual,
            LinearSolver::BiConjugateGradientStabilized => Self::Bicgstab,
            LinearSolver::SparseLu => Self::SparseLu,
        }
    }
}
impl From<WireAlgorithm> for LinearSolver {
    fn from(value: WireAlgorithm) -> Self {
        match value {
            WireAlgorithm::ConjugateGradient => Self::ConjugateGradient,
            WireAlgorithm::MinimumResidual => Self::MinimumResidual,
            WireAlgorithm::Bicgstab => Self::BiConjugateGradientStabilized,
            WireAlgorithm::SparseLu => Self::SparseLu,
        }
    }
}
impl From<PreconditionerPolicy> for WirePreconditioner {
    fn from(value: PreconditionerPolicy) -> Self {
        match value {
            PreconditionerPolicy::Identity => Self::Identity,
            PreconditionerPolicy::Jacobi => Self::Jacobi,
        }
    }
}
impl From<WirePreconditioner> for PreconditionerPolicy {
    fn from(value: WirePreconditioner) -> Self {
        match value {
            WirePreconditioner::Identity => Self::Identity,
            WirePreconditioner::Jacobi => Self::Jacobi,
        }
    }
}
impl From<ReductionPolicy> for WireReduction {
    fn from(value: ReductionPolicy) -> Self {
        match value {
            ReductionPolicy::Reproducible => Self::Reproducible,
            ReductionPolicy::Fast => Self::Fast,
        }
    }
}
impl From<WireReduction> for ReductionPolicy {
    fn from(value: WireReduction) -> Self {
        match value {
            WireReduction::Reproducible => Self::Reproducible,
            WireReduction::Fast => Self::Fast,
        }
    }
}
impl From<ConvergenceReason> for WireConvergenceReason {
    fn from(value: ConvergenceReason) -> Self {
        match value {
            ConvergenceReason::InitialResidualSatisfied => Self::InitialResidualSatisfied,
            ConvergenceReason::ResidualToleranceSatisfied => Self::ResidualToleranceSatisfied,
        }
    }
}
impl From<WireConvergenceReason> for ConvergenceReason {
    fn from(value: WireConvergenceReason) -> Self {
        match value {
            WireConvergenceReason::InitialResidualSatisfied => Self::InitialResidualSatisfied,
            WireConvergenceReason::ResidualToleranceSatisfied => Self::ResidualToleranceSatisfied,
        }
    }
}
