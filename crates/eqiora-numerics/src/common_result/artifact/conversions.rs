//! Exact wire conversions for common Result artifacts.

use eqiora_core::{Diagnostic, DimExponents};
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

pub(super) fn dimension_to_wire(value: DimExponents) -> [(i32, i32); 7] {
    value.exponents()
}

pub(super) fn dimension_from_wire(value: [(i32, i32); 7]) -> Result<DimExponents, Diagnostic> {
    let dimension = DimExponents::from_rationals(value)
        .ok_or_else(|| invalid("invalid rational Result dimension"))?;
    if dimension.exponents() != value {
        return Err(invalid(
            "Result dimension must contain canonical reduced fractions",
        ));
    }
    Ok(dimension)
}

impl From<CommonResultFamily> for WireResultFamily {
    fn from(value: CommonResultFamily) -> Self {
        match value {
            CommonResultFamily::Eigen => Self::Eigen,
            CommonResultFamily::Algebraic => Self::Algebraic,
            CommonResultFamily::Linear => Self::Linear,
            CommonResultFamily::SteadyStokes => Self::SteadyStokes,
            CommonResultFamily::Ode => Self::Ode,
            CommonResultFamily::TransientFlow => Self::TransientFlow,
            CommonResultFamily::FixedReferenceFsi => Self::FixedReferenceFsi,
        }
    }
}

impl From<WireResultFamily> for CommonResultFamily {
    fn from(value: WireResultFamily) -> Self {
        match value {
            WireResultFamily::Eigen => Self::Eigen,
            WireResultFamily::Algebraic => Self::Algebraic,
            WireResultFamily::Linear => Self::Linear,
            WireResultFamily::SteadyStokes => Self::SteadyStokes,
            WireResultFamily::Ode => Self::Ode,
            WireResultFamily::TransientFlow => Self::TransientFlow,
            WireResultFamily::FixedReferenceFsi => Self::FixedReferenceFsi,
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

#[cfg(test)]
mod dimension_tests {
    use super::{dimension_from_wire, dimension_to_wire};
    use eqiora_core::DimExponents;

    #[test]
    fn result_dimensions_use_exact_si_order_and_reject_noncanonical_input() {
        let fractions = [(1, 3), (-1, 2), (3, 4), (0, 1), (0, 1), (0, 1), (0, 1)];
        let dimension = DimExponents::from_rationals(fractions).unwrap();
        assert_eq!(dimension_to_wire(dimension), fractions);
        assert_eq!(dimension_from_wire(fractions).unwrap(), dimension);
        for pair in [(1, 0), (2, 4), (1, -2), (0, 2), (i32::MIN, 1)] {
            let mut invalid = fractions;
            invalid[1] = pair;
            assert!(dimension_from_wire(invalid).is_err());
        }
    }
}
