//! Method-specific scalar Cartesian solution reconstruction.
use super::*;

/// Method-specific Cartesian result selected from one typed Realization plan.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedScalarEllipticCartesianSolution {
    /// Continuous Q1 Galerkin finite-element result.
    FiniteElement(ScalarEllipticCartesianFemSolution),
    /// Cell-centred orthogonal two-point-flux finite-volume result.
    FiniteVolume(ScalarEllipticCartesianFvmSolution),
}

impl ResolvedScalarEllipticCartesianSolution {
    pub(crate) fn into_point(self) -> crate::cartesian_elliptic::CartesianPoint {
        use crate::cartesian_elliptic::CartesianPoint;
        match self {
            Self::FiniteElement(solution) => CartesianPoint::FiniteElement(solution.into_point()),
            Self::FiniteVolume(solution) => CartesianPoint::FiniteVolume(solution.into_point()),
        }
    }

    /// Consume the method-native solution into its complete primary Field.
    #[must_use]
    pub fn into_primary_field_values(self) -> Vec<f64> {
        match self {
            Self::FiniteElement(solution) => solution.into_field_values(),
            Self::FiniteVolume(solution) => solution.into_field_values(),
        }
    }
}
