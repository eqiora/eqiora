//! Native finite algebra uses the ordinary source call and reference owners.
use super::DraftExpression;

impl DraftExpression {
    /// Algebraic dual or map transpose, without conjugating coefficients.
    #[must_use]
    pub fn transpose(self) -> Self {
        Self::call("transpose", vec![self])
    }
    /// Conjugate transpose in the declared orthonormal finite bases.
    #[must_use]
    pub fn adjoint(self) -> Self {
        Self::call("adjoint", vec![self])
    }
    /// Apply this map to coordinates in its exact input basis.
    #[must_use]
    pub fn apply(self, coordinates: Self) -> Self {
        Self::call("apply", vec![self, coordinates])
    }
    /// Compose this map after `right`; equal extents do not imply equal bases.
    #[must_use]
    pub fn compose_map(self, right: Self) -> Self {
        Self::call("compose", vec![self, right])
    }
    /// Bilinear pairing of dual coordinates with the matching primal basis.
    #[must_use]
    pub fn pair(self, right: Self) -> Self {
        Self::call("pair", vec![self, right])
    }
}
