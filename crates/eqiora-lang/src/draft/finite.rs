//! Native finite algebra uses the ordinary source call and reference owners.
use super::DraftExpression;

impl DraftExpression {
    /// Algebraic diagonal sum on an exact finite endomorphism or spatial tensor.
    #[must_use]
    pub fn matrix_trace(self) -> Self {
        Self::call("matrix_trace", vec![self])
    }
    /// Determinant of an admitted real finite endomorphism.
    #[must_use]
    pub fn determinant(self) -> Self {
        Self::call("determinant", vec![self])
    }
    /// Inverse real finite map under the execution owner's regularity policy.
    #[must_use]
    pub fn inverse(self) -> Self {
        Self::call("inverse", vec![self])
    }

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
    /// Ordered tensor product; the right factor varies fastest.
    #[must_use]
    pub fn tensor_product(self, right: Self) -> Self {
        Self::call("tensor_product", vec![self, right])
    }
    /// Explicitly reorder the two factors of coordinates or both map endpoints.
    #[must_use]
    pub fn permute_factors(self, order: [u8; 2]) -> Self {
        Self::call(
            "permute_factors",
            vec![
                self,
                Self::array(order.map(|index| {
                    Self::constant(
                        crate::DecimalLiteral::parse(&index.to_string()).expect("factor index"),
                    )
                })),
            ],
        )
    }
}
