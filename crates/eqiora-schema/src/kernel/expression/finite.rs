use super::{ExprDagBuilder, ExprId, ExprNode};
use eqiora_core::Diagnostic;

/// Unary operations on finite-basis values; storage and solver choices are separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FiniteUnaryOperation {
    /// Algebraic dual/transpose without complex conjugation.
    Transpose,
    /// Algebraic trace of a real or complex endomorphism.
    MatrixTrace,
    /// Determinant of a real endomorphism, with coefficient units raised to its extent.
    Determinant,
    /// Real inverse of an invertible map, exchanging its endpoints and coefficient units.
    Inverse,
    /// Conjugate transpose in the declared orthonormal component bases.
    Adjoint,
    /// Explicit ordering of the two tensor factors, applied to both map endpoints.
    PermuteFactors([u8; 2]),
}

/// Binary operations on finite-basis values; endpoint bases must match by declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FiniteBinaryOperation {
    /// Apply a linear map to coordinates in its input basis.
    Apply,
    /// Compose left after right, retaining the ordered outer endpoints.
    Compose,
    /// Bilinear evaluation of a dual coordinate and the matching primal coordinate.
    Pair,
    /// Ordered two-factor tensor product, with the right factor varying fastest.
    TensorProduct,
}

impl ExprDagBuilder {
    /// Add a finite unary operation such as a transpose or adjoint; types are checked
    /// during semantic admission.
    pub fn finite_unary(
        &mut self,
        operation: FiniteUnaryOperation,
        value: ExprId,
    ) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::FiniteUnary(operation, value))
    }
    /// Add a finite binary operation; endpoint types are checked during inference.
    pub fn finite_binary(
        &mut self,
        operation: FiniteBinaryOperation,
        left: ExprId,
        right: ExprId,
    ) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::FiniteBinary(operation, left, right))
    }
}
