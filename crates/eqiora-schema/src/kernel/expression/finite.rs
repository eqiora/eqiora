use super::{ExprDagBuilder, ExprId, ExprNode};
use eqiora_core::Diagnostic;

/// Closed finite-basis unary algebra; storage and solver choices are separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FiniteUnaryOperation {
    /// Algebraic dual/transpose without complex conjugation.
    Transpose,
    /// Algebraic trace on one exact real or complex endomorphism space.
    MatrixTrace,
    /// Determinant of a real endomorphism, with coefficient units raised to its extent.
    Determinant,
    /// Real inverse map under invertibility, exchanging exact endpoints and coefficient units.
    Inverse,
    /// Conjugate transpose in the declared orthonormal component bases.
    Adjoint,
    /// Explicit ordering of the two tensor factors, applied to both map endpoints.
    PermuteFactors([u8; 2]),
}

/// Closed finite-basis binary algebra with exact nominal endpoint matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FiniteBinaryOperation {
    /// Apply a linear map to coordinates in its exact input basis.
    Apply,
    /// Compose left after right, retaining the ordered outer endpoints.
    Compose,
    /// Bilinear evaluation of a dual coordinate and its exact primal coordinate.
    Pair,
    /// Ordered two-factor tensor product, with the right factor varying fastest.
    TensorProduct,
}

impl ExprDagBuilder {
    /// Retain an explicit transpose or adjoint for shared semantic type admission.
    pub fn finite_unary(
        &mut self,
        operation: FiniteUnaryOperation,
        value: ExprId,
    ) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::FiniteUnary(operation, value))
    }
    /// Retain an explicit finite operation; actual endpoint types are checked by inference.
    pub fn finite_binary(
        &mut self,
        operation: FiniteBinaryOperation,
        left: ExprId,
        right: ExprId,
    ) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::FiniteBinary(operation, left, right))
    }
}
