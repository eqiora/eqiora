use super::{ExprDagBuilder, ExprId, ExprNode};
use eqiora_core::Diagnostic;

/// Closed finite-basis unary algebra; storage and solver choices are separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FiniteUnaryOperation {
    /// Algebraic dual/transpose without complex conjugation.
    Transpose,
    /// Conjugate transpose in the declared orthonormal component bases.
    Adjoint,
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
