//! Construct physical-space differential and exact boundary expressions.
use super::{Diagnostic, ExprDagBuilder, ExprId, ExprNode, Id, kinds};

impl ExprDagBuilder {
    /// Take the physical-space gradient.
    pub fn gradient(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Gradient(value))
    }

    /// Take the physical-space divergence.
    pub fn divergence(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Divergence(value))
    }

    /// Take the symmetric part of a square Cartesian rank-two tensor.
    pub fn symmetric_part(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::SymmetricPart(value))
    }

    /// Lift a supported invariant scalar to an isotropic Cartesian tensor.
    pub fn isotropic_lift(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::IsotropicLift(value))
    }

    /// Restrict an expression to one exact boundary Domain.
    pub fn trace(&mut self, value: ExprId, on: Id<kinds::Domain>) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Trace { value, on })
    }

    /// Take the outward-normal component on one exact boundary Domain.
    pub fn normal_component(
        &mut self,
        value: ExprId,
        on: Id<kinds::Domain>,
    ) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::NormalComponent { value, on })
    }
}
