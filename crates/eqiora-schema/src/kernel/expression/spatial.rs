//! Construct physical-space differential and exact boundary expressions.
use super::{CoordinateMapFactor, Diagnostic, ExprDagBuilder, ExprId, ExprNode, Id, codes, kinds};

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

    /// Take a full trace on one exact boundary or physical-interface Domain.
    /// The operand's support selects its adjacent parent.
    pub fn trace(&mut self, value: ExprId, on: Id<kinds::Domain>) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Trace { value, on })
    }

    /// Contract with the normal of one exact boundary or physical-interface Domain.
    /// Exterior boundaries use the parent-outward normal; physical interfaces
    /// use the common normal of their first declared boundary.
    pub fn normal_component(
        &mut self,
        value: ExprId,
        on: Id<kinds::Domain>,
    ) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::NormalComponent { value, on })
    }

    /// Retain a map between exact coordinate supports, before numerical sampling.
    /// Type admission proves complete inventories and each coordinate's own unit.
    pub fn pullback(
        &mut self,
        value: ExprId,
        source: Vec<ExprId>,
        at: Vec<(ExprId, ExprId)>,
    ) -> Result<ExprId, Diagnostic> {
        if source.is_empty() || at.is_empty() {
            return Err(Diagnostic::error(
                codes::INVALID_EXPRESSION_DAG,
                "coordinate pullback requires nonempty source and target coordinates",
            ));
        }
        self.push(ExprNode::Pullback { value, source, at })
    }

    /// Derive a local Jacobian factor from a complete coordinate map.
    /// # Errors
    /// Rejects empty inventories and operands outside this expression arena.
    pub fn coordinate_map_factor(
        &mut self,
        factor: CoordinateMapFactor,
        source: Vec<ExprId>,
        at: Vec<(ExprId, ExprId)>,
    ) -> Result<ExprId, Diagnostic> {
        if source.is_empty() || at.is_empty() {
            return Err(Diagnostic::error(
                codes::INVALID_EXPRESSION_DAG,
                "coordinate map factor requires nonempty coordinate inventories",
            ));
        }
        self.push(ExprNode::CoordinateMapFactor { factor, source, at })
    }

    /// Apply a dimensioned mapped-row direction to an exact Jacobian factor.
    /// This is a directional action, not an assertion that the directions are
    /// time derivatives. A conservation correspondence must establish that fact.
    /// # Errors
    /// Rejects a non-factor value, wrong row count, or unavailable operands.
    pub fn coordinate_map_factor_action(
        &mut self,
        value: ExprId,
        parameter: ExprId,
        directions: Vec<ExprId>,
    ) -> Result<ExprId, Diagnostic> {
        if !matches!(self.nodes.get(value.index() as usize),
            Some(ExprNode::CoordinateMapFactor { at, .. }) if at.len() == directions.len())
        {
            return Err(Diagnostic::error(
                codes::INVALID_EXPRESSION_DAG,
                "coordinate-map action requires its exact factor and one direction per mapped row",
            ));
        }
        self.push(ExprNode::CoordinateMapFactorAction {
            value,
            parameter,
            directions,
        })
    }
}
