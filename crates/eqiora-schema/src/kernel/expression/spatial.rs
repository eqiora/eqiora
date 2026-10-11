//! Construct physical-space differential and boundary expressions.
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

    /// Take a full trace on a boundary or physical-interface Domain.
    /// The operand's support selects the adjacent parent.
    pub fn trace(&mut self, value: ExprId, on: Id<kinds::Domain>) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Trace { value, on })
    }

    /// Contract with the normal of a boundary or physical-interface Domain.
    /// Exterior boundaries use the parent-outward normal; physical interfaces
    /// use the common normal of their first declared boundary.
    pub fn normal_component(
        &mut self,
        value: ExprId,
        on: Id<kinds::Domain>,
    ) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::NormalComponent { value, on })
    }

    /// Pull a value back along a map between coordinate supports, before any
    /// numerical sampling. Type admission requires every coordinate to be bound,
    /// each with its own unit.
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
    ///
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

    /// Apply a dimensioned direction, one per mapped row, to a Jacobian factor.
    /// The directions are not assumed to be time derivatives; a conservation
    /// correspondence has to establish that separately.
    ///
    /// # Errors
    /// Rejects a non-factor value, a wrong row count, or unavailable operands.
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
