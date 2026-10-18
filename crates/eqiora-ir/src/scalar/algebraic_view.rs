//! Borrowed proof views preserve the executable SSA and its demanded domains.
use super::*;
use std::hash::{Hash, Hasher};

/// One borrowed scalar expression. Equality and hashing identify an SSA node
/// within its program; use [`Self::same_expression`] for structural comparison.
#[derive(Debug, Clone, Copy)]
pub struct ScalarExpressionView<'a> {
    program: &'a ScalarInputOperatorIr,
    root: ValueId,
}

impl PartialEq for ScalarExpressionView<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.program, other.program) && self.root == other.root
    }
}
impl Eq for ScalarExpressionView<'_> {}
impl Hash for ScalarExpressionView<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::from_ref(self.program).hash(state);
        self.root.0.hash(state);
    }
}

/// Algebra visible to symbolic coefficient proofs. Other operations stay
/// opaque; this view never authorizes replacing ordinary value/domain checks.
#[derive(Debug, Clone, Copy)]
pub enum ScalarAlgebraicNode<'a> {
    /// A literal, never a sampled input value.
    Constant(f64),
    /// An exact retained input coordinate.
    Input(&'a ScalarInputSlot),
    /// Ordered negation.
    Neg(ScalarExpressionView<'a>),
    /// Ordered addition.
    Add(ScalarExpressionView<'a>, ScalarExpressionView<'a>),
    /// Ordered subtraction.
    Sub(ScalarExpressionView<'a>, ScalarExpressionView<'a>),
    /// Ordered multiplication.
    Mul(ScalarExpressionView<'a>, ScalarExpressionView<'a>),
    /// Ordered division, with the denominator's domain retained by execution.
    Div(ScalarExpressionView<'a>, ScalarExpressionView<'a>),
    /// A primitive or guarded expression, not an algebraic rewrite permission.
    Opaque,
}

impl ScalarInputOperatorIr {
    pub(crate) fn expression_view(&self) -> ScalarExpressionView<'_> {
        ScalarExpressionView {
            program: self,
            root: self.roots[0],
        }
    }
}

impl<'a> ScalarExpressionView<'a> {
    /// Inspect the current node without copying or evaluating its program.
    #[must_use]
    pub fn node(self) -> ScalarAlgebraicNode<'a> {
        use ScalarAlgebraicNode as Node;
        let at = |root| Self { root, ..self };
        match self.program.instructions[self.root.0 as usize] {
            Instruction::Constant(value) => Node::Constant(value.value()),
            Instruction::Read(slot) => Node::Input(&self.program.slots[slot.0 as usize]),
            Instruction::Neg(a) => Node::Neg(at(a)),
            Instruction::Add(a, b) => Node::Add(at(a), at(b)),
            Instruction::Sub(a, b) => Node::Sub(at(a), at(b)),
            Instruction::Mul(a, b) => Node::Mul(at(a), at(b)),
            Instruction::Div(a, b) => Node::Div(at(a), at(b)),
            _ => Node::Opaque,
        }
    }

    /// Compare exact operation structure with caller-owned input equivalence.
    /// Guards and primitive operands remain part of the comparison. Unsupported
    /// structure or proof work exceeding the scalar budget returns false.
    #[must_use]
    pub fn same_expression(
        self,
        other: ScalarExpressionView<'_>,
        mut same_input: impl FnMut(&ScalarInputSlot, &ScalarInputSlot) -> bool,
    ) -> bool {
        let mut pending = vec![(self.root, other.root)];
        let mut visited = std::collections::HashSet::new();
        while let Some((a, b)) = pending.pop() {
            if !visited.insert((a.0, b.0)) {
                continue;
            }
            if visited.len() > MAX_COMPONENT_PRODUCTS {
                return false;
            }
            let (left, right) = (
                self.program.instructions[a.0 as usize],
                other.program.instructions[b.0 as usize],
            );
            if let (Instruction::Read(a), Instruction::Read(b)) = (left, right) {
                if !same_input(
                    &self.program.slots[a.0 as usize],
                    &other.program.slots[b.0 as usize],
                ) {
                    return false;
                }
                continue;
            }
            let (Some((left, a)), Some((right, b))) = (structure(left), structure(right)) else {
                return false;
            };
            if left != right || a.len() != b.len() {
                return false;
            }
            pending.extend(a.into_iter().zip(b));
        }
        true
    }
}

fn structure(instruction: Instruction) -> Option<(Instruction, Vec<ValueId>)> {
    let mut operands = Vec::new();
    let instruction = match instruction {
        Instruction::MapInvariant {
            start,
            extent,
            output,
        } => {
            operands
                .extend((start.0..start.0.checked_add(extent.checked_mul(extent)?)?).map(ValueId));
            Instruction::MapInvariant {
                start: ValueId(0),
                extent,
                output,
            }
        }
        Instruction::CoefficientJet { start, len, shape } => {
            operands.extend((start.0..start.0.checked_add(len.checked_mul(4)?)?).map(ValueId));
            Instruction::CoefficientJet {
                start: ValueId(0),
                len,
                shape,
            }
        }
        other => other.map_scalar_operands(|id| {
            operands.push(id);
            ValueId(0)
        })?,
    };
    Some((instruction, operands))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, DynQuantity, Id, ScalarDomain, ValueType};
    use eqiora_schema::kernel::ComparisonOp;

    #[test]
    fn proof_views_ignore_ssa_numbering_but_retain_guards_and_input_identity() {
        let unit = DimExponents::DIMENSIONLESS;
        let ty = ValueType::scalar(ScalarDomain::Real, unit).unwrap();
        let p = SymbolRef::Parameter(Id::new());
        let q = SymbolRef::Parameter(Id::new());
        let build = |symbol, lower, unused| {
            let mut builder = ScalarInputIrBuilder::new();
            if unused {
                builder.constant(DynQuantity::new(17., unit)).unwrap();
            }
            let coordinate = ScalarSymbolCoordinate::for_value(symbol, &ty)
                .unwrap()
                .remove(0);
            let value = builder
                .input(ScalarInputSlot::new(0, coordinate, unit))
                .unwrap();
            let bound = builder.constant(DynQuantity::new(lower, unit)).unwrap();
            let condition = builder
                .compare(ComparisonOp::Greater, value, bound)
                .unwrap();
            let guarded = builder.require(condition, value).unwrap();
            builder.finish([guarded]).unwrap()
        };
        let original = build(p, 0., false);
        let renumbered = build(p, 0., true);
        let different_domain = build(p, 1., false);
        let different_input = build(q, 0., false);
        let same_input = |a: &ScalarInputSlot, b: &ScalarInputSlot| a.source() == b.source();
        let view = original.expression_view();
        assert!(matches!(view.node(), ScalarAlgebraicNode::Opaque));
        assert!(view.same_expression(renumbered.expression_view(), same_input));
        assert!(!view.same_expression(different_domain.expression_view(), same_input));
        assert!(!view.same_expression(different_input.expression_view(), same_input));
        assert!(original.evaluate(&[-1.]).is_err());
        assert_eq!(original.evaluate(&[2.]).unwrap(), [2.]);
    }
}
