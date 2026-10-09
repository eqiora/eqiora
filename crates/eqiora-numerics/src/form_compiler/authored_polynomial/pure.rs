//! One component projection for checked Model and authored-form pure calculus.
use super::*;
use eqiora_schema::kernel::pure_operator::{CalculusNode, PureOperatorDefinition};

impl Context<'_> {
    pub(super) fn pure_component(
        &mut self,
        definition: &PureOperatorDefinition,
        coordinate: &[usize],
        depth: usize,
        mut component: impl FnMut(&mut Self, u16, &[usize], usize) -> Option<Polynomial>,
    ) -> Option<Polynomial> {
        self.remaining = self.remaining.checked_sub(definition.nodes().len())?;
        let coordinate = coordinate
            .iter()
            .map(|i| u32::try_from(*i).ok())
            .collect::<Option<Vec<_>>>()?;
        let mut mapped: Vec<Polynomial> = Vec::new();
        for node in definition.nodes() {
            let value = match node {
                CalculusNode::FormalComponent { formal, axes } => {
                    let indices = axes
                        .iter()
                        .map(|axis| axis.resolve(&coordinate).ok().map(|i| i as usize))
                        .collect::<Option<Vec<_>>>()?;
                    component(self, *formal, &indices, depth + 1)?
                }
                CalculusNode::Rational { value, .. } => Polynomial::constant(*value),
                CalculusNode::KroneckerDelta(a, b) => Polynomial::constant(ExactRational::integer(
                    i64::from(a.resolve(&coordinate).ok()? == b.resolve(&coordinate).ok()?),
                )),
                CalculusNode::Neg(a) => mapped[a.index() as usize].checked_neg().ok()?,
                CalculusNode::UnaryMath(eqiora_schema::kernel::UnaryMathFunction::Conj, a) => {
                    mapped[a.index() as usize].conjugate().ok()?
                }
                CalculusNode::Add(a, b) => mapped[a.index() as usize]
                    .checked_add(&mapped[b.index() as usize])
                    .ok()?,
                CalculusNode::Mul(a, b) => mapped[a.index() as usize]
                    .checked_mul(&mapped[b.index() as usize])
                    .ok()?,
                _ => return None,
            };
            mapped.push(value);
        }
        mapped.get(definition.root().index() as usize).cloned()
    }
}
