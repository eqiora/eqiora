//! A conservative smooth-density profile for quadrature on bounded coordinate factors.
use super::*;
use eqiora_schema::kernel::UnaryMathFunction;

impl ScalarOperatorIr {
    /// Require polynomial, sine, cosine and exponential composition with no varying poles or radicals.
    ///
    /// `varying` must identify every input that depends on a coordinate in the density's
    /// support. Other inputs are held fixed; their numerical values must still pass normal
    /// evaluation (including nonzero divisors and finite intermediates). This structural
    /// check establishes no quadrature error bound and does not admit improper integrals.
    /// # Errors
    /// Rejects varying denominators, negative powers, logarithms or square roots, and operations outside
    /// this bounded smooth profile. Callers must separately validate derived input densities.
    pub fn require_regular_density(
        &self,
        mut varying: impl FnMut(SymbolRef) -> bool,
    ) -> Result<(), Diagnostic> {
        let mut dependencies = Vec::with_capacity(self.instructions.len());
        for (index, instruction) in self.instructions.iter().copied().enumerate() {
            let depends = |id: ValueId| dependencies[id.0 as usize];
            let dependency = match instruction {
                Instruction::Constant(_) => false,
                Instruction::Read(slot) => varying(self.symbols[slot_index(slot, index)?]),
                Instruction::Neg(a)
                | Instruction::Math(
                    UnaryMathFunction::Sin | UnaryMathFunction::Cos | UnaryMathFunction::Exp,
                    a,
                ) => depends(a),
                Instruction::Add(a, b) | Instruction::Sub(a, b) | Instruction::Mul(a, b) => {
                    depends(a) || depends(b)
                }
                Instruction::Div(a, b) if !depends(b) => depends(a),
                // Fixed validity checks preserve the smooth value's regularity.
                // A varying predicate cannot establish validity over an interval.
                Instruction::Compare(_, a, b) | Instruction::And(a, b) | Instruction::Or(a, b)
                    if !depends(a) && !depends(b) =>
                {
                    false
                }
                Instruction::Not(a) if !depends(a) => false,
                Instruction::Require { condition, value } if !depends(condition) => depends(value),
                Instruction::Demand { value, .. } => depends(value),
                Instruction::PowI(a, exponent) if exponent >= 0 || !depends(a) => {
                    exponent != 0 && depends(a)
                }
                Instruction::Math(UnaryMathFunction::Sqrt | UnaryMathFunction::Log, a)
                    if !depends(a) =>
                {
                    false
                }
                _ => {
                    return Err(ir_builder_error(
                        "regular factor density requires polynomial, sine, cosine or exponential composition with fixed nonzero denominators and no varying logarithms or radicals",
                    ));
                }
            };
            dependencies.push(dependency);
        }
        Ok(())
    }
}
