//! A conservative smooth-density profile for quadrature on bounded coordinate factors.
use super::*;

impl ScalarOperatorIr {
    /// Require polynomial, sine and exponential composition with no varying poles or radicals.
    ///
    /// `varying` must identify every input that depends on a coordinate in the density's
    /// support. Other inputs are held fixed; their numerical values must still pass normal
    /// evaluation (including nonzero divisors and finite intermediates). This structural
    /// check establishes no quadrature error bound and does not admit improper integrals.
    /// # Errors
    /// Rejects varying denominators, negative powers or square roots, and operations outside
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
                Instruction::Neg(a) | Instruction::Sin(a) | Instruction::Exp(a) => depends(a),
                Instruction::Add(a, b) | Instruction::Sub(a, b) | Instruction::Mul(a, b) => {
                    depends(a) || depends(b)
                }
                Instruction::Div(a, b) if !depends(b) => depends(a),
                Instruction::PowI(a, exponent) if exponent >= 0 || !depends(a) => {
                    exponent != 0 && depends(a)
                }
                Instruction::Sqrt(a) if !depends(a) => false,
                _ => {
                    return Err(ir_builder_error(
                        "regular factor density requires polynomial, sine or exponential composition with fixed nonzero denominators and no varying radicals",
                    ));
                }
            };
            dependencies.push(dependency);
        }
        Ok(())
    }
}
