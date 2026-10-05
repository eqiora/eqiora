//! Homogeneous pencils reuse the scalar SSA affine proof and coefficient owner.
use super::*;

impl ScalarInputOperatorIr {
    pub(crate) fn bind_affine_pencil(
        &self,
        selected: &[ScalarSymbolCoordinate],
        spectral: &ScalarSymbolCoordinate,
        bindings: &[(ScalarSymbolCoordinate, f64)],
    ) -> Result<
        (
            BoundAffineScalarIr<ScalarSymbolCoordinate>,
            BoundAffineScalarIr<ScalarSymbolCoordinate>,
        ),
        Diagnostic,
    > {
        if selected.is_empty() || selected.contains(spectral) {
            return Err(ir_builder_error(
                "pencil requires mode coordinates disjoint from its spectral coordinate",
            ));
        }
        require_bidegree(self, selected, spectral)?;
        let mut at_zero = bindings.to_vec();
        at_zero.push((spectral.clone(), 0.));
        let a = self.bind_affine(selected, &at_zero)?;
        if a.offsets().iter().any(|&value| value != 0.) {
            return Err(ir_builder_error(
                "pencil must be homogeneous in its mode coordinates",
            ));
        }
        let mut mode_bindings = bindings.to_vec();
        mode_bindings.extend(selected.iter().cloned().map(|coordinate| (coordinate, 0.)));
        let lambda = std::slice::from_ref(spectral);
        let zero_mode = self.bind_affine(lambda, &mode_bindings)?;
        if zero_mode
            .coefficients()
            .iter()
            .chain(zero_mode.offsets())
            .any(|&value| value != 0.)
        {
            return Err(ir_builder_error(
                "pencil has a spectral term independent of its mode",
            ));
        }
        let count = self
            .roots
            .len()
            .checked_mul(selected.len())
            .ok_or_else(|| ir_builder_error("pencil coefficient count overflowed"))?;
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(count)
            .map_err(|_| ir_builder_error("cannot allocate pencil coefficients"))?;
        coefficients.resize(count, 0.);
        for column in 0..selected.len() {
            mode_bindings[bindings.len() + column].1 = 1.;
            // An affine coefficient, not f(1)-f(0): a large A cannot erase C.
            let row = self.bind_affine(lambda, &mode_bindings)?;
            for root in 0..self.roots.len() {
                coefficients[root * selected.len() + column] = row.coefficients()[root];
            }
            mode_bindings[bindings.len() + column].1 = 0.;
        }
        let c = BoundAffineScalarIr {
            selected_symbols: selected.to_vec(),
            residuals: self.roots.len(),
            coefficients,
            offsets: vec![0.; self.roots.len()],
        };
        Ok((a, c))
    }
}

fn require_bidegree(
    program: &ScalarInputOperatorIr,
    selected: &[ScalarSymbolCoordinate],
    spectral: &ScalarSymbolCoordinate,
) -> Result<(), Diagnostic> {
    let mut degrees: Vec<[u8; 2]> = Vec::with_capacity(program.instructions.len());
    for (index, instruction) in program.instructions.iter().copied().enumerate() {
        let at = |id: ValueId| {
            degrees
                .get(id.0 as usize)
                .copied()
                .ok_or_else(|| ir_builder_error("invalid pencil SSA operand"))
        };
        let nonlinear = || {
            ir_builder_error(format!(
                "pencil requires at most first degree in each of mode and spectral coordinates at instruction {index}"
            ))
        };
        let independent = |degree| {
            if degree == [0, 0] {
                Ok(degree)
            } else {
                Err(nonlinear())
            }
        };
        let degree = match instruction {
            Instruction::Constant(_) => [0, 0],
            Instruction::Read(slot) => {
                let coordinate = program
                    .slots
                    .get(slot.0 as usize)
                    .ok_or_else(|| ir_builder_error("invalid pencil input slot"))?
                    .source();
                [
                    u8::from(selected.contains(coordinate)),
                    u8::from(coordinate == spectral),
                ]
            }
            Instruction::Neg(value) => at(value)?,
            Instruction::Add(left, right) | Instruction::Sub(left, right) => {
                let (left, right) = (at(left)?, at(right)?);
                [left[0].max(right[0]), left[1].max(right[1])]
            }
            Instruction::Mul(left, right) => {
                let (left, right) = (at(left)?, at(right)?);
                let degree = [left[0] + right[0], left[1] + right[1]];
                if degree.iter().any(|&value| value > 1) {
                    return Err(nonlinear());
                }
                degree
            }
            Instruction::Div(left, right) => {
                independent(at(right)?)?;
                at(left)?
            }
            Instruction::ComplexDiv {
                operands: [a, b, c, d],
                ..
            } => {
                independent(at(c)?)?;
                independent(at(d)?)?;
                let (a, b) = (at(a)?, at(b)?);
                [a[0].max(b[0]), a[1].max(b[1])]
            }
            Instruction::PowI(_, 0) => [0, 0],
            Instruction::PowI(value, 1) => at(value)?,
            Instruction::PowI(value, _)
            | Instruction::Sin(value)
            | Instruction::Exp(value)
            | Instruction::Sqrt(value) => independent(at(value)?)?,
            Instruction::MapInvariant { start, extent, .. } => {
                let range = map_evaluation::operand_range(start, extent, index)?;
                for degree in &degrees[range] {
                    independent(*degree)?;
                }
                [0, 0]
            }
            _ => {
                return Err(ir_builder_error(
                    "instruction is outside the affine pencil profile",
                ));
            }
        };
        degrees.push(degree);
    }
    Ok(())
}
