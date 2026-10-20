//! Structural polynomial degrees, independent of sampled coefficient values.
use super::*;

impl ScalarInputOperatorIr {
    pub(super) fn polynomial_degrees<const N: usize>(
        &self,
        inputs: &[Option<[usize; N]>],
    ) -> Result<Vec<Option<[usize; N]>>, Diagnostic> {
        require_length(inputs, self.slots.len(), "polynomial degree inputs")?;
        let mut degrees = Vec::with_capacity(self.instructions.len());
        for (index, instruction) in self.instructions.iter().copied().enumerate() {
            let at = |id: ValueId| degrees.get(id.0 as usize).copied().flatten();
            let independent = |degree| (degree == Some([0; N])).then_some([0; N]);
            let combine = |left: Option<[usize; N]>, right: Option<[usize; N]>, product| {
                let (left, right) = (left?, right?);
                let mut degree = [0; N];
                for axis in 0..N {
                    degree[axis] = if product {
                        left[axis].checked_add(right[axis])?
                    } else {
                        left[axis].max(right[axis])
                    };
                }
                Some(degree)
            };
            let degree = match instruction {
                Instruction::Constant(_) | Instruction::TypedConstant(_) => Some([0; N]),
                Instruction::Read(slot) => inputs[slot.0 as usize],
                Instruction::Neg(value) => at(value),
                Instruction::Add(a, b) | Instruction::Sub(a, b) => combine(at(a), at(b), false),
                Instruction::Mul(a, b) => combine(at(a), at(b), true),
                Instruction::Div(a, b) => independent(at(b)).and_then(|_| at(a)),
                Instruction::Compare(_, a, b) | Instruction::And(a, b) | Instruction::Or(a, b) => {
                    independent(at(a)).and(independent(at(b)))
                }
                Instruction::Not(value) => independent(at(value)),
                // Require restricts the domain, not the value's polynomial.
                // Its predicate remains in the instruction degrees, so an
                // affine/constant proof cannot erase a selected primal guard.
                Instruction::Require { value, .. } | Instruction::DerivativeLimit { value, .. } => {
                    at(value)
                }
                Instruction::Select {
                    condition,
                    then_value,
                    else_value,
                } => independent(at(condition))
                    .and_then(|_| combine(at(then_value), at(else_value), false)),
                Instruction::ComplexDiv {
                    operands: [a, b, c, d],
                    ..
                } => independent(at(c))
                    .and(independent(at(d)))
                    .and_then(|_| combine(at(a), at(b), false)),
                Instruction::PowI(_, 0) => Some([0; N]),
                Instruction::PowI(value, exponent) if exponent > 0 => {
                    at(value).and_then(|before| {
                        let mut degree = [0; N];
                        for axis in 0..N {
                            degree[axis] = before[axis].checked_mul(exponent as usize)?;
                        }
                        Some(degree)
                    })
                }
                Instruction::PowI(value, _) | Instruction::Math(_, value) => independent(at(value)),
                Instruction::CoefficientJet { start, len, shape } => {
                    let operands =
                        coefficient_jet::operands(start, len, shape, &self.instructions[..index])?;
                    if matches!(shape, coefficient_jet::Shape::ComplexMath { .. }) {
                        operands
                            .iter()
                            .all(|id| independent(at(id)).is_some())
                            .then_some([0; N])
                    } else if operands
                        .b
                        .iter()
                        .flatten()
                        .all(|&id| independent(at(id)).is_some())
                    {
                        operands
                            .a
                            .iter()
                            .flatten()
                            .try_fold([0; N], |degree, &id| combine(Some(degree), at(id), false))
                    } else {
                        None
                    }
                }
                Instruction::MapInvariant { start, extent, .. } => {
                    let range = map_evaluation::operand_range(start, extent, index)?;
                    degrees[range]
                        .iter()
                        .all(|&d| independent(d).is_some())
                        .then_some([0; N])
                }
                _ => None,
            };
            degrees.push(degree);
        }
        Ok(degrees)
    }

    pub(crate) fn affine_in(
        &self,
        selected: &[ScalarSymbolCoordinate],
    ) -> Result<bool, Diagnostic> {
        let mut coordinates = std::collections::HashSet::new();
        for coordinate in selected {
            if !coordinates.insert(coordinate) {
                return Err(ir_builder_error(
                    "affine classification repeats a selected coordinate",
                ));
            }
        }
        let inputs = self
            .slots
            .iter()
            .map(|slot| Some([usize::from(coordinates.contains(slot.source()))]))
            .collect::<Vec<_>>();
        // Classify every retained instruction, not just the outputs. A zero
        // power or cancellation must not hide an unknown-dependent domain.
        let degrees = self.polynomial_degrees(&inputs)?;
        for (index, instruction) in self.instructions.iter().enumerate() {
            if let Instruction::DerivativeLimit {
                start,
                len,
                order: 0,
                ..
            } = *instruction
                && derivative_limit::inputs(start, len, index)?
                    .any(|input| degrees[input] != Some([0]))
            {
                return Ok(false);
            }
        }
        Ok(degrees
            .iter()
            .all(|degree| degree.is_some_and(|[degree]| degree <= 1)))
    }

    pub(crate) fn polynomial_degree(&self, inputs: &[Option<usize>]) -> Option<usize> {
        let inputs = inputs
            .iter()
            .map(|degree| degree.map(|d| [d]))
            .collect::<Vec<_>>();
        let degrees = self.polynomial_degrees(&inputs).ok()?;
        let degree = self.roots.iter().try_fold(0, |degree, root| {
            Some(degree.max(degrees.get(root.0 as usize)?.as_ref()?[0]))
        })?;
        // A zero output degree must not erase a spatially nonpolynomial primal
        // domain, e.g. sqrt(x)^0 or (1/x)^0. Consumers also use degree zero as
        // a constancy proof; all demanded intermediates must support that proof.
        (degree != 0 || degrees.iter().all(Option::is_some)).then_some(degree)
    }
}
