//! Exact homogeneous polynomial pencils over real component coordinates.
use super::*;
use std::collections::BTreeMap;
mod polynomial;
use polynomial::Polynomial;

impl ScalarInputOperatorIr {
    pub(crate) fn bind_polynomial_pencil(
        &self,
        selected: &[ScalarSymbolCoordinate],
        spectral: &[ScalarSymbolCoordinate],
        maximum_degree: u32,
        term_budget: usize,
        bindings: &[(ScalarSymbolCoordinate, f64)],
    ) -> Result<BTreeMap<Vec<u32>, BoundAffineScalarIr<ScalarSymbolCoordinate>>, Diagnostic> {
        if selected.is_empty() || spectral.is_empty() || term_budget == 0 {
            return Err(ir_builder_error(
                "pencil requires mode and spectral coordinates and a nonzero coefficient resource budget",
            ));
        }
        let mut coordinates = HashMap::new();
        for (index, coordinate) in selected.iter().chain(spectral).enumerate() {
            if coordinates.insert(coordinate, index).is_some() {
                return Err(ir_builder_error(
                    "pencil repeats a mode or spectral coordinate",
                ));
            }
        }
        let mut constants = HashMap::new();
        for (coordinate, value) in bindings {
            if !value.is_finite()
                || coordinates.contains_key(coordinate)
                || constants.insert(coordinate, *value).is_some()
            {
                return Err(ir_builder_error(
                    "invalid, repeated or selected pencil binding",
                ));
            }
        }
        let inputs = self
            .slots
            .iter()
            .map(|slot| {
                if coordinates.contains_key(slot.source()) {
                    Ok(0.)
                } else {
                    constants
                        .get(slot.source())
                        .copied()
                        .ok_or_else(|| ir_builder_error("pencil coordinate is unbound"))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Prove the structural profile before evaluating any bound coefficient.
        // In particular, zero bindings cannot erase nonlinear mode dependence.
        let degrees = require_degrees(self, selected, spectral, maximum_degree)?;
        let values = evaluate_instructions(&self.instructions, &inputs)?;
        let variables = selected.len() + spectral.len();
        let mut polynomials: Vec<Polynomial> = Vec::new();
        let mut remaining = term_budget;
        for (index, instruction) in self.instructions.iter().copied().enumerate() {
            let polynomial = if degrees[index] == [0, 0] {
                Polynomial::constant(variables, values[index])
            } else {
                polynomial::evaluate(
                    instruction,
                    &polynomials,
                    &values,
                    variables,
                    &self.slots,
                    &coordinates,
                    term_budget,
                )?
            };
            remaining = remaining.checked_sub(polynomial.len()).ok_or_else(|| {
                ir_builder_error("polynomial pencil coefficient resource budget exhausted")
            })?;
            polynomials.push(polynomial);
        }
        let count = self
            .roots
            .len()
            .checked_mul(selected.len())
            .filter(|count| *count <= term_budget)
            .ok_or_else(|| {
                ir_builder_error("pencil matrix coefficient resource budget exhausted")
            })?;
        let mut result = BTreeMap::new();
        for (row, root) in self.roots.iter().enumerate() {
            let polynomial = polynomials
                .get(root.0 as usize)
                .ok_or_else(|| ir_builder_error("invalid pencil root"))?;
            for (powers, value) in polynomial.iter() {
                if *value == 0. {
                    continue;
                }
                let mode = powers[..selected.len()]
                    .iter()
                    .position(|power| *power != 0)
                    .ok_or_else(|| {
                        ir_builder_error("pencil must be homogeneous in its mode coordinates")
                    })?;
                let spectral_powers = powers[selected.len()..].to_vec();
                if !result.contains_key(&spectral_powers) {
                    remaining = remaining.checked_sub(count).ok_or_else(|| {
                        ir_builder_error("pencil output coefficient resource budget exhausted")
                    })?;
                }
                let coefficient =
                    result
                        .entry(spectral_powers)
                        .or_insert_with(|| BoundAffineScalarIr {
                            selected_symbols: selected.to_vec(),
                            residuals: self.roots.len(),
                            coefficients: vec![0.; count],
                            offsets: vec![0.; self.roots.len()],
                        });
                coefficient.coefficients[row * selected.len() + mode] = *value;
            }
        }
        Ok(result)
    }
}

fn require_degrees(
    program: &ScalarInputOperatorIr,
    selected: &[ScalarSymbolCoordinate],
    spectral: &[ScalarSymbolCoordinate],
    maximum_degree: u32,
) -> Result<Vec<[u32; 2]>, Diagnostic> {
    let mut degrees: Vec<[u32; 2]> = Vec::with_capacity(program.instructions.len());
    for (index, instruction) in program.instructions.iter().copied().enumerate() {
        let at = |id: ValueId| {
            degrees
                .get(id.0 as usize)
                .copied()
                .ok_or_else(|| ir_builder_error("invalid pencil SSA operand"))
        };
        let nonlinear = || {
            ir_builder_error(format!(
                "pencil requires first mode degree and the admitted spectral degree at instruction {index}"
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
                    u32::from(selected.contains(coordinate)),
                    u32::from(spectral.contains(coordinate)),
                ]
            }
            Instruction::Neg(value) => at(value)?,
            Instruction::Add(left, right) | Instruction::Sub(left, right) => {
                let (left, right) = (at(left)?, at(right)?);
                [left[0].max(right[0]), left[1].max(right[1])]
            }
            Instruction::Mul(left, right) => {
                let (left, right) = (at(left)?, at(right)?);
                let degree = [
                    left[0].checked_add(right[0]).ok_or_else(nonlinear)?,
                    left[1].checked_add(right[1]).ok_or_else(nonlinear)?,
                ];
                if degree[0] > 1 || degree[1] > maximum_degree {
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
            Instruction::PowI(value, exponent) if exponent > 1 => {
                let before = at(value)?;
                let degree = [
                    before[0]
                        .checked_mul(exponent as u32)
                        .ok_or_else(nonlinear)?,
                    before[1]
                        .checked_mul(exponent as u32)
                        .ok_or_else(nonlinear)?,
                ];
                if degree[0] > 1 || degree[1] > maximum_degree {
                    return Err(nonlinear());
                }
                degree
            }
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
                    "instruction is outside the polynomial pencil profile",
                ));
            }
        };
        if degree[0] > 1 || degree[1] > maximum_degree {
            return Err(nonlinear());
        }
        degrees.push(degree);
    }
    Ok(degrees)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, DynQuantity, Id, ScalarDomain, ValueType};

    fn program(
        instructions: Vec<Instruction>,
        roots: &[u32],
    ) -> (ScalarInputOperatorIr, Vec<ScalarSymbolCoordinate>) {
        let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
        let coordinates = (0..3)
            .map(|_| {
                ScalarSymbolCoordinate::for_value(SymbolRef::Field(Id::new()), &real)
                    .unwrap()
                    .remove(0)
            })
            .collect::<Vec<_>>();
        let slots = coordinates
            .iter()
            .enumerate()
            .map(|(index, coordinate)| ScalarInputSlot::new(index as u32, coordinate.clone()))
            .collect();
        (
            ScalarInputOperatorIr {
                slots,
                instructions,
                roots: roots.iter().copied().map(ValueId).collect(),
            },
            coordinates,
        )
    }
    fn constant(value: f64) -> Instruction {
        Instruction::Constant(DynQuantity::new(value, DimExponents::DIMENSIONLESS))
    }

    #[test]
    fn polynomial_coefficients_preserve_quadratic_complex_parts_and_small_terms() {
        // (1e30 + 3*x + 2*(x²-y²))*u and (3*y + 4*x*y)*u.
        // These are real and imaginary parts of (1e30+3λ+2λ²)u
        // for a real u; the 3 coefficient must not disappear beside 1e30.
        let (ir, coordinates) = program(
            vec![
                Instruction::Read(SymbolSlot(0)),
                Instruction::Read(SymbolSlot(1)),
                Instruction::Read(SymbolSlot(2)),
                constant(1e30),
                constant(3.),
                constant(2.),
                constant(4.),
                Instruction::Mul(ValueId(4), ValueId(1)),
                Instruction::PowI(ValueId(1), 2),
                Instruction::PowI(ValueId(2), 2),
                Instruction::Sub(ValueId(8), ValueId(9)),
                Instruction::Mul(ValueId(5), ValueId(10)),
                Instruction::Add(ValueId(3), ValueId(7)),
                Instruction::Add(ValueId(12), ValueId(11)),
                Instruction::Mul(ValueId(13), ValueId(0)),
                Instruction::Mul(ValueId(1), ValueId(2)),
                Instruction::Mul(ValueId(6), ValueId(15)),
                Instruction::Mul(ValueId(4), ValueId(2)),
                Instruction::Add(ValueId(16), ValueId(17)),
                Instruction::Mul(ValueId(18), ValueId(0)),
            ],
            &[14, 19],
        );
        let coefficients = ir
            .bind_polynomial_pencil(&coordinates[..1], &coordinates[1..], 2, 1000, &[])
            .unwrap();
        for (powers, expected) in [
            (vec![0, 0], [1e30, 0.]),
            (vec![1, 0], [3., 0.]),
            (vec![0, 1], [0., 3.]),
            (vec![2, 0], [2., 0.]),
            (vec![0, 2], [-2., 0.]),
            (vec![1, 1], [0., 4.]),
        ] {
            assert_eq!(coefficients[&powers].coefficients(), expected);
            assert_eq!(coefficients[&powers].offsets(), [0., 0.]);
        }
        assert!(
            ir.bind_polynomial_pencil(&coordinates[..1], &coordinates[1..], 1, 1000, &[])
                .is_err()
        );
        assert!(
            ir.bind_polynomial_pencil(&coordinates[..1], &coordinates[1..], 2, 1, &[])
                .is_err()
        );
    }

    #[test]
    fn polynomial_proof_rejects_nonlinear_modes_and_dependent_denominators_before_binding() {
        for instruction in [
            Instruction::Mul(ValueId(0), ValueId(0)),
            Instruction::Div(ValueId(0), ValueId(1)),
            Instruction::Sin(ValueId(1)),
        ] {
            let (ir, coordinates) = program(
                vec![
                    Instruction::Read(SymbolSlot(0)),
                    Instruction::Read(SymbolSlot(1)),
                    instruction,
                    constant(0.),
                    Instruction::Mul(ValueId(2), ValueId(3)),
                ],
                &[4],
            );
            assert!(
                ir.bind_polynomial_pencil(&coordinates[..1], &coordinates[1..], 2, 1000, &[])
                    .is_err()
            );
        }
    }

    #[test]
    fn polynomial_binding_rejects_missing_repeated_bound_and_inhomogeneous_coordinates() {
        let (ir, coordinates) = program(
            vec![
                Instruction::Read(SymbolSlot(0)),
                Instruction::Read(SymbolSlot(1)),
                Instruction::Add(ValueId(0), ValueId(1)),
            ],
            &[2],
        );
        assert!(
            ir.bind_polynomial_pencil(&coordinates[..1], &coordinates[1..], 2, 1000, &[])
                .is_err()
        );
        assert!(
            ir.bind_polynomial_pencil(&coordinates[..1], &coordinates[1..2], 2, 1000, &[])
                .is_err()
        );
        assert!(
            ir.bind_polynomial_pencil(
                &coordinates[..1],
                &coordinates[1..],
                2,
                1000,
                &[(coordinates[0].clone(), 0.)]
            )
            .is_err()
        );
        assert!(
            ir.bind_polynomial_pencil(&coordinates[..1], &coordinates[..1], 2, 1000, &[])
                .is_err()
        );
    }
}
