//! Structural polynomial degree; never inferred from sampled values.
use super::*;

impl<S: Scalar> ScalarSpatialExpression<S> {
    /// Total coordinate degree bound, or None for non-polynomial data.
    /// Saturation retains an impossible degree so resource admission rejects
    /// it instead of silently treating overflow as unknown mathematics.
    pub(crate) fn polynomial_degree(&self) -> Option<usize> {
        let mut degrees: Vec<Option<usize>> = Vec::with_capacity(self.instructions.len());
        for instruction in &self.instructions {
            let degree = match *instruction {
                Instruction::Constant(_) | Instruction::Parameter(_) => Some(0),
                Instruction::Coordinate(_) => Some(1),
                Instruction::Neg(a) | Instruction::Conjugate(a) => degrees[a],
                Instruction::Add(a, b) | Instruction::Sub(a, b) => {
                    degrees[a].zip(degrees[b]).map(|(a, b)| a.max(b))
                }
                Instruction::Mul(a, b) => {
                    degrees[a].zip(degrees[b]).map(|(a, b)| a.saturating_add(b))
                }
                Instruction::Div(a, b) => (degrees[b] == Some(0)).then_some(degrees[a]).flatten(),
                Instruction::PowI(a, n) => match (degrees[a], n) {
                    (_, 0) | (Some(0), _) => Some(0),
                    (Some(degree), n) if n > 0 => Some(degree.saturating_mul(n as usize)),
                    _ => None,
                },
                Instruction::Sin(a) | Instruction::Sqrt(a) => (degrees[a] == Some(0)).then_some(0),
            };
            degrees.push(degree);
        }
        degrees[self.root]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree_is_structural_and_does_not_infer_polynomials_from_sampled_parameters() {
        let mut tape = ScalarSpatialExpression::<f64> {
            coordinate_dimension: 1,
            instructions: vec![
                Instruction::Coordinate(0),
                Instruction::Parameter(0),
                Instruction::PowI(0, 8),
                Instruction::Mul(1, 2),
            ],
            root: 3,
            coordinate_dependent: true,
            parameter_fields: vec![Id::new()],
            parameter_values: vec![0.],
        };
        assert_eq!(tape.polynomial_degree(), Some(8));
        tape.instructions.push(Instruction::Div(2, 0));
        tape.root = 4;
        // Algebraic cancellation is not presumed across possible zero divisors.
        assert_eq!(tape.polynomial_degree(), None);
        tape.instructions[4] = Instruction::PowI(0, -1);
        assert_eq!(tape.polynomial_degree(), None);
        tape.instructions[4] = Instruction::Sin(0);
        assert_eq!(tape.polynomial_degree(), None);
        tape.instructions[4] = Instruction::Sin(1);
        assert_eq!(tape.polynomial_degree(), Some(0));
    }
}
