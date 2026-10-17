//! Affine dependency proof and coefficient arithmetic share one owner.
use super::super::*;

#[derive(Debug, Clone, PartialEq)]
pub(in crate::scalar) struct AffineSummary {
    pub(in crate::scalar) constant: Option<AffineValue>,
    pub(in crate::scalar) coefficients: Vec<AffineValue>,
}

impl AffineSummary {
    pub(in crate::scalar) fn bound(value: AffineValue, dimension: usize) -> Self {
        Self {
            constant: Some(value),
            coefficients: vec![AffineValue::constant(0.0); dimension],
        }
    }

    pub(in crate::scalar) fn constant(value: f64, dimension: usize) -> Self {
        Self::bound(AffineValue::constant(value), dimension)
    }

    pub(in crate::scalar) fn independent(dimension: usize) -> Self {
        Self {
            constant: None,
            coefficients: vec![AffineValue::constant(0.0); dimension],
        }
    }

    pub(in crate::scalar) fn variable(column: usize, dimension: usize) -> Self {
        let mut coefficients = vec![AffineValue::constant(0.0); dimension];
        coefficients[column] = AffineValue::constant(1.0);
        Self {
            constant: Some(AffineValue::constant(0.0)),
            coefficients,
        }
    }

    pub(in crate::scalar) fn depends_on_selected(&self) -> bool {
        self.coefficients
            .iter()
            .any(|coefficient| !coefficient.is_zero())
    }

    pub(in crate::scalar) fn scaled(
        &self,
        scale: AffineValue,
        instruction: usize,
    ) -> Result<Self, SymbolicLinearityFailure> {
        let constant = self.constant.map(|value| value.multiply(scale));
        let coefficients = self
            .coefficients
            .iter()
            .map(|coefficient| coefficient.multiply(scale))
            .collect::<Vec<_>>();
        Self::finite(constant, coefficients, instruction)
    }

    pub(in crate::scalar) fn sum(
        left: &Self,
        right: &Self,
        right_scale: f64,
        instruction: usize,
    ) -> Result<Self, SymbolicLinearityFailure> {
        let constant = left
            .constant
            .zip(right.constant)
            .map(|(left, right)| left.add(right.scale(right_scale)));
        let coefficients = left
            .coefficients
            .iter()
            .zip(&right.coefficients)
            .map(|(left, right)| left.add(right.scale(right_scale)))
            .collect::<Vec<_>>();
        Self::finite(constant, coefficients, instruction)
    }

    pub(in crate::scalar) fn product(
        left: &Self,
        right: &Self,
        instruction: usize,
    ) -> Result<Self, SymbolicLinearityFailure> {
        match (left.depends_on_selected(), right.depends_on_selected()) {
            (true, true) => Err(SymbolicLinearityFailure::Nonlinear { instruction }),
            (true, false) => right.constant.map_or_else(
                || Err(SymbolicLinearityFailure::VariableCoefficient { instruction }),
                |scale| left.scaled(scale, instruction),
            ),
            (false, true) => left.constant.map_or_else(
                || Err(SymbolicLinearityFailure::VariableCoefficient { instruction }),
                |scale| right.scaled(scale, instruction),
            ),
            (false, false) => {
                let constant = left
                    .constant
                    .zip(right.constant)
                    .map(|(left, right)| left.multiply(right));
                Self::finite(
                    constant,
                    vec![AffineValue::constant(0.0); left.coefficients.len()],
                    instruction,
                )
            }
        }
    }

    pub(in crate::scalar) fn quotient(
        numerator: &Self,
        denominator: &Self,
        instruction: usize,
    ) -> Result<Self, SymbolicLinearityFailure> {
        if denominator.depends_on_selected() {
            return Err(SymbolicLinearityFailure::Nonlinear { instruction });
        }
        if numerator.depends_on_selected() {
            return denominator.constant.map_or_else(
                || Err(SymbolicLinearityFailure::VariableCoefficient { instruction }),
                |denominator| {
                    numerator.scaled(AffineValue::constant(1.0).divide(denominator), instruction)
                },
            );
        }
        let constant = numerator
            .constant
            .zip(denominator.constant)
            .map(|(numerator, denominator)| numerator.divide(denominator));
        Self::finite(
            constant,
            vec![AffineValue::constant(0.0); numerator.coefficients.len()],
            instruction,
        )
    }

    pub(in crate::scalar) fn integer_power(
        &self,
        exponent: i32,
        instruction: usize,
    ) -> Result<Self, SymbolicLinearityFailure> {
        match exponent {
            0 => Ok(Self::constant(1.0, self.coefficients.len())),
            1 => Ok(self.clone()),
            _ if self.depends_on_selected() => {
                Err(SymbolicLinearityFailure::Nonlinear { instruction })
            }
            _ => {
                let constant = self.constant.map(|value| value.powi(exponent));
                Self::finite(
                    constant,
                    vec![AffineValue::constant(0.0); self.coefficients.len()],
                    instruction,
                )
            }
        }
    }

    pub(in crate::scalar) fn finite(
        constant: Option<AffineValue>,
        coefficients: Vec<AffineValue>,
        instruction: usize,
    ) -> Result<Self, SymbolicLinearityFailure> {
        if constant.is_some_and(|value| !value.is_finite())
            || coefficients.iter().any(|value| !value.is_finite())
        {
            Err(SymbolicLinearityFailure::NonFiniteCoefficient { instruction })
        } else {
            Ok(Self {
                constant,
                coefficients,
            })
        }
    }
}
