//! Optional forward direction in the existing affine coefficient arithmetic.
//! A missing direction preserves primal-only admission (notably sqrt at zero).
use super::super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::scalar) struct AffineValue {
    pub(in crate::scalar) value: f64,
    pub(in crate::scalar) tangent: Option<f64>,
}

impl AffineValue {
    pub(in crate::scalar) fn constant(value: f64) -> Self {
        Self {
            value,
            tangent: None,
        }
    }

    pub(in crate::scalar) fn is_finite(self) -> bool {
        self.value.is_finite() && self.tangent.is_none_or(f64::is_finite)
    }

    pub(in crate::scalar) fn is_zero(self) -> bool {
        self.value == 0. && self.tangent.is_none_or(|v| v == 0.)
    }

    fn direction(self, other: Self, evaluate: impl FnOnce(f64, f64) -> f64) -> Option<f64> {
        (self.tangent.is_some() || other.tangent.is_some())
            .then(|| evaluate(self.tangent.unwrap_or(0.), other.tangent.unwrap_or(0.)))
    }

    pub(in crate::scalar) fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            tangent: self.direction(other, |a, b| a + b),
        }
    }

    pub(in crate::scalar) fn scale(self, scale: f64) -> Self {
        Self {
            value: self.value * scale,
            tangent: self.tangent.map(|t| t * scale),
        }
    }

    pub(in crate::scalar) fn multiply(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            tangent: self.direction(other, |a, b| a * other.value + self.value * b),
        }
    }

    pub(in crate::scalar) fn divide(self, other: Self) -> Self {
        let value = self.value / other.value;
        Self {
            value,
            tangent: self.direction(other, |a, b| a / other.value - value * (b / other.value)),
        }
    }

    pub(in crate::scalar) fn powi(self, exponent: i32) -> Self {
        Self {
            value: self.value.powi(exponent),
            tangent: self
                .tangent
                .map(|t| powi_derivative(self.value, exponent) * t),
        }
    }

    pub(in crate::scalar) fn math(
        self,
        function: eqiora_schema::kernel::UnaryMathFunction,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            value: real_math::value(function, self.value)?,
            tangent: self
                .tangent
                .map(|t| real_math::action(function, self.value, t))
                .transpose()?,
        })
    }

    pub(in crate::scalar) fn complex_quotient(a: [Self; 2], b: [Self; 2]) -> [Self; 2] {
        let values = complex_quotient::quotient(a.map(|v| v.value), b.map(|v| v.value));
        let direction = a.iter().chain(&b).any(|v| v.tangent.is_some()).then(|| {
            complex_quotient::direction(
                a.map(|v| v.value),
                b.map(|v| v.value),
                a.map(|v| v.tangent.unwrap_or(0.)),
                b.map(|v| v.tangent.unwrap_or(0.)),
            )
        });
        std::array::from_fn(|i| Self {
            value: values[i],
            tangent: direction.map(|v| v[i]),
        })
    }

    pub(in crate::scalar) fn map(
        values: &[Self],
        extent: usize,
        component: Option<u32>,
    ) -> Result<Self, Diagnostic> {
        let entries = values.iter().map(|v| v.value).collect::<Vec<_>>();
        let mut map = map_evaluation::MapEvaluation::new(&entries, extent)?;
        let value = map.value(component)?;
        let tangent = if values.iter().any(|v| v.tangent.is_some()) {
            Some(
                map.gradient(component)?
                    .iter()
                    .zip(values)
                    .map(|(g, v)| g * v.tangent.unwrap_or(0.))
                    .sum(),
            )
        } else {
            None
        };
        Ok(Self { value, tangent })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_map_coefficients_share_exact_determinant_and_inverse_actions() {
        let values = [(2., 1.), (3., 0.), (5., 0.), (7., 2.)].map(|(value, tangent)| AffineValue {
            value,
            tangent: Some(tangent),
        });
        let determinant = AffineValue::map(&values, 2, None).unwrap();
        assert!((determinant.value + 1.).abs() < 1e-13);
        assert!((determinant.tangent.unwrap() - 11.).abs() < 1e-12);
        // A^-1=[[-7,3],[5,-2]], d(A^-1)=-A^-1 diag(1,2) A^-1.
        for (i, (value, tangent)) in [(-7., -79.), (3., 33.), (5., 55.), (-2., -23.)]
            .into_iter()
            .enumerate()
        {
            let actual = AffineValue::map(&values, 2, Some(i as u32)).unwrap();
            assert!((actual.value - value).abs() < 1e-12);
            assert!((actual.tangent.unwrap() - tangent).abs() < 1e-11);
        }
    }
}
