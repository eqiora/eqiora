//! Optional forward direction in the existing affine coefficient arithmetic.
//! A missing direction preserves primal-only admission (notably sqrt at zero).
use super::super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::scalar) struct AffineValue {
    pub(in crate::scalar) value: f64,
    pub(in crate::scalar) tangent: Option<Scaled>,
}

impl AffineValue {
    pub(in crate::scalar) fn constant(value: f64) -> Self {
        Self {
            value,
            tangent: None,
        }
    }

    pub(in crate::scalar) fn is_finite(self) -> bool {
        self.value.is_finite() && self.tangent.is_none_or(Scaled::is_finite)
    }

    pub(in crate::scalar) fn is_zero(self) -> bool {
        self.value == 0. && self.tangent.is_none_or(Scaled::is_zero)
    }

    fn direction(
        self,
        other: Self,
        evaluate: impl FnOnce(Scaled, Scaled) -> Scaled,
    ) -> Option<Scaled> {
        (self.tangent.is_some() || other.tangent.is_some()).then(|| {
            evaluate(
                self.tangent.unwrap_or(Scaled::ZERO),
                other.tangent.unwrap_or(Scaled::ZERO),
            )
        })
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
            tangent: differential::quotient_direction(
                &mut differential::Numerical,
                Scaled::new(self.value),
                Scaled::new(other.value),
                self.tangent,
                other.tangent,
            )
            .expect("numerical quotient arithmetic is infallible"),
        }
    }

    pub(in crate::scalar) fn powi(self, exponent: i32) -> Self {
        Self {
            value: self.value.powi(exponent),
            tangent: self.tangent.map(|t| {
                differential::power_slope(
                    &mut differential::Numerical,
                    Scaled::new(self.value),
                    exponent,
                )
                .expect("numerical arithmetic is infallible")
                    * t
            }),
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
                .map(|t| {
                    real_math::action_in(
                        &mut differential::Numerical,
                        function,
                        Scaled::new(self.value),
                        t,
                    )
                })
                .transpose()?,
        })
    }

    pub(in crate::scalar) fn complex_quotient(a: [Self; 2], b: [Self; 2]) -> [Self; 2] {
        let values = complex_quotient::quotient(a.map(|v| v.value), b.map(|v| v.value));
        let direction = a.iter().chain(&b).any(|v| v.tangent.is_some()).then(|| {
            complex_quotient::direction_in(
                &mut differential::Numerical,
                a.map(|v| Scaled::new(v.value)),
                b.map(|v| Scaled::new(v.value)),
                Some(a.map(|v| v.tangent.unwrap_or(Scaled::ZERO))),
                Some(b.map(|v| v.tangent.unwrap_or(Scaled::ZERO))),
            )
            .expect("numerical arithmetic is infallible")
            .expect("directions present")
        });
        std::array::from_fn(|i| Self {
            value: values[i],
            tangent: direction.map(|v| v[i]),
        })
    }

    pub(in crate::scalar) fn map(
        values: &[Self],
        extent: usize,
        output: MapOutput,
    ) -> Result<Self, Diagnostic> {
        let entries = values.iter().map(|v| v.value).collect::<Vec<_>>();
        let mut map = map_evaluation::MapEvaluation::new(&entries, extent)?;
        let value = map.value(output)?;
        let tangent = if values.iter().any(|v| v.tangent.is_some()) {
            Some(
                map.gradient(output, &|index| {
                    values[index].tangent.is_some_and(|value| !value.is_zero())
                })?
                .iter()
                .zip(values)
                .map(|(g, v)| *g * v.tangent.unwrap_or(Scaled::ZERO))
                .sum::<Scaled>(),
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
    fn affine_inverse_direction_survives_outer_scaling() {
        for exponent in [-700, -400, 400, 700] {
            let scale = 2f64.powi(exponent);
            let diagonal = AffineValue {
                value: scale,
                tangent: Some(Scaled::new(1.)),
            };
            let zero = AffineValue::constant(0.);
            // A=pI: s*(A^-1)_00=s/p, with derivative -1/s at p=s.
            let inverse =
                AffineValue::map(&[diagonal, zero, zero, diagonal], 2, MapOutput::Inverse(0))
                    .unwrap();
            assert!(inverse.is_finite());
            let result = inverse.scale(scale);
            assert_eq!(result.value, 1.);
            assert_eq!(result.tangent.unwrap().value() * scale, -1.);
        }
    }

    #[test]
    fn affine_map_coefficients_share_exact_determinant_and_inverse_actions() {
        let values = [(2., 1.), (3., 0.), (5., 0.), (7., 2.)].map(|(value, tangent)| AffineValue {
            value,
            tangent: Some(Scaled::new(tangent)),
        });
        let determinant = AffineValue::map(&values, 2, MapOutput::Determinant).unwrap();
        assert!((determinant.value + 1.).abs() < 1e-13);
        assert!((determinant.tangent.unwrap().value() - 11.).abs() < 1e-12);
        // A^-1=[[-7,3],[5,-2]], d(A^-1)=-A^-1 diag(1,2) A^-1.
        for (i, (value, tangent)) in [(-7., -79.), (3., 33.), (5., 55.), (-2., -23.)]
            .into_iter()
            .enumerate()
        {
            let actual = AffineValue::map(&values, 2, MapOutput::Inverse(i as u32)).unwrap();
            assert!((actual.value - value).abs() < 1e-12);
            assert!((actual.tangent.unwrap().value() - tangent).abs() < 1e-11);
        }
    }
}
