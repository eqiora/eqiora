//! Execution-owned dense reference factorization for local finite maps.
//! Sizes come from typed operands; resource admission belongs to scalarization.
use super::*;

/// Evaluate a differential factor with the same factorization and conditioning
/// profile as finite-map determinants and inverses. Entries use coherent units;
/// the typed coordinate-map owner retains their individual dimensions.
/// # Errors
/// Rejects invalid shape, resource exhaustion, nonfinite arithmetic, and (for
/// volume/orientation) a singular or numerically unresolved differential.
pub fn coordinate_map_factor(
    entries: &[f64],
    extent: usize,
    factor: eqiora_schema::kernel::CoordinateMapFactor,
) -> Result<f64, Diagnostic> {
    use eqiora_schema::kernel::CoordinateMapFactor;
    let mut map = MapEvaluation::new(entries, extent)?;
    if factor != CoordinateMapFactor::SignedJacobian {
        map.inverse()?;
    }
    if factor == CoordinateMapFactor::Orientation {
        return Ok((0..extent).fold(map.sign, |sign, k| sign * map.lu[k * extent + k].signum()));
    }
    let determinant = map.value(None)?;
    Ok(match factor {
        CoordinateMapFactor::SignedJacobian => determinant,
        CoordinateMapFactor::VolumeScale => determinant.abs(),
        CoordinateMapFactor::Orientation => determinant.signum(),
    })
}

pub(super) fn operand_range(
    start: ValueId,
    extent: u32,
    end: usize,
) -> Result<std::ops::Range<usize>, Diagnostic> {
    let start = start.0 as usize;
    let count = (extent as usize)
        .checked_mul(extent as usize)
        .filter(|count| *count > 0)
        .ok_or_else(|| ir_builder_error("invalid map operand extent"))?;
    let limit = start
        .checked_add(count)
        .filter(|limit| *limit <= end)
        .ok_or_else(|| ir_builder_error("map operands are unavailable"))?;
    Ok(start..limit)
}

#[derive(Default)]
pub(super) struct MapCache(HashMap<(u32, u32), MapEvaluation>);
impl MapCache {
    pub(super) fn get(
        &mut self,
        start: ValueId,
        extent: u32,
        values: &[f64],
    ) -> Result<&mut MapEvaluation, Diagnostic> {
        let range = operand_range(start, extent, values.len())?;
        if let std::collections::hash_map::Entry::Vacant(entry) = self.0.entry((start.0, extent)) {
            entry.insert(MapEvaluation::new(&values[range], extent as usize)?);
        }
        Ok(self.0.get_mut(&(start.0, extent)).expect("inserted map"))
    }
}

pub(super) struct MapEvaluation {
    entries: Vec<f64>,
    lu: Vec<f64>,
    permutation: Vec<usize>,
    n: usize,
    scale: f64,
    norm: f64,
    sign: f64,
    singular: bool,
    normalized_inverse: Option<Vec<f64>>,
}

impl MapEvaluation {
    pub(super) fn new(entries: &[f64], n: usize) -> Result<Self, Diagnostic> {
        if n == 0 || n.checked_mul(n) != Some(entries.len()) {
            return Err(ir_builder_error(
                "finite map extent does not match operands",
            ));
        }
        if n.checked_pow(3)
            .and_then(|work| work.checked_mul(2))
            .is_none_or(|work| work > 1_000_000)
        {
            return Err(ir_builder_error(
                "finite map factorization exceeds one million component products",
            ));
        }
        require_finite(entries, "finite map operands")?;
        let scale = entries.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
        let scale = if scale == 0.0 { 1.0 } else { scale };
        let mut lu = entries.iter().map(|x| x / scale).collect::<Vec<_>>();
        if entries
            .iter()
            .zip(&lu)
            .any(|(input, scaled)| *input != 0.0 && *scaled == 0.0)
        {
            return Err(Diagnostic::error(
                codes::NONFINITE_EVALUATION,
                "finite map normalization underflows binary64; numerical zero is not evidence of singularity",
            ));
        }
        let norm = infinity_norm(&lu, n);
        let mut permutation = (0..n).collect::<Vec<_>>();
        let mut sign = 1.0;
        let mut singular = false;
        for k in 0..n {
            let pivot = (k..n)
                .max_by(|&a, &b| lu[a * n + k].abs().total_cmp(&lu[b * n + k].abs()))
                .expect("nonempty pivot column");
            if lu[pivot * n + k] == 0.0 {
                singular = true;
                break;
            }
            if pivot != k {
                for j in 0..n {
                    lu.swap(k * n + j, pivot * n + j);
                }
                permutation.swap(k, pivot);
                sign = -sign;
            }
            for i in k + 1..n {
                lu[i * n + k] /= lu[k * n + k];
                for j in k + 1..n {
                    lu[i * n + j] -= lu[i * n + k] * lu[k * n + j];
                }
            }
        }
        require_finite(&lu, "finite map factorization")?;
        Ok(Self {
            entries: entries.to_vec(),
            lu,
            permutation,
            n,
            scale,
            norm,
            sign,
            singular,
            normalized_inverse: None,
        })
    }

    fn determinant(&self) -> f64 {
        if self.singular {
            return 0.0;
        }
        // Combine exponents before materializing the result, so intermediate
        // products do not overflow solely because of coefficient units.
        let mut sign = self.sign;
        let mut logarithm = 0.0;
        for k in 0..self.n {
            let pivot = self.lu[k * self.n + k];
            sign *= pivot.signum();
            logarithm += pivot.abs().ln() + self.scale.ln();
        }
        sign * logarithm.exp()
    }

    fn factor_inverse(&mut self) -> Result<(), Diagnostic> {
        if self.singular {
            return Err(ir_builder_error(
                "finite map inverse is singular or numerically unresolved",
            ));
        }
        if self.normalized_inverse.is_some() {
            return Ok(());
        }
        let n = self.n;
        let mut inverse = vec![0.0; n * n];
        for column in 0..n {
            let mut x = self
                .permutation
                .iter()
                .map(|&row| f64::from(row == column))
                .collect::<Vec<_>>();
            for i in 0..n {
                for j in 0..i {
                    x[i] -= self.lu[i * n + j] * x[j];
                }
            }
            for i in (0..n).rev() {
                for j in i + 1..n {
                    x[i] -= self.lu[i * n + j] * x[j];
                }
                x[i] /= self.lu[i * n + i];
            }
            for i in 0..n {
                inverse[i * n + column] = x[i];
            }
        }
        self.normalized_inverse = Some(inverse);
        Ok(())
    }

    fn inverse(&mut self) -> Result<&[f64], Diagnostic> {
        self.factor_inverse()?;
        let inverse = self
            .normalized_inverse
            .as_deref()
            .expect("factored inverse");
        let reciprocal = (1.0 / self.norm) / infinity_norm(inverse, self.n);
        if !reciprocal.is_finite() || reciprocal <= 64.0 * f64::EPSILON {
            return Err(ir_builder_error(
                "finite map inverse is numerically ill-conditioned: reciprocal infinity-norm estimate must exceed 64 binary64 epsilons",
            ));
        }
        Ok(inverse)
    }

    pub(super) fn value(&mut self, component: Option<u32>) -> Result<f64, Diagnostic> {
        let scale = self.scale;
        let value = match component {
            None => {
                let value = self.determinant();
                if value == 0.0 && !self.singular {
                    return Err(Diagnostic::error(
                        codes::NONFINITE_EVALUATION,
                        "finite map determinant underflows binary64; numerical zero is not evidence of singularity",
                    ));
                }
                value
            }
            Some(index) => {
                *self
                    .inverse()?
                    .get(index as usize)
                    .ok_or_else(|| ir_builder_error("inverse component is outside map"))?
                    / scale
            }
        };
        require_finite_value(value, "finite map", 0)?;
        Ok(value)
    }

    pub(super) fn gradient(&mut self, component: Option<u32>) -> Result<Vec<f64>, Diagnostic> {
        let n = self.n;
        let scale = self.scale;
        let mut gradient = vec![0.0; n * n];
        if let Some(output) = component {
            if output as usize >= n * n {
                return Err(ir_builder_error("inverse component is outside map"));
            }
            let (r, c) = (output as usize / n, output as usize % n);
            let inverse = self.inverse()?;
            for i in 0..n {
                for j in 0..n {
                    gradient[i * n + j] =
                        -(inverse[r * n + i] / scale) * (inverse[j * n + c] / scale);
                }
            }
        } else {
            gradient = self.determinant_gradient()?;
        }
        require_finite(&gradient, "finite map derivative")?;
        Ok(gradient)
    }
    fn determinant_gradient(&mut self) -> Result<Vec<f64>, Diagnostic> {
        let n = self.n;
        let mut gradient = vec![0.0; n * n];
        let determinant = self.determinant();
        if !self.singular && determinant != 0.0 {
            self.factor_inverse()?;
            let inverse = self
                .normalized_inverse
                .as_deref()
                .expect("factored inverse");
            for i in 0..n {
                for j in 0..n {
                    gradient[i * n + j] = (determinant / self.scale) * inverse[j * n + i];
                }
            }
            // A representable cofactor can be lost when det/scale underflows
            // before multiplication by a large inverse entry. Re-evaluate with
            // minors rather than accepting that intermediate zero.
            if gradient.iter().enumerate().all(|(index, value)| {
                value.is_finite() && (*value != 0.0 || inverse[(index % n) * n + index / n] == 0.0)
            }) {
                return Ok(gradient);
            }
        }
        // The determinant is differentiable even when the inverse is undefined
        // or unrepresentable. Cofactors do not impose inverse admission on it.
        if n.checked_pow(5)
            .and_then(|work| work.checked_mul(2))
            .is_none_or(|work| work > 1_000_000)
        {
            return Err(ir_builder_error(
                "determinant derivative exceeds one million component products",
            ));
        }
        for i in 0..n {
            for j in 0..n {
                let mut minor = Vec::new();
                for r in 0..n {
                    for c in 0..n {
                        if r != i && c != j {
                            minor.push(self.entries[r * n + c]);
                        }
                    }
                }
                let value = if n == 1 {
                    1.0
                } else {
                    Self::new(&minor, n - 1)?.determinant()
                };
                gradient[i * n + j] = if (i + j) % 2 == 0 { value } else { -value };
            }
        }
        Ok(gradient)
    }
}

fn infinity_norm(entries: &[f64], n: usize) -> f64 {
    entries
        .chunks_exact(n)
        .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_does_not_require_a_representable_volume_factor() {
        use eqiora_schema::kernel::CoordinateMapFactor::{Orientation, VolumeScale};
        for n in [2, 6, 9, 16] {
            let mut entries = vec![0.0; n * n];
            for i in 0..n {
                entries[i * n + i] = 1e-200;
            }
            entries[0] = -1e-200;
            assert_eq!(
                coordinate_map_factor(&entries, n, Orientation).unwrap(),
                -1.0
            );
            assert!(
                coordinate_map_factor(&entries, n, VolumeScale)
                    .unwrap_err()
                    .message()
                    .contains("underflows")
            );
        }
    }

    #[test]
    fn normalization_cannot_turn_nonzero_coefficients_into_singular_success() {
        // The diagonal product is near one, despite the >binary64 dynamic range.
        let error = match MapEvaluation::new(&[1e308, 0.0, 0.0, 1e-308], 2) {
            Err(error) => error,
            Ok(_) => panic!("normalization must preserve nonzero operands or reject"),
        };
        assert!(
            error
                .message()
                .contains("normalization underflows binary64")
        );
    }

    #[test]
    fn determinant_cofactors_survive_intermediate_underflow() {
        let mut map =
            MapEvaluation::new(&[1e-200, 0.0, 0.0, 0.0, 1e-200, 0.0, 0.0, 0.0, 1e100], 3).unwrap();
        let gradient = map.gradient(None).unwrap();
        // Independently, diagonal cofactors are products of the other entries:
        // 1e-100, 1e-100, 1e-400 (the last is unrepresentable in binary64).
        for index in [0, 4] {
            assert!((gradient[index] / 1e-100 - 1.0).abs() < 1e-12);
        }
        for index in [1, 2, 3, 5, 6, 7, 8] {
            assert_eq!(gradient[index], 0.0);
        }
    }

    #[test]
    fn factorization_work_budget_precedes_arithmetic_and_has_an_admitted_neighbor() {
        // The reference work bound is 2*n^3, not a mathematical extent rule:
        // 2*79^3=986078 <=1e6, while 2*80^3=1024000 >1e6.
        let n = 79;
        let identity = (0..n * n)
            .map(|i| f64::from(i / n == i % n))
            .collect::<Vec<_>>();
        assert_eq!(
            MapEvaluation::new(&identity, n)
                .unwrap()
                .value(None)
                .unwrap(),
            1.0
        );
        let error = match MapEvaluation::new(&vec![f64::NAN; 80 * 80], 80) {
            Err(error) => error,
            Ok(_) => panic!("work budget must reject before numerical work"),
        };
        assert!(
            error
                .message()
                .contains("factorization exceeds one million component products")
        );
    }
}
