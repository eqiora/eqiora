//! Execution-owned dense reference factorization for local finite maps.
//! Sizes come from typed operands; resource admission belongs to scalarization.
use super::*;

impl ScalarOperatorIr {
    /// Apply one row-major entry tangent to the existing differential-factor calculus.
    /// Entries and tangents use coherent units; the typed map owns their dimensions.
    /// # Errors
    /// Rejects wrong tangent cardinality, nonfinite arithmetic, or the factor's
    /// ordinary singularity/conditioning failures.
    pub fn coordinate_map_factor_jvp(
        entries: &[f64],
        direction: &[f64],
        extent: usize,
        factor: eqiora_schema::kernel::CoordinateMapFactor,
    ) -> Result<f64, Diagnostic> {
        use eqiora_schema::kernel::CoordinateMapFactor;
        if direction.len() != entries.len() {
            return Err(ir_builder_error(
                "coordinate-map tangent cardinality differs",
            ));
        }
        require_finite(direction, "coordinate-map tangent")?;
        let mut map = MapEvaluation::new(entries, extent)?;
        if factor != CoordinateMapFactor::SignedJacobian {
            map.inverse()?;
        }
        if factor == CoordinateMapFactor::Orientation {
            return Ok(0.0);
        }
        let sign = if factor == CoordinateMapFactor::VolumeScale {
            (0..extent).fold(map.sign, |sign, k| sign * map.lu[k * extent + k].signum())
        } else {
            1.0
        };
        let derivative = sign
            * map
                .gradient(MapOutput::Determinant, &|index| direction[index] != 0.)?
                .iter()
                .zip(direction)
                .map(|(gradient, direction)| *gradient * *direction)
                .sum::<Scaled>()
                .value();
        require_finite_value(derivative, "coordinate-map factor derivative", 0)?;
        Ok(derivative)
    }

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
        let determinant = map.value(MapOutput::Determinant)?;
        Ok(match factor {
            CoordinateMapFactor::SignedJacobian => determinant,
            CoordinateMapFactor::VolumeScale => determinant.abs(),
            CoordinateMapFactor::Orientation => determinant.signum(),
        })
    }
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

/// Gather scalar coordinates without changing units or signed zero. Both initial
/// lowering and active-program projection use this contiguous operand convention.
pub(super) fn gather_entries(
    instructions: &mut Vec<Instruction>,
    entries: impl ExactSizeIterator<Item = ValueId> + Clone,
) -> Result<ValueId, Diagnostic> {
    let end = instructions.len();
    let limit = end
        .checked_add(entries.len())
        .and_then(|n| n.checked_add(1))
        .ok_or_else(ir_size_error)?;
    u32::try_from(limit).map_err(|_| ir_size_error())?;
    if entries.clone().any(|entry| entry.0 as usize >= end) {
        return Err(ir_builder_error(
            "map gather requires prior scalar operands",
        ));
    }
    let one = ValueId(end as u32);
    instructions.push(Instruction::Constant(eqiora_core::DynQuantity::new(
        1.,
        eqiora_core::DimExponents::DIMENSIONLESS,
    )));
    let start = ValueId(one.0 + 1);
    instructions.extend(entries.map(|entry| Instruction::Mul(entry, one)));
    Ok(start)
}

#[derive(Default)]
pub(super) struct MapCache(HashMap<(u32, u32), MapEvaluation>);
impl MapCache {
    fn cached(
        &mut self,
        start: ValueId,
        extent: u32,
        build: impl FnOnce() -> Result<MapEvaluation, Diagnostic>,
    ) -> Result<&mut MapEvaluation, Diagnostic> {
        if let std::collections::hash_map::Entry::Vacant(entry) = self.0.entry((start.0, extent)) {
            entry.insert(build()?);
        }
        Ok(self.0.get_mut(&(start.0, extent)).expect("inserted map"))
    }

    pub(super) fn get(
        &mut self,
        start: ValueId,
        extent: u32,
        values: &[f64],
    ) -> Result<&mut MapEvaluation, Diagnostic> {
        let range = operand_range(start, extent, values.len())?;
        self.cached(start, extent, || {
            MapEvaluation::new(&values[range], extent as usize)
        })
    }

    /// Typed demand shares factorization, admission and values across every
    /// component of the same map at this point, just as numerical execution does.
    pub(super) fn typed(
        &mut self,
        start: ValueId,
        extent: u32,
        output: MapOutput,
        values: &[Option<eqiora_core::ValueLiteral>],
        end: usize,
    ) -> Result<eqiora_core::ValueLiteral, Diagnostic> {
        let range = operand_range(start, extent, end)?;
        let read = |index: usize| {
            values
                .get(index)
                .and_then(Option::as_ref)
                .and_then(eqiora_core::ValueLiteral::real_scalar_value)
                .ok_or_else(|| {
                    ir_builder_error("finite map operands require demanded real scalar coordinates")
                })
        };
        let dimension = read(range.start)?.dim();
        let map = self.cached(start, extent, || {
            let entries = range.map(read).collect::<Result<Vec<_>, _>>()?;
            if entries.iter().any(|value| value.dim() != dimension) {
                return Err(ir_builder_error("finite map coefficient units differ"));
            }
            MapEvaluation::new(
                &entries
                    .iter()
                    .map(|value| value.value())
                    .collect::<Vec<_>>(),
                extent as usize,
            )
        })?;
        let power = output.dimension_power(extent)?;
        let dimension = dimension
            .pow(power, 1)
            .ok_or_else(|| ir_builder_error("finite map dimension overflow"))?;
        eqiora_core::ValueLiteral::try_from(eqiora_core::DynQuantity::new(
            map.value(output)?,
            dimension,
        ))
        .map_err(|error| ir_builder_error(error.to_string()))
    }
}

pub(super) struct MapEvaluation {
    entries: Vec<Scaled>,
    lu: Vec<Scaled>,
    permutation: Vec<usize>,
    n: usize,
    norm: Scaled,
    sign: f64,
    singular: bool,
    inverse_values: Option<Vec<Scaled>>,
    cofactors: Vec<Option<Scaled>>,
}

impl MapEvaluation {
    pub(super) fn new<T: Copy + Into<Scaled>>(entries: &[T], n: usize) -> Result<Self, Diagnostic> {
        if n == 0 || n.checked_mul(n) != Some(entries.len()) {
            return Err(ir_builder_error(
                "finite map extent does not match operands",
            ));
        }
        if n.checked_pow(3)
            .and_then(|work| work.checked_mul(2))
            .is_none_or(|work| work > MAX_COMPONENT_PRODUCTS)
        {
            return Err(ir_builder_error(
                "finite map factorization exceeds one million component products",
            ));
        }
        let entries = entries
            .iter()
            .map(|&value| value.into())
            .collect::<Vec<_>>();
        if entries.iter().any(|value| !value.is_finite()) {
            return Err(invalid_linearization(
                "finite map operands must contain only finite values",
            ));
        }
        let mut lu = entries.clone();
        let norm = infinity_norm(&lu, n);
        let mut permutation = (0..n).collect::<Vec<_>>();
        let mut sign = 1.0;
        let mut singular = false;
        for k in 0..n {
            let pivot = (k..n)
                .max_by(|&a, &b| lu[a * n + k].abs_cmp(lu[b * n + k]))
                .expect("nonempty pivot column");
            if lu[pivot * n + k].is_zero() {
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
                lu[i * n + k] = lu[i * n + k] / lu[k * n + k];
                for j in k + 1..n {
                    lu[i * n + j] = lu[i * n + j] - lu[i * n + k] * lu[k * n + j];
                }
            }
        }
        for &value in &lu {
            require_valid_derivative(value, "finite map factorization", 0)?;
        }
        Ok(Self {
            entries,
            lu,
            permutation,
            n,
            norm,
            sign,
            singular,
            inverse_values: None,
            cofactors: vec![None; n * n],
        })
    }

    fn determinant(&self) -> f64 {
        self.determinant_scaled().value()
    }

    fn determinant_scaled(&self) -> Scaled {
        if self.singular {
            return Scaled::ZERO;
        }
        (0..self.n).fold(Scaled::new(self.sign), |value, k| {
            value * self.lu[k * self.n + k]
        })
    }

    fn factor_inverse(&mut self) -> Result<(), Diagnostic> {
        if self.singular {
            return Err(ir_builder_error(
                "finite map inverse is singular or numerically unresolved",
            ));
        }
        if self.inverse_values.is_some() {
            return Ok(());
        }
        let n = self.n;
        let mut inverse = vec![Scaled::ZERO; n * n];
        for column in 0..n {
            let mut x = self
                .permutation
                .iter()
                .map(|&row| Scaled::new(f64::from(row == column)))
                .collect::<Vec<_>>();
            for i in 0..n {
                for j in 0..i {
                    x[i] = x[i] - self.lu[i * n + j] * x[j];
                }
            }
            for i in (0..n).rev() {
                for j in i + 1..n {
                    x[i] = x[i] - self.lu[i * n + j] * x[j];
                }
                x[i] = x[i] / self.lu[i * n + i];
            }
            for i in 0..n {
                inverse[i * n + column] = x[i];
            }
        }
        for &value in &inverse {
            require_valid_derivative(value, "finite map inverse factorization", 0)?;
        }
        self.inverse_values = Some(inverse);
        Ok(())
    }

    fn inverse(&mut self) -> Result<&[Scaled], Diagnostic> {
        self.factor_inverse()?;
        let inverse = self.inverse_values.as_deref().expect("factored inverse");
        let reciprocal = ((1.0 / self.norm) / infinity_norm(inverse, self.n)).value();
        if !reciprocal.is_finite() || reciprocal <= 64.0 * f64::EPSILON {
            return Err(ir_builder_error(
                "finite map inverse is numerically ill-conditioned: reciprocal infinity-norm estimate must exceed 64 binary64 epsilons",
            ));
        }
        Ok(inverse)
    }

    pub(super) fn value(&mut self, output: MapOutput) -> Result<f64, Diagnostic> {
        let value = match output {
            MapOutput::Determinant => {
                let value = self.determinant();
                if value == 0.0 && !self.singular {
                    return Err(Diagnostic::error(
                        codes::NONFINITE_EVALUATION,
                        "finite map determinant underflows binary64; numerical zero is not evidence of singularity",
                    ));
                }
                value
            }
            MapOutput::Inverse(index) => self
                .inverse()?
                .get(index as usize)
                .ok_or_else(|| ir_builder_error("inverse component is outside map"))?
                .value(),
            MapOutput::Cofactor(index) => self.cofactor(index)?.value(),
        };
        require_finite_value(value, "finite map", 0)?;
        Ok(value)
    }

    fn minor(&self, index: u32) -> Result<(f64, Vec<usize>, Self), Diagnostic> {
        let (sign, coordinates) = map_calculus::minor_coordinates(self.n, index)?;
        let entries = coordinates
            .iter()
            .map(|&index| self.entries[index])
            .collect::<Vec<_>>();
        Ok((sign, coordinates, Self::new(&entries, self.n - 1)?))
    }

    pub(super) fn gradient(
        &mut self,
        output: MapOutput,
        required: &dyn Fn(usize) -> bool,
    ) -> Result<Vec<Scaled>, Diagnostic> {
        let n = self.n;
        let gradient = match output {
            MapOutput::Inverse(output) => {
                if output as usize >= n * n {
                    return Err(ir_builder_error("inverse component is outside map"));
                }
                let (r, c) = (output as usize / n, output as usize % n);
                let inverse = self.inverse()?;
                map_calculus::inverse_gradient(
                    &mut differential::Numerical,
                    n,
                    |_, i| Ok(inverse[r * n + i]),
                    |_, j| Ok(inverse[j * n + c]),
                    required,
                )?
                .into_iter()
                .map(|value| value.unwrap_or(Scaled::ZERO))
                .collect()
            }
            MapOutput::Determinant => (0..n * n)
                .map(|index| {
                    if required(index) {
                        self.cofactor(index as u32)
                    } else {
                        Ok(Scaled::ZERO)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?,
            MapOutput::Cofactor(index) => {
                let mut gradient = vec![Scaled::ZERO; n * n];
                if n == 1 {
                    map_calculus::minor_coordinates(n, index)?;
                } else {
                    let (sign, coordinates, mut minor) = self.minor(index)?;
                    let partials = minor.gradient(MapOutput::Determinant, &|local| {
                        required(coordinates[local])
                    })?;
                    for (coordinate, partial) in coordinates.into_iter().zip(partials) {
                        gradient[coordinate] = sign * partial;
                    }
                }
                gradient
            }
        };
        for &value in &gradient {
            require_valid_derivative(value, "finite map derivative", 0)?;
        }
        Ok(gradient)
    }
    pub(super) fn curvature(
        &mut self,
        output: MapOutput,
        left: &[Scaled],
        right: &[Scaled],
    ) -> Result<Scaled, Diagnostic> {
        let n = self.n;
        require_length(left, n * n, "finite map first direction")?;
        require_length(right, n * n, "finite map second direction")?;
        for &value in left.iter().chain(right) {
            require_valid_derivative(value, "finite map direction", 0)?;
        }
        let value = if let MapOutput::Inverse(output) = output {
            if output as usize >= n * n {
                return Err(ir_builder_error("inverse component is outside map"));
            }
            let inverse = self.inverse()?.to_vec();
            let (r, c) = (output as usize / n, output as usize % n);
            // D²(A^-1)[H,K] = B H B K B + B K B H B, B=A^-1.
            // Only the requested row/column is contracted; no Hessian tensor.
            let product = |h: &[Scaled], k: &[Scaled]| {
                let kb = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| k[i * n + j] * inverse[j * n + c])
                            .sum::<Scaled>()
                    })
                    .collect::<Vec<_>>();
                let bkb = (0..n)
                    .map(|i| (0..n).map(|j| inverse[i * n + j] * kb[j]).sum::<Scaled>())
                    .collect::<Vec<_>>();
                (0..n)
                    .map(|i| {
                        inverse[r * n + i] * (0..n).map(|j| h[i * n + j] * bkb[j]).sum::<Scaled>()
                    })
                    .sum::<Scaled>()
            };
            product(left, right) + product(right, left)
        } else if let MapOutput::Cofactor(index) = output {
            if n == 1 {
                map_calculus::minor_coordinates(n, index)?;
                Scaled::ZERO
            } else {
                let (sign, coordinates, mut minor) = self.minor(index)?;
                let left = coordinates
                    .iter()
                    .map(|&index| left[index])
                    .collect::<Vec<_>>();
                let right = coordinates
                    .iter()
                    .map(|&index| right[index])
                    .collect::<Vec<_>>();
                sign * minor.curvature(MapOutput::Determinant, &left, &right)?
            }
        } else {
            // Multilinearity in rows is valid at singular matrices too. A
            // determinant derivative must not acquire inverse admission.
            if n.checked_pow(5)
                .and_then(|work| work.checked_mul(2))
                .is_none_or(|work| work > MAX_COMPONENT_PRODUCTS)
            {
                return Err(ir_builder_error(
                    "determinant mixed derivative exceeds one million component products",
                ));
            }
            let mut sum = Scaled::ZERO;
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let mut entries = self.entries.clone();
                    for (row, direction) in [(i, left), (j, right)] {
                        let range = row * n..(row + 1) * n;
                        entries[range.clone()].copy_from_slice(&direction[range]);
                    }
                    sum += Self::new(&entries, n)?.determinant_scaled();
                }
            }
            sum
        };
        require_valid_derivative(value, "finite map mixed derivative", 0)?;
        Ok(value)
    }

    fn cofactor(&mut self, index: u32) -> Result<Scaled, Diagnostic> {
        let n = self.n;
        let cached = self
            .cofactors
            .get(index as usize)
            .ok_or_else(|| ir_builder_error("cofactor component is outside map"))?;
        if let Some(value) = *cached {
            return Ok(value);
        }
        let determinant = self.determinant_scaled();
        if !self.singular {
            self.factor_inverse()?;
            let inverse = self.inverse_values.as_deref().expect("factored inverse");
            let transposed = (index as usize % n) * n + index as usize / n;
            let value = determinant * inverse[transposed];
            require_valid_derivative(value, "finite map cofactor", 0)?;
            self.cofactors[index as usize] = Some(value);
            return Ok(value);
        }
        // Cofactors are polynomials even at singular maps. Keep the existing
        // aggregate work bound, but evaluate only the requested coordinate.
        if n.checked_pow(5)
            .and_then(|work| work.checked_mul(2))
            .is_none_or(|work| work > MAX_COMPONENT_PRODUCTS)
        {
            return Err(ir_builder_error(
                "determinant derivative exceeds one million component products",
            ));
        }
        let value = if n == 1 {
            Scaled::new(1.)
        } else {
            let (sign, _, minor) = self.minor(index)?;
            sign * minor.determinant_scaled()
        };
        require_valid_derivative(value, "finite map cofactor", 0)?;
        self.cofactors[index as usize] = Some(value);
        Ok(value)
    }
}

fn infinity_norm(entries: &[Scaled], n: usize) -> Scaled {
    entries
        .chunks_exact(n)
        .map(|row| row.iter().map(|x| x.abs()).sum::<Scaled>())
        .max_by(|a, b| a.abs_cmp(*b))
        .unwrap_or(Scaled::ZERO)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod factor_rate_tests {
    use super::*;
    use eqiora_schema::kernel::CoordinateMapFactor::{Orientation, SignedJacobian, VolumeScale};
    #[test]
    fn determinant_rate_reuses_cofactors_at_regular_and_singular_maps() {
        // det([[2,1],[0,3]] + e*[[1,4],[2,-1]]) = 6-e-9e².
        for factor in [SignedJacobian, VolumeScale] {
            assert_eq!(
                ScalarOperatorIr::coordinate_map_factor_jvp(
                    &[2., 1., 0., 3.],
                    &[1., 4., 2., -1.],
                    2,
                    factor
                )
                .unwrap(),
                -1.
            );
        }
        // det(diag(1,0)+e*diag(0,1))=e, even though the inverse is undefined.
        assert_eq!(
            ScalarOperatorIr::coordinate_map_factor_jvp(
                &[1., 0., 0., 0.],
                &[0., 0., 0., 1.],
                2,
                SignedJacobian
            )
            .unwrap(),
            1.
        );
        for factor in [VolumeScale, Orientation] {
            assert!(
                ScalarOperatorIr::coordinate_map_factor_jvp(
                    &[1., 0., 0., 0.],
                    &[0., 0., 0., 1.],
                    2,
                    factor
                )
                .is_err()
            );
        }
        assert!(
            ScalarOperatorIr::coordinate_map_factor_jvp(&[1.], &[], 1, SignedJacobian).is_err()
        );
        assert!(
            ScalarOperatorIr::coordinate_map_factor_jvp(&[1.], &[f64::NAN], 1, SignedJacobian)
                .is_err()
        );
    }
}
