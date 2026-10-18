//! Finite-map partials retained by both numerical execution and spatial SSA.
use super::*;
use differential::Arithmetic;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MapOutput {
    Determinant,
    Inverse(u32),
    // A determinant partial is a polynomial, including at singular matrices.
    // This internal output retains the numerical cofactor rounding convention.
    Cofactor(u32),
}

impl MapOutput {
    pub(super) fn dimension_power(self, extent: u32) -> Result<i32, Diagnostic> {
        let n = i32::try_from(extent)
            .map_err(|_| ir_builder_error("finite map dimension exponent overflow"))?;
        Ok(match self {
            Self::Determinant => n,
            Self::Inverse(_) => -1,
            Self::Cofactor(_) => n - 1,
        })
    }
}

/// Exact row-major projection and sign of one cofactor's minor.
pub(super) fn minor_coordinates(n: usize, output: u32) -> Result<(f64, Vec<usize>), Diagnostic> {
    let count = n.checked_mul(n).ok_or_else(ir_size_error)?;
    if output as usize >= count {
        return Err(ir_builder_error("cofactor component is outside map"));
    }
    let (row, column) = (output as usize / n, output as usize % n);
    let coordinates = (0..count)
        .filter(|index| index / n != row && index % n != column)
        .collect();
    Ok((if (row + column) % 2 == 0 { 1. } else { -1. }, coordinates))
}

/// D(A^-1)[H]_(r,c) = -sum_ij B_(r,i) H_(i,j) B_(j,c).
/// Only the selected row and column are demanded; unused inverse outputs need
/// not be representable. Numerical and emitted scalar partials share this rule.
pub(super) fn inverse_gradient<A: Arithmetic>(
    algebra: &mut A,
    extent: usize,
    mut row: impl FnMut(&mut A, usize) -> Result<A::Value, Diagnostic>,
    mut column: impl FnMut(&mut A, usize) -> Result<A::Value, Diagnostic>,
    required: impl Fn(usize) -> bool,
) -> Result<Vec<Option<A::Value>>, Diagnostic> {
    let mut gradient = Vec::with_capacity(extent.checked_mul(extent).ok_or_else(ir_size_error)?);
    for i in 0..extent {
        for j in 0..extent {
            let partial = if required(gradient.len()) {
                let left = row(algebra, i)?;
                let right = column(algebra, j)?;
                let product = algebra.mul(left, right)?;
                Some(algebra.neg(product)?)
            } else {
                None
            };
            gradient.push(partial);
        }
    }
    Ok(gradient)
}

impl ScalarInputIrBuilder {
    pub(crate) fn map_output(
        &mut self,
        start: ScalarInputValueId,
        extent: u32,
        output: MapOutput,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let range = map_evaluation::operand_range(start.0, extent, self.instructions.len())?;
        if let MapOutput::Inverse(index) | MapOutput::Cofactor(index) = output
            && index as usize >= range.len()
        {
            return Err(ir_builder_error(
                "finite map output coordinate is outside its extent",
            ));
        }
        let key = (start.0.0, extent, output);
        if let Some(&value) = self.map_values.get(&key) {
            return Ok(value);
        }
        let value = if extent == 1 && output == MapOutput::Cofactor(0) {
            // The empty minor is exactly one. Retain source-domain checks in
            // the caller, without inventing dependence on this scalar entry.
            self.constant(eqiora_core::DynQuantity::new(
                1.,
                eqiora_core::DimExponents::DIMENSIONLESS,
            ))?
        } else {
            self.push(Instruction::MapInvariant {
                start: start.0,
                extent,
                output,
            })?
        };
        self.map_values.insert(key, value);
        Ok(value)
    }

    pub(super) fn map_direction(
        &mut self,
        start: ValueId,
        extent: u32,
        output: MapOutput,
        directions: &[Option<ScalarInputValueId>],
        charge: &mut impl FnMut(usize) -> Result<(), Diagnostic>,
    ) -> Result<Option<ScalarInputValueId>, Diagnostic> {
        let range = map_evaluation::operand_range(start, extent, directions.len())?;
        // At most six scalar operations per source entry cover inverse partials,
        // minor gathers, signs and contraction. Charge before emitting anything,
        // against the same caller-owned budget as authored finite-map products.
        charge(range.len().checked_mul(6).ok_or_else(ir_size_error)?)?;
        let n = extent as usize;
        let mut sum = None;
        let mut accumulate = |builder: &mut Self, partial, direction| -> Result<(), Diagnostic> {
            let term = builder.mul(partial, direction)?;
            sum = Some(match sum {
                None => term,
                Some(sum) => builder.add(sum, term)?,
            });
            Ok(())
        };
        if let MapOutput::Inverse(index) = output {
            let (r, c) = (index / extent, index % extent);
            let gradient = inverse_gradient(
                self,
                n,
                |builder, i| {
                    builder.map_output(
                        ScalarInputValueId(start),
                        extent,
                        MapOutput::Inverse(r * extent + i as u32),
                    )
                },
                |builder, j| {
                    builder.map_output(
                        ScalarInputValueId(start),
                        extent,
                        MapOutput::Inverse(j as u32 * extent + c),
                    )
                },
                |index| directions[range.start + index].is_some(),
            )?;
            for (partial, coordinate) in gradient.into_iter().zip(range) {
                if let (Some(partial), Some(direction)) = (partial, directions[coordinate]) {
                    accumulate(self, partial, direction)?;
                }
            }
        } else {
            let (map_start, size, sign, coordinates) = match output {
                MapOutput::Determinant => {
                    (ScalarInputValueId(start), extent, 1., (0..n * n).collect())
                }
                MapOutput::Cofactor(index) => {
                    let (sign, coordinates) = minor_coordinates(n, index)?;
                    if n == 1 {
                        return Ok(None);
                    }
                    let key = (start.0, extent, index);
                    let minor_start = if let Some(&minor) = self.map_minors.get(&key) {
                        minor
                    } else {
                        let entries = coordinates
                            .iter()
                            .map(|&index| ValueId(start.0 + index as u32));
                        let minor = ScalarInputValueId(map_evaluation::gather_entries(
                            &mut self.instructions,
                            entries,
                        )?);
                        self.map_minors.insert(key, minor);
                        minor
                    };
                    (minor_start, extent - 1, sign, coordinates)
                }
                MapOutput::Inverse(_) => unreachable!(),
            };
            for (index, coordinate) in coordinates.into_iter().enumerate() {
                if let Some(direction) = directions[range.start + coordinate] {
                    let partial =
                        self.map_output(map_start, size, MapOutput::Cofactor(index as u32))?;
                    let partial = if sign < 0. {
                        self.neg(partial)?
                    } else {
                        partial
                    };
                    accumulate(self, partial, direction)?;
                }
            }
        }
        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitted_inverse_action_only_evaluates_required_inverse_coordinates() {
        // A(p)=[[p,s],[s,0]], with the smallest positive subnormal s.
        // B00 and its p derivative are zero; B01=1/s is unrepresentable.
        let mut builder = ScalarInputIrBuilder::new();
        let mut constant = |x| {
            builder
                .constant(eqiora_core::DynQuantity::new(
                    x,
                    eqiora_core::DimExponents::DIMENSIONLESS,
                ))
                .unwrap()
        };
        let zero = constant(0.);
        let tiny = constant(f64::from_bits(1));
        let one = constant(1.);
        let start = builder.map_entries(&[zero, tiny, tiny, zero]).unwrap();
        let primal = builder.map_output(start, 2, MapOutput::Inverse(0)).unwrap();
        let mut directions = vec![None; builder.instructions.len()];
        directions[start.0.0 as usize] = Some(one);
        let derivative = builder
            .map_direction(
                start.0,
                2,
                MapOutput::Inverse(0),
                &directions,
                &mut |work| {
                    assert!(work <= MAX_COMPONENT_PRODUCTS);
                    Ok(())
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            builder
                .finish([primal, derivative])
                .unwrap()
                .evaluate(&[])
                .unwrap(),
            [0., 0.]
        );
    }

    #[test]
    fn derivative_work_admission_precedes_emission() {
        let mut builder = ScalarInputIrBuilder::new();
        let one = builder
            .constant(eqiora_core::DynQuantity::new(
                1.,
                eqiora_core::DimExponents::DIMENSIONLESS,
            ))
            .unwrap();
        let start = builder.map_entries(&[one; 4]).unwrap();
        let before = builder.instructions.len();
        let directions = vec![Some(one); before];
        let error = builder
            .map_direction(
                start.0,
                2,
                MapOutput::Inverse(0),
                &directions,
                &mut |work| {
                    assert!(work > 0);
                    Err(ir_builder_error("owner scalarization budget exhausted"))
                },
            )
            .unwrap_err();
        assert!(error.message().contains("owner scalarization budget"));
        assert_eq!(builder.instructions.len(), before);
        assert!(builder.map_values.is_empty());
        assert!(builder.map_minors.is_empty());
    }
}
