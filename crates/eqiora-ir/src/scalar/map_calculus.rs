//! Finite-map partials retained by both numerical execution and spatial SSA.
use super::*;
use differential::Arithmetic;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MapOutput {
    Determinant,
    VolumeScale,
    Orientation,
    Inverse(u32),
    // A determinant partial is a polynomial, including at singular matrices.
    // This internal output retains the numerical cofactor rounding convention.
    Cofactor(u32),
}

impl MapOutput {
    pub(crate) fn coordinate_factor(factor: eqiora_schema::kernel::CoordinateMapFactor) -> Self {
        use eqiora_schema::kernel::CoordinateMapFactor;
        match factor {
            CoordinateMapFactor::SignedJacobian => Self::Determinant,
            CoordinateMapFactor::VolumeScale => Self::VolumeScale,
            CoordinateMapFactor::Orientation => Self::Orientation,
        }
    }

    pub(super) fn dimension_power(self, extent: u32) -> Result<i32, Diagnostic> {
        let n = i32::try_from(extent)
            .map_err(|_| ir_builder_error("finite map dimension exponent overflow"))?;
        Ok(match self {
            Self::Determinant | Self::VolumeScale => n,
            Self::Orientation => 0,
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
        if output == MapOutput::Orientation {
            // directional() retains the demanded primal and its conditioning check.
            return Ok(None);
        }
        if output == MapOutput::VolumeScale {
            let derivative =
                self.map_direction(start, extent, MapOutput::Determinant, directions, charge)?;
            return derivative
                .map(|derivative| {
                    let orientation =
                        self.map_output(ScalarInputValueId(start), extent, MapOutput::Orientation)?;
                    self.mul(orientation, derivative)
                })
                .transpose();
        }
        // Charge the retained two-block solve gather (flags and values) or
        // signed-minor contraction before emission, against the caller's budget.
        let work = if matches!(output, MapOutput::Inverse(_)) {
            range.len().checked_mul(8).and_then(|n| n.checked_add(3))
        } else {
            range.len().checked_mul(6)
        };
        charge(work.ok_or_else(ir_size_error)?)?;
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
            if directions[range.clone()].iter().all(Option::is_none) {
                return Ok(None);
            }
            let shape = coefficient_jet::Shape::Solve {
                extent,
                component: index,
            };
            shape.admit(2 * range.len())?;
            let one = self.constant(eqiora_core::DynQuantity::new(
                1.,
                eqiora_core::DimExponents::DIMENSIONLESS,
            ))?;
            let mut a = vec![None; 2 * range.len()];
            for diagonal in 0..n {
                a[diagonal * n + diagonal] = Some(one);
            }
            let mut b = range
                .clone()
                .map(|i| Some(ScalarInputValueId(ValueId(i as u32))))
                .collect::<Vec<_>>();
            b.extend_from_slice(&directions[range]);
            return self.coefficient_jet(shape, &a, &b).map(Some);
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
                        let minor = ScalarInputValueId(slot_composition::gather_entries(
                            &mut self.instructions,
                            entries,
                        )?);
                        self.map_minors.insert(key, minor);
                        minor
                    };
                    (minor_start, extent - 1, sign, coordinates)
                }
                MapOutput::Inverse(_) | MapOutput::VolumeScale | MapOutput::Orientation => {
                    unreachable!()
                }
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
    fn guarded_unitful_inverse_jet_keeps_projected_forward_reverse_and_mixed_actions() {
        use eqiora_core::{DimExponents, DynQuantity, Id, ScalarDomain, ValueType};
        let unitless = DimExponents::DIMENSIONLESS;
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let ty = ValueType::scalar(ScalarDomain::Real, unitless).unwrap();
        for exponent in [-700, 700] {
            let scale = 2f64.powi(exponent);
            let coordinate =
                ScalarSymbolCoordinate::for_value(SymbolRef::Parameter(Id::new()), &ty)
                    .unwrap()
                    .remove(0);
            let mut builder = ScalarInputIrBuilder::new();
            let p = builder
                .input(ScalarInputSlot::new(0, coordinate, unitless))
                .unwrap();
            let one = builder.constant(DynQuantity::new(1., unitless)).unwrap();
            let negative_one = builder.neg(one).unwrap();
            let factor = builder.constant(DynQuantity::new(scale, length)).unwrap();
            let zero = builder.constant(DynQuantity::new(0., length)).unwrap();
            let sum = builder.add(one, p).unwrap();
            let diagonal = builder.mul(sum, factor).unwrap();
            let entries = builder
                .map_entries(&[diagonal, zero, zero, diagonal])
                .unwrap();
            let inverse = builder
                .map_output(entries, 2, MapOutput::Inverse(0))
                .unwrap();
            let condition = builder
                .compare(
                    eqiora_schema::kernel::ComparisonOp::Greater,
                    p,
                    negative_one,
                )
                .unwrap();
            let guarded = builder.require(condition, inverse).unwrap();
            let derivative = builder
                .directional(
                    guarded,
                    &[Some(one)],
                    length.pow(-1, 1).unwrap(),
                    |_| Ok(()),
                )
                .unwrap()
                .unwrap();
            let ir = builder.finish([derivative]).unwrap();
            // f(p)=1/(s*(1+p)); the emitted root is f'. Its value, first
            // action and mixed action at zero are -1/s, 2/s and -6/s.
            assert_eq!(ir.evaluate(&[0.]).unwrap(), [-1. / scale]);
            let action = ir
                .linearize(&[0.], &[DifferentiationRole::Unknown])
                .unwrap();
            let mut output = [0.];
            action
                .jvp(RelationTangent::Unknown(&[1.]), &mut output)
                .unwrap();
            assert_eq!(output, [2. / scale]);
            action
                .vjp(&[scale], RelationCotangent::Unknown(&mut output))
                .unwrap();
            assert_eq!(output, [2.]);
            action
                .mixed_jvp(
                    RelationTangent::Unknown(&[1.]),
                    RelationTangent::Unknown(&[1.]),
                    RelationTangent::Unknown(&[0.]),
                    &mut output,
                )
                .unwrap();
            assert_eq!(output, [-6. / scale]);
            assert!(ir.evaluate(&[-1.]).is_err());
        }
    }

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
