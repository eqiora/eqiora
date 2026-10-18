//! Complex quotients share retained binary exponents with scalar derivatives.
//! The squared norm stays scaled until the requested output is materialized.

use super::Scaled;

pub(super) fn quotient(numerator: [f64; 2], denominator: [f64; 2]) -> [f64; 2] {
    scaled_quotient(numerator.map(Scaled::new), denominator.map(Scaled::new)).map(Scaled::value)
}

pub(super) fn scaled_quotient([ar, ai]: [Scaled; 2], [br, bi]: [Scaled; 2]) -> [Scaled; 2] {
    let norm = br * br + bi * bi;
    [(ar * br + ai * bi) / norm, (ai * br - ar * bi) / norm]
}

pub(super) fn evaluate(
    operands: [super::ValueId; 4],
    values: &[f64],
    index: usize,
) -> Result<[f64; 2], super::Diagnostic> {
    let [a, b, c, d] = operands;
    Ok(quotient(
        [
            super::read(values, a, index)?,
            super::read(values, b, index)?,
        ],
        [
            super::read(values, c, index)?,
            super::read(values, d, index)?,
        ],
    ))
}

/// Typed demand uses the same scaled numerical quotient, with exact paired units.
pub(super) fn typed(
    operands: [&eqiora_core::ValueLiteral; 4],
    imaginary: bool,
    instruction: usize,
) -> Result<eqiora_core::ValueLiteral, super::Diagnostic> {
    let real = |index: usize| {
        operands[index].real_scalar_value().ok_or_else(|| {
            super::ir_builder_error("complex quotient requires real component coordinates")
        })
    };
    let [a, b, c, d] = [real(0)?, real(1)?, real(2)?, real(3)?];
    if a.dim() != b.dim() || c.dim() != d.dim() {
        return Err(super::ir_builder_error(
            "complex quotient coordinate units differ",
        ));
    }
    let dimension = a
        .dim()
        .div(c.dim())
        .ok_or_else(|| super::ir_builder_error("complex quotient dimension overflow"))?;
    let value = quotient([a.value(), b.value()], [c.value(), d.value()])[usize::from(imaginary)];
    super::require_finite_value(value, "complex quotient", instruction)?;
    eqiora_core::ValueLiteral::try_from(eqiora_core::DynQuantity::new(value, dimension))
        .map_err(|error| super::ir_builder_error(error.to_string()))
}

// Differentiate the quotient action directly. Forming 1/b first can overflow
// even when a scaled tangent or cotangent has a finite derivative.
pub(super) fn jvp(
    operands: [super::ValueId; 4],
    values: &[f64],
    tangents: &[Scaled],
    index: usize,
    imaginary: bool,
) -> Result<Scaled, super::Diagnostic> {
    let [a, b, c, d] = operands;
    let denominator = [
        Scaled::new(super::read(values, c, index)?),
        Scaled::new(super::read(values, d, index)?),
    ];
    let numerator = [
        Scaled::new(super::read(values, a, index)?),
        Scaled::new(super::read(values, b, index)?),
    ];
    let numerator_tangent = [
        super::read(tangents, a, index)?,
        super::read(tangents, b, index)?,
    ];
    let denominator_tangent = [
        super::read(tangents, c, index)?,
        super::read(tangents, d, index)?,
    ];
    Ok(direction_in(
        &mut super::differential::Numerical,
        numerator,
        denominator,
        Some(numerator_tangent),
        Some(denominator_tangent),
    )?
    .expect("directions present")[usize::from(imaginary)])
}

pub(super) fn direction(
    numerator: [f64; 2],
    denominator: [f64; 2],
    numerator_tangent: [f64; 2],
    denominator_tangent: [f64; 2],
) -> [f64; 2] {
    direction_in(
        &mut super::differential::Numerical,
        numerator.map(Scaled::new),
        denominator.map(Scaled::new),
        Some(numerator_tangent.map(Scaled::new)),
        Some(denominator_tangent.map(Scaled::new)),
    )
    .expect("numerical arithmetic is infallible")
    .expect("numerical directions are present")
    .map(Scaled::value)
}

/// The robust quotient differential has one owner for evaluation and SSA emission.
pub(super) fn direction_in<A: super::differential::Arithmetic>(
    algebra: &mut A,
    numerator: [A::Value; 2],
    denominator: [A::Value; 2],
    numerator_tangent: Option<[A::Value; 2]>,
    denominator_tangent: Option<[A::Value; 2]>,
) -> Result<Option<[A::Value; 2]>, super::Diagnostic> {
    let da = numerator_tangent
        .map(|d| algebra.complex_quotient(d, denominator))
        .transpose()?;
    let Some(db) = denominator_tangent else {
        return Ok(da);
    };
    let db = algebra.complex_quotient(db, denominator)?;
    let q = algebra.complex_quotient(numerator, denominator)?;
    let rr = algebra.mul(q[0], db[0])?;
    let ii = algebra.mul(q[1], db[1])?;
    let real = algebra.sub(rr, ii)?;
    let ri = algebra.mul(q[0], db[1])?;
    let ir = algebra.mul(q[1], db[0])?;
    let imaginary = algebra.add(ri, ir)?;
    Ok(Some(if let Some(da) = da {
        [algebra.sub(da[0], real)?, algebra.sub(da[1], imaginary)?]
    } else {
        [algebra.neg(real)?, algebra.neg(imaginary)?]
    }))
}

// For q=a/b, mixed curvature is -(q_left*b_right+q_right*b_left)/b.
// Divide the denominator directions first, retaining the scaled quotient owner.
pub(super) fn curvature(
    operands: [super::ValueId; 4],
    values: &[f64],
    left: &[Scaled],
    right: &[Scaled],
    index: usize,
    imaginary: bool,
) -> Result<Scaled, super::Diagnostic> {
    let denominator = [
        Scaled::new(super::read(values, operands[2], index)?),
        Scaled::new(super::read(values, operands[3], index)?),
    ];
    let direction = |tangent: &[Scaled]| -> Result<([Scaled; 2], [Scaled; 2]), super::Diagnostic> {
        Ok((
            [
                jvp(operands, values, tangent, index, false)?,
                jvp(operands, values, tangent, index, true)?,
            ],
            scaled_quotient(
                [
                    super::read(tangent, operands[2], index)?,
                    super::read(tangent, operands[3], index)?,
                ],
                denominator,
            ),
        ))
    };
    let (ql, bl) = direction(left)?;
    let (qr, br) = direction(right)?;
    Ok(if imaginary {
        -(ql[0] * br[1] + ql[1] * br[0] + qr[0] * bl[1] + qr[1] * bl[0])
    } else {
        -(ql[0] * br[0] - ql[1] * br[1] + qr[0] * bl[0] - qr[1] * bl[1])
    })
}

pub(super) fn vjp(
    operands: [super::ValueId; 4],
    values: &[f64],
    index: usize,
    imaginary: bool,
    seed: Scaled,
) -> Result<[Scaled; 4], super::Diagnostic> {
    let denominator = [
        Scaled::new(super::read(values, operands[2], index)?),
        -Scaled::new(super::read(values, operands[3], index)?),
    ];
    let weight = if imaginary {
        [Scaled::ZERO, seed]
    } else {
        [seed, Scaled::ZERO]
    };
    let a = scaled_quotient(weight, denominator);
    let q = scaled_quotient(
        [
            Scaled::new(super::read(values, operands[0], index)?),
            Scaled::new(super::read(values, operands[1], index)?),
        ],
        [denominator[0], -denominator[1]],
    );
    // Euclidean pairing on real/imaginary coordinates: a*=w/conj(b),
    // b*=-conj(a/b) a*.
    Ok([
        a[0],
        a[1],
        -(q[0] * a[0] + q[1] * a[1]),
        -(q[0] * a[1] - q[1] * a[0]),
    ])
}

pub(super) fn affine(
    arguments: [&super::AffineSummary; 4],
    imaginary: bool,
    index: usize,
) -> Result<super::AffineSummary, super::SymbolicLinearityFailure> {
    use super::{AffineSummary, SymbolicLinearityFailure};
    let [ar, ai, br, bi] = arguments;
    if br.depends_on_selected() || bi.depends_on_selected() {
        return Err(SymbolicLinearityFailure::Nonlinear { instruction: index });
    }
    let Some((br, bi)) = br.constant.zip(bi.constant) else {
        return if ar.depends_on_selected() || ai.depends_on_selected() {
            Err(SymbolicLinearityFailure::VariableCoefficient { instruction: index })
        } else {
            Ok(AffineSummary::independent(ar.coefficients.len()))
        };
    };
    let component = usize::from(imaginary);
    let constant = ar
        .constant
        .zip(ai.constant)
        .map(|(a, b)| super::AffineValue::complex_quotient([a, b], [br, bi])[component]);
    let coefficients = ar
        .coefficients
        .iter()
        .zip(&ai.coefficients)
        .map(|(&a, &b)| super::AffineValue::complex_quotient([a, b], [br, bi])[component])
        .collect();
    AffineSummary::finite(constant, coefficients, index)
}

#[cfg(test)]
mod tests {
    use super::quotient;

    #[test]
    fn numerical_quotient_retains_ieee_zero_signs_on_the_negative_real_axis() {
        // The real-coordinate formula gives -1-i0 for (1+i0)/(-1+i0),
        // and -1+i0 for (1-i0)/(-1-i0). atan2 distinguishes these sides.
        // This is numerical arithmetic; ValueLiteral separately canonicalizes
        // stored zeros and its contract is unchanged.
        for zero in [0.0_f64, -0.0] {
            let [real, imaginary] = quotient([1., zero], [-1., zero]);
            assert_eq!(real, -1.);
            assert_eq!(imaginary.to_bits(), (-zero).to_bits());
            assert_eq!(
                imaginary.atan2(real),
                (-zero).signum() * std::f64::consts::PI
            );
        }
    }

    #[test]
    fn quotient_preserves_finite_results_across_normal_and_subnormal_scales() {
        // (2+3i)/(4-i)=(5+14i)/17, derived by multiplying by 4+i.
        let expected = [5. / 17., 14. / 17.];
        for power in [-1000, -500, 0, 500, 1000] {
            let unit = 2_f64.powi(power);
            let actual = quotient([2. * unit, 3. * unit], [4. * unit, -unit]);
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 4. * f64::EPSILON);
            }
        }
        let tiny = f64::from_bits(1);
        assert_eq!(quotient([tiny, tiny], [tiny, tiny]), [1., 0.]);
        assert_eq!(
            quotient([f64::MAX, f64::MAX], [f64::MAX, f64::MAX]),
            [1., 0.]
        );
        assert_eq!(quotient([0., 0.], [tiny, tiny]), [0., 0.]);
        assert!(quotient([1., 0.], [0., 0.]).into_iter().all(f64::is_nan));
    }
    #[test]
    fn quotient_affine_analysis_requires_an_independent_fixed_denominator() {
        use super::super::{AffineSummary as A, SymbolicLinearityFailure as F};
        let x = A::variable(0, 2);
        let y = A::variable(1, 2);
        let four = A::constant(4., 2);
        let minus_one = A::constant(-1., 2);
        // Division by 4-i has real-coordinate matrix [[4,-1],[1,4]]/17.
        for (imaginary, expected) in [(false, [4. / 17., -1. / 17.]), (true, [1. / 17., 4. / 17.])]
        {
            let summary = super::affine([&x, &y, &four, &minus_one], imaginary, 7).unwrap();
            assert_eq!(summary.constant.map(|v| v.value), Some(0.));
            assert_eq!(
                summary
                    .coefficients
                    .iter()
                    .map(|v| v.value)
                    .collect::<Vec<_>>(),
                expected
            );
        }
        let independent = A::independent(2);
        assert_eq!(
            super::affine([&x, &y, &independent, &minus_one], false, 7),
            Err(F::VariableCoefficient { instruction: 7 })
        );
        assert_eq!(
            super::affine([&four, &minus_one, &x, &y], false, 7),
            Err(F::Nonlinear { instruction: 7 })
        );
        let zero = A::constant(0., 2);
        assert_eq!(
            super::affine([&x, &y, &zero, &zero], false, 7),
            Err(F::NonFiniteCoefficient { instruction: 7 })
        );
    }
}
