//! Scale each operand by a power of two before forming a complex quotient.
//!
//! No squared physical denominator or overflowing reciprocal is required.

pub(super) fn quotient(numerator: [f64; 2], denominator: [f64; 2]) -> [f64; 2] {
    let a = numerator[0].abs().max(numerator[1].abs());
    let b = denominator[0].abs().max(denominator[1].abs());
    if !a.is_finite() || !b.is_finite() || b == 0. {
        return [f64::NAN; 2];
    }
    let ae = exponent(a);
    let be = exponent(b);
    let [ar, ai] = numerator.map(|value| scale(value, -ae));
    let [br, bi] = denominator.map(|value| scale(value, -be));
    let norm = br * br + bi * bi;
    [
        scale((ar * br + ai * bi) / norm, ae - be),
        scale((ai * br - ar * bi) / norm, ae - be),
    ]
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

// Differentiate the quotient action directly. Forming 1/b first can overflow
// even when a scaled tangent or cotangent has a finite derivative.
pub(super) fn jvp(
    operands: [super::ValueId; 4],
    values: &[f64],
    tangents: &[f64],
    index: usize,
    imaginary: bool,
) -> Result<f64, super::Diagnostic> {
    let [a, b, c, d] = operands;
    let denominator = [
        super::read(values, c, index)?,
        super::read(values, d, index)?,
    ];
    let da = quotient(
        [
            super::read(tangents, a, index)?,
            super::read(tangents, b, index)?,
        ],
        denominator,
    );
    let db = quotient(
        [
            super::read(tangents, c, index)?,
            super::read(tangents, d, index)?,
        ],
        denominator,
    );
    let q = evaluate(operands, values, index)?;
    let derivative = [
        da[0] - (q[0] * db[0] - q[1] * db[1]),
        da[1] - (q[0] * db[1] + q[1] * db[0]),
    ];
    Ok(derivative[usize::from(imaginary)])
}

pub(super) fn vjp(
    operands: [super::ValueId; 4],
    values: &[f64],
    index: usize,
    imaginary: bool,
    seed: f64,
) -> Result<[f64; 4], super::Diagnostic> {
    let denominator = [
        super::read(values, operands[2], index)?,
        -super::read(values, operands[3], index)?,
    ];
    let weight = if imaginary { [0., seed] } else { [seed, 0.] };
    let a = quotient(weight, denominator);
    let q = evaluate(operands, values, index)?;
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
        .map(|(a, b)| quotient([a, b], [br, bi])[component]);
    let coefficients = ar
        .coefficients
        .iter()
        .zip(&ai.coefficients)
        .map(|(&a, &b)| quotient([a, b], [br, bi])[component])
        .collect();
    AffineSummary::finite(constant, coefficients, index)
}

fn exponent(value: f64) -> i32 {
    if value == 0. {
        return 0;
    }
    let bits = value.to_bits();
    let stored = ((bits >> 52) & 0x7ff) as i32;
    if stored == 0 {
        // A subnormal is its integer significand times 2^-1074.
        63 - bits.leading_zeros() as i32 - 1074
    } else {
        stored - 1023
    }
}

fn scale(mut value: f64, mut exponent: i32) -> f64 {
    // Every factor is normal and exactly represented. Applying bounded chunks
    // avoids first constructing 2^1074 or 2^-2097 as an f64.
    while exponent != 0 {
        let part = exponent.clamp(-512, 512);
        value *= 2_f64.powi(part);
        exponent -= part;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::quotient;

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
            assert_eq!(summary.constant, Some(0.));
            assert_eq!(summary.coefficients, expected);
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
