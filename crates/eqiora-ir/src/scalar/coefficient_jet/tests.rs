use super::*;
use num_rational::BigRational as Rational;
use num_traits::{Signed, ToPrimitive, Zero};

#[test]
fn quotient_coefficients_match_independent_rational_derivatives_across_scales() {
    let exact = |value| Rational::from_float(value).unwrap();
    let mut seed = 0x76d8_4e91_0ba2_c335u64;
    let mut sample = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let exponent = [-1000, -700, -500, 0, 500, 700, 1000][(seed >> 32) as usize % 7];
        let mantissa = ((seed >> 48) % 7 + 1) as f64 / 4.;
        let sign = if (seed >> 25) & 1 == 0 { 1. } else { -1. };
        sign * mantissa * 2f64.powi(exponent)
    };
    // Through order two, each expanded rational term crosses fewer than 32
    // rounded operations in the coefficient recurrence. gamma_64 also covers
    // alignment rounding; one half-subnormal accounts for final conversion.
    // The bound scales with absolute analytic terms, not the observed error.
    let ku = exact(64. * f64::EPSILON);
    let gamma = &ku / (exact(1.) - &ku);
    let final_round = exact(f64::from_bits(1)) / exact(2.);
    let mut checked = [0; 2];
    for _ in 0..512 {
        let a = std::array::from_fn::<_, 4, _>(|_| sample());
        let b = std::array::from_fn::<_, 4, _>(|_| sample());
        let ar = a.map(exact);
        let br = b.map(exact);
        let b2 = &br[0] * &br[0];
        let b3 = &b2 * &br[0];
        // Differentiate a/b directly, independently of the subset recurrence.
        let references = [
            vec![&ar[1] / &br[0], -&ar[0] * &br[1] / &b2],
            vec![
                &ar[3] / &br[0],
                -&ar[1] * &br[2] / &b2,
                -&ar[2] * &br[1] / &b2,
                -&ar[0] * &br[3] / &b2,
                exact(2.) * &ar[0] * &br[1] * &br[2] / &b3,
            ],
        ];
        for (order, terms) in references.into_iter().enumerate() {
            let expected = terms.iter().cloned().sum::<Rational>();
            if !expected.to_f64().is_some_and(f64::is_finite) {
                continue;
            }
            let n = 1 << (order + 1);
            let actual = Shape::SCALAR
                .jet(
                    &a[..n].iter().copied().map(Some).collect::<Vec<_>>(),
                    &b[..n].iter().copied().map(Some).collect::<Vec<_>>(),
                )
                .unwrap();
            assert!(actual.is_finite());
            let magnitude = terms
                .iter()
                .fold(Rational::zero(), |sum, term| sum + term.abs());
            let bound = &gamma * magnitude + &final_round;
            assert!(
                (exact(actual) - expected).abs() <= bound,
                "order={order}, a={a:?}, b={b:?}"
            );
            checked[order] += 1;
        }
    }
    assert!(checked.into_iter().all(|count| count >= 128));
}

#[test]
fn quotient_work_admission_precedes_gather_and_retains_structural_absence() {
    assert_eq!(
        Shape::SCALAR
            .jet(&[Some(2.), None], &[Some(4.), None])
            .unwrap(),
        0.
    );
    assert!(Shape::SCALAR.jet(&[Some(1.); 3], &[Some(1.); 3]).is_err());
    assert!(
        Shape::SCALAR
            .jet(&[Some(1.); 8192], &[Some(1.); 8192])
            .is_err()
    );
    assert_eq!(
        Shape::SCALAR
            .jet(&[Some(1.); 4096], &[Some(1.); 4096])
            .unwrap(),
        0.
    );
    let mut instructions = vec![Instruction::Constant(eqiora_core::DynQuantity::new(
        1.,
        eqiora_core::DimExponents::DIMENSIONLESS,
    ))];
    let prior = instructions.clone();
    assert!(
        gather(
            &mut instructions,
            &[Some(ValueId(0)); 3],
            &[Some(ValueId(0)); 3],
            Shape::SCALAR
        )
        .is_err()
    );
    assert_eq!(instructions, prior);
}
