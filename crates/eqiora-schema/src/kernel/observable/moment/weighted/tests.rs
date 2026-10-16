use super::*;
use eqiora_core::DimExponents;

fn quantity(value: f64, units: [i32; 7]) -> ValueLiteral {
    ValueLiteral::from_real(
        ValueType::scalar(
            ScalarDomain::Real,
            DimExponents::from_integers(units).unwrap(),
        )
        .unwrap(),
        value,
    )
    .unwrap()
}

const LENGTH: [i32; 7] = [0, 1, 0, 0, 0, 0, 0];
const TIME: [i32; 7] = [0, 0, 1, 0, 0, 0, 0];

fn accumulated(
    moment: ProbabilityMoment,
    weights: &[f64],
) -> (WeightedMoments, Vec<Vec<ValueLiteral>>) {
    let rows = [0.0, 2.0, 2.0]
        .into_iter()
        .map(|x| {
            let mut row = vec![quantity(x, LENGTH)];
            if moment == ProbabilityMoment::Covariance {
                row.push(quantity(3.0 - x, TIME));
            }
            row
        })
        .collect::<Vec<_>>();
    let mut sums = WeightedMoments::new(
        moment,
        &rows[0]
            .iter()
            .map(|v| v.value_type().clone())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for (row, &weight) in rows.iter().zip(weights) {
        sums.push(row, weight).unwrap();
    }
    (sums, rows)
}

#[test]
fn duplicated_occurrences_have_hand_derived_weighted_moments_and_derivatives() {
    // [0,2,2] with weights [1,2,1]: W=4, sum=6, mean=3/2,
    // E[X²]=3, Var=3-9/4=3/4. Y=3-X has Cov(X,Y)=-3/4.
    for (moment, expected, units, derivatives) in [
        (
            ProbabilityMoment::Expectation,
            1.5,
            LENGTH,
            vec![vec![0.25], vec![0.5], vec![0.25]],
        ),
        (
            ProbabilityMoment::Variance,
            0.75,
            [0, 2, 0, 0, 0, 0, 0],
            vec![vec![-0.75], vec![0.5], vec![0.25]],
        ),
        (
            ProbabilityMoment::Covariance,
            -0.75,
            [0, 1, 1, 0, 0, 0, 0],
            vec![vec![0.375, -0.375], vec![-0.25, 0.25], vec![-0.125, 0.125]],
        ),
    ] {
        let (sums, rows) = accumulated(moment, &[1.0, 2.0, 1.0]);
        assert_eq!(sums.value(0).unwrap(), quantity(expected, units));
        for ((row, &weight), expected) in rows.iter().zip(&[1.0, 2.0, 1.0]).zip(derivatives) {
            assert_eq!(sums.derivative(row, weight, 0).unwrap(), expected);
        }
        if moment == ProbabilityMoment::Expectation {
            assert_eq!(sums.sum().unwrap(), quantity(6.0, LENGTH));
        } else {
            assert!(sums.sum().is_err());
        }
        assert!(
            sums.value(1).is_err(),
            "unequal weights do not define an n-1 convention"
        );
    }
}

#[test]
fn finite_count_correction_is_exact_and_explicit() {
    // Equal weights give mean=4/3 and squared deviations sum=8/3.
    // Dividing by n-1=2 gives 4/3, with derivatives [-4/3,2/3,2/3].
    let (sums, rows) = accumulated(ProbabilityMoment::Variance, &[2.0; 3]);
    assert_eq!(
        sums.value(1).unwrap(),
        quantity(4.0 / 3.0, [0, 2, 0, 0, 0, 0, 0])
    );
    for (row, expected) in rows.iter().zip([-4.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]) {
        assert_eq!(sums.derivative(row, 2.0, 1).unwrap(), [expected]);
    }
    assert!(sums.value(3).is_err());
    assert!(sums.value(usize::MAX).is_err());
    let (means, _) = accumulated(ProbabilityMoment::Expectation, &[1.0; 3]);
    assert!(means.value(1).is_err());
}

#[test]
fn streaming_reordering_preserves_cancellation_and_failed_push_is_atomic() {
    let ty = quantity(0.0, LENGTH).value_type().clone();
    for order in [[1e16, 1.0, -1e16], [1e16, -1e16, 1.0], [-1e16, 1.0, 1e16]] {
        let mut sums =
            WeightedMoments::new(ProbabilityMoment::Expectation, std::slice::from_ref(&ty))
                .unwrap();
        assert!(sums.value(0).is_err());
        for value in order {
            sums.push(&[quantity(value, LENGTH)], 1.0).unwrap();
        }
        assert_eq!(sums.sum().unwrap(), quantity(1.0, LENGTH));
        assert_eq!(sums.value(0).unwrap(), quantity(1.0 / 3.0, LENGTH));
        for weight in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(sums.push(&[quantity(7.0, LENGTH)], weight).is_err());
            assert_eq!(sums.sum().unwrap(), quantity(1.0, LENGTH));
        }
        assert!(sums.push(&[quantity(7.0, TIME)], 1.0).is_err());
        assert!(sums.push(&[], 1.0).is_err());
        assert_eq!(sums.sum().unwrap(), quantity(1.0, LENGTH));
    }
    let mut sums = WeightedMoments::new(ProbabilityMoment::Expectation, &[ty]).unwrap();
    sums.push(&[quantity(f64::MAX, LENGTH)], f64::MAX).unwrap();
    assert_eq!(sums.value(0).unwrap(), quantity(f64::MAX, LENGTH));
    assert!(sums.sum().is_err());
}

#[test]
fn mean_error_keeps_units_and_extreme_range_without_squared_float() {
    let (mean, _) = accumulated(ProbabilityMoment::Expectation, &[1.0; 3]);
    assert_eq!(
        mean.mean_standard_error().unwrap(),
        quantity(2.0 / 3.0, LENGTH)
    );
    for x in [0.0, f64::from_bits(1), f64::MIN_POSITIVE, 1.0, f64::MAX] {
        let mut mean = WeightedMoments::new(
            ProbabilityMoment::Expectation,
            &[quantity(x, LENGTH).value_type().clone()],
        )
        .unwrap();
        mean.push(&[quantity(x, LENGTH)], 1.0).unwrap();
        assert!(mean.mean_standard_error().is_err());
        mean.push(&[quantity(-x, LENGTH)], 1.0).unwrap();
        // For two opposite points SE=|x|, even when x² is not binary64.
        assert_eq!(mean.mean_standard_error().unwrap(), quantity(x, LENGTH));
    }
    let (unequal, _) = accumulated(ProbabilityMoment::Expectation, &[1.0, 2.0, 1.0]);
    assert!(unequal.mean_standard_error().is_err());
    let (variance, _) = accumulated(ProbabilityMoment::Variance, &[1.0; 3]);
    assert!(variance.mean_standard_error().is_err());
}
