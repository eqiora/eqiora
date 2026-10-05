use super::*;
use eqiora_core::{DimExponents, Id, entity::kinds};
use eqiora_schema::kernel::{ComparisonOp, ExprDagBuilder};

fn scalar(domain: ScalarDomain, dimension: DimExponents, re: f64, im: f64) -> ValueLiteral {
    ValueLiteral::new(ValueType::scalar(domain, dimension).unwrap(), [(re, im)]).unwrap()
}

fn evaluate(builder: ExprDagBuilder, roots: &[ExprId]) -> Result<Vec<ValueLiteral>, Diagnostic> {
    let dag = builder.finish(roots.iter().copied()).unwrap();
    evaluate_selected(
        Id::<kinds::Relation>::new().erase(),
        &dag,
        roots,
        &mut |_| None,
    )
}

#[test]
fn equation_residuals_keep_complete_channel_parts_and_physical_units() {
    let unit = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let ty = ValueType::scalar(ScalarDomain::Complex, unit)
        .unwrap()
        .array(6)
        .unwrap();
    let left = ValueLiteral::new(
        ty.clone(),
        (0..6).map(|i| (i as f64 + 2., 2. * i as f64 - 1.)),
    )
    .unwrap();
    let right = ValueLiteral::new(ty, [(1., -2.); 6]).unwrap();
    let real = scalar(ScalarDomain::Real, unit, 7., 0.);
    let zero = scalar(ScalarDomain::Real, unit, 0., 0.);
    assert_eq!(
        numerical_differences(vec![left.clone(), right.clone(), real, zero]).unwrap(),
        [1., 1., 2., 3., 3., 5., 4., 7., 5., 9., 6., 11., 7.]
    );
    assert!(numerical_differences(vec![left.clone()]).is_err());
    assert!(
        numerical_differences(vec![left, scalar(ScalarDomain::Complex, unit, 1., -2.)]).is_err()
    );
    assert!(
        numerical_differences(vec![
            right,
            ValueLiteral::new(
                ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                    .unwrap()
                    .array(6)
                    .unwrap(),
                [(0., 0.); 6],
            )
            .unwrap()
        ])
        .is_err()
    );
    assert!(
        numerical_differences(vec![
            scalar(ScalarDomain::Complex, unit, f64::MAX, 0.),
            scalar(ScalarDomain::Complex, unit, -f64::MAX, 0.),
        ])
        .is_err()
    );
}

#[test]
fn typed_complex_product_division_and_constructor_keep_both_components() {
    let unit = DimExponents::DIMENSIONLESS;
    let mut b = ExprDagBuilder::new();
    let re = b
        .constant(scalar(ScalarDomain::Real, unit, 3., 0.))
        .unwrap();
    let im = b
        .constant(scalar(ScalarDomain::Real, unit, 4., 0.))
        .unwrap();
    let a = b.complex(re, im).unwrap();
    let divisor = b
        .constant(scalar(ScalarDomain::Complex, unit, 1., -2.))
        .unwrap();
    let product = b.mul(a, divisor).unwrap();
    let quotient = b.div(a, divisor).unwrap();
    let negative = b.neg(a).unwrap();
    let squared = b.powi(a, 2).unwrap();
    let embedded = b.add(divisor, re).unwrap();
    let results = evaluate(b, &[product, quotient, negative, squared, embedded]).unwrap();
    // Expand (3+4i)(1-2i), rationalize the quotient, and use i^2=-1.
    let expected = [(11., -2.), (-1., 2.), (-3., -4.), (-7., 24.), (4., -2.)];
    for (result, pair) in results.iter().zip(expected) {
        let actual = result.component(0).unwrap();
        for (observed, expected) in [(actual.0, pair.0), (actual.1, pair.1)] {
            assert!((observed - expected).abs() <= 8. * f64::EPSILON * expected.abs().max(1.));
        }
        assert_eq!(result.value_type().scalar_domain(), ScalarDomain::Complex);
        assert!(result.real_scalar_value().is_none());
    }
}

#[test]
fn complex_channels_share_addition_scaling_indexing_and_dimensions() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let mut b = ExprDagBuilder::new();
    let a = b
        .constant(scalar(ScalarDomain::Complex, length, 1., 2.))
        .unwrap();
    let c = b
        .constant(scalar(ScalarDomain::Complex, length, 3., -4.))
        .unwrap();
    let values = b.array([a, c]).unwrap();
    let r = b
        .constant(scalar(ScalarDomain::Real, length, 2., 0.))
        .unwrap();
    let real_values = b.array([r, r]).unwrap();
    let sum = b.add(values, real_values).unwrap();
    let factor = b
        .constant(scalar(
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
            0.,
            1.,
        ))
        .unwrap();
    let scaled = b.mul(sum, factor).unwrap();
    let selected = b.index(scaled, 1).unwrap();
    let result = evaluate(b, &[scaled, selected]).unwrap();
    assert_eq!(result[0].component(0), Some((-2., 3.)));
    assert_eq!(result[0].component(1), Some((4., 5.)));
    assert_eq!(result[1].component(0), Some((4., 5.)));
    assert_eq!(result[0].value_type().dimension(), length);
    assert_eq!(result[0].value_type().array_rank(), 1);
}

#[test]
fn complex_execution_rejects_ordering_wrong_dimensions_and_nonfinite_results() {
    let unit = DimExponents::DIMENSIONLESS;
    for operation in ["order", "divide-zero", "overflow", "dimension"] {
        let mut b = ExprDagBuilder::new();
        let a = b
            .constant(scalar(ScalarDomain::Complex, unit, 1., 2.))
            .unwrap();
        let root = match operation {
            "order" => b.compare(ComparisonOp::Less, a, a).unwrap(),
            "divide-zero" => {
                let zero = b
                    .constant(scalar(ScalarDomain::Complex, unit, -0., -0.))
                    .unwrap();
                b.div(a, zero).unwrap()
            }
            "overflow" => {
                let huge = b
                    .constant(scalar(ScalarDomain::Complex, unit, 1e308, 1e308))
                    .unwrap();
                b.mul(huge, huge).unwrap()
            }
            _ => {
                let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
                let real = b
                    .constant(scalar(ScalarDomain::Real, length, 1., 0.))
                    .unwrap();
                b.add(a, real).unwrap()
            }
        };
        let error = evaluate(b, &[root]).unwrap_err();
        let expected_code = match operation {
            "order" => codes::NOT_IMPLEMENTED,
            "dimension" => codes::DIMENSION_MISMATCH,
            _ => codes::NONFINITE_EVALUATION,
        };
        assert_eq!(error.code(), expected_code, "{operation}: {error:?}");
    }
}

#[test]
fn complex_division_does_not_square_extreme_denominators() {
    let unit = DimExponents::DIMENSIONLESS;
    for scale in [1e-300, 1e300] {
        let mut b = ExprDagBuilder::new();
        let numerator = b
            .constant(scalar(ScalarDomain::Complex, unit, scale, 2. * scale))
            .unwrap();
        let denominator = b
            .constant(scalar(ScalarDomain::Complex, unit, scale, 0.))
            .unwrap();
        let quotient = b.div(numerator, denominator).unwrap();
        let result = evaluate(b, &[quotient]).unwrap();
        let (real, imag) = result[0].component(0).unwrap();
        // The common nonzero scale cancels exactly in the mathematical quotient.
        assert!((real - 1.).abs() <= 8. * f64::EPSILON);
        assert!((imag - 2.).abs() <= 16. * f64::EPSILON);
    }
}

#[test]
fn complex_quotients_and_negative_powers_handle_extreme_norms() {
    let unit = DimExponents::DIMENSIONLESS;
    for scale in [f64::from_bits(1), 1e-300, 1e300, f64::MAX] {
        let mut b = ExprDagBuilder::new();
        let z = b
            .constant(scalar(ScalarDomain::Complex, unit, scale, scale))
            .unwrap();
        let quotient = b.div(z, z).unwrap();
        let result = evaluate(b, &[quotient]).unwrap();
        let (real, imag) = result[0].component(0).unwrap();
        assert!((real - 1.).abs() <= 8. * f64::EPSILON, "{scale}: {real}");
        assert!(imag.abs() <= 8. * f64::EPSILON, "{scale}: {imag}");
    }
    for scale in [1e-300, 1e300] {
        let mut b = ExprDagBuilder::new();
        let z = b
            .constant(scalar(ScalarDomain::Complex, unit, scale, 0.))
            .unwrap();
        let inverse = b.powi(z, -1).unwrap();
        let result = evaluate(b, &[inverse]).unwrap();
        let (real, imag) = result[0].component(0).unwrap();
        assert!((real * scale - 1.).abs() <= 8. * f64::EPSILON);
        assert_eq!(imag, 0.);
    }
}
