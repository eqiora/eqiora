use super::*;
use num_complex::Complex64 as C;

#[test]
fn coupled_boundary_elimination_and_its_direction_preserve_real_and_complex_pairings() {
    check(1., [7., 11., 13.], [2., 3., 5.], [0.5, 3.5, 11., 21.5]);
    check(
        C::new(2., 1.),
        [C::new(7., 2.), C::new(11., -3.), C::new(13., 5.)],
        [C::new(2., -1.), C::new(3., 2.), C::new(5., -4.)],
        [
            C::new(1., 0.5),
            C::new(7., 3.5),
            C::new(13.5, 12.),
            C::new(40.5, 27.),
        ],
    );
}
fn check<S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + From<f64>>(
    phase: S,
    values: [S; 3],
    partials: [S; 3],
    expected: [S; 4],
) {
    // T=phase*[[4,1,2],[3,0,2],[5,1,3]]. Only its N,N block scales by 1+t.
    // Solving P*T_NN=T_DN gives P=(-1/2,1)/(1+t); a row pivot is essential.
    // E=phase*(4-(7/2)/(1+t)); l=E*g_D+P*g_N.
    // Differentiate these rational expressions, with the independently supplied
    // data directions, to obtain the literal values above.
    let matrix = [[4., 1., 2.], [3., 0., 2.], [5., 1., 3.]]
        .map(|row| row.map(|x| phase * <S as From<f64>>::from(x)).to_vec())
        .to_vec();
    let derivative = [[0., 0., 0.], [0., 0., 2.], [0., 1., 3.]]
        .map(|row| row.map(|x| phase * <S as From<f64>>::from(x)).to_vec())
        .to_vec();
    let data = [
        CartesianBoundaryValue::Essential(values[0]),
        CartesianBoundaryValue::Natural(values[1]),
        CartesianBoundaryValue::Natural(values[2]),
    ];
    let direction = [
        CartesianBoundaryValue::Essential(partials[0]),
        CartesianBoundaryValue::Natural(partials[1]),
        CartesianBoundaryValue::Natural(partials[2]),
    ];
    let prescribed = vec![S::zero(); data.len()];
    let (a, b) = response(&matrix, &data, &prescribed, None).unwrap();
    let (da, db) = response(
        &matrix,
        &data,
        &prescribed,
        Some((&derivative, &direction, &prescribed)),
    )
    .unwrap();
    for (actual, expected) in [a[0][0], da[0][0], b[0], db[0]].into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
    for i in 0..3 {
        for j in 0..3 {
            if (i, j) != (0, 0) {
                assert_eq!(a[i][j], S::zero());
                assert_eq!(da[i][j], S::zero());
            }
        }
    }
    assert_eq!(&b[1..], &values[1..]);
    assert_eq!(&db[1..], &partials[1..]);
}

#[test]
fn singular_coupled_boundary_rejects_but_unused_natural_trace_is_not_inverted() {
    let data = [
        CartesianBoundaryValue::Essential(7.),
        CartesianBoundaryValue::Natural(0.),
    ];
    assert!(
        response(&vec![vec![2., 1.], vec![0., 0.]], &data, &[0.; 2], None)
            .unwrap_err()
            .message()
            .contains("singular")
    );
    let (matrix, rhs) = response(&vec![vec![2., 0.], vec![0., 0.]], &data, &[0.; 2], None).unwrap();
    assert_eq!(matrix, vec![vec![2., 0.], vec![0., 0.]]);
    assert_eq!(rhs, vec![14., 0.]);
    let bad = [
        CartesianBoundaryValue::Natural(0.),
        CartesianBoundaryValue::Natural(0.),
    ];
    assert!(
        response(&matrix, &data, &[0.; 2], Some((&matrix, &bad, &[0.; 2])))
            .unwrap_err()
            .message()
            .contains("boundary kind")
    );
}
