use super::*;
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorOrientation as O, LinearOperatorProperties,
};
use num_complex::Complex64 as C;

#[test]
fn rectangular_complex32_actions_preserve_empty_rows_and_distinct_orientations() {
    use num_complex::Complex32 as F;
    // [1+i, 0, 2-3i; 0, 0, 0], including one explicit zero entry.
    let matrix = CsrMatrix::from_sorted_csr(
        2,
        3,
        vec![0, 3, 3],
        vec![0, 1, 2],
        vec![F::new(1., 1.), F::new(0., 0.), F::new(2., -3.)],
    )
    .unwrap();
    let input = [F::new(2., -1.), F::new(9., 2.), F::new(-1., 4.)];
    assert_eq!(
        matrix.multiply(&input).unwrap(),
        [F::new(13., 12.), F::new(0., 0.)]
    );
    let mut empty_row = [F::new(9., 9.)];
    matrix.apply_rows(1..2, &input, &mut empty_row).unwrap();
    assert_eq!(empty_row, [F::new(0., 0.)]);
    for (orientation, expected) in [
        (
            O::Transposed,
            [F::new(1., 5.), F::new(0., 0.), F::new(12., -5.)],
        ),
        (
            O::ConjugateTransposed,
            [F::new(5., -1.), F::new(0., 0.), F::new(0., 13.)],
        ),
    ] {
        let mut output = [F::new(8., 9.); 3];
        matrix
            .apply_oriented(orientation, &[F::new(3., 2.), F::new(7., -1.)], &mut output)
            .unwrap();
        assert_eq!(output, expected);
    }
}

#[test]
fn complex_csr_assembly_and_shared_solver_view_distinguish_transpose_from_adjoint() {
    let local = LocalContribution::new(
        2,
        2,
        vec![
            C::new(1., 1.),
            C::new(2., -1.),
            C::new(-3., 2.),
            C::new(4., 1.),
        ],
        vec![C::new(4., 8.), C::new(-11., 18.)],
    )
    .unwrap();
    let map = AssemblyMap::new(
        vec![Some(DofId::new(0)), Some(DofId::new(1))],
        vec![
            LocalUnknown::Free(DofId::new(0)),
            LocalUnknown::Free(DofId::new(1)),
        ],
    )
    .unwrap();
    let mut assembler = CooAssembler::new(2).unwrap();
    assembler.scatter(&map, &local).unwrap();
    let system = assembler.finish().unwrap();
    let canonical =
        CanonicalCsrSystemView::new(&system, system.rhs(), LinearOperatorProperties::General)
            .unwrap();
    let input = [C::new(2., -1.), C::new(-1., 3.)];
    // Direct Gaussian-integer multiplication, with neither symmetry nor
    // Hermiticity: the three orientations must produce different values.
    for (orientation, expected) in [
        (O::Normal, [C::new(4., 8.), C::new(-11., 18.)]),
        (O::Transposed, [C::new(0., -10.), C::new(-4., 7.)]),
        (O::ConjugateTransposed, [C::new(10., -10.), C::new(4., 13.)]),
    ] {
        for operator in [
            system.matrix() as &dyn OrientedLinearOperator<Scalar = C>,
            &canonical,
        ] {
            let mut actual = [C::new(0., 0.); 2];
            operator
                .apply_oriented(orientation, &input, &mut actual)
                .unwrap();
            assert_eq!(actual, expected);
        }
    }
    assert_eq!(system.rhs(), &[C::new(4., 8.), C::new(-11., 18.)]);
}

#[test]
fn complex_csr_scatter_failure_is_atomic_and_exact_cancellation_preserves_empty_rows() {
    let map = AssemblyMap::new(
        vec![Some(DofId::new(0))],
        vec![LocalUnknown::Free(DofId::new(0))],
    )
    .unwrap();
    let local =
        LocalContribution::new(1, 1, vec![C::new(0., f64::MAX)], vec![C::new(1., -2.)]).unwrap();
    let mut assembler = CooAssembler::new(1).unwrap();
    assembler.scatter(&map, &local).unwrap();
    let before = assembler.clone().finish().unwrap();
    assert!(assembler.scatter(&map, &local).is_err());
    assert_eq!(assembler.finish().unwrap(), before);
    let mut cancelled = CooAssembler::new(1).unwrap();
    cancelled.scatter(&map, &local).unwrap();
    let negative =
        LocalContribution::new(1, 1, vec![C::new(0., -f64::MAX)], vec![C::new(0., 0.)]).unwrap();
    cancelled.scatter(&map, &negative).unwrap();
    let cancelled = cancelled.finish().unwrap();
    assert_eq!(cancelled.matrix().row_offsets(), &[0, 0]);
    assert!(cancelled.matrix().values().is_empty());
    assert_eq!(cancelled.rhs(), &[C::new(1., -2.)]);
}
