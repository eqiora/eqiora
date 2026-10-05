use super::*;
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorOrientation as O, LinearOperatorProperties,
};
use num_complex::Complex64 as C;

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
        CanonicalCsrSystemView::new(&system, LinearOperatorProperties::General).unwrap();
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
fn complex_csr_scatter_failure_is_atomic_and_exact_cancellation_rejects_empty_rows() {
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
    assert!(cancelled.finish().is_err());
}
