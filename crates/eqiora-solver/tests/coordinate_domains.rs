//! Exact block equivalence, independent of either coordinate conversion.
use eqiora_solver::{
    CanonicalCsrSystemView, CsrStorage, LinearOperator, LinearOperatorProperties as P,
};
use num_complex::Complex64 as C;

struct Storage {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
}
impl CsrStorage for Storage {
    fn rows(&self) -> usize {
        self.offsets.len() - 1
    }
    fn columns(&self) -> usize {
        self.rows()
    }
    fn row_offsets(&self) -> &[usize] {
        &self.offsets
    }
    fn column_indices(&self) -> &[usize] {
        &self.columns
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
}
fn dense(values: &[f64], rhs: &[f64]) -> CanonicalCsrSystemView {
    let n = rhs.len();
    let storage = Storage {
        offsets: (0..=n).map(|i| i * n).collect(),
        columns: (0..n).flat_map(|_| 0..n).collect(),
        values: values.to_vec(),
    };
    CanonicalCsrSystemView::new(&storage, rhs, P::General).unwrap()
}

#[test]
fn paired_real_blocks_recover_the_complete_nonsymmetric_complex_action() {
    // Independently expand A=[[1+2i,3-4i],[5+6i,7+8i]].
    let values = [
        1., -2., 3., 4., 2., 1., -4., 3., 5., -6., 7., -8., 6., 5., 8., 7.,
    ];
    let real = dense(&values, &[9., 10., 11., 12.]);
    let complex = real.complex_coordinate_system().unwrap().unwrap();
    assert_eq!(
        complex.values(),
        &[
            C::new(1., 2.),
            C::new(3., -4.),
            C::new(5., 6.),
            C::new(7., 8.)
        ]
    );
    assert_eq!(
        complex.right_hand_side(),
        &[C::new(9., 10.), C::new(11., 12.)]
    );
    assert_eq!(complex.properties(), P::General);
    assert_ne!(
        real.agreement_fingerprint(),
        complex.agreement_fingerprint()
    );
    let mut actual = [C::new(0., 0.); 2];
    complex
        .apply(&[C::new(2., -1.), C::new(-3., 4.)], &mut actual)
        .unwrap();
    assert_eq!(actual, [C::new(11., 27.), C::new(-37., 11.)]);
    // Every block equality is exact, including changes smaller than ordinary
    // numerical acceptance tolerances. No approximate C-linearity is inferred.
    for index in 0..values.len() {
        let mut changed = values;
        changed[index] = changed[index].next_up();
        assert!(
            dense(&changed, &[9., 10., 11., 12.])
                .complex_coordinate_system()
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn implicit_zeros_are_checked_as_parts_of_the_same_block() {
    // A=[[2,i],[-2i,3]], with no explicit zero stored in its real coordinates.
    let mut storage = Storage {
        offsets: vec![0, 2, 4, 6, 8],
        columns: vec![0, 3, 1, 2, 1, 2, 0, 3],
        values: vec![2., -1., 2., 1., 2., 3., -2., 3.],
    };
    let captured = CanonicalCsrSystemView::new(&storage, &[1., 2., 3., 4.], P::General).unwrap();
    let complex = captured.complex_coordinate_system().unwrap().unwrap();
    assert_eq!(
        complex.values(),
        &[
            C::new(2., 0.),
            C::new(0., 1.),
            C::new(0., -2.),
            C::new(3., 0.)
        ]
    );
    // Remove just one nonzero partner. Its implicit replacement is zero, so
    // the resulting sparse map must not be silently projected to complex.
    storage.values.remove(4);
    storage.columns.remove(4);
    storage.offsets[3] -= 1;
    storage.offsets[4] -= 1;
    let changed = CanonicalCsrSystemView::new(&storage, &[1., 2., 3., 4.], P::General).unwrap();
    assert!(changed.complex_coordinate_system().unwrap().is_none());
}

#[test]
fn conjugation_and_unpaired_coordinates_remain_real_linear() {
    assert!(
        dense(&[1., 0., 0., -1.], &[2., 3.])
            .complex_coordinate_system()
            .unwrap()
            .is_none()
    );
    assert!(
        dense(&[1.], &[2.])
            .complex_coordinate_system()
            .unwrap()
            .is_none()
    );
    let storage = Storage {
        offsets: vec![0, 1, 2],
        columns: vec![0, 1],
        values: vec![2., 2.],
    };
    let real =
        CanonicalCsrSystemView::new(&storage, &[1., 3.], P::SymmetricPositiveDefinite).unwrap();
    assert_eq!(
        real.complex_coordinate_system()
            .unwrap()
            .unwrap()
            .properties(),
        P::General
    );
}
