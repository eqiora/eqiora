//! Tiny exact results must survive cancellation in all sparse action orientations.
use eqiora_core::Scalar;
use eqiora_solver::{CsrStorage, LinearOperatorOrientation as Orientation};
use num_complex::{Complex32, Complex64, ComplexFloat};

struct Matrix<S>([S; 9]);
impl<S> CsrStorage<S> for Matrix<S> {
    fn rows(&self) -> usize {
        3
    }
    fn columns(&self) -> usize {
        3
    }
    fn row_offsets(&self) -> &[usize] {
        &[0, 3, 6, 9]
    }
    fn column_indices(&self) -> &[usize] {
        &[0, 1, 2, 0, 1, 2, 0, 1, 2]
    }
    fn values(&self) -> &[S] {
        &self.0
    }
}
fn check<S: Scalar + ComplexFloat + Sync>(values: [S; 9], one: S, zero: S, expected: [S; 3]) {
    let matrix = Matrix(values);
    let storage = &matrix as &dyn CsrStorage<S>;
    for orientation in [
        Orientation::Normal,
        Orientation::Transposed,
        Orientation::ConjugateTransposed,
    ] {
        let mut actual = [zero; 3];
        storage
            .apply_oriented(orientation, &[one; 3], &mut actual)
            .unwrap();
        assert_eq!(
            actual,
            if orientation == Orientation::ConjugateTransposed {
                expected.map(|v| v.conj())
            } else {
                expected
            }
        );
    }
    let mut row = [zero];
    storage.apply_rows(2..3, &[one; 3], &mut row).unwrap();
    assert_eq!(row, [expected[2]]);
}
#[test]
fn real_and_complex_storage_preserves_small_components_in_both_precisions() {
    let large = 2f64.powi(54);
    let values = [large, 1., -large, 1., 2., 3., -large, 3., large];
    check(values, 1., 0., [1., 6., 3.]);
    check(values.map(|v| v as f32), 1f32, 0f32, [1f32, 6f32, 3f32]);
    let phase = Complex64::new(1., 2.);
    check(
        values.map(|v| phase * v),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        [phase, 6. * phase, 3. * phase],
    );
    let phase = Complex32::new(1., 2.);
    check(
        values.map(|v| phase * (v as f32)),
        Complex32::new(1., 0.),
        Complex32::new(0., 0.),
        [phase, 6. * phase, 3. * phase],
    );
}

#[test]
fn captured_transpose_row_partitions_preserve_compensated_cancellation() {
    use eqiora_solver::{
        CanonicalCsrSystemView, LinearOperator, LinearOperatorProperties, Oriented,
    };
    let large = 2f64.powi(54);
    let phase = Complex64::new(1., 2.);
    let storage = Matrix([large, 1., -large, 1., 2., 3., -large, 3., large].map(|v| phase * v));
    let one = Complex64::new(1., 0.);
    let zero = Complex64::new(0., 0.);
    let system =
        CanonicalCsrSystemView::new(&storage, &[zero; 3], LinearOperatorProperties::General)
            .unwrap();
    for orientation in [Orientation::Transposed, Orientation::ConjugateTransposed] {
        let oriented = Oriented::new(&system, orientation).unwrap();
        let mut actual = [zero; 3];
        let action = oriented.row_action().unwrap();
        action
            .apply_rows(0..1, &[one; 3], &mut actual[..1])
            .unwrap();
        action
            .apply_rows(1..3, &[one; 3], &mut actual[1..])
            .unwrap();
        let expected = [phase, 6. * phase, 3. * phase];
        assert_eq!(
            actual,
            if orientation == Orientation::ConjugateTransposed {
                expected.map(|v| v.conj())
            } else {
                expected
            }
        );
    }
}
