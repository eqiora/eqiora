//! Fixed upper-triangular diagonal-Schur action over exact caller-owned groups.
use std::collections::BTreeMap;

use crate::sparse_lu_factor::{
    SparseLuNumericFactor, SparseLuScalar, SparseLuSymbolicFactor, factor_numeric, factor_symbolic,
    solve_factored_oriented,
};
use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_solver::{
    CanonicalCsrSystemView, CsrStorage, LinearOperatorOrientation, LinearOperatorProperties,
};

struct Sparse<S> {
    rows: usize,
    columns: usize,
    offsets: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<S>,
}
impl<S> CsrStorage<S> for Sparse<S> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn columns(&self) -> usize {
        self.columns
    }
    fn row_offsets(&self) -> &[usize] {
        &self.offsets
    }
    fn column_indices(&self) -> &[usize] {
        &self.indices
    }
    fn values(&self) -> &[S] {
        &self.values
    }
}
struct Factors<S> {
    symbolic: SparseLuSymbolicFactor,
    numeric: SparseLuNumericFactor<S>,
}
impl<S: SparseLuScalar> Factors<S> {
    fn new(matrix: &Sparse<S>) -> Result<Self, Diagnostic> {
        let system = CanonicalCsrSystemView::new(
            matrix,
            &vec![S::zero(); matrix.rows],
            LinearOperatorProperties::General,
        )?;
        let symbolic = factor_symbolic(&system)?;
        let numeric = factor_numeric(&symbolic, &system)?;
        // Faer's numeric preparation may retain a zero pivot. The raw fixed
        // triangular action on zero exposes division by that pivot as a
        // nonfinite value; it performs no RHS-dependent refinement or choice.
        solve_factored_oriented(
            &symbolic,
            &numeric,
            &vec![S::zero(); matrix.rows],
            LinearOperatorOrientation::Normal,
        )?;
        Ok(Self { symbolic, numeric })
    }
    fn solve(&self, rhs: &[S]) -> Result<Vec<S>, Diagnostic> {
        solve_factored_oriented(
            &self.symbolic,
            &self.numeric,
            rhs,
            LinearOperatorOrientation::Normal,
        )
    }
}
/// Both fixed factors form one aggregate; callers must commit/discard this whole value.
pub(super) struct UpperDiagonalSchur<S> {
    groups: [Vec<usize>; 2],
    a: Factors<S>,
    schur: Factors<S>,
    c: Sparse<S>,
    dimension: usize,
}
impl<S: SparseLuScalar> UpperDiagonalSchur<S> {
    pub(super) fn new(
        system: &CanonicalCsrSystemView<S>,
        groups: [Vec<usize>; 2],
        orientation: LinearOperatorOrientation,
        maximum_entries: usize,
        maximum_products: usize,
    ) -> Result<Self, Diagnostic> {
        let n = system.rows();
        if orientation != LinearOperatorOrientation::Normal
            || n != system.columns()
            || groups.iter().any(Vec::is_empty)
        {
            return Err(invalid(
                "upper diagonal-Schur requires a normal square action and two nonempty exact groups",
            ));
        }
        let mut owner = vec![None; n];
        for (g, group) in groups.iter().enumerate() {
            for (local, &global) in group.iter().enumerate() {
                if global >= n || owner[global].replace((g, local)).is_some() {
                    return Err(invalid(
                        "upper diagonal-Schur groups overlap or contain foreign coordinates",
                    ));
                }
            }
        }
        if owner.iter().any(Option::is_none) {
            return Err(invalid("upper diagonal-Schur groups omit coordinates"));
        }
        let sizes = [groups[0].len(), groups[1].len()];
        let mut blocks: [Vec<BTreeMap<usize, S>>; 4] =
            std::array::from_fn(|i| vec![BTreeMap::new(); sizes[i / 2]]);
        let mut entries = 0usize;
        for row in 0..n {
            let (g, r) = owner[row].ok_or_else(|| invalid("missing row owner"))?;
            for k in system.row_offsets()[row]..system.row_offsets()[row + 1] {
                let (h, c) = owner[system.column_indices()[k]]
                    .ok_or_else(|| invalid("missing column owner"))?;
                entries = entries
                    .checked_add(1)
                    .filter(|&v| v <= maximum_entries)
                    .ok_or_else(|| invalid("upper diagonal-Schur sparse entry budget exceeded"))?;
                blocks[g * 2 + h][r].insert(c, system.values()[k]);
            }
        }
        let mut inverse = Vec::with_capacity(sizes[0]);
        for (i, row) in blocks[0].iter().enumerate() {
            let diagonal = row.get(&i).copied().unwrap_or(S::zero());
            if diagonal == S::zero() || !diagonal.is_finite() {
                return Err(invalid(
                    "upper diagonal-Schur requires finite nonzero A diagonal",
                ));
            }
            let value = S::one() / diagonal;
            if !value.is_finite() {
                return Err(invalid(
                    "upper diagonal-Schur inverse diagonal is nonfinite",
                ));
            }
            inverse.push(value);
        }
        let mut schur = blocks[3].clone();
        let mut products = 0usize;
        for (row, drow) in blocks[2].iter().enumerate() {
            for (&k, &d) in drow {
                for (&col, &c) in &blocks[1][k] {
                    products = products
                        .checked_add(1)
                        .filter(|&v| v <= maximum_products)
                        .ok_or_else(|| invalid("upper diagonal-Schur product budget exceeded"))?;
                    if !schur[row].contains_key(&col) {
                        entries = entries
                            .checked_add(1)
                            .filter(|&v| v <= maximum_entries)
                            .ok_or_else(|| invalid("upper diagonal-Schur fill budget exceeded"))?;
                    }
                    let previous = schur[row].get(&col).copied().unwrap_or(S::zero());
                    let value = previous - d * inverse[k] * c;
                    if !value.is_finite() {
                        return Err(invalid("upper diagonal-Schur product is nonfinite"));
                    }
                    schur[row].insert(col, value);
                }
            }
        }
        let a = Factors::new(&sparse(&blocks[0], sizes[0]))?;
        let schur = Factors::new(&sparse(&schur, sizes[1]))?;
        Ok(Self {
            groups,
            a,
            schur,
            c: sparse(&blocks[1], sizes[1]),
            dimension: n,
        })
    }
    pub(super) fn apply(&self, input: &[S], output: &mut [S]) -> Result<(), Diagnostic> {
        if input.len() != self.dimension
            || output.len() != self.dimension
            || input.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "upper diagonal-Schur action requires matching finite vectors",
            ));
        }
        let z2 = self
            .schur
            .solve(&self.groups[1].iter().map(|&i| input[i]).collect::<Vec<_>>())?;
        let mut rhs = self.groups[0].iter().map(|&i| input[i]).collect::<Vec<_>>();
        let mut image = vec![S::zero(); rhs.len()];
        (&self.c as &dyn CsrStorage<S>).apply_rows(0..rhs.len(), &z2, &mut image)?;
        for (value, c) in rhs.iter_mut().zip(image) {
            *value = *value - c;
        }
        let z1 = self.a.solve(&rhs)?;
        for (group, values) in self.groups.iter().zip([z1, z2]) {
            for (&i, value) in group.iter().zip(values) {
                output[i] = value;
            }
        }
        Ok(())
    }
}
fn sparse<S: Copy>(rows: &[BTreeMap<usize, S>], columns: usize) -> Sparse<S> {
    let mut result = Sparse {
        rows: rows.len(),
        columns,
        offsets: vec![0],
        indices: Vec::new(),
        values: Vec::new(),
    };
    for row in rows {
        for (&col, &value) in row {
            result.indices.push(col);
            result.values.push(value);
        }
        result.offsets.push(result.values.len());
    }
    result
}
fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> CanonicalCsrSystemView<f64> {
        // Local order is A coordinates [2,0], Schur coordinates [3,1].
        let local = [
            [2., 1., 1., 0.],
            [0., 4., 2., 1.],
            [3., 0., 5., 1.],
            [0., 2., 1., 3.],
        ];
        let map = [2, 0, 3, 1];
        let mut rows = vec![BTreeMap::new(); 4];
        for i in 0..4 {
            for j in 0..4 {
                if local[i][j] != 0. {
                    rows[map[i]].insert(map[j], local[i][j]);
                }
            }
        }
        CanonicalCsrSystemView::new(
            &sparse(&rows, 4),
            &[0.; 4],
            LinearOperatorProperties::General,
        )
        .unwrap()
    }
    #[test]
    fn exact_noncontiguous_upper_action_and_linearity() {
        let action = UpperDiagonalSchur::new(
            &fixture(),
            [vec![2, 0], vec![3, 1]],
            LinearOperatorOrientation::Normal,
            64,
            64,
        )
        .unwrap();
        assert!(action.apply(&[0.; 3], &mut [0.; 4]).is_err());
        assert!(action.apply(&[f64::NAN, 0., 0., 0.], &mut [0.; 4]).is_err());
        // Fixed factors only: no RHS-adaptive refinement or rescaling. For these
        // bounded dyadic vectors, 2x2 partial-pivot factor/backsolve backward
        // error 4 gamma_12 ||M||inf, kappa(A)<=2.5, kappa(S)<=1.8 and
        // Schur rounding <=35u give propagated action error <240u.
        // Reference/slack 512u <5.7e-14 supports 1e-13; three action
        // errors plus dyadic arithmetic <2e-13 supports 1e-12 linearity.
        // This is independent of the original-K residual acceptance gate.
        let r = [2. / 4., 4. / 4., 1. / 4., 3. / 4.];
        let mut ar = [0.; 4];
        action.apply(&r, &mut ar).unwrap();
        for (actual, expected) in ar.iter().zip([-1. / 40., 2. / 5., 7. / 80., 1. / 10.]) {
            assert!((actual - expected).abs() <= 1e-13);
        }
        let r = [2. / 8., 4. / 8., 1. / 8., 3. / 8.];
        let s = [1. / 8., -1. / 8., -2. / 8., 1. / 8.];
        action.apply(&r, &mut ar).unwrap();
        let mut as_ = [0.; 4];
        action.apply(&s, &mut as_).unwrap();
        let combined = std::array::from_fn::<_, 4, _>(|i| r[i] - s[i]);
        let mut applied = [0.; 4];
        action.apply(&combined, &mut applied).unwrap();
        for i in 0..4 {
            assert!((applied[i] - (ar[i] - as_[i])).abs() <= 1e-12);
        }
        // Predeclared cancellation and mixed-sign inputs, local coordinates.
        for local in [
            [1. / 8., -1. / 8., 1. / 8., -1. / 8.],
            [-1., 1., -1. / 2., 1. / 2.],
            [0., 0., 0., 0.],
        ] {
            let input = [local[1], local[3], local[0], local[2]];
            action.apply(&input, &mut applied).unwrap();
            let z20 = 2. * local[2] / 7. - 4. * local[3] / 35.;
            let z21 = 2. * local[3] / 5.;
            let f0 = local[0] - z20;
            let f1 = local[1] - 2. * z20 - z21;
            let expected = [f1 / 4., z21, f0 / 2. - f1 / 8., z20];
            for i in 0..4 {
                assert!((applied[i] - expected[i]).abs() <= 1e-13);
            }
        }
    }
    #[test]
    fn rejects_singular_factors_and_zero_diagonal() {
        for local in [
            [
                [0., 1., 0., 0.],
                [1., 2., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
            [
                [1., 1., 0., 0.],
                [1., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 1.],
                [0., 0., 1., 1.],
            ],
        ] {
            let mut rows = vec![BTreeMap::new(); 4];
            for i in 0..4 {
                for j in 0..4 {
                    if local[i][j] != 0. {
                        rows[i].insert(j, local[i][j]);
                    }
                }
            }
            let system = CanonicalCsrSystemView::new(
                &sparse(&rows, 4),
                &[0.; 4],
                LinearOperatorProperties::General,
            )
            .unwrap();
            assert!(
                UpperDiagonalSchur::new(
                    &system,
                    [vec![0, 1], vec![2, 3]],
                    LinearOperatorOrientation::Normal,
                    64,
                    64
                )
                .is_err()
            );
        }
    }
    #[test]
    fn rejects_wrong_orientation_partition_and_resource_limits() {
        let system = fixture();
        for orientation in [
            LinearOperatorOrientation::Transposed,
            LinearOperatorOrientation::ConjugateTransposed,
        ] {
            assert!(
                UpperDiagonalSchur::new(&system, [vec![2, 0], vec![3, 1]], orientation, 64, 64)
                    .is_err()
            );
        }
        for groups in [[vec![0, 1], vec![1, 2, 3]], [vec![0], vec![1, 2]]] {
            assert!(
                UpperDiagonalSchur::new(&system, groups, LinearOperatorOrientation::Normal, 64, 64)
                    .is_err()
            );
        }
        for (entries, products) in [(1, 64), (64, 0)] {
            assert!(
                UpperDiagonalSchur::new(
                    &system,
                    [vec![2, 0], vec![3, 1]],
                    LinearOperatorOrientation::Normal,
                    entries,
                    products
                )
                .is_err()
            );
        }
    }
}
