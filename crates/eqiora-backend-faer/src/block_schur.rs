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

#[derive(Debug)]
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
#[derive(Debug)]
struct Factors<S> {
    symbolic: std::sync::Arc<SparseLuSymbolicFactor>,
    offsets: Vec<usize>,
    indices: Vec<usize>,
    numeric: SparseLuNumericFactor<S>,
}
impl<S: SparseLuScalar> Factors<S> {
    fn new(matrix: &Sparse<S>, previous: Option<&Self>) -> Result<Self, Diagnostic> {
        let system = CanonicalCsrSystemView::new(
            matrix,
            &vec![S::zero(); matrix.rows],
            LinearOperatorProperties::General,
        )?;
        let symbolic = match previous.filter(|previous| {
            previous.offsets == matrix.offsets && previous.indices == matrix.indices
        }) {
            Some(previous) => std::sync::Arc::clone(&previous.symbolic),
            None => std::sync::Arc::new(factor_symbolic(&system)?),
        };
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
        Ok(Self {
            symbolic,
            numeric,
            offsets: matrix.offsets.clone(),
            indices: matrix.indices.clone(),
        })
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
#[derive(Debug)]
pub(super) struct UpperDiagonalSchur<S> {
    groups: [Vec<usize>; 2],
    a: Factors<S>,
    schur: Factors<S>,
    c: Sparse<S>,
    dimension: usize,
}
impl<S: SparseLuScalar> UpperDiagonalSchur<S> {
    #[cfg(test)]
    pub(super) fn new(
        system: &CanonicalCsrSystemView<S>,
        groups: [Vec<usize>; 2],
        orientation: LinearOperatorOrientation,
        maximum_entries: usize,
        maximum_products: usize,
    ) -> Result<Self, Diagnostic> {
        Self::new_reusing(
            system,
            groups,
            orientation,
            maximum_entries,
            maximum_products,
            None,
        )
    }
    fn new_reusing(
        system: &CanonicalCsrSystemView<S>,
        groups: [Vec<usize>; 2],
        orientation: LinearOperatorOrientation,
        maximum_entries: usize,
        maximum_products: usize,
        previous: Option<&Self>,
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
                if g == 1 && h == 1 {
                    // Charge the retained E seed before its sparse entry allocation.
                    entries = entries
                        .checked_add(1)
                        .filter(|&value| value <= maximum_entries)
                        .ok_or_else(|| {
                            invalid("upper diagonal-Schur seed entry budget exceeded")
                        })?;
                }
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
        // Move the already charged E seed; no unbudgeted second sparse copy.
        let mut schur = std::mem::take(&mut blocks[3]);
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
        let a = Factors::new(
            &sparse(&blocks[0], sizes[0]),
            previous.map(|previous| &previous.a),
        )?;
        let schur = Factors::new(
            &sparse(&schur, sizes[1]),
            previous.map(|previous| &previous.schur),
        )?;
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
            *value -= c;
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

pub(super) fn require_factor_choices(
    config: &eqiora_solver::UpperDiagonalSchurConfig,
) -> Result<(), Diagnostic> {
    for choice in [config.a_factorization(), config.schur_factorization()] {
        if choice.algorithm() != eqiora_solver::LinearSolver::SparseLu
            || choice.provider() != crate::FAER_SOLVER_PROVIDER.id().as_str()
        {
            return Err(invalid(
                "fixed block factor provider does not match the resolved Faer provider",
            ));
        }
    }
    Ok(())
}

/// Full-policy bound fixed action, admitted against retained original coordinates.
#[derive(Debug)]
pub(super) struct BoundUpperDiagonalSchur<S> {
    policy: eqiora_solver::PreconditionerPolicy,
    action: UpperDiagonalSchur<S>,
}
impl<S: SparseLuScalar> BoundUpperDiagonalSchur<S> {
    pub(super) fn new(
        problem: &eqiora_solver::LinearProblem<'_, S>,
        plan: &eqiora_solver::SolverPlan,
    ) -> Result<Self, Diagnostic> {
        Self::new_reusing(problem, plan, None)
    }
    #[cfg(test)]
    pub(super) fn shares_symbolic_with(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.action.a.symbolic, &other.action.a.symbolic)
            && std::sync::Arc::ptr_eq(&self.action.schur.symbolic, &other.action.schur.symbolic)
    }
    pub(super) fn require_problem(
        &self,
        problem: &eqiora_solver::LinearProblem<'_, S>,
        plan: &eqiora_solver::SolverPlan,
    ) -> Result<(), Diagnostic> {
        if self.policy != *plan.preconditioner()
            || problem.operator().orientation() != LinearOperatorOrientation::Normal
            || problem.operator().rows() != self.action.dimension
            || problem.operator().columns() != self.action.dimension
        {
            return Err(invalid(
                "prepared Schur action differs from original problem orientation, dimensions or policy",
            ));
        }
        let eqiora_solver::PreconditionerPolicy::UpperDiagonalSchur(config) = plan.preconditioner()
        else {
            return Err(invalid("prepared Schur policy is missing"));
        };
        crate::FaerLinearSolver.capabilities().require_problem(
            plan,
            S::DOMAIN,
            S::STORAGE,
            problem.properties(),
        )?;
        let system = problem
            .canonical_csr_system()
            .ok_or_else(|| invalid("prepared Schur action requires canonical CSR"))?;
        if system.properties() != problem.properties() {
            return Err(invalid("prepared Schur canonical properties differ"));
        }
        let structure = problem
            .algebraic_structure()
            .ok_or_else(|| invalid("prepared Schur action requires exact ownership"))?;
        if structure.partition_coordinates(config.groups())?.as_slice()
            != self.action.groups.as_slice()
        {
            return Err(invalid("prepared Schur action coordinate groups differ"));
        }
        Ok(())
    }
    pub(super) fn new_reusing(
        problem: &eqiora_solver::LinearProblem<'_, S>,
        plan: &eqiora_solver::SolverPlan,
        previous: Option<&Self>,
    ) -> Result<Self, Diagnostic> {
        let eqiora_solver::PreconditionerPolicy::UpperDiagonalSchur(config) = plan.preconditioner()
        else {
            return Err(invalid(
                "upper diagonal-Schur action requires its complete policy",
            ));
        };
        if S::DOMAIN != eqiora_core::ScalarDomain::Real
            || S::STORAGE != eqiora_core::ScalarType::F64
            || plan.algorithm() != eqiora_solver::LinearSolver::BiConjugateGradientStabilized
            || plan.reduction() != eqiora_solver::ReductionPolicy::Fast
        {
            return Err(invalid(
                "upper diagonal-Schur requires real binary64 BiCGSTAB/Fast",
            ));
        }
        require_factor_choices(config)?;
        crate::FaerLinearSolver.capabilities().require_problem(
            plan,
            S::DOMAIN,
            S::STORAGE,
            problem.properties(),
        )?;
        let system = problem
            .canonical_csr_system()
            .ok_or_else(|| invalid("upper diagonal-Schur requires canonical CSR coefficients"))?;
        if system.rows() != problem.operator().rows()
            || system.columns() != problem.operator().columns()
            || system.properties() != problem.properties()
        {
            return Err(invalid(
                "upper diagonal-Schur canonical source and original operator differ",
            ));
        }
        let structure = problem
            .algebraic_structure()
            .ok_or_else(|| invalid("upper diagonal-Schur requires retained realized ownership"))?;
        let groups = structure.partition_coordinates(config.groups())?;
        let groups: [Vec<usize>; 2] = groups
            .try_into()
            .map_err(|_| invalid("upper diagonal-Schur requires exactly two coordinate groups"))?;
        let previous = previous.filter(|previous| &previous.policy == plan.preconditioner());
        let action = UpperDiagonalSchur::new_reusing(
            system,
            groups,
            problem.operator().orientation(),
            config.maximum_schur_entries().get(),
            config.maximum_schur_products().get(),
            previous.map(|previous| &previous.action),
        )?;
        Ok(Self {
            policy: plan.preconditioner().clone(),
            action,
        })
    }
}
impl<S: SparseLuScalar> eqiora_solver::FixedLinearPreconditioner<S> for BoundUpperDiagonalSchur<S> {
    fn dimension(&self) -> usize {
        self.action.dimension
    }
    fn policy(&self) -> &eqiora_solver::PreconditionerPolicy {
        &self.policy
    }
    fn apply(&self, input: &[S], output: &mut [S]) -> Result<(), Diagnostic> {
        self.action.apply(input, output)
    }
}

pub(super) fn solve(
    problem: &eqiora_solver::LinearProblem<'_>,
    plan: &eqiora_solver::SolverPlan,
    execution: &dyn eqiora_solver::ReplicatedLinearExecution,
) -> Result<eqiora_solver::LinearSolution, Diagnostic> {
    let action = BoundUpperDiagonalSchur::new(problem, plan)?;
    eqiora_solver::solve_bicgstab_with_preconditioner(
        crate::FAER_SOLVER_PROVIDER,
        problem,
        plan,
        execution,
        &action,
    )
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
            for (i, values) in local.iter().enumerate() {
                for (j, &value) in values.iter().enumerate() {
                    if value != 0. {
                        rows[i].insert(j, value);
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
    fn schur_seed_and_new_fill_are_charged_before_entry_insertion() {
        let seed_error = UpperDiagonalSchur::new(
            &fixture(),
            [vec![2, 0], vec![3, 1]],
            LinearOperatorOrientation::Normal,
            3,
            3,
        )
        .unwrap_err();
        assert!(seed_error.message().contains("seed entry budget exceeded"));
        // E has two seeds; D diag(A)^-1 C introduces exactly one new slot.
        let local = [
            [1., 0., 1., 1.],
            [0., 1., 0., 1.],
            [1., 0., 2., 0.],
            [0., 1., 0., 2.],
        ];
        let rows = local
            .iter()
            .map(|row| {
                row.iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, value)| *value != 0.)
                    .collect()
            })
            .collect::<Vec<BTreeMap<usize, f64>>>();
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
                3,
                3
            )
            .is_ok()
        );
        let fill_error = UpperDiagonalSchur::new(
            &system,
            [vec![0, 1], vec![2, 3]],
            LinearOperatorOrientation::Normal,
            2,
            3,
        )
        .unwrap_err();
        assert!(fill_error.message().contains("fill budget exceeded"));
    }
    #[test]
    fn rejects_wrong_orientation_partition_and_resource_limits() {
        let system = fixture();
        assert!(
            UpperDiagonalSchur::new(
                &system,
                [vec![2, 0], vec![3, 1]],
                LinearOperatorOrientation::Normal,
                4,
                3
            )
            .is_ok()
        );
        for (entries, products) in [(3, 3), (4, 2)] {
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
