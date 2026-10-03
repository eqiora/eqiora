use super::*;
use crate::{
    FixedOrderInnerProduct, LinearOperator, LinearSolveRequest, LinearSolver, ReductionPolicy,
};
use eqiora_core::Diagnostic;

struct ComplexStorage;
impl CompleteCsrStorage<Complex64> for ComplexStorage {
    fn rows(&self) -> usize {
        2
    }
    fn columns(&self) -> usize {
        2
    }
    fn row_offsets(&self) -> &[usize] {
        &[0, 2, 4]
    }
    fn column_indices(&self) -> &[usize] {
        &[0, 1, 0, 1]
    }
    fn values(&self) -> &[Complex64] {
        const VALUES: [Complex64; 4] = [
            Complex64::new(1., 1.),
            Complex64::new(2., 0.),
            Complex64::new(0., 3.),
            Complex64::new(4., -1.),
        ];
        &VALUES
    }
    fn right_hand_side(&self) -> &[Complex64] {
        const RHS: [Complex64; 2] = [Complex64::new(-5., 5.), Complex64::new(-13., 9.)];
        &RHS
    }
}

#[test]
fn real_block_actions_equal_the_independent_complex_products_in_each_orientation() {
    let source =
        CanonicalCsrSystemView::new(&ComplexStorage, LinearOperatorProperties::General).unwrap();
    for (orientation, expected) in [
        (LinearOperatorOrientation::Normal, [-5., 5., -13., 9.]),
        (LinearOperatorOrientation::Transposed, [-4., -3., -5., 10.]),
        (
            LinearOperatorOrientation::ConjugateTransposed,
            [6., 7., -7., 6.],
        ),
    ] {
        let oriented = crate::Oriented::new(&source, orientation).unwrap();
        let problem =
            LinearProblem::from_oriented_canonical(&oriented, &source, source.right_hand_side())
                .unwrap();
        let block = CanonicalCsrSystemView::new(
            &RealBlockStorage::new(&source, &problem).unwrap(),
            LinearOperatorProperties::General,
        )
        .unwrap();
        let mut actual = [0.; 4];
        block.apply(&[1., 2., -2., 1.], &mut actual).unwrap();
        assert_eq!(actual, expected);
    }
}

#[derive(Debug)]
struct UnadmittedExecution;

impl ReplicatedLinearExecution for UnadmittedExecution {
    fn provider(&self) -> crate::ExecutionProvider {
        crate::ExecutionProvider::new(crate::ExecutionId::new("test.unadmitted"), "1", &[])
    }
    fn report(&self) -> crate::ExecutionReport {
        SERIAL_LINEAR_EXECUTION.report()
    }
    fn require_reduction(&self, _: ReductionPolicy) -> Result<(), Diagnostic> {
        panic!("must reject before numerical work")
    }
    fn apply(
        &self,
        _: &dyn LinearOperator<Scalar = f64>,
        _: &[f64],
        _: &mut [f64],
    ) -> Result<(), Diagnostic> {
        panic!("must reject before numerical work")
    }
    fn inner_product(&self, _: FixedOrderInnerProduct<'_>) -> Result<f64, Diagnostic> {
        panic!("must reject before numerical work")
    }
}

#[test]
fn complex_reference_rejects_unadmitted_execution_before_any_numerical_operation() {
    let source =
        CanonicalCsrSystemView::new(&ComplexStorage, LinearOperatorProperties::General).unwrap();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-12,
        32.try_into().unwrap(),
    )
    .unwrap();
    let error = ReferenceLinearSolver
        .solve_with_execution(
            &source.linear_problem().unwrap(),
            plan,
            &UnadmittedExecution,
        )
        .unwrap_err();
    assert!(error.message().contains("host-serial"));

    // A matrix-free clone of the same action has no captured coefficient authority.
    let uncaptured =
        LinearProblem::new(&source, source.right_hand_side(), source.properties()).unwrap();
    let error = LinearSolveRequest::new(&ReferenceLinearSolver, plan)
        .solve(&uncaptured)
        .unwrap_err();
    assert!(error.message().contains("captured canonical"));
}
