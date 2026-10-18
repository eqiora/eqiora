use super::*;
use crate::{
    CanonicalCsrSystemView, CsrStorage, FixedOrderInnerProduct, LinearOperator,
    LinearOperatorOrientation, LinearSolveRequest, LinearSolver, ReductionPolicy,
};
use eqiora_core::Diagnostic;

struct ComplexStorage;
impl CsrStorage<Complex64> for ComplexStorage {
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
}

impl ComplexStorage {
    pub(crate) fn rhs(&self) -> &[Complex64] {
        const RHS: [Complex64; 2] = [Complex64::new(-5., 5.), Complex64::new(-13., 9.)];
        &RHS
    }
}

#[test]
fn real_block_actions_equal_the_independent_complex_products_in_each_orientation() {
    let source = CanonicalCsrSystemView::new(
        &ComplexStorage,
        ComplexStorage.rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
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
        let block = source
            .real_coordinate_system(problem.operator().orientation(), problem.right_hand_side())
            .unwrap();
        let mut actual = [0.; 4];
        block.apply(&[1., 2., -2., 1.], &mut actual).unwrap();
        assert_eq!(actual, expected);
    }
}

#[derive(Debug)]
struct UnadmittedExecution;

impl ReplicatedLinearExecution<Complex64> for UnadmittedExecution {
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
        _: &dyn LinearOperator<Scalar = Complex64>,
        _: &[Complex64],
        _: &mut [Complex64],
    ) -> Result<(), Diagnostic> {
        panic!("must reject before numerical work")
    }
    fn inner_product(
        &self,
        _: FixedOrderInnerProduct<'_, Complex64>,
    ) -> Result<Complex64, Diagnostic> {
        panic!("must reject before numerical work")
    }
}

#[test]
fn complex_reference_rejects_unadmitted_execution_before_any_numerical_operation() {
    let source = CanonicalCsrSystemView::new(
        &ComplexStorage,
        ComplexStorage.rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
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

#[test]
fn real_coordinates_preserve_rhs_and_the_adjoint_pairing_without_reusing_source_identity() {
    let source = CanonicalCsrSystemView::new(
        &ComplexStorage,
        ComplexStorage.rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    let rhs = [Complex64::new(7., -3.), Complex64::new(11., 2.)];
    let real = source
        .real_coordinate_system(LinearOperatorOrientation::Normal, &rhs)
        .unwrap();
    assert_eq!(real.right_hand_side(), [7., -3., 11., 2.]);
    assert_eq!(source.right_hand_side(), ComplexStorage.rhs());
    assert_ne!(real.agreement_fingerprint(), source.agreement_fingerprint());
    let transpose = crate::Oriented::new(&real, LinearOperatorOrientation::Transposed).unwrap();
    let adjoint = source
        .real_coordinate_system(LinearOperatorOrientation::ConjugateTransposed, &rhs)
        .unwrap();
    let mut transposed = [0.; 4];
    let mut conjugated = [0.; 4];
    transpose
        .apply(&[1., 2., -2., 1.], &mut transposed)
        .unwrap();
    adjoint.apply(&[1., 2., -2., 1.], &mut conjugated).unwrap();
    assert_eq!(transposed, [6., 7., -7., 6.]);
    assert_eq!(transposed, conjugated);
    assert!(
        source
            .real_coordinate_system(LinearOperatorOrientation::Normal, &rhs[..1])
            .is_err()
    );
    assert!(
        source
            .real_coordinate_system(
                LinearOperatorOrientation::Normal,
                &[Complex64::new(1., f64::NAN); 2]
            )
            .is_err()
    );
}

#[derive(Debug, Default)]
struct ObservedSerial {
    actions: std::sync::atomic::AtomicUsize,
    reductions: std::sync::atomic::AtomicUsize,
}

impl ReplicatedLinearExecution<Complex64> for ObservedSerial {
    fn provider(&self) -> crate::ExecutionProvider {
        SERIAL_LINEAR_EXECUTION.provider()
    }
    fn report(&self) -> crate::ExecutionReport {
        SERIAL_LINEAR_EXECUTION.report()
    }
    fn require_reduction(&self, _: ReductionPolicy) -> Result<(), Diagnostic> {
        Ok(())
    }
    fn apply(
        &self,
        operator: &dyn LinearOperator<Scalar = Complex64>,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), Diagnostic> {
        self.actions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        SERIAL_LINEAR_EXECUTION.apply(operator, input, output)
    }
    fn inner_product(
        &self,
        action: FixedOrderInnerProduct<'_, Complex64>,
    ) -> Result<Complex64, Diagnostic> {
        self.reductions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        SERIAL_LINEAR_EXECUTION.inner_product(action)
    }
}

#[test]
fn native_complex_iteration_and_acceptance_use_the_admitted_typed_execution() {
    let source = CanonicalCsrSystemView::new(
        &ComplexStorage,
        ComplexStorage.rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-12,
        32.try_into().unwrap(),
    )
    .unwrap();
    let execution = ObservedSerial::default();
    let solved = ReferenceLinearSolver
        .solve_with_execution(&source.linear_problem().unwrap(), plan, &execution)
        .unwrap();
    for (actual, expected) in solved
        .values()
        .iter()
        .zip([Complex64::new(1.0, 2.0), Complex64::new(-2.0, 1.0)])
    {
        assert!((*actual - expected).norm() < 1e-12);
    }
    // Acceptance alone applies twice and reduces three norms. The producer
    // must also use this admitted execution, rather than bypassing its actions.
    assert!(execution.actions.load(std::sync::atomic::Ordering::Relaxed) > 2);
    assert!(
        execution
            .reductions
            .load(std::sync::atomic::Ordering::Relaxed)
            > 3
    );
}
