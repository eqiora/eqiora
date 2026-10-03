use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_solver::{
    ExecutionProvider, ExecutionReport, FixedOrderInnerProduct, LinearOperator,
    LinearOperatorProperties, LinearProblem, LinearSolver, LinearSolverBackend,
    PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy, ReplicatedLinearExecution,
    SERIAL_LINEAR_EXECUTION, SolverPlan,
};

const HADAMARD_CONDITIONING_INTEGERS: [u64; 64] = [
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    12,
    17,
    24,
    34,
    48,
    68,
    97,
    138,
    197,
    280,
    398,
    565,
    804,
    1_143,
    1_625,
    2_311,
    3_287,
    4_674,
    6_647,
    9_452,
    13_440,
    19_112,
    27_178,
    38_648,
    54_958,
    78_151,
    111_131,
    158_030,
    224_721,
    319_557,
    454_415,
    646_185,
    918_884,
    1_306_667,
    1_858_100,
    2_642_246,
    3_757_313,
    5_342_955,
    7_597_761,
    10_804_128,
    15_363_631,
    21_847_311,
    31_067_200,
    44_178_019,
    62_821_798,
    89_333_529,
    127_033_602,
    180_643_666,
    256_877_971,
    365_284_284,
    519_439_668,
    738_650_910,
    1_050_372_547,
    1_493_645_335,
    2_123_985_813,
    3_020_339_320,
    4_294_967_296,
];

#[derive(Debug)]
struct HadamardConditionedSymmetricIndefinite;

impl LinearOperator for HadamardConditionedSymmetricIndefinite {
    type Scalar = f64;

    fn rows(&self) -> usize {
        HADAMARD_CONDITIONING_INTEGERS.len()
    }

    fn columns(&self) -> usize {
        HADAMARD_CONDITIONING_INTEGERS.len()
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        if input.len() != self.columns() || output.len() != self.rows() {
            return Err(Diagnostic::error(
                codes::INVALID_REALIZATION,
                "Hadamard witness shape mismatch",
            ));
        }
        let mut spectral = input.to_vec();
        sylvester_hadamard_transform(&mut spectral);
        for (index, value) in spectral.iter_mut().enumerate() {
            let sign = if index % 2 == 0 { -1.0 } else { 1.0 };
            *value *= sign * (HADAMARD_CONDITIONING_INTEGERS[index] as f64) / (8.0 * 65_536.0);
        }
        sylvester_hadamard_transform(&mut spectral);
        for (output, value) in output.iter_mut().zip(spectral) {
            *output = value / 8.0;
        }
        Ok(())
    }
}

fn sylvester_hadamard_transform(values: &mut [f64]) {
    let mut width = 1;
    while width < values.len() {
        for start in (0..values.len()).step_by(2 * width) {
            for offset in 0..width {
                let left = values[start + offset];
                let right = values[start + width + offset];
                values[start + offset] = left + right;
                values[start + width + offset] = left - right;
            }
        }
        width *= 2;
    }
}

fn right_hand_side() -> Vec<f64> {
    let mut right_hand_side = HADAMARD_CONDITIONING_INTEGERS
        .iter()
        .enumerate()
        .map(|(index, magnitude)| {
            let sign = if index % 2 == 0 { -1.0 } else { 1.0 };
            sign * (*magnitude as f64) / 65_536.0
        })
        .collect::<Vec<_>>();
    sylvester_hadamard_transform(&mut right_hand_side);
    for value in &mut right_hand_side {
        *value /= 8.0;
    }
    right_hand_side
}

#[derive(Debug, Default)]
struct RecordingExecution {
    inner_products: AtomicUsize,
}

impl RecordingExecution {
    fn inner_products(&self) -> usize {
        self.inner_products.load(Ordering::Relaxed)
    }
}

impl ReplicatedLinearExecution for RecordingExecution {
    fn provider(&self) -> ExecutionProvider {
        SERIAL_LINEAR_EXECUTION.provider()
    }

    fn report(&self) -> ExecutionReport {
        SERIAL_LINEAR_EXECUTION.report()
    }

    fn require_reduction(&self, policy: ReductionPolicy) -> Result<(), Diagnostic> {
        SERIAL_LINEAR_EXECUTION.require_reduction(policy)
    }

    fn apply(
        &self,
        operator: &dyn LinearOperator<Scalar = f64>,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        SERIAL_LINEAR_EXECUTION.apply(operator, input, output)
    }

    fn inner_product(&self, action: FixedOrderInnerProduct<'_>) -> Result<f64, Diagnostic> {
        self.inner_products.fetch_add(1, Ordering::Relaxed);
        SERIAL_LINEAR_EXECUTION.inner_product(action)
    }
}

#[test]
fn reference_minres_solves_the_hadamard_conditioned_indefinite_witness() {
    let operator = HadamardConditionedSymmetricIndefinite;
    let right_hand_side = right_hand_side();
    let mut first_column = [0.0; 64];
    let mut first_basis = [0.0; 64];
    first_basis[0] = 1.0;
    operator.apply(&first_basis, &mut first_column).unwrap();

    assert_eq!(HADAMARD_CONDITIONING_INTEGERS[0], 1);
    assert_eq!(HADAMARD_CONDITIONING_INTEGERS[63], 1_u64 << 32);
    assert_eq!(first_column[0], 601.211_506_366_729_7);
    assert_eq!(first_column[1], -3_450.455_029_010_772_7);
    assert_eq!(right_hand_side[0], 4_809.692_050_933_838);
    assert_eq!(right_hand_side[1], -27_603.640_232_086_18);
    assert_eq!(right_hand_side[63], 4_991.299_930_572_51);

    let problem = LinearProblem::new(
        &operator,
        &right_hand_side,
        LinearOperatorProperties::SymmetricIndefinite,
    )
    .unwrap();
    let plan = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-12,
        0.0,
        NonZeroUsize::new(128).unwrap(),
    )
    .unwrap();
    let solution = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap();

    assert_eq!(solution.report().completed_iterations(), 64);
    assert!((solution.report().residual_target() - 9.217_895_952_054_019e-8).abs() < 2.0e-20);
    assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
    assert!((solution.values()[0] - 8.0).abs() <= 0.01);
    assert!(
        solution.values()[1..]
            .iter()
            .all(|value| value.abs() <= 0.01)
    );
    assert_eq!(solution.report().algorithm(), LinearSolver::MinimumResidual);
    assert_eq!(
        solution.report().preconditioner(),
        PreconditionerPolicy::Identity
    );
    assert_eq!(solution.report().reduction(), ReductionPolicy::Reproducible);
    assert_eq!(solution.report().backend().as_str(), "eqiora.reference");
}

#[test]
fn reference_minres_fails_closed_below_the_attainable_floor() {
    let operator = HadamardConditionedSymmetricIndefinite;
    let right_hand_side = right_hand_side();
    let problem = LinearProblem::new(
        &operator,
        &right_hand_side,
        LinearOperatorProperties::SymmetricIndefinite,
    )
    .unwrap();
    let plan = SolverPlan::new(
        LinearSolver::MinimumResidual,
        0.0,
        1.0e-20,
        NonZeroUsize::new(128).unwrap(),
    )
    .unwrap();
    let error = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap_err();

    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert!(error.message().contains("Krylov space closed"));
    assert!(!error.message().contains("plan limit"));
}

#[test]
fn reference_minres_fails_at_the_plan_limit_before_the_krylov_grade() {
    let operator = HadamardConditionedSymmetricIndefinite;
    let right_hand_side = right_hand_side();
    let problem = LinearProblem::new(
        &operator,
        &right_hand_side,
        LinearOperatorProperties::SymmetricIndefinite,
    )
    .unwrap();
    let plan = SolverPlan::new(
        LinearSolver::MinimumResidual,
        0.0,
        1.0e-20,
        NonZeroUsize::new(32).unwrap(),
    )
    .unwrap();
    let error = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap_err();

    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert!(error.message().contains("plan limit of 32 iterations"));
    assert!(!error.message().contains("Krylov space closed"));
}

#[test]
fn reference_minres_deflates_an_invariant_krylov_space_without_early_exit() {
    #[derive(Debug)]
    struct GradeTwoIndefinite;

    impl LinearOperator for GradeTwoIndefinite {
        type Scalar = f64;

        fn rows(&self) -> usize {
            3
        }

        fn columns(&self) -> usize {
            3
        }

        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            output[0] = input[0];
            output[1] = -input[1];
            output[2] = 4.0 * input[2];
            Ok(())
        }
    }

    let problem = LinearProblem::new(
        &GradeTwoIndefinite,
        &[3.0, 5.0, 0.0],
        LinearOperatorProperties::SymmetricIndefinite,
    )
    .unwrap();
    let plan = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-12,
        0.0,
        NonZeroUsize::new(10).unwrap(),
    )
    .unwrap();
    let solution = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap();

    assert_eq!(solution.report().completed_iterations(), 2);
    for (actual, expected) in solution.values().iter().zip([3.0, -5.0, 0.0]) {
        assert!((actual - expected).abs() <= 1.0e-13);
    }
}

#[test]
fn reference_minres_routes_both_reorthogonalization_passes_through_the_execution() {
    let operator = HadamardConditionedSymmetricIndefinite;
    let right_hand_side = right_hand_side();
    let problem = LinearProblem::new(
        &operator,
        &right_hand_side,
        LinearOperatorProperties::SymmetricIndefinite,
    )
    .unwrap();
    let plan = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-12,
        0.0,
        NonZeroUsize::new(128).unwrap(),
    )
    .unwrap();
    let execution = RecordingExecution::default();
    let solution = REFERENCE_LINEAR_SOLVER
        .solve_with_execution(&problem, plan, &execution)
        .unwrap();
    let iterations = solution.report().completed_iterations();

    assert!(execution.inner_products() >= iterations * (iterations + 1));
}
