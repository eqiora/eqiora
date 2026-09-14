use std::num::NonZeroUsize;

use eqiora::solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, LinearOperatorProperties, LinearSolver,
    LinearSolverBackend, PreconditionerPolicy, PreparedLinearStructureIdentity, ReductionPolicy,
    SolverPlan,
};
use eqiora_backend_faer::FaerLinearSolver;

const ABSOLUTE_TOLERANCE: f64 = f64::from_bits(0x3e10_0000_0000_0000);
const SOLUTION_ERROR_CEILING: f64 = f64::from_bits(0x3df0_0000_0000_0000);

struct Storage {
    value: f64,
    right_hand_side: f64,
}

impl CompleteCsrStorage for Storage {
    fn rows(&self) -> usize {
        1
    }
    fn columns(&self) -> usize {
        1
    }
    fn row_offsets(&self) -> &[usize] {
        &[0, 1]
    }
    fn column_indices(&self) -> &[usize] {
        &[0]
    }
    fn values(&self) -> &[f64] {
        std::slice::from_ref(&self.value)
    }
    fn right_hand_side(&self) -> &[f64] {
        std::slice::from_ref(&self.right_hand_side)
    }
}

#[test]
fn prepared_provider_preserves_independent_solutions_and_reports() {
    run_independent_oracle_checker();
    let plan = sparse_lu_plan();
    let structure =
        PreparedLinearStructureIdentity::new(&b"two-element-q1-ordering-and-constraints"[..])
            .unwrap();
    let storage = [
        Storage {
            value: 4.0,
            right_hand_side: 1.0,
        },
        Storage {
            value: 4.0,
            right_hand_side: 2.0,
        },
        Storage {
            value: 5.0,
            right_hand_side: 1.0,
        },
    ];
    let systems = storage
        .iter()
        .map(|storage| {
            CanonicalCsrSystemView::new(
                storage,
                LinearOperatorProperties::SymmetricPositiveDefinite,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let cold = systems
        .iter()
        .map(|system| {
            let mut prepared = FaerLinearSolver.prepare_linear(plan).unwrap().unwrap();
            prepared
                .solve(&structure, &system.linear_problem().unwrap())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut prepared = FaerLinearSolver.prepare_linear(plan).unwrap().unwrap();
    let warm = systems
        .iter()
        .map(|system| {
            prepared
                .solve(&structure, &system.linear_problem().unwrap())
                .unwrap()
        })
        .collect::<Vec<_>>();
    for (index, ((warm, cold), expected)) in
        warm.iter().zip(&cold).zip([0.25, 0.5, 0.2]).enumerate()
    {
        assert_eq!(warm, cold, "warm/cold solution {index}");
        assert!((warm.values()[0] - expected).abs() <= SOLUTION_ERROR_CEILING);
        assert!(warm.report().true_residual_norm() <= ABSOLUTE_TOLERANCE);
    }
}

fn sparse_lu_plan() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::SparseLu,
        0.0,
        ABSOLUTE_TOLERANCE,
        NonZeroUsize::MIN,
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast)
}

fn run_independent_oracle_checker() {
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .unwrap();
    let output = std::process::Command::new("python3")
        .current_dir(repository)
        .arg("verify/numerics/faer-sparse-lu-reuse/run_case.py")
        .arg("--check")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "sparse-LU reuse oracle failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
