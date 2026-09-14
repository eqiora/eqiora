use std::num::NonZeroUsize;
use std::path::Path;
use std::process::Command;

use eqiora::solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, ExecutionProvider, HostSerialSolverProfile,
    LinearOperatorProperties, LinearSolveRequest, LinearSolver, LinearSolverBackend,
    PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy, ResolvedHostSerialSolverPlan,
    SERIAL_EXECUTION_PROVIDER, SolverPlan, SolverPlanningObjective, SolverProvider,
};
use eqiora_backend_faer::FaerLinearSolver;
use serde_json::Value;

const ORACLE: &str =
    include_str!("../../../verify/numerics/host-serial-solver-planning/expected/policy-v2.json");
const REFERENCE_ID: &str = "eqiora.reference.bicgstab-general-jacobi-reproducible-f64";
const FAER_BICGSTAB_ID: &str = "eqiora.faer.bicgstab-general-jacobi-fast-f64";
const FAER_SPARSE_LU_ID: &str = "eqiora.faer.sparse-lu-general-identity-fast-f64";

#[derive(Debug)]
struct Fixture;

impl CompleteCsrStorage for Fixture {
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

    fn values(&self) -> &[f64] {
        &[4.0, 1.0, 2.0, 3.0]
    }

    fn right_hand_side(&self) -> &[f64] {
        &[6.0, 8.0]
    }
}

fn plan(
    algorithm: LinearSolver,
    preconditioner: PreconditionerPolicy,
    reduction: ReductionPolicy,
) -> SolverPlan {
    SolverPlan::new(algorithm, 1.0e-12, 1.0e-14, NonZeroUsize::new(100).unwrap())
        .unwrap()
        .with_preconditioner(preconditioner)
        .with_reduction(reduction)
}

fn reference_plan() -> SolverPlan {
    plan(
        LinearSolver::BiConjugateGradientStabilized,
        PreconditionerPolicy::Jacobi,
        ReductionPolicy::Reproducible,
    )
}

fn faer_bicgstab_plan() -> SolverPlan {
    plan(
        LinearSolver::BiConjugateGradientStabilized,
        PreconditionerPolicy::Jacobi,
        ReductionPolicy::Fast,
    )
}

fn faer_sparse_lu_plan() -> SolverPlan {
    plan(
        LinearSolver::SparseLu,
        PreconditionerPolicy::Identity,
        ReductionPolicy::Fast,
    )
}

fn objective(name: &str) -> SolverPlanningObjective {
    match name {
        "Robust" => SolverPlanningObjective::Robust,
        "Fast" => SolverPlanningObjective::Fast,
        "LowMemory" => SolverPlanningObjective::LowMemory,
        other => panic!("unexpected frozen objective {other}"),
    }
}

fn assert_solver_provider(actual: SolverProvider, expected: &Value) {
    assert_eq!(actual.id().as_str(), expected["id"].as_str().unwrap());
    assert_eq!(
        expected["implementation_version"].as_str(),
        Some("0.1.0-alpha.3")
    );
    assert_eq!(actual.implementation_version(), env!("CARGO_PKG_VERSION"));
    let expected_libraries = expected["libraries"].as_array().unwrap();
    assert_eq!(actual.libraries().len(), expected_libraries.len());
    for (actual, expected) in actual.libraries().iter().zip(expected_libraries) {
        assert_eq!(actual.name(), expected["name"].as_str().unwrap());
        assert_eq!(actual.version(), expected["version"].as_str().unwrap());
    }
}

fn assert_execution_provider(actual: ExecutionProvider, expected: &Value) {
    assert_eq!(actual.id().as_str(), expected["id"].as_str().unwrap());
    assert_eq!(
        expected["implementation_version"].as_str(),
        Some("0.1.0-alpha.3")
    );
    assert_eq!(actual.implementation_version(), env!("CARGO_PKG_VERSION"));
    assert!(actual.libraries().is_empty());
    assert!(expected["libraries"].as_array().unwrap().is_empty());
}

fn assert_plan(candidate_id: &str, actual: SolverPlan, expected: &Value) {
    let tuple = &expected["tuple"];
    match candidate_id {
        REFERENCE_ID => assert_eq!(actual, reference_plan()),
        FAER_BICGSTAB_ID => assert_eq!(actual, faer_bicgstab_plan()),
        FAER_SPARSE_LU_ID => assert_eq!(actual, faer_sparse_lu_plan()),
        other => panic!("unexpected frozen candidate {other}"),
    }
    assert_eq!(
        format!("{:?}", actual.algorithm()),
        tuple["algorithm"].as_str().unwrap()
    );
    assert_eq!(
        format!("{:?}", actual.preconditioner()),
        tuple["preconditioner"].as_str().unwrap()
    );
    assert_eq!(
        format!("{:?}", actual.reduction()),
        tuple["reduction"].as_str().unwrap()
    );
    assert_eq!(tuple["operator_properties"].as_str().unwrap(), "General");
    assert_eq!(tuple["scalar_type"].as_str().unwrap(), "F64");
}

fn run_derivation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap();
    let output = Command::new("python3")
        .current_dir(root)
        .arg("verify/numerics/host-serial-solver-planning/references/derive_policy_v2.py")
        .output()
        .expect("registered planning evidence requires Python 3");
    assert!(
        output.status.success(),
        "independent policy derivation failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn full_catalog_decisions_match_exact_manual_execution_and_rational_oracle() {
    run_derivation();
    let oracle: Value = serde_json::from_str(ORACLE).unwrap();
    assert_eq!(
        oracle["policy_id"].as_str().unwrap(),
        "eqiora.host-serial-solver-planning/v2"
    );
    assert_eq!(1.0e-12_f64.to_bits(), 0x3d71_9799_812d_ea11);
    assert_eq!(1.0e-14_f64.to_bits(), 0x3d06_849b_86a1_2b9b);
    assert_eq!(
        oracle["common_controls"]["relative_tolerance_f64_bits"]
            .as_str()
            .unwrap(),
        "0x3d719799812dea11"
    );
    assert_eq!(
        oracle["common_controls"]["absolute_tolerance_f64_bits"]
            .as_str()
            .unwrap(),
        "0x3d06849b86a12b9b"
    );

    let system = CanonicalCsrSystemView::new(&Fixture, LinearOperatorProperties::General).unwrap();
    assert_eq!(system.row_offsets(), &[0, 2, 4]);
    assert_eq!(system.column_indices(), &[0, 1, 0, 1]);
    assert_eq!(system.values(), &[4.0, 1.0, 2.0, 3.0]);
    assert_eq!(system.right_hand_side(), &[6.0, 8.0]);
    assert_eq!(system.column_indices()[0], 0);
    assert_eq!(system.values()[0], 4.0);
    assert_eq!(system.column_indices()[3], 1);
    assert_eq!(system.values()[3], 3.0);
    let problem = system.linear_problem().unwrap();
    assert!(problem.initial_guess().is_none());

    let faer = FaerLinearSolver;
    let candidates = [
        (
            REFERENCE_ID,
            "fluid.cartesian-advection-diffusion-fvm-2d",
            REFERENCE_LINEAR_SOLVER.provider(),
            reference_plan(),
        ),
        (
            FAER_SPARSE_LU_ID,
            "numerics.linear-backends",
            faer.provider(),
            faer_sparse_lu_plan(),
        ),
        (
            FAER_BICGSTAB_ID,
            "numerics.linear-backends",
            faer.provider(),
            faer_bicgstab_plan(),
        ),
    ];

    let frozen_candidates = oracle["candidates"].as_array().unwrap();
    assert_eq!(frozen_candidates.len(), 3);
    for (candidate_id, evidence_case, provider, plan) in candidates {
        let expected = frozen_candidates
            .iter()
            .find(|expected| expected["id"].as_str() == Some(candidate_id))
            .unwrap();
        assert_eq!(evidence_case, expected["evidence_case"].as_str().unwrap());
        assert_solver_provider(provider, &expected["solver_provider"]);
        assert_plan(candidate_id, plan, expected);
    }

    let bound = 2.0_f64.powi(-40);
    assert_eq!(bound.to_bits(), 0x3d70_0000_0000_0000);
    for expected in oracle["objectives"].as_array().unwrap() {
        let objective = objective(expected["objective"].as_str().unwrap());
        let decision = ResolvedHostSerialSolverPlan::resolve(
            HostSerialSolverProfile::general_canonical_csr(),
            objective,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(100).unwrap(),
            &REFERENCE_LINEAR_SOLVER,
            &faer,
        )
        .unwrap();
        assert_eq!(decision.policy_id(), oracle["policy_id"].as_str().unwrap());
        assert_eq!(decision.objective(), objective);
        assert_eq!(
            decision.selected_candidate_id(),
            expected["selected_candidate_id"].as_str().unwrap()
        );
        let expected_reasons = expected["ordered_reasons"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| (pair[0].as_str().unwrap(), pair[1].as_str().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(decision.reasons().collect::<Vec<_>>(), expected_reasons);
        let selected_expected = frozen_candidates
            .iter()
            .find(|candidate| candidate["id"].as_str() == Some(decision.selected_candidate_id()))
            .unwrap();
        assert_eq!(
            decision.selected_evidence_case(),
            selected_expected["evidence_case"].as_str().unwrap()
        );
        assert_solver_provider(
            decision.solver_provider(),
            &selected_expected["solver_provider"],
        );
        assert_execution_provider(
            decision.execution_provider(),
            &oracle["common_controls"]["execution_provider"],
        );
        assert_eq!(decision.execution_provider(), SERIAL_EXECUTION_PROVIDER);
        assert_plan(
            decision.selected_candidate_id(),
            decision.solver_plan(),
            selected_expected,
        );

        let manual = match decision.selected_candidate_id() {
            REFERENCE_ID => LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, reference_plan()),
            FAER_BICGSTAB_ID => LinearSolveRequest::new(&faer, faer_bicgstab_plan()),
            FAER_SPARSE_LU_ID => LinearSolveRequest::new(&faer, faer_sparse_lu_plan()),
            _ => unreachable!("closed catalog candidate"),
        }
        .solve(&problem)
        .unwrap();
        let planned = decision.solve(&problem, None).unwrap();
        assert_eq!(planned.values(), manual.values());
        assert_eq!(planned.report(), manual.report());
        assert_eq!(planned, manual);

        assert_eq!(planned.values().len(), 2);
        assert!((planned.values()[0] - 1.0).abs() <= bound);
        assert!((planned.values()[1] - 2.0).abs() <= bound);
        let applied = [
            4.0 * planned.values()[0] + planned.values()[1],
            2.0 * planned.values()[0] + 3.0 * planned.values()[1],
        ];
        let residual: [f64; 2] = [6.0 - applied[0], 8.0 - applied[1]];
        let independent_true_residual = residual[0].hypot(residual[1]);
        assert_eq!(
            planned.report().true_residual_norm().to_bits(),
            independent_true_residual.to_bits()
        );
        assert!(planned.report().true_residual_norm() <= planned.report().residual_target());
        assert_eq!(
            planned.report().solver_provider(),
            decision.solver_provider()
        );
        assert_eq!(
            planned.report().execution_provider(),
            decision.execution_provider()
        );
        assert_eq!(planned.report().solver_plan(), decision.solver_plan());
    }
}

#[test]
fn spd_and_saddle_point_profiles_select_exact_current_backends() {
    #[derive(Debug)]
    struct Matrix {
        rows: usize,
        offsets: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<f64>,
        rhs: Vec<f64>,
    }
    impl CompleteCsrStorage for Matrix {
        fn rows(&self) -> usize {
            self.rows
        }
        fn columns(&self) -> usize {
            self.rows
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
        fn right_hand_side(&self) -> &[f64] {
            &self.rhs
        }
    }
    // SPD: leading principal minors are 4 and 11, and A[1,2] = [6,7].
    let spd = Matrix {
        rows: 2,
        offsets: vec![0, 2, 4],
        columns: vec![0, 1, 0, 1],
        values: vec![4., 1., 1., 3.],
        rhs: vec![6., 7.],
    };
    // Gauge-resolved saddle point: K=diag(2,3), B=[1,1], Schur=-5/6.
    // For a zero-integral gauge, B[1,-1]=0 and A[1,-1,3]=[5,0,0].
    // The multiplier diagonal is structurally absent.
    let saddle = Matrix {
        rows: 3,
        offsets: vec![0, 2, 4, 6],
        columns: vec![0, 2, 1, 2, 0, 1],
        values: vec![2., 1., 3., 1., 1., 1.],
        rhs: vec![5., 0., 0.],
    };
    let faer = FaerLinearSolver;
    for (matrix, properties, diagonal, expected, iterative) in [
        (
            &spd,
            LinearOperatorProperties::SymmetricPositiveDefinite,
            true,
            vec![1., 2.],
            LinearSolver::ConjugateGradient,
        ),
        (
            &saddle,
            LinearOperatorProperties::SymmetricIndefinite,
            false,
            vec![1., -1., 3.],
            LinearSolver::MinimumResidual,
        ),
    ] {
        let field = eqiora::Id::<eqiora::entity::kinds::Field>::new();
        let structure = eqiora_solver::AlgebraicStructure::new(
            [field],
            (!diagonal).then_some(eqiora_solver::AlgebraicConstraint::ZeroIntegral { field }),
        )
        .unwrap();
        let system = CanonicalCsrSystemView::new(matrix, properties).unwrap();
        let problem = system.linear_problem().unwrap();
        for objective in [
            SolverPlanningObjective::Robust,
            SolverPlanningObjective::Fast,
            SolverPlanningObjective::LowMemory,
        ] {
            let decision = ResolvedHostSerialSolverPlan::resolve(
                HostSerialSolverProfile::canonical_csr(properties, Some(diagonal), None)
                    .with_structure(structure.clone())
                    .unwrap(),
                objective,
                1e-12,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
                &REFERENCE_LINEAR_SOLVER,
                &faer,
            )
            .unwrap();
            let backend: &dyn LinearSolverBackend = if objective == SolverPlanningObjective::Fast {
                assert_eq!(decision.solver_plan().algorithm(), LinearSolver::SparseLu);
                &faer
            } else if diagonal && objective == SolverPlanningObjective::LowMemory {
                // Both CG candidates have the same fixed-vector rank; exact ID
                // order puts eqiora.faer before eqiora.reference.
                assert_eq!(
                    decision.solver_plan().algorithm(),
                    LinearSolver::ConjugateGradient
                );
                &faer
            } else {
                assert_eq!(decision.solver_plan().algorithm(), iterative);
                &REFERENCE_LINEAR_SOLVER
            };
            let uses_faer = objective == SolverPlanningObjective::Fast
                || (diagonal && objective == SolverPlanningObjective::LowMemory);
            let expected_plan = plan(
                if objective == SolverPlanningObjective::Fast {
                    LinearSolver::SparseLu
                } else {
                    iterative
                },
                if diagonal && objective == SolverPlanningObjective::LowMemory {
                    PreconditionerPolicy::Jacobi
                } else {
                    PreconditionerPolicy::Identity
                },
                if uses_faer {
                    ReductionPolicy::Fast
                } else {
                    ReductionPolicy::Reproducible
                },
            );
            assert_eq!(decision.solver_plan(), expected_plan);
            assert_eq!(decision.solver_provider(), backend.provider());
            assert_eq!(decision.execution_provider(), SERIAL_EXECUTION_PROVIDER);
            assert!(decision.solve(&problem, None).is_err());
            let foreign = eqiora_solver::AlgebraicStructure::new([eqiora::Id::new()], []).unwrap();
            assert!(decision.solve(&problem, Some(&foreign)).is_err());
            if !diagonal {
                let omitted_gauge = eqiora_solver::AlgebraicStructure::new([field], []).unwrap();
                assert!(decision.solve(&problem, Some(&omitted_gauge)).is_err());
            }
            let ranked = decision.solve(&problem, Some(&structure)).unwrap();
            let manual = LinearSolveRequest::new(backend, expected_plan)
                .solve(&problem)
                .unwrap();
            assert_eq!(ranked.values(), manual.values());
            assert_eq!(ranked.report(), manual.report());
            for (actual, expected) in ranked.values().iter().zip(&expected) {
                assert!((actual - expected).abs() <= 2_f64.powi(-40));
            }
            // Independently apply the literal mathematical matrix, not provider output.
            let x = ranked.values();
            let residual = if diagonal {
                vec![4. * x[0] + x[1] - 6., x[0] + 3. * x[1] - 7.]
            } else {
                vec![2. * x[0] + x[2] - 5., 3. * x[1] + x[2], x[0] + x[1]]
            };
            assert!(residual.iter().all(|r| r.abs() <= 2_f64.powi(-38)));
            let mismatched =
                CanonicalCsrSystemView::new(matrix, LinearOperatorProperties::General).unwrap();
            assert!(
                decision
                    .solve(&mismatched.linear_problem().unwrap(), Some(&structure))
                    .unwrap_err()
                    .message()
                    .contains("profile.operator-properties-mismatch")
            );
        }
    }
}

#[test]
fn unclaimed_diagonal_ranks_only_independently_admissible_candidates() {
    let faer = FaerLinearSolver;
    let system = CanonicalCsrSystemView::new(&Fixture, LinearOperatorProperties::General).unwrap();
    let problem = system.linear_problem().unwrap();
    for objective in [
        SolverPlanningObjective::Robust,
        SolverPlanningObjective::Fast,
        SolverPlanningObjective::LowMemory,
    ] {
        let profile =
            HostSerialSolverProfile::canonical_csr(LinearOperatorProperties::General, None, None);
        let decision = ResolvedHostSerialSolverPlan::resolve(
            profile,
            objective,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
            &REFERENCE_LINEAR_SOLVER,
            &faer,
        )
        .unwrap();
        // A diagonal has not been established during planning: both Jacobi
        // candidates are ineligible. Identity LU is the sole admissible tuple.
        assert_eq!(decision.selected_candidate_id(), FAER_SPARSE_LU_ID);
        assert_eq!(decision.solver_plan(), faer_sparse_lu_plan());
        assert_eq!(decision.solver_provider(), faer.provider());
        let solution = decision.solve(&problem, None).unwrap();
        let manual = LinearSolveRequest::new(&faer, faer_sparse_lu_plan())
            .solve(&problem)
            .unwrap();
        assert_eq!(solution.values(), manual.values());
        assert_eq!(solution.report(), manual.report());
        assert!((solution.values()[0] - 1.).abs() <= 2_f64.powi(-40));
        assert!((solution.values()[1] - 2.).abs() <= 2_f64.powi(-40));
        assert!((4. * solution.values()[0] + solution.values()[1] - 6.).abs() <= 2_f64.powi(-38));
        assert!(
            (2. * solution.values()[0] + 3. * solution.values()[1] - 8.).abs() <= 2_f64.powi(-38)
        );
    }
    // Unknown is not fabricated absence. A genuinely asserted absence must
    // still reject this matrix's present diagonal before any numerical work.
    assert!(
        HostSerialSolverProfile::canonical_csr(
            LinearOperatorProperties::General,
            Some(false),
            None
        )
        .require_problem(&problem)
        .unwrap_err()
        .message()
        .contains("diagonal-availability-mismatch")
    );
}
