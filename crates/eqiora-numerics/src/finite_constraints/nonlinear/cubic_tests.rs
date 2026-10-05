use super::*;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER};
use std::num::NonZeroUsize;

fn compile(source: &str) -> KernelProgram {
    let (transaction, model, _) = eqiora_compiler::compile("cubic.eqi", source)
        .unwrap()
        .pop()
        .unwrap()
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

#[test]
fn physical_residual_scale_is_shared_by_original_operands_and_differentials() {
    use eqiora_realization::PositivePhysicalScale;
    let kernel = compile(
        "model M(){parameter b:complex<1>=math.complex(2.25,4.5);variable z:complex<1>;relation r{z+0.25*math.abs2(z)*z=b;}}",
    );
    let problem = lower_finite_constraints(&kernel, None, true).unwrap();
    let reference = ConstraintRef::new(problem.relations[0].id, 0);
    let scale =
        PositivePhysicalScale::new(DynQuantity::new(4., DimExponents::DIMENSIONLESS)).unwrap();
    let scaled = problem
        .clone()
        .with_residual_scales(&[(reference, scale)])
        .unwrap();
    // At z=0, F=-b and J=I. Dividing both complex parts by four gives
    // F=(-9/16,-9/8), J=I/4, preserving the exact Newton correction b.
    assert_eq!(
        scaled.original_residual(&[0., 0.]).unwrap(),
        [-0.5625, -1.125]
    );
    let (actions, _) = scaled.equality_jacobian(&[0., 0.], &[]).unwrap();
    assert_eq!(actions.values, [-0.5625, -1.125]);
    assert_eq!(actions.unknown_jacobian, [0.25, 0., 0., 0.25]);
    assert!(
        problem
            .clone()
            .with_residual_scales(&[(reference, scale), (reference, scale)])
            .is_err()
    );
    assert!(
        problem
            .clone()
            .with_residual_scales(&[(ConstraintRef::new(reference.relation(), 1), scale)])
            .is_err()
    );
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let wrong = PositivePhysicalScale::new(DynQuantity::new(4., length)).unwrap();
    assert!(problem.with_residual_scales(&[(reference, wrong)]).is_err());
}

#[test]
fn scaling_cannot_erase_a_nonzero_residual_or_derivative() {
    // Both divisions overflow/underflow in binary64 although their operands
    // are finite. Neither is an admissible normalized numerical value.
    assert!(scaling::normalize(f64::from_bits(1), 2.).is_err());
    assert!(scaling::normalize(f64::MAX, 0.5).is_err());
    assert_eq!(scaling::normalize(0., f64::MAX).unwrap(), 0.);
    assert_eq!(
        scaling::normalize(f64::MIN_POSITIVE, 2.).unwrap(),
        f64::MIN_POSITIVE / 2.
    );
}

#[test]
fn nonlinear_norm_does_not_square_away_a_nonzero_residual() {
    let kernel = compile("model M(){parameter b:1=1e-200;variable w:1;relation r{w=b;}}");
    let problem = lower_finite_constraints(&kernel, None, true).unwrap();
    assert_eq!(
        problem.assess_seed(&[0.]).unwrap().equality_residual_norm(),
        1e-200
    );
    let controls = NonlinearSolvePlan::new(0., 1e-250, NonZeroUsize::new(4).unwrap(), 4).unwrap();
    assert!(
        problem
            .validate_nonlinear_values(&[0.], &[0.], controls)
            .is_err()
    );
    let kernel = compile("model M(){parameter b:1=1e200;variable w:1;relation r{w=b;}}");
    let problem = lower_finite_constraints(&kernel, None, true).unwrap();
    assert_eq!(
        problem.assess_seed(&[0.]).unwrap().equality_residual_norm(),
        1e200
    );
}

#[test]
fn real_and_complex_cubic_roots_share_newton_without_ordered_constraints() {
    // In complex coordinates the exact root is (1,2), b=(9/4,9/2).
    // The radial map is strongly monotone: its real Jacobian has eigenvalues
    // 1+alpha*r² and 1+3alpha*r², both >=1. A residual bound therefore bounds
    // the root error by the same value; no global Newton convergence is claimed.
    for (source, root) in [
        (
            "model M(){parameter alpha:1=0.25;parameter b:complex<1>=math.complex(2.25,4.5);variable z:complex<1>;relation r{z+alpha*math.abs2(z)*z=b;}}",
            vec![1., 2.],
        ),
        (
            "model M(){parameter alpha:1=0.25;parameter b:1=4;variable z:1;relation r{z+alpha*z*z*z=b;}}",
            vec![2.],
        ),
    ] {
        let kernel = compile(source);
        // Selecting Linear must still reject the nonlinear original operands.
        assert!(lower_finite_constraints(&kernel, None, false).is_err());
        let problem = lower_finite_constraints(&kernel, None, true).unwrap();
        let controls =
            NonlinearSolvePlan::new(0., 1e-12, NonZeroUsize::new(32).unwrap(), 16).unwrap();
        let linear = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-15,
            NonZeroUsize::new(32).unwrap(),
        )
        .unwrap();
        for seed in [vec![0.; root.len()], root.clone()] {
            let solved = problem
                .solve_at_point(
                    &seed,
                    controls,
                    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, linear),
                )
                .unwrap();
            for (actual, exact) in solved.values.iter().zip(&root) {
                assert!((actual - exact).abs() <= 1e-12);
            }
            assert!(solved.assessment.equality_residual_norm() <= 1e-12);
            assert_eq!(solved.iterations == 0, seed == root);
            assert_eq!(solved.iterations, solved.linear_solves.len());
            problem
                .validate_nonlinear_values(&seed, &solved.values, controls)
                .unwrap();
        }
        assert!(
            problem
                .validate_nonlinear_values(&root, &vec![0.; root.len()], controls)
                .is_err()
        );
    }
}
