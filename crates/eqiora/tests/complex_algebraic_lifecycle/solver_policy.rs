//! Scalar domain does not replace the explicitly admitted solver policy.
use super::*;

const SOURCE: &str = "model M(){variable x:1;variable y:1;relation r{4*x+y=6;2*x+3*y=8;}observable output:1=x+10*y;}";

#[test]
fn real_finite_execution_preserves_admitted_iterative_controls_and_replay() {
    for reference in [false, true] {
        let (document, plan, result) = solve_with(SOURCE, reference);
        let output = result
            .observe(
                plan.model_artifact(),
                document.aliases()["output"].downcast().unwrap(),
                &Default::default(),
                None,
            )
            .unwrap();
        // 4x+y=6 and 2x+3y=8 imply x=1,y=2, hence x+10y=21.
        assert!((output.value().real_scalar_value().unwrap().value() - 21.).abs() < 1e-10);
        let evidence = result.solve_evidence(None, None).unwrap();
        assert_eq!(evidence.algorithm(), plan.linear().algorithm());
        assert_eq!(evidence.reduction(), plan.linear().reduction());
        assert!(evidence.true_residual_norm() <= evidence.residual_target());
        if reference {
            assert_eq!(
                plan.linear().algorithm(),
                LinearSolver::BiConjugateGradientStabilized
            );
            assert_eq!(plan.linear().reduction(), ReductionPolicy::Reproducible);
            assert!(evidence.completed_iterations() > 0);
            let initial = plan.initial_state(&[]).unwrap();
            assert!(plan.run_result(&initial, &FaerLinearSolver).is_err());
        }
    }
}

#[test]
fn finite_policy_retains_capability_and_iteration_failure_boundaries() {
    let document = ModelDocument::compile("finite-policy.eqi", SOURCE).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    for algorithm in [
        LinearSolver::ConjugateGradient,
        LinearSolver::BiConjugateGradientStabilized,
    ] {
        let solver = SolverPlan::new(algorithm, 1e-13, 1e-15, NonZeroUsize::MIN)
            .unwrap()
            .with_reduction(ReductionPolicy::Reproducible);
        let admitted = CommonAlgebraicPlan::resolve(
            &model,
            CommonSolvePolicy::Linear(
                CommonLinearRequest::exact(solver, REFERENCE_LINEAR_SOLVER.provider()).unwrap(),
            ),
            None,
            &[],
            None,
            &REFERENCE_LINEAR_SOLVER,
        );
        if algorithm == LinearSolver::ConjugateGradient {
            // This nonsymmetric system provides no positive-definite assertion.
            assert!(admitted.is_err());
        } else {
            let plan = admitted.unwrap();
            let initial = plan.initial_state(&[]).unwrap();
            let before = initial.to_bytes().unwrap();
            assert!(plan.run_result(&initial, &REFERENCE_LINEAR_SOLVER).is_err());
            assert_eq!(initial.to_bytes().unwrap(), before);
        }
    }
}

#[test]
fn finite_newton_preserves_iterative_controls_for_real_and_nonholomorphic_values() {
    use eqiora::artifact::CanonicalModelArtifact;
    use eqiora_numerics::CommonInitialField;
    use eqiora_realization::NonlinearSolvePlan;
    for (kind, equation, seed, expected) in [
        ("1", "z*z=4", (3., 0.), 2.),
        (
            "complex<1>",
            "z+0.1*math.abs2(z)=math.complex(5.5,4)",
            (2., 3.),
            43.,
        ),
    ] {
        let source = format!(
            "model M(){{variable z:{kind};relation r{{{equation};}}observable output:1=math.real(z)+10*math.imag(z);}}"
        );
        let document = ModelDocument::compile("finite-newton-policy.eqi", &source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let linear = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-15,
            NonZeroUsize::new(8).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Reproducible);
        let plan = CommonAlgebraicPlan::resolve(
            &model,
            CommonSolvePolicy::Newton {
                nonlinear: NonlinearSolvePlan::new(0., 1e-12, NonZeroUsize::new(32).unwrap(), 16)
                    .unwrap(),
                linear: CommonLinearRequest::exact(linear, REFERENCE_LINEAR_SOLVER.provider())
                    .unwrap(),
            },
            None,
            &[],
            None,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
        let initial = plan
            .initial_state(&[CommonInitialField::finite(
                plan.model_artifact()
                    .artifact_reference()
                    .unwrap()
                    .artifact()
                    .clone(),
                document.aliases()["z"].downcast().unwrap(),
                eqiora::ValueShape::scalar(),
                vec![seed],
            )
            .unwrap()])
            .unwrap();
        let before = initial.to_bytes().unwrap();
        let result = plan.run_result(&initial, &REFERENCE_LINEAR_SOLVER).unwrap();
        let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
        let replay = CommonResult::from_bytes(&result.to_bytes().unwrap(), &resolved).unwrap();
        let output = replay
            .observe(
                plan.model_artifact(),
                document.aliases()["output"].downcast().unwrap(),
                &Default::default(),
                None,
            )
            .unwrap();
        // Real: positive root 2. Complex: z=3+4i, so z+0.1|z|²=5.5+4i.
        assert!((output.value().real_scalar_value().unwrap().value() - expected).abs() < 1e-10);
        assert_eq!(initial.to_bytes().unwrap(), before);
        assert_eq!(
            replay.solve_evidence(None, Some(0)).unwrap().algorithm(),
            linear.algorithm()
        );
    }
}
