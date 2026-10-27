use super::*;

#[test]
fn isotropic_bilinear_rhs_has_an_exact_one_step_hermitian_solution() {
    // det(A)=-3, eigenvalues -1 and3. b=[z,iz], z=-1-2i, is an
    // eigenvector of eigenvalue-1: b^T*b=0 but b^H*b=10. Thus Hermitian
    // BiCG alpha=10/(-10)=-1 and s=0, in one iteration, for A and A^H.
    let source = Matrix([
        [C::new(1., 0.), C::new(0., 2.)],
        [C::new(0., -2.), C::new(1., 0.)],
    ]);
    let rhs = [C::new(-1., -2.), C::new(2., -1.)];
    for orientation in [Orientation::Normal, Orientation::ConjugateTransposed] {
        let oriented = Oriented::new(&source, orientation).unwrap();
        for preconditioner in [PreconditionerPolicy::Identity, PreconditionerPolicy::Jacobi] {
            let request = SolverPlan::new(
                LinearSolver::BiConjugateGradientStabilized,
                1e-13,
                1e-13,
                1.try_into().unwrap(),
            )
            .unwrap()
            .with_preconditioner(preconditioner)
            .with_reduction(ReductionPolicy::Fast);
            for method in [false, true] {
                let criterion = |x: &[C]| {
                    // Both original physical rows must be satisfied separately.
                    Ok((x[0] + C::new(0., 2.) * x[1] - rhs[0]).norm() <= 1e-13
                        && (C::new(0., -2.) * x[0] + x[1] - rhs[1]).norm() <= 1e-13)
                };
                let mut problem =
                    LinearProblem::new(&oriented, &rhs, LinearOperatorProperties::Hermitian)
                        .unwrap();
                if method {
                    problem = problem.with_candidate_check(&criterion);
                }
                let solved = FaerLinearSolver.solve(&problem, request).unwrap();
                assert_eq!(solved.values(), &X);
                assert_eq!(solved.report().completed_iterations(), 1);
                assert_eq!(solved.report().solver_plan(), request);
                assert_eq!(solved.report().solver_provider(), FAER_SOLVER_PROVIDER);
                assert_eq!(solved.report().orientation(), orientation);
                assert_eq!(solved.report().true_residual_norm(), 0.);
                assert!(criterion(solved.values()).unwrap());
                // Refusing the exact candidate cannot be bypassed by the happy
                // alpha exit or a restart outside this same one-iteration cap.
                let reject = |_: &[C]| Ok(false);
                assert!(
                    FaerLinearSolver
                        .solve(&problem.with_candidate_check(&reject), request)
                        .is_err()
                );
            }
        }
    }
}

#[test]
fn complex_omega_uses_the_hermitian_least_residual_product() {
    // A=diag(1+i,2), b=(1,1), x0=0. Exact arithmetic gives
    // alpha=(3-i)/5, s=(1-2i)/5*(1,-1), omega=(3-i)/6.
    // Bilinear omega would instead be (7-i)/10. The independent first
    // omega point is ((19-13i)/30,(17+i)/30), residual norm sqrt(1/15).
    let source = Matrix([
        [C::new(1., 1.), C::new(0., 0.)],
        [C::new(0., 0.), C::new(2., 0.)],
    ]);
    let rhs = [C::new(1., 0.); 2];
    let problem = LinearProblem::new(&source, &rhs, LinearOperatorProperties::General).unwrap();
    // This independently chosen threshold separates alpha and omega:
    // sqrt(1/15) < 1/3 < sqrt(2/5). One iteration therefore must publish omega.
    let request = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        0.,
        1. / 3.,
        1.try_into().unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let first = FaerLinearSolver.solve(&problem, request).unwrap();
    assert_values(
        first.values(),
        &[C::new(19., -13.) / 30., C::new(17., 1.) / 30.],
    );
    assert_eq!(first.report().completed_iterations(), 1);
    assert_eq!(first.report().solver_plan(), request);
    assert_eq!(first.report().solver_provider(), FAER_SOLVER_PROVIDER);
    assert!((first.report().true_residual_norm() - (1_f64 / 15.).sqrt()).abs() <= TOLERANCE);
    // The ordinary provider still solves the ORIGINAL problem to its original
    // tight policy; the intermediate omega witness is not a relaxed solve.
    let solved = FaerLinearSolver
        .solve(
            &problem,
            plan(
                LinearSolver::BiConjugateGradientStabilized,
                PreconditionerPolicy::Identity,
            ),
        )
        .unwrap();
    assert_values(solved.values(), &[C::new(0.5, -0.5), C::new(0.5, 0.)]);
    assert!(solved.report().true_residual_norm() <= solved.report().residual_target());
}
