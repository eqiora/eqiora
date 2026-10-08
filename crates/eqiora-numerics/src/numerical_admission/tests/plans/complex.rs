use super::*;
use num_complex::Complex64 as C;

#[test]
fn complex_spatial_plan_retains_types_exact_solver_and_replay() {
    let geometry = cartesian_interval();
    let source = POISSON_INTERVAL
        .replace("variable potential: 1", "variable potential: complex<1>")
        .replace(
            "-div(grad(potential)) - source_scale = 0",
            "-div(grad(potential)) - math.complex(source_scale, 2*source_scale) = 0",
        );
    let model = scalar_box_model(&geometry, &source, "PoissonInterval", &["left", "right"]);
    let exact = exact_reference_linear(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(64).unwrap(),
    );
    let resolve = |method, request| {
        ResolvedCommonPlan::resolve(
            &model,
            cartesian_box_resources(&geometry, &[4]),
            method,
            CommonSolvePolicy::Linear(request),
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
    };
    let plan = resolve(CommonSpatialPolicy::Q1, exact).unwrap();
    let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
    let scalar = plan.as_scalar().unwrap();
    assert_eq!(
        scalar.fields().next().unwrap().1.scalar_domain(),
        eqiora_core::ScalarDomain::Complex
    );
    assert_eq!(scalar.solver_provider(), REFERENCE_LINEAR_SOLVER.provider());
    scalar.reauthenticate_portable_realization().unwrap();
    let RecognizedNativeModel::ComplexScalar(equations) = scalar.admission.recognized_model()
    else {
        panic!("typed complex equations");
    };
    let NativeMeshResources::Cartesian { mesh, .. } = scalar.admission.resources() else {
        panic!("Cartesian resources");
    };
    let structure = equations.algebraic_structure(None).unwrap();
    let checked = scalar
        .admission
        .linear
        .checked_complex_backend(&REFERENCE_LINEAR_SOLVER, Some(&structure))
        .unwrap();
    let output = equations
        .execute(
            NonZeroUsize::MIN,
            LinearSolveRequest::new(&checked, scalar.linear()),
            mesh.mesh(),
            |reactions, values| reactions.recover(values),
        )
        .unwrap();
    // -u''=1+2i on [0,1], zero endpoints: nodal Q1 values equal
    // (1+2i)x(1-x)/2, independently of the implementation's matrix.
    for (value, x) in output.fields[0].2.iter().zip([0., 0.25, 0.5, 0.75, 1.]) {
        assert!((*value - C::new(1., 2.) * (x * (1. - x) / 2.)).norm() < 1e-11);
    }
    let mut forged = scalar.clone();
    forged.fields[0].1 = eqiora_core::ValueType::scalar(
        eqiora_core::ScalarDomain::Real,
        forged.fields[0].1.dimension(),
    )
    .unwrap();
    assert!(forged.reauthenticate_portable_realization().is_err());
    assert!(resolve(CommonSpatialPolicy::CellCenteredTpfa, exact).is_err());
    let automatic = CommonLinearRequest::program_controlled(
        1e-12,
        1e-14,
        NonZeroUsize::new(64).unwrap(),
        SolverPlanningObjective::Robust,
    )
    .unwrap();
    assert!(
        resolve(CommonSpatialPolicy::Q1, automatic)
            .unwrap_err()
            .message()
            .contains("exact linear controls")
    );
    let cg = exact_reference_linear(
        LinearSolver::ConjugateGradient,
        1e-12,
        1e-14,
        NonZeroUsize::new(64).unwrap(),
    );
    assert!(resolve(CommonSpatialPolicy::Q1, cg).is_err());
}
