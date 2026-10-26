//! Exact Jacobi admission consumes the actual fixed affine operator diagonal.
use super::*;
use eqiora_schema::kernel::SymbolRef;

fn request(preconditioner: PreconditionerPolicy) -> CommonLinearRequest {
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(preconditioner)
    .with_reduction(ReductionPolicy::Reproducible);
    CommonLinearRequest::exact(plan, REFERENCE_LINEAR_SOLVER.provider()).unwrap()
}

fn check_result(result: &crate::CommonResult, preconditioner: PreconditionerPolicy) {
    let evidence = result.solve_evidence(None, None).unwrap();
    assert_eq!(
        evidence.algorithm(),
        LinearSolver::BiConjugateGradientStabilized
    );
    assert_eq!(evidence.preconditioner(), preconditioner);
    assert_eq!(evidence.reduction(), ReductionPolicy::Reproducible);
    assert_eq!(
        evidence.solver().id(),
        REFERENCE_LINEAR_SOLVER.provider().id().as_str()
    );
    assert!(evidence.true_residual_norm() <= evidence.residual_target());
    evidence.validate_plan(result.plan()).unwrap();
    let bytes = result.to_bytes().unwrap();
    assert_eq!(
        crate::CommonResult::from_bytes(&bytes, result.plan())
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    let plan = ResolvedCommonPlan::from_bytes(
        &result.plan().to_bytes().unwrap(),
        &REFERENCE_LINEAR_SOLVER,
        &[],
    )
    .unwrap();
    assert_eq!(&plan, result.plan());
}

#[test]
fn jacobi_solver_diagonal_admits_ordinary_steady_spatial_execution_and_reopening() {
    let geometry = cartesian_interval();
    let model = scalar_box_model(
        &geometry,
        POISSON_INTERVAL,
        "PoissonInterval",
        &["left", "right"],
    );
    for preconditioner in [PreconditionerPolicy::Identity, PreconditionerPolicy::Jacobi] {
        let resolved = ResolvedCommonPlan::resolve(
            &model,
            cartesian_box_resources(&geometry, &[4]),
            CommonSpatialPolicy::Q1,
            CommonSolvePolicy::Linear(request(preconditioner)),
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
            None,
        )
        .unwrap();
        let plan = resolved.as_spatial().unwrap();
        assert_eq!(
            plan.linear(),
            request(preconditioner).exact_request().unwrap().0
        );
        let result = plan.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        // -u''=1, u(0)=u(1)=0 gives u=x(1-x)/2 at these Q1 nodes.
        let expected = [0., 3. / 32., 1. / 8., 3. / 32., 0.];
        let (_, values, _) = result.field_block(0, 0).unwrap();
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-11);
        }
        check_result(&result, preconditioner);
    }
}

fn finite(
    source: &str,
) -> (
    ModelEnvelope,
    eqiora_core::RawId,
    eqiora_core::Id<eqiora_core::entity::kinds::Observable>,
) {
    let compiled =
        CompiledModel::compile_selected("solver-diagonal.eqi", source, "Finite", &[]).unwrap();
    let (transaction, model, symbols) = compiled.into_parts();
    let x = symbols.get("x").unwrap();
    let observable = symbols.get("objective").unwrap().downcast().unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    (ModelEnvelope::from_program(&kernel).unwrap(), x, observable)
}

#[test]
fn jacobi_solver_diagonal_admits_a_structurally_different_finite_model() {
    let (model, x, observable) = finite(
        "model Finite() { variable x:1; variable y:1; relation first {4*x+y=6;} relation second {2*x+3*y=8;} observable objective:1=x+10*y; }",
    );
    for preconditioner in [PreconditionerPolicy::Identity, PreconditionerPolicy::Jacobi] {
        let plan = CommonAlgebraicPlan::resolve(
            &model,
            CommonSolvePolicy::Linear(request(preconditioner)),
            None,
            &[],
            None,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
        let initial = plan.initial_state(&[]).unwrap();
        let before = initial.to_bytes().unwrap();
        let result = plan.run_result(&initial, &REFERENCE_LINEAR_SOLVER).unwrap();
        for (symbol, actual) in plan.symbols().iter().zip(result.finite_values().unwrap()) {
            let SymbolRef::Field(field) = symbol else {
                unreachable!()
            };
            let expected = if field.erase() == x { 1. } else { 2. };
            assert!((actual - expected).abs() < 1e-11);
        }
        assert!(result.finite_reference_residual_norm().unwrap() <= 1e-11);
        let observation = result
            .observe(&model, observable, &Default::default(), None)
            .unwrap();
        assert!((observation.value().real_scalar_value().unwrap().value() - 21.).abs() < 1e-10);
        check_result(&result, preconditioner);
        assert_eq!(initial.to_bytes().unwrap(), before);
    }
}

#[test]
fn jacobi_solver_diagonal_rejects_an_ordinary_missing_finite_diagonal() {
    // One declared coordinate is absent from both rows, independently of row/Field ordering.
    let (model, _, _) = finite(
        "model Finite() { variable x:1; variable y:1; relation first {x=1;} relation second {2*x=2;} observable objective:1=x+y; }",
    );
    let error = CommonAlgebraicPlan::resolve(
        &model,
        CommonSolvePolicy::Linear(request(PreconditionerPolicy::Jacobi)),
        None,
        &[],
        None,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap_err();
    assert!(
        error
            .message()
            .contains("profile.complete-diagonal-required"),
        "{error}"
    );
}
