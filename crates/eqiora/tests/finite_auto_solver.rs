//! Ordinary real finite objectives share the admitted host-serial chooser.
use eqiora::artifact::ModelEnvelope;
use eqiora::compiler::CompiledModel;
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    LinearSolver, LinearSolverBackend, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverPlan, SolverPlanningObjective,
};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonAlgebraicState, CommonLinearRequest, CommonResult,
    CommonSolvePolicy, ResolvedCommonPlan,
};
use std::num::NonZeroUsize;

fn model() -> (
    ModelEnvelope,
    eqiora::RawId,
    eqiora::Id<eqiora::kinds::Observable>,
) {
    let compiled = CompiledModel::compile_selected("finite-auto.eqi", "model Finite() { variable x:1; variable y:1; relation first {4*x+y=6;} relation second {2*x+3*y=8;} observable objective:1=x+10*y; }", "Finite", &[]).unwrap();
    let (transaction, model, symbols) = compiled.into_parts();
    let x = symbols.get("x").unwrap();
    let observable = symbols.get("objective").unwrap().downcast().unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    (
        ModelEnvelope::from_program(&program).unwrap(),
        x,
        observable,
    )
}
fn request(objective: SolverPlanningObjective) -> CommonLinearRequest {
    CommonLinearRequest::program_controlled(
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
        objective,
    )
    .unwrap()
}
fn resolve(
    model: &ModelEnvelope,
    objective: SolverPlanningObjective,
    backend: &dyn LinearSolverBackend,
) -> CommonAlgebraicPlan {
    CommonAlgebraicPlan::resolve(
        model,
        CommonSolvePolicy::Linear(request(objective)),
        None,
        &[],
        None,
        backend,
    )
    .unwrap()
}
fn expected_plan(algorithm: LinearSolver, reduction: ReductionPolicy) -> SolverPlan {
    SolverPlan::new(algorithm, 1e-12, 1e-14, NonZeroUsize::new(100).unwrap())
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(reduction)
}

#[test]
fn real_finite_objectives_execute_reopen_and_preserve_exact_selection() {
    let (model, x, observable) = model();
    let faer = FaerLinearSolver;
    for (objective, candidate, backend, expected) in [
        (
            SolverPlanningObjective::Robust,
            "eqiora.reference.bicgstab-general-identity-reproducible-f64",
            &REFERENCE_LINEAR_SOLVER as &dyn LinearSolverBackend,
            expected_plan(
                LinearSolver::BiConjugateGradientStabilized,
                ReductionPolicy::Reproducible,
            ),
        ),
        (
            SolverPlanningObjective::Fast,
            "eqiora.faer.sparse-lu-general-identity-fast-f64",
            &faer,
            expected_plan(LinearSolver::SparseLu, ReductionPolicy::Fast),
        ),
        (
            // Known Reference numeric payload is preferred over an unproved Faer bound.
            SolverPlanningObjective::LowMemory,
            "eqiora.reference.bicgstab-general-identity-reproducible-f64",
            &REFERENCE_LINEAR_SOLVER,
            expected_plan(
                LinearSolver::BiConjugateGradientStabilized,
                ReductionPolicy::Reproducible,
            ),
        ),
    ] {
        let plan = resolve(&model, objective, &faer);
        let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
        assert_eq!(resolved.solver_planning_objective(), Some(objective));
        assert_eq!(
            resolved.solver_planning_policy_id(),
            Some("eqiora.solver-planning/v11")
        );
        assert_eq!(resolved.selected_solver_candidate_id(), Some(candidate));
        assert_eq!(resolved.linear_solver_provider(), Some(backend.provider()));
        assert_eq!(plan.linear(), expected);
        assert!(resolved.solver_planning_reasons().contains(&(
            "eqiora.reference.bicgstab-general-jacobi-reproducible-f64",
            "candidate.admitted"
        )));
        let initial = plan.initial_state(&[]).unwrap();
        let before = initial.to_bytes().unwrap();
        let result = plan.run_result(&initial, backend).unwrap();
        for (symbol, actual) in plan.symbols().iter().zip(result.finite_values().unwrap()) {
            let eqiora::kernel::SymbolRef::Field(field) = symbol else {
                unreachable!()
            };
            assert!((actual - if field.erase() == x { 1. } else { 2. }).abs() < 1e-11);
        }
        let observation = result
            .observe(&model, observable, &Default::default(), None)
            .unwrap();
        assert!((observation.value().real_scalar_value().unwrap().value() - 21.).abs() < 1e-10);
        assert!(result.finite_reference_residual_norm().unwrap() <= 1e-11);
        let evidence = result.solve_evidence(None, None).unwrap();
        assert_eq!(evidence.algorithm(), expected.algorithm());
        assert_eq!(evidence.preconditioner(), expected.preconditioner());
        assert_eq!(evidence.reduction(), expected.reduction());
        assert_eq!(evidence.solver().id(), backend.provider().id().as_str());
        assert!(evidence.true_residual_norm() <= evidence.residual_target());
        evidence.validate_plan(&resolved).unwrap();
        let reopened_plan =
            ResolvedCommonPlan::from_bytes(&resolved.to_bytes().unwrap(), &faer, &[]).unwrap();
        assert_eq!(reopened_plan, resolved);
        let reopened =
            CommonResult::from_bytes(&result.to_bytes().unwrap(), &reopened_plan).unwrap();
        assert_eq!(reopened, result);
        assert_eq!(initial.to_bytes().unwrap(), before);
        // Explicit manual controls continue to select the exact same solver/provider.
        let manual = CommonAlgebraicPlan::resolve(
            &model,
            CommonSolvePolicy::Linear(
                CommonLinearRequest::exact(expected, backend.provider()).unwrap(),
            ),
            None,
            &[],
            None,
            backend,
        )
        .unwrap();
        let manual = manual
            .run_result(&manual.initial_state(&[]).unwrap(), backend)
            .unwrap();
        assert_eq!(manual.finite_values(), result.finite_values());
        assert_eq!(
            manual.solve_evidence(None, None),
            result.solve_evidence(None, None)
        );
    }
}

#[test]
fn finite_auto_identity_binds_effective_selection_and_replay_does_not_switch_it() {
    let (model, _, _) = model();
    let faer = FaerLinearSolver;
    let available = resolve(&model, SolverPlanningObjective::Fast, &faer);
    let unavailable = resolve(
        &model,
        SolverPlanningObjective::Fast,
        &REFERENCE_LINEAR_SOLVER,
    );
    assert_ne!(available.identity(), unavailable.identity());
    assert_eq!(available.linear().algorithm(), LinearSolver::SparseLu);
    assert_eq!(
        unavailable.linear(),
        expected_plan(
            LinearSolver::BiConjugateGradientStabilized,
            ReductionPolicy::Reproducible
        )
    );
    let full = ResolvedCommonPlan::Algebraic(Box::new(available.clone()));
    let limited = ResolvedCommonPlan::Algebraic(Box::new(unavailable.clone()));
    assert!(limited.solver_planning_reasons().contains(&(
        "eqiora.faer.sparse-lu-general-identity-fast-f64",
        "catalog.provider-mismatch"
    )));
    let initial = available.initial_state(&[]).unwrap();
    assert!(CommonAlgebraicState::from_bytes(&initial.to_bytes().unwrap(), &unavailable).is_err());
    assert!(
        available
            .run_result(&initial, &REFERENCE_LINEAR_SOLVER)
            .is_err()
    );
    let result = available.run_result(&initial, &faer).unwrap();
    assert!(CommonResult::from_bytes(&result.to_bytes().unwrap(), &limited).is_err());
    assert!(
        ResolvedCommonPlan::from_bytes(&full.to_bytes().unwrap(), &REFERENCE_LINEAR_SOLVER, &[])
            .is_err()
    );
}

#[test]
fn complex_finite_objectives_execute_and_reopen_the_selected_typed_provider() {
    // Independent product: A*[1+2i,-2+i]=[-5+5i,-13+9i]. Both squared
    // magnitudes are 5, hence the real selected observable equals 55.
    let compiled = CompiledModel::compile_selected(
        "finite-complex-auto.eqi",
        "model Finite() { variable z:array<complex<1>,2>; relation value {math.complex(1,1)*z[0]+2*z[1]=math.complex(-5,5);math.complex(0,3)*z[0]+math.complex(4,-1)*z[1]=math.complex(-13,9);} observable objective:1=math.abs2(z[0])+10*math.abs2(z[1]); }",
        "Finite", &[],
    ).unwrap();
    let (transaction, model, symbols) = compiled.into_parts();
    let observable = symbols.get("objective").unwrap().downcast().unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let model = ModelEnvelope::from_program(&kernel).unwrap();
    let faer = FaerLinearSolver;
    for (objective, id, backend, algorithm, reduction) in [
        (
            SolverPlanningObjective::Robust,
            "eqiora.reference.bicgstab-general-identity-reproducible-complex-f64",
            &REFERENCE_LINEAR_SOLVER as &dyn LinearSolverBackend,
            LinearSolver::BiConjugateGradientStabilized,
            ReductionPolicy::Reproducible,
        ),
        (
            SolverPlanningObjective::Fast,
            "eqiora.faer.sparse-lu-general-identity-fast-complex-f64",
            &faer,
            LinearSolver::SparseLu,
            ReductionPolicy::Fast,
        ),
        (
            SolverPlanningObjective::LowMemory,
            "eqiora.faer.bicgstab-general-identity-fast-complex-f64",
            &faer,
            LinearSolver::BiConjugateGradientStabilized,
            ReductionPolicy::Fast,
        ),
    ] {
        let plan = resolve(&model, objective, &faer);
        let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
        assert_eq!(resolved.selected_solver_candidate_id(), Some(id));
        assert_eq!(
            resolved.solver_planning_policy_id(),
            Some("eqiora.solver-planning/v11")
        );
        assert_eq!(plan.linear(), expected_plan(algorithm, reduction));
        assert_eq!(resolved.linear_solver_provider(), Some(backend.provider()));
        let initial = plan.initial_state(&[]).unwrap();
        let result = plan.run_result(&initial, backend).unwrap();
        for (actual, expected) in result
            .finite_values()
            .unwrap()
            .iter()
            .zip([1., 2., -2., 1.])
        {
            assert!((actual - expected).abs() <= 2_f64.powi(-40));
        }
        let observation = result
            .observe(&model, observable, &Default::default(), None)
            .unwrap();
        assert!((observation.value().real_scalar_value().unwrap().value() - 55.).abs() < 1e-10);
        let evidence = result.solve_evidence(None, None).unwrap();
        evidence.validate_plan(&resolved).unwrap();
        assert_eq!(evidence.algorithm(), algorithm);
        assert_eq!(evidence.solver().id(), backend.provider().id().as_str());
        let reopened =
            ResolvedCommonPlan::from_bytes(&resolved.to_bytes().unwrap(), &faer, &[]).unwrap();
        assert_eq!(reopened, resolved);
        assert_eq!(
            CommonResult::from_bytes(&result.to_bytes().unwrap(), &reopened).unwrap(),
            result
        );
        let manual = CommonAlgebraicPlan::resolve(
            &model,
            CommonSolvePolicy::Linear(
                CommonLinearRequest::exact(plan.linear(), backend.provider()).unwrap(),
            ),
            None,
            &[],
            None,
            backend,
        )
        .unwrap();
        let manual_result = manual
            .run_result(&manual.initial_state(&[]).unwrap(), backend)
            .unwrap();
        assert_eq!(manual_result.finite_values(), result.finite_values());
        assert_eq!(
            manual_result.solve_evidence(None, None),
            result.solve_evidence(None, None)
        );
    }
    let fast = resolve(&model, SolverPlanningObjective::Fast, &faer);
    let reference = resolve(
        &model,
        SolverPlanningObjective::Fast,
        &REFERENCE_LINEAR_SOLVER,
    );
    assert_ne!(fast.identity(), reference.identity());
    assert!(
        ResolvedCommonPlan::from_bytes(
            &ResolvedCommonPlan::Algebraic(Box::new(fast))
                .to_bytes()
                .unwrap(),
            &REFERENCE_LINEAR_SOLVER,
            &[]
        )
        .is_err()
    );
}

#[test]
fn canonical_finite_request_retains_forecasts_and_payload_limit_on_reopen() {
    let (model, _, _) = model();
    let cost = eqiora::solver::SolverCostRequest::new(
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(2).unwrap(),
    )
    .unwrap()
    .with_iteration_forecasts(NonZeroUsize::new(3), NonZeroUsize::new(1))
    .with_numeric_payload_limit(NonZeroUsize::new(4096).unwrap());
    let exact = CommonLinearRequest::exact(
        expected_plan(
            LinearSolver::BiConjugateGradientStabilized,
            ReductionPolicy::Reproducible,
        ),
        REFERENCE_LINEAR_SOLVER.provider(),
    )
    .unwrap();
    for requested in [exact, request(SolverPlanningObjective::LowMemory)] {
        let requested = requested.with_cost(cost);
        let plan = CommonAlgebraicPlan::resolve(
            &model,
            CommonSolvePolicy::Linear(requested),
            None,
            &[],
            None,
            &FaerLinearSolver,
        )
        .unwrap();
        let plan = ResolvedCommonPlan::Algebraic(Box::new(plan));
        assert_eq!(
            plan.canonical_solve_request(),
            Some(CommonSolvePolicy::Linear(requested))
        );
        let bytes = plan.to_bytes().unwrap();
        let reopened = ResolvedCommonPlan::from_bytes(&bytes, &FaerLinearSolver, &[]).unwrap();
        assert_eq!(reopened.to_bytes().unwrap(), bytes);
        assert_eq!(
            reopened.canonical_solve_request(),
            plan.canonical_solve_request()
        );
        assert_eq!(
            reopened.solver_planning_costs(),
            plan.solver_planning_costs()
        );
    }
}
