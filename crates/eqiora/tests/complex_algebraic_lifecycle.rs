//! One mathematical Field retains both parts through the ordinary finite lifecycle.
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::solver::{
    LinearSolver, LinearSolverBackend, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan,
};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonAlgebraicState, CommonLinearRequest, CommonResult,
    CommonSolvePolicy, ResolvedCommonPlan,
};
use std::num::NonZeroUsize;

fn solve(source: &str) -> (ModelDocument, CommonAlgebraicPlan, CommonResult) {
    solve_with(source, true)
}

fn solve_with(source: &str, complex: bool) -> (ModelDocument, CommonAlgebraicPlan, CommonResult) {
    let backend: &dyn LinearSolverBackend = if complex {
        &REFERENCE_LINEAR_SOLVER
    } else {
        &FaerLinearSolver
    };
    let document = ModelDocument::compile("complex.eqi", source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        if complex {
            LinearSolver::BiConjugateGradientStabilized
        } else {
            LinearSolver::SparseLu
        },
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(if complex {
        ReductionPolicy::Reproducible
    } else {
        ReductionPolicy::Fast
    });
    let request =
        CommonSolvePolicy::Linear(CommonLinearRequest::exact(solver, backend.provider()).unwrap());
    let plan = CommonAlgebraicPlan::resolve(&model, request, None, backend).unwrap();
    let initial = plan.initial_state(&[]).unwrap();
    assert_eq!(
        CommonAlgebraicState::from_bytes(&initial.to_bytes().unwrap(), &plan).unwrap(),
        initial
    );
    let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
    let replayed = ResolvedCommonPlan::from_bytes(
        &resolved.to_bytes().unwrap(),
        &FaerLinearSolver,
        eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
    )
    .unwrap();
    assert_eq!(replayed, resolved);
    let result = replayed
        .as_algebraic()
        .unwrap()
        .run_result(&initial, backend)
        .unwrap();
    assert_eq!(
        CommonResult::from_bytes(&result.to_bytes().unwrap(), &replayed).unwrap(),
        result
    );
    (document, plan, result)
}

#[test]
fn complex_equation_and_real_equation_share_plan_run_and_real_observables() {
    for (domain, equation, observable, expected, coordinates) in [
        (
            "complex<1>",
            "math.complex(1,-2)*z=math.complex(11,-2)",
            "math.abs2(z)",
            25.,
            2,
        ),
        ("1", "2*z=6", "z*z", 9., 1),
    ] {
        let source = format!(
            "model M(){{variable z:{domain};relation r{{{equation};}}observable output:1={observable};}}"
        );
        let (document, plan, result) = solve_with(&source, domain == "complex<1>");
        assert_eq!(plan.symbols().len(), 1);
        assert_eq!(plan.coordinate_count(), coordinates);
        let observed = result
            .observe(
                plan.model_artifact(),
                document.aliases()["output"].downcast().unwrap(),
                None,
            )
            .unwrap();
        // (1-2i)(3+4i)=11-2i, hence |z|²=25. Binary64 solver
        // tolerance and this small well-conditioned system bound output error below 1e-10.
        assert!((observed.value().real_scalar_value().unwrap().value() - expected).abs() < 1e-10);
    }
}

#[test]
fn shaped_complex_field_preserves_channel_order_and_explicit_real_output() {
    let source = "model M(){variable z: array<complex<1>,2>;relation r{math.complex(1,-2)*z=[math.complex(11,-2),math.complex(0,5)];}observable output:1=math.real(z[0])+math.imag(z[1])+math.abs2(z[1]);}";
    let (document, plan, result) = solve(source);
    assert_eq!(plan.symbols().len(), 1);
    assert_eq!(plan.coordinate_count(), 4);
    // z=[3+4i,-2+i], so Re(z[0])+Im(z[1])+|z[1]|²=3+1+5=9.
    let observed = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            None,
        )
        .unwrap();
    assert!((observed.value().real_scalar_value().unwrap().value() - 9.).abs() < 1e-10);
}

#[test]
fn replay_rejects_a_discarded_imaginary_part_before_digest_validation() {
    let (_, plan, result) = solve(
        "model M(){variable z:complex<1>;relation r{z=math.complex(3,4);}observable output:1=math.abs2(z);}",
    );
    let mut payload: serde_json::Value =
        serde_json::from_slice(&result.to_bytes().unwrap()).unwrap();
    payload["content"]["payload"]["values"][1] = 0.into();
    let error = CommonResult::from_bytes(
        &serde_json::to_vec(&payload).unwrap(),
        &ResolvedCommonPlan::Algebraic(Box::new(plan)),
    )
    .unwrap_err();
    assert!(
        error.message().contains("original Model equality residual"),
        "{error:?}"
    );
}

#[test]
fn contextual_zero_accepts_a_complete_complex_array_residual() {
    let source = "model M(){variable z:array<complex<1>,2>;relation r{z-[math.complex(3,4),math.complex(-2,1)]=0;}observable output:1=math.abs2(z[1]);}";
    let (document, plan, result) = solve(source);
    let observed = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            None,
        )
        .unwrap();
    assert!((observed.value().real_scalar_value().unwrap().value() - 5.).abs() < 1e-10);
}

#[test]
fn finite_admission_counts_both_parts_before_expanding_large_fields() {
    let source = "model M(){variable z:array<complex<1>,129>;relation r{z=0;}}";
    let document = ModelDocument::compile("bounded.eqi", source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let request = CommonSolvePolicy::Linear(
        CommonLinearRequest::exact(solver, FaerLinearSolver.provider()).unwrap(),
    );
    let error = CommonAlgebraicPlan::resolve(&model, request, None, &FaerLinearSolver).unwrap_err();
    assert!(
        error.message().contains("256 real coordinates"),
        "{error:?}"
    );
}

#[test]
fn general_complex_two_by_two_system_uses_the_typed_reference_path_through_plan_replay() {
    let source = "model M(){
      variable z:array<complex<1>,2>;
      relation r{
        math.complex(1,1)*z[0]+2*z[1]=math.complex(-5,5);
        math.complex(0,3)*z[0]+math.complex(4,-1)*z[1]=math.complex(-13,9);
      }
      observable first:complex<1>=z[0];
      observable second:complex<1>=z[1];
    }";
    let (document, plan, result) = solve(source);
    assert_eq!(plan.solver_provider(), REFERENCE_LINEAR_SOLVER.provider());
    for (name, (re, im)) in [("first", (1., 2.)), ("second", (-2., 1.))] {
        let observation = result
            .observe(
                plan.model_artifact(),
                document.aliases()[name].downcast().unwrap(),
                None,
            )
            .unwrap();
        let actual = observation.value().component(0).unwrap();
        assert!((actual.0 - re).hypot(actual.1 - im) < 1e-10);
    }
}

#[test]
fn conjugate_dependence_retains_its_real_linear_profile() {
    let (document, plan, result) = solve_with(
        "model M(){variable z:complex<1>;relation r{math.conj(z)=math.complex(3,-4);}observable output:complex<1>=z;}",
        false,
    );
    assert_eq!(plan.solver_provider(), FaerLinearSolver.provider());
    let observation = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            None,
        )
        .unwrap();
    assert_eq!(observation.value().component(0).unwrap(), (3., 4.));
}
