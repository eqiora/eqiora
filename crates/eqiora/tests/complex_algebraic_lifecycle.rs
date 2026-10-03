//! One mathematical Field retains both parts through the ordinary finite lifecycle.
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonAlgebraicState, CommonLinearRequest, CommonResult,
    CommonSolvePolicy, ResolvedCommonPlan,
};
use std::num::NonZeroUsize;

fn solve(source: &str) -> (ModelDocument, CommonAlgebraicPlan, CommonResult) {
    let document = ModelDocument::compile("complex.eqi", source).unwrap();
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
    let plan = CommonAlgebraicPlan::resolve(&model, request, None, &FaerLinearSolver).unwrap();
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
        .run_result(&initial, &FaerLinearSolver)
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
        let (document, plan, result) = solve(&source);
        assert_eq!(plan.symbols().len(), 1);
        assert_eq!(plan.coordinate_count(), coordinates);
        let observed = result
            .observe(
                plan.model_artifact(),
                document.aliases()["output"].downcast().unwrap(),
                None,
            )
            .unwrap();
        // (1-2i)(3+4i)=11-2i, hence |z|²=25. Binary64 SparseLU
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
