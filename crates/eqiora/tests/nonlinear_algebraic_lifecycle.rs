//! Positive-branch nonlinear execution uses the ordinary exact no-Mesh lifecycle.
use eqiora::api::ModelDocument;
use eqiora::artifact::{CanonicalModelArtifact, ModelEnvelope};
use eqiora::solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
use eqiora::{DimExponents, DynQuantity};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::finite_constraints::{
    ConstraintRef, ConstraintTolerance, FiniteConstraintEnforcement,
};
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonAlgebraicState, CommonInitialField, CommonLinearRequest,
    CommonResult, CommonSolvePolicy, ResolvedCommonPlan,
};
use eqiora_realization::NonlinearSolvePlan;
use std::num::NonZeroUsize;

fn fixture(p: f64, margin: f64, equality: &str) -> (ModelDocument, CommonAlgebraicPlan) {
    let source = format!(
        "model Root(){{parameter p:1={p};variable w:1;relation root{{{equality};inequality(p>=0);inequality(w>=0);}}observable output:1=w+p;}}"
    );
    let document = ModelDocument::compile("root.eqi", &source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let relation = document.aliases()["root"].downcast().unwrap();
    let enforcement = FiniteConstraintEnforcement::strict_interior(
        [1, 2]
            .into_iter()
            .map(|ordinal| {
                ConstraintTolerance::inequality(
                    ConstraintRef::new(relation, ordinal),
                    DynQuantity::new(margin, DimExponents::DIMENSIONLESS),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let linear = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let solve = CommonSolvePolicy::Newton {
        nonlinear: NonlinearSolvePlan::new(0.0, 1e-12, NonZeroUsize::new(32).unwrap(), 16).unwrap(),
        linear: CommonLinearRequest::exact(linear, FaerLinearSolver.provider()).unwrap(),
    };
    let plan = CommonAlgebraicPlan::resolve(
        &model,
        solve,
        Some(enforcement),
        &[],
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    (document, plan)
}
fn seed(document: &ModelDocument, plan: &CommonAlgebraicPlan, value: f64) -> CommonInitialField {
    CommonInitialField::finite(
        plan.model_artifact()
            .artifact_reference()
            .unwrap()
            .artifact()
            .clone(),
        document.aliases()["w"].downcast().unwrap(),
        eqiora::ValueShape::scalar(),
        vec![(value, 0.)],
    )
    .unwrap()
}

#[test]
fn nonlinear_plan_state_result_and_observable_replay_share_the_positive_root() {
    let (document, plan) = fixture(4.0, 1e-8, "w*w=p");
    let state = plan.initial_state(&[seed(&document, &plan, 1.0)]).unwrap();
    let other_seed = plan.initial_state(&[seed(&document, &plan, 3.0)]).unwrap();
    assert_ne!(state.identity(), other_seed.identity());
    assert_eq!(
        CommonAlgebraicState::from_bytes(&state.to_bytes().unwrap(), &plan).unwrap(),
        state
    );
    let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
    let replayed = ResolvedCommonPlan::from_bytes(
        &resolved.to_bytes().unwrap(),
        &FaerLinearSolver,
        eqiora::time::TimeBackendCapabilities::new(
            eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
            &[eqiora::ScalarDomain::Real, eqiora::ScalarDomain::Complex],
            &[eqiora::ScalarType::F64],
        ),
    )
    .unwrap();
    assert_eq!(replayed, resolved);
    let result = replayed
        .as_algebraic()
        .unwrap()
        .run_result(&state, &FaerLinearSolver)
        .unwrap();
    // w²=4 on w>0 implies w=2, so O=w+p=6. Residual tolerance 1e-12
    // bounds root error by 1e-12/(w+2); the bound below is conservative.
    assert!((result.finite_values().unwrap()[0] - 2.0).abs() <= 1e-12);
    let observed = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
        )
        .unwrap();
    assert!((observed.value().real_scalar_value().unwrap().value() - 6.0).abs() <= 1e-12);
    assert!(result.nonlinear_iterations().unwrap() > 0);
    assert_eq!(
        CommonResult::from_bytes(&result.to_bytes().unwrap(), &resolved).unwrap(),
        result
    );
    let exact = plan.initial_state(&[seed(&document, &plan, 2.0)]).unwrap();
    let exact_result = plan.run_result(&exact, &FaerLinearSolver).unwrap();
    assert_eq!(exact_result.nonlinear_iterations(), Some(0));
    assert_eq!(
        CommonResult::from_bytes(&exact_result.to_bytes().unwrap(), &resolved).unwrap(),
        exact_result
    );
}

#[test]
fn branch_seed_and_plan_ownership_fail_closed() {
    let (document, plan) = fixture(4.0, 1e-8, "w*w=p");
    assert!(plan.initial_state(&[]).is_err());
    assert!(plan.initial_state(&[seed(&document, &plan, -2.0)]).is_err());
    let (other, other_plan) = fixture(9.0, 1e-8, "w*w=p");
    assert!(
        plan.initial_state(&[seed(&other, &other_plan, 1.0)])
            .is_err()
    );
    let state = plan.initial_state(&[seed(&document, &plan, 1.0)]).unwrap();
    assert!(other_plan.run_result(&state, &FaerLinearSolver).is_err());
    let (_, margin_plan) = fixture(4.0, 1e-6, "w*w=p");
    assert_ne!(plan.identity(), margin_plan.identity());
    assert!(CommonAlgebraicState::from_bytes(&state.to_bytes().unwrap(), &margin_plan).is_err());
    let (zero, zero_plan) = fixture(0.0, 1e-8, "w*w=p");
    assert!(
        zero_plan
            .initial_state(&[seed(&zero, &zero_plan, 1.0)])
            .is_err()
    );
    let (singular, singular_plan) = fixture(4.0, 1e-8, "(w-2)*(w-2)=0");
    let singular_state = singular_plan
        .initial_state(&[seed(&singular, &singular_plan, 2.0)])
        .unwrap();
    let error = singular_plan
        .run_result(&singular_state, &FaerLinearSolver)
        .unwrap_err();
    assert!(
        error.message().contains("Jacobian is singular"),
        "{error:?}"
    );
}

#[test]
fn result_replay_rejects_a_fabricated_zero_update_record_before_digest_check() {
    let (document, plan) = fixture(4.0, 1e-8, "w*w=p");
    let state = plan.initial_state(&[seed(&document, &plan, 1.0)]).unwrap();
    let result = plan.run_result(&state, &FaerLinearSolver).unwrap();
    let mut payload: serde_json::Value =
        serde_json::from_slice(&result.to_bytes().unwrap()).unwrap();
    payload["content"]["payload"]["solve"]["iterations"] = 0.into();
    payload["content"]["payload"]["solve"]["linear_solves"] = serde_json::json!([]);
    let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan));
    let error =
        CommonResult::from_bytes(&serde_json::to_vec(&payload).unwrap(), &resolved).unwrap_err();
    assert!(error.message().contains("zero-update"), "{error:?}");
}

#[test]
fn accepted_finite_point_separates_residual_partials_and_reduced_output_actions() {
    use eqiora::differentiation::{
        AcceptedOutputLinearization, adjoint_output_gradient, forward_output_sensitivity,
    };
    use eqiora::ir::{LinearizedOutput, LinearizedRelation, RelationTangent};
    use eqiora::solver::{LinearOperatorProperties, LinearSolveRequest};
    let (document, plan) = fixture(4.0, 1e-8, "w*w=p");
    let initial = plan.initial_state(&[seed(&document, &plan, 1.0)]).unwrap();
    let parameter = document.parameter_ref("p").unwrap().id();
    let observable = document.observable_ref("output").unwrap().id();
    for (p, w) in [(4.0, 2.0), (9.0, 3.0)] {
        let point = plan
            .differentiate(
                &initial,
                &[parameter],
                Some(&[p]),
                observable,
                &FaerLinearSolver,
            )
            .unwrap();
        assert!(point.receipt().is_none());
        assert_eq!(point.nonlinear_initial_state().unwrap(), &initial);
        assert!(point.nonlinear_iterations().unwrap() > 0);
        assert!((point.output_values()[0] - (w + p)).abs() < 1e-12);
        let mut partial = [0.0];
        point
            .relation()
            .jvp(RelationTangent::Parameter(&[1.0]), &mut partial)
            .unwrap();
        assert_eq!(partial, [-1.0]);
        LinearizedOutput::jvp(&point, &[0.0], &[1.0], &mut partial).unwrap();
        assert_eq!(partial, [1.0]);
        let accepted = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
            point.relation(),
            &point,
            point.relation().state_jacobian(),
            point.residual_target(),
        )
        .unwrap();
        let forward = forward_output_sensitivity(
            &accepted,
            &[1.0],
            LinearOperatorProperties::General,
            LinearSolveRequest::new(&FaerLinearSolver, plan.linear()),
        )
        .unwrap();
        let (_, tangent) = forward.into_parts();
        let reverse = adjoint_output_gradient(
            &accepted,
            &[1.0],
            LinearOperatorProperties::General,
            LinearSolveRequest::new(&FaerLinearSolver, plan.linear()),
        )
        .unwrap();
        let (_, gradient) = reverse.into_parts();
        // R_w=2w, R_p=-1 imply dw/dp=1/(2w); O=w+p adds the direct 1.
        let expected = 1.0 + 1.0 / (2.0 * w);
        assert!((tangent[0] - expected).abs() < 1e-12);
        assert!((gradient[0] - expected).abs() < 1e-12);
    }
    let error = plan
        .differentiate(
            &initial,
            &[parameter],
            Some(&[0.0]),
            observable,
            &FaerLinearSolver,
        )
        .unwrap_err();
    assert!(error.message().contains("inequality"), "{error:?}");
    let exact = plan.initial_state(&[seed(&document, &plan, 2.0)]).unwrap();
    let point = plan
        .differentiate(&exact, &[parameter], None, observable, &FaerLinearSolver)
        .unwrap();
    assert_eq!(point.nonlinear_iterations(), Some(0));
    assert!(point.receipt().is_none());
}

#[test]
fn common_program_separates_partial_and_reduced_actions_at_owned_nonlinear_points() {
    use eqiora::api::{DerivativeImplementation, DifferentiableProgram};
    let (document, plan) = fixture(4.0, 1e-8, "w*w=p");
    let initial = plan.initial_state(&[seed(&document, &plan, 2.0)]).unwrap();
    let input = document.parameter_ref("p").unwrap();
    let output = document.observable_ref("output").unwrap();
    let program = DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(plan.clone())),
        std::slice::from_ref(&input),
        &output,
        Some(initial.clone()),
        &FaerLinearSolver,
    )
    .unwrap();
    let default = program.evaluate(&[4.0]).unwrap();
    assert_eq!(default.primal().evidence().nonlinear_iterations(), Some(0));
    assert!(default.primal().evidence().receipt().is_none());
    assert!(default.primal().evidence().primal_solve().is_none());
    assert_eq!(
        program.identity().initial_state_identity(),
        Some(initial.identity())
    );
    for (p, w, total) in [(4.0, 2.0, 1.25), (9.0, 3.0, 7.0 / 6.0)] {
        let point = program.evaluate(&[p]).unwrap();
        assert!((point.accepted_unknowns()[0] - w).abs() <= 1e-12);
        assert!((point.primal().output()[0] - (w + p)).abs() <= 1e-12);
        assert_eq!(point.residual_jvp(&[0.0], &[1.0]).unwrap(), [-1.0]);
        assert!((point.residual_jvp(&[1.0], &[0.0]).unwrap()[0] - 2.0 * w).abs() <= 2e-12);
        let (rw, rp) = point.residual_vjp(&[1.0]).unwrap();
        assert!((rw[0] - 2.0 * w).abs() <= 2e-12);
        assert_eq!(rp, [-1.0]);
        assert_eq!(point.output_partial_jvp(&[0.0], &[1.0]).unwrap(), [1.0]);
        assert_eq!(
            point.output_partial_vjp(&[1.0]).unwrap(),
            (vec![1.0], vec![1.0])
        );
        assert!((point.jvp(&[1.0]).unwrap().tangent()[0] - total).abs() <= 1e-12);
        let reverse = point.vjp(&[1.0]).unwrap();
        assert!((reverse.input_cotangent()[0] - total).abs() <= 1e-12);
        assert_eq!(
            reverse.evidence().implementation(),
            DerivativeImplementation::OperatorIr
        );
        assert_eq!(
            reverse.evidence().nonlinear_initial_state_identity(),
            Some(initial.identity())
        );
        assert_eq!(
            reverse.evidence().nonlinear_accepted_unknowns(),
            Some(point.accepted_unknowns())
        );
        assert!(point.residual_jvp(&[], &[1.0]).is_err());
        assert!(point.output_partial_vjp(&[]).is_err());
    }
    // Another evaluation cannot replace the point retained for a paired reverse action.
    assert_eq!(default.vjp(&[1.0]).unwrap().input_cotangent(), [1.25]);
    assert!(program.evaluate(&[0.0]).is_err());
    assert!(program.evaluate(&[f64::NAN]).is_err());
    assert!(program.evaluate(&[]).is_err());
    let (foreign, foreign_plan) = fixture(9.0, 1e-8, "w*w=p");
    let foreign_initial = foreign_plan
        .initial_state(&[seed(&foreign, &foreign_plan, 2.0)])
        .unwrap();
    let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan));
    assert!(
        DifferentiableProgram::compile(
            resolved.clone(),
            std::slice::from_ref(&input),
            &output,
            Some(foreign_initial),
            &FaerLinearSolver
        )
        .is_err()
    );
    assert!(
        DifferentiableProgram::compile(
            resolved.clone(),
            &[foreign.parameter_ref("p").unwrap()],
            &output,
            Some(initial.clone()),
            &FaerLinearSolver
        )
        .is_err()
    );
    assert!(
        DifferentiableProgram::compile(
            resolved.clone(),
            std::slice::from_ref(&input),
            &foreign.observable_ref("output").unwrap(),
            Some(initial.clone()),
            &FaerLinearSolver
        )
        .is_err()
    );
    assert!(
        DifferentiableProgram::compile(resolved, &[input], &output, None, &FaerLinearSolver)
            .is_err()
    );
}
