//! The ordinary Plan boundary must not silently reinterpret a process as an ODE.
use eqiora_api::ModelDocument;
use eqiora_artifact::ModelEnvelope;
use eqiora_core::TimeStateCoordinate;
use eqiora_numerics::{CommonOdePlan, CommonOdePolicy};
use eqiora_time::{ImplicitMidpointTimeBackend, TimeMethod};

#[test]
fn deterministic_plan_rejects_stochastic_models_before_selecting_a_flow() {
    for body in [
        "state x: 1; initial {x=1;} noise w: wiener(covariance=1); law evolve for x {calculus ito; drift -1[1/s]*x; diffusion w=1[1/s^(1/2)];}",
        "state x: integer; initial {x=0;} noise n: poisson(intensity=2[1/s]); law evolve for x {jump n=1;}",
    ] {
        let model =
            ModelDocument::compile("stochastic.eqi", &format!("model M() {{{body}}}")).unwrap();
        let bytes = model.canonical_json().unwrap();
        let replay = ModelDocument::replay(&bytes).unwrap();
        let field = model.aliases()["x"].downcast().unwrap();
        for document in [&model, &replay] {
            let envelope = ModelEnvelope::from_program(document.program()).unwrap();
            let policy = CommonOdePolicy::new(
                TimeMethod::ImplicitMidpoint,
                0.01,
                1e-6,
                vec![(TimeStateCoordinate::new(field, 0, 0, false), 1e-8)],
            )
            .unwrap();
            let error = CommonOdePlan::resolve(
                &envelope,
                document.program(),
                policy,
                ImplicitMidpointTimeBackend::CAPABILITIES,
                Some(eqiora_numerics::CommonSolvePolicy::Linear(
                    eqiora_numerics::CommonLinearRequest::exact(
                        eqiora_solver::SolverPlan::new(
                            eqiora_solver::LinearSolver::BiConjugateGradientStabilized,
                            1e-12,
                            1e-14,
                            std::num::NonZeroUsize::new(100).unwrap(),
                        )
                        .unwrap(),
                        eqiora_solver::REFERENCE_SOLVER_PROVIDER,
                    )
                    .unwrap(),
                )),
                &eqiora_solver::REFERENCE_LINEAR_SOLVER,
            )
            .unwrap_err();
            assert!(
                error.message().contains("explicit stochastic execution"),
                "{error:?}"
            );
        }
        assert_eq!(model.canonical_json().unwrap(), bytes);
    }
}
