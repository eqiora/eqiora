//! Source formal scopes execute through ordinary typed finite Plans and replay.
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::solver::{LinearSolver, ReductionPolicy, SolverPlan};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonLinearRequest, CommonResult, CommonSolvePolicy, ResolvedCommonPlan,
};

#[test]
fn source_partials_keep_real_complex_and_shaped_held_inputs() {
    for (ty, component, input, first, second) in [
        ("1", "z", "2", vec![(12., 0.)], vec![(4., 0.)]),
        (
            "complex<1>",
            "z",
            "math.complex(2,5)",
            vec![(12., -30.)],
            vec![(4., -10.)],
        ),
        (
            "array<1,2>",
            "component(z,0)",
            "[2,7]",
            vec![(12., 0.), (42., 0.)],
            vec![(4., 0.), (14., 0.)],
        ),
        (
            "array<complex<1>,2>",
            "component(z,0)",
            "[math.complex(2,5),math.complex(7,-4)]",
            vec![(12., -30.), (42., 24.)],
            vec![(4., -10.), (14., 8.)],
        ),
    ] {
        let source = format!(
            r#"
            operator slope(input x:1,input z:{ty}):{ty} =
                partial(x*x*math.conj({component}),wrt=x,holding=(z));
            operator curvature(input x:1,input z:{ty}):{ty} =
                partial(partial(x*x*math.conj({component}),wrt=x,holding=(z)),wrt=x,holding=(z));
            model M() {{ parameter x:1=3; parameter z:{ty}={input};
                variable first:{ty}; variable second:{ty};
                relation r {{ first=slope(x=x,z=z); second=curvature(x=x,z=z); }}
                observable first_value:{ty}=first;
                observable second_value:{ty}=second;
            }}
        "#
        );
        let document = ModelDocument::compile("typed-partial.eqi", &source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let backend = &FaerLinearSolver;
        let solver = SolverPlan::new(
            LinearSolver::SparseLu,
            1e-13,
            1e-15,
            std::num::NonZeroUsize::new(8).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast);
        let plan = CommonAlgebraicPlan::resolve(
            &model,
            CommonSolvePolicy::Linear(
                CommonLinearRequest::exact(solver, backend.provider()).unwrap(),
            ),
            None,
            &[],
            None,
            backend,
        )
        .unwrap();
        let state = plan.initial_state(&[]).unwrap();
        let before = state.to_bytes().unwrap();
        let result = plan.run_result(&state, backend).unwrap();
        let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
        let replay = CommonResult::from_bytes(&result.to_bytes().unwrap(), &resolved).unwrap();
        assert_eq!(result, replay);
        assert_eq!(state.to_bytes().unwrap(), before);
        // At x=3, d_x[x²*conj(z)]=6*conj(z), d_xx=2*conj(z).
        // The identity solve and these integer operations are exact in binary64.
        for (name, expected) in [("first_value", first), ("second_value", second)] {
            let observed = replay
                .observe(
                    plan.model_artifact(),
                    document.aliases()[name].downcast().unwrap(),
                    &Default::default(),
                    None,
                )
                .unwrap();
            assert_eq!(
                observed.value().components().unwrap().collect::<Vec<_>>(),
                expected
            );
        }
    }
}
