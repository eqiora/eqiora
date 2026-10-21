//! Complete material inputs retain their existing mathematical value types.
use super::*;
use eqiora::ValueLiteral;

#[test]
fn value_only_properties_preserve_complex_and_shaped_inputs() {
    for (ty, input, component, values, expected) in [
        ("1", "1", "x", vec![(1., 0.)], vec![(3., 0.)]),
        (
            "array<1,2>",
            "[1,3]",
            "component(x,0)",
            vec![(1., 0.), (3., 0.)],
            vec![(3., 0.), (15., 0.)],
        ),
        (
            "complex<1>",
            "math.complex(1,2)",
            "x",
            vec![(1., 2.)],
            vec![(3., 1.)],
        ),
        (
            "array<complex<1>,2>",
            "[math.complex(1,2),math.complex(3,-4)]",
            "component(x,0)",
            vec![(1., 2.), (3., -4.)],
            vec![(3., 1.), (29., 3.)],
        ),
    ] {
        let value = if ty.contains("complex") {
            format!("{component}*math.conj({component})+math.complex(0,1)*{component}")
        } else {
            format!("{component}*{component}+2*{component}")
        };
        let source = format!(
            r#"
            property contract Response(input x:{ty}):{ty} {{ derivatives value_only; }}
            property release Measured:Response {{
                analytic {{ value={value}; source_unit:1=1; }}
                validity math.real({component})>=0; outside reject; branch single;
                citation org.example.manufactured; license spdx.CC0_1_0;
            }}
            component Consumer(parameter x:{ty},property response:Response) {{
                variable y:{ty}; relation law {{y=response(x=x);}}
                observable output:{ty}=y;
            }}
            model Main() {{parameter p:{ty}={input};
                instance sample:Consumer(x=p,response=Measured);}}
            "#
        );
        let document = ModelDocument::compile("typed-property-input.eqi", &source).unwrap();
        ordinary_run(&document, &expected);
        let encoded = ModelEnvelope::from_program(document.program())
            .unwrap()
            .canonical_json()
            .unwrap();
        let reopened = ModelEnvelope::from_json(&encoded, Default::default())
            .unwrap()
            .to_program()
            .unwrap();
        assert_eq!(document.program(), &reopened);
        for program in [document.program(), &reopened] {
            let release = program
                .nodes()
                .find_map(|node| match node {
                    KernelNode::Relation(value) => value.expression().properties().values().next(),
                    _ => None,
                })
                .unwrap();
            let definition = release.meaning().definition().unwrap();
            let input_type = definition.formals()[0].exact_type().unwrap().clone();
            let point = ValueLiteral::new(input_type.clone(), values.iter().copied()).unwrap();
            let actual =
                eqiora::ir::ComponentScalarization::evaluate_pure_operator(definition, &[&point])
                    .unwrap();
            assert_eq!(actual.value_type(), &input_type);
            // Complex: a²+b²-b+i*a. Real: x²+2*x.
            assert_eq!(actual.components().unwrap().collect::<Vec<_>>(), expected);
            assert!(
                release.partial("x").is_err(),
                "value-only grants no derivative"
            );
            let mut invalid = values.clone();
            invalid[0].0 = -1.;
            let outside = ValueLiteral::new(input_type, invalid).unwrap();
            assert!(eqiora::ir::ComponentScalarization::evaluate_pure_operator(
                definition,
                &[&outside],
            ).is_err());
        }
    }
}

fn ordinary_run(document: &ModelDocument, expected: &[(f64, f64)]) {
    use eqiora::solver::{LinearSolver, ReductionPolicy, SolverPlan};
    use eqiora_backend_faer::FaerLinearSolver;
    use eqiora_numerics::{
        CommonAlgebraicPlan, CommonLinearRequest, CommonResult, CommonSolvePolicy,
        ResolvedCommonPlan,
    };
    let backend = &FaerLinearSolver;
    let solver = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-13,
        1e-15,
        std::num::NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = CommonAlgebraicPlan::resolve(
        &model,
        CommonSolvePolicy::Linear(CommonLinearRequest::exact(solver, backend.provider()).unwrap()),
        None,
        &[],
        None,
        backend,
    )
    .unwrap();
    let initial = plan.initial_state(&[]).unwrap();
    let before = initial.to_bytes().unwrap();
    let result = plan.run_result(&initial, backend).unwrap();
    let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
    assert_eq!(
        CommonResult::from_bytes(&result.to_bytes().unwrap(), &resolved).unwrap(),
        result
    );
    assert_eq!(initial.to_bytes().unwrap(), before);
    let denied = plan
        .differentiate(
            &initial,
            &[document.aliases()["p"].downcast().unwrap()],
            None,
            &[document.aliases()["sample.output"]],
            backend,
        )
        .unwrap_err();
    assert!(
        denied.message().contains("value-only contract"),
        "{denied:?}"
    );
    let observed = result
        .observe(
            plan.model_artifact(),
            document.aliases()["sample.output"].downcast().unwrap(),
            &Default::default(),
            None,
        )
        .unwrap();
    let values = observed.value().components().unwrap().collect::<Vec<_>>();
    assert_eq!(values.len(), expected.len());
    for ((re, im), &(a, b)) in values.into_iter().zip(expected) {
        assert!((re - a).abs() < 1e-10 && (im - b).abs() < 1e-10);
    }
}
