use super::*;

#[test]
fn spatial_integrals_reconstruct_only_requested_fields() {
    let geometry = geometry(true);
    let body = geometry.entity_set("body").unwrap();
    let mut supports = vec![("body", body, None)];
    supports.extend(["left", "right", "bottom", "top"].map(|name| {
        (
            name,
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    // On the unit square, u=(1,0), grad(p)=grad(f)=(1,0), and
    // zero right traction fixes p=x-1. Hence integral(p)=-1/2,
    // integral_left(p)=-1, and integral(1)=1. The unrequested MINI
    // velocity must not invalidate these scalar P1/constant observations.
    let source = FLOW.replace(
        "f=0[kg/(m*s^2)]",
        "f=1[kg/(m^2*s^2)]*coordinate(0)-1[kg/(m*s^2)]",
    );
    for (declaration, expected) in [
        (
            "observable output:kg*m/s^2=integral(p,measure(body));",
            Some((-0.5, [1, 1, -2, 0, 0, 0, 0])),
        ),
        (
            "observable output:kg/s^2=integral(trace(p),measure(left));",
            Some((-1.0, [1, 0, -2, 0, 0, 0, 0])),
        ),
        (
            "observable output:m^2=integral(1,measure(body));",
            Some((1.0, [0, 2, 0, 0, 0, 0, 0])),
        ),
        (
            "observable output:kg*m/s^2=integral(f,measure(body));",
            Some((-0.5, [1, 1, -2, 0, 0, 0, 0])),
        ),
        (
            "observable output:m^3/s=integral(component(u,indices=(0,)),measure(body));",
            None,
        ),
    ] {
        let source = source.replace("\n}\n", &format!("\n{declaration}\n}}\n"));
        let model = compile_model(
            "requested-fields.eqi",
            &source,
            &geometry,
            "Throughflow",
            &supports,
            &[],
        );
        for permuted in [false, true] {
            let plan = ResolvedCommonPlan::resolve(
                &model,
                open_resources(&geometry, permuted),
                CommonSpatialPolicy::MiniP1,
                CommonSolvePolicy::Linear(exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-11,
                    1e-13,
                    NonZeroUsize::new(1000).unwrap(),
                )),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
            )
            .unwrap();
            let spatial = plan.as_spatial().unwrap();
            let observable = spatial
                .kernel()
                .nodes()
                .find_map(|node| match node {
                    eqiora_schema::kernel::KernelNode::Observable(definition) => {
                        Some(definition.id())
                    }
                    _ => None,
                })
                .unwrap();
            let result = spatial.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
            let bytes = result.to_bytes().unwrap();
            let replay = crate::CommonResult::from_bytes(&bytes, &plan).unwrap();
            for result in [&result, &replay] {
                let rules = result
                    .plan()
                    .observation_quadratures(&[observable], 2)
                    .unwrap();
                let observation = result.observe(&model, observable, &rules, None);
                if let Some((expected, exponents)) = expected {
                    let observation = observation.unwrap();
                    let value = observation.value().real_scalar_value().unwrap();
                    assert!(
                        (value.value() - expected).abs() < 1e-9,
                        "{declaration}: {value:?}"
                    );
                    assert_eq!(value.dim(), DimExponents::from_integers(exponents).unwrap());
                    assert_eq!(observation.result_identity(), result.identity());
                } else {
                    // A requested MINI velocity still requires its own admitted
                    // reconstruction; ignoring unrelated fields must not widen it.
                    assert!(
                        observation
                            .unwrap_err()
                            .message()
                            .contains("linear nodal or cell-constant Field space")
                    );
                }
                assert_eq!(result.to_bytes().unwrap(), bytes);
            }
        }
    }
}
