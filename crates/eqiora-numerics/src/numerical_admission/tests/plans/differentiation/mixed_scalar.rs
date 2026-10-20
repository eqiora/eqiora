//! Real Fields retain one coordinate inside a complex coupled problem.
use super::*;
use eqiora_ir::LinearizedOutput;
use num_complex::Complex64 as C;

#[test]
fn mixed_scalar_fields_share_primal_replay_and_observation_products() {
    let geometry = cartesian_interval();
    for nonlinear in [false, true] {
        let (law, forcing) = if nonlinear {
            ("u*(1+v)", "math.complex(p*(1+p),2*p*(1+p))")
        } else {
            ("u", "math.complex(p,2*p)")
        };
        let source = format!(
            "public component Mixed(
                support body:volume(ambient_dimension=1),
                support left:boundary(parent=body),support right:boundary(parent=body),
                parameter p:1) {{
                variable u:complex<1> on body in smooth;
                variable v:1 on body in smooth;
                relation wave on body {{-div(1[m^2]*grad(u))+{law}={forcing};}}
                relation real_part on body {{-div(1[m^2]*grad(v))+v-math.real(u)=0;}}
                relation left_u on left {{trace(u)=math.complex(p,2*p);}}
                relation right_u on right {{trace(u)=math.complex(p,2*p);}}
                relation left_v on left {{trace(v)=p;}}
                relation right_v on right {{trace(v)=p;}}
                observable combined:complex<m>=integral(u+math.complex(v,0),measure(body));
            }}"
        );
        let body = geometry.entity_set("body").unwrap();
        let build_model = |source: &str| {
            compile_model(
                "mixed-scalar-products.eqi",
                source,
                &geometry,
                "Mixed",
                &[
                    ("body", body, None),
                    (
                        "left",
                        geometry.entity_set("left").unwrap(),
                        Some(("body", body)),
                    ),
                    (
                        "right",
                        geometry.entity_set("right").unwrap(),
                        Some(("body", body)),
                    ),
                ],
                &[("p", DynQuantity::new(1., DimExponents::DIMENSIONLESS))],
            )
        };
        let model = build_model(&source);
        let linear = exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        );
        let solve = if nonlinear {
            CommonSolvePolicy::Newton {
                linear,
                nonlinear: NonlinearSolvePlan::new(0., 1e-11, NonZeroUsize::new(16).unwrap(), 16)
                    .unwrap(),
            }
        } else {
            CommonSolvePolicy::Linear(linear)
        };
        let resolved = replay_plan(
            ResolvedCommonPlan::resolve(
                &model,
                cartesian_box_resources(&geometry, &[2]),
                CommonSpatialPolicy::Q1,
                solve,
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
            )
            .unwrap(),
            &REFERENCE_LINEAR_SOLVER,
        );
        // A real test Field cannot discard the imaginary equation. Changing only
        // its residual domain must reach the component-type admission check.
        let invalid_model =
            build_model(&source.replace("v-math.real(u)=0;", "v-math.real(u)=math.complex(0,1);"));
        let error = ResolvedCommonPlan::resolve(
            &invalid_model,
            cartesian_box_resources(&geometry, &[2]),
            CommonSpatialPolicy::Q1,
            solve,
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap_err();
        assert!(
            error
                .message()
                .contains("residual and test Field have different component roles"),
            "{error}"
        );
        let plan = resolved.as_spatial().unwrap();
        let real = plan
            .fields()
            .find(|(_, ty)| ty.scalar_domain() == eqiora_core::ScalarDomain::Real)
            .unwrap()
            .0;
        let complex = plan
            .fields()
            .find(|(_, ty)| ty.scalar_domain() == eqiora_core::ScalarDomain::Complex)
            .unwrap()
            .0;
        let seed = plan
            .initial_state(
                None,
                plan.fields()
                    .map(|(id, ty)| {
                        CommonInitialField::new(
                            model.digest().unwrap(),
                            id,
                            Some(
                                CommonInitialValues::new(
                                    ty.shape().clone(),
                                    vec![
                                        if id == real {
                                            C::new(1., 0.)
                                        } else {
                                            C::new(1., 2.)
                                        };
                                        3
                                    ],
                                )
                                .unwrap(),
                            ),
                            None,
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap();
        let initial = nonlinear.then_some(&seed);
        let result = plan.run_result(initial, &REFERENCE_LINEAR_SOLVER).unwrap();
        assert_eq!(
            crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &resolved).unwrap(),
            result
        );
        let observable = plan
            .kernel()
            .nodes()
            .find_map(|node| match node {
                eqiora_schema::kernel::KernelNode::Observable(value) => Some(value),
                _ => None,
            })
            .unwrap();
        let parameter = plan
            .kernel()
            .nodes()
            .find_map(|node| match node {
                eqiora_schema::kernel::KernelNode::Parameter(value) => Some(value.id()),
                _ => None,
            })
            .unwrap();
        let rules = std::collections::HashMap::from([(
            observable.reduction().domain().unwrap(),
            eqiora_meshing::QuadratureRule::gauss_legendre(2).unwrap(),
        )]);
        let outputs = [real.erase(), complex.erase(), observable.id().erase()];
        // The same independent constant solution solves both systems:
        // v=p, u=p+2ip. The unit-interval integral of u+v is 2p+2ip.
        let derivative = [vec![1.; 3], [1., 2.].repeat(3), vec![2., 2.]].concat();
        let free_derivative = plan
            .fields()
            .flat_map(|(id, _)| if id == real { vec![1.] } else { vec![1., 2.] })
            .collect::<Vec<_>>();
        for p in [0.5, 1., 2.] {
            let point = plan
                .differentiate(
                    initial,
                    &[parameter],
                    Some(&[p]),
                    &outputs,
                    &rules,
                    &REFERENCE_LINEAR_SOLVER,
                )
                .unwrap();
            let accepted = point.linearization();
            let expected = derivative.iter().map(|value| p * value).collect::<Vec<_>>();
            for (actual, expected) in accepted.output_values().iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
            }
            assert_eq!(accepted.output_values().len(), expected.len());
            assert_eq!(accepted.relation().accepted_unknowns().len(), 3);
            let mut tangent = vec![0.; derivative.len()];
            accepted.jvp(&free_derivative, &[1.], &mut tangent).unwrap();
            for (actual, expected) in tangent.iter().zip(&derivative) {
                assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
            }
        }
    }
}
