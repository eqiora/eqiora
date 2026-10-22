//! An affine manufactured solution exercises a state-dependent full tensor flux.
use super::*;
use eqiora_ir::{LinearizedOutput, RelationCotangent};

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }
}

#[test]
fn tensor_material_laws_retain_nonlinear_state_and_parameter_products() {
    let geometry = CanonicalGeometryV1::decode_cartesian_box_v1_canonical(
        br#"{"schema":"eqiora.cartesian-box-envelope/v1","encoding":"eqiora.canonical-json/v1","length_unit":"metre","bounds":[[0.0,1.0],[0.0,1.0]],"entity_sets":[{"name":"bottom","dimension":1,"members":[2]},{"name":"left","dimension":1,"members":[0]},{"name":"right","dimension":1,"members":[1]},{"name":"top","dimension":1,"members":[3]},{"name":"body","dimension":2,"members":[0]}]}"#,
        eqiora_geometry::CanonicalGeometryLimits::default(),
    )
    .unwrap();
    for (property, complex) in [(false, false), (true, false), (false, true), (true, true)] {
        let supports = "support body:volume(ambient_dimension=2),
            support left:boundary(parent=body),support right:boundary(parent=body),
            support bottom:boundary(parent=body),support top:boundary(parent=body)";
        let law = if complex {
            "math.complex(1+T,1)*delta(0,1)+p"
        } else {
            "(1+T)*delta(0,1)+p"
        };
        let scalar = if complex { "complex<1>" } else { "1" };
        let branch = if complex { "principal" } else { "single" };
        let declarations = if property {
            format!(
                "property contract Conductivity(input T:1,input p:1):tensor<{scalar},2,2> {{
                    derivatives first_partials;branch {branch};
                }}
                property release Measured:Conductivity {{
                    analytic {{value={law};source_unit:1=1;}}
                    validity T>=0 and T<=20 and p>=0 and p<=3;outside reject;branch {branch};
                    citation org.example.manufactured;license spdx.CC0_1_0;
                }}"
            )
        } else {
            format!("operator k(input T:1,input p:1):tensor<{scalar},2,2>={law};")
        };
        let argument = if property {
            ",property k:Conductivity"
        } else {
            ""
        };
        let binding = if property { ",k=Measured" } else { "" };
        let real_exact = "p*(1+coordinate(0)/1[m]+2*coordinate(1)/1[m])";
        let exact = if complex {
            format!("math.complex({real_exact},p*(2-coordinate(0)/1[m]+coordinate(1)/1[m]))")
        } else {
            real_exact.to_owned()
        };
        let temperature = if complex { "math.real(u)" } else { "u" };
        let flux = format!("contract(k(T={temperature},p=p),grad(u),axes=((1,0),))");
        let forcing = if complex {
            "-math.complex(5,1)*p*p"
        } else {
            "-5*p*p"
        };
        let (right, top) = if complex {
            (
                format!("math.complex(1+{real_exact},1)*p*math.complex(1,-1)+3*p*p"),
                format!("math.complex(1+{real_exact},1)*p*math.complex(2,1)+3*p*p"),
            )
        } else {
            (
                format!("(1+{real_exact})*p+3*p*p"),
                format!("2*(1+{real_exact})*p+3*p*p"),
            )
        };
        let source = format!(
            "{declarations}
            component Equation({supports},parameter p:1{argument}) {{
                variable u:{scalar} on body in smooth;
                relation balance on body {{-div({flux})={forcing}/1[m^2];}}
                relation lower_x on left {{trace(u)={exact};}}
                relation lower_y on bottom {{trace(u)={exact};}}
                relation upper_x on right {{normal({flux})=({right})/1[m];}}
                relation upper_y on top {{normal({flux})=({top})/1[m];}}
            }}
            public component Material({supports},parameter p:1) {{
                instance physics:Equation(body=body,left=left,right=right,
                    bottom=bottom,top=top,p=p{binding});
            }}"
        );
        let body = geometry.entity_set("body").unwrap();
        let mut bindings = vec![("body", body, None)];
        bindings.extend(["left", "right", "bottom", "top"].map(|name| {
            (
                name,
                geometry.entity_set(name).unwrap(),
                Some(("body", body)),
            )
        }));
        let model = compile_model(
            "tensor-material.eqi",
            &source,
            &geometry,
            "Material",
            &bindings,
            &[("p", DynQuantity::new(1., DimExponents::DIMENSIONLESS))],
        );
        let resources = cartesian_box_resources(&geometry, &[2, 2]);
        let NativeMeshResources::Cartesian { mesh, .. } = &resources.resources else {
            unreachable!()
        };
        let mesh = mesh.mesh().clone();
        let resolved = ResolvedCommonPlan::resolve(
            &model,
            resources,
            CommonSpatialPolicy::Q1,
            CommonSolvePolicy::Newton {
                linear: exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-13,
                    1e-14,
                    NonZeroUsize::new(100).unwrap(),
                ),
                nonlinear: NonlinearSolvePlan::new(0., 1e-11, NonZeroUsize::new(16).unwrap(), 16)
                    .unwrap(),
            },
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
            None,
        )
        .unwrap();
        let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
        let plan = resolved.as_spatial().unwrap();
        let field = plan.fields().next().unwrap().0;
        let points = plan
            .field_coefficient_entities(field)
            .unwrap()
            .into_iter()
            .map(|entity| mesh.vertex_coordinates(entity).unwrap().to_vec())
            .collect::<Vec<_>>();
        let width = if complex { 2 } else { 1 };
        let profile_at = |x: &[f64]| [1. + x[0] + 2. * x[1], 2. - x[0] + x[1]];
        let profile = points
            .iter()
            .flat_map(|x| profile_at(x).into_iter().take(width))
            .collect::<Vec<_>>();
        let free_profile = points
            .iter()
            .filter(|x| x[0] > 0. && x[1] > 0.)
            .flat_map(|x| profile_at(x).into_iter().take(width))
            .collect::<Vec<_>>();
        let initial = plan
            .scalar_state(
                None,
                points
                    .iter()
                    .flat_map(|x| {
                        let value = profile_at(x);
                        [
                            value[0] - 0.25 * x[0] * x[1],
                            value[1] - 0.125 * x[0] * x[1],
                        ]
                        .into_iter()
                        .take(width)
                    })
                    .collect(),
            )
            .unwrap();
        let initial = CommonState::from_bytes(&initial.to_bytes().unwrap(), &resolved).unwrap();
        let result = plan
            .run_result(Some(&initial), &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        close(result.field_block(0, 0).unwrap().1, &profile);
        assert_eq!(
            crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &resolved).unwrap(),
            result
        );
        let mut corrupted = profile.clone();
        let interior = points.iter().position(|x| x[0] > 0. && x[1] > 0.).unwrap();
        corrupted[interior * width + width - 1] += 0.01;
        assert!(
            crate::CommonResult::from_bytes(
                &super::super::result_replay::reseal_values(&result, 0, &corrupted),
                &resolved,
            )
            .is_err()
        );
        let before = initial.to_bytes().unwrap();
        let parameter = plan
            .kernel()
            .nodes()
            .find_map(|node| match node {
                eqiora_schema::kernel::KernelNode::Parameter(value) => Some(value.id()),
                _ => None,
            })
            .unwrap();
        let outputs = [field.erase()];
        let quadratures = std::collections::HashMap::new();
        for p in [0.5, 1., 2.] {
            // Real: u=p*phi, K=(1+u)I+pJ, div(K grad u)=5p².
            // Complex: u=p*(phi+i*psi), K=(1+Re(u)+i)I+pJ,
            // phi=1+x+2y, psi=2-x+y, J_ij=1, div(K grad u)=p²(5+i).
            // Q1 represents both affine profiles exactly. du/dp=phi+i*psi,
            // including material-state dependence and nonzero essential/natural data.
            let candidate = free_profile
                .iter()
                .map(|value| p * value)
                .collect::<Vec<_>>();
            let accepted = plan
                .validate_differentiated_values(
                    Some(&initial),
                    &[parameter],
                    Some(&[p]),
                    &outputs,
                    &quadratures,
                    &candidate,
                )
                .unwrap();
            close(
                accepted.output_values(),
                &profile.iter().map(|value| p * value).collect::<Vec<_>>(),
            );
            let relation = accepted.relation();
            let mut residual = vec![0.; free_profile.len()];
            relation.primal(&mut residual).unwrap();
            close(&residual, &vec![0.; residual.len()]);
            let mut state_action = residual.clone();
            relation
                .jvp(RelationTangent::Unknown(&free_profile), &mut state_action)
                .unwrap();
            relation
                .jvp(RelationTangent::Parameter(&[1.]), &mut residual)
                .unwrap();
            close(
                &residual,
                &state_action.iter().map(|value| -value).collect::<Vec<_>>(),
            );
            let weights = (1..=residual.len())
                .map(|i| i as f64 / 8.)
                .collect::<Vec<_>>();
            let mut reverse = [0.];
            relation
                .vjp(&weights, RelationCotangent::Parameter(&mut reverse))
                .unwrap();
            close(
                &reverse,
                &[-weights
                    .iter()
                    .zip(&state_action)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()],
            );
            let mut tangent = vec![0.; profile.len()];
            accepted.jvp(&free_profile, &[1.], &mut tangent).unwrap();
            close(&tangent, &profile);
            let live = plan
                .differentiate(
                    Some(&initial),
                    &[parameter],
                    Some(&[p]),
                    &outputs,
                    &quadratures,
                    &REFERENCE_LINEAR_SOLVER,
                )
                .unwrap();
            assert!(live.nonlinear_iterations().unwrap() > 0);
            close(
                live.linearization().output_values(),
                accepted.output_values(),
            );
            let mut wrong = candidate;
            wrong[0] += 0.01;
            assert!(
                plan.validate_differentiated_values(
                    Some(&initial),
                    &[parameter],
                    Some(&[p]),
                    &outputs,
                    &quadratures,
                    &wrong,
                )
                .is_err()
            );
        }
        if property {
            assert!(
                plan.differentiate(
                    Some(&initial),
                    &[parameter],
                    Some(&[4.]),
                    &outputs,
                    &quadratures,
                    &REFERENCE_LINEAR_SOLVER,
                )
                .is_err()
            );
        }
        assert_eq!(initial.to_bytes().unwrap(), before);
    }
}
