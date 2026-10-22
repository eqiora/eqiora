//! Nonholomorphic residual and prescribed trace share one Parameter derivative.
use super::*;
use eqiora_core::ScalarDomain;
use eqiora_ir::{LinearizedOutput, LinearizedRelation, RelationTangent};
use eqiora_schema::kernel::KernelNode;
use std::collections::HashMap;

#[test]
fn conjugated_unknown_derivatives_retain_parameter_load_and_boundary_lifts() {
    let source = r#"
      public component Conjugated(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body), support right:boundary(parent=body),
        parameter p:complex<1>) {
        variable u:complex<1> on body in h1;
        relation balance on body {
          -div(math.complex(2,3)*grad(u))+
          (math.complex(3,-2)*u+p*math.conj(u)
          -(math.complex(3,-2)*p+p*math.conj(p))*(1+coordinate(0)/1[m]))/1[m^2]=0;
        }
        relation left_value on left {trace(u)=p;}
        relation right_value on right {trace(u)=2*p;}
      }
    "#;
    check_derivatives(
        source,
        &[
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ],
        false,
        ScalarDomain::Real,
    );
}

#[test]
fn conjugated_gradient_and_prescribed_flux_retain_natural_trace_derivatives() {
    // u=p*(1+x), q=(4+3i)*grad(u)+p*conj(grad(u))+p*x*e.
    // Thus -div(q)=-p and q(1)=(4+3i)*p+p*conj(p)+p.
    let source = r#"
      public component Conjugated(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body), support right:boundary(parent=body),
        parameter p:complex<1>) {
        parameter e:vector<1,1>=tensor_value(frame=body,components=[1]);
        variable g:vector<complex<1/m>,1> on body in h1;
        relation drift on body {g=p*coordinate(0)/1[m^2]*e;}
        variable u:complex<1> on body in smooth;
        relation balance on body {
          -div(math.complex(4,3)*grad(u)+p*math.conj(grad(u))+g)=-p/1[m^2];
        }
        relation left_value on left {trace(u)=p;}
        relation right_value on right {
          normal(math.complex(4,3)*grad(u)+p*math.conj(grad(u))+g)
            =(math.complex(4,3)*p+p*math.conj(p)+p)/1[m];
        }
      }
    "#;
    check_derivatives(
        source,
        &[
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ],
        true,
        ScalarDomain::Real,
    );
}

#[test]
fn scalar_diffusion_orientation_ignores_transport_and_prescribed_flux() {
    // All forms admit u=p*(1+x): u''=0, (u/m)'=p/m^2,
    // and (p*x/m^2)'=p/m^2. Reversing a whole equation preserves its solution.
    for equation in [
        "-div(grad(u)-e*u/1[m])=p/1[m^2]",
        "-div(grad(u))+div(e*u/1[m])=p/1[m^2]",
        "div(e*u/1[m])-div(grad(u))=p/1[m^2]",
        "div(grad(u))-div(e*u/1[m])=-p/1[m^2]",
        "-div(grad(u))+div(p*coordinate(0)*e/1[m^2])=p/1[m^2]",
    ] {
        let source = format!(
            r#"
          public component Conjugated(support body:volume(ambient_dimension=1),
            support left:boundary(parent=body), support right:boundary(parent=body),
            parameter p:complex<1>) {{
            parameter e:vector<1,1>=tensor_value(frame=body,components=[1]);
            variable u:complex<1> on body in smooth;
            relation balance on body {{{equation};}}
            relation left_value on left {{trace(u)=p;}}
            relation right_value on right {{trace(u)=2*p;}}
          }}
        "#
        );
        check_derivatives(
            &source,
            &[CommonSpatialPolicy::Q1],
            false,
            ScalarDomain::Complex,
        );
    }
}

fn check_derivatives(
    source: &str,
    methods: &[CommonSpatialPolicy],
    right_natural: bool,
    nonzero_state_domain: ScalarDomain,
) {
    let geometry = cartesian_interval();
    let body = geometry.entity_set("body").unwrap();
    let model = compile_typed_model(
        "conjugated-boundary.eqi",
        source,
        &geometry,
        "Conjugated",
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
        &[(
            "p",
            eqiora_core::ValueLiteral::new(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Complex,
                    DimExponents::DIMENSIONLESS,
                )
                .unwrap(),
                vec![(2., 3.)],
            )
            .unwrap(),
        )],
    );
    for &spatial in methods {
        let cells = spatial == CommonSpatialPolicy::CellCenteredTpfa;
        let locations = if cells {
            vec![0.125, 0.375, 0.625, 0.875]
        } else {
            vec![0., 0.25, 0.5, 0.75, 1.]
        };
        let free = locations
            .iter()
            .copied()
            .filter(|x| cells || (*x > 0. && (right_natural || *x < 1.)))
            .collect::<Vec<_>>();
        let resolved = ResolvedCommonPlan::resolve(
            &model,
            cartesian_box_resources(&geometry, &[4]),
            spatial,
            CommonSolvePolicy::Linear(exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-12,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
            )),
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
            None,
        )
        .unwrap();
        let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
        let plan = resolved.as_spatial().unwrap();
        let result = plan.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        let bytes = result.to_bytes().unwrap();
        assert_eq!(
            crate::CommonResult::from_bytes(&bytes, &resolved)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let parameter = plan
            .admission
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Parameter(value)
                    if value.value_type().scalar_domain() == eqiora_core::ScalarDomain::Complex =>
                {
                    Some(value.id())
                }
                _ => None,
            })
            .unwrap();
        let outputs = plan
            .fields()
            .map(|(field, _)| field.erase())
            .collect::<Vec<_>>();
        // The exact affine solution is u=p*(1+x), so both real input directions
        // have independent analytic tangents, including their nonzero traces.
        for point in [[2., 3.], [0., 0.], [-1., 4.]] {
            let accepted = plan
                .differentiate(
                    None,
                    &[parameter],
                    Some(&point),
                    &outputs,
                    &HashMap::new(),
                    &REFERENCE_LINEAR_SOLVER,
                )
                .unwrap();
            let linear = accepted.linearization();
            assert_eq!(
                linear.state_scalar_domain(),
                if point == [0., 0.] {
                    eqiora_core::ScalarDomain::Complex
                } else {
                    nonzero_state_domain
                }
            );
            let relation = linear.relation();
            assert_eq!(relation.unknown_dimension(), free.len() * 2);
            for (pair, x) in linear
                .output_values()
                .as_chunks::<2>()
                .0
                .iter()
                .zip(&locations)
            {
                for part in 0..2 {
                    assert!((pair[part] - point[part] * (1. + x)).abs() < 2e-10);
                }
            }
            for direction in [[1., 0.], [0., 1.], [3., -2.]] {
                let unknown = free
                    .iter()
                    .flat_map(|x| direction.map(|v| v * (1. + x)))
                    .collect::<Vec<_>>();
                let mut state_action = vec![0.; free.len() * 2];
                let mut parameter_action = vec![0.; free.len() * 2];
                relation
                    .jvp(RelationTangent::Unknown(&unknown), &mut state_action)
                    .unwrap();
                relation
                    .jvp(
                        RelationTangent::Parameter(&direction),
                        &mut parameter_action,
                    )
                    .unwrap();
                for (state, parameter) in state_action.into_iter().zip(parameter_action) {
                    assert!((state + parameter).abs() < 2e-10);
                }
                let mut output = vec![0.; locations.len() * 2];
                linear.jvp(&unknown, &direction, &mut output).unwrap();
                let expected = locations
                    .iter()
                    .flat_map(|x| direction.map(|v| v * (1. + x)));
                for (actual, expected) in output.into_iter().zip(expected) {
                    assert!((actual - expected).abs() < 2e-10);
                }
            }
            let replay = plan
                .validate_differentiated_values(
                    None,
                    &[parameter],
                    Some(&point),
                    &outputs,
                    &HashMap::new(),
                    relation.accepted_unknowns(),
                )
                .unwrap();
            assert_eq!(&replay, linear);
            let mut altered = relation.accepted_unknowns().to_vec();
            altered[1] += 0.01;
            assert!(
                plan.validate_differentiated_values(
                    None,
                    &[parameter],
                    Some(&point),
                    &outputs,
                    &HashMap::new(),
                    &altered
                )
                .is_err()
            );
        }
    }
}
