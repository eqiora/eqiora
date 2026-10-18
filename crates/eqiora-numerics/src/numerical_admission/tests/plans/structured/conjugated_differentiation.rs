//! Nonholomorphic residual and prescribed trace share one Parameter derivative.
use super::*;
use eqiora_ir::{LinearizedOutput, LinearizedRelation, RelationTangent};
use eqiora_schema::kernel::KernelNode;
use std::collections::HashMap;

#[test]
fn conjugated_unknown_derivatives_retain_parameter_load_and_boundary_lifts() {
    let geometry = cartesian_interval();
    let body = geometry.entity_set("body").unwrap();
    let source = r#"
      public component Conjugated(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body), support right:boundary(parent=body),
        parameter p:complex<1>) {
        variable u:complex<1> on body in h1;
        relation balance on body {
          -div(math.complex(2,3)*grad(u))+
          (math.complex(3,-2)*u+math.complex(1,2)*math.conj(u)
          -(math.complex(3,-2)*p+math.complex(1,2)*math.conj(p))*(1+coordinate(0)/1[m]))/1[m^2]=0;
        }
        relation left_value on left {trace(u)=p;}
        relation right_value on right {trace(u)=2*p;}
      }
    "#;
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
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[4]),
        CommonSpatialPolicy::Q1,
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
    )
    .unwrap();
    let plan = resolved.as_spatial().unwrap();
    let parameter = plan
        .admission
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Parameter(value) => Some(value.id()),
            _ => None,
        })
        .unwrap();
    let outputs = plan
        .fields()
        .map(|(field, _)| field.erase())
        .collect::<Vec<_>>();
    // The exact affine solution is u=p*(1+x), so both real input directions
    // have independent analytic tangents, including their nonzero traces.
    for point in [[2., 3.], [-1., 4.]] {
        let accepted = plan
            .differentiate(
                &[parameter],
                Some(&point),
                &outputs,
                &HashMap::new(),
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
        let linear = accepted.linearization();
        let relation = linear.relation();
        assert_eq!(relation.unknown_dimension(), 6);
        for (node, pair) in linear.output_values().as_chunks::<2>().0.iter().enumerate() {
            for part in 0..2 {
                assert!((pair[part] - point[part] * (1. + node as f64 / 4.)).abs() < 2e-10);
            }
        }
        for direction in [[1., 0.], [0., 1.], [3., -2.]] {
            let unknown = (1..4)
                .flat_map(|node| direction.map(|v| v * (1. + node as f64 / 4.)))
                .collect::<Vec<_>>();
            let mut state_action = vec![0.; 6];
            let mut parameter_action = vec![0.; 6];
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
            let mut output = vec![0.; 10];
            linear.jvp(&unknown, &direction, &mut output).unwrap();
            let expected = (0..5).flat_map(|node| direction.map(|v| v * (1. + node as f64 / 4.)));
            for (actual, expected) in output.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 2e-10);
            }
        }
        let replay = plan
            .validate_differentiated_values(
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
