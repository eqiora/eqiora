//! Independently integrated affine channels and real-linear complex directions.
use super::*;
use eqiora_schema::kernel::{KernelNode, ObservableReduction};

pub(super) fn source(source: &str, complex: bool) -> String {
    let (prefix, _) = source.rsplit_once('}').unwrap();
    let scalar = if complex { "complex<1>" } else { "1" };
    let output = if complex { "complex<m>" } else { "m" };
    let bias = if complex {
        "[math.complex(1+2*x/1[m],2-x/1[m]),math.complex(3-x/1[m],-1+4*x/1[m])]"
    } else {
        "[1+2*x/1[m],3-x/1[m]]"
    };
    let reflected = if complex {
        "2*math.conj(potential[1])"
    } else {
        "2*potential[1]"
    };
    let energy = if complex {
        "math.abs2(potential[0])+2*math.abs2(potential[1])"
    } else {
        "potential[0]^2+2*potential[1]^2"
    };
    let weights = if complex {
        "[math.complex(2,3),math.complex(-1,2)]"
    } else {
        "[2,-3]"
    };
    let composed = if complex {
        "math.conj([weights[0]*summary[1],weights[1]*summary[0]])"
    } else {
        "[weights[0]*summary[1],weights[1]*summary[0]]"
    };
    let norm = if complex {
        "math.abs2(composed[0])+math.abs2(composed[1])"
    } else {
        "composed[0]^2+composed[1]^2"
    };
    format!(
        "{prefix}
        variable bias:array<{scalar},2> on body in smooth;
        relation bias_definition on body {{ bias={bias}; }}
        observable summary:array<{output},7> = integral(
            [potential[0],potential[1],{reflected},1[m]*partial(potential,wrt=x)[0],
             {energy},bias[0],bias[1]], measure(body));
        parameter weights:array<{scalar},2>={weights};
        observable composed:array<{output},2>={composed};
        observable norm:m^2={norm};
    }}"
    )
}

pub(super) fn check(
    model: &ModelEnvelope,
    plan: &ResolvedCommonPlan,
    result: &crate::CommonResult,
    method: CommonSpatialPolicy,
    complex: bool,
) {
    let definition = plan
        .as_spatial()
        .unwrap()
        .kernel()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Observable(definition)
                if matches!(
                    definition.reduction(),
                    ObservableReduction::SpatialIntegral { .. }
                ) =>
            {
                Some(definition)
            }
            _ => None,
        })
        .unwrap();
    let domain = definition.reduction().domain().unwrap();
    let rules = std::collections::HashMap::from([(
        domain,
        eqiora_meshing::QuadratureRule::gauss_legendre(2).unwrap(),
    )]);
    let (field, ty) = plan.as_spatial().unwrap().fields().next().unwrap();
    let nodal = method == CommonSpatialPolicy::Q1;
    let energy = if complex { 42. } else { 9. }
        - if nodal {
            0.
        } else if complex {
            15. / 192.
        } else {
            6. / 192.
        };
    let expected = [
        (2., 0.5),
        (-1.5, 4.),
        (-3., -8.),
        if nodal { (2., -1.) } else { (0., 0.) },
        (energy, 0.),
        (2., 1.5),
        (2.5, 1.),
    ];
    let derivative = [
        (3., -2.),
        (-1., 4.),
        (-2., -8.),
        (0., 0.),
        (if complex { 80. } else { 18. }, 0.),
        (0., 0.),
        (0., 0.),
    ];
    let replay = crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), plan).unwrap();
    for result in [result, &replay] {
        let direction = if complex {
            vec![3., -2., -1., 4.]
        } else {
            vec![3., -1.]
        };
        let coefficients = (0..if nodal { 5 } else { 4 })
            .flat_map(|_| direction.iter().copied())
            .map(|value| DynQuantity::new(value, ty.dimension()))
            .collect();
        let direction = result
            .observable_state_tangent([(field, coefficients)])
            .unwrap();
        let primal = result
            .observe(model, definition.id(), &rules, None)
            .unwrap();
        let tangent = result
            .observe_state_jvp(model, definition.id(), &rules, &direction)
            .unwrap();
        let mut composed_count = 0;
        for definition in
            plan.as_spatial()
                .unwrap()
                .kernel()
                .nodes()
                .filter_map(|node| match node {
                    KernelNode::Observable(definition)
                        if definition.reduction() == ObservableReduction::Value
                            && plan
                                .as_spatial()
                                .unwrap()
                                .kernel()
                                .observable_output_support(definition.id())
                                .unwrap()
                                .is_none() =>
                    {
                        Some(definition)
                    }
                    _ => None,
                })
        {
            composed_count += 1;
            let (value, jvp) = if definition.value_type().shape().is_scalar() {
                if complex {
                    (vec![(258.5, 0.)], vec![(505., 0.)])
                } else {
                    (vec![(45., 0.)], vec![(120., 0.)])
                }
            } else if complex {
                (
                    vec![(-15., -3.5), (-3., -3.5)],
                    vec![(-14., -5.), (1., -8.)],
                )
            } else {
                (vec![(-3., 0.), (-6., 0.)], vec![(-2., 0.), (-9., 0.)])
            };
            let primal = result
                .observe(model, definition.id(), &rules, None)
                .unwrap();
            let tangent = result
                .observe_state_jvp(model, definition.id(), &rules, &direction)
                .unwrap();
            for (observation, expected) in [(primal, value), (tangent, jvp)] {
                assert_eq!(observation.value().value_type(), definition.value_type());
                assert_eq!(observation.value().component_count(), expected.len());
                for ((a, b), (real, imaginary)) in
                    observation.value().components().unwrap().zip(expected)
                {
                    assert!(
                        (a - real).abs() < 1e-8 && (b - imaginary).abs() < 1e-8,
                        "{method:?} complex={complex}: ({a},{b}) != ({real},{imaginary})"
                    );
                }
            }
        }
        assert_eq!(composed_count, 2);
        for (value, expected) in [(primal.value(), expected), (tangent.value(), derivative)] {
            assert_eq!(value.value_type(), definition.value_type());
            assert_eq!(value.component_count(), 7);
            for ((real, imaginary), (a, b)) in value.components().unwrap().zip(expected) {
                assert!(
                    (real - a).abs() < 1e-9
                        && (imaginary - if complex { b } else { 0. }).abs() < 1e-9,
                    "{method:?} complex={complex}: ({real},{imaginary}) != ({a},{b})"
                );
            }
        }
    }
}

pub(super) fn check_vector(
    model: &ModelEnvelope,
    plan: &ResolvedCommonPlan,
    result: &crate::CommonResult,
    complex: bool,
) {
    use eqiora_schema::kernel::ExprNode;
    let replay = crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), plan).unwrap();
    for result in [result, &replay] {
        for definition in
            plan.as_spatial()
                .unwrap()
                .kernel()
                .nodes()
                .filter_map(|node| match node {
                    KernelNode::Observable(definition)
                        if matches!(
                            definition.reduction(),
                            ObservableReduction::SpatialIntegral { .. }
                        ) =>
                    {
                        Some(definition)
                    }
                    _ => None,
                })
        {
            let expression = definition.expression();
            let expected = match expression.node(expression.roots()[0]).unwrap() {
                // On [0,2] x [0,3], integrating each affine component is area
                // six times its value at the centroid (1,3/2).
                ExprNode::Symbol(_) => vec![(27., 24.), (-33., 15.)],
                ExprNode::Gradient(_) => vec![(12., -6.), (6., 12.), (6., 18.), (-18., -6.)],
                // Right flux (31+53i,-48+71i)/m, times edge length three.
                ExprNode::NormalComponent { .. } if complex => vec![(93., 159.), (-144., 213.)],
                ExprNode::NormalComponent { .. } => vec![(51., 0.), (27., 0.)],
                _ => panic!("independently specified integrand"),
            };
            let rules = result.observation_quadratures(definition.id(), 2).unwrap();
            let value = result
                .observe(model, definition.id(), &rules, None)
                .unwrap();
            assert_eq!(value.value().value_type(), definition.value_type());
            assert_eq!(
                value.value().value_type().frame(),
                eqiora_core::ValueFrame::SpatialCartesian
            );
            assert_eq!(value.value().component_count(), expected.len());
            for ((a, b), (real, imaginary)) in value.value().components().unwrap().zip(expected) {
                assert!(
                    (a - real).abs() < 1e-9
                        && (b - if complex { imaginary } else { 0. }).abs() < 1e-9,
                    "complex={complex}: ({a},{b}) != ({real},{imaginary})"
                );
            }
        }
    }
}
