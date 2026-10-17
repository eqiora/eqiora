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
        relation bias_definition on body {{ bias=observation_scale*({bias}); }}
        observable summary:array<{output},8> = integral(
            [observation_scale*potential[0],potential[1],{reflected},1[m]*partial(potential,wrt=x)[0],
             {energy},bias[0],bias[1],1[m]*partial(bias,wrt=x)[0]], measure(body));
        observable composed:array<{output},2>={composed};
        observable norm:m^2={norm};
    }}"
    ).replace(
        "parameter source_scale: 1 / m ^ 2",
        &format!("parameter source_scale: 1 / m ^ 2, parameter weights:array<{scalar},2>={weights}, parameter observation_scale:1=1"),
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
        (2., -1.),
    ];
    let derivative = [
        (3., -2.),
        (-1., 4.),
        (-2., -8.),
        (0., 0.),
        (if complex { 80. } else { 18. }, 0.),
        (0., 0.),
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
            assert_eq!(value.component_count(), 8);
            for ((real, imaginary), (a, b)) in value.components().unwrap().zip(expected) {
                assert!(
                    (real - a).abs() < 1e-9
                        && (imaginary - if complex { b } else { 0. }).abs() < 1e-9,
                    "{method:?} complex={complex}: ({real},{imaginary}) != ({a},{b})"
                );
            }
        }
    }
    check_parameter_point(plan, result, complex);
}

fn check_parameter_point(plan: &ResolvedCommonPlan, result: &crate::CommonResult, complex: bool) {
    let spatial = plan.as_spatial().unwrap();
    let reduced = spatial
        .kernel()
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Observable(observable)
                if observable.reduction() == ObservableReduction::Value
                    && spatial
                        .kernel()
                        .observable_output_support(observable.id())
                        .unwrap()
                        .is_none() =>
            {
                Some(observable)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(reduced.len(), 2);
    let composed = reduced
        .iter()
        .find(|value| !value.value_type().shape().is_scalar())
        .unwrap()
        .id();
    let norm = reduced
        .iter()
        .find(|value| value.value_type().shape().is_scalar())
        .unwrap()
        .id();
    // Request a dependent output first; its shared integral is evaluated once.
    let summary = spatial
        .kernel()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Observable(observable)
                if matches!(
                    observable.reduction(),
                    ObservableReduction::SpatialIntegral { .. }
                ) =>
            {
                Some(observable.id())
            }
            _ => None,
        })
        .unwrap();
    let parameter_of = |id| {
        let Some(KernelNode::Observable(definition)) = spatial.kernel().node(id) else {
            panic!("exact Observable");
        };
        let parameters = definition
            .expression()
            .nodes()
            .iter()
            .filter_map(|node| match node {
                eqiora_schema::kernel::ExprNode::Symbol(
                    eqiora_schema::kernel::SymbolRef::Parameter(id),
                ) => Some(*id),
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(parameters.len(), 1);
        *parameters.iter().next().unwrap()
    };
    let selected = [
        parameter_of(composed.erase()),
        parameter_of(summary.erase()),
    ];
    let coordinates = crate::parameter_point::coordinates(spatial.kernel(), &selected).unwrap();
    let values: &[f64] = if complex {
        &[3., -1., 2., 4., 2.]
    } else {
        &[3., 2., 2.]
    };
    let point =
        crate::parameter_point::ParameterPoint::bind(spatial.kernel(), &selected, values).unwrap();
    let outputs = [norm, composed, summary];
    let rules = plan.observation_quadratures(&outputs, 2).unwrap();
    let fields = spatial
        .fields()
        .enumerate()
        .map(|(index, (id, _))| (id, result.field_block(index, 0).unwrap().1))
        .collect::<Vec<_>>();
    let original = result.to_bytes().unwrap();
    let evaluated = crate::common_result::observe::spatial_point_outputs(
        spatial,
        &point,
        fields.iter().copied(),
        &outputs,
        &rules,
        Some(&coordinates),
    )
    .unwrap();
    let partials = evaluated.partials.unwrap();
    let evaluated = evaluated.values;
    // Scale two changes the independent integrals to (4+i,-3/2+4i).
    // Weights (3-i,2+4i) give conjugated products (-1/2-27i/2,4-18i).
    let expected = if complex {
        vec![vec![(522.5, 0.)], vec![(-0.5, -13.5), (4., -18.)]]
    } else {
        vec![vec![(84.25, 0.)], vec![(-4.5, 0.), (8., 0.)]]
    };
    assert_eq!(evaluated.len(), 3);
    for (value, expected) in evaluated.iter().zip(expected) {
        assert_eq!(value.component_count(), expected.len());
        for ((a, b), (real, imaginary)) in value.components().unwrap().zip(expected) {
            assert!((a - real).abs() < 1e-9 && (b - imaginary).abs() < 1e-9);
        }
    }
    // The same point also binds prescribed Field expressions before reconstruction.
    for (index, (real, imaginary)) in [(5, (4., 3.)), (6, (5., 2.)), (7, (4., -2.))] {
        let (a, b) = evaluated[2].component(index).unwrap();
        assert!((a - real).abs() < 1e-9 && (b - if complex { imaginary } else { 0. }).abs() < 1e-9);
    }
    assert!(
        crate::common_result::observe::spatial_point_outputs(
            spatial,
            &point,
            std::iter::empty(),
            &outputs,
            &rules,
            None,
        )
        .is_err()
    );
    // At fixed Field coefficients, |w0*A|² + |w1*s*B|² has exact real partials.
    let expected: &[f64] = if complex {
        &[109.5, -36.5, 68., 136., 340.]
    } else {
        &[13.5, 64., 64.]
    };
    for (actual, expected) in partials[0][0].parameters.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-8,
            "norm Parameter partial {actual} != {expected}"
        );
    }
    // Prescribed value and coordinate-gradient chains use the same coefficient derivatives.
    let width = if complex { 2 } else { 1 };
    for (component, expected) in [
        (0, (2., 0.5)),
        (5, (2., 1.5)),
        (6, (2.5, 1.)),
        (7, (2., -1.)),
    ] {
        let row = component * width;
        let scale = coordinates.len() - 1;
        assert!((partials[2][row].parameters[scale] - expected.0).abs() < 1e-9);
        if complex {
            assert!((partials[2][row + 1].parameters[scale] - expected.1).abs() < 1e-9);
        }
    }
    // Selecting only one coordinate holds every other Parameter fixed at this same point.
    let restricted = crate::common_result::observe::spatial_point_outputs(
        spatial,
        &point,
        fields.iter().copied(),
        &outputs,
        &rules,
        Some(&coordinates[..1]),
    )
    .unwrap();
    assert_eq!(restricted.values, evaluated);
    assert!((restricted.partials.unwrap()[0][0].parameters[0] - expected[0]).abs() < 1e-8);
    // Contract the reverse Field covector against an independently specified uniform direction.
    let direction: &[f64] = if complex {
        &[3., -2., -1., 4.]
    } else {
        &[3., -1.]
    };
    let (field, coefficients) = fields[0];
    let product = |row: &crate::common_result::observe::Covector| {
        row.fields.get(&field.erase()).map_or(0., |values| {
            values
                .iter()
                .enumerate()
                .map(|(index, value)| value * direction[index % direction.len()])
                .sum::<f64>()
        })
    };
    assert!((product(&partials[0][0]) - if complex { 1150. } else { 219. }).abs() < 1e-8);
    let expected: &[f64] = if complex {
        &[1., -13., 28., -16.]
    } else {
        &[-3., 12.]
    };
    for (row, expected) in partials[1].iter().zip(expected) {
        assert!((product(row) - expected).abs() < 1e-8);
    }
    // A unit affine variation of channel zero has integrated gradient one for Q1,
    // while cell-constant reconstruction has identically zero gradient.
    let count = coefficients.len() / direction.len();
    let derivative = partials[2][3 * width].fields[&field.erase()]
        .iter()
        .enumerate()
        .filter(|(index, _)| index % direction.len() == 0)
        .map(|(index, value)| value * (index / direction.len()) as f64 / (count - 1) as f64)
        .sum::<f64>();
    assert!((derivative - if count == 5 { 1. } else { 0. }).abs() < 1e-9);
    assert_eq!(result.to_bytes().unwrap(), original);
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
            let rules = result
                .plan()
                .observation_quadratures(&[definition.id()], 2)
                .unwrap();
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
