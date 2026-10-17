//! Point queries preserve the represented Field, including P0 cell averages.
use super::*;
use eqiora_schema::kernel::KernelNode;

pub(super) fn channel_source(source: &str, complex: bool) -> String {
    let source = source.replace("on body in h1;", "on body in smooth;");
    let (prefix, _) = source.rsplit_once('}').unwrap();
    let scalar = if complex { "complex<1>" } else { "1" };
    let rate = if complex { "complex<1/m>" } else { "1/m" };
    let projected = if complex {
        "math.conj(potential)"
    } else {
        "potential"
    };
    format!(
        "{prefix}
        coordinate x:m on body from body[0];
        observable profile:array<{scalar},2> on body=potential;
        observable reflected:array<{scalar},2> on body=2*{projected};
        observable slope:array<{rate},2> on body=partial(potential,wrt=x);
    }}"
    )
}

pub(super) fn check_vector(
    model: &ModelEnvelope,
    plan: &ResolvedCommonPlan,
    result: &crate::CommonResult,
    complex: bool,
) {
    use eqiora_schema::kernel::{ExprNode, SymbolRef};
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let spatial = plan.as_spatial().unwrap();
    for definition in spatial.kernel().nodes().filter_map(|node| match node {
        KernelNode::Observable(definition)
            if definition.reduction() == eqiora_schema::kernel::ObservableReduction::Value =>
        {
            Some(definition)
        }
        _ => None,
    }) {
        let dag = definition.expression();
        let axis = dag.nodes().iter().find_map(|node| match node {
            ExprNode::CoordinatePartial { wrt, .. } => match dag.node(*wrt) {
                Some(ExprNode::Symbol(SymbolRef::Coordinate { axis, .. })) => Some(*axis),
                _ => unreachable!(),
            },
            _ => None,
        });
        let expected = match axis {
            None => [(2.97, 4.09), (-5.32, 0.88)], // x=.37, y=1.23
            Some(0) => [(2., -1.), (1., 3.)],
            Some(1) => [(1., 2.), (-3., -1.)],
            _ => unreachable!(),
        };
        let observation = result
            .observe_at(
                model,
                definition.id(),
                &[0.37, 1.23].map(|x| DynQuantity::new(x, length)),
                &Default::default(),
                None,
            )
            .unwrap();
        assert_eq!(observation.value().value_type(), definition.value_type());
        assert_eq!(
            observation.value().value_type().frame(),
            eqiora_core::ValueFrame::SpatialCartesian
        );
        assert_eq!(observation.value().value_type().array_rank(), 0);
        for ((real, imaginary), (a, b)) in observation.value().components().unwrap().zip(expected) {
            assert!(
                (real - a).abs() < 1e-9 && (imaginary - if complex { b } else { 0. }).abs() < 1e-9,
                "complex={complex}, axis={axis:?}: ({real},{imaginary}) != ({a},{b})"
            );
        }
    }
}

pub(super) fn check_channels(
    model: &ModelEnvelope,
    plan: &ResolvedCommonPlan,
    result: &crate::CommonResult,
    method: CommonSpatialPolicy,
    complex: bool,
) {
    let spatial = plan.as_spatial().unwrap();
    let ty = spatial.fields().next().unwrap().1;
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let replay = crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), plan).unwrap();
    for result in [result, &replay] {
        for definition in spatial.kernel().nodes().filter_map(|node| match node {
            KernelNode::Observable(definition)
                if definition.reduction() == eqiora_schema::kernel::ObservableReduction::Value =>
            {
                Some(definition)
            }
            _ => None,
        }) {
            // Distinct off-center points in the same [1/4,1/2] cell distinguish
            // nodal interpolation from the represented constant cell average.
            for x in [0.31, 0.44] {
                let observation = result
                    .observe_at(
                        model,
                        definition.id(),
                        &[DynQuantity::new(x, length)],
                        &Default::default(),
                        None,
                    )
                    .unwrap();
                assert_eq!(observation.value().value_type(), definition.value_type());
                assert_eq!(observation.value().value_type().array_rank(), 1);
                assert_eq!(observation.value().value_type().frame(), ty.frame());
                let slope = definition.value_type().dimension() != ty.dimension();
                let sampled_x = if method == CommonSpatialPolicy::Q1 {
                    x
                } else {
                    0.375
                };
                let values = if slope {
                    if method == CommonSpatialPolicy::Q1 {
                        [
                            (2., if complex { -1. } else { 0. }),
                            (1., if complex { 2. } else { 0. }),
                        ]
                    } else {
                        [(0., 0.); 2]
                    }
                } else {
                    [
                        (
                            1. + 2. * sampled_x,
                            if complex { 1. - sampled_x } else { 0. },
                        ),
                        (
                            -2. + sampled_x,
                            if complex { 3. + 2. * sampled_x } else { 0. },
                        ),
                    ]
                };
                // The ordinary expression interpreter owns conjugation and scaling.
                let reflected = definition
                    .expression()
                    .nodes()
                    .iter()
                    .any(|node| matches!(node, eqiora_schema::kernel::ExprNode::Mul(..)));
                for ((real, imaginary), (a, b)) in
                    observation.value().components().unwrap().zip(values)
                {
                    let expected = if reflected { (2. * a, -2. * b) } else { (a, b) };
                    assert!(
                        (real - expected.0).abs() < 1e-9 && (imaginary - expected.1).abs() < 1e-9,
                        "{method:?}, complex={complex}, slope={slope}, x={x}: ({real},{imaginary}) != {expected:?}"
                    );
                }
            }
            assert!(
                result
                    .observe_at(
                        model,
                        definition.id(),
                        &[DynQuantity::new(0.31, DimExponents::DIMENSIONLESS)],
                        &Default::default(),
                        None
                    )
                    .is_err()
            );
            assert!(
                result
                    .observe_at(
                        model,
                        definition.id(),
                        &[DynQuantity::new(1.1, length)],
                        &Default::default(),
                        None
                    )
                    .is_err()
            );
            if method != CommonSpatialPolicy::Q1
                || definition.value_type().dimension() != ty.dimension()
            {
                assert!(
                    result
                        .observe_at(
                            model,
                            definition.id(),
                            &[DynQuantity::new(0.5, length)],
                            &Default::default(),
                            None
                        )
                        .unwrap_err()
                        .message()
                        .contains("explicit side")
                );
            }
        }
    }
}
