//! Independent component sensitivities through the ordinary spatial Plan.
use super::*;
use eqiora_ir::{LinearizedOutput, LinearizedRelation, RelationCotangent, RelationTangent};
use eqiora_schema::kernel::KernelNode;

#[test]
fn channel_parameter_points_preserve_complete_primal_and_partial_actions() {
    let geometry = cartesian_interval();
    let source = POISSON_INTERVAL
        .replace("variable potential: 1", "variable potential: array<1,2>")
        .replace(
            "parameter source_scale: 1 / m ^ 2",
            "parameter source_scale: 1 / m ^ 2, parameter amplitudes:array<1,2>",
        )
        .replace(
            "-div(grad(potential)) - source_scale = 0",
            "-div(grad(potential)) + source_scale*(potential-amplitudes) = 0",
        )
        .replace("trace(potential) = 0", "trace(potential) = amplitudes");
    let body = geometry.entity_set("body").unwrap();
    let ty = eqiora_core::ValueType::scalar(
        eqiora_core::ScalarDomain::Real,
        DimExponents::DIMENSIONLESS,
    )
    .unwrap()
    .array(2)
    .unwrap();
    let model = compile_typed_model(
        "channel-parameters.eqi",
        &source,
        &geometry,
        "PoissonInterval",
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
        &[
            (
                "source_scale",
                eqiora_core::ValueLiteral::try_from(DynQuantity::new(
                    1.,
                    DimExponents::from_integers([0, -2, 0, 0, 0, 0, 0]).unwrap(),
                ))
                .unwrap(),
            ),
            (
                "amplitudes",
                eqiora_core::ValueLiteral::new(ty, vec![(2., 0.), (-3., 0.)]).unwrap(),
            ),
        ],
    );
    for spatial in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
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
        )
        .unwrap();
        let plan = resolved.as_spatial().unwrap();
        let parameter = plan
            .admission
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Parameter(value) if value.value_type().array_rank() == 1 => {
                    Some(value.id())
                }
                _ => None,
            })
            .unwrap();
        let scale = plan
            .admission
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Parameter(value) if value.value_type().array_rank() == 0 => {
                    Some(value.id())
                }
                _ => None,
            })
            .unwrap();
        let selected = [parameter, scale];
        let values = [5., -7., 2.];
        // Each independent channel solves -u''+s*u=s*p with u=p at both ends.
        // The exact solution and its Parameter tangent are constant on the mesh.
        let point = plan.differentiate(&selected, Some(&values)).unwrap();
        let linearization = point.linearization();
        let count = if spatial == CommonSpatialPolicy::Q1 {
            5
        } else {
            4
        };
        assert_eq!(linearization.output_values().len(), 2 * count);
        for value in linearization.output_values().as_chunks::<2>().0 {
            assert!((value[0] - 5.).abs() < 1e-11);
            assert!((value[1] + 7.).abs() < 1e-11);
        }
        let relation = linearization.relation();
        assert_eq!(relation.parameter_dimension(), 3);
        let direction = [3., -2., 4.];
        // Check the state operator independently: constant solutions alone would
        // also pass if the implementation silently dropped reaction and source.
        // On [0,1], Q1 has K=(1/h)[[1,-1],[-1,1]] and
        // M=(h/6)[[2,1],[1,2]]. TPFA integrates s*u over each cell;
        // an endpoint trace adds flux over the half-cell distance.
        let vertices = if spatial == CommonSpatialPolicy::Q1 {
            3
        } else {
            4
        };
        assert_eq!(relation.unknown_dimension(), 2 * vertices);
        let probe = (0..2 * vertices)
            .map(|i| (i + 1) as f64 * if i % 2 == 0 { 1. } else { -1. })
            .collect::<Vec<_>>();
        let mut actual = vec![0.; probe.len()];
        relation
            .jvp(RelationTangent::Unknown(&probe), &mut actual)
            .unwrap();
        let h = 0.25;
        for vertex in 0..vertices {
            let (diagonal, adjacent) = if spatial == CommonSpatialPolicy::Q1 {
                (
                    2. / h + values[2] * 2. * h / 3.,
                    -1. / h + values[2] * h / 6.,
                )
            } else {
                let boundary = vertex == 0 || vertex + 1 == vertices;
                (
                    (if boundary { 3. } else { 2. }) / h + values[2] * h,
                    -1. / h,
                )
            };
            for channel in 0..2 {
                let index = 2 * vertex + channel;
                let mut expected = diagonal * probe[index];
                if vertex > 0 {
                    expected += adjacent * probe[index - 2];
                }
                if vertex + 1 < vertices {
                    expected += adjacent * probe[index + 2];
                }
                assert!((actual[index] - expected).abs() < 1e-11);
            }
        }
        let unknown = (0..relation.unknown_dimension())
            .map(|i| direction[i % 2])
            .collect::<Vec<_>>();
        let mut state_action = vec![0.; relation.residual_dimension()];
        let mut parameter_action = state_action.clone();
        relation
            .jvp(RelationTangent::Unknown(&unknown), &mut state_action)
            .unwrap();
        relation
            .jvp(
                RelationTangent::Parameter(&direction),
                &mut parameter_action,
            )
            .unwrap();
        for (state, parameter) in state_action.iter().zip(&parameter_action) {
            assert!((state + parameter).abs() < 1e-11);
        }
        let mut output = vec![0.; 2 * count];
        linearization
            .jvp(&unknown, &direction, &mut output)
            .unwrap();
        for value in output.as_chunks::<2>().0 {
            assert!((value[0] - 3.).abs() < 1e-11);
            assert!((value[1] + 2.).abs() < 1e-11);
        }
        let weights = (0..relation.residual_dimension())
            .map(|i| (i + 1) as f64)
            .collect::<Vec<_>>();
        let mut pullback = [0.; 3];
        relation
            .vjp(&weights, RelationCotangent::Parameter(&mut pullback))
            .unwrap();
        let forward = weights
            .iter()
            .zip(parameter_action)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let reverse = direction
            .iter()
            .zip(pullback)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        assert!((forward - reverse).abs() < 1e-11);
        let replay = plan
            .validate_differentiated_values(&selected, Some(&values), relation.accepted_unknowns())
            .unwrap();
        assert_eq!(&replay, linearization);
    }
}

#[test]
fn coefficient_load_and_trace_partials_obey_the_same_discrete_balance() {
    let geometry = cartesian_interval();
    for natural in [false, true] {
        let source = POISSON_INTERVAL
        .replace("on body in h1", "on body in smooth")
        .replace("parameter source_scale: 1 / m ^ 2", "parameter source_scale: 1 / m ^ 2, parameter diffusivity:1, parameter boundary_value:1, parameter boundary_flux:1/m")
        .replace("-div(grad(potential))", "-div(diffusivity*grad(potential))")
        .replace("trace(potential) = 0", "trace(potential) = boundary_value")
        .replace("relation right_value on right { trace(potential) = boundary_value; }",
            if natural { "relation right_value on right { normal(diffusivity*grad(potential)) = boundary_flux; }" } else { "relation right_value on right { trace(potential) = 2*boundary_value; }" });
        let body = geometry.entity_set("body").unwrap();
        let model = compile_model(
            "coefficient-trace-partials.eqi",
            &source,
            &geometry,
            "PoissonInterval",
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
            &[
                (
                    "boundary_flux",
                    DynQuantity::new(
                        11.,
                        DimExponents::from_integers([0, -1, 0, 0, 0, 0, 0]).unwrap(),
                    ),
                ),
                (
                    "diffusivity",
                    DynQuantity::new(2., DimExponents::DIMENSIONLESS),
                ),
                (
                    "source_scale",
                    DynQuantity::new(
                        5.,
                        DimExponents::from_integers([0, -2, 0, 0, 0, 0, 0]).unwrap(),
                    ),
                ),
                (
                    "boundary_value",
                    DynQuantity::new(7., DimExponents::DIMENSIONLESS),
                ),
            ],
        );
        for spatial in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            let plan =
                resolve_scalar_box(&model, cartesian_box_resources(&geometry, &[4]), spatial);
            let selected = [2., 5., 7., 11.].map(|canonical| {
                plan.kernel()
                    .nodes()
                    .find_map(|node| match node {
                        KernelNode::Parameter(p)
                            if p.value().component(0) == Some((canonical, 0.)) =>
                        {
                            Some(p.id())
                        }
                        _ => None,
                    })
                    .unwrap()
            });
            let point = plan.differentiate(&selected, None).unwrap();
            let linear = point.linearization();
            let relation = linear.relation();
            let h = 0.25;
            let cells = spatial == CommonSpatialPolicy::CellCenteredTpfa;
            let locations = if cells {
                vec![h / 2., 3. * h / 2., 5. * h / 2., 7. * h / 2.]
            } else {
                vec![0., h, 2. * h, 3. * h, 1.]
            };
            // -k*u''=q, u(0)=p, with either u(1)=2p or k*u'(1)=r.
            // Q1 gives exact nodal samples of the quadratic polynomial.
            // Half-cell TPFA trace resistance adds q*h^2/(8*k) at every cell.
            let shape = |x: f64| {
                (if natural {
                    x - x * x / 2.
                } else {
                    x * (1. - x) / 2.
                }) + if cells { h * h / 8. } else { 0. }
            };
            let lift = |x: f64| if natural { 1. } else { 1. + x };
            let flux = |x: f64| if natural { x } else { 0. };
            let direction = [-0.5, 3., -2., 4.];
            let tangent = |x: f64| {
                direction[2] * lift(x)
                    + (direction[1] / 2. - 5. * direction[0] / 4.) * shape(x)
                    + (direction[3] / 2. - 11. * direction[0] / 4.) * flux(x)
            };
            for (value, x) in linear.output_values().iter().zip(&locations) {
                let expected = 7. * lift(*x) + 5. / 2. * shape(*x) + 11. / 2. * flux(*x);
                assert!(
                    (value - expected).abs() < 1e-11,
                    "{spatial:?}: {value} != {expected}"
                );
            }
            let unknown = locations
                .iter()
                .copied()
                .filter(|x| cells || (*x > 0. && (natural || *x < 1.)))
                .map(tangent)
                .collect::<Vec<_>>();
            let mut state = vec![0.; unknown.len()];
            let mut parameters = state.clone();
            relation
                .jvp(RelationTangent::Unknown(&unknown), &mut state)
                .unwrap();
            relation
                .jvp(RelationTangent::Parameter(&direction), &mut parameters)
                .unwrap();
            for (state, parameter) in state.iter().zip(parameters) {
                assert!(
                    (state + parameter).abs() < 1e-11,
                    "{spatial:?}: {state} + {parameter}"
                );
            }
            let mut output = vec![0.; locations.len()];
            linear.jvp(&unknown, &direction, &mut output).unwrap();
            for (value, x) in output.iter().zip(&locations) {
                assert!((value - tangent(*x)).abs() < 1e-11);
            }
        }
    }
}
