//! Independently derived trajectory products for u'=p and z=u².
use super::*;
use eqiora_ir::{DiscreteStepLinearization, LinearizedOutput, LinearizedRelation, RelationTangent};
use eqiora_schema::kernel::KernelNode;
use eqiora_solver::LinearOperatorOrientation;

fn resolve_products(
    complex: bool,
    spatial: CommonSpatialPolicy,
    mixed: bool,
    constrained: bool,
) -> (ResolvedCommonPlan, eqiora_core::RawId) {
    let geometry = cartesian_interval();
    let scalar = if complex { "complex<1>" } else { "1" };
    let first = if mixed {
        "(1[s]*derivative(u)-p)+(z-u*u)"
    } else {
        "1[s]*derivative(u)-p"
    };
    let second = if mixed {
        "(1[s]*derivative(u)-p)-(z-u*u)"
    } else {
        "z-u*u"
    };
    let z = if constrained {
        format!(
            "variable z:{scalar} on body in smooth; relation constraint on body {{{second}=0;}}"
        )
    } else {
        String::new()
    };
    let form = if constrained && spatial == CommonSpatialPolicy::Q1 {
        format!(
            "form mixed for evolution,constraint {{ test wu:1 for u; test wz:1 for z; evolution: integrate(body,inner(wu,({first})))=0; constraint: integrate(body,inner(wz,({second})))=0; }}"
        )
    } else {
        String::new()
    };
    let observation_type = if complex { "complex<m>" } else { "m" };
    let observed_parameter = if complex { "math.complex(p,0)" } else { "p" };
    let source = format!(
        "model Evolution(support body:volume(ambient_dimension=1),support left:boundary(parent=body),support right:boundary(parent=body)) {{ parameter p:1=1; state u:{scalar} on body in smooth; {z} relation evolution on body {{{first}=0;}} observable sample:{observation_type}=integral((1+time()/1[s])*u+{observed_parameter},measure(body)); {form} }}"
    );
    let bindings = ["body", "left", "right"].map(|name| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: (name != "body").then(|| geometry.entity_set("body").unwrap()),
            },
        )
    });
    let compiled =
        CompiledModel::compile_selected("midpoint-products.eqi", &source, "Evolution", &bindings)
            .unwrap();
    let projection = compiled
        .authored_formulations()
        .next()
        .map(|form| form.projection().clone());
    let (transaction, model, symbols) = compiled.into_parts();
    let dynamic_field = symbols.get("u").unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry]).unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    let base = ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[2]),
        spatial,
        if constrained {
            CommonSolvePolicy::Newton {
                linear: exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    0.,
                    1e-14,
                    NonZeroUsize::new(100).unwrap(),
                ),
                nonlinear: NonlinearSolvePlan::new(0., 1e-12, NonZeroUsize::new(16).unwrap(), 16)
                    .unwrap(),
            }
        } else {
            CommonSolvePolicy::Linear(exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                0.,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
            ))
        },
        None,
        Some(CommonTemporalPolicy::backward_euler(0.1).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
        projection.as_ref(),
    )
    .unwrap();
    (
        policy::with_policy(
            &base,
            policy::policy(&base, eqiora_time::TimeMethod::ImplicitMidpoint, 0.1),
        )
        .unwrap(),
        dynamic_field,
    )
}

#[test]
fn midpoint_runs_compose_requested_outputs_and_reopen_every_internal_step() {
    for complex in [false, true] {
        for (spatial, mixed) in [
            (CommonSpatialPolicy::Q1, false),
            (CommonSpatialPolicy::Q1, true),
            (CommonSpatialPolicy::CellCenteredTpfa, false),
        ] {
            for constrained in [false, true] {
                if mixed && !constrained {
                    continue;
                }
                let (resolved, dynamic_field) =
                    resolve_products(complex, spatial, mixed, constrained);
                let plan = resolved.as_spatial().unwrap();
                let parameter = plan
                    .kernel()
                    .nodes()
                    .find_map(|node| match node {
                        KernelNode::Parameter(parameter) => Some(parameter.id()),
                        _ => None,
                    })
                    .unwrap();
                let fields = plan
                    .fields()
                    .map(|(field, _)| field.erase())
                    .collect::<Vec<_>>();
                let observable = plan
                    .kernel()
                    .nodes()
                    .find_map(|node| match node {
                        KernelNode::Observable(value) => Some(value),
                        _ => None,
                    })
                    .unwrap();
                let rules = std::collections::HashMap::from([(
                    observable.reduction().domain().unwrap(),
                    eqiora_meshing::QuadratureRule::gauss_legendre(2).unwrap(),
                )]);
                let mut outputs = fields.clone();
                outputs.push(observable.id().erase());
                let count = if spatial == CommonSpatialPolicy::Q1 {
                    3
                } else {
                    2
                };
                let initial_fields = plan
                    .fields()
                    .map(|(field, ty)| {
                        let values = CommonInitialValues::new(
                            ty.shape().clone(),
                            vec![C::new(1., 0.); count],
                        )
                        .unwrap();
                        CommonInitialField::new(
                            resolved.model_artifact().digest().unwrap(),
                            field,
                            (spatial == CommonSpatialPolicy::Q1).then(|| values.clone()),
                            (spatial != CommonSpatialPolicy::Q1).then_some(values),
                        )
                        .unwrap()
                    })
                    .collect();
                let initial = plan.initial_state(Some(0.), initial_fields).unwrap();
                let request = CommonTransientRunRequest::from_times(
                    resolved.clone(),
                    initial.clone(),
                    0.25,
                    vec![0.025, 0.075, 0.1, 0.225, 0.25],
                )
                .unwrap();
                let trajectory = plan
                    .differentiate_trajectory(
                        &request,
                        &[parameter],
                        Some(&[2.]),
                        &outputs,
                        &rules,
                        &REFERENCE_LINEAR_SOLVER,
                    )
                    .unwrap_or_else(|error| panic!("complex={complex}, spatial={spatial:?}, mixed={mixed}, constrained={constrained}: {error}"));
                assert_eq!(trajectory.steps().len(), 3);
                let candidates = trajectory
                    .steps()
                    .iter()
                    .map(|point| point.candidate())
                    .collect::<Vec<_>>();
                let candidates: Vec<CommonDifferentiationCandidate> =
                    serde_json::from_slice(&serde_json::to_vec(&candidates).unwrap()).unwrap();
                let replay = plan
                    .validate_differentiated_trajectory_values(
                        &request,
                        &[parameter],
                        Some(&[2.]),
                        &outputs,
                        &rules,
                        &candidates,
                    )
                    .unwrap();
                for (point, reopened) in trajectory.steps().iter().zip(&replay) {
                    assert_eq!(point.linearization(), reopened);
                }
                let mut previous = vec![
                    0.;
                    trajectory.steps()[0]
                        .linearization()
                        .discrete_step()
                        .unwrap()
                        .previous_state()
                        .len()
                ];
                let mut observed = Vec::new();
                for (index, point) in trajectory.steps().iter().enumerate() {
                    let linearization = point.linearization();
                    let step = linearization.discrete_step().unwrap();
                    assert_eq!(step.start_time(), [0., 0.1, 0.2][index]);
                    assert_eq!(step.end_time(), [0.1, 0.2, 0.25][index]);
                    if index == 1 {
                        assert!(linearization.observation_times().unwrap().is_empty());
                        assert_eq!(linearization.output_dimension(), 0);
                    }
                    let candidate = point.candidate();
                    assert_eq!(
                        candidate.solve_count(),
                        if constrained {
                            2 + linearization.observation_times().unwrap().len()
                        } else {
                            1
                        }
                    );
                    let mut direction = previous.clone();
                    direction.push(1.);
                    let mut rhs = vec![0.; linearization.relation().residual_dimension()];
                    LinearizedRelation::jvp(
                        &step,
                        RelationTangent::Parameter(&direction),
                        &mut rhs,
                    )
                    .unwrap();
                    for value in &mut rhs {
                        *value = -*value;
                    }
                    let tangent = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan.linear())
                        .solve_canonical_oriented(
                            linearization.relation().state_jacobian(),
                            &rhs,
                            LinearOperatorOrientation::Normal,
                        )
                        .unwrap();
                    let output = step.selected_output();
                    let mut selected = vec![0.; output.output_dimension()];
                    output
                        .jvp(tangent.values(), &direction, &mut selected)
                        .unwrap();
                    let mut cursor = 0;
                    for &time in linearization.observation_times().unwrap() {
                        for (field, (_, ty)) in fields.iter().zip(plan.fields()) {
                            let is_u = field == &dynamic_field;
                            let primal = if is_u {
                                1. + 2. * time
                            } else {
                                (1. + 2. * time).powi(2)
                            };
                            let derivative = if is_u {
                                time
                            } else {
                                2. * (1. + 2. * time) * time
                            };
                            let width = if ty.scalar_domain() == eqiora_core::ScalarDomain::Complex
                            {
                                2
                            } else {
                                1
                            };
                            for _ in 0..count {
                                assert!(
                                    (linearization.output_values()[cursor] - primal).abs() < 1e-9
                                );
                                assert!(
                                    (selected[cursor] - derivative).abs() < 1e-9,
                                    "time {time}: {} vs {derivative}",
                                    selected[cursor]
                                );
                                cursor += 1;
                                if width == 2 {
                                    assert!(selected[cursor].abs() < 1e-9);
                                    cursor += 1;
                                }
                            }
                        }
                        // Unit interval: Q(t)=(1+t)*u(t)+p. Time and p direct actions
                        // must be evaluated at the actual observation, not the endpoint.
                        let expected = (1. + time) * (1. + 2. * time) + 2.;
                        let derivative = (1. + time) * time + 1.;
                        assert!((linearization.output_values()[cursor] - expected).abs() < 1e-9);
                        assert!((selected[cursor] - derivative).abs() < 1e-9);
                        cursor += 1;
                        if complex {
                            assert!(selected[cursor].abs() < 1e-9);
                            cursor += 1;
                        }
                        observed.push(time);
                    }
                    previous = vec![0.; step.next_state().len()];
                    LinearizedOutput::jvp(&step, tangent.values(), &direction, &mut previous)
                        .unwrap();
                }
                assert_eq!(observed, request.output_times_s());
                let mut missing = candidates.clone();
                missing.pop();
                assert!(
                    plan.validate_differentiated_trajectory_values(
                        &request,
                        &[parameter],
                        Some(&[2.]),
                        &outputs,
                        &rules,
                        &missing
                    )
                    .is_err()
                );
                let mut forged = candidates.clone();
                let crate::numerical_admission::differentiation::Candidate::Midpoint {
                    stage, ..
                } = &mut forged[0].kind
                else {
                    unreachable!()
                };
                stage.unknown[0] += 0.5;
                assert!(
                    plan.validate_differentiated_trajectory_values(
                        &request,
                        &[parameter],
                        Some(&[2.]),
                        &outputs,
                        &rules,
                        &forged
                    )
                    .is_err()
                );
                if constrained {
                    for missing_endpoint in [false, true] {
                        let mut incomplete = candidates.clone();
                        let crate::numerical_admission::differentiation::Candidate::Midpoint {
                            endpoint,
                            observations,
                            ..
                        } = &mut incomplete[0].kind
                        else {
                            unreachable!()
                        };
                        if missing_endpoint {
                            *endpoint = None;
                        } else {
                            observations.pop();
                        }
                        assert!(
                            plan.validate_differentiated_trajectory_values(
                                &request,
                                &[parameter],
                                Some(&[2.]),
                                &outputs,
                                &rules,
                                &incomplete
                            )
                            .is_err()
                        );
                    }
                }
                let mut extra = candidates.clone();
                let crate::numerical_admission::differentiation::Candidate::Midpoint {
                    stage,
                    observations,
                    ..
                } = &mut extra[1].kind
                else {
                    unreachable!()
                };
                observations.push(stage.clone());
                assert!(
                    plan.validate_differentiated_trajectory_values(
                        &request,
                        &[parameter],
                        Some(&[2.]),
                        &outputs,
                        &rules,
                        &extra
                    )
                    .is_err()
                );
                assert_eq!(request.state(), &initial);
            }
        }
    }
}
