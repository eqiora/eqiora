//! Independent consistency chart with and without invertible equation row mixing.
use super::*;
use eqiora_core::RawId;

fn resolve_constraint(
    spatial: CommonSpatialPolicy,
    mixed: bool,
    constraint: &str,
) -> ResolvedCommonPlan {
    resolve_typed_constraint(spatial, mixed, constraint, false, "1[s]*derivative(u)-1").0
}

fn resolve_typed_constraint(
    spatial: CommonSpatialPolicy,
    mixed: bool,
    constraint: &str,
    complex: bool,
    dynamic: &str,
) -> (ResolvedCommonPlan, [RawId; 2]) {
    resolve_typed_constraint_with_data(spatial, mixed, constraint, complex, dynamic, "")
}

fn resolve_typed_constraint_with_data(
    spatial: CommonSpatialPolicy,
    mixed: bool,
    constraint: &str,
    complex: bool,
    dynamic: &str,
    data: &str,
) -> (ResolvedCommonPlan, [RawId; 2]) {
    let scalar = if complex { "complex<1>" } else { "1" };
    let geometry = cartesian_interval();
    let first = if mixed {
        format!("({dynamic})+({constraint})")
    } else {
        dynamic.into()
    };
    let second = if mixed {
        format!("({dynamic})-({constraint})")
    } else {
        constraint.into()
    };
    let form = if spatial == CommonSpatialPolicy::Q1 {
        format!(
            "form mixed for evolution,constraint {{
            test wu:1 for u; test wz:1 for z;
            evolution: integrate(body,inner(wu,({first})))=0;
            constraint: integrate(body,inner(wz,({second})))=0;
        }}"
        )
    } else {
        String::new()
    };
    let source = format!(
        "public component Mixed(support body:volume(ambient_dimension=1), support left:boundary(parent=body), support right:boundary(parent=body)) {{
        state u:{scalar} on body in smooth;
        variable z:{scalar} on body in smooth;
        {data}
        relation evolution on body {{{first}=0;}}
        relation constraint on body {{{second}=0;}}
        {form}
    }}"
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
        CompiledModel::compile_selected("recovery.eqi", &source, "Mixed", &bindings).unwrap();
    let projection = compiled
        .authored_formulations()
        .next()
        .map(|form| form.projection().clone());
    let (transaction, model, symbols) = compiled.into_parts();
    let fields = ["definition.u", "definition.z"].map(|name| symbols.get(name).unwrap());
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry]).unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    let base = ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[2]),
        spatial,
        CommonSolvePolicy::Newton {
            linear: exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                0.,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
            ),
            nonlinear: NonlinearSolvePlan::new(0., 1e-12, NonZeroUsize::new(16).unwrap(), 16)
                .unwrap(),
        },
        None,
        Some(CommonTemporalPolicy::backward_euler(0.1).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
        projection.as_ref(),
    )
    .unwrap_or_else(|error| panic!("{spatial:?}, mixed={mixed}, dynamic={dynamic}, constraint={constraint}, data={data}: {error}"));
    let resolved = policy::with_policy(
        &base,
        policy::policy(&base, eqiora_time::TimeMethod::ImplicitMidpoint, 0.1),
    )
    .unwrap();
    let bytes = resolved.to_bytes().unwrap();
    let replay = ResolvedCommonPlan::from_bytes(
        &bytes,
        &REFERENCE_LINEAR_SOLVER,
        &[eqiora_time::ImplicitMidpointTimeBackend::CAPABILITIES],
    )
    .unwrap();
    assert_eq!(replay.to_bytes().unwrap(), bytes);
    (replay, fields)
}

#[test]
fn original_residual_recovers_rates_and_algebraic_state_without_row_labels() {
    for (spatial, mixed) in [
        (CommonSpatialPolicy::Q1, false),
        (CommonSpatialPolicy::Q1, true),
        (CommonSpatialPolicy::CellCenteredTpfa, false),
    ] {
        let resolved = resolve_constraint(spatial, mixed, "z-u*u");
        let plan = resolved.as_spatial().unwrap();
        let RecognizedNativeModel::Spatial(equations) = plan.admission.recognized_model() else {
            unreachable!()
        };
        let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
        // u=1.1 is held. The equations independently give v_u=1,z=u^2=1.21.
        // L=[[1,1],[1,-1]] has determinant -2, so row mixing cannot change this.
        let fixed = vec![1.1; system.dimension()];
        let seed = vec![0.1; system.dimension()];
        let original = (fixed.clone(), seed.clone());
        let (unknown, _) = system
            .recover_consistent(0.1, 0.1, &fixed, &seed, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let (state, rate) = system.recovery_coordinates(0.1, &fixed, &unknown).unwrap();
        let mut algebraic = 0;
        for (&state, &rate) in state.iter().zip(&rate) {
            if rate == 0. {
                assert!((state - 1.21).abs() < 1e-11);
                algebraic += 1;
            } else {
                assert_eq!(state, 1.1);
                assert!((rate - 1.).abs() < 1e-11);
            }
        }
        assert_eq!(algebraic * 2, system.dimension());
        let mut residual = vec![f64::NAN; system.dimension()];
        system.residual(0.1, &state, &rate, &mut residual).unwrap();
        assert!(residual.iter().all(|value| value.abs() < 1e-12));
        assert_eq!((fixed, seed), original);
    }
}

#[test]
fn midpoint_stage_and_endpoint_recovery_satisfy_distinct_physical_constraints() {
    for (spatial, mixed) in [
        (CommonSpatialPolicy::Q1, false),
        (CommonSpatialPolicy::Q1, true),
        (CommonSpatialPolicy::CellCenteredTpfa, false),
    ] {
        let resolved = resolve_constraint(spatial, mixed, "z-u*u");
        let plan = resolved.as_spatial().unwrap();
        let RecognizedNativeModel::Spatial(equations) = plan.admission.recognized_model() else {
            unreachable!()
        };
        let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
        let initial = vec![1.; system.dimension()];
        let (unknown, _) = system
            .midpoint_step(0., 0.1, &initial, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        // u(t)=1+t. At the midpoint, z=1.05^2; at the endpoint z=1.1^2.
        // The stage unknown for z is twice its midpoint value, never z(0.1).
        let seed = unknown
            .iter()
            .map(|value| if *value > 2. { value * 0.5 } else { value - 1. })
            .collect::<Vec<_>>();
        assert!(
            unknown
                .iter()
                .all(|value| (value - 1.1).abs() < 1e-11 || (value - 2.205).abs() < 1e-11)
        );
        let (recovered, _) = system
            .recover_consistent(0.1, 0.1, &unknown, &seed, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let (state, rate) = system
            .recovery_coordinates(0.1, &unknown, &recovered)
            .unwrap();
        for (&state, &rate) in state.iter().zip(&rate) {
            let expected = if rate == 0. { 1.21 } else { 1.1 };
            assert!((state - expected).abs() < 1e-11);
        }
        assert_eq!(initial, vec![1.; system.dimension()]);
    }
}

#[test]
fn recovery_rejects_singular_chart_even_with_zero_original_residual() {
    for spatial in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let resolved = resolve_constraint(spatial, false, "z*z-u*u");
        let plan = resolved.as_spatial().unwrap();
        let RecognizedNativeModel::Spatial(equations) = plan.admission.recognized_model() else {
            unreachable!()
        };
        let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
        let fixed = vec![0.; system.dimension()];
        // All unknowns zero give z=0 and v_u=0. Supply v_u=1 using a
        // preceding regular recovery's rate partition, without assuming order.
        let regular = vec![1.; system.dimension()];
        let guess = vec![0.1; system.dimension()];
        let (unknown, _) = system
            .recover_consistent(0., 0.1, &regular, &guess, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let (_, rate) = system
            .recovery_coordinates(0.1, &regular, &unknown)
            .unwrap();
        let seed = rate
            .iter()
            .map(|value| if *value == 0. { 0. } else { 0.1 })
            .collect::<Vec<_>>();
        let (state, rate) = system.recovery_coordinates(0.1, &fixed, &seed).unwrap();
        let mut residual = vec![f64::NAN; system.dimension()];
        system.residual(0., &state, &rate, &mut residual).unwrap();
        assert!(residual.iter().all(|value| *value == 0.));
        let error = system
            .recover_consistent(0., 0.1, &fixed, &seed, &REFERENCE_LINEAR_SOLVER)
            .unwrap_err();
        assert!(error.message().contains("locally regular"), "{error}");
    }
}

#[test]
fn constrained_midpoint_runs_recover_physical_outputs_and_replay_without_resolving() {
    for (complex, conjugate) in [(false, false), (true, false), (true, true)] {
        for (spatial, mixed) in [
            (CommonSpatialPolicy::Q1, false),
            (CommonSpatialPolicy::Q1, true),
            (CommonSpatialPolicy::CellCenteredTpfa, false),
        ] {
            let constraint = if conjugate {
                "z-u*math.conj(u)"
            } else {
                "z-u*u"
            };
            let (resolved, fields) = resolve_typed_constraint(
                spatial,
                mixed,
                constraint,
                complex,
                "1[s]*derivative(u)-1",
            );
            let plan = resolved.as_spatial().unwrap();
            let q1 = spatial == CommonSpatialPolicy::Q1;
            let count = if q1 { 3 } else { 2 };
            let u0 = C::new(1., if complex { 2. } else { 0. });
            let z = |u: C| u * if conjugate { u.conj() } else { u };
            let assignments = plan
                .fields()
                .map(|(field, ty)| {
                    let value = if field.erase() == fields[0] {
                        u0
                    } else {
                        z(u0)
                    };
                    let values =
                        CommonInitialValues::new(ty.shape().clone(), vec![value; count]).unwrap();
                    CommonInitialField::new(
                        resolved.model_artifact().digest().unwrap(),
                        field,
                        q1.then(|| values.clone()),
                        (!q1).then_some(values),
                    )
                    .unwrap()
                })
                .collect();
            let initial = plan.initial_state(Some(0.), assignments).unwrap();
            let before = initial.to_bytes().unwrap();
            let run = |state, times| {
                CommonTransientRunRequest::from_times(resolved.clone(), state, 0.3, times)
                    .unwrap()
                    .run(
                        &REFERENCE_LINEAR_SOLVER,
                        |_, _| panic!("midpoint uses the production stage owner"),
                        |_, _| panic!("unexpected residual-native backend"),
                    )
                    .unwrap()
            };
            let dense = run(initial.clone(), vec![0.05, 0.1, 0.15, 0.2, 0.3]);
            let sparse = run(initial.clone(), vec![0.3]);
            let interior = run(initial.clone(), vec![0.05]);
            assert_eq!(dense.spatial_time_history(), sparse.spatial_time_history());
            assert_eq!(
                dense.spatial_time_history(),
                interior.spatial_time_history()
            );
            assert_eq!(dense.spatial_state(0), interior.spatial_state(0));
            assert_eq!(dense.spatial_state(4), sparse.spatial_state(0));
            for (index, time) in [0.05, 0.1, 0.15, 0.2, 0.3].into_iter().enumerate() {
                let state = dense.spatial_state(index).unwrap();
                for (field, _, _, values) in plan.state_fields(state).unwrap() {
                    let u = u0 + time;
                    let expected = if field.erase() == fields[0] { u } else { z(u) };
                    close(values, &coordinates(&vec![expected; count], complex));
                }
            }
            let resumed = run(dense.spatial_state(1).unwrap().clone(), vec![0.3]);
            close(
                resumed.spatial_state(0).unwrap().spatial_values().unwrap(),
                dense.spatial_state(4).unwrap().spatial_values().unwrap(),
            );
            assert_eq!(initial.to_bytes().unwrap(), before);
            let bytes = dense.to_bytes().unwrap();
            assert_eq!(
                crate::CommonTrajectory::from_bytes(&bytes, &resolved)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
            let result = crate::CommonResult::accept_trajectory(0., dense).unwrap();
            let result_bytes = result.to_bytes().unwrap();
            assert_eq!(
                crate::CommonResult::from_bytes(&result_bytes, &resolved)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                result_bytes
            );
            let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            let witness = &mut forged["payload"]["execution"]["endpoints"][0];
            let wrong = witness["unknown"][0].as_f64().unwrap() + 0.1;
            witness["unknown"][0] = serde_json::json!(wrong);
            witness["convergence"] = serde_json::json!({"kind":"exact-residual-zero"});
            let error = crate::CommonTrajectory::from_bytes(
                &serde_json::to_vec(&forged).unwrap(),
                &resolved,
            )
            .unwrap_err();
            assert!(error.message().contains("zero-residual"), "{error}");
        }
    }
}

#[test]
fn algebraic_midpoint_feedback_changes_the_differential_update() {
    for (spatial, mixed) in [
        (CommonSpatialPolicy::Q1, false),
        (CommonSpatialPolicy::Q1, true),
        (CommonSpatialPolicy::CellCenteredTpfa, false),
    ] {
        let (resolved, _) =
            resolve_typed_constraint(spatial, mixed, "z-u*u", false, "1[s]*derivative(u)+z");
        let plan = resolved.as_spatial().unwrap();
        let RecognizedNativeModel::Spatial(equations) = plan.admission.recognized_model() else {
            unreachable!()
        };
        let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
        let initial = vec![1.; system.dimension()];
        let (unknown, _) = system
            .midpoint_step(0., 0.1, &initial, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        // u'=-z,z=u²: h*w²+2*w-2*u0=0 at the collocation point.
        // This cancellation-free positive root is independent of the solver.
        let w = 2. / (1. + 1.2_f64.sqrt());
        let end = 2. * w - 1.;
        assert!(
            unknown
                .iter()
                .all(|value| (value - end).abs() < 1e-11 || (value - 2. * w * w).abs() < 1e-11)
        );
        let seed = unknown
            .iter()
            .map(|value| if *value > 1. { value / 2. } else { value - 1. })
            .collect::<Vec<_>>();
        let (recovered, _) = system
            .recover_consistent(0.1, 0.1, &unknown, &seed, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let (state, rate) = system
            .recovery_coordinates(0.1, &unknown, &recovered)
            .unwrap();
        for (&state, &rate) in state.iter().zip(&rate) {
            if rate == 0. {
                assert!((state - end * end).abs() < 1e-11);
            } else {
                assert!((state - end).abs() < 1e-11);
                assert!((rate + end * end).abs() < 1e-11);
            }
        }
    }
}

#[test]
fn one_complex_field_retains_differential_real_and_algebraic_imaginary_coordinates() {
    for spatial in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let base = resolve(
            true,
            spatial,
            0.1,
            "1[s]*derivative(math.complex(math.real(u),0))+math.complex(0,math.imag(u)-math.real(u)*math.real(u))=1;",
            None,
        );
        let resolved = policy::with_policy(
            &base,
            policy::policy(&base, eqiora_time::TimeMethod::ImplicitMidpoint, 0.1),
        )
        .unwrap();
        let plan = resolved.as_spatial().unwrap();
        let (field, ty) = plan.fields().next().unwrap();
        let q1 = spatial == CommonSpatialPolicy::Q1;
        let count = if q1 { 3 } else { 2 };
        let values =
            CommonInitialValues::new(ty.shape().clone(), vec![C::new(1., 1.); count]).unwrap();
        let assignment = CommonInitialField::new(
            resolved.model_artifact().digest().unwrap(),
            field,
            q1.then(|| values.clone()),
            (!q1).then_some(values),
        )
        .unwrap();
        let initial = plan.initial_state(Some(0.), vec![assignment]).unwrap();
        let before = initial.to_bytes().unwrap();
        let trajectory = CommonTransientRunRequest::from_times(
            resolved.clone(),
            initial.clone(),
            0.2,
            vec![0.05, 0.2],
        )
        .unwrap()
        .run(
            &REFERENCE_LINEAR_SOLVER,
            |_, _| panic!("production midpoint"),
            |_, _| panic!("unexpected residual-native backend"),
        )
        .unwrap();
        assert_eq!(
            trajectory.spatial_time_history().unwrap().dimension(),
            count
        );
        for (index, t) in [0.05, 0.2].into_iter().enumerate() {
            let state = trajectory.spatial_state(index).unwrap();
            let values = plan.state_fields(state).unwrap().next().unwrap().3;
            close(
                values,
                &coordinates(&vec![C::new(1. + t, (1. + t) * (1. + t)); count], true),
            );
        }
        assert_eq!(initial.to_bytes().unwrap(), before);
        let bytes = trajectory.to_bytes().unwrap();
        assert_eq!(
            crate::CommonTrajectory::from_bytes(&bytes, &resolved)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
    }
}

#[test]
fn midpoint_rejects_a_singular_collocation_chart_even_when_endpoints_are_regular() {
    let (base, fields) = resolve_typed_constraint(
        CommonSpatialPolicy::CellCenteredTpfa,
        false,
        "u+(time()/1[s]-0.0625)*(time()/1[s]-0.0625)*z",
        false,
        "1[s]*derivative(u)+z",
    );
    let resolved = policy::with_policy(
        &base,
        policy::policy(&base, eqiora_time::TimeMethod::ImplicitMidpoint, 0.125),
    )
    .unwrap();
    let plan = resolved.as_spatial().unwrap();
    let assignments = plan
        .fields()
        .map(|(field, ty)| {
            let value = if field.erase() == fields[0] {
                1.
            } else {
                -256.
            };
            let values =
                CommonInitialValues::new(ty.shape().clone(), vec![C::new(value, 0.); 2]).unwrap();
            CommonInitialField::new(
                resolved.model_artifact().digest().unwrap(),
                field,
                None,
                Some(values),
            )
            .unwrap()
        })
        .collect();
    let initial = plan.initial_state(Some(0.), assignments).unwrap();
    let before = initial.to_bytes().unwrap();
    // At t=h/2=1/16, the algebraic coefficient is exactly zero. The
    // midpoint stage matrix is regular, but [F_vD/h,F_yA] is singular.
    // Endpoint-only observations must not hide that failed local chart.
    for outputs in [vec![0.125], vec![0.0625, 0.125]] {
        let request = CommonTransientRunRequest::from_times(
            resolved.clone(),
            initial.clone(),
            0.125,
            outputs,
        )
        .unwrap();
        let error = match request.run(
            &REFERENCE_LINEAR_SOLVER,
            |_, _| panic!("production midpoint"),
            |_, _| panic!("unexpected residual-native backend"),
        ) {
            Ok(_) => panic!("midpoint accepted a singular collocation recovery chart"),
            Err(error) => error,
        };
        assert!(error.message().contains("locally regular"), "{error}");
        assert_eq!(initial.to_bytes().unwrap(), before);
    }
}

#[test]
fn physical_initial_rates_differentiate_constraints_without_changing_state() {
    for (spatial, mixed) in [
        (CommonSpatialPolicy::Q1, false),
        (CommonSpatialPolicy::Q1, true),
        (CommonSpatialPolicy::CellCenteredTpfa, false),
    ] {
        for (constraint, expected) in [("z-u*u", 2.), ("z-u*u-time()/1[s]", 3.)] {
            // u'=1, z=u^2 (+t); at u=z=1,t=0 the physical z' is 2 (+1).
            // A positive State-dependent multiplier and invertible row mixing
            // do not change this independently differentiated solution.
            let (resolved, _) = resolve_typed_constraint(
                spatial,
                mixed,
                constraint,
                false,
                "(1+u*u)*(1[s]*derivative(u)-1)",
            );
            let plan = resolved.as_spatial().unwrap();
            let RecognizedNativeModel::Spatial(equations) = plan.admission.recognized_model()
            else {
                unreachable!()
            };
            let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
            let initial = vec![1.; system.dimension()];
            let before = initial.clone();
            let rates = system
                .initial_rates(0., &initial, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
            assert_eq!(initial, before);
            assert_eq!(
                rates.iter().filter(|v| (**v - 1.).abs() < 1e-11).count() * 2,
                rates.len()
            );
            assert_eq!(
                rates
                    .iter()
                    .filter(|v| (**v - expected).abs() < 1e-11)
                    .count()
                    * 2,
                rates.len()
            );
            let mut residual = vec![f64::NAN; rates.len()];
            system
                .residual(0., &initial, &rates, &mut residual)
                .unwrap();
            assert!(residual.iter().all(|v| v.abs() < 1e-12));
        }
    }
}

#[test]
fn physical_initial_rates_do_not_differentiate_regular_row_forcing() {
    // u'=sqrt(t), z=u², u(0)=z(0)=1 gives u'(0)=z'(0)=0.
    // The physical pair exists even though the nuisance acceleration u''(0)
    // does not; differentiating the dynamic forcing is not needed for z'.
    check_initial_zero_rates(
        CommonSpatialPolicy::CellCenteredTpfa,
        false,
        "(1+u*u)*(1[s]*derivative(u)-math.sqrt(time()/1[s]))",
    );
}

#[test]
fn physical_initial_rates_q1_do_not_differentiate_regular_row_forcing() {
    // Authored weak forms refer to the existing prescribed-Field owner. This
    // exercises the same sqrt(t) forcing without extending their closed syntax.
    for mixed in [false, true] {
        check_initial_zero_rates_with_data(
            CommonSpatialPolicy::Q1,
            mixed,
            "(1+u*u)*(1[s]*derivative(u)-forcing)",
            "variable forcing:1 on body; relation forcing_data on body {forcing=math.sqrt(time()/1[s]);}",
        );
    }
}

#[test]
fn physical_initial_rates_do_not_differentiate_affine_forcing() {
    check_initial_zero_rates(
        CommonSpatialPolicy::CellCenteredTpfa,
        false,
        "1[s]*derivative(u)-math.sqrt(time()/1[s])",
    );
}

fn check_initial_zero_rates(spatial: CommonSpatialPolicy, mixed: bool, dynamic: &str) {
    check_initial_zero_rates_with_data(spatial, mixed, dynamic, "");
}

fn check_initial_zero_rates_with_data(
    spatial: CommonSpatialPolicy,
    mixed: bool,
    dynamic: &str,
    data: &str,
) {
    let rates = initial_rate_result(spatial, mixed, "z-u*u", dynamic, data).unwrap();
    assert!(rates.iter().all(|rate| rate.abs() < 1e-12));
}

fn initial_rate_result(
    spatial: CommonSpatialPolicy,
    mixed: bool,
    constraint: &str,
    dynamic: &str,
    data: &str,
) -> Result<Vec<f64>, eqiora_core::Diagnostic> {
    let (resolved, _) =
        resolve_typed_constraint_with_data(spatial, mixed, constraint, false, dynamic, data);
    let plan = resolved.as_spatial().unwrap();
    let RecognizedNativeModel::Spatial(equations) = plan.admission.recognized_model() else {
        unreachable!()
    };
    let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
    let initial = vec![1.; system.dimension()];
    system.initial_rates(0., &initial, &REFERENCE_LINEAR_SOLVER)
}

#[test]
fn physical_initial_rates_project_coefficients_before_differentiation() {
    for (spatial, mixed) in [
        (CommonSpatialPolicy::Q1, false),
        (CommonSpatialPolicy::Q1, true),
        (CommonSpatialPolicy::CellCenteredTpfa, false),
    ] {
        for forcing in [
            "math.sqrt(time()/1[s])",
            "coordinate(0)/1[m]*math.sqrt(time()/1[s])",
        ] {
            let data = format!(
                "variable forcing:1 on body; relation forcing_data on body {{forcing={forcing};}}"
            );
            // u'=sqrt(t)*g(x), z=u² at u=z=1 has both initial rates zero.
            check_initial_zero_rates_with_data(spatial, mixed, "1[s]*derivative(u)-forcing", &data);
            // a'(0) need not exist when its entire action a'(0)*u'(0) is zero.
            check_initial_zero_rates_with_data(
                spatial,
                mixed,
                "(1+forcing)*1[s]*derivative(u)",
                &data,
            );
            // The prescribed forcing and nonlinear u²*forcing actions cancel
            // only after their complete equation actions have been assembled.
            let rates = initial_rate_result(
                spatial,
                mixed,
                "z-u*u+forcing-u*u*forcing",
                "1[s]*derivative(u)",
                &data,
            )
            .unwrap();
            assert!(rates.iter().all(|rate| rate.abs() < 1e-12));
        }
        // With u'=0 and z=u²+t the algebraic rate is exactly one; the Time
        // partial must be retained when it lies outside the differential range.
        let rates = initial_rate_result(
            spatial,
            mixed,
            "z-u*u-time()/1[s]",
            "1[s]*derivative(u)",
            "",
        )
        .unwrap();
        assert_eq!(
            rates.iter().filter(|rate| rate.abs() < 1e-12).count() * 2,
            rates.len()
        );
        assert_eq!(
            rates
                .iter()
                .filter(|rate| (**rate - 1.).abs() < 1e-12)
                .count()
                * 2,
            rates.len()
        );
        // z=u²-sqrt(t) genuinely has no finite initial algebraic rate.
        let error = initial_rate_result(spatial, mixed, "z-u*u+forcing", "1[s]*derivative(u)",
            "variable forcing:1 on body; relation forcing_data on body {forcing=math.sqrt(time()/1[s]);}").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("scalar derivative requires a positive argument"),
            "{error}"
        );
    }
}

#[test]
fn physical_initial_rates_complex_coefficient_actions_preserve_both_parts() {
    for (spatial, mixed) in [
        (CommonSpatialPolicy::Q1, false),
        (CommonSpatialPolicy::Q1, true),
        (CommonSpatialPolicy::CellCenteredTpfa, false),
    ] {
        let (resolved, fields) = resolve_typed_constraint_with_data(
            spatial,
            mixed,
            "z-u*u-math.complex(4,5)*time()/1[s]",
            true,
            "1[s]*derivative(u)-math.complex(2,3)*forcing",
            "variable forcing:1 on body; relation forcing_data on body {forcing=math.sqrt(time()/1[s]);}",
        );
        let plan = resolved.as_spatial().unwrap();
        let q1 = spatial == CommonSpatialPolicy::Q1;
        let count = if q1 { 3 } else { 2 };
        let u = C::new(1., 2.);
        let assignments = plan
            .fields()
            .map(|(field, ty)| {
                let value = if field.erase() == fields[0] { u } else { u * u };
                let values =
                    CommonInitialValues::new(ty.shape().clone(), vec![value; count]).unwrap();
                CommonInitialField::new(
                    resolved.model_artifact().digest().unwrap(),
                    field,
                    q1.then(|| values.clone()),
                    (!q1).then_some(values),
                )
                .unwrap()
            })
            .collect();
        let initial = plan.initial_state(Some(0.), assignments).unwrap();
        let before = initial.to_bytes().unwrap();
        let RecognizedNativeModel::ComplexSpatial(equations) = plan.admission.recognized_model()
        else {
            unreachable!()
        };
        let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
        let rates = system
            .initial_rates(
                0.,
                initial.spatial_values().unwrap(),
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
        // u'(0)=0 and z'(0)=2u(0)u'(0)+(4+5i)=4+5i.
        assert_eq!(
            rates
                .as_chunks::<2>()
                .0
                .iter()
                .filter(|pair| pair[0].abs() < 1e-11 && pair[1].abs() < 1e-11)
                .count(),
            count
        );
        assert_eq!(
            rates
                .as_chunks::<2>()
                .0
                .iter()
                .filter(|pair| (pair[0] - 4.).abs() < 1e-11 && (pair[1] - 5.).abs() < 1e-11)
                .count(),
            count
        );
        assert_eq!(initial.to_bytes().unwrap(), before);
    }
}
