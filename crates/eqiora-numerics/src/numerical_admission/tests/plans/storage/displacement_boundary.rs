use super::*;

#[test]
fn displacement_boundary_uses_discrete_history_and_consistent_mass() {
    let source = "model DrivenWave() {
        domain body=box(0,1);
        domain left=boundary(body,axis=0,side=lower);
        domain right=boundary(body,axis=0,side=upper);
        state d:m on body in smooth;
        state v:m/s on body in smooth;
        relation kinematics on body {derivative(d)=v;}
        relation momentum on body {derivative(v)=1[m^2/s^2]*div(grad(d));}
        initial {d=0;v=0;}
        relation driven_left on left {trace(d)=1[m/s^2]*time()*time();}
        relation driven_right on right {trace(d)=1[m/s^2]*time()*time();}
    }";
    let (transaction, id, symbols) = eqiora_compiler::compile("driven-wave.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let d = symbols.get("d").unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), id).unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&cartesian_interval(), &[2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )),
        None,
        Some(CommonBackwardEuler::from_seconds(0.25).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
    let plan = resolved.as_linear().unwrap();
    let mut state = plan.initial_state(0., vec![]).unwrap();
    // Independent two-element consistent matrices: Mii=1/3, each Mib=1/12,
    // Kii=4, each Kib=-2. Endpoint g=t^2 gives vb+=(g+-g)/dt.
    // (Mii+dt^2*Kii)d+=Mii*d+dt*Mii*v-2*dt*Mib*(vb+-vb)-2*dt^2*Kib*g+.
    // These exact rational values include the boundary capacity-history term.
    for (index, (boundary_d, boundary_v, interior_d, interior_v)) in [
        (1. / 16., 1. / 4., 1. / 112., 1. / 28.),
        (1. / 4., 3. / 4., 4. / 49., 57. / 196.),
        (9. / 16., 5. / 4., 1611. / 5488., 1163. / 1372.),
    ]
    .into_iter()
    .enumerate()
    {
        let old = state.to_bytes().unwrap();
        let next = plan
            .advance_scalar(&state, &REFERENCE_LINEAR_SOLVER, (index + 1) as f64 * 0.25)
            .unwrap();
        assert_eq!(state.to_bytes().unwrap(), old);
        state = CommonState::from_bytes(&next.to_bytes().unwrap(), &resolved).unwrap();
        let assignments = |bad_boundary: bool| {
            plan.fields()
                .enumerate()
                .map(|(field_index, (field, _))| {
                    let mut values = state.linear_values().unwrap()
                        [3 * field_index..3 * field_index + 3]
                        .to_vec();
                    if bad_boundary && field.erase() == d {
                        values[0] += 1.;
                    }
                    CommonInitialField::new(
                        model.digest().unwrap(),
                        field,
                        Some(
                            CommonInitialValues::new(eqiora_core::ValueShape::scalar(), values)
                                .unwrap(),
                        ),
                        None,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let valid = assignments(false);
        assert!(
            plan.initial_state(state.time_s(), assignments(true))
                .unwrap_err()
                .message()
                .contains("prescribed displacement boundary")
        );
        let restarted = plan.initial_state(state.time_s(), valid).unwrap();
        assert_eq!(restarted.to_bytes().unwrap(), state.to_bytes().unwrap());
        state = restarted;
        let values = state.linear_values().unwrap();
        for (field_index, (field, _)) in plan.fields().enumerate() {
            let (boundary, interior) = if field.erase() == d {
                (boundary_d, interior_d)
            } else {
                (boundary_v, interior_v)
            };
            assert_eq!(values[3 * field_index], boundary);
            assert!((values[3 * field_index + 1] - interior).abs() < 1e-12);
            assert_eq!(values[3 * field_index + 2], boundary);
        }
    }
}

#[test]
fn displacement_and_rate_boundaries_meet_without_surrogate_corner_constraints() {
    let source = "model MixedBoundaryWave() {
        domain region=box(0,1,0,1);
        domain left=boundary(region,axis=0,side=lower);
        domain right=boundary(region,axis=0,side=upper);
        domain bottom=boundary(region,axis=1,side=lower);
        domain top=boundary(region,axis=1,side=upper);
        state d:m on region in smooth;
        state v:m/s on region in smooth;
        relation pair on region {derivative(d)=v;}
        relation balance on region {derivative(v)=1[m^2/s^2]*div(grad(d));}
        initial {d=0;v=1[m/s];}
        relation left_value on left {trace(d)=1[m/s]*time();}
        relation right_value on right {trace(d)=1[m/s]*time();}
        relation bottom_value on bottom {trace(v)=1[m/s];}
        relation top_value on top {trace(v)=1[m/s];}
    }";
    let (transaction, model, symbols) = eqiora_compiler::compile("mixed-boundary.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let d = symbols.get("d").unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let geometry = CanonicalGeometryV1::decode_cartesian_box_v1_canonical(
        br#"{"schema":"eqiora.cartesian-box-envelope/v1","encoding":"eqiora.canonical-json/v1","length_unit":"metre","bounds":[[0.0,1.0],[0.0,1.0]],"entity_sets":[{"name":"bottom","dimension":1,"members":[2]},{"name":"left","dimension":1,"members":[0]},{"name":"right","dimension":1,"members":[1]},{"name":"top","dimension":1,"members":[3]},{"name":"region","dimension":2,"members":[0]}]}"#,
        eqiora_geometry::CanonicalGeometryLimits::default(),
    ).unwrap();

    let resolved = ResolvedCommonPlan::resolve(
        &ModelEnvelope::from_program(&program).unwrap(),
        cartesian_box_resources(&geometry, &[2, 2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )),
        None,
        Some(CommonBackwardEuler::from_seconds(0.25).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let plan = resolved.as_linear().unwrap();
    let mut state = plan.initial_state(0., vec![]).unwrap();
    // Uniform d=t, v=1 has zero spatial gradient and zero acceleration.
    // Both boundary laws agree at every corner, including the initial state.
    for step in 1..=3 {
        let t = step as f64 * 0.25;
        state = plan
            .advance_scalar(&state, &REFERENCE_LINEAR_SOLVER, t)
            .unwrap();
        state = CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap();
        for (index, (field, _)) in plan.fields().enumerate() {
            let expected = if field.erase() == d { t } else { 1. };
            for value in &state.linear_values().unwrap()[9 * index..9 * index + 9] {
                assert!((value - expected).abs() < 1e-12);
            }
        }
    }
}
