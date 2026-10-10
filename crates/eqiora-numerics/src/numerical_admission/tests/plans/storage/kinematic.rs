use super::*;

#[test]
fn common_linear_state_retains_displacement_and_rate_across_replay() {
    let source = "model Wave() {
        domain body=box(0,1);
        domain left=boundary(body,axis=0,side=lower);
        domain right=boundary(body,axis=0,side=upper);
        variable g:m on body in smooth;
        relation potential on body {g=4*coordinate(0)*(1[m]-coordinate(0))/1[m];}
        state d:m on body in smooth;
        state v:m/s on body in smooth;
        relation kinematics on body {derivative(d)=v;}
        relation momentum on body {derivative(v)=1[m^2/s^2]*div(grad(d));}
        initial {d=g;v=0;}
        relation fixed_left on left {trace(v)=0;}
        relation fixed_right on right {trace(v)=0;}
    }";
    let (transaction, id, symbols) = eqiora_compiler::compile("wave.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let d = symbols.get("d").unwrap();
    let v = symbols.get("v").unwrap();
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
    assert_eq!(plan.fields().len(), 2);
    let graph = plan.portable_realization();
    assert_eq!(graph.systems()[0].blocks().len(), 1);
    assert!(graph.transformations().iter().any(|node| match node {
        eqiora_realization::TransformationNode::BackwardEulerElimination {
            state,
            rate,
            duration,
            ..
        } => {
            graph.field(*state).unwrap().field().erase() == d
                && graph.field(*rate).unwrap().field().erase() == v
                && duration.value() == 0.25
        }
        _ => false,
    }));
    for field in [d, v] {
        assert_eq!(
            plan.field_coefficient_entities(field.downcast().unwrap())
                .unwrap()
                .len(),
            3
        );
    }
    let mut state = plan.initial_state(0., vec![]).unwrap();
    let mut displacement = 1.;
    let mut rate = 0.;
    // Two half-length P1 elements give M_ii=1/3, K_ii=4, hence lambda=12.
    // Solve BE kinematics and momentum independently: d+=(d+dt*v)/(1+12dt²).
    for step in 1..=3 {
        let next_d = (displacement + 0.25 * rate) / 1.75;
        let next_v = rate - 3. * next_d;
        displacement = next_d;
        rate = next_v;
        state = plan
            .advance_scalar(&state, &REFERENCE_LINEAR_SOLVER, 0.25 * step as f64)
            .unwrap();
        state = CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap();
        let values = state.linear_values().unwrap();
        for (index, (field, _)) in plan.fields().enumerate() {
            let expected = if field.erase() == d {
                displacement
            } else {
                rate
            };
            assert_eq!(values[3 * index], 0.);
            assert!((values[3 * index + 1] - expected).abs() < 1e-12);
            assert_eq!(values[3 * index + 2], 0.);
        }
    }
}
