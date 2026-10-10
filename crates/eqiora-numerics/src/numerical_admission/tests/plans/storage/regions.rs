use super::*;

fn resolve(discontinuous: bool) -> ResolvedCommonPlan {
    let mut source = String::from("model HeatRegions() {");
    for index in 0..3 {
        let factor = if discontinuous && index == 1 { 2 } else { 1 };
        source += &format!(
            "domain body{index}=box({index},{});
             domain left{index}=boundary(body{index},axis=0,side=lower);
             domain right{index}=boundary(body{index},axis=0,side=upper);
             state u{index}:1 on body{index} in smooth;
             coordinate x{index}:m on body{index} from body{index}[0];
             initial {{ u{index}={factor}*x{index}*(3[m]-x{index})/1[m^2]; }}
             law balance{index} on body{index} {{ storage 1[s/m^2]*u{index};
                 flux -grad(u{index}); source 0[1/m^2]; }}",
            index + 1,
        );
    }
    source += "relation fixed_left on left0 { trace(u0)=0; }
               relation fixed_right on right2 { trace(u2)=0; }";
    for index in 0..2 {
        let next = index + 1;
        source += &format!(
            "domain contact{index}=interface(right{index},left{next});
             relation continuity{index} on contact{index} {{ trace(u{index})=trace(u{next}); }}
             relation balance_interface{index} on contact{index} {{ normal(grad(u{index}))=normal(grad(u{next})); }}"
        );
    }
    source += "}";
    let (transaction, model, _) = eqiora_compiler::compile("heat-regions.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let graph = GeometryGraph::new();
    let interval = graph.interval([0., 3.]).unwrap();
    let geometry = graph
        .build(
            &interval,
            &BTreeMap::from([
                ("body".to_owned(), vec![interval.region().into()]),
                ("left".to_owned(), vec![interval.boundaries()[0].into()]),
                ("right".to_owned(), vec![interval.boundaries()[1].into()]),
            ]),
        )
        .unwrap();
    ResolvedCommonPlan::resolve(
        &ModelEnvelope::from_program(&program).unwrap(),
        cartesian_box_resources(&geometry, &[3]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )),
        None,
        Some(CommonBackwardEuler::from_seconds(1.).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap()
}

#[test]
fn three_regions_share_transient_interface_coordinates() {
    let resolved = replay_plan(resolve(false), &REFERENCE_LINEAR_SOLVER);
    let initial = resolved.as_linear().unwrap().initial_state().unwrap();
    let initial = CommonState::from_bytes(&initial.to_bytes().unwrap(), &resolved).unwrap();
    assert_eq!(initial.scalar_values().unwrap().len(), 6);
    let baseline = initial.scalar_values().unwrap().to_vec();
    let mut initial_values = baseline.clone();
    initial_values.sort_by(f64::total_cmp);
    // x(3-x) at each Region's two endpoints, including shared endpoints twice.
    assert_eq!(initial_values, [0., 0., 2., 2., 2., 2.]);
    // Three unit elements, two free nodes: M has diagonal 2/3 and off-diagonal
    // 1/6; K has diagonal 2 and off-diagonal -1. The equal-interior mode
    // therefore decays by (5/6)/(5/6+1)=5/11 for dt=1.
    let run = CommonTransientRunRequest::from_steps(resolved, initial, 3, vec![1, 2, 3]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("three steps must complete")
    };
    for (step, (_, state)) in (1..=3).zip(outputs) {
        for (&actual, &start) in state.scalar_values().unwrap().iter().zip(&baseline) {
            assert!((actual - start * (5.0_f64 / 11.0).powi(step)).abs() < 1e-12);
        }
    }
    assert!(
        resolve(true)
            .as_linear()
            .unwrap()
            .initial_state()
            .unwrap_err()
            .message()
            .contains("trace quotient")
    );
}
