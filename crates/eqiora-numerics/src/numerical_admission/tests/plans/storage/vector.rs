use super::*;

#[test]
fn vector_storage_retains_each_component_through_steps_and_replay() {
    let source = r#"model VectorStorage() {
        domain region=box(0,1,0,1);
        domain left=boundary(region,axis=0,side=lower);
        domain right=boundary(region,axis=0,side=upper);
        domain bottom=boundary(region,axis=1,side=lower);
        domain top=boundary(region,axis=1,side=upper);
        variable g:m on region in smooth;
        relation potential on region {
            g=256[m]*(2*coordinate(0)/1[m]+3*coordinate(1)/1[m]-2.5)
                *(coordinate(0)/1[m])^2*(1-coordinate(0)/1[m])^2
                *(coordinate(1)/1[m])^2*(1-coordinate(1)/1[m])^2;
        }
        observable profile:vector<1,2> on region=u;
        observable total:m^2=integral(component(u,indices=(0,)),measure(region));
        state u:vector<1,2> on region in smooth;
        initial { u=grad(g); }
        relation balance on region { 1[s/m^2]*derivative(u)=div(grad(u)); }
        relation fixed_left on left { trace(u)=0; }
        relation fixed_right on right { trace(u)=0; }
        relation fixed_bottom on bottom { trace(u)=0; }
        relation fixed_top on top { trace(u)=0; }
    }"#;
    let (transaction, model, _) = eqiora_compiler::compile("vector-storage.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
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
        Some(CommonBackwardEuler::from_seconds(1. / 24.).unwrap().into()),
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
    )
    .unwrap();
    let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
    let initial = resolved
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.0), Vec::new())
        .unwrap();
    let initial = CommonState::from_bytes(&initial.to_bytes().unwrap(), &resolved).unwrap();
    let mut wire: serde_json::Value = serde_json::from_slice(&initial.to_bytes().unwrap()).unwrap();
    wire["schema"] = "eqiora.common-spatial-state/v1".into();
    assert!(
        CommonState::from_bytes(&serde_json::to_vec(&wire).unwrap(), &resolved)
            .unwrap_err()
            .message()
            .contains("unknown schema")
    );

    // The potential's gradient is (2,3) at the center and zero at all boundary nodes.
    let mut expected = [0.; 18];
    expected[8] = 2.;
    expected[9] = 3.;
    assert_eq!(initial.spatial_values().unwrap(), expected);
    let fields = resolved
        .as_spatial()
        .unwrap()
        .state_fields(&initial)
        .unwrap()
        .collect::<Vec<_>>();
    let [(_, value_type, entities, values)] = fields.as_slice() else {
        panic!("one represented vector State Field required");
    };
    assert_eq!(value_type.shape().component_count(), Some(2));
    assert_eq!(
        entities,
        &(0..9)
            .map(|index| eqiora_meshing::MeshEntity::new(0, index))
            .collect::<Vec<_>>()
    );
    assert_eq!(*values, expected);
    // Four Q1 squares: M_ii=1/9 and K_ii=8/3 per component. With dt=1/24,
    // M/(M+dt*K)=1/2, independently of the assembled operator.
    crate::region_assembly::InterfaceReactions::<f64>::reset_evaluations();
    let run =
        CommonTransientRunRequest::from_steps(resolved.clone(), initial, 3, vec![1, 2, 3]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("three accepted vector steps required")
    };
    assert_eq!(
        crate::region_assembly::InterfaceReactions::<f64>::evaluations(),
        0
    );
    let trajectory =
        crate::CommonTrajectory::accept_spatial_transient(run, outputs.clone()).unwrap();
    let result = crate::CommonResult::accept_trajectory(0., trajectory).unwrap();
    for (step, (_, state)) in (1..=3).zip(outputs) {
        let state = state.state();
        for definition in resolved
            .as_spatial()
            .unwrap()
            .kernel()
            .nodes()
            .filter_map(|node| match node {
                eqiora_schema::kernel::KernelNode::Observable(value) => Some(value),
                _ => None,
            })
        {
            let index = usize::try_from(step - 1).unwrap();
            let observation = if definition.value_type().shape().is_scalar() {
                let rules = result
                    .plan()
                    .observation_quadratures(&[definition.id()], 2)
                    .unwrap();
                let observation = result
                    .observe(
                        resolved.model_artifact(),
                        definition.id(),
                        &rules,
                        Some(index),
                    )
                    .unwrap();
                // The Cartesian center hat has integral 1/4 on the unit square.
                assert!(
                    (observation.value().real_scalar_value().unwrap().value()
                        - 0.5 * 0.5_f64.powi(step))
                    .abs()
                        < 1e-12
                );
                observation
            } else {
                let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
                let observation = result
                    .observe_at(
                        resolved.model_artifact(),
                        definition.id(),
                        &[DynQuantity::new(0.5, length); 2],
                        &std::collections::HashMap::new(),
                        Some(index),
                    )
                    .unwrap();
                for (component, value) in [2., 3.].into_iter().enumerate() {
                    assert!(
                        (observation.value().component(component).unwrap().0
                            - value * 0.5_f64.powi(step))
                        .abs()
                            < 1e-12
                    );
                }
                observation
            };
            assert_eq!(
                observation.output(),
                Some((index, state.time_s().unwrap(), state.identity()))
            );
        }
        for (&actual, expected) in state.spatial_values().unwrap().iter().zip(expected) {
            assert!((actual - expected * 0.5_f64.powi(step)).abs() < 1e-12);
        }
    }
}
