use super::*;

#[test]
fn simplicial_displacement_trace_uses_previous_physical_state() {
    let geometry = geometry(false);
    let body = geometry.entity_set("body").unwrap();
    let model = compile_model(
        "simplicial-driven-wave.eqi",
        r#"
public model DrivenWave(
    support body: volume(ambient_dimension=2),
    support outer: boundary(parent=body)
) {
    state d:m on body in smooth;
    state v:m/s on body in smooth;
    relation kinematics on body {derivative(d)=v;}
    relation momentum on body {derivative(v)=1[m^2/s^2]*div(grad(d));}
    initial {d=0;v=0;}
    relation driven on outer {trace(d)=1[m/s^2]*time()*time();}
}
"#,
        &geometry,
        "DrivenWave",
        &[
            ("body", body, None),
            (
                "outer",
                geometry.entity_set("outer").unwrap(),
                Some(("body", body)),
            ),
        ],
        &[],
    );
    for permuted in [false, true] {
        let resolved = ResolvedCommonPlan::resolve(
            &model,
            resources(&geometry, permuted),
            CommonSpatialPolicy::P1,
            CommonSolvePolicy::Linear(exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-12,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
            )),
            None,
            Some(CommonBackwardEuler::from_seconds(0.25).unwrap().into()),
            &REFERENCE_LINEAR_SOLVER,
            None,
            None,
        )
        .unwrap();
        let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
        let plan = resolved.as_spatial().unwrap();
        let mut state = plan.initial_state(Some(0.), vec![]).unwrap();
        let (mut d, mut v, mut boundary_d, mut boundary_v) = (0., 0., 0., 0.);
        // Four area-1/4 triangles give Mcc=1/6, sum(Mcb)=1/6,
        // Kcc=4, sum(Kcb)=-4. The consistent boundary mass contributes
        // -(vb_next-vb)/6 to the interior momentum equation.
        for step in 1..=3 {
            let dt = 0.25;
            let time = step as f64 * dt;
            let next_boundary_d = time * time;
            let next_boundary_v = (next_boundary_d - boundary_d) / dt;
            v = (v / 6. - 4. * dt * d - (next_boundary_v - boundary_v) / 6.
                + 4. * dt * next_boundary_d)
                / (1. / 6. + 4. * dt * dt);
            d += dt * v;
            boundary_d = next_boundary_d;
            boundary_v = next_boundary_v;
            let retained = state.to_bytes().unwrap();
            let next = plan
                .advance_scalar(&state, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
            assert_eq!(state.to_bytes().unwrap(), retained);
            state = CommonState::from_bytes(&next.state().to_bytes().unwrap(), &resolved).unwrap();
            for (index, (_, ty)) in plan.fields().enumerate() {
                let (boundary, interior) = if ty.dimension()
                    == DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap()
                {
                    (boundary_d, d)
                } else {
                    (boundary_v, v)
                };
                let values = &state.spatial_values().unwrap()[5 * index..5 * index + 5];
                for value in &values[..4] {
                    assert!((value - boundary).abs() < 1e-12);
                }
                assert!((values[4] - interior).abs() < 1e-12);
            }
        }
    }
}
