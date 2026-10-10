use super::*;

#[test]
fn algebraic_snapshot_uses_current_displacement_with_nonzero_rate() {
    let source = r#"
public model Constrained(
    support body: volume(ambient_dimension = 1),
    support left: boundary(parent = body),
    support right: boundary(parent = body)
) {
    coordinate x:m on body from body[0];
    state w:m on body in h1;
    state u:m/s on body in h1;
    state v:m on body in h1;
    initial {
        w=4*x*(1[m]-x)/1[m];
        u=4*x*(1[m]-x)/1[m*s];
        v=(1/13)*4*x*(1[m]-x)/1[m];
    }
    relation kinematic on body {derivative(w)=u;}
    relation evolution on body {1[s^2/m^2]*derivative(u)=div(grad(w))-v*1[1/m^2];}
    relation constraint on body {-div(grad(v))+(v-w)*1[1/m^2]=0;}
    relation left_values on left {trace(u)=0; trace(v)=0;}
    relation right_values on right {trace(u)=0; trace(v)=0;}
}
"#;
    let (resolved, fields) = coupled::resolve(source, "Constrained").unwrap();
    let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
    let plan = resolved.as_linear().unwrap();
    let indices = fields.map(|field| {
        3 * plan
            .fields()
            .position(|(id, _)| id.erase() == field)
            .unwrap()
            + 1
    });
    let initial = plan.initial_state(0., Vec::new()).unwrap();
    let initial = CommonState::from_bytes(&initial.to_bytes().unwrap(), &resolved).unwrap();
    let run =
        CommonTransientRunRequest::from_steps(resolved.clone(), initial, 3, vec![1, 2, 3]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("complete run")
    };
    // Mii=1/3 and Kii=4 imply v=w/13 and u' = -(12+1/13)w.
    // Invert the two-by-two Backward Euler equations, independently of assembly.
    let (mut displacement, mut velocity) = (1., 1.);
    let step = 0.25;
    let lambda = 157. / 13.;
    for (_, state) in outputs {
        displacement = (displacement + step * velocity) / (1. + step * step * lambda);
        velocity -= step * lambda * displacement;
        for (index, expected) in
            indices
                .into_iter()
                .zip([velocity, displacement / 13., displacement])
        {
            assert!((state.linear_values().unwrap()[index] - expected).abs() < 1e-11);
        }
        assert_eq!(
            CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap(),
            state
        );
    }
}
