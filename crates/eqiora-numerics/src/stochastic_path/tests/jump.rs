use super::super::realization::RealizedNoise;
use super::super::run::ProcessMethod;
use super::*;
use num_traits::ToPrimitive;

#[test]
fn poisson_events_apply_state_dependent_jumps_once_per_arrival_under_rejection() {
    let kernel = compile_process(
        r#"model Process() {
        noise N: poisson(intensity=8[1/s]);
        state x: 1;
        state y: 1;
        initial { x=1; y=0; }
        law first for x { jump N=x; }
        law second for y { jump N=1; }
    }"#,
    );
    let program = program::ProcessProgram::new(&kernel).unwrap();
    let initial = program.initialize(2.).unwrap();
    let x = initial
        .values()
        .iter()
        .position(|value| *value == 1.)
        .unwrap();
    let y = initial
        .values()
        .iter()
        .position(|value| *value == 0.)
        .unwrap();
    let id = identity(1, b"jump-run", &[]);
    let noise = program.noises[0].realize(&[7; 32], &id, 1., 0, 64).unwrap();
    let count = noise.poisson_increment(&exact(0.), &exact(1.)).unwrap()[0];
    assert!(
        count > 1,
        "fixture must distinguish event stepping from frozen count multiplication"
    );
    let request = process_run(
        &kernel,
        initial,
        &[7; 32],
        &id,
        1.,
        ProcessMethod::PoissonEvents { max_events: 64 },
    )
    .unwrap();
    let mut whole = request.start();
    whole.accept(whole.trial(&exact(1.)).unwrap()).unwrap();
    let mut divided = request.start();
    for elapsed in [0.3, 0.7, 1.] {
        let before = divided.clone();
        drop(divided.trial(&exact(1.)).unwrap());
        assert_eq!(divided, before);
        divided
            .accept(divided.trial(&exact(elapsed)).unwrap())
            .unwrap();
    }
    assert_eq!(whole.state(), divided.state());
    // Each event maps x -> 2x and y -> y+1. This identity is independent
    // of the sampler, event timing, trial partition and source field order.
    assert_eq!(whole.state().values()[x], 2_f64.powi(count as i32));
    assert_eq!(whole.state().values()[y], count as f64);
    assert_eq!(whole.state().time_s(), 3.);

    let RealizedNoise::Poisson(paths) = noise else {
        panic!("Poisson law")
    };
    let first = paths[0].arrivals().next().unwrap();
    let mut boundary = request.start();
    boundary.accept(boundary.trial(&first).unwrap()).unwrap();
    assert_eq!(boundary.state().values()[x], 2.);
    assert_eq!(boundary.state().values()[y], 1.);
    boundary
        .accept(boundary.trial(&exact(1.)).unwrap())
        .unwrap();
    assert_eq!(boundary.state(), whole.state());
}

#[test]
fn jump_coefficients_read_arrival_time_and_the_same_pre_jump_state() {
    let kernel = compile_process(
        r#"model Process() {
        noise N: poisson(intensity=8[1/s]);
        state x: 1;
        state y: 1;
        initial { x=1; y=-2; }
        law first for x { jump N=time()/1[s]; }
        law second for y { jump N=x; }
    }"#,
    );
    let program = program::ProcessProgram::new(&kernel).unwrap();
    let initial = program.initialize(2.).unwrap();
    let id = identity(1, b"timed-jumps", &[]);
    let RealizedNoise::Poisson(paths) =
        program.noises[0].realize(&[7; 32], &id, 1., 0, 64).unwrap()
    else {
        panic!("Poisson law")
    };
    let x = initial
        .values()
        .iter()
        .position(|value| *value == 1.)
        .unwrap();
    let y = initial
        .values()
        .iter()
        .position(|value| *value == -2.)
        .unwrap();
    let mut expected_x = 1.;
    let mut expected_y = -2.;
    for arrival in paths[0].arrivals() {
        expected_y += expected_x;
        expected_x += (exact(2.) + arrival).to_f64().unwrap();
    }
    assert!(expected_x > 0.);
    let request = process_run(
        &kernel,
        initial,
        &[7; 32],
        &id,
        1.,
        ProcessMethod::PoissonEvents { max_events: 64 },
    )
    .unwrap();
    let mut cursor = request.start();
    cursor.accept(cursor.trial(&exact(1.)).unwrap()).unwrap();
    assert_eq!(cursor.state().values()[x], expected_x);
    assert_eq!(cursor.state().values()[y], expected_y);
}

#[test]
fn failed_jump_trials_and_invalid_method_admission_preserve_the_accepted_boundary() {
    let kernel = compile_process(
        r#"model Process() {
        noise N: poisson(intensity=8[1/s]);
        state x: 1;
        initial { x=0; }
        law dynamics for x { jump N=1e308; }
    }"#,
    );
    let program = program::ProcessProgram::new(&kernel).unwrap();
    let initial = program.initialize(0.).unwrap();
    let id = identity(1, b"overflow-jumps", &[]);
    let RealizedNoise::Poisson(paths) =
        program.noises[0].realize(&[7; 32], &id, 1., 0, 64).unwrap()
    else {
        panic!("Poisson law")
    };
    assert!(paths[0].arrivals().count() >= 2);
    let first = paths[0].arrivals().next().unwrap();
    let request = process_run(
        &kernel,
        initial.clone(),
        &[7; 32],
        &id,
        1.,
        ProcessMethod::PoissonEvents { max_events: 64 },
    )
    .unwrap();
    let mut cursor = request.start();
    let before = cursor.clone();
    assert!(cursor.trial(&exact(1.)).is_err());
    assert_eq!(cursor, before);
    cursor.accept(cursor.trial(&first).unwrap()).unwrap();
    assert_eq!(cursor.state().values(), [1e308]);
    let accepted = cursor.clone();
    assert!(cursor.trial(&exact(1.)).is_err());
    assert_eq!(cursor, accepted);
    assert!(
        process_run(
            &kernel,
            initial.clone(),
            &[7; 32],
            &id,
            1.,
            ProcessMethod::PoissonEvents { max_events: 0 }
        )
        .is_err()
    );
    assert!(
        process_run(
            &kernel,
            initial,
            &[7; 32],
            &id,
            1.,
            ProcessMethod::EulerMaruyama { depth: 8 }
        )
        .is_err()
    );
}

#[test]
fn independent_jump_channels_follow_arrival_order_and_one_total_event_budget() {
    let kernel = compile_process(
        r#"model Process() {
        noise N: poisson(intensity=5[1/s], independent=true);
        noise M: poisson(intensity=7[1/s], independent=true);
        state x: 1;
        initial { x=1; }
        law dynamics for x { jump N=1; jump M=x; }
    }"#,
    );
    let program = program::ProcessProgram::new(&kernel).unwrap();
    let state = program.initialize(0.).unwrap();
    let id = identity(1, b"ordered-jumps", &[]);
    let mut additions = Vec::new();
    let mut doublings = Vec::new();
    for (group, coefficient) in &program.laws[0].jumps {
        let noise = program
            .noises
            .iter()
            .find(|noise| noise.id == *group)
            .unwrap();
        let RealizedNoise::Poisson(paths) = noise.realize(&[7; 32], &id, 1., 0, 64).unwrap() else {
            panic!("Poisson law")
        };
        // The constant jump adds one; the other doubles the current State.
        if coefficient.constant() {
            additions = paths[0].arrivals().collect();
        } else {
            doublings = paths[0].arrivals().collect();
        }
    }
    assert!(!additions.is_empty() && !doublings.is_empty());
    let count = additions.len() + doublings.len();
    // x starts at one. Every doubling multiplies that one by two. An
    // additive event contributes one times two for each subsequent doubling.
    // This closed form is independent of the executor's chronological loop.
    let expected = 2_f64.powi(doublings.len() as i32)
        + additions
            .iter()
            .map(|arrival| {
                2_f64.powi(doublings.iter().filter(|later| *later > arrival).count() as i32)
            })
            .sum::<f64>();
    let request = process_run(
        &kernel,
        state.clone(),
        &[7; 32],
        &id,
        1.,
        ProcessMethod::PoissonEvents { max_events: count },
    )
    .unwrap();
    let mut cursor = request.start();
    cursor.accept(cursor.trial(&exact(1.)).unwrap()).unwrap();
    assert_eq!(cursor.state().values(), [expected]);
    assert!(
        process_run(
            &kernel,
            state,
            &[7; 32],
            &id,
            1.,
            ProcessMethod::PoissonEvents {
                max_events: count - 1
            }
        )
        .is_err()
    );
}

#[test]
fn zero_intensity_runs_without_an_event_or_event_budget() {
    let kernel = compile_process(
        r#"model Process() {
        noise N: poisson(intensity=0[1/s]);
        state x: 1;
        initial { x=7; }
        law dynamics for x { jump N=1; }
    }"#,
    );
    let program = program::ProcessProgram::new(&kernel).unwrap();
    let state = program.initialize(0.).unwrap();
    let request = process_run(
        &kernel,
        state,
        &[7; 32],
        &identity(0, b"zero", &[]),
        1.,
        ProcessMethod::PoissonEvents { max_events: 0 },
    )
    .unwrap();
    let mut cursor = request.start();
    cursor.accept(cursor.trial(&exact(1.)).unwrap()).unwrap();
    assert_eq!(cursor.state().values(), [7.]);
    assert_eq!(cursor.state().time_s(), 1.);
}
