use super::super::run::ProcessMethod;
use super::*;

#[test]
fn rejected_wiener_trials_do_not_advance_or_replace_the_accepted_path() {
    let source = r#"model Process() {
        noise W: wiener(covariance=1);
        state x: 1;
        state y: 1;
        initial { x=1; y=-2; }
        law first for x {
            calculus ito;
            drift -x * 2[1/s];
            diffusion W=1[1/s^(1/2)];
        }
        law second for y {
            calculus ito;
            drift x * 4[1/s];
            diffusion W=-2[1/s^(1/2)];
        }
    }"#;
    let kernel = compile_process(source);
    let program = program::ProcessProgram::new(&kernel).unwrap();
    let initial = program.initialize(2.).unwrap();
    let id = identity(1, b"execution", &[]);
    let request = process_run(
        &kernel,
        initial.clone(),
        &[7; 32],
        &id,
        1.,
        ProcessMethod::EulerMaruyama { depth: 8 },
    )
    .unwrap();
    assert!(
        request
            .start()
            .trial(&BigRational::new(1.into(), 3.into()))
            .is_err()
    );
    let mut plain = request.start();
    let mut retried = request.start();
    for (tick, extra) in [(64, 32), (128, 96), (256, 192)] {
        let before = retried.clone();
        drop(retried.trial(&exact(1.)).unwrap());
        drop(retried.trial(&(exact(extra as f64) / exact(256.))).unwrap());
        assert_eq!(retried, before);
        plain
            .accept(plain.trial(&(exact(tick as f64) / exact(256.))).unwrap())
            .unwrap();
        retried
            .accept(retried.trial(&(exact(tick as f64) / exact(256.))).unwrap())
            .unwrap();
        assert_eq!(plain, retried);
    }
    assert_eq!(plain.state().time_s(), 3.);
    assert_ne!(plain.state().values(), initial.values());
    // d(y+2x)=0 follows directly from both authored drift and diffusion,
    // independently of the random draws. Both coefficients use the same left State.
    let coordinates = initial.values();
    let x = coordinates.iter().position(|value| *value == 1.).unwrap();
    let y = coordinates.iter().position(|value| *value == -2.).unwrap();
    assert_eq!(
        plain.state().values()[y] + 2. * plain.state().values()[x],
        0.
    );

    let mut cursor = request.start();
    let stale = cursor.trial(&exact(1.)).unwrap();
    cursor.accept(cursor.trial(&exact(0.25)).unwrap()).unwrap();
    let before = cursor.clone();
    assert!(cursor.accept(stale).is_err());
    assert_eq!(cursor, before);
    let foreign = process_run(
        &kernel,
        initial,
        &[8; 32],
        &id,
        1.,
        ProcessMethod::EulerMaruyama { depth: 8 },
    )
    .unwrap();
    let mut cursor = request.start();
    assert!(
        cursor
            .accept(foreign.start().trial(&exact(0.25)).unwrap())
            .is_err()
    );
    assert_eq!(cursor, request.start());

    let stratonovich_kernel =
        compile_process(&source.replace("calculus ito", "calculus stratonovich"));
    let stratonovich = program::ProcessProgram::new(&stratonovich_kernel).unwrap();
    let state = stratonovich.initialize(0.).unwrap();
    assert!(
        process_run(
            &stratonovich_kernel,
            state,
            &[7; 32],
            &id,
            1.,
            ProcessMethod::EulerMaruyama { depth: 8 }
        )
        .is_err()
    );
}

#[test]
fn failed_wiener_trial_keeps_the_last_physical_and_stochastic_boundary() {
    let kernel = compile_process(
        r#"model Process() {
        noise W: wiener(covariance=1);
        state x: 1;
        initial { x=0; }
        law dynamics for x {
            calculus ito;
            drift 1e308[1/s];
            diffusion W=1[1/s^(1/2)];
        }
    }"#,
    );
    let program = program::ProcessProgram::new(&kernel).unwrap();
    let state = program.initialize(0.).unwrap();
    let request = process_run(
        &kernel,
        state,
        &[7; 32],
        &identity(1, b"overflow", &[]),
        2.,
        ProcessMethod::EulerMaruyama { depth: 8 },
    )
    .unwrap();
    let mut cursor = request.start();
    let before = cursor.clone();
    assert!(cursor.trial(&exact(2.)).is_err());
    assert_eq!(cursor, before);
    cursor.accept(cursor.trial(&exact(1.)).unwrap()).unwrap();
    assert_eq!(cursor.state().time_s(), 1.);
    assert_eq!(cursor.state().values(), [1e308]);
    let before = cursor.clone();
    assert!(cursor.trial(&exact(2.)).is_err());
    assert_eq!(cursor, before);
}

#[test]
fn constant_coefficient_run_matches_independently_derived_physical_moments() {
    let kernel = compile_process(
        r#"model Process() {
        noise W: wiener(covariance=0.25);
        state x: 1;
        initial { x=1; }
        law dynamics for x {
            calculus ito;
            drift 3[1/s];
            diffusion W=2[1/s^(1/2)];
        }
    }"#,
    );
    let program = program::ProcessProgram::new(&kernel).unwrap();
    let state = program.initialize(0.).unwrap();
    // X(2) = 1 + 3*2 + 2 W(2), with Var(W(2))=0.25*2.
    // Thus E[X]=7 and E[(X-7)^2]=2. Predeclare six-standard-error
    // bounds: SE(mean)=sqrt(2/N), SE(centered second moment)=sqrt(8/N).
    const N: u64 = 4096;
    let mut mean = 0.;
    let mut centered_second = 0.;
    for sample in 0..N {
        let request = process_run(
            &kernel,
            state.clone(),
            &[7; 32],
            &identity(sample, b"physical-evidence", &[]),
            2.,
            ProcessMethod::EulerMaruyama { depth: 8 },
        )
        .unwrap();
        let mut cursor = request.start();
        cursor.accept(cursor.trial(&exact(2.)).unwrap()).unwrap();
        let value = cursor.state().values()[0];
        mean += value;
        centered_second += (value - 7.).powi(2);
    }
    let n = N as f64;
    assert!((mean / n - 7.).abs() <= 6. * (2. / n).sqrt());
    assert!((centered_second / n - 2.).abs() <= 6. * (8. / n).sqrt());
}
