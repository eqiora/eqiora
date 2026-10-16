use super::*;
use num_traits::ToPrimitive;

fn identity(sample: u64, channel: &[u8], lineage: &[u64]) -> SamplingIdentity {
    SamplingIdentity::new(&sample.to_be_bytes(), channel, lineage, None).unwrap()
}

fn path(sample: u64) -> WienerPath {
    WienerPath::new(&[17; 32], &identity(sample, b"W", &[]), 1., 1., 8).unwrap()
}

#[test]
fn retained_increments_compose_exactly_under_reordering_and_refinement() {
    let path = path(1);
    let whole = path.increment(0, 256).unwrap();
    let left = path.increment(0, 113).unwrap();
    let right = path.increment(113, 256).unwrap();
    assert_eq!(&left + &right, whole);
    let fine = (0..256)
        .rev()
        .map(|tick| path.increment(tick, tick + 1).unwrap())
        .sum::<BigRational>();
    assert_eq!(fine, whole);
    // A rejected trial cannot consume a generator or replace an endpoint.
    for (from, to) in [(13, 177), (0, 256), (71, 72), (113, 256)] {
        let _rejected = path.increment(from, to).unwrap();
    }
    assert_eq!(path.increment(0, 113).unwrap(), left);
    assert_eq!(path.increment(113, 0).unwrap(), -left);
    assert_eq!(path.increment(19, 19).unwrap(), BigRational::zero());
    assert!(path.increment(0, 257).is_err());
}

#[test]
fn sample_channel_and_explicit_coupling_own_paths_not_occurrence_order() {
    let original = path(3);
    let reordered = WienerPath::new(&[17; 32], &identity(3, b"W", &[8, 2]), 1., 1., 8).unwrap();
    assert_eq!(original, reordered);
    assert_ne!(original, path(4));
    let changed = WienerPath::new(&[17; 32], &identity(3, b"V", &[]), 1., 1., 8).unwrap();
    assert_ne!(original, changed);
    let coupled = |sample| {
        let id = SamplingIdentity::new(&[sample], b"W", &[], Some(b"shared")).unwrap();
        WienerPath::new(&[17; 32], &id, 1., 1., 8).unwrap()
    };
    assert_eq!(coupled(0), coupled(1));
    // Grid refinement preserves every previously admitted endpoint.
    let finer = WienerPath::new(&[17; 32], &identity(3, b"W", &[]), 1., 1., 12).unwrap();
    for tick in [0, 1, 59, 128, 255, 256] {
        assert_eq!(
            original.value(tick).unwrap(),
            finer.value(tick * 16).unwrap()
        );
    }
}

#[test]
fn independently_derived_wiener_and_conditional_bridge_moments() {
    // W(1) ~ N(0,1). Given W(1), Y=W(1/2)-W(1)/2 has mean 0,
    // variance 1/4 and covariance 0 with W(1). Fix N and six-standard-error
    // bounds before sampling: SE(mean)=sigma/sqrt(N),
    // SE(second moment)=sigma^2*sqrt(2/N), SE(E[WY])=1/(2sqrt(N)).
    // This checks these bounded moments, not independence of every hashed stream.
    const N: u64 = 4096;
    let mut sums = [0.; 5];
    for sample in 0..N {
        let path = path(sample);
        let end = path.value(256).unwrap().to_f64().unwrap();
        let midpoint = path.value(128).unwrap().to_f64().unwrap();
        let residual = midpoint - end * 0.5;
        for (sum, value) in sums.iter_mut().zip([
            end,
            end * end,
            residual,
            residual * residual,
            end * residual,
        ]) {
            *sum += value;
        }
    }
    let n = N as f64;
    let expected = [0., 1., 0., 0.25, 0.];
    let errors = [
        6. / n.sqrt(),
        6. * (2. / n).sqrt(),
        3. / n.sqrt(),
        1.5 * (2. / n).sqrt(),
        3. / n.sqrt(),
    ];
    for ((sum, expected), bound) in sums.into_iter().zip(expected).zip(errors) {
        assert!(
            (sum / n - expected).abs() <= bound,
            "moment {sum}/{n}, expected {expected}, bound {bound}"
        );
    }
}

#[test]
fn endpoint_conversion_and_degenerate_law_are_explicit() {
    assert!(open_unit(0) > 0.);
    assert!(open_unit(u64::MAX) < 1.);
    let id = identity(0, b"W", &[]);
    let zero = WienerPath::new(&[0; 32], &id, 0., 1., 52).unwrap();
    assert_eq!(
        zero.increment(0, zero.ticks()).unwrap(),
        BigRational::zero()
    );
    for (variance, duration, depth) in [
        (-1., 1., 1),
        (f64::NAN, 1., 1),
        (1., 0., 1),
        (1., f64::INFINITY, 1),
        (f64::MAX, 2., 1),
        (f64::from_bits(1), 0.5, 1),
        (1., 1., 53),
    ] {
        assert!(WienerPath::new(&[0; 32], &id, variance, duration, depth).is_err());
    }
}

pub(super) fn compile_process(source: &str) -> eqiora_sem::KernelProgram {
    use eqiora_graph::{GraphStore, InMemoryGraphStore};
    let compiled = eqiora_compiler::compile("process.eqi", source)
        .unwrap()
        .pop()
        .unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    eqiora_sem::KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

#[test]
fn canonical_process_initialization_uses_model_owned_initial_conditions() {
    use eqiora_schema::kernel::KernelNode;
    use eqiora_sem::{Interpreter, ReferenceConfig};

    let source = r#"model Process() {
        noise W: wiener(covariance=1, independent=true);
        state x: 1;
        initial { x=2; }
        law dynamics for x {
            calculus ito;
            drift -x * 3[1/s];
            diffusion W=0.5[1/s^(1/2)];
        }
    }"#;
    let kernel = compile_process(source);
    let field = kernel
        .nodes()
        .find_map(|node| match node {
            KernelNode::Field(field) => Some(field.id()),
            _ => None,
        })
        .unwrap();
    let initial = Interpreter::new()
        .initialize(&kernel, 0., ReferenceConfig::new(0., 1.).unwrap())
        .unwrap();
    assert_eq!(
        initial.fields()[&field.erase()]
            .real_scalar_value()
            .unwrap()
            .value(),
        2.
    );
    assert!(initial.derivatives().is_empty());
    assert!(
        Interpreter::new()
            .run(&kernel, ReferenceConfig::new(1., 0.1).unwrap())
            .is_err()
    );
    let missing = compile_process(&source.replace("initial { x=2; }", ""));
    assert!(
        Interpreter::new()
            .initialize(&missing, 0., ReferenceConfig::new(0., 1.).unwrap())
            .is_err(),
        "a stochastic law must not supply an implicit zero initial value"
    );
}

#[test]
fn stochastic_initialization_solves_coupled_conditions_and_rejects_inconsistency() {
    use eqiora_sem::{Interpreter, ReferenceConfig};
    let source = r#"model Process() {
        noise W: wiener(covariance=1, independent=true);
        state x: 1;
        state y: 1;
        initial { x+y=5; x-y=1; }
        law first for x {
            calculus ito;
            drift -x * 3[1/s];
            diffusion W=0.5[1/s^(1/2)];
        }
        law second for y {
            calculus ito;
            drift x * 2[1/s];
            diffusion W=y * 1[1/s^(1/2)];
        }
    }"#;
    let kernel = compile_process(source);
    let config = ReferenceConfig::new(0., 1.).unwrap();
    let initial = Interpreter::new().initialize(&kernel, 0., config).unwrap();
    let mut values = initial
        .fields()
        .values()
        .map(|value| value.real_scalar_value().unwrap().value())
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    assert_eq!(values, [2., 3.]);
    assert!(initial.derivatives().is_empty());
    for conditions in ["x+y=5;", "x+y=5; x-y=1; x=0;"] {
        let bad = compile_process(&source.replace("x+y=5; x-y=1;", conditions));
        assert!(Interpreter::new().initialize(&bad, 0., config).is_err());
    }
}

#[test]
fn process_coefficients_reuse_typed_ir_and_preserve_channel_axes() {
    use super::program::ProcessProgram;
    let kernel = compile_process(
        r#"model Process() {
        parameter rate: 1/s = 3[1/s];
        noise W: wiener(covariance=[[1,0],[0,1]], independent=true);
        state x: array<1,2>;
        initial { x=[2,4]; }
        law dynamics for x {
            calculus ito;
            drift -x * rate;
            diffusion W=[[1[1/s^(1/2)],2[1/s^(1/2)]],[3[1/s^(1/2)],4[1/s^(1/2)]]];
        }
    }"#,
    );
    let process = ProcessProgram::new(&kernel).unwrap();
    let initial = process.initialize(0.).unwrap();
    assert_eq!(initial.values(), [2., 4.]);
    let law = &process.laws[0];
    assert_eq!(law.state, 0..2);
    let drift = law.drift.as_ref().unwrap();
    assert!(!drift.constant());
    assert_eq!(drift.evaluate(initial.values(), 0.).unwrap(), [-6., -12.]);
    assert_eq!(drift.evaluate(&[5., 7.], 0.).unwrap(), [-15., -21.]);
    assert!(drift.evaluate(&[5.], 0.).is_err());
    let diffusion = &law.diffusions[0].1;
    assert!(diffusion.constant());
    assert_eq!(
        diffusion.evaluate(initial.values(), 0.).unwrap(),
        [1., 2., 3., 4.]
    );
    let noise = &process.noises[0];
    assert_eq!(law.diffusions[0].0, noise.id);
    assert!(noise.parameter.constant());
    assert_eq!(
        noise.parameter.evaluate(initial.values(), 0.).unwrap(),
        [1., 0., 0., 1.]
    );
    // The semantic evaluator independently inspects the original shaped operands.
    let fields = eqiora_sem::Interpreter::new()
        .initialize(
            &kernel,
            0.,
            eqiora_sem::ReferenceConfig::new(0., 1.).unwrap(),
        )
        .unwrap()
        .fields()
        .iter()
        .map(|(id, value)| (eqiora_core::Id::from_ulid(id.ulid()), value.clone()))
        .collect::<Vec<_>>();
    let operands = kernel
        .evaluate_relation_operands(law.id, &fields, &[], &[])
        .unwrap();
    for (value, coefficient) in operands.iter().zip([drift, diffusion]) {
        assert_eq!(value.value_type(), &coefficient.value_type);
        assert_eq!(
            value
                .components()
                .unwrap()
                .map(|(real, _)| real)
                .collect::<Vec<_>>(),
            coefficient.evaluate(initial.values(), 0.).unwrap()
        );
    }
}

#[test]
fn process_coefficients_bind_physical_time_without_a_parallel_evaluator() {
    let kernel = compile_process(
        r#"model Process() {
        noise W: wiener(covariance=1, independent=true);
        state x: 1;
        initial { x=2; }
        law dynamics for x {
            calculus stratonovich;
            drift time()/1[s^2];
            diffusion W=x * 1[1/s^(1/2)];
        }
    }"#,
    );
    let process = program::ProcessProgram::new(&kernel).unwrap();
    let law = &process.laws[0];
    assert_eq!(
        law.calculus,
        Some(eqiora_schema::kernel::StochasticCalculus::Stratonovich)
    );
    assert_eq!(
        law.drift.as_ref().unwrap().evaluate(&[2.], 7.).unwrap(),
        [7.]
    );
    assert_eq!(law.diffusions[0].1.evaluate(&[3.], 7.).unwrap(), [3.]);
}

#[test]
fn model_owned_correlated_paths_preserve_composition_and_nominal_channels() {
    let kernel = compile_process(
        r#"model Process() {
        noise W: wiener(covariance=[[1,-1],[-1,1]], independent=true);
        noise V: wiener(covariance=[[1,-1],[-1,1]], independent=true);
        state x: 1;
        initial { x=0; }
        law dynamics for x {
            calculus ito;
            drift 0[1/s];
            diffusion W=[1[1/s^(1/2)],2[1/s^(1/2)]];
            diffusion V=[3[1/s^(1/2)],4[1/s^(1/2)]];
        }
    }"#,
    );
    let process = program::ProcessProgram::new(&kernel).unwrap();
    let id = identity(5, b"experiment", &[7]);
    let first = process.noises[0].realize(&[9; 32], &id, 1., 8, 64).unwrap();
    let whole = first.wiener_increment(0, 256).unwrap();
    assert_eq!(whole[0], -&whole[1]);
    let left = first.wiener_increment(0, 71).unwrap();
    let right = first.wiener_increment(71, 256).unwrap();
    for channel in 0..2 {
        assert_eq!(&left[channel] + &right[channel], whole[channel]);
    }
    let second = process.noises[1].realize(&[9; 32], &id, 1., 8, 64).unwrap();
    assert_ne!(whole, second.wiener_increment(0, 256).unwrap());
    let reordered = process.noises[0]
        .realize(&[9; 32], &identity(5, b"experiment", &[2, 1]), 1., 8, 64)
        .unwrap();
    assert_eq!(first, reordered);
    assert!(first.poisson_increment(&exact(0.), &exact(1.)).is_err());
}

#[test]
fn poisson_realization_binds_constant_intensity_and_rejects_state_dependent_rates() {
    let source = r#"model Process() {
        parameter rate: 1/s=2[1/s];
        noise N: poisson(intensity=rate);
        state x: 1;
        initial { x=1; }
        law dynamics for x { jump N=1; }
    }"#;
    let process = program::ProcessProgram::new(&compile_process(source)).unwrap();
    let id = identity(5, b"experiment", &[]);
    let path = process.noises[0].realize(&[9; 32], &id, 1., 8, 64).unwrap();
    assert_eq!(
        path.poisson_increment(&exact(0.), &exact(0.5)).unwrap()[0]
            + path.poisson_increment(&exact(0.5), &exact(1.)).unwrap()[0],
        path.poisson_increment(&exact(0.), &exact(1.)).unwrap()[0]
    );
    assert!(path.wiener_increment(0, 256).is_err());
    let dynamic = program::ProcessProgram::new(&compile_process(
        &source.replace("intensity=rate", "intensity=rate*x"),
    ))
    .unwrap();
    assert!(dynamic.noises[0].realize(&[9; 32], &id, 1., 8, 64).is_err());
}

#[test]
fn physical_time_state_replays_through_one_model_bound_codec() {
    use crate::common_time::TimeStateSpace;
    let source = r#"model Process() {
        noise W: wiener(covariance=1);
        state x: array<m,2>;
        initial { x=[2[m],4[m]]; }
        law dynamics for x {
            calculus ito;
            drift x * 1[1/s];
            diffusion W=x * 1[1/s^(1/2)];
        }
    }"#;
    let kernel = compile_process(source);
    let model = eqiora_artifact::ModelEnvelope::from_program(&kernel).unwrap();
    let process = program::ProcessProgram::new(&kernel).unwrap();
    let state = process.initialize(0.).unwrap();
    let bytes = state.to_bytes().unwrap();
    assert_eq!(process.state_from_bytes(&bytes).unwrap(), state);
    let field = state.state_coordinates()[0].field();
    let value = state.field_value(&model, field, 0).unwrap();
    assert_eq!(
        value.components().unwrap().collect::<Vec<_>>(),
        [(2., 0.), (4., 0.)]
    );
    assert!(state.field_value(&model, field, 1).is_err());
    assert!(TimeStateSpace::new(&model, state.state_coordinates()[..1].to_vec(), b"test").is_err());
    assert!(TimeStateSpace::new(&model, vec![state.state_coordinates()[0]; 2], b"test").is_err());
    assert!(process.accept_state(0., vec![1.], "result").is_err());
    assert!(process.accept_state(-0., vec![1., 2.], "result").is_err());
    assert!(
        process
            .accept_state(0., vec![1., f64::NAN], "result")
            .is_err()
    );
    let foreign = program::ProcessProgram::new(&compile_process(
        &source.replace("[2[m],4[m]]", "[3[m],4[m]]"),
    ))
    .unwrap();
    assert!(foreign.state_from_bytes(&bytes).is_err());
    let mut tampered: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    tampered["values"][0] = 99.into();
    assert!(
        process
            .state_from_bytes(&serde_json::to_vec(&tampered).unwrap())
            .is_err()
    );
    let retired = String::from_utf8(bytes.clone())
        .unwrap()
        .replace("eqiora.common-time-state/v1", "eqiora.common-ode-state/v1");
    assert!(process.state_from_bytes(retired.as_bytes()).is_err());
    let mut noncanonical = bytes;
    noncanonical.push(b'\n');
    assert!(process.state_from_bytes(&noncanonical).is_err());
}

mod jump;
mod plan;
mod run;

pub(super) fn process_run(
    kernel: &eqiora_sem::KernelProgram,
    state: crate::CommonTimeState,
    master: &[u8; 32],
    identity: &SamplingIdentity,
    duration: f64,
    method: super::run::ProcessMethod,
) -> Result<std::sync::Arc<super::run::ProcessRun>, eqiora_core::Diagnostic> {
    let model = eqiora_artifact::ModelEnvelope::from_program(kernel)?;
    let plan = crate::CommonTimePlan::resolve_process(&model, kernel, method)?;
    let until = state.time_s() + duration;
    let request = crate::CommonTimeRunRequest::sampled(
        plan,
        state,
        until,
        vec![until],
        *master,
        identity.clone(),
    )?;
    super::run::ProcessRun::new(request)
}

/// Registered claim: exact retained bridge algebra and independently derived moments.
#[test]
fn retained_wiener_bridge_claim() {
    retained_increments_compose_exactly_under_reordering_and_refinement();
    sample_channel_and_explicit_coupling_own_paths_not_occurrence_order();
    independently_derived_wiener_and_conditional_bridge_moments();
    endpoint_conversion_and_degenerate_law_are_explicit();
}
