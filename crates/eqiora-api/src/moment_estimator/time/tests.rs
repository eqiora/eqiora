//! Fixed scientific campaigns; references were derived before candidate execution.
use super::*;
use crate::ModelDocument;
use eqiora_artifact::ModelEnvelope;
use eqiora_core::{DimExponents, ScalarDomain};
use eqiora_numerics::{CommonTimePlan, CommonTimeState, SamplingIdentity};
use serde_json::Value;
use std::ops::ControlFlow;

fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../../../../../verify/time/ito-distribution-moments/expected/campaign.json"
    ))
    .unwrap()
}

fn number(value: &Value, key: &str) -> f64 {
    value[key].as_str().unwrap().parse().unwrap()
}

fn scalar(value: &ValueLiteral) -> f64 {
    value.real_scalar_value().unwrap().value()
}

fn run(
    plan: &CommonTimePlan,
    initial: &CommonTimeState,
    campaign: &Value,
    channel: &str,
    sample: u32,
    times: &[f64],
) -> CommonResult {
    let request = CommonTimeRunRequest::sampled(
        plan.clone(),
        initial.clone(),
        campaign["duration_s"].as_f64().unwrap(),
        times.to_vec(),
        [campaign["master_seed_byte"]
            .as_u64()
            .unwrap()
            .try_into()
            .unwrap(); 32],
        SamplingIdentity::new(&sample.to_le_bytes(), channel.as_bytes(), &[], None).unwrap(),
    )
    .unwrap();
    CommonResult::accept_trajectory(
        0.,
        request
            .advance_process(None, |_, _| ControlFlow::Continue(()))
            .unwrap()
            .continue_value()
            .unwrap(),
    )
    .unwrap()
}

fn observed(result: &CommonResult, output: &Output, index: usize) -> ValueLiteral {
    result
        .observe(
            result.plan().model_artifact(),
            output.id(),
            &HashMap::new(),
            Some(index),
        )
        .unwrap()
        .value()
        .clone()
}

fn accepted(value: f64, expected: f64, radius: f64) -> bool {
    value.is_finite() && (value - expected).abs() <= radius
}

#[test]
fn ito_distribution_moments_claim() {
    let campaign = reference();
    let count: u32 = campaign["sample_count"]
        .as_u64()
        .unwrap()
        .try_into()
        .unwrap();
    let components =
        include_str!("../../../packages/Eqiora.Stochastic.Processes/src/processes.eqi");
    let composition = include_str!("../../../../../examples/stochastic-processes/src/main.eqi")
        .replace(
            "import Eqiora.Stochastic.Processes.processes as processes;",
            "",
        )
        .replace("processes.", "");
    let document =
        ModelDocument::compile("distribution.eqi", &format!("{components}\n{composition}"))
            .unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let outputs = [
        ("ou", document.observable_ref("ou.sample").unwrap()),
        ("gbm", document.observable_ref("gbm.sample").unwrap()),
    ];
    let mut preceding_bias = [f64::INFINITY; 4];
    for reference in campaign["rows"].as_array().unwrap() {
        let depth = reference["depth"].as_u64().unwrap().try_into().unwrap();
        let plan =
            CommonTimePlan::resolve_euler_maruyama(&model, document.program(), depth).unwrap();
        let initial = plan.initial_state(0.).unwrap();
        let members: Arc<[CommonResult]> = (0..count)
            .map(|sample| {
                run(
                    &plan,
                    &initial,
                    &campaign,
                    campaign["weak_channel"].as_str().unwrap(),
                    sample,
                    &[1.],
                )
            })
            .collect::<Vec<_>>()
            .into();
        for (process_index, (name, output)) in outputs.iter().enumerate() {
            let expected = &reference[*name];
            for (moment_index, (kind, name)) in [
                (ProbabilityMoment::Expectation, "mean"),
                (ProbabilityMoment::Variance, "variance"),
            ]
            .into_iter()
            .enumerate()
            {
                let estimator = MomentEstimator::from_time_results(
                    members.clone(),
                    1.,
                    MomentReduction::UnbiasedIid(kind),
                    std::slice::from_ref(output),
                    vec![1.; count as usize],
                )
                .unwrap();
                let value = scalar(estimator.value());
                let finite = number(expected, name);
                let radius = number(expected, &format!("{name}_six_se"));
                assert!(
                    accepted(value, finite, radius),
                    "{} depth={depth} {name}: observed={value}, finite-grid={finite}, six-SE={radius}",
                    outputs[process_index].0
                );
                // Time-discretization bias is derived separately from the SDE law.
                let continuous = number(expected, &format!("continuous_{name}"));
                let bias = (finite - continuous).abs();
                assert!(accepted(value, continuous, bias + radius));
                let previous = &mut preceding_bias[2 * process_index + moment_index];
                assert!(
                    bias < *previous,
                    "the fixed-grid continuum bias must decrease"
                );
                *previous = bias;
                eprintln!(
                    "{} depth={depth} {name}={value} finite_grid={finite} bias={bias} six_se={radius}",
                    outputs[process_index].0
                );
            }
        }
        // Numerical falsifiers reach the moment predicate, not a foreign-Model gate.
        // OU is affine in volatility: replacing sqrt(h)*Z by h*Z scales its
        // centered output by sqrt(h), so the accepted variance scales by h.
        let ou = &reference["ou"];
        let deterministic_mean = number(ou, "mean");
        let h = 1. / reference["steps"].as_f64().unwrap();
        let value_type =
            ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
        let mut wrong_scaling = WeightedMoments::new(
            ProbabilityMoment::Variance,
            std::slice::from_ref(&value_type),
        )
        .unwrap();
        let mut shared_stream = wrong_scaling.clone();
        let first = observed(&members[0], &outputs[0].1, 0);
        for member in members.iter() {
            let correct = scalar(&observed(member, &outputs[0].1, 0));
            let wrong = deterministic_mean + h.sqrt() * (correct - deterministic_mean);
            wrong_scaling
                .push(
                    &[ValueLiteral::from_real(value_type.clone(), wrong).unwrap()],
                    1.,
                )
                .unwrap();
            shared_stream
                .push(std::slice::from_ref(&first), 1.)
                .unwrap();
        }
        for mutant in [wrong_scaling, shared_stream] {
            assert!(!accepted(
                scalar(&mutant.value(1).unwrap()),
                number(ou, "variance"),
                number(ou, "variance_six_se")
            ));
        }
    }
}

#[test]
fn ito_path_refinement_claim() {
    let campaign = reference();
    let count: u32 = campaign["sample_count"]
        .as_u64()
        .unwrap()
        .try_into()
        .unwrap();
    let document = ModelDocument::compile(
        "coupled-gbm.eqi",
        include_str!(
            "../../../../../verify/time/ito-path-refinement/models/geometric-brownian.eqi"
        ),
    )
    .unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let x = document.observable_ref("value").unwrap();
    let w = document.observable_ref("driver").unwrap();
    let rows = campaign["rows"].as_array().unwrap();
    let plans = rows
        .iter()
        .map(|row| {
            CommonTimePlan::resolve_euler_maruyama(
                &model,
                document.program(),
                row["depth"].as_u64().unwrap().try_into().unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let initial = plans[0].initial_state(0.).unwrap();
    let outputs = (1..=8).map(|tick| f64::from(tick) / 8.).collect::<Vec<_>>();
    let value_type = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
    let moment = WeightedMoments::new(
        ProbabilityMoment::Expectation,
        std::slice::from_ref(&value_type),
    )
    .unwrap();
    let mut errors = vec![moment.clone(); rows.len()];
    let mut wrong_ito = errors.clone();
    for sample in 0..count {
        let mut coarse_driver = Vec::new();
        for (index, plan) in plans.iter().enumerate() {
            let result = run(
                plan,
                &initial,
                &campaign,
                campaign["strong_channel"].as_str().unwrap(),
                sample,
                &outputs,
            );
            let driver = outputs
                .iter()
                .enumerate()
                .map(|(output, _)| scalar(&observed(&result, &w, output)))
                .collect::<Vec<_>>();
            if index == 0 {
                coarse_driver = driver.clone();
            } else {
                // Exact retained increments telescope (#970). This projection
                // adds binary64 values in up to 128 steps, so allow roundoff only.
                for (coarse, fine) in coarse_driver.iter().zip(&driver) {
                    assert!((coarse - fine).abs() <= 4096. * f64::EPSILON * (1. + coarse.abs()));
                }
            }
            let value = scalar(&observed(&result, &x, outputs.len() - 1));
            let brownian = driver[outputs.len() - 1];
            // Ito's formula: log X_T = (a-b^2/2)T + b W_T, x0=T=1.
            let exact = (0.25 - 0.5 * 0.5_f64.powi(2) + 0.5 * brownian).exp();
            let missing_correction = (0.25 + 0.5 * brownian).exp();
            for (accumulator, target) in [
                (&mut errors[index], exact),
                (&mut wrong_ito[index], missing_correction),
            ] {
                accumulator
                    .push(
                        &[
                            ValueLiteral::from_real(value_type.clone(), (value - target).powi(2))
                                .unwrap(),
                        ],
                        1.,
                    )
                    .unwrap();
            }
        }
    }
    let mut measured = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let error = scalar(&errors[index].value(0).unwrap());
        let expected = number(&row["gbm"], "path_mse");
        let radius = number(&row["gbm"], "path_mse_six_se");
        assert!(
            accepted(error, expected, radius),
            "GBM depth={} MSE={error}, exact-grid MSE={expected}, six-SE={radius}",
            row["depth"]
        );
        assert!(
            !accepted(
                scalar(&wrong_ito[index].value(0).unwrap()),
                expected,
                radius
            ),
            "omitting the Ito drift correction must fail the path criterion"
        );
        measured.push(error);
        eprintln!(
            "GBM depth={} MSE={error} expected={expected} six_se={radius}",
            row["depth"]
        );
    }
    assert!(
        4. * measured.last().unwrap() < measured[0],
        "the fixed 8-to-128-step refinement must reduce strong MSE"
    );
}
