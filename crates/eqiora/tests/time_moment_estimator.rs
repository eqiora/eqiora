//! Accepted time Results share the existing typed moment arithmetic and conventions.
use eqiora::{
    DimExponents, ValueLiteral,
    api::{ModelDocument, MomentEstimator, MomentReduction},
    artifact::ModelEnvelope,
};
use eqiora_numerics::{CommonResult, CommonTimePlan, CommonTimeRunRequest, SamplingIdentity};
use eqiora_schema::kernel::ProbabilityMoment;
use std::{collections::HashMap, ops::ControlFlow, sync::Arc};

fn document() -> ModelDocument {
    ModelDocument::compile(
        "time-statistic.eqi",
        r#"model Process() {
          noise W:wiener(covariance=1);
          state x:m; state y:s;
          initial{x=0[m]; y=0[s];}
          law first for x {calculus ito; drift 0[m/s]; diffusion W=1[m/s^(1/2)];}
          law second for y {calculus ito; drift 0; diffusion W=2[s^(1/2)];}
          observable position:m=x; observable duration:s=y;
        }"#,
    )
    .unwrap()
}

fn run(
    document: &ModelDocument,
    depth: u8,
    identity: SamplingIdentity,
    seed: [u8; 32],
    outputs: Vec<f64>,
) -> CommonResult {
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = CommonTimePlan::resolve_euler_maruyama(&model, document.program(), depth).unwrap();
    let request = CommonTimeRunRequest::sampled(
        plan.clone(),
        plan.initial_state(0.).unwrap(),
        1.,
        outputs,
        seed,
        identity,
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

fn identity(sample: u8) -> SamplingIdentity {
    SamplingIdentity::new(&[sample], b"path", &[], None).unwrap()
}

fn scalar(value: &ValueLiteral) -> f64 {
    value.real_scalar_value().unwrap().value()
}

#[test]
fn time_statistics_bind_complete_results_units_time_and_sampling_assumptions() {
    let document = document();
    let outputs = [
        document.observable_ref("position").unwrap(),
        document.observable_ref("duration").unwrap(),
    ];
    let members: Arc<[CommonResult]> = vec![
        run(&document, 3, identity(1), [7; 32], vec![0.5, 1.]),
        // Different output cadence does not change the selected physical time.
        run(&document, 3, identity(2), [7; 32], vec![1.]),
    ]
    .into();
    let model = members[0].plan().model_artifact();
    let x = members
        .iter()
        .enumerate()
        .map(|(index, member)| {
            scalar(
                member
                    .observe(model, outputs[0].id(), &HashMap::new(), Some(1 - index))
                    .unwrap()
                    .value(),
            )
        })
        .collect::<Vec<_>>();
    let mean = MomentEstimator::from_time_results(
        members.clone(),
        1.,
        MomentReduction::UnbiasedIid(ProbabilityMoment::Expectation),
        &outputs[..1],
        vec![1., 1.],
    )
    .unwrap();
    assert_eq!(mean.model(), outputs[0].model());
    assert_eq!(mean.time_observation().unwrap(), (1., &outputs[..1]));
    assert_eq!(
        mean.time_results().unwrap()[0].identity(),
        members[0].identity()
    );
    assert!(mean.map().is_none());
    assert!(
        mean.vjp(1 << 20).is_err(),
        "statistics confer no trajectory AD"
    );
    let expected_mean = (x[0] + x[1]) / 2.;
    let expected_error = (x[0] - x[1]).abs() / 2.;
    let roundoff = 16. * f64::EPSILON * x.iter().copied().map(f64::abs).sum::<f64>().max(1.);
    assert!((scalar(mean.value()) - expected_mean).abs() <= roundoff);
    assert!((scalar(&mean.standard_error().unwrap()) - expected_error).abs() <= roundoff);
    assert_eq!(
        mean.value_type().dimension(),
        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap()
    );
    let covariance = MomentEstimator::from_time_results(
        members,
        1.,
        MomentReduction::UnbiasedIid(ProbabilityMoment::Covariance),
        &outputs,
        vec![1., 1.],
    )
    .unwrap();
    // Y=2X in numerical coordinates: n=2 sample covariance is (x0-x1)^2.
    assert!(
        (scalar(covariance.value()) - (x[0] - x[1]).powi(2)).abs()
            <= 4. * roundoff * x.iter().copied().map(f64::abs).sum::<f64>().max(1.)
    );
    assert_eq!(
        covariance.value_type().dimension(),
        DimExponents::from_integers([0, 1, 1, 0, 0, 0, 0]).unwrap()
    );
    assert!(covariance.standard_error().is_err());
}

#[test]
fn time_iid_admission_rejects_repeated_coupled_and_mixed_design_samples() {
    let document = document();
    let output = document.observable_ref("position").unwrap();
    let first = run(&document, 3, identity(1), [7; 32], vec![1.]);
    for second in [
        first.clone(),
        run(
            &document,
            3,
            SamplingIdentity::new(&[1], b"path", &[99], None).unwrap(),
            [7; 32],
            vec![1.],
        ),
        run(
            &document,
            3,
            SamplingIdentity::new(&[2], b"path", &[], Some(b"paired")).unwrap(),
            [7; 32],
            vec![1.],
        ),
        run(
            &document,
            3,
            SamplingIdentity::new(&[2], b"other", &[], None).unwrap(),
            [7; 32],
            vec![1.],
        ),
        run(&document, 3, identity(2), [8; 32], vec![1.]),
    ] {
        let members: Arc<[CommonResult]> = vec![first.clone(), second].into();
        assert!(
            MomentEstimator::from_time_results(
                members.clone(),
                1.,
                MomentReduction::UnbiasedIid(ProbabilityMoment::Expectation),
                std::slice::from_ref(&output),
                vec![1., 1.],
            )
            .is_err()
        );
        // Empirical moments preserve the actual supplied membership without an IID claim.
        MomentEstimator::from_time_results(
            members,
            1.,
            MomentReduction::Empirical(ProbabilityMoment::Expectation),
            std::slice::from_ref(&output),
            vec![1., 1.],
        )
        .unwrap();
    }
}

#[test]
fn missing_outputs_and_foreign_plans_never_become_partial_statistics() {
    let document = document();
    let output = document.observable_ref("position").unwrap();
    let first = run(&document, 3, identity(1), [7; 32], vec![1.]);
    for second in [
        run(&document, 3, identity(2), [7; 32], vec![0.5]),
        run(&document, 4, identity(2), [7; 32], vec![1.]),
    ] {
        assert!(
            MomentEstimator::from_time_results(
                vec![first.clone(), second].into(),
                1.,
                MomentReduction::Empirical(ProbabilityMoment::Expectation),
                std::slice::from_ref(&output),
                vec![1., 0.],
            )
            .is_err(),
            "even zero-weight members must be admitted"
        );
    }
    for reduction in [
        MomentReduction::Population(ProbabilityMoment::Expectation),
        MomentReduction::UnbiasedIid(ProbabilityMoment::Variance),
    ] {
        assert!(
            MomentEstimator::from_time_results(
                vec![first.clone()].into(),
                1.,
                reduction,
                std::slice::from_ref(&output),
                vec![1.],
            )
            .is_err()
        );
    }
}
