//! Estimator admission uses exact accepted membership and declared sample design.
use eqiora::api::{
    CompleteEvaluationMap, DifferentiableProgram, EvaluationMapEstimator,
    EvaluationMapExecutionPolicy, EvaluationMapPlan, ModelDocument, ModelEntityRef,
    MomentEstimatorConvention, ParameterSampler, SampledParameterPoint, SamplingIdentity,
};
use eqiora_artifact::ModelEnvelope;
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_core::DimExponents;
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy, ResolvedCommonPlan,
};
use eqiora_schema::kernel::ProbabilityMoment;
use eqiora_solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
use std::{num::NonZeroUsize, sync::Arc};

fn fixture() -> (Arc<DifferentiableProgram>, [ModelEntityRef; 2]) {
    named_fixture("M")
}

fn named_fixture(name: &str) -> (Arc<DifferentiableProgram>, [ModelEntityRef; 2]) {
    let document = ModelDocument::compile(
        "estimator.eqi",
        &r#"
        model M() {
          parameter position:m=1[m];
          variable x:m; variable y:s;
          relation system {x=position; y=3[s]-1[s/m]*position;}
        }
    "#
        .replace("model M()", &format!("model {name}()")),
    )
    .unwrap();
    let outputs = [
        document.field_ref("x").unwrap().erase(),
        document.field_ref("y").unwrap().erase(),
    ];
    let artifact = ModelEnvelope::from_program(document.program()).unwrap();
    let policy = CommonSolvePolicy::Linear(
        CommonLinearRequest::exact(
            SolverPlan::new(
                LinearSolver::SparseLu,
                1e-12,
                1e-14,
                NonZeroUsize::new(16).unwrap(),
            )
            .unwrap()
            .with_reduction(ReductionPolicy::Fast),
            FaerLinearSolver.provider(),
        )
        .unwrap(),
    );
    let plan = CommonAlgebraicPlan::resolve(&artifact, policy, None, &[], None, &FaerLinearSolver)
        .unwrap();
    let program = DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(plan)),
        &[document.parameter_ref("position").unwrap()],
        &outputs,
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    (Arc::new(program), outputs)
}

fn sample(
    sampler: &ParameterSampler,
    id: u8,
    channel: &[u8],
    coupling: Option<&[u8]>,
) -> SampledParameterPoint {
    sampler
        .sample(&SamplingIdentity::new(&[id], channel, &[], coupling).unwrap())
        .unwrap()
}

fn accepted(
    program: &Arc<DifferentiableProgram>,
    samples: &[SampledParameterPoint],
) -> CompleteEvaluationMap {
    EvaluationMapPlan::from_samples(
        program.clone(),
        samples,
        &[],
        EvaluationMapExecutionPolicy::retained(1 << 28),
    )
    .unwrap()
    .execute()
    .unwrap()
}

#[test]
fn accepted_estimator_binds_units_order_weights_and_full_sample_identity() {
    let (program, outputs) = fixture();
    // Distinct identities deliberately have identical numeric values on this
    // adjacent-f64 interval. Multiplicity and independence requests are separate.
    let bounds = [[1.0, 1.0f64.next_up()]];
    let sampler = ParameterSampler::new(program.clone(), [42; 32], &bounds).unwrap();
    let samples = [
        sample(&sampler, 1, b"position", None),
        sample(&sampler, 2, b"position", None),
    ];
    assert_eq!(samples[0].point(), samples[1].point());
    assert_eq!(samples[0].bounds(), bounds);
    assert_eq!(samples[0].generator(), ParameterSampler::GENERATOR);
    let map = accepted(&program, &samples);
    for (moment, selected, units) in [
        (
            ProbabilityMoment::Expectation,
            &outputs[..1],
            [0, 1, 0, 0, 0, 0, 0],
        ),
        (
            ProbabilityMoment::Variance,
            &outputs[..1],
            [0, 2, 0, 0, 0, 0, 0],
        ),
        (
            ProbabilityMoment::Covariance,
            &outputs[..],
            [0, 1, 1, 0, 0, 0, 0],
        ),
    ] {
        for convention in [
            MomentEstimatorConvention::Empirical,
            MomentEstimatorConvention::UnbiasedIid,
        ] {
            let bound =
                EvaluationMapEstimator::new(&map, moment, selected, &[2.0, 2.0], convention)
                    .unwrap();
            assert!(std::ptr::eq(bound.map(), &map));
            assert_eq!(bound.map().plan().samples().unwrap(), samples);
            assert_eq!(bound.moment(), moment);
            assert_eq!(bound.convention(), convention);
            assert_eq!(bound.weights(), [2.0, 2.0]);
            assert_eq!(
                bound.value_type().dimension(),
                DimExponents::from_integers(units).unwrap()
            );
        }
    }
    let reversed = EvaluationMapEstimator::new(
        &map,
        ProbabilityMoment::Covariance,
        &[outputs[1].clone(), outputs[0].clone()],
        &[1.0, 3.0],
        MomentEstimatorConvention::Empirical,
    )
    .unwrap();
    assert_eq!(reversed.output_indices(), [1, 0]);
    assert_eq!(reversed.weights(), [1.0, 3.0]);

    let (foreign_program, foreign_outputs) = named_fixture("Foreign");
    assert_ne!(foreign_program.identity(), program.identity());
    assert!(
        EvaluationMapEstimator::new(
            &map,
            ProbabilityMoment::Expectation,
            &foreign_outputs[..1],
            &[1.0, 1.0],
            MomentEstimatorConvention::Empirical
        )
        .unwrap_err()
        .message()
        .contains("exact accepted Program")
    );
    assert!(
        EvaluationMapPlan::from_samples(
            foreign_program,
            &samples,
            &[],
            EvaluationMapExecutionPolicy::retained(1 << 28)
        )
        .unwrap_err()
        .message()
        .contains("foreign Program")
    );
}

#[test]
fn iid_interpretation_rejects_replayed_coupled_or_different_design_samples() {
    let (program, outputs) = fixture();
    let sampler = ParameterSampler::new(program.clone(), [42; 32], &[[1.0, 2.0]]).unwrap();
    let different_bounds = ParameterSampler::new(program.clone(), [42; 32], &[[2.0, 3.0]]).unwrap();
    let different_master = ParameterSampler::new(program.clone(), [43; 32], &[[1.0, 2.0]]).unwrap();
    let first = sample(&sampler, 1, b"position", None);
    for second in [
        first.clone(),
        sampler
            .sample(&SamplingIdentity::new(&[1], b"position", &[999], None).unwrap())
            .unwrap(),
        sample(&sampler, 2, b"position", Some(b"paired")),
        sample(&sampler, 2, b"other-channel", None),
        sample(&different_bounds, 2, b"position", None),
        sample(&different_master, 2, b"position", None),
    ] {
        let map = accepted(&program, &[first.clone(), second]);
        EvaluationMapEstimator::new(
            &map,
            ProbabilityMoment::Variance,
            &outputs[..1],
            &[1.0, 1.0],
            MomentEstimatorConvention::Empirical,
        )
        .unwrap();
        let error = EvaluationMapEstimator::new(
            &map,
            ProbabilityMoment::Variance,
            &outputs[..1],
            &[1.0, 1.0],
            MomentEstimatorConvention::UnbiasedIid,
        )
        .unwrap_err();
        assert!(
            error.message().contains("one declared sampling design"),
            "{error:?}"
        );
    }
    let deterministic = EvaluationMapPlan::new(
        program,
        &[first.point().clone()],
        &[],
        EvaluationMapExecutionPolicy::retained(1 << 28),
    )
    .unwrap()
    .execute()
    .unwrap();
    EvaluationMapEstimator::new(
        &deterministic,
        ProbabilityMoment::Expectation,
        &outputs[..1],
        &[1.0],
        MomentEstimatorConvention::Empirical,
    )
    .unwrap();
    assert!(
        EvaluationMapEstimator::new(
            &deterministic,
            ProbabilityMoment::Expectation,
            &outputs[..1],
            &[1.0],
            MomentEstimatorConvention::UnbiasedIid
        )
        .unwrap_err()
        .message()
        .contains("sample-design metadata")
    );
}

#[test]
fn estimator_normalization_and_minimum_count_are_explicit() {
    let (program, outputs) = fixture();
    let sampler = ParameterSampler::new(program.clone(), [42; 32], &[[1.0, 2.0]]).unwrap();
    let samples = [
        sample(&sampler, 1, b"position", None),
        sample(&sampler, 2, b"position", None),
    ];
    let map = accepted(&program, &samples);
    for weights in [
        vec![],
        vec![1.0],
        vec![0.0, 0.0],
        vec![-1.0, 2.0],
        vec![f64::NAN, 1.0],
        vec![1.0, f64::INFINITY],
    ] {
        assert!(
            EvaluationMapEstimator::new(
                &map,
                ProbabilityMoment::Variance,
                &outputs[..1],
                &weights,
                MomentEstimatorConvention::Empirical
            )
            .is_err()
        );
    }
    for weights in [[0.0, 1.0], [1.0, 3.0]] {
        EvaluationMapEstimator::new(
            &map,
            ProbabilityMoment::Variance,
            &outputs[..1],
            &weights,
            MomentEstimatorConvention::Empirical,
        )
        .unwrap();
        assert!(
            EvaluationMapEstimator::new(
                &map,
                ProbabilityMoment::Variance,
                &outputs[..1],
                &weights,
                MomentEstimatorConvention::UnbiasedIid
            )
            .is_err()
        );
    }
    // Validation must not overflow a floating total whose normalized ratios are finite.
    EvaluationMapEstimator::new(
        &map,
        ProbabilityMoment::Variance,
        &outputs[..1],
        &[f64::MAX, f64::MAX],
        MomentEstimatorConvention::UnbiasedIid,
    )
    .unwrap();
    let singleton = accepted(&program, &samples[..1]);
    EvaluationMapEstimator::new(
        &singleton,
        ProbabilityMoment::Expectation,
        &outputs[..1],
        &[1.0],
        MomentEstimatorConvention::UnbiasedIid,
    )
    .unwrap();
    assert!(
        EvaluationMapEstimator::new(
            &singleton,
            ProbabilityMoment::Variance,
            &outputs[..1],
            &[1.0],
            MomentEstimatorConvention::UnbiasedIid
        )
        .is_err()
    );
    let empty = accepted(&program, &[]);
    assert!(
        EvaluationMapEstimator::new(
            &empty,
            ProbabilityMoment::Expectation,
            &outputs[..1],
            &[],
            MomentEstimatorConvention::Empirical
        )
        .is_err()
    );
    assert!(
        EvaluationMapEstimator::new(
            &map,
            ProbabilityMoment::Covariance,
            &outputs[..1],
            &[1.0, 1.0],
            MomentEstimatorConvention::Empirical
        )
        .is_err()
    );
}
