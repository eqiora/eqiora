//! Estimator execution uses exact accepted membership and declared sample design.
#[path = "probability_estimators/execution.rs"]
mod execution;
use eqiora::api::{
    CompleteEvaluationMap, DifferentiableProgram, EvaluationMapExecutionPolicy, EvaluationMapPlan,
    ModelDocument, ModelEntityRef, MomentEstimator, MomentReduction, ParameterSampler,
    SampledParameterPoint, SamplingIdentity,
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
    compile_fixture(name, false)
}

fn compile_fixture(
    name: &str,
    with_gain: bool,
) -> (Arc<DifferentiableProgram>, [ModelEntityRef; 2]) {
    let document = ModelDocument::compile(
        "estimator.eqi",
        &r#"
        model M() {
          parameter position:m=1[m]; parameter gain:1=1;
          variable x:m; variable y:s;
          relation system {x=gain*position; y=3[s]-1[s/m]*position;}
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
    let mut inputs = vec![document.parameter_ref("position").unwrap()];
    if with_gain {
        inputs.insert(0, document.parameter_ref("gain").unwrap());
    }
    let program = DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(plan)),
        &inputs,
        &outputs,
        Default::default(),
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
    let sampler = ParameterSampler::new(
        program.clone(),
        [42; 32],
        &eqiora::api::DifferentiableValues::from_components(&[], &[]).unwrap(),
        &bounds,
    )
    .unwrap();
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
        for reduction in [
            MomentReduction::Empirical(moment),
            MomentReduction::UnbiasedIid(moment),
        ] {
            let bound = MomentEstimator::evaluate(&map, reduction, selected, &[2.0, 2.0]).unwrap();
            let recorded = map.record().unwrap().with_estimator(0, &bound).unwrap();
            let bytes = recorded.to_bytes().unwrap();
            let reopened = eqiora::api::EvaluationCollection::from_bytes(
                &bytes,
                &FaerLinearSolver,
                eqiora_time::ImplicitMidpointTimeBackend::CAPABILITIES,
            )
            .unwrap();
            assert_eq!(reopened.to_bytes().unwrap(), bytes);
            let restored = &reopened.groups()[0].estimators()[0];
            assert_eq!(restored.value(), bound.value());
            assert_eq!(restored.reduction(), bound.reduction());
            assert_eq!(restored.output_indices(), bound.output_indices());
            assert_eq!(restored.weights(), bound.weights());
            assert!(std::ptr::eq(bound.map().unwrap(), &map));
            assert_eq!(bound.map().unwrap().plan().samples().unwrap(), samples);
            assert_eq!(bound.reduction(), reduction);
            assert_eq!(bound.weights(), [2.0, 2.0]);
            assert_eq!(
                bound.value_type().dimension(),
                DimExponents::from_integers(units).unwrap()
            );
        }
    }
    let reversed = MomentEstimator::evaluate(
        &map,
        MomentReduction::Empirical(ProbabilityMoment::Covariance),
        &[outputs[1].clone(), outputs[0].clone()],
        &[1.0, 3.0],
    )
    .unwrap();
    assert_eq!(reversed.output_indices().unwrap(), [1, 0]);
    assert_eq!(reversed.weights(), [1.0, 3.0]);

    let (foreign_program, foreign_outputs) = named_fixture("Foreign");
    assert_ne!(foreign_program.identity(), program.identity());
    assert!(
        MomentEstimator::evaluate(
            &map,
            MomentReduction::Empirical(ProbabilityMoment::Expectation),
            &foreign_outputs[..1],
            &[1.0, 1.0]
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
    let sampler = ParameterSampler::new(
        program.clone(),
        [42; 32],
        &eqiora::api::DifferentiableValues::from_components(&[], &[]).unwrap(),
        &[[1.0, 2.0]],
    )
    .unwrap();
    let different_bounds = ParameterSampler::new(
        program.clone(),
        [42; 32],
        &eqiora::api::DifferentiableValues::from_components(&[], &[]).unwrap(),
        &[[2.0, 3.0]],
    )
    .unwrap();
    let different_master = ParameterSampler::new(
        program.clone(),
        [43; 32],
        &eqiora::api::DifferentiableValues::from_components(&[], &[]).unwrap(),
        &[[1.0, 2.0]],
    )
    .unwrap();
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
        MomentEstimator::evaluate(
            &map,
            MomentReduction::Empirical(ProbabilityMoment::Variance),
            &outputs[..1],
            &[1.0, 1.0],
        )
        .unwrap();
        let error = MomentEstimator::evaluate(
            &map,
            MomentReduction::UnbiasedIid(ProbabilityMoment::Variance),
            &outputs[..1],
            &[1.0, 1.0],
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
    MomentEstimator::evaluate(
        &deterministic,
        MomentReduction::Empirical(ProbabilityMoment::Expectation),
        &outputs[..1],
        &[1.0],
    )
    .unwrap();
    assert!(
        MomentEstimator::evaluate(
            &deterministic,
            MomentReduction::UnbiasedIid(ProbabilityMoment::Expectation),
            &outputs[..1],
            &[1.0]
        )
        .unwrap_err()
        .message()
        .contains("sample-design metadata")
    );
}

#[test]
fn estimator_normalization_and_minimum_count_are_explicit() {
    let (program, outputs) = fixture();
    let sampler = ParameterSampler::new(
        program.clone(),
        [42; 32],
        &eqiora::api::DifferentiableValues::from_components(&[], &[]).unwrap(),
        &[[1.0, 2.0]],
    )
    .unwrap();
    let samples = [
        sample(&sampler, 1, b"position", None),
        sample(&sampler, 2, b"position", None),
    ];
    let map = accepted(&program, &samples);
    assert!(
        MomentEstimator::evaluate(
            &map,
            MomentReduction::Population(ProbabilityMoment::Expectation),
            &outputs[..1],
            &[1.0, 1.0],
        )
        .unwrap_err()
        .message()
        .contains("exhaustive declared-law")
    );
    for weights in [
        vec![],
        vec![1.0],
        vec![0.0, 0.0],
        vec![-1.0, 2.0],
        vec![f64::NAN, 1.0],
        vec![1.0, f64::INFINITY],
    ] {
        assert!(
            MomentEstimator::evaluate(
                &map,
                MomentReduction::Empirical(ProbabilityMoment::Variance),
                &outputs[..1],
                &weights
            )
            .is_err()
        );
    }
    for weights in [[0.0, 1.0], [1.0, 3.0]] {
        MomentEstimator::evaluate(
            &map,
            MomentReduction::Empirical(ProbabilityMoment::Variance),
            &outputs[..1],
            &weights,
        )
        .unwrap();
        assert!(
            MomentEstimator::evaluate(
                &map,
                MomentReduction::UnbiasedIid(ProbabilityMoment::Variance),
                &outputs[..1],
                &weights
            )
            .is_err()
        );
    }
    // Validation must not overflow a floating total whose normalized ratios are finite.
    MomentEstimator::evaluate(
        &map,
        MomentReduction::UnbiasedIid(ProbabilityMoment::Variance),
        &outputs[..1],
        &[f64::MAX, f64::MAX],
    )
    .unwrap();
    let singleton = accepted(&program, &samples[..1]);
    MomentEstimator::evaluate(
        &singleton,
        MomentReduction::UnbiasedIid(ProbabilityMoment::Expectation),
        &outputs[..1],
        &[1.0],
    )
    .unwrap();
    assert!(
        MomentEstimator::evaluate(
            &singleton,
            MomentReduction::UnbiasedIid(ProbabilityMoment::Variance),
            &outputs[..1],
            &[1.0]
        )
        .is_err()
    );
    let empty = accepted(&program, &[]);
    assert!(
        MomentEstimator::evaluate(
            &empty,
            MomentReduction::Empirical(ProbabilityMoment::Expectation),
            &outputs[..1],
            &[]
        )
        .is_err()
    );
    assert!(
        MomentEstimator::evaluate(
            &map,
            MomentReduction::Empirical(ProbabilityMoment::Covariance),
            &outputs[..1],
            &[1.0, 1.0]
        )
        .is_err()
    );
}
