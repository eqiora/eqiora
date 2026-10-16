//! A frozen finite objective is distinct from an admitted population derivative.
use eqiora::api::{
    CompleteEvaluationMap, DifferentiableProgram, DifferentiableValues, EnsembleReduction,
    EvaluationMapEstimator, EvaluationMapExecutionPolicy, EvaluationMapPlan, ModelDocument,
    ParameterSampler, SamplingIdentity,
};
use eqiora_artifact::{CanonicalModelArtifact, ModelEnvelope};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_compiler::StaticBindingValue;
use eqiora_core::{DimExponents, DynQuantity, ValueShape};
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonInitialField, CommonLinearRequest, CommonSolvePolicy,
    ResolvedCommonPlan,
};
use eqiora_realization::NonlinearSolvePlan;
use eqiora_schema::kernel::{AxisBounds, ProbabilityMoment};
use eqiora_solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
use std::{num::NonZeroUsize, sync::Arc};

#[test]
fn fixed_sample_objective_retains_law_realization_and_parameter_paths() {
    let document = ModelDocument::compile_selected(
        "gradient.eqi",
        r#"
      model Loss(support noise:interval(1)) {
        parameter location:1=2; parameter scale:1=3; parameter draw:1=0;
        coordinate z:1 on noise from noise[0];
        distribution base:normalize_density(measure(noise),density=1);
        observable x:1=random_variable(base,location+scale*z);
        observable loss:1=evaluate((x-1)^2,at=(z=draw));
        variable result:1; relation equation {result=loss;}
      }"#,
        "Loss",
        &[(
            "noise",
            StaticBindingValue::CoordinateInterval(
                AxisBounds::new(
                    DynQuantity::new(-2.0, DimExponents::DIMENSIONLESS),
                    DynQuantity::new(2.0, DimExponents::DIMENSIONLESS),
                )
                .unwrap(),
            ),
        )],
    )
    .unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let linear = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-12,
        1e-14,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let plan = CommonAlgebraicPlan::resolve(
        &model,
        CommonSolvePolicy::Newton {
            nonlinear: NonlinearSolvePlan::new(0.0, 1e-12, NonZeroUsize::new(8).unwrap(), 8)
                .unwrap(),
            linear: CommonLinearRequest::exact(linear, FaerLinearSolver.provider()).unwrap(),
        },
        None,
        &[],
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    let initial = plan
        .initial_state(&[CommonInitialField::finite(
            model.artifact_reference().unwrap().artifact().clone(),
            document.aliases()["result"].downcast().unwrap(),
            ValueShape::scalar(),
            vec![(0.0, 0.0)],
        )
        .unwrap()])
        .unwrap();
    let inputs = ["location", "scale", "draw"].map(|name| document.parameter_ref(name).unwrap());
    let outputs = [document.field_ref("result").unwrap().erase()];
    let program = Arc::new(
        DifferentiableProgram::compile(
            ResolvedCommonPlan::Algebraic(Box::new(plan)),
            &inputs,
            &outputs,
            Some(initial),
            &FaerLinearSolver,
        )
        .unwrap(),
    );
    let fixed = DifferentiableValues::from_components(
        &program.identity().inputs()[..2],
        &[&[(2.0, 0.0)], &[(3.0, 0.0)]],
    )
    .unwrap();
    let sampler = ParameterSampler::new(program.clone(), [17; 32], &fixed, &[[-0.5, 0.5]]).unwrap();
    let samples = (0..3)
        .map(|id| {
            sampler
                .sample(&SamplingIdentity::new(&[id], b"base-noise", &[], None).unwrap())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let draws = samples
        .iter()
        .map(|sample| {
            sample
                .point()
                .leaf(2)
                .unwrap()
                .components()
                .next()
                .unwrap()
                .0
        })
        .collect::<Vec<_>>();
    let map: CompleteEvaluationMap = EvaluationMapPlan::from_samples(
        program.clone(),
        &samples,
        &program.identity().inputs()[..2]
            .iter()
            .map(|input| input.entity().downcast().unwrap())
            .collect::<Vec<_>>(),
        EvaluationMapExecutionPolicy::retained(1 << 28),
    )
    .unwrap()
    .execute()
    .unwrap();
    let estimator = EvaluationMapEstimator::evaluate(
        &map,
        EnsembleReduction::Empirical(ProbabilityMoment::Expectation),
        &outputs,
        &[1.0; 3],
    )
    .unwrap();
    let gradient = estimator.vjp(1 << 20).unwrap();
    let close = |actual: f64, expected: f64| {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}")
    };
    // At location=2, scale=3 and target=1: each loss=(1+3z)^2,
    // d_location=2(1+3z), d_scale=2z(1+3z). This is the exact
    // finite-sample objective; the uniform population gradient differs.
    close(
        estimator.value().real_scalar_value().unwrap().value(),
        draws.iter().map(|z| (1.0 + 3.0 * z).powi(2)).sum::<f64>() / 3.0,
    );
    for (index, expected) in [
        draws.iter().map(|z| 2.0 * (1.0 + 3.0 * z)).sum::<f64>() / 3.0,
        draws.iter().map(|z| 2.0 * z * (1.0 + 3.0 * z)).sum::<f64>() / 3.0,
    ]
    .into_iter()
    .enumerate()
    {
        close(
            gradient.shared_cotangents()[0]
                .components(index)
                .unwrap()
                .next()
                .unwrap()
                .0,
            expected,
        );
    }
    for sample in &samples {
        assert_eq!(sampler.sample(sample.identity()).unwrap(), *sample);
    }
    let outside = DifferentiableValues::from_components(
        program.identity().inputs(),
        &[&[(2.0, 0.0)], &[(3.0, 0.0)], &[(2.5, 0.0)]],
    )
    .unwrap();
    assert!(program.evaluate(&outside).is_err());
}
