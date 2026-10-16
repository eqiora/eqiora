use super::*;
use eqiora::api::{DifferentiableValues, EvaluationMapRetention};
use eqiora_realization::Target;

fn policies() -> [EvaluationMapExecutionPolicy; 3] {
    [
        EvaluationMapExecutionPolicy::retained(1 << 28),
        EvaluationMapExecutionPolicy::new(
            2,
            Target::HostCpu {
                threads: NonZeroUsize::new(2).unwrap(),
            },
            EvaluationMapRetention::Retain,
            1 << 28,
        )
        .unwrap(),
        EvaluationMapExecutionPolicy::new(
            3,
            Target::HostCpu {
                threads: NonZeroUsize::new(2).unwrap(),
            },
            EvaluationMapRetention::Recompute,
            1 << 28,
        )
        .unwrap(),
    ]
}

fn partitioned(
    program: &Arc<DifferentiableProgram>,
    values: &[f64],
    policy: EvaluationMapExecutionPolicy,
) -> CompleteEvaluationMap {
    let shared =
        DifferentiableValues::from_components(&program.identity().inputs()[..1], &[&[(1.0, 0.0)]])
            .unwrap();
    let mapped = values
        .iter()
        .map(|&x| {
            DifferentiableValues::from_components(&program.identity().inputs()[1..], &[&[(x, 0.0)]])
                .unwrap()
        })
        .collect::<Vec<_>>();
    EvaluationMapPlan::from_partition(program.clone(), &shared, &mapped, policy)
        .unwrap()
        .execute()
        .unwrap()
}

#[test]
fn hand_derived_statistics_chain_into_shared_and_mapped_covectors() {
    let (program, outputs) = compile_fixture("M", true);
    // X=gain*position, Y=3-position, gain=1, positions=[0,2,2], weights=[1,2,1].
    // Sum=6; mean=3/2; variance=3/4; covariance=-3/4.
    // gain derivatives are sum=6, mean=3/2, variance=3/2, covariance=-3/4.
    for policy in policies() {
        let map = partitioned(&program, &[0.0, 2.0, 2.0], policy);
        for (reduction, selected, value, shared, mapped) in [
            (
                EnsembleReduction::Sum,
                &outputs[..1],
                6.0,
                6.0,
                [1.0, 2.0, 1.0],
            ),
            (
                EnsembleReduction::Empirical(ProbabilityMoment::Expectation),
                &outputs[..1],
                1.5,
                1.5,
                [0.25, 0.5, 0.25],
            ),
            (
                EnsembleReduction::Empirical(ProbabilityMoment::Variance),
                &outputs[..1],
                0.75,
                1.5,
                [-0.75, 0.5, 0.25],
            ),
            (
                EnsembleReduction::Empirical(ProbabilityMoment::Covariance),
                &outputs[..],
                -0.75,
                -0.75,
                [0.75, -0.5, -0.25],
            ),
        ] {
            let estimator =
                EvaluationMapEstimator::evaluate(&map, reduction, selected, &[1.0, 2.0, 1.0])
                    .unwrap();
            assert_eq!(
                estimator.value().real_scalar_value().unwrap().value(),
                value
            );
            assert!(estimator.standard_error().is_err());
            let gradient = estimator.vjp(1 << 20).unwrap();
            assert!(gradient.shared_shape().is_empty());
            assert_eq!(gradient.mapped_shape(), [3]);
            assert_eq!(
                gradient.shared_cotangents()[0]
                    .components(0)
                    .unwrap()
                    .collect::<Vec<_>>(),
                [(shared, 0.0)]
            );
            for (product, expected) in gradient.mapped_cotangents().iter().zip(mapped) {
                assert_eq!(
                    product.components(0).unwrap().collect::<Vec<_>>(),
                    [(expected, 0.0)]
                );
            }
            assert_eq!(gradient.products().len(), 3);
            assert!(estimator.vjp(0).is_err());
        }
        // Repeated output selections combine both chain-rule contributions.
        let same = [outputs[0].clone(), outputs[0].clone()];
        let covariance = EvaluationMapEstimator::evaluate(
            &map,
            EnsembleReduction::Empirical(ProbabilityMoment::Covariance),
            &same,
            &[1.0, 2.0, 1.0],
        )
        .unwrap();
        assert_eq!(
            covariance.value().real_scalar_value().unwrap().value(),
            0.75
        );
        assert_eq!(
            covariance.vjp(1 << 20).unwrap().shared_cotangents()[0]
                .components(0)
                .unwrap()
                .collect::<Vec<_>>(),
            [(1.5, 0.0)]
        );
    }
}

#[test]
fn cancellation_is_independent_of_chunks_workers_and_retention() {
    let (program, outputs) = compile_fixture("M", true);
    for policy in policies() {
        let map = partitioned(&program, &[1e16, 1.0, -1e16], policy);
        for (reduction, expected) in [
            (EnsembleReduction::Sum, 1.0),
            (
                EnsembleReduction::Empirical(ProbabilityMoment::Expectation),
                1.0 / 3.0,
            ),
        ] {
            let result =
                EvaluationMapEstimator::evaluate(&map, reduction, &outputs[..1], &[1.0; 3])
                    .unwrap();
            assert_eq!(
                result.value().real_scalar_value().unwrap().value(),
                expected
            );
        }
    }
}

#[test]
fn standard_error_requires_explicit_iid_mean_and_finite_count() {
    let (program, outputs) = fixture();
    let sampler = ParameterSampler::new(program.clone(), [42; 32], &[[1.0, 2.0]]).unwrap();
    let samples = [
        sample(&sampler, 1, b"x", None),
        sample(&sampler, 2, b"x", None),
    ];
    let map = accepted(&program, &samples);
    let values = map
        .members()
        .unwrap()
        .iter()
        .map(|member| {
            member
                .primal()
                .output()
                .leaf(0)
                .unwrap()
                .coefficient(0)
                .unwrap()
                .real_scalar_value()
                .unwrap()
                .value()
        })
        .collect::<Vec<_>>();
    // For two points, s²=(x1-x2)²/2 and SE=sqrt(s²/2)=|x1-x2|/2.
    let expected = (values[0] - values[1]).abs() / 2.0;
    let mean = EvaluationMapEstimator::evaluate(
        &map,
        EnsembleReduction::UnbiasedIid(ProbabilityMoment::Expectation),
        &outputs[..1],
        &[3.0, 3.0],
    )
    .unwrap();
    let error = mean.standard_error().unwrap();
    assert_eq!(error.value_type(), mean.value_type());
    assert_eq!(error.real_scalar_value().unwrap().value(), expected);
    let singleton = accepted(&program, &samples[..1]);
    assert!(
        EvaluationMapEstimator::evaluate(
            &singleton,
            EnsembleReduction::UnbiasedIid(ProbabilityMoment::Expectation),
            &outputs[..1],
            &[1.0]
        )
        .unwrap()
        .standard_error()
        .is_err()
    );
    let variance = EvaluationMapEstimator::evaluate(
        &map,
        EnsembleReduction::UnbiasedIid(ProbabilityMoment::Variance),
        &outputs[..1],
        &[1.0, 1.0],
    )
    .unwrap();
    assert!(variance.standard_error().is_err());
}
