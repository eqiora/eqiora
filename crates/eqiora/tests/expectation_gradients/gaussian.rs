//! Independently declared Gaussian squared-loss moments and frozen pathwise actions.
use super::*;

#[test]
fn gaussian_pushforward_fixed_paths_match_independent_moments() {
    // With U,V uniform on the unit square, polar change of variables gives
    // Z=sqrt(-2 log U) cos(2 pi V) a standard normal distribution.
    // X=mu+sigma*Z, L=(X-t)^2, a=mu-t give
    // E[L]=a²+sigma², E[d_mu L]=2a, E[d_sigma L]=2sigma.
    // Their variances are 4a²sigma²+2sigma⁴, 4sigma² and 4a²+8sigma².
    // Fix N and the six-reference-standard-error criterion before drawing.
    // This deterministic finite-grid comparison is not an unbiasedness proof.
    const N: usize = 1024;
    const K: f64 = 6.0;
    let unit = StaticBindingValue::CoordinateInterval(
        AxisBounds::new(
            DynQuantity::new(0.0, DimExponents::DIMENSIONLESS),
            DynQuantity::new(1.0, DimExponents::DIMENSIONLESS),
        )
        .unwrap(),
    );
    let document = ModelDocument::compile_selected(
        "gaussian.eqi",
        r#"
      model Loss(support radial:interval(1), support angular:interval(1)) {
        support noise:product(radial,angular);
        parameter location:1=2; parameter scale:1=3;
        parameter radial_draw:1=0.5; parameter angular_draw:1=0.5;
        coordinate u:1 on noise from radial[0]; coordinate v:1 on noise from angular[0];
        distribution base:normalize_density(measure(noise),density=1);
        observable z:1=random_variable(base,math.sqrt(-2*math.log(u))*math.cos(2*math.pi*v));
        observable x:1=random_variable(base,location+scale*z);
        observable loss:1=evaluate((x-1)^2,at=(u=radial_draw,v=angular_draw));
        variable result:1; relation equation {result=loss;}
      }"#,
        "Loss",
        &[("radial", unit), ("angular", unit)],
    )
    .unwrap();
    let (program, output) = program(
        &document,
        &["location", "scale", "radial_draw", "angular_draw"],
    );
    let fixed = DifferentiableValues::from_components(
        &program.identity().inputs()[..2],
        &[&[(2.0, 0.0)], &[(3.0, 0.0)]],
    )
    .unwrap();
    let sampler =
        ParameterSampler::new(program.clone(), [29; 32], &fixed, &[[0.0, 1.0]; 2]).unwrap();
    let samples = (0..N)
        .map(|id| {
            sampler
                .sample(
                    &SamplingIdentity::new(&(id as u64).to_le_bytes(), b"gaussian-base", &[], None)
                        .unwrap(),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut finite = [0.0; 3];
    for sample in &samples {
        let coordinate = |index| {
            sample
                .point()
                .leaf(index)
                .unwrap()
                .components()
                .next()
                .unwrap()
                .0
        };
        let z = (-2.0 * coordinate(2).ln()).sqrt() * (std::f64::consts::TAU * coordinate(3)).cos();
        let residual = 1.0 + 3.0 * z;
        for (sum, value) in
            finite
                .iter_mut()
                .zip([residual * residual, 2.0 * residual, 2.0 * z * residual])
        {
            *sum += value / (N as f64);
        }
    }
    let shared = program.identity().inputs()[..2]
        .iter()
        .map(|input| input.entity().downcast().unwrap())
        .collect::<Vec<_>>();
    let map = EvaluationMapPlan::from_samples(
        program.clone(),
        &samples,
        &shared,
        EvaluationMapExecutionPolicy::retained(1 << 28),
    )
    .unwrap()
    .execute()
    .unwrap();
    let outputs = [output];
    let weights = vec![1.0; N];
    let estimator = EvaluationMapEstimator::evaluate(
        &map,
        EnsembleReduction::Empirical(ProbabilityMoment::Expectation),
        &outputs,
        &weights,
    )
    .unwrap();
    let gradient = estimator.vjp(1 << 28).unwrap();
    let actual = [
        estimator.value().real_scalar_value().unwrap().value(),
        gradient.shared_cotangents()[0]
            .components(0)
            .unwrap()
            .next()
            .unwrap()
            .0,
        gradient.shared_cotangents()[0]
            .components(1)
            .unwrap()
            .next()
            .unwrap()
            .0,
    ];
    for ((actual, finite), (mean, variance)) in
        actual
            .into_iter()
            .zip(finite)
            .zip([(10.0, 198.0_f64), (2.0, 36.0), (6.0, 76.0)])
    {
        let arithmetic = 16.0 * N as f64 * f64::EPSILON * finite.abs().max(1.0);
        assert!(
            (actual - finite).abs() <= arithmetic,
            "finite AD: {actual} != {finite}"
        );
        let criterion = K * (variance / N as f64).sqrt();
        assert!(
            (actual - mean).abs() <= criterion,
            "Gaussian reference: {actual} != {mean} within {criterion}"
        );
    }
    for sample in &samples {
        assert_eq!(sampler.sample(sample.identity()).unwrap(), *sample);
    }
    // The transform's singular endpoint rejects the entire evaluation. It must
    // never be repaired by replacing that draw or removing it from a mean.
    let singular = DifferentiableValues::from_components(
        program.identity().inputs(),
        &[&[(2.0, 0.0)], &[(3.0, 0.0)], &[(0.0, 0.0)], &[(0.5, 0.0)]],
    )
    .unwrap();
    assert!(program.evaluate(&singular).is_err());
}
