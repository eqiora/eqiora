//! Admission probes reach the estimator after ordinary finite objectives succeed.
use super::*;

fn document(variable: &str, loss: &str) -> ModelDocument {
    let source = r#"operator squared(input value:1):1=value*value;
      model Loss(support noise:interval(1)) {
        parameter location:1=2; parameter scale:1=3; parameter draw:1=0;
        coordinate z:1 on noise from noise[0];
        distribution base:normalize_density(measure(noise),density=1);
        observable x:1=random_variable(base,VARIABLE);
        observable loss:1=evaluate(LOSS,at=(z=draw));
        variable result:1; relation equation {result=loss;}
    }"#
    .replace("VARIABLE", variable)
    .replace("LOSS", loss);
    ModelDocument::compile_selected(
        "pathwise.eqi",
        &source,
        "Loss",
        &[(
            "noise",
            StaticBindingValue::CoordinateInterval(
                AxisBounds::new(
                    DynQuantity::new(-1.0, DimExponents::DIMENSIONLESS),
                    DynQuantity::new(1.0, DimExponents::DIMENSIONLESS),
                )
                .unwrap(),
            ),
        )],
    )
    .unwrap()
}

fn map(
    document: &ModelDocument,
    bounds: [f64; 2],
    scale: f64,
    coupled: bool,
) -> (CompleteEvaluationMap, ModelEntityRef) {
    let (program, output) = program(document, &["location", "scale", "draw"]);
    let fixed = DifferentiableValues::from_components(
        &program.identity().inputs()[..2],
        &[&[(2.0, 0.0)], &[(scale, 0.0)]],
    )
    .unwrap();
    let sampler = ParameterSampler::new(program.clone(), [31; 32], &fixed, &[bounds]).unwrap();
    let samples = (0..3)
        .map(|id| {
            sampler
                .sample(
                    &SamplingIdentity::new(
                        &[id],
                        b"base",
                        &[],
                        if coupled { Some(b"comparison") } else { None },
                    )
                    .unwrap(),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    let shared = program.identity().inputs()[..2]
        .iter()
        .map(|input| input.entity().downcast().unwrap())
        .collect::<Vec<_>>();
    (
        EvaluationMapPlan::from_samples(
            program,
            &samples,
            &shared,
            EvaluationMapExecutionPolicy::retained(1 << 24),
        )
        .unwrap()
        .execute()
        .unwrap(),
        output,
    )
}

#[test]
fn pathwise_admission_preserves_coupling_and_rejects_detached_designs() {
    for (variable, loss, rejection) in [
        ("location+scale*z", "(x-1)^2", None),
        ("scale*z+location", "squared(value=x-1)", None),
        ("2+scale*z", "(x-1)^2", Some("selected location + scale")),
        ("location+3*z", "(x-1)^2", Some("selected location + scale")),
        (
            "location+scale*(z+location)",
            "(x-1)^2",
            Some("design-independent base law"),
        ),
        (
            "location+scale*z",
            "(location+scale*z-1)^2",
            Some("detached"),
        ),
    ] {
        let document = document(variable, loss);
        let (map, output) = map(&document, [-1.0, 1.0], 3.0, true);
        let estimator = MomentEstimator::evaluate(
            &map,
            MomentReduction::Empirical(ProbabilityMoment::Expectation),
            &[output],
            &[1.0; 3],
        )
        .unwrap();
        // Every mutant is a valid finite objective and gradient. Rejection
        // must come from the stronger declared expectation interpretation.
        let finite = estimator.vjp(1 << 24).unwrap();
        let admitted = estimator.pathwise_location_scale(
            &document.observable_ref("x").unwrap(),
            &document.parameter_ref("location").unwrap(),
            &document.parameter_ref("scale").unwrap(),
        );
        if let Some(rejection) = rejection {
            assert!(admitted.unwrap_err().message().contains(rejection));
        } else {
            let estimator = admitted.unwrap();
            let gradient = estimator.vjp(1 << 24).unwrap();
            assert_eq!(
                gradient.derivative().shared_cotangents(),
                finite.derivative().shared_cotangents()
            );
            assert_eq!(
                gradient.gradient_estimator(),
                estimator.gradient_estimator()
            );
            for sample in gradient.map().unwrap().plan().samples().unwrap() {
                assert_eq!(
                    sample.identity().coupling_id(),
                    Some(b"comparison".as_slice())
                );
            }
        }
    }
}

#[test]
fn pathwise_admission_rejects_changed_support_scale_weights_and_model() {
    let document = document("location+scale*z", "(x-1)^2");
    for (bounds, scale, weights, rejection) in [
        ([-0.5, 0.5], 3.0, [1.0; 3], "sample design differs"),
        ([-1.0, 1.0], -3.0, [1.0; 3], "positive scale"),
        (
            [-1.0, 1.0],
            3.0,
            [1.0, 2.0, 1.0],
            "equal positive mean weights",
        ),
    ] {
        let (map, output) = map(&document, bounds, scale, false);
        let estimator = MomentEstimator::evaluate(
            &map,
            MomentReduction::Empirical(ProbabilityMoment::Expectation),
            &[output],
            &weights,
        )
        .unwrap();
        estimator.vjp(1 << 24).unwrap();
        let error = estimator
            .pathwise_location_scale(
                &document.observable_ref("x").unwrap(),
                &document.parameter_ref("location").unwrap(),
                &document.parameter_ref("scale").unwrap(),
            )
            .unwrap_err();
        assert!(error.message().contains(rejection), "{error:?}");
    }
    let (map, output) = map(&document, [-1.0, 1.0], 3.0, false);
    let foreign = self::document("location+scale*z", "(x-2)^2");
    let estimator = MomentEstimator::evaluate(
        &map,
        MomentReduction::Empirical(ProbabilityMoment::Expectation),
        &[output],
        &[1.0; 3],
    )
    .unwrap();
    assert!(
        estimator
            .pathwise_location_scale(
                &foreign.observable_ref("x").unwrap(),
                &document.parameter_ref("location").unwrap(),
                &document.parameter_ref("scale").unwrap(),
            )
            .unwrap_err()
            .message()
            .contains("exact location-scale selection")
    );
}
