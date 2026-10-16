//! A named finite law supplies the weights, outcomes and loss for native AD.
use super::*;
use eqiora_core::ValueLiteral;
use eqiora_schema::kernel::SymbolRef;

fn document() -> ModelDocument {
    ModelDocument::compile_selected(
        "bernoulli.eqi",
        r#"enum Outcome {Low,High}
        model Loss() {
          parameter theta:1=1;
          distribution law:normalize_mass(Outcome,weights=[3,theta]);
          observable loss:1=random_variable(law,
            if outcome(law)==Outcome.Low then 2+theta else 6+2*theta);
          observable mean:1=expectation(loss);
        }"#,
        "Loss",
        &[],
    )
    .unwrap()
}

fn real(value: f64) -> ValueLiteral {
    ValueLiteral::try_from(DynQuantity::new(value, DimExponents::DIMENSIONLESS)).unwrap()
}

#[test]
fn finite_law_weights_and_losses_resolve_the_same_design_point() {
    let document = document();
    let parameter = document.parameter_ref("theta").unwrap().id();
    let mean = document.observable_ref("mean").unwrap().id();
    // E=(3(2+t)+t(6+2t))/(3+t). Both weights and losses must
    // use t=2 instead of the declaration's default t=1.
    let value = document
        .program()
        .evaluate_finite_observable(mean, &mut |symbol| {
            assert_eq!(symbol, SymbolRef::Parameter(parameter));
            Some(real(2.0))
        })
        .unwrap();
    assert_eq!(value, real(32.0 / 5.0));
    assert!(
        document
            .program()
            .evaluate_finite_observable(mean, &mut |_| Some(real(-1.0)))
            .is_err()
    );
    assert!(
        document
            .program()
            .evaluate_finite_observable(mean, &mut |_| None)
            .is_err()
    );
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    assert!(
        document
            .program()
            .evaluate_finite_observable(mean, &mut |_| {
                Some(ValueLiteral::try_from(DynQuantity::new(2.0, length)).unwrap())
            })
            .unwrap_err()
            .message()
            .contains("admitted type")
    );
}

#[test]
fn bernoulli_score_reference_differentiates_declared_normalized_weights() {
    let document = document();
    let parameter = document.parameter_ref("theta").unwrap();
    let loss = document.observable_ref("loss").unwrap();
    // Polynomial division gives E=2t+3-3/(t+3), hence E'=2+3/(t+3)^2.
    // At t=1 the direct loss term is 5/4 and the score term is 15/16.
    for (theta, expected, gradient) in [
        (1.0, 17.0 / 4.0, 35.0 / 16.0),
        (2.0, 32.0 / 5.0, 53.0 / 25.0),
    ] {
        let point =
            DifferentiableValues::from_parameters(&document, &[(parameter.clone(), real(theta))])
                .unwrap();
        let result = MomentEstimator::enumerated_score(&document, &loss, &point, 4096).unwrap();
        assert_eq!(result.value(), &real(expected));
        assert!((coordinate(result.derivative(), 0) - gradient).abs() <= 8.0 * f64::EPSILON);
        assert_eq!(
            result.reduction(),
            MomentReduction::Population(ProbabilityMoment::Expectation)
        );
        assert_eq!(result.weights(), &[3.0, theta]);
        assert_eq!(result.point(), Some(&point));
        assert_eq!(result.model(), parameter.model());
        assert!(result.map().is_none());
        assert!(result.output_indices().is_none());
        assert_eq!(result.outcomes().unwrap().len(), 2);
        assert_ne!(result.outcomes().unwrap()[0], result.outcomes().unwrap()[1]);
        assert_eq!(
            result.outcome_values().unwrap(),
            &[real(2.0 + theta), real(6.0 + 2.0 * theta)]
        );
        let scores = result.scores().unwrap();
        for (score, expected) in scores
            .iter()
            .zip([-1.0 / (3.0 + theta), 3.0 / (theta * (3.0 + theta))])
        {
            assert!((coordinate(score, 0) - expected).abs() <= 4.0 * f64::EPSILON);
        }
        match result.gradient_estimator() {
            eqiora::api::GradientEstimator::EnumeratedScore {
                variable,
                distribution_parameters,
                ..
            } => {
                assert_eq!(*variable, loss.id());
                assert_eq!(distribution_parameters, &[parameter.id()]);
            }
            _ => panic!("score interpretation was lost"),
        }
        assert!(result.standard_error().is_err());
    }
}

fn coordinate(value: &eqiora::api::DifferentiableCotangent, column: usize) -> f64 {
    value.components(column).unwrap().next().unwrap().0
}

#[test]
fn score_normalization_cancels_common_mass_scale_in_either_parameter_order() {
    let document = ModelDocument::compile_selected(
        "scale.eqi",
        r#"
        enum Outcome {Low,High}
        model Loss() {
          parameter theta:1=1; parameter common:1=2;
          distribution law:normalize_mass(Outcome,weights=[3*common,theta*common]);
          observable loss:1=random_variable(law,
            if outcome(law)==Outcome.Low then 2+theta else 6+2*theta);
        }"#,
        "Loss",
        &[],
    )
    .unwrap();
    for names in [["theta", "common"], ["common", "theta"]] {
        let point = DifferentiableValues::from_parameters(
            &document,
            &names
                .iter()
                .map(|name| {
                    (
                        document.parameter_ref(name).unwrap(),
                        real(if *name == "theta" { 1.0 } else { 2.0 }),
                    )
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let result = MomentEstimator::enumerated_score(
            &document,
            &document.observable_ref("loss").unwrap(),
            &point,
            4096,
        )
        .unwrap();
        assert_eq!(result.value(), &real(17.0 / 4.0));
        for (column, name) in names.iter().enumerate() {
            assert_eq!(
                coordinate(result.derivative(), column),
                if *name == "theta" { 35.0 / 16.0 } else { 0.0 }
            );
            if *name == "common" {
                for score in result.scores().unwrap() {
                    assert_eq!(coordinate(score, column), 0.0);
                }
            }
        }
    }
}

#[test]
fn score_rejects_support_boundaries_foreign_models_and_exhausted_resources() {
    let document = document();
    let parameter = document.parameter_ref("theta").unwrap();
    let loss = document.observable_ref("loss").unwrap();
    for theta in [0.0, -1.0] {
        let point =
            DifferentiableValues::from_parameters(&document, &[(parameter.clone(), real(theta))])
                .unwrap();
        let error = MomentEstimator::enumerated_score(&document, &loss, &point, 4096).unwrap_err();
        assert!(error.message().contains("strictly positive"), "{error}");
    }
    let point = DifferentiableValues::from_parameters(&document, &[(parameter.clone(), real(1.0))])
        .unwrap();
    assert!(
        MomentEstimator::enumerated_score(&document, &loss, &point, 0)
            .unwrap_err()
            .message()
            .contains("byte limit")
    );
    let foreign = ModelDocument::compile_selected(
        "foreign.eqi",
        "model Other() {parameter theta:1=1; variable x:1; relation balance {x=theta;}}",
        "Other",
        &[],
    )
    .unwrap();
    assert!(
        DifferentiableValues::from_parameters(&foreign, &[(parameter.clone(), real(1.0))]).is_err()
    );
    let foreign_point = DifferentiableValues::from_parameters(
        &foreign,
        &[(foreign.parameter_ref("theta").unwrap(), real(1.0))],
    )
    .unwrap();
    assert!(
        MomentEstimator::enumerated_score(&document, &loss, &foreign_point, 4096)
            .unwrap_err()
            .message()
            .contains("exact real scalar")
    );
    assert!(
        MomentEstimator::enumerated_score(&foreign, &loss, &point, 4096)
            .unwrap_err()
            .message()
            .contains("exact Model")
    );
    assert!(
        DifferentiableValues::from_parameters(
            &document,
            &[(parameter.clone(), real(1.0)), (parameter, real(1.0))]
        )
        .is_err()
    );
}

#[test]
fn score_units_and_frozen_defaults_follow_the_exact_design_signature() {
    let document = ModelDocument::compile_selected(
        "units.eqi",
        r#"
        enum Outcome {Low,High}
        model Loss() {
          parameter duration:s=1[s]; parameter length:m=2[m];
          distribution law:normalize_mass(Outcome,weights=[3,duration/1[s]]);
          observable loss:m=random_variable(law,
            if outcome(law)==Outcome.Low then length+duration*1[m/s]
            else 3*length+2*duration*1[m/s]);
        }"#,
        "Loss",
        &[],
    )
    .unwrap();
    let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let input = document.parameter_ref("duration").unwrap();
    let point = DifferentiableValues::from_parameters(
        &document,
        &[(
            input.clone(),
            ValueLiteral::try_from(DynQuantity::new(2.0, time)).unwrap(),
        )],
    )
    .unwrap();
    let result = MomentEstimator::enumerated_score(
        &document,
        &document.observable_ref("loss").unwrap(),
        &point,
        4096,
    )
    .unwrap();
    assert_eq!(
        result.value(),
        &ValueLiteral::try_from(DynQuantity::new(32.0 / 5.0, length)).unwrap()
    );
    assert!((coordinate(result.derivative(), 0) - 53.0 / 25.0).abs() <= 8.0 * f64::EPSILON);
    assert_eq!(result.derivative().base_signatures(), point.signatures());
    assert!(DifferentiableValues::from_parameters(&document, &[(input, real(2.0))]).is_err());
}

#[test]
fn score_rejects_an_active_loss_boundary_and_unresolved_observable_dependencies() {
    for (expression, diagnostic) in [
        ("if theta==1 then theta else 2*theta", "boundary"),
        ("base+theta", "inter-Observable"),
    ] {
        let source = format!(
            r#"
          enum Outcome {{Low,High}}
          model Loss() {{
            parameter theta:1=1;
            distribution law:normalize_mass(Outcome,weights=[3,theta]);
            observable base:1=random_variable(law,2*theta);
            observable loss:1=random_variable(law,{expression});
          }}"#
        );
        let document =
            ModelDocument::compile_selected("rejection.eqi", &source, "Loss", &[]).unwrap();
        let point = DifferentiableValues::from_parameters(
            &document,
            &[(document.parameter_ref("theta").unwrap(), real(1.0))],
        )
        .unwrap();
        let error = MomentEstimator::enumerated_score(
            &document,
            &document.observable_ref("loss").unwrap(),
            &point,
            4096,
        )
        .unwrap_err();
        assert!(error.message().contains(diagnostic), "{error}");
    }
}
