//! Finite moving limits retain their exact factor and dimension before quadrature.
use eqiora::api::ModelDocument;
use eqiora::compiler::StaticBindingValue;
use eqiora::kernel::AxisBounds;
use eqiora::{DimExponents, DynQuantity};

const SOURCE: &str = r#"
model Moving(support line:interval(m)) {
  coordinate x:m on line from line;
  parameter a:m=1[m];
  variable anchor:1;
  relation fixed {anchor=1;}
  observable total:m^3=integral(x*x,measure(line),lower=0[m],upper=a);
  observable slope:m^2=partial(total,wrt=a);
}
"#;

fn document(source: &str) -> Result<ModelDocument, Vec<eqiora::Diagnostic>> {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    ModelDocument::compile_selected(
        "moving-integral.eqi",
        source,
        "Moving",
        &[(
            "line",
            StaticBindingValue::CoordinateInterval(
                AxisBounds::new(
                    DynQuantity::new(-4.0, length),
                    DynQuantity::new(4.0, length),
                )
                .unwrap(),
            ),
        )],
    )
}

#[test]
fn moving_limits_retain_their_parameter_and_dimension_through_model_replay() {
    let document = document(SOURCE).unwrap();
    let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
    assert_eq!(document.program(), replay.program());
}

#[test]
fn finite_polynomial_limits_preserve_orientation_and_exact_support() {
    use eqiora::artifact::ModelEnvelope;
    use eqiora::meshing::QuadratureRule;
    use eqiora_backend_faer::FaerLinearSolver;
    use std::collections::HashMap;
    for a in [-1.0_f64, 0.0, 1.0, 2.0] {
        for (density, lower, upper, power, expected, derivative) in [
            ("x*x", "0[m]", "a", 3, a.powi(3) / 3.0, a * a),
            (
                "a*x",
                "a",
                "2*a",
                3,
                3.0 * a.powi(3) / 2.0,
                9.0 * a * a / 2.0,
            ),
            ("1", "a", "2*a", 1, a, 1.0),
        ] {
            let source = SOURCE.replace("a:m=1[m]", &format!("a:m={a}[m]")).replace(
                "integral(x*x,measure(line),lower=0[m],upper=a)",
                &format!("integral({density},measure(line),lower={lower},upper={upper})"),
            );
            let source = source
                .replace("total:m^3", &format!("total:m^{power}"))
                .replace("slope:m^2", &format!("slope:m^{}", power - 1));
            let document = document(&source).unwrap();
            let model = ModelEnvelope::from_program(document.program()).unwrap();
            let plan = super::resolve(&model).unwrap();
            let result = plan
                .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
                .unwrap();
            let rules = HashMap::from([(
                document.aliases()["line"].downcast().unwrap(),
                QuadratureRule::gauss_legendre(2).unwrap(),
            )]);
            let value = result
                .observe(
                    &model,
                    document.aliases()["total"].downcast().unwrap(),
                    &rules,
                )
                .unwrap();
            assert!((value.value().real_scalar_value().unwrap().value() - expected).abs() < 1e-12);
            let slope = result
                .observe(
                    &model,
                    document.aliases()["slope"].downcast().unwrap(),
                    &rules,
                )
                .unwrap();
            let slope = slope.value().real_scalar_value().unwrap();
            assert!((slope.value() - derivative).abs() < 1e-12);
            assert_eq!(
                slope.dim(),
                DimExponents::from_integers([0, power - 1, 0, 0, 0, 0, 0]).unwrap()
            );
        }
    }
}

#[test]
fn moving_limits_reject_capture_units_and_unjustified_exchange() {
    document(SOURCE).unwrap();
    for (from, to, message) in [
        ("upper=a", "upper=x", "bound coordinate cannot be captured"),
        ("upper=a", "upper=a*a", "exact coordinate unit"),
        (",lower=0[m]", "", "both lower and upper"),
        ("x*x,measure", "1[m^4]/(x-a)^2,measure", "polynomial"),
    ] {
        let errors = document(&SOURCE.replace(from, to)).unwrap_err();
        assert!(
            errors.iter().any(|error| error.message().contains(message)),
            "{to}: {errors:?}"
        );
    }
}

#[test]
fn limits_bind_semantic_identity_and_reject_displaced_wire() {
    let original = document(SOURCE).unwrap();
    let renamed = document(
        &SOURCE
            .replace("coordinate x:", "coordinate position:")
            .replace("x*x", "position*position"),
    )
    .unwrap();
    assert!(original.structurally_equivalent(&renamed).unwrap());
    let reordered = document(&SOURCE.replace("  parameter a:m=1[m];", "").replace(
        "  observable slope",
        "  parameter a:m=1[m];\n  observable slope",
    ))
    .unwrap();
    assert!(original.structurally_equivalent(&reordered).unwrap());
    let swapped = document(&SOURCE.replace("lower=0[m],upper=a", "lower=a,upper=0[m]")).unwrap();
    assert!(!original.structurally_equivalent(&swapped).unwrap());
    let wire = String::from_utf8(original.canonical_json().unwrap()).unwrap();
    let displaced = wire.replace("eqiora.model-envelope/v43", "eqiora.model-envelope/v37");
    assert_ne!(wire, displaced);
    assert!(ModelDocument::replay(displaced.as_bytes()).is_err());
}

#[test]
fn moving_limits_cannot_escape_the_declared_coordinate_interval() {
    use eqiora::artifact::ModelEnvelope;
    use eqiora::meshing::QuadratureRule;
    use eqiora_backend_faer::FaerLinearSolver;
    use std::collections::HashMap;
    let document = document(&SOURCE.replace("a:m=1[m]", "a:m=5[m]")).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = super::resolve(&model).unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    let rules = HashMap::from([(
        document.aliases()["line"].downcast().unwrap(),
        QuadratureRule::gauss_legendre(2).unwrap(),
    )]);
    for name in ["total", "slope"] {
        let error = result
            .observe(&model, document.aliases()[name].downcast().unwrap(), &rules)
            .unwrap_err();
        assert!(error.message().contains("outside"), "{error:?}");
    }
}

#[test]
fn moving_limits_reject_a_foreign_factor_even_with_identical_units_and_bounds() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let interval = StaticBindingValue::CoordinateInterval(
        AxisBounds::new(
            DynQuantity::new(-4.0, length),
            DynQuantity::new(4.0, length),
        )
        .unwrap(),
    );
    let source = SOURCE.replace(
        "support line:interval(m)",
        "support line:interval(m), support other:interval(m)",
    );
    ModelDocument::compile_selected(
        "positive.eqi",
        &source,
        "Moving",
        &[("line", interval), ("other", interval)],
    )
    .unwrap();
    let source = source.replace(
        "coordinate x:m on line from line",
        "coordinate x:m on other from other",
    );
    let errors = ModelDocument::compile_selected(
        "foreign.eqi",
        &source,
        "Moving",
        &[("line", interval), ("other", interval)],
    )
    .unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e.message().contains("support") || e.message().contains("interval")),
        "{errors:?}"
    );
}

#[test]
fn explicit_fixed_limits_specialize_the_same_integral_and_partial() {
    use eqiora::artifact::ModelEnvelope;
    use eqiora::meshing::QuadratureRule;
    use eqiora_backend_faer::FaerLinearSolver;
    use std::collections::HashMap;
    for limits in ["", ",lower=-4[m],upper=4[m]"] {
        let source = SOURCE.replace(",lower=0[m],upper=a", limits);
        let document = document(&source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let plan = super::resolve(&model).unwrap();
        let result = plan
            .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
            .unwrap();
        let rules = HashMap::from([(
            document.aliases()["line"].downcast().unwrap(),
            QuadratureRule::gauss_legendre(2).unwrap(),
        )]);
        for (name, expected) in [("total", 128.0 / 3.0), ("slope", 0.0)] {
            let value = result
                .observe(&model, document.aliases()[name].downcast().unwrap(), &rules)
                .unwrap();
            assert!((value.value().real_scalar_value().unwrap().value() - expected).abs() < 1e-12);
        }
    }
}
