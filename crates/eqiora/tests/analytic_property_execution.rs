//! Constitutive inputs remain typed and nominal through ordinary source execution.
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::kernel::KernelNode;
use eqiora::sem::{Interpreter, ReferenceConfig};

const SOURCE: &str = r#"
property contract Conductivity(input T: K, input p: kg / m / s ^ 2): kg * m / s ^ 3 / K {
  derivatives first_partials;
  branch liquid;
}
property release Fluid: Conductivity {
  analytic { value = 10[kg * m / s ^ 3 / K]
    + 0.1[kg * m / s ^ 3 / K ^ 2] * (T - 300[K])
    + 0.001[kg * m / s ^ 3 / K ^ 3] * (T - 300[K]) ^ 2
    + 0.00001[m ^ 2 / s / K] * p;
  source_unit: kg * m / s ^ 3 / K = 1; }
  validity T >= 300[K] and T <= 400[K] and p >= 0[kg / m / s ^ 2] and p <= 300000[kg / m / s ^ 2];
  outside reject;
  branch liquid;
  citation org.example.analytic;
  license spdx.CC0_1_0;
}
component Thermal(parameter temperature: K, parameter pressure: kg / m / s ^ 2, property conductivity: Conductivity) {
  variable k: kg * m / s ^ 3 / K;
  relation law { k = conductivity(T = temperature, p = pressure); }
}
model Main() {
  instance warm: Thermal(temperature = 320[K], pressure = 100000[kg / m / s ^ 2], conductivity = Fluid);
  instance hot: Thermal(temperature = 340[K], pressure = 200000[kg / m / s ^ 2], conductivity = Fluid);
}
"#;

#[test]
fn two_analytic_component_consumers_keep_operating_inputs_and_release_on_replay() {
    let document = ModelDocument::compile("conductivity.eqi", SOURCE).unwrap();
    let fields = [
        document.field_ref("warm.k").unwrap().id().erase(),
        document.field_ref("hot.k").unwrap().id().erase(),
    ];
    let program = document.program();
    let bytes = ModelEnvelope::from_program(program)
        .unwrap()
        .canonical_json()
        .unwrap();
    let reopened = ModelEnvelope::from_json(&bytes, Default::default())
        .unwrap()
        .to_program()
        .unwrap();
    assert_eq!(program, &reopened);
    for program in [program, &reopened] {
        let bindings = program
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Relation(value) => Some(value.expression().properties()),
                _ => None,
            })
            .flat_map(|properties| properties.values())
            .collect::<Vec<_>>();
        for relation in program.nodes().filter_map(|node| match node {
            KernelNode::Relation(value) => Some(value),
            _ => None,
        }) {
            assert_eq!(
                program
                    .numerical_residuals(relation.id().erase())
                    .unwrap()
                    .properties(),
                relation.expression().properties()
            );
        }
        assert_eq!(bindings.len(), 2);
        assert!(bindings.iter().all(|release| release.inputs() == ["T", "p"]
            && release.guarded()
            && release.branch() == Some("liquid")));
        let result = Interpreter::new()
            .run(
                program,
                ReferenceConfig::new(0.0, 1.0)
                    .unwrap()
                    .with_nonlinear_tolerances(1e-13, 0.0)
                    .unwrap(),
            )
            .unwrap();
        for (field, expected) in fields.into_iter().zip([13.4, 17.6]) {
            assert!((result.last_value(field).unwrap().value() - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn independent_partials_keep_other_inputs_fixed_and_reject_outside_validity() {
    use eqiora::kernel::{ExprDagBuilder, ExprNode};
    use eqiora::{ScalarDomain, ValueLiteral, ValueType};
    let document = ModelDocument::compile("conductivity.eqi", SOURCE).unwrap();
    let release = document
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(value) => value.expression().properties().values().next(),
            _ => None,
        })
        .unwrap();
    // Direct differentiation of the declared polynomial gives dk/dT=.1+.002(T-300)
    // and dk/dp=.00001, with the other independent variable held fixed.
    for (name, expected) in [("T", [0.14, 0.18]), ("p", [0.00001, 0.00001])] {
        let derivative = release.partial(name).unwrap();
        for (point, expected) in [[320., 100000.], [340., 200000.]].into_iter().zip(expected) {
            let mut builder = ExprDagBuilder::new();
            let arguments = point
                .into_iter()
                .zip(derivative.formals())
                .map(|(value, formal)| {
                    let kind =
                        ValueType::scalar(ScalarDomain::Real, formal.dimension().unwrap()).unwrap();
                    builder
                        .push(ExprNode::Constant(
                            ValueLiteral::from_real(kind, value).unwrap(),
                        ))
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let root = builder.pure_operator(&derivative, arguments).unwrap();
            let dag = builder.finish([root]).unwrap();
            let actual = eqiora::ir::ScalarOperatorIr::lower(&dag)
                .unwrap()
                .evaluate_typed(&[root], &mut |_| None)
                .unwrap();
            assert!((actual[0].real_scalar_value().unwrap().value() - expected).abs() < 1e-12);
        }
    }
    assert!(release.partial("absent").is_err());
    for invalid in [
        SOURCE.replace("320[K]", "299[K]"),
        SOURCE.replace("100000[kg / m / s ^ 2]", "400000[kg / m / s ^ 2]"),
    ] {
        let document = ModelDocument::compile("outside.eqi", &invalid).unwrap();
        let error = Interpreter::new()
            .run(document.program(), ReferenceConfig::new(0., 1.).unwrap())
            .unwrap_err();
        assert!(
            error.iter().any(|diagnostic| diagnostic
                .message()
                .contains("required expression domain condition")),
            "{error:?}"
        );
    }
}

fn packaged_model(source: &str) -> eqiora::package::PackagedModelDocument {
    use eqiora::package::*;
    fn package(name: &str, source: &str, dependencies: &[PackageReleaseV1]) -> PackageReleaseV1 {
        let path = NormalizedRelativePath::parse("src/main.eqi").unwrap();
        let manifest = PackageManifestV1::new(
            "main",
            QualifiedName::parse(name).unwrap(),
            ExactVersion::parse("1.0.0").unwrap(),
            dependencies
                .iter()
                .map(|release| PackageDependencyV1::new(release.package_identity().unwrap()))
                .collect(),
            vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
        )
        .unwrap();
        let sources = PackageSourcesV1::new(
            manifest,
            vec![SourceFileV1::new(
                path,
                BundleRoleV1::ModelSource,
                source.as_bytes().to_vec(),
            )],
        )
        .unwrap();
        prepare_package_release_v1(sources, dependencies).unwrap()
    }
    let (declarations, consumer) = source.split_once("component Thermal").unwrap();
    let properties = package(
        "org.example.Fluid",
        &declarations
            .replace("property contract", "public property contract")
            .replace("property release", "public property release"),
        &[],
    );
    let consumer = format!(
        "import org.example.Fluid.main as fluid;\npublic component Thermal{}",
        consumer
            .replace(
                "conductivity: Conductivity",
                "conductivity: fluid.Conductivity"
            )
            .replace("conductivity = Fluid", "conductivity = fluid.Fluid")
            .replace("model Main", "public model Main")
    );
    let root = package(
        "org.example.Consumer",
        &consumer,
        std::slice::from_ref(&properties),
    );
    let resolution =
        ResolutionRecordV1::from_exact_releases(&root, std::slice::from_ref(&properties)).unwrap();
    let mut store = InMemoryPackageStore::default();
    store.insert(&properties).unwrap();
    store.insert(&root).unwrap();
    PackagedModelDocument::compile_locked(&store, &resolution, "Main").unwrap()
}

#[test]
fn exact_offline_package_keeps_public_analytic_contract_and_release() {
    let packaged = packaged_model(SOURCE);
    let result = Interpreter::new()
        .run(
            packaged.model().program(),
            ReferenceConfig::new(0., 1.).unwrap(),
        )
        .unwrap();
    let field = packaged.model().field_ref("warm.k").unwrap().id().erase();
    assert!((result.last_value(field).unwrap().value() - 13.4).abs() < 1e-10);
}

#[test]
fn analytic_admission_rejects_incomplete_typed_inputs_and_foreign_branch() {
    for source in [
        SOURCE.replace(
            "conductivity(T = temperature, p = pressure)",
            "conductivity(T = temperature)",
        ),
        SOURCE.replace(
            "conductivity(T = temperature, p = pressure)",
            "conductivity(T = temperature, p = temperature)",
        ),
        SOURCE.replacen(
            "  branch liquid;\n  citation",
            "  branch gas;\n  citation",
            1,
        ),
        SOURCE.replace("value = 10", "value = unknown_capture + 10"),
    ] {
        assert!(ModelDocument::compile("invalid-property.eqi", &source).is_err());
    }
}

#[test]
fn release_provenance_changes_exact_artifacts_without_changing_mathematical_structure() {
    let first = ModelDocument::compile("conductivity.eqi", SOURCE).unwrap();
    let changed = SOURCE
        .replace("Fluid", "OtherFluid")
        .replace("org.example.analytic", "org.example.other")
        .replace("spdx.CC0_1_0", "spdx.MIT");
    let second = ModelDocument::compile("other.eqi", &changed).unwrap();
    assert_ne!(
        first.canonical_json().unwrap(),
        second.canonical_json().unwrap()
    );
    assert!(first.structurally_equivalent(&second).unwrap());
    for changed_math in [
        SOURCE.replace("first_partials", "value_only"),
        SOURCE.replace("T <= 400[K]", "T <= 390[K]"),
        SOURCE.replace("+ 0.1[", "+ 0.2["),
    ] {
        let changed = ModelDocument::compile("changed-math.eqi", &changed_math).unwrap();
        assert!(!first.structurally_equivalent(&changed).unwrap());
    }
}

fn evolving_source() -> String {
    SOURCE
        .replace("parameter temperature: K", "parameter initial_temperature: K")
        .replace(
            "  variable k:",
            "  state temperature: K;\n  initial { temperature = initial_temperature; }\n  relation heating { derivative(temperature) = 5[K / s]; }\n  variable k:",
        )
        .replace("temperature = 320[K]", "initial_temperature = 320[K]")
        .replace("temperature = 340[K]", "initial_temperature = 340[K]")
}

#[test]
fn evolving_state_inputs_match_the_direct_law_without_losing_release_lineage() {
    let source = evolving_source();
    let direct = source
        .replace(
            "conductivity(T = temperature, p = pressure)",
            "10[kg * m / s ^ 3 / K] + 0.1[kg * m / s ^ 3 / K ^ 2] * (temperature - 300[K]) + 0.001[kg * m / s ^ 3 / K ^ 3] * (temperature - 300[K]) ^ 2 + 0.00001[m ^ 2 / s / K] * pressure",
        )
        .replace(", property conductivity: Conductivity", "")
        .replace(", conductivity = Fluid", "");
    let config = ReferenceConfig::new(2.0, 0.25)
        .unwrap()
        .with_initial_guess(320.0)
        .unwrap()
        .with_nonlinear_tolerances(1e-11, 0.0)
        .unwrap();
    let mut traces = Vec::new();
    for (source, expected_bindings) in [(&source, 2), (&direct, 0)] {
        let model = ModelDocument::compile("evolving-conductivity.eqi", source).unwrap();
        let packaged = packaged_model(source);
        for model in [&model, packaged.model()] {
            let envelope = ModelEnvelope::from_program(model.program()).unwrap();
            let reopened = envelope.to_program().unwrap();
            for program in [model.program(), &reopened] {
                let bindings = program
                    .nodes()
                    .filter_map(|node| match node {
                        KernelNode::Relation(value) => Some(value.expression().properties().len()),
                        _ => None,
                    })
                    .sum::<usize>();
                assert_eq!(bindings, expected_bindings);
                let trajectory = Interpreter::new().run(program, config).unwrap();
                let mut trace = Vec::new();
                for (name, initial, pressure) in
                    [("warm", 320.0, 100000.0), ("hot", 340.0, 200000.0)]
                {
                    let field = model.field_ref(&format!("{name}.k")).unwrap().id().erase();
                    let samples = trajectory
                        .samples()
                        .iter()
                        .filter(|sample| sample.coordinate() == field);
                    let mut count = 0;
                    for sample in samples {
                        // Constant heating integrates exactly under backward Euler:
                        // T(t)=T(0)+5t. Substitute into the declared two-input polynomial.
                        let delta = initial + 5.0 * sample.time() - 300.0;
                        let expected =
                            10.0 + 0.1 * delta + 0.001 * delta * delta + 0.00001 * pressure;
                        assert!((sample.value().value() - expected).abs() < 1e-9);
                        trace.push((sample.time(), sample.value().value()));
                        count += 1;
                    }
                    assert_eq!(count, 9);
                }
                traces.push(trace);
            }
        }
    }
    for trace in &traces[1..] {
        for ((time, value), (reference_time, reference)) in trace.iter().zip(&traces[0]) {
            assert_eq!(time, reference_time);
            assert!((value - reference).abs() < 1e-9);
        }
    }
}

#[test]
fn state_input_near_validity_boundary_uses_an_admissible_newton_probe() {
    let source = evolving_source()
        .replace(
            "initial_temperature = 320[K]",
            "initial_temperature = 399.999999[K]",
        )
        .replace(
            "initial_temperature = 340[K]",
            "initial_temperature = 399.999999[K]",
        );
    let model = ModelDocument::compile("boundary-conductivity.eqi", &source).unwrap();
    let config = ReferenceConfig::new(0.0, 0.25)
        .unwrap()
        .with_initial_guess(399.999999)
        .unwrap()
        .with_nonlinear_tolerances(1e-11, 0.0)
        .unwrap();
    let trajectory = Interpreter::new().run(model.program(), config).unwrap();
    // This interior point has a valid analytic derivative, but the default positive
    // finite-difference probe (about 6e-6 K) crosses the 400 K guard.
    let delta = 399.999999 - 300.0;
    let thermal = 10.0 + 0.1 * delta + 0.001 * delta * delta;
    for (name, expected) in [("warm.k", thermal + 1.0), ("hot.k", thermal + 2.0)] {
        let field = model.field_ref(name).unwrap().id().erase();
        assert!((trajectory.last_value(field).unwrap().value() - expected).abs() < 1e-9);
    }
    // A value at the exact endpoint does not supply the analytic derivative
    // demanded by initial regularity. Numerical probing must not bypass it.
    let boundary = source.replace("399.999999[K]", "400[K]");
    let boundary = ModelDocument::compile("endpoint-conductivity.eqi", &boundary).unwrap();
    let error = Interpreter::new()
        .run(
            boundary.program(),
            config.with_initial_guess(400.0).unwrap(),
        )
        .unwrap_err();
    assert!(
        error.iter().any(|diagnostic| diagnostic
            .message()
            .contains("derivative is undefined at a demanded comparison boundary")),
        "{error:?}"
    );
    let invalid = source.replace(
        "initial_temperature = 399.999999[K]",
        "initial_temperature = 401[K]",
    );
    let model = ModelDocument::compile("outside-conductivity.eqi", &invalid).unwrap();
    let error = Interpreter::new()
        .run(model.program(), config.with_initial_guess(401.0).unwrap())
        .unwrap_err();
    assert!(
        error.iter().any(|diagnostic| diagnostic
            .message()
            .contains("required expression domain condition")),
        "{error:?}"
    );
}
