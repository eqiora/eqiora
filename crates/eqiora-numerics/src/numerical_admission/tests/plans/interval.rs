//! Interval proof selection must not narrow existing automatic TPFA execution.
use super::*;

#[test]
fn interval_proof_inventory_preserves_automatic_tpfa_and_reauthenticates_rules() {
    let geometry = cartesian_interval();
    let source = POISSON_INTERVAL.replace(
        "relation balance on body {\n    -div(grad(potential)) - source_scale = 0;\n  }",
        "law balance on body { flux -grad(potential); source source_scale; }",
    );
    for unsupported in [false, true] {
        let source = if unsupported {
            source.replace(
                "source source_scale;",
                "source math.sqrt(1) * source_scale;",
            )
        } else {
            source.clone()
        };
        let model = scalar_box_model(&geometry, &source, "PoissonInterval", &["left", "right"]);
        let plan = resolve_scalar_box(
            &model,
            cartesian_box_resources(&geometry, &[4]),
            CommonSpatialPolicy::CellCenteredTpfa,
        );
        assert_eq!(plan.formulation().is_none(), unsupported);
        let output = plan.run(&REFERENCE_LINEAR_SOLVER).unwrap();
        // Four unit-interval cells, unit source/diffusion and half-cell boundary resistance.
        for (actual, expected) in output.fields[0]
            .2
            .iter()
            .zip([0.0625, 0.125, 0.125, 0.0625])
        {
            assert!((actual - expected).abs() < 1.0e-9);
        }
        if !unsupported {
            let mut forged = plan.clone();
            forged.formulation.as_mut().unwrap().rule_ids = Box::new(["forged-rule"]);
            assert!(forged.run(&REFERENCE_LINEAR_SOLVER).is_err());
        }
    }
}

#[test]
fn source_cartesian_law_retains_automatic_tpfa_without_geometry_interval_claim() {
    let source = r#"model SourceInterval() {
      domain body = box(0, 1);
      domain left = boundary(body, axis = 0, side = lower);
      domain right = boundary(body, axis = 0, side = upper);
      parameter source_scale: 1 / m ^ 2 = 1;
      variable potential: 1 on body;
      law balance on body { flux -grad(potential); source source_scale; }
      relation lower on left { trace(potential) = 0; }
      relation upper on right { trace(potential) = 0; }
    }"#;
    let (transaction, model, _) = eqiora_compiler::compile("source-interval.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    let geometry = cartesian_interval();
    let plan = resolve_scalar_box(
        &model,
        cartesian_box_resources(&geometry, &[4]),
        CommonSpatialPolicy::CellCenteredTpfa,
    );
    assert!(plan.formulation().is_none());
    assert!(plan.run(&REFERENCE_LINEAR_SOLVER).is_ok());
}

const NEUMANN_INTERVAL: &str = r#"
public component NeumannInterval(
    support body: volume(ambient_dimension = 1),
    support left: boundary(parent = body), support right: boundary(parent = body),
    parameter source_value: 1 / m ^ 2,
    parameter lower_load: 1 / m,
    parameter upper_load: 1 / m
) {
    variable potential: 1 on body;
    law balance on body { flux -grad(potential); source source_value; }
    relation lower on left { normal(grad(potential)) = lower_load; }
    relation upper on right { normal(grad(potential)) = upper_load; }
    form conservative for balance {
        interval segment(a, b) on body;
        gauge potential {
            reference integrate(body, potential) = 0;
            compatibility integrate(body, source_value) + lower_load + upper_load = 0;
        }
        outward_flux(segment, a, -grad(potential)) + outward_flux(segment, b, -grad(potential)) = integrate(segment, source_value);
    }
}
"#;

fn neumann_model(
    geometry: &CanonicalGeometryV1,
    source: &str,
) -> (ModelEnvelope, AuthoredFormulationProjection) {
    let body = geometry.entity_set("body").unwrap();
    let source_value = eqiora_core::ValueLiteral::try_from(DynQuantity::new(
        -2.,
        DimExponents::from_integers([0, -2, 0, 0, 0, 0, 0]).unwrap(),
    ))
    .unwrap();
    let load = eqiora_core::ValueLiteral::try_from(DynQuantity::new(
        1.,
        DimExponents::from_integers([0, -1, 0, 0, 0, 0, 0]).unwrap(),
    ))
    .unwrap();
    let compiled = CompiledModel::compile_selected(
        "neumann.eqi",
        source,
        "NeumannInterval",
        &[
            ("source_value", StaticBindingValue::Value(&source_value)),
            ("lower_load", StaticBindingValue::Value(&load)),
            ("upper_load", StaticBindingValue::Value(&load)),
            (
                "body",
                StaticBindingValue::GeometrySupport {
                    geometry,
                    selection: body,
                    parent: None,
                },
            ),
            (
                "left",
                StaticBindingValue::GeometrySupport {
                    geometry,
                    selection: geometry.entity_set("left").unwrap(),
                    parent: Some(body),
                },
            ),
            (
                "right",
                StaticBindingValue::GeometrySupport {
                    geometry,
                    selection: geometry.entity_set("right").unwrap(),
                    parent: Some(body),
                },
            ),
        ],
    )
    .unwrap_or_else(|errors| panic!("{errors:?}"));
    let projection = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection()
        .clone();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[geometry]).unwrap();
    (ModelEnvelope::from_program(&program).unwrap(), projection)
}

#[test]
fn gauge_compatibility_matches_actual_source_and_signed_boundary_loads() {
    let geometry = cartesian_interval();
    let admit = |source: &str| {
        let (model, projection) = neumann_model(&geometry, source);
        let recognized =
            RecognizedNativeAdmission::recognize(&model, cartesian_box_resources(&geometry, &[4]))
                .unwrap();
        let RecognizedNativeModel::Scalar(equations) = &recognized.recognized else {
            panic!("scalar");
        };
        super::super::super::scalar::interval::admit_gauge(
            &recognized.program,
            equations,
            &projection,
        )
    };
    assert!(admit(NEUMANN_INTERVAL).unwrap().is_some());
    // Whole boundary-equation sign reversal leaves exactly the same conormal datum.
    assert!(
        admit(&NEUMANN_INTERVAL.replace(
            "normal(grad(potential)) = lower_load",
            "-normal(grad(potential)) = -lower_load"
        ))
        .unwrap()
        .is_some()
    );
    for mutant in [
        NEUMANN_INTERVAL.replace(
            "+ lower_load + upper_load = 0",
            "- lower_load + upper_load = 0",
        ),
        NEUMANN_INTERVAL.replace(
            "+ lower_load + upper_load = 0",
            "+ lower_load - upper_load = 0",
        ),
        NEUMANN_INTERVAL.replace(
            "integrate(body, source_value) +",
            "-integrate(body, source_value) +",
        ),
        NEUMANN_INTERVAL.replace(
            "normal(grad(potential)) = lower_load",
            "trace(potential) = 0",
        ),
        NEUMANN_INTERVAL.replace(
            "reference integrate(body, potential) = 0",
            "reference integrate(body, potential) = integrate(body, potential)",
        ),
    ] {
        assert!(admit(&mutant).is_err(), "{mutant}");
    }
}
