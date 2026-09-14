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
