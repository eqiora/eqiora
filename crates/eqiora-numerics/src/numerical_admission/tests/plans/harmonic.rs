use super::*;

const SOURCE: &str =
    include_str!("../../../../../../verify/numerics/harmonic-response/models/wave.eqi");

#[test]
fn harmonic_response_has_independent_circuit_wave_and_transient_evidence() {
    crate::form_compiler::harmonic::tests::check_finite_profiles();
    let graph = GeometryGraph::new();
    let interval = graph.interval([0., 1.]).unwrap();
    let geometry = graph
        .build(
            &interval,
            &BTreeMap::from([
                ("body".into(), vec![interval.region().into()]),
                ("left".into(), vec![interval.boundaries()[0].into()]),
                ("right".into(), vec![interval.boundaries()[1].into()]),
            ]),
        )
        .unwrap();
    let bindings = ["body", "left", "right"].map(|name| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: (name != "body").then(|| geometry.entity_set("body").unwrap()),
            },
        )
    });
    let compiled = CompiledModel::compile_selected("wave.eqi", SOURCE, "Wave", &bindings).unwrap();
    let form = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection()
        .clone();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry]).unwrap();
    let original = ModelEnvelope::from_program(&program).unwrap();
    let plan = ResolvedCommonPlan::resolve(
        &original,
        cartesian_box_resources(&geometry, &[2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-15,
            NonZeroUsize::new(64).unwrap(),
        )),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        Some(&form),
    )
    .unwrap();
    let description = plan.formulation().unwrap();
    assert_eq!(description.requested(), FormulationSelectionMode::Authored);
    assert_eq!(description.effective(), FormulationKind::HarmonicResponse);
    assert_eq!(
        description.requested_source_identity(),
        Some(form.source_identity())
    );
    let scalar = plan.as_linear().unwrap();
    assert_eq!(scalar.harmonic_original_model(), Some(&original));
    assert_eq!(scalar.harmonic_amplitudes().count(), 1);
    assert_ne!(
        plan.model_id(),
        original.model().unwrap().ulid().to_string()
    );
    let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
    let result = plan
        .as_linear()
        .unwrap()
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let (_, values, shape) = result.field_block(0, 0).unwrap();
    assert_eq!(shape, &[3]);
    assert_eq!(values.len(), 6);
    // U=2+i has zero spatial gradient, and (-omega^2-i*omega)*U=-1-3i.
    // It meets the prescribed left amplitude and zero right normal derivative exactly.
    for node in values.as_chunks::<2>().0 {
        assert!(
            (node[0] - 2.).abs() < 1e-10 && (node[1] - 1.).abs() < 1e-10,
            "{node:?}"
        );
    }
    assert_eq!(
        crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
        result
    );
    let scalar = plan.as_linear().unwrap();
    let (_, original_field, amplitude) = scalar.harmonic_amplitudes().next().unwrap();
    assert_eq!(scalar.harmonic_angular_frequency(), Some(1.));
    let seconds = eqiora_core::DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    // Re((2+i) exp(-it)) = 2 cos(t) + sin(t): peak, not RMS amplitudes.
    for (time, expected) in [
        (0., 2.),
        (std::f64::consts::FRAC_PI_2, 1.),
        (std::f64::consts::PI, -2.),
        (-std::f64::consts::FRAC_PI_2, -1.),
    ] {
        let reconstructed = scalar
            .reconstruct_harmonic_field_block(
                &result,
                original_field,
                0,
                DynQuantity::new(time, seconds),
            )
            .unwrap();
        assert_eq!(reconstructed.len(), 3);
        assert!(
            reconstructed
                .iter()
                .all(|value| (value - expected).abs() < 1e-10)
        );
    }
    for (field, block, time, diagnostic) in [
        (
            amplitude,
            0,
            DynQuantity::new(0., seconds),
            "Field has no mapping",
        ),
        (
            original_field,
            1,
            DynQuantity::new(0., seconds),
            "block is absent",
        ),
        (
            original_field,
            0,
            DynQuantity::new(
                0.,
                eqiora_core::DimExponents::from_integers([1, 0, 0, 0, 0, 0, 0]).unwrap(),
            ),
            "time in seconds",
        ),
        (
            original_field,
            0,
            DynQuantity::new(f64::NAN, seconds),
            "time in seconds",
        ),
        (
            original_field,
            0,
            DynQuantity::new(f64::INFINITY, seconds),
            "time in seconds",
        ),
    ] {
        let error = scalar
            .reconstruct_harmonic_field_block(&result, field, block, time)
            .unwrap_err();
        assert!(error.to_string().contains(diagnostic), "{error}");
    }
    for (old, new, diagnostic) in [
        (
            "balance, fixed, flux",
            "balance, fixed",
            "every original noninitial Relation",
        ),
        (
            "balance, fixed, flux",
            "balance, flux",
            "every original noninitial Relation",
        ),
        (
            "excitation force=math.complex(-1[1/s^2],-3[1/s^2]);",
            "",
            "without an excitation",
        ),
        (
            "excitation boundary_value=math.complex(2,1);",
            "",
            "without an excitation",
        ),
    ] {
        assert!(SOURCE.contains(old));
        let errors = CompiledModel::compile_selected(
            "wave.eqi",
            &SOURCE.replace(old, new),
            "Wave",
            &bindings,
        )
        .unwrap_err();
        assert!(format!("{errors:?}").contains(diagnostic), "{errors:?}");
    }
}
