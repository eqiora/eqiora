use super::*;

mod transport;

#[test]
fn storage_chart_selection_cannot_hide_an_extra_equation_region() {
    use crate::numerical_admission::native::polyhedral::bind_model_support;
    let (geometry, _, program) = fixture(SOURCE);
    let owner = resources(&geometry, false);
    assert_eq!(
        bind_model_support(&program, &owner.resources)
            .unwrap()
            .len(),
        1
    );
    let extra_equation = SOURCE.replace("    state u:", "    variable extra: 1 on physical;\n    relation extra_balance on physical { extra = 0; }\n    state u:");
    let mut missing_storage_map = SOURCE.to_owned();
    let start = missing_storage_map.find("        storage").unwrap();
    let end = missing_storage_map.find("        flux").unwrap();
    missing_storage_map.replace_range(start..end, "        storage 3[s/m^2] * u;\n");
    for source in [extra_equation, missing_storage_map] {
        let (geometry, _, program) = fixture(&source);
        let owner = resources(&geometry, false);
        let error = bind_model_support(&program, &owner.resources).unwrap_err();
        assert!(error.message().contains("foreign Geometry"), "{error:?}");
    }
}

const SOURCE: &str = r#"
public component Inventory(
    support body: volume(ambient_dimension = 2),
    support physical: volume(ambient_dimension = 2),
    support outer: boundary(parent = body)
) {
    coordinate xi: m on body from body[0];
    coordinate eta: m on body from body[1];
    coordinate x: m on physical from physical[0];
    coordinate y: m on physical from physical[1];
    state u: 1 on body in h1;
    initial { u = 2; }
    law balance on body {
        storage 3[s/m^2] * u * volume_jacobian(from=(xi,eta),at=(
            x=(1+0.5[1/s]*time())*xi,
            y=(1+0.5[1/s]*time())*eta));
        flux -grad(u);
        source 0[1/m^2];
    }
    relation prescribed on outer { trace(u) = 2 / (1+0.5[1/s]*time())^2; }
}
"#;

#[test]
fn mapped_form_binds_boundary_time_without_changing_initial_time() {
    use crate::form_compiler::linear::CompiledLinearBlockForm;
    let source = SOURCE.replace("initial { u = 2; }", "initial { u = 2 + 1[1/s]*time(); }");
    let (geometry, _, program) = fixture(&source);
    let supports = crate::numerical_admission::native::polyhedral::bind_model_support(
        &program,
        &resources(&geometry, false).resources,
    )
    .unwrap();
    // The sole PDE Region remains distinct from the map's coordinate-only target.
    let domain = *supports.keys().next().unwrap();
    assert!(
        CompiledLinearBlockForm::<f64>::derive(
            &program,
            domain,
            2,
            &std::collections::BTreeSet::new(),
        )
        .is_err()
    );
    for (time, expected) in [(0., 2.), (1., 8. / 9.), (2., 0.5)] {
        let form = CompiledLinearBlockForm::<f64>::derive_at_time(
            &program,
            domain,
            2,
            &std::collections::BTreeSet::new(),
            Some(time),
        )
        .unwrap();
        assert!(form.is_transient());
        assert_eq!(
            form.initial_values_at(&[0., 0.])
                .unwrap()
                .into_values()
                .flatten()
                .collect::<Vec<_>>(),
            vec![2.]
        );
        for law in form.boundary_laws().values().flat_map(|laws| laws.values()) {
            assert!((law.evaluate(&[0., 0.], &[]).unwrap()[0] - expected).abs() < 1e-14);
        }
    }
}

fn fixture(source: &str) -> (CanonicalGeometryV1, CanonicalGeometryV1, KernelProgram) {
    fixture_result(source).unwrap()
}

fn fixture_result(
    source: &str,
) -> Result<(CanonicalGeometryV1, CanonicalGeometryV1, KernelProgram), Vec<eqiora_core::Diagnostic>>
{
    let geometry = geometry(false);
    let body = geometry.entity_set("body").unwrap();
    let physical = CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            vec![[-10., -10.], [10., -10.], [10., 10.], [-10., 10.]],
            vec![PlanarFace::new(vec![0, 1, 2, 3], vec![])],
            vec![NamedEntitySet::new("chart", 2, vec![0])],
            1e-12,
        )
        .unwrap(),
    )
    .unwrap();
    let compiled = CompiledModel::compile_selected(
        "mapped-inventory.eqi",
        source,
        "Inventory",
        &[
            (
                "body",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: body,
                    parent: None,
                },
            ),
            (
                "outer",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set("outer").unwrap(),
                    parent: Some(body),
                },
            ),
            (
                "physical",
                StaticBindingValue::GeometrySupport {
                    geometry: &physical,
                    selection: physical.entity_set("chart").unwrap(),
                    parent: None,
                },
            ),
        ],
    )?;
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot_with_geometry(
        &store.snapshot(),
        model,
        &[&geometry, &physical],
    )?;
    Ok((geometry, physical, program))
}

fn mapped_plan(source: &str) -> ResolvedCommonPlan {
    mapped_plan_at_step(source, 1.0)
}

fn mapped_plan_at_step(source: &str, step: f64) -> ResolvedCommonPlan {
    let (geometry, physical, program) = fixture(source);
    let model = ModelEnvelope::from_program(&program).unwrap();
    ResolvedCommonPlan::resolve(
        &model,
        resources(&geometry, false)
            .with_model_geometries(vec![physical.clone()])
            .unwrap(),
        CommonSpatialPolicy::P1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )),
        None,
        Some(CommonBackwardEuler::from_seconds(step).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap()
}

#[test]
fn translating_material_profile_uses_current_boundary_coordinates() {
    let source = SOURCE
        .replace("initial { u = 2; }", "initial { u = 2 + xi/1[m]; }")
        .replace("(1+0.5[1/s]*time())*xi", "xi+0.25[m/s]*time()")
        .replace("(1+0.5[1/s]*time())*eta", "eta")
        .replace("2 / (1+0.5[1/s]*time())^2", "2 + coordinate(0)/1[m]");
    let plan = mapped_plan_at_step(&source, 0.1);
    let linear = plan.as_linear().unwrap();
    let NativeMeshResources::GmshSimplicial { mesh, .. } = linear.admission.resources() else {
        unreachable!()
    };
    let expected = mesh
        .mesh()
        .vertices()
        .iter()
        .map(|point| 2.0 + point[0])
        .collect::<Vec<_>>();
    let initial = linear.initial_state().unwrap();
    assert_eq!(initial.linear_values().unwrap(), expected);
    let run =
        CommonTransientRunRequest::from_steps(plan.clone(), initial, 10, vec![3, 6, 10]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("translated profile must complete")
    };
    let RecognizedNativeModel::Linear(equations) = linear.admission.recognized_model() else {
        unreachable!()
    };
    let motion = equations.single().unwrap().form.motion().unwrap();
    for (_, state) in outputs {
        let geometry = motion
            .bind(linear.admission.program(), state.time_s())
            .unwrap()
            .geometry_state(mesh.mesh())
            .unwrap();
        // q(x,t)=2+x-t/4, so partial_t q + (1/4) partial_x q=0.
        // A translated unit box carries the same nonuniform material profile.
        for ((value, reference), physical) in state
            .linear_values()
            .unwrap()
            .iter()
            .zip(&expected)
            .zip(geometry.coordinates())
        {
            assert!((value - reference).abs() < 1e-11);
            assert!((value - (2.0 + physical[0] - 0.25 * state.time_s())).abs() < 1e-11);
        }
    }
}

#[test]
fn moving_storage_rejects_unrepresented_capacity_history() {
    for coefficient in ["(1+0.25[1/s]*time())", "(1+xi/1[m])"] {
        let source = SOURCE.replace("3[s/m^2] * u", &format!("3[s/m^2] * {coefficient} * u"));
        let (geometry, _, program) = match fixture_result(&source) {
            Ok(fixture) => fixture,
            Err(errors) => {
                // The semantic storage correspondence may reject this profile
                // before numerical admission. Never bypass that proof to test
                // the later history guard.
                assert!(
                    errors.iter().any(|error| error
                        .message()
                        .contains("storage accumulation correspondence")),
                    "{errors:?}"
                );
                continue;
            }
        };
        let error = crate::numerical_admission::native::polyhedral::bind_model_support(
            &program,
            &resources(&geometry, false).resources,
        )
        .unwrap_err();
        assert!(error.message().contains("normalized capacity"), "{error:?}");
    }
}

fn stationary_density_source() -> String {
    SOURCE
        .replace(
            "    state u:",
            "    parameter vx: m/s = 0[m/s];\n    parameter vy: m/s = 0[m/s];\n    state u:",
        )
        .replace(
            "flux -grad(u);",
            r#"flux -grad(u) + 3[s/m^2]*u*(1+0.5[1/s]*time()) * (
            (vx-derivative((1+0.5[1/s]*time())*xi))*grad(xi)
          + (vy-derivative((1+0.5[1/s]*time())*eta))*grad(eta));"#,
        )
        .replace("2 / (1+0.5[1/s]*time())^2", "2")
}

#[test]
fn stationary_physical_density_closes_expanding_volume_balance() {
    let source = stationary_density_source();
    let plan = mapped_plan(&source);
    let initial = plan.as_linear().unwrap().initial_state().unwrap();
    let run = CommonTransientRunRequest::from_steps(plan, initial, 2, vec![1, 2]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("expanding stationary density must complete");
    };
    let mut previous_inventory = 2.0;
    for ((_, state), (scale, expected, influx)) in
        outputs.iter().zip([(1.5, 4.5, 2.5), (2.0, 8.0, 3.5)])
    {
        for density in state.linear_values().unwrap() {
            assert!(
                (density - 2.0).abs() < 1e-11,
                "stationary physical density changed: {density}"
            );
        }
        // rho=2 and v=0 on the fixed physical chart. Only the observation
        // volume moves: inventory is 2*lambda^2, not the material invariant 2.
        // First-step influx is 2*(1.5^2-1)=2.5; endpoint mesh flux gives 3.
        // Each reference triangle has area 1/4. Its P1 integral is area
        // times the mean of its three nodal coefficients: every corner
        // occurs twice and the center four times in this fixture.
        let values = state.linear_values().unwrap();
        let inventory = scale * scale * (values[..4].iter().sum::<f64>() / 6.0 + values[4] / 3.0);
        assert!((inventory - expected).abs() < 1e-11);
        assert!((inventory - previous_inventory - influx).abs() < 1e-11);
        previous_inventory = inventory;
    }
}

#[test]
fn mapped_material_inventory_executes_through_plan_and_restart() {
    let plan = mapped_plan(SOURCE);
    let time = eqiora_time::TimeBackendCapabilities::new(
        eqiora_time::TimeBackendIdentity::new("eqiora.test.time", "1"),
        &[
            eqiora_core::ScalarDomain::Real,
            eqiora_core::ScalarDomain::Complex,
        ],
        &[eqiora_core::ScalarType::F64],
    );
    let bytes = plan.to_bytes().unwrap();
    let replay = ResolvedCommonPlan::from_bytes(&bytes, &REFERENCE_LINEAR_SOLVER, time).unwrap();
    assert_eq!(replay.to_bytes().unwrap(), bytes);
    let initial = replay.as_linear().unwrap().initial_state().unwrap();
    assert_eq!(initial.linear_values().unwrap(), &[2.; 5]);
    let run =
        CommonTransientRunRequest::from_steps(replay.clone(), initial, 2, vec![1, 2]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("mapped inventory must complete");
    };
    // Reference unit box expands with lambda=1+t/2. Zero relative advective
    // flux conserves total inventory: rho=2/lambda^2, independently of the solve.
    for ((_, state), (scale, expected)) in outputs.iter().zip([(1.5, 8. / 9.), (2., 0.5)]) {
        for value in state.linear_values().unwrap() {
            assert!((value - expected).abs() < 1e-11);
        }
        assert!((expected * scale * scale - 2.).abs() < 1e-12);
        assert_eq!(
            CommonState::from_bytes(&state.to_bytes().unwrap(), &replay).unwrap(),
            *state
        );
    }
    let restart = CommonState::from_bytes(&outputs[0].1.to_bytes().unwrap(), &replay).unwrap();
    let run = CommonTransientRunRequest::from_steps(replay, restart, 1, vec![1]).unwrap();
    let std::ops::ControlFlow::Continue(restarted) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("mapped restart must complete");
    };
    assert_eq!(restarted[0].1, outputs[1].1);
}

#[test]
fn moving_transport_rejects_missing_relative_rate_and_wrong_cofactor() {
    use crate::form_compiler::linear::CompiledLinearBlockForm;
    let source = stationary_density_source();
    for mutant in [
        source.replace("vx-derivative((1+0.5[1/s]*time())*xi)", "vx"),
        source.replace(
            "vx-derivative((1+0.5[1/s]*time())*xi)",
            "vx-derivative((1+0.25[1/s]*time())*xi)",
        ),
        source.replace(
            "vx-derivative((1+0.5[1/s]*time())*xi)",
            "vx-derivative((1+0.5[1/s]*time())*eta)",
        ),
        source.replace("3[s/m^2]*u*(1+0.5[1/s]*time())", "3[s/m^2]*u"),
        source.replace(
            "3[s/m^2]*u*(1+0.5[1/s]*time())",
            "6[s/m^2]*u*(1+0.5[1/s]*time())",
        ),
    ] {
        let (geometry, _, program) = fixture(&mutant);
        let supports = crate::numerical_admission::native::polyhedral::bind_model_support(
            &program,
            &resources(&geometry, false).resources,
        )
        .unwrap();
        let domain = *supports.keys().next().unwrap();
        let error = CompiledLinearBlockForm::<f64>::derive_at_time(
            &program,
            domain,
            2,
            &std::collections::BTreeSet::new(),
            Some(0.0),
        )
        .expect_err("incorrect moving transport must reject");
        assert!(
            error
                .message()
                .contains("explicit material-minus-mesh velocity correspondence"),
            "{error:?}"
        );
    }
}
