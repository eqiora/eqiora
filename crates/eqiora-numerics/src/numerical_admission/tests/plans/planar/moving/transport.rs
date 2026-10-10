use super::*;

#[test]
fn a_capacity_subtraction_cannot_impersonate_relative_velocity() {
    use crate::form_compiler::linear::CompiledLinearBlockForm;
    let source = stationary_density_source()
        .replace("(1+0.5[1/s]*time())", "1")
        .replace("3[s/m^2]*u*1", "(3[s/m^2]-0[s/m^2])*u*1")
        .replace("vx-derivative(1*xi)", "3[m/s]")
        .replace("vy-derivative(1*eta)", "3[m/s]");
    let (geometry, _, program) = fixture(&source);
    let supports = crate::numerical_admission::native::polyhedral::bind_model_support(
        &program,
        &resources(&geometry, false).resources,
    )
    .unwrap();
    let error = CompiledLinearBlockForm::<f64>::derive_at_time(
        &program,
        *supports.keys().next().unwrap(),
        2,
        &std::collections::BTreeSet::new(),
        Some(0.0),
    )
    .expect_err("a capacity difference has the wrong dimensions for velocity");
    assert!(
        error
            .message()
            .contains("explicit material-minus-mesh velocity correspondence"),
        "{error:?}"
    );
}

fn run(plan: ResolvedCommonPlan, steps: usize) -> Vec<CommonState> {
    let initial = plan
        .as_linear()
        .unwrap()
        .initial_state(0.0, Vec::new())
        .unwrap();
    let request =
        CommonTransientRunRequest::from_steps(plan, initial, steps, (1..=steps).collect()).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = request
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("all prescribed transport steps must complete");
    };
    outputs.into_iter().map(|(_, state)| state).collect()
}

#[test]
fn material_velocity_is_distinct_from_mesh_velocity_and_declaration_order() {
    let source = stationary_density_source()
        .replace(
            "parameter vx: m/s = 0[m/s];",
            "parameter vx: m/s = 0.75[m/s];",
        )
        .replace(
            "parameter vy: m/s = 0[m/s];",
            "parameter vy: m/s = 0.5[m/s];",
        )
        .replace("(1+0.5[1/s]*time())*xi", "xi+0.25[m/s]*time()")
        .replace("(1+0.5[1/s]*time())*eta", "eta+0.125[m/s]*time()")
        .replace("3[s/m^2]*u*(1+0.5[1/s]*time())", "3[s/m^2]*u")
        .replace(
            "initial { u = 2; }",
            "initial { u = 4 + xi/1[m] + 2*eta/1[m]; }",
        )
        .replace(
            "trace(u) = 2;",
            "trace(u) = 4 + coordinate(0)/1[m] + 2*coordinate(1)/1[m] - 1.25[1/s]*time();",
        );
    let permuted = source
        .replace("parameter vx: m/s = 0.75[m/s];\n    parameter vy: m/s = 0.5[m/s];", "parameter decoy: m/s = 9[m/s];\n    parameter vy: m/s = 0.5[m/s];\n    parameter vx: m/s = 0.75[m/s];")
        .replace("(vx-derivative(xi+0.25[m/s]*time()))*grad(xi)\n          + (vy-derivative(eta+0.125[m/s]*time()))*grad(eta)", "(vy-derivative(eta+0.125[m/s]*time()))*grad(eta)\n          + (vx-derivative(xi+0.25[m/s]*time()))*grad(xi)")
        .replace("from=(xi,eta)", "from=(eta,xi)")
        .replace("x=xi+0.25[m/s]*time(),\n            y=eta+0.125[m/s]*time()", "y=eta+0.125[m/s]*time(),\n            x=xi+0.25[m/s]*time()");
    let vertices = [[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.5, 0.5]];
    let mut reference = None;
    for source in [source, permuted] {
        let outputs = run(mapped_plan_at_step(&source, 0.1), 10);
        for state in &outputs {
            for (value, [xi, eta]) in state.linear_values().unwrap().iter().zip(vertices) {
                // q(x,y,t)=4+x+2y-(3/4+2*1/2)t has zero material
                // derivative. Pulling it onto chi=(xi+t/4,eta+t/8)
                // gives 4+xi+2eta-5t/4, exactly affine in space/time.
                // Mesh velocity has nonzero dot product 1/2 with grad(q),
                // so omitting it cannot accidentally preserve this profile.
                let expected = 4.0 + xi + 2.0 * eta - 1.25 * state.time_s();
                assert!((value - expected).abs() < 1e-10, "{value} != {expected}");
            }
        }
        let values = outputs
            .iter()
            .map(|state| state.linear_values().unwrap().to_vec())
            .collect::<Vec<_>>();
        if let Some(reference) = &reference {
            assert_eq!(&values, reference);
        } else {
            reference = Some(values);
        }
    }
}

#[test]
fn fixed_map_specialization_matches_fixed_domain_law() {
    let mapped = stationary_density_source()
        .replace("(1+0.5[1/s]*time())", "1")
        .replace(
            "parameter vx: m/s = 0[m/s];",
            "parameter vx: m/s = 0.75[m/s];",
        )
        .replace(
            "parameter vy: m/s = 0[m/s];",
            "parameter vy: m/s = 0.5[m/s];",
        )
        .replace("source 0[1/m^2];", "source 3[1/m^2];")
        .replace("3[s/m^2]*u*1", "(3[s/m^2]-0[s/m^2])*u*1");
    let fixed = mapped
        .replace("    support physical: volume(ambient_dimension = 2),\n", "")
        .replace("    coordinate x: m on physical from physical[0];\n", "")
        .replace("    coordinate y: m on physical from physical[1];\n", "")
        .replace(
            " * volume_jacobian(from=(xi,eta),at=(\n            x=1*xi,\n            y=1*eta))",
            "",
        )
        .replace("vx-derivative(1*xi)", "vx")
        .replace("vy-derivative(1*eta)", "vy");
    let geometry = geometry(false);
    let body = geometry.entity_set("body").unwrap();
    let model = compile_model(
        "fixed-chart-law.eqi",
        &fixed,
        &geometry,
        "Inventory",
        &[
            ("body", body, None),
            (
                "outer",
                geometry.entity_set("outer").unwrap(),
                Some(("body", body)),
            ),
        ],
        &[],
    );
    let fixed = ResolvedCommonPlan::resolve(
        &model,
        resources(&geometry, false),
        CommonSpatialPolicy::P1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )),
        None,
        Some(CommonBackwardEuler::from_seconds(0.25).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let mapped = run(mapped_plan_at_step(&mapped, 0.25), 3);
    let fixed = run(fixed, 3);
    for (mapped, fixed) in mapped.iter().zip(&fixed) {
        for (a, b) in mapped
            .linear_values()
            .unwrap()
            .iter()
            .zip(fixed.linear_values().unwrap())
        {
            assert!((a - b).abs() < 1e-12);
        }
        assert!(fixed.linear_values().unwrap()[4] > 2.0);
    }
}
