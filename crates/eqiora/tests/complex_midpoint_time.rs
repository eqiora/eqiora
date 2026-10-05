//! One ordinary Model/Plan/State/Run path for complex and real oscillators.
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::runtime::FirstOrderProgram;
use eqiora::time::{ImplicitMidpointTimeBackend, TimeMethod};
use eqiora_numerics::{
    CommonOdePlan, CommonOdePolicy, CommonOdeRunRequest, CommonOdeState, CommonTimeTolerance,
    CommonTrajectory, ResolvedCommonPlan,
};

const SOURCE: &str = r#"
model Oscillators() {
    parameter omega:1/s=1;
    state z:array<complex<1>,2>;
    state u:1;
    state v:1;
    initial { z=[math.complex(1,0),math.complex(2,0)]; u=3; v=0; }
    relation flow {
        derivative(z)=math.complex(0,1)*omega*z;
        derivative(u)=-omega*v;
        derivative(v)=omega*u;
    }
}
"#;
fn run(
    plan: CommonOdePlan,
    state: CommonOdeState,
    end: f64,
    outputs: Vec<f64>,
) -> CommonTrajectory {
    let request = CommonOdeRunRequest::new(plan, state, end, outputs).unwrap();
    let solution = ImplicitMidpointTimeBackend::new()
        .solve(&request.problem().unwrap(), request.time_plan())
        .unwrap();
    CommonTrajectory::accept_ode(request, solution).unwrap()
}
fn resolve(document: &ModelDocument, step: f64) -> CommonOdePlan {
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let flow = FirstOrderProgram::lower(
        document.program(),
        document.aliases()["flow"].downcast().unwrap(),
    )
    .unwrap();
    let temporal = CommonOdePolicy::new(
        TimeMethod::ImplicitMidpoint,
        step,
        1e-12,
        flow.state_coordinates()
            .iter()
            .map(|&c| CommonTimeTolerance::new(c, 1e-14).unwrap())
            .collect(),
    )
    .unwrap();
    CommonOdePlan::resolve(
        &model,
        document.program(),
        temporal,
        ImplicitMidpointTimeBackend::CAPABILITIES,
    )
    .unwrap()
}
#[test]
fn complex_and_real_oscillators_share_exact_plan_state_and_restart() {
    let document = ModelDocument::compile("oscillators.eqi", SOURCE).unwrap();
    let plan = resolve(&document, 0.05);
    let resolved = ResolvedCommonPlan::Ode(Box::new(plan.clone()));
    let bytes = resolved.to_bytes().unwrap();
    let replay = ResolvedCommonPlan::from_bytes(
        &bytes,
        &eqiora::solver::REFERENCE_LINEAR_SOLVER,
        ImplicitMidpointTimeBackend::CAPABILITIES,
    )
    .unwrap();
    assert_eq!(replay, resolved);
    let state = plan.initial_state(0.).unwrap();
    let state = CommonOdeState::from_bytes(&state.to_bytes().unwrap(), &plan).unwrap();
    let whole = run(plan.clone(), state.clone(), 1., vec![0.025, 0.375, 0.5, 1.]);
    let sparse = run(plan.clone(), state.clone(), 1., vec![1.]);
    assert_eq!(whole.ode_history(), sparse.ode_history());
    let last = whole.ode_states().unwrap().last().unwrap();
    assert_eq!(last.values(), sparse.ode_states().unwrap()[0].values());
    let angle = 40. * (0.025_f64).atan();
    let field = document.aliases()["z"].downcast().unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let value = last.field_value(&model, field, 0).unwrap();
    assert_eq!(value.component_count(), 2);
    for component in 0..2 {
        let actual = value.component(component).unwrap();
        let amplitude = (component + 1) as f64;
        assert!((actual.0 - amplitude * angle.cos()).abs() < 1e-11);
        assert!((actual.1 - amplitude * angle.sin()).abs() < 1e-11);
    }
    assert!(last.field_value(&model, field, 1).is_err());
    let other = ModelDocument::compile("other.eqi", &SOURCE.replace("u=3", "u=4")).unwrap();
    assert!(
        last.field_value(
            &ModelEnvelope::from_program(other.program()).unwrap(),
            field,
            0
        )
        .is_err()
    );

    for (coordinate, actual) in last.state_coordinates().iter().zip(last.values()) {
        let expected = if coordinate.field().erase() == document.aliases()["z"] {
            let amplitude = (coordinate.component() + 1) as f64;
            amplitude
                * if coordinate.is_imaginary() {
                    angle.sin()
                } else {
                    angle.cos()
                }
        } else if coordinate.field().erase() == document.aliases()["u"] {
            3. * angle.cos()
        } else {
            3. * angle.sin()
        };
        assert!((actual - expected).abs() < 1e-11);
    }
    let prefix = run(plan.clone(), state, 0.5, vec![0.5]);
    let checkpoint = prefix.ode_states().unwrap()[0].to_bytes().unwrap();
    let restart = CommonOdeState::from_bytes(&checkpoint, &plan).unwrap();
    let resumed = run(plan, restart, 1., vec![1.]);
    for (a, b) in resumed.ode_states().unwrap()[0]
        .values()
        .iter()
        .zip(last.values())
    {
        assert!((a - b).abs() < 1e-11);
    }
}

#[test]
fn midpoint_phase_converges_without_erasing_error_or_normalizing_samples() {
    for dense_mass in [false, true] {
        let source = if dense_mass {
            SOURCE.replace(
                "derivative(z)=math.complex(0,1)*omega*z;",
                r#"
                2*derivative(z)[0]+derivative(z)[1]=math.complex(0,1)*omega*(2*z[0]+z[1]);
                derivative(z)[0]+2*derivative(z)[1]=math.complex(0,1)*omega*(z[0]+2*z[1]);
            "#,
            )
        } else {
            SOURCE.to_owned()
        };
        let document = ModelDocument::compile("oscillators.eqi", &source).unwrap();
        let mut errors = Vec::new();
        for step in [0.1, 0.05] {
            let plan = resolve(&document, step);
            if dense_mass {
                assert_eq!(
                    plan.equation_class(),
                    eqiora::time::TimeEquationClass::MassMatrix {
                        rank: eqiora::time::MassMatrixRank::Full
                    }
                );
            }
            let result = run(
                plan.clone(),
                plan.initial_state(0.).unwrap(),
                1.,
                vec![step / 2., 1.],
            );
            let projection = |state: &CommonOdeState, imaginary| {
                state
                    .state_coordinates()
                    .iter()
                    .zip(state.values())
                    .find_map(|(c, &value)| {
                        (c.field().erase() == document.aliases()["z"]
                            && c.component() == 0
                            && c.is_imaginary() == imaginary)
                            .then_some(value)
                    })
                    .unwrap()
            };
            let samples = result.ode_states().unwrap();
            let a = projection(&samples[1], false);
            let b = projection(&samples[1], true);
            assert!((a.hypot(b) - 1.).abs() < 1e-11);
            let error = (b.atan2(a) - 1.).abs();
            // atan(x)=x-x^3/3+... gives positive phase lag bounded by h²/12.
            assert!(error > step * step / 13. && error < step * step / 12. + 1e-11);
            errors.push(error);
            let observed_norm = projection(&samples[0], false).hypot(projection(&samples[0], true));
            let expected_norm = 1. / (1. + step * step / 4.).sqrt();
            assert!((observed_norm - expected_norm).abs() < 1e-11);
            assert!(observed_norm < 0.9999);
        }
        assert!((3.95..4.05).contains(&(errors[0] / errors[1])));
    }
}

#[test]
fn stale_and_nonfinite_states_and_wrong_initial_domains_fail_closed() {
    let document = ModelDocument::compile("oscillators.eqi", SOURCE).unwrap();
    let plan = resolve(&document, 0.05);
    let initial = plan.initial_state(0.).unwrap();
    let bytes = initial.to_bytes().unwrap();
    let changed = ModelDocument::compile(
        "oscillators.eqi",
        &SOURCE.replace("omega:1/s=1", "omega:1/s=2"),
    )
    .unwrap();
    let changed_plan = resolve(&changed, 0.05);
    assert!(CommonOdeState::from_bytes(&bytes, &changed_plan).is_err());
    assert!(CommonOdeRunRequest::new(changed_plan, initial.clone(), 1., vec![1.]).is_err());
    let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let imaginary = initial
        .state_coordinates()
        .iter()
        .position(|c| c.is_imaginary())
        .unwrap();
    wire["values"][imaginary] = serde_json::json!(f64::MAX);
    let finite = serde_json::to_string(&wire).unwrap();
    // An overflowing JSON real in the imaginary coordinate cannot enter State.
    let nonfinite = finite.replace(&f64::MAX.to_string(), "1e999");
    // serde_json uses exponent notation, so substitute its exact scalar spelling.
    let nonfinite = nonfinite.replace(&serde_json::to_string(&f64::MAX).unwrap(), "1e999");
    assert!(nonfinite.contains("1e999"));
    assert!(CommonOdeState::from_bytes(nonfinite.as_bytes(), &plan).is_err());
    assert!(
        ModelDocument::compile(
            "wrong-domain.eqi",
            &SOURCE.replace("[math.complex(1,0),math.complex(2,0)]", "[true,false]")
        )
        .is_err()
    );
    let bad_controls = plan
        .temporal()
        .clone()
        .with_events(
            1,
            vec![(
                eqiora::Id::new(),
                eqiora::DynQuantity::new(1e-9, eqiora::DimExponents::DIMENSIONLESS),
            )],
        )
        .unwrap();
    let error = CommonOdePlan::resolve(
        &ModelEnvelope::from_program(document.program()).unwrap(),
        document.program(),
        bad_controls,
        ImplicitMidpointTimeBackend::CAPABILITIES,
    )
    .unwrap_err();
    assert!(
        error
            .message()
            .contains("event and forward-sensitivity execution is not implemented")
    );
}

#[test]
fn typed_observables_preserve_complex_components_and_physical_mass_rates() {
    for mass in [false, true] {
        let source = SOURCE.replace("    relation flow {", "    observable sample:array<complex<1>,2> = z;\n    observable rate:array<complex<1/s>,2> = derivative(z);\n    relation flow {");
        let source = if mass {
            source.replace("derivative(z)=math.complex(0,1)*omega*z;", "2*derivative(z)[0]+derivative(z)[1]=math.complex(0,1)*omega*(2*z[0]+z[1]); derivative(z)[0]+2*derivative(z)[1]=math.complex(0,1)*omega*(z[0]+2*z[1]);")
        } else {
            source
        };
        let document = ModelDocument::compile("observable-oscillators.eqi", &source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let plan = resolve(&document, 0.05);
        let result = run(plan.clone(), plan.initial_state(0.).unwrap(), 1., vec![1.]);
        let angle = 40. * (0.025_f64).atan();
        let integral = result
            .observe_time_integral(
                &model,
                document.aliases()["rate"].downcast().unwrap(),
                eqiora_numerics::TimeFunctionalQuadrature::AcceptedStepSimpson,
            )
            .unwrap();
        for component in 0..2 {
            let amplitude = (component + 1) as f64;
            let actual = integral.value().component(component).unwrap();
            assert!((actual.0 - amplitude * (angle.cos() - 1.)).abs() < 1e-11);
            assert!((actual.1 - amplitude * angle.sin()).abs() < 1e-11);
        }
        for name in ["sample", "rate"] {
            let observed = result
                .observe_terminal(&model, document.aliases()[name].downcast().unwrap())
                .unwrap();
            assert_eq!(observed.value().component_count(), 2);
            for component in 0..2 {
                let amplitude = (component + 1) as f64;
                let expected = if name == "sample" {
                    (amplitude * angle.cos(), amplitude * angle.sin())
                } else {
                    (-amplitude * angle.sin(), amplitude * angle.cos())
                };
                let actual = observed.value().component(component).unwrap();
                assert!((actual.0 - expected.0).abs() < 1e-11, "{name}, mass={mass}");
                assert!((actual.1 - expected.1).abs() < 1e-11, "{name}, mass={mass}");
            }
        }
    }
}

#[test]
fn complex_parameter_components_drive_the_same_time_program_and_jvp() {
    use eqiora::time::ParametricTimeSystem;
    for shaped in [false, true] {
        let declaration = if shaped {
            "parameter rotation:array<complex<1/s>,2>=[math.complex(0,1),math.complex(0,1)];"
        } else {
            "parameter rotation:complex<1/s>=math.complex(0,1);"
        };
        let flow = if shaped {
            "derivative(z)[0]=rotation[0]*z[0]; derivative(z)[1]=rotation[1]*z[1];"
        } else {
            "derivative(z)=rotation*z;"
        };
        let source = SOURCE
            .replace(
                "parameter omega:1/s=1;",
                &format!("parameter omega:1/s=1; {declaration}"),
            )
            .replace("derivative(z)=math.complex(0,1)*omega*z;", flow);
        let document = ModelDocument::compile("parameter-oscillators.eqi", &source).unwrap();
        let program = FirstOrderProgram::lower(
            document.program(),
            document.aliases()["flow"].downcast().unwrap(),
        )
        .unwrap();
        let plan = resolve(&document, 0.05);
        let initial = plan.initial_state(0.).unwrap();
        let mut direction = vec![0.; program.parameter_coordinates().len()];
        let mut output = vec![0.; initial.values().len()];
        for (index, coordinate) in program.parameter_coordinates().iter().enumerate() {
            if coordinate.symbol()
                == eqiora::kernel::SymbolRef::Parameter(
                    document.aliases()["rotation"].downcast().unwrap(),
                )
                && coordinate.is_imaginary()
            {
                direction[index] = 1.;
            }
        }
        program
            .rhs_parameter_jvp(0., initial.values(), &direction, &mut output)
            .unwrap();
        for (coordinate, actual) in initial.state_coordinates().iter().zip(output) {
            let expected = if coordinate.field().erase() == document.aliases()["z"]
                && coordinate.is_imaginary()
            {
                (coordinate.component() + 1) as f64
            } else {
                0.
            };
            assert!((actual - expected).abs() < 1e-13);
        }
        let result = run(plan, initial, 1., vec![1.]);
        let last = &result.ode_states().unwrap()[0];
        let angle = 40. * (0.025_f64).atan();
        for (coordinate, actual) in last.state_coordinates().iter().zip(last.values()) {
            if coordinate.field().erase() == document.aliases()["z"] {
                let expected = (coordinate.component() + 1) as f64
                    * if coordinate.is_imaginary() {
                        angle.sin()
                    } else {
                        angle.cos()
                    };
                assert!((actual - expected).abs() < 1e-11);
            }
        }
    }
}

#[test]
fn complex_payload_and_coordinate_precision_require_advertised_capabilities() {
    use eqiora::time::TimeBackendCapabilities;
    use eqiora::{ScalarDomain, ScalarType};
    let document = ModelDocument::compile("capabilities.eqi", SOURCE).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = resolve(&document, 0.05);
    for unsupported in [
        TimeBackendCapabilities::new(
            ImplicitMidpointTimeBackend::IDENTITY,
            &[ScalarDomain::Real],
            &[ScalarType::F64],
        ),
        TimeBackendCapabilities::new(
            ImplicitMidpointTimeBackend::IDENTITY,
            &[ScalarDomain::Real, ScalarDomain::Complex],
            &[ScalarType::F32],
        ),
    ] {
        let error = CommonOdePlan::resolve(
            &model,
            document.program(),
            plan.temporal().clone(),
            unsupported,
        )
        .unwrap_err();
        assert!(error.message().contains("does not support"));
    }
    let resolved = ResolvedCommonPlan::Ode(Box::new(plan));
    let bytes = resolved.to_bytes().unwrap();
    let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire["temporal"]["coordinates"], "real-f64");
    wire["temporal"]["coordinates"] = "real-f32".into();
    assert!(
        ResolvedCommonPlan::from_bytes(
            &serde_json::to_vec(&wire).unwrap(),
            &eqiora::solver::REFERENCE_LINEAR_SOLVER,
            ImplicitMidpointTimeBackend::CAPABILITIES
        )
        .is_err()
    );
}
