use super::*;
use eqiora_numerics::{CommonState, CommonTrajectory, CommonTransientRunRequest};

fn heated(conductivity: u32, capacity: i32, heating: u32) -> String {
    format!(
        "{IMPORT}{SIGNATURE}
 state temperature: K on body in h1;
 initial {{ temperature = 300[K]; }}
 instance heat: scalar.TransientHeat2d(body=body, temperature=temperature,
 conductivity={conductivity}[W/(m*K)], capacity={capacity}[J/(m^3*K)], heating={heating}[W/m^3]);
 {} }}",
        ["left", "right", "bottom", "top"].map(fixed).join("\n")
    )
}

fn assert_temperatures(
    plan: &ResolvedCommonPlan,
    state: &CommonState,
    expected: impl Fn(&[f64]) -> f64,
) {
    let spatial = plan.as_spatial().unwrap();
    let fields = spatial.state_fields(state).unwrap().collect::<Vec<_>>();
    assert_eq!(fields.len(), 1);
    let mesh = plan
        .authenticated_mesh()
        .unwrap()
        .cartesian_mesh()
        .unwrap()
        .mesh();
    for (entity, value) in fields[0].2.iter().zip(fields[0].3) {
        let point = mesh.vertex_coordinates(*entity).unwrap();
        assert!(value.is_finite());
        assert!(
            (value - expected(&point)).abs() <= 1e-10,
            "point {point:?}: {value}"
        );
    }
}

#[test]
fn imported_transient_heat_matches_independent_mass_stiffness_load_and_replays() {
    // Four h=1/2 Q1 cells give M=c/9, K=8k/3 and F=q/4, per unit depth.
    // This derives the discrete recurrence independently of Eqiora's output.
    for (conductivity, capacity, heating, dt) in [
        (1, 1, 1, 1. / 24.),
        (2, 1, 1, 1. / 24.),
        (1, 2, 1, 1. / 24.),
        (1, 1, 2, 1. / 24.),
        (1, 1, 1, 1. / 48.),
        (1, -1, 1, 1. / 48.),
    ] {
        let source = heated(conductivity, capacity, heating);
        let plan = resolve_temporal(&source, Some(dt)).unwrap();
        let bytes = plan.to_bytes().unwrap();
        let plan = ResolvedCommonPlan::from_bytes(&bytes, &REFERENCE_LINEAR_SOLVER, &[]).unwrap();
        assert_eq!(plan.to_bytes().unwrap(), bytes);
        let initial = plan
            .as_spatial()
            .unwrap()
            .initial_state(Some(0.), Vec::new())
            .unwrap();
        assert_temperatures(&plan, &initial, |_| 300.);
        let initial_bytes = initial.to_bytes().unwrap();
        let initial = CommonState::from_bytes(&initial_bytes, &plan).unwrap();
        let run =
            CommonTransientRunRequest::from_steps(plan.clone(), initial.clone(), 3, vec![1, 2, 3])
                .unwrap();
        let std::ops::ControlFlow::Continue(outputs) = run
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("three accepted heat steps")
        };
        let mass = f64::from(capacity) / 9.;
        let stiffness = 8. * f64::from(conductivity) / 3.;
        let load = f64::from(heating) / 4.;
        let ratio = mass / (mass + stiffness * dt);
        let mut previous = 0.;
        for (index, (_, output)) in outputs.iter().enumerate() {
            let theta = load / stiffness * (1. - ratio.powi(index as i32 + 1));
            assert_temperatures(&plan, output.state(), |point| {
                if point == [0.5, 0.5].as_slice() {
                    300. + theta
                } else {
                    300.
                }
            });
            // The pre-existing heat-balance budget covers cancellation near 300 K.
            let actual = output.state().spatial_values().unwrap()[4] - 300.;
            assert!(
                ((mass / dt + stiffness) * actual - load - mass / dt * previous).abs() <= 3e-10
            );
            previous = actual;
        }
        assert_eq!(initial.to_bytes().unwrap(), initial_bytes);
        let restart = outputs[0].1.state().clone();
        let resumed =
            CommonTransientRunRequest::from_steps(plan.clone(), restart, 2, vec![2]).unwrap();
        let std::ops::ControlFlow::Continue(resumed) = resumed
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("heat restart")
        };
        assert_eq!(resumed[0].1.state(), outputs[2].1.state());
        let result = CommonResult::accept_trajectory(
            0.,
            CommonTrajectory::accept_spatial_transient(run, outputs).unwrap(),
        )
        .unwrap();
        let bytes = result.to_bytes().unwrap();
        assert_eq!(
            CommonResult::from_bytes(&bytes, &plan)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
    }
}

#[test]
fn imported_transient_heat_flux_preserves_time_linear_affine_manufactured_solution() {
    // T=300+t +/- x: dT/dt=1 K/s, Laplacian(T)=0, source=c*1 K/s.
    // Backward Euler is exact for this time-linear profile. Outward Fourier
    // flux on x=1 is -k*dT/dx; the other natural boundaries are insulated.
    for slope in [-1, 1] {
        let flux = -2 * slope;
        let source = format!("{IMPORT}{SIGNATURE}
 state temperature: K on body in smooth;
 initial {{temperature = 300[K] + {slope}[K/m] * coordinate(0);}}
 instance heat: scalar.TransientHeat2d(body=body, temperature=temperature,
 conductivity=2[W/(m*K)], capacity=2[J/(m^3*K)], heating=2[W/m^3]);
 instance left_value: scalar.FixedTemperature2d(body=body, face=left, temperature=temperature, value=300[K]+time()*1[K/s]);
 instance right_flux: scalar.OutwardHeatFlux2d(body=body, face=right, temperature=temperature, conductivity=2[W/(m*K)], flux={flux}[W/m^2]);
 instance bottom_flux: scalar.OutwardHeatFlux2d(body=body, face=bottom, temperature=temperature, conductivity=2[W/(m*K)], flux=0[W/m^2]);
 instance top_flux: scalar.OutwardHeatFlux2d(body=body, face=top, temperature=temperature, conductivity=2[W/(m*K)], flux=0[W/m^2]);
 }}");
        let plan = resolve_temporal(&source, Some(1. / 24.)).unwrap();
        let initial = plan
            .as_spatial()
            .unwrap()
            .initial_state(Some(0.), Vec::new())
            .unwrap();
        assert_temperatures(&plan, &initial, |point| 300. + f64::from(slope) * point[0]);
        let run =
            CommonTransientRunRequest::from_steps(plan.clone(), initial, 3, vec![1, 2, 3]).unwrap();
        let std::ops::ControlFlow::Continue(outputs) = run
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("manufactured heat steps")
        };
        for (index, (_, state)) in outputs.iter().enumerate() {
            let time = (index + 1) as f64 / 24.;
            assert_temperatures(&plan, state.state(), |point| {
                300. + time + f64::from(slope) * point[0]
            });
        }
        let result = CommonResult::accept_trajectory(
            0.,
            CommonTrajectory::accept_spatial_transient(run, outputs).unwrap(),
        )
        .unwrap();
        let bytes = result.to_bytes().unwrap();
        assert_eq!(
            CommonResult::from_bytes(&bytes, &plan)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
    }
}

#[test]
fn transient_heat_requires_borrowed_state_and_typed_capacity() {
    let good = heated(1, 1, 1);
    let wrong_role = good
        .replace("state temperature", "variable temperature")
        .replace("initial { temperature = 300[K]; }", "");
    let error = fixture(&wrong_role).expect_err("an algebraic Field cannot bind a State slot");
    assert!(
        error.diagnostics().is_some_and(|diagnostics| diagnostics
            .iter()
            .any(|d| d.message().contains("declared state role"))),
        "{error}"
    );
    for wrong in [
        good.replace("capacity=1[J/(m^3*K)]", "capacity=1[W/m^3]"),
        good.replace("heating=1[W/m^3]", "heating=1[K]"),
        good.replace("ambient_dimension = 2", "ambient_dimension = 3"),
    ] {
        let error = fixture(&wrong).expect_err("typed transient binding must reject");
        assert!(
            error
                .diagnostics()
                .is_some_and(|diagnostics| !diagnostics.is_empty())
        );
    }
}
