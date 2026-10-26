//! Exact rational recurrences, independent of implementation-produced trajectories.
use super::*;
use crate::region_assembly::mapping::{RegionAlgebraicCoordinates, RegionDofMap};
use crate::spatial_expression::Coefficient;
use eqiora_time::TimeMethod;

fn exercise<S: Coefficient + Send + Sync>(
    plan: &CommonSpatialPlan,
    equations: &ExecutableSpatialEquations<S>,
    complex: bool,
    q1: bool,
    nonlinear_storage: bool,
) where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
    let initial = coordinates(
        &vec![
            C::new(1., if complex { 2. } else { 0. });
            if q1 {
                if nonlinear_storage { 3 } else { 1 }
            } else {
                2
            }
        ],
        complex,
    );
    let saved = initial.clone();
    let mut point = initial.clone();
    let step = if nonlinear_storage { 0.25 } else { 1. / 16. };
    // Heat: Q1 lambda=K/M=4/(1/3)=12; TPFA uniform mode lambda=4/(1/2)=8.
    // Midpoint amplification is (1-h*lambda/2)/(1+h*lambda/2).
    // d(u²)/dt=-u² has the nonzero branch u'=-u/2: h=1/4 gives 15/17.
    let factor: f64 = if nonlinear_storage {
        15. / 17.
    } else if q1 {
        5. / 11.
    } else {
        3. / 5.
    };
    // The same rational amplification differentiates the initial uniform mode.
    // For d(u²)/dt=-u² this checks the nonlinear storage Jacobian, not an
    // assumed constant mass. Complex coordinates include both channels.
    {
        use eqiora_solver::{CanonicalCsrSystemView, LinearOperator, LinearOperatorOrientation};
        let mut full_step = system
            .midpoint_step(0., step, &initial, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let time_plan = eqiora_time::TimePlan::new(
            TimeMethod::ImplicitMidpoint,
            0.,
            step,
            1e-10,
            vec![1e-12; initial.len()],
            vec![step],
        )
        .unwrap();
        let solution = eqiora_time::ImplicitMidpointTimeBackend::new()
            .solve_with_stages(
                &initial,
                &time_plan,
                eqiora_time::TimeEquationClass::GeneralImplicitDae,
                |_, _, _| Ok(full_step.physical.clone()),
            )
            .unwrap();
        for time in [0.25 * step, 0.5 * step, step] {
            full_step.observations.push(
                system
                    .solve_midpoint_observation(
                        &solution.history().unwrap().steps()[0],
                        &full_step.physical,
                        None,
                        time,
                        &REFERENCE_LINEAR_SOLVER,
                    )
                    .unwrap(),
            );
        }
        let partials = system
            .midpoint_partials(0., step, &initial, &full_step, &[])
            .unwrap();
        assert_eq!(partials.unknowns.len(), initial.len());
        assert!(partials.parameters.is_empty());
        assert_eq!(partials.physical, (0..initial.len()).collect::<Vec<_>>());
        let mut rhs = vec![0.; initial.len()];
        partials.history.apply(&initial, &mut rhs).unwrap();
        for value in &mut rhs {
            *value = -*value;
        }
        let canonical =
            CanonicalCsrSystemView::new(&partials.state, &rhs, LinearOperatorProperties::General)
                .unwrap();
        let derivative = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan.linear())
            .solve_canonical_oriented(&canonical, &rhs, LinearOperatorOrientation::Normal)
            .unwrap();
        assert_eq!(partials.observations.len(), 3);
        for observation in &partials.observations {
            let beta = observation.time / step;
            let alpha = 1. - beta;
            for i in 0..initial.len() {
                assert_eq!(observation.unknowns[i], i);
                assert_eq!(observation.weights[i], beta);
                assert_eq!(observation.history[i], alpha);
            }
            close(
                &observation.physical,
                &initial
                    .iter()
                    .map(|value| (alpha + beta * factor) * value)
                    .collect::<Vec<_>>(),
            );
            close(
                &observation
                    .unknowns
                    .iter()
                    .enumerate()
                    .map(|(i, &unknown)| {
                        observation.weights[i] * derivative.values()[unknown]
                            + observation.history[i] * initial[i]
                    })
                    .collect::<Vec<_>>(),
                &initial
                    .iter()
                    .map(|value| (alpha + beta * factor) * value)
                    .collect::<Vec<_>>(),
            );
        }
        close(
            derivative.values(),
            &initial
                .iter()
                .map(|value| factor * value)
                .collect::<Vec<_>>(),
        );
    }
    for index in 0..4 {
        let (next, evidence) = system
            .midpoint_stage(index as f64 * step, step, &point, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        close(
            &next,
            &initial
                .iter()
                .map(|value| value * factor.powi(index + 1))
                .collect::<Vec<_>>(),
        );
        assert!(evidence.iterations > 0);
        assert_eq!(evidence.iterations, evidence.linear_solves.len());
        assert!(evidence.residual_target().is_err());
        for report in &evidence.linear_solves {
            assert_eq!(report.solver_provider(), plan.admission.linear.provider);
            assert_eq!(report.solver_plan(), plan.linear());
        }
        point = next;
    }
    assert_eq!(initial, saved);
    for bad_step in [0., -1., f64::INFINITY, f64::NAN] {
        assert!(
            system
                .midpoint_stage(0., bad_step, &initial, &REFERENCE_LINEAR_SOLVER)
                .is_err()
        );
        assert_eq!(initial, saved);
    }
}

#[test]
fn admitted_midpoint_stages_preserve_real_complex_heat_and_nonlinear_storage() {
    for complex in [false, true] {
        for spatial in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            for nonlinear in [false, true] {
                // Pure local storage needs no boundary condition; use a spatially constant mode.
                let balance = if nonlinear {
                    "derivative(u*u)=-u*u/1[s];"
                } else {
                    "1[s/m^2]*derivative(u)-div(grad(u))=0[1/m^2];"
                };
                let base = resolve(complex, spatial, 0.25, balance, (!nonlinear).then_some("0"));
                let requested = policy::policy(
                    &base,
                    TimeMethod::ImplicitMidpoint,
                    if nonlinear { 0.25 } else { 1. / 16. },
                );
                let resolved = policy::with_policy(&base, requested).unwrap();
                let plan = resolved.as_spatial().unwrap();
                match plan.admission.recognized_model() {
                    RecognizedNativeModel::Spatial(equations) => exercise(
                        plan,
                        equations,
                        complex,
                        spatial == CommonSpatialPolicy::Q1,
                        nonlinear,
                    ),
                    RecognizedNativeModel::ComplexSpatial(equations) => exercise(
                        plan,
                        equations,
                        complex,
                        spatial == CommonSpatialPolicy::Q1,
                        nonlinear,
                    ),
                    _ => unreachable!(),
                }
            }
        }
    }
}

#[test]
fn a_backward_euler_plan_cannot_execute_an_unrequested_midpoint_stage() {
    let resolved = resolve(
        false,
        CommonSpatialPolicy::Q1,
        0.25,
        "1[s/m^2]*derivative(u)-div(grad(u))=0[1/m^2];",
        Some("0"),
    );
    let plan = resolved.as_spatial().unwrap();
    let RecognizedNativeModel::Spatial(equations) = plan.admission.recognized_model() else {
        unreachable!()
    };
    let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
    let initial = [1.];
    let error = system
        .midpoint_stage(0., 0.25, &initial, &REFERENCE_LINEAR_SOLVER)
        .unwrap_err();
    assert!(error.message().contains("midpoint policy"));
    assert_eq!(initial, [1.]);
}

#[test]
fn a_failed_nonlinear_midpoint_stage_preserves_the_input() {
    let base = resolve(
        false,
        CommonSpatialPolicy::Q1,
        0.25,
        "derivative(u*u)=-u*u/1[s];",
        None,
    );
    let requested = policy::policy(&base, TimeMethod::ImplicitMidpoint, 0.25);
    let CommonSolvePolicy::Newton { linear, .. } = base.canonical_solve_request().unwrap() else {
        unreachable!()
    };
    let resolved = ResolvedCommonPlan::resolve(
        base.model_artifact(),
        base.authenticated_mesh().unwrap(),
        base.canonical_method_request().unwrap(),
        CommonSolvePolicy::Newton {
            linear,
            nonlinear: NonlinearSolvePlan::new(0., 1e-12, NonZeroUsize::new(1).unwrap(), 16)
                .unwrap(),
        },
        base.canonical_scaling_request(),
        Some(requested.into()),
        &REFERENCE_LINEAR_SOLVER,
        Some(eqiora_time::ImplicitMidpointTimeBackend::CAPABILITIES),
        None,
    )
    .unwrap();
    let plan = resolved.as_spatial().unwrap();
    let RecognizedNativeModel::Spatial(equations) = plan.admission.recognized_model() else {
        unreachable!()
    };
    let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
    let initial = [1.; 3];
    // One Newton update from u=1 for this quadratic cannot meet a 1e-10 correction tolerance.
    let error = system
        .midpoint_stage(0., 0.25, &initial, &REFERENCE_LINEAR_SOLVER)
        .unwrap_err();
    assert_eq!(
        error.code(),
        eqiora_core::diagnostic::codes::NUMERICAL_SOLVE_FAILED
    );
    assert!(error.message().contains("iterations"));
    assert_eq!(initial, [1.; 3]);
}
