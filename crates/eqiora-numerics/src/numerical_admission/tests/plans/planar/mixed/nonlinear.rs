use super::*;
use crate::common_result::evidence::CommonSolveHistory;

#[test]
fn nonlinear_mini_history_checks_continuity_and_recovers_pressure_through_common_newton() {
    let geometry = geometry(true);
    let body = geometry.entity_set("body").unwrap();
    let mut supports = vec![("body", body, None)];
    supports.extend(["left", "right", "bottom", "top"].map(|name| {
        (
            name,
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    for force in [0., 1.] {
        let source = String::from(
            "public operator dyad(input left:spatial[1],input right:spatial[1]):spatial[2]=component(left,0)*component(right,1);\n",
        ) + &FLOW
            .replace("variable u:", "state u:")
            .replace("variable p:", "state p:")
            .replace(
                "{-div(2[kg/(m*s)]",
                "{1[kg/m^3]*derivative(u)+div(1[kg/m^3]*dyad(left=u,right=u))-div(2[kg/(m*s)]",
            )
            .replace(
                "relation constraint on body",
                "initial {u=grad(g); p=0[kg/(m*s^2)];} relation constraint on body",
            )
            .replace(
                "f=0[kg/(m*s^2)]",
                &format!("f={force}[kg/(m^2*s^2)]*(coordinate(0)-1[m])"),
            );
        let model = compile_model(
            "nonlinear-mini-history.eqi",
            &source,
            &geometry,
            "Throughflow",
            &supports,
            &[],
        );
        for permuted in [false, true] {
            let resources = open_resources(&geometry, permuted);
            let vertices = resources
                .simplicial_mesh()
                .unwrap()
                .mesh()
                .vertices()
                .to_vec();
            let resolved = ResolvedCommonPlan::resolve(
                &model,
                resources,
                CommonMethodRequest::Exact {
                    spatial: CommonSpatialPolicy::MiniP1,
                    formulation: FormulationKind::MixedGalerkin,
                },
                CommonSolvePolicy::Newton {
                    linear: exact_reference_linear(
                        LinearSolver::BiConjugateGradientStabilized,
                        0.,
                        1e-12,
                        NonZeroUsize::new(1000).unwrap(),
                    ),
                    nonlinear: NonlinearSolvePlan::new(
                        0.,
                        1e-12,
                        NonZeroUsize::new(12).unwrap(),
                        12,
                    )
                    .unwrap(),
                },
                None,
                Some(CommonBackwardEuler::from_seconds(0.25).unwrap()),
                &REFERENCE_LINEAR_SOLVER,
                None,
            )
            .unwrap();
            let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
            let plan = resolved.as_spatial().unwrap();
            let initial = plan.initial_state(0., vec![]).unwrap();
            let before = initial.to_bytes().unwrap();
            let mut invalid = initial.spatial_values().unwrap().to_vec();
            let mut offset = 0;
            for (_, ty) in plan.fields() {
                if ty.shape().is_scalar() {
                    offset += 6;
                } else {
                    // An interior MINI bubble preserves every boundary trace,
                    // but its divergence violates the pressure test equations.
                    invalid[offset + 12] += 0.1;
                    break;
                }
            }
            let error = plan.scalar_state(0., invalid).unwrap_err();
            assert!(error.message().contains("algebraic residual"), "{error:?}");
            let run = CommonTransientRunRequest::from_steps(
                resolved.clone(),
                initial.clone(),
                2,
                vec![1, 2],
            )
            .unwrap();
            let std::ops::ControlFlow::Continue(states) = run
                .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
                .unwrap()
            else {
                panic!("uninterrupted mixed Newton Run");
            };
            assert_eq!(initial.to_bytes().unwrap(), before);
            for (_, accepted) in &states {
                let CommonSolveHistory::Newton { iterations, .. } =
                    &accepted.execution.as_ref().unwrap().solve
                else {
                    panic!("Newton history");
                };
                if force == 0. {
                    assert_eq!(*iterations, 0);
                }
                let state = accepted.state();
                let replayed =
                    CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap();
                assert_eq!(replayed, *state);
                let mut values = state.spatial_values().unwrap();
                for (_, ty) in plan.fields() {
                    if ty.shape().is_scalar() {
                        // Uniform u=(1,0) has zero acceleration, convection and
                        // strain. grad(p)=grad(f), with p=f=0 at the outlet.
                        for (actual, point) in values[..6].iter().zip(&vertices) {
                            assert!((actual - force * (point[0] - 1.)).abs() < 1e-9);
                        }
                        values = &values[6..];
                    } else {
                        for pair in values[..12].as_chunks::<2>().0 {
                            assert!((pair[0] - 1.).abs() < 1e-9);
                            assert!(pair[1].abs() < 1e-9);
                        }
                        assert!(values[12..22].iter().all(|value| value.abs() < 1e-9));
                        values = &values[22..];
                    }
                }
                assert!(values.is_empty());
            }
            if force != 0. {
                assert!(matches!(states[0].1.execution.as_ref().unwrap().solve,
                    CommonSolveHistory::Newton { iterations, .. } if iterations > 0));
            }
            let trajectory =
                crate::CommonTrajectory::accept_spatial_transient(run, states).unwrap();
            let result = crate::CommonResult::accept_trajectory(0., trajectory).unwrap();
            let bytes = result.to_bytes().unwrap();
            assert_eq!(
                crate::CommonResult::from_bytes(&bytes, &resolved)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
        }
    }
}
