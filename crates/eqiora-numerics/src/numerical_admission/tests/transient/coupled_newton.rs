use super::*;
use crate::common_result::evidence::CommonSolveHistory;

fn resolve(count: usize, renamed: bool, force: f64, maximum: usize) -> ResolvedCommonPlan {
    resolve_with_boundary(count, renamed, force, maximum, false)
}

fn resolve_with_boundary(
    count: usize,
    renamed: bool,
    force: f64,
    maximum: usize,
    natural: bool,
) -> ResolvedCommonPlan {
    let geometry = rectangle();
    let names = if renamed {
        ["cobalt", "amber", "violet"]
    } else {
        ["a", "b", "c"]
    };
    let mut source = String::from(
        "public operator dyad(input left: spatial[1], input right: spatial[1]): spatial[2] = component(left, 0) * component(right, 1);\npublic component Coupled(\n support body: volume(ambient_dimension = 2),\n support left: boundary(parent = body),\n support right: boundary(parent = body),\n support bottom: boundary(parent = body),\n support top: boundary(parent = body),\n parameter rho: kg/m^3, parameter mu: kg/(m*s),\n parameter rate: kg/(m^3*s), parameter force: kg/(m^2*s^2)\n) {\n",
    );
    if natural {
        source += "variable g: m^2/s on body in smooth; relation potential on body { g = 1[m/s]*coordinate(0) + 2[m/s]*coordinate(1); }\n";
    }
    let initial = if natural { "grad(g)" } else { "0" };
    for name in &names[..count] {
        source += &format!(
            "state {name}: vector<m/s, 2> on body in smooth;\ninitial {{ {name} = {initial}; }}\n"
        );
    }
    for (i, name) in names.iter().enumerate().take(count) {
        source += &format!(
            "relation balance{i} on body {{ rho*derivative({name}) + div(rho*dyad(left = {name}, right = {name})) - div(mu*grad({name}))"
        );
        for (j, other) in names.iter().enumerate().take(count) {
            if i != j {
                source += &format!(" + rate*({name} - {other})");
            }
        }
        if i == 0 {
            source += " - force*grad(coordinate(0))";
        }
        source += " = 0; }\n";
        for boundary in ["left", "right", "bottom", "top"] {
            if natural && boundary == "right" {
                source +=
                    &format!("relation outlet{i} on right {{ normal(mu*grad({name})) = 0; }}\n");
            } else {
                let datum = if natural { "trace(grad(g))" } else { "0" };
                source += &format!(
                    "relation {boundary}{i} on {boundary} {{ trace({name}) = {datum}; }}\n"
                );
            }
        }
    }
    source += "}";
    let supports = [
        ("body", "region", None),
        ("left", "left", Some("body")),
        ("right", "right", Some("body")),
        ("bottom", "bottom", Some("body")),
        ("top", "top", Some("body")),
    ]
    .map(|(name, set, parent)| {
        (
            name,
            geometry.entity_set(set).unwrap(),
            parent.map(|parent| (parent, geometry.entity_set("region").unwrap())),
        )
    });
    let model = compile_model(
        "coupled.eqi",
        &source,
        &geometry,
        "Coupled",
        &supports,
        &[
            (
                "rho",
                DynQuantity::new(
                    4.,
                    DimExponents::from_integers([1, -3, 0, 0, 0, 0, 0]).unwrap(),
                ),
            ),
            (
                "mu",
                DynQuantity::new(
                    0.1,
                    DimExponents::from_integers([1, -1, -1, 0, 0, 0, 0]).unwrap(),
                ),
            ),
            (
                "rate",
                DynQuantity::new(
                    2.,
                    DimExponents::from_integers([1, -3, -1, 0, 0, 0, 0]).unwrap(),
                ),
            ),
            (
                "force",
                DynQuantity::new(
                    force,
                    DimExponents::from_integers([1, -2, -2, 0, 0, 0, 0]).unwrap(),
                ),
            ),
        ],
    );
    let linear = exact_reference_linear(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(1000).unwrap(),
    );
    let plan = ResolvedCommonPlan::resolve(
        &model,
        cartesian_rectangle_resources(&geometry, [3, 3]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Newton {
            linear,
            nonlinear: NonlinearSolvePlan::new(
                1e-10,
                1e-12,
                NonZeroUsize::new(maximum).unwrap(),
                12,
            )
            .unwrap(),
        },
        None,
        Some(CommonBackwardEuler::from_seconds(0.125).unwrap().into()),
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
    )
    .unwrap();
    replay_plan(plan, &REFERENCE_LINEAR_SOLVER)
}

#[test]
fn coupled_region_newton_executes_and_replays_through_the_common_run() {
    for count in [2, 3] {
        let mut norms = Vec::new();
        for renamed in [false, true] {
            let resolved = resolve(count, renamed, 3., 12);
            let plan = resolved.as_spatial().unwrap();
            assert_eq!(plan.fields().len(), count);
            assert!(plan.portable_realization().domains().iter().all(|domain| {
                domain.discretization().quadrature()
                    == QuadraturePolicy::GaussLegendre {
                        points_per_axis: NonZeroUsize::new(3).unwrap(),
                    }
            }));
            assert!(
                RecognizedNativeAdmission::recognize(
                    plan.admission.model(),
                    AuthenticatedCommonMesh {
                        resources: plan.admission.resources().clone(),
                        model_geometries: plan.admission.model_geometries().to_vec(),
                    },
                    &CommonSpatialRequest::Uniform(CommonSpatialPolicy::CellCentered),
                    None
                )
                .is_err(),
                "a weak form cannot silently stand in for an unavailable cell-centered method"
            );
            let initial = plan.initial_state(Some(0.), vec![]).unwrap();
            let before = initial.to_bytes().unwrap();
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
                panic!("uninterrupted Run");
            };
            assert_eq!(before, initial.to_bytes().unwrap());
            for (_, state) in &states {
                let CommonSolveHistory::Newton {
                    iterations,
                    linear_solves,
                    ..
                } = &state.execution.as_ref().unwrap().solve
                else {
                    panic!("Newton history");
                };
                assert!(*iterations > 1);
                assert_eq!(*iterations, linear_solves.len());
            }
            let trajectory =
                crate::CommonTrajectory::accept_spatial_transient(run.clone(), states.clone())
                    .unwrap();
            let bytes = trajectory.to_bytes().unwrap();
            assert_eq!(
                crate::CommonTrajectory::from_bytes(&bytes, &resolved)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
            let result = crate::CommonResult::accept_trajectory(0., trajectory).unwrap();
            let bytes = result.to_bytes().unwrap();
            let result = crate::CommonResult::from_bytes(&bytes, &resolved).unwrap();
            assert_eq!(result.to_bytes().unwrap(), bytes);
            assert_eq!(result.family_name(), "spatial");
            for output in [0, 1] {
                let iterations = result.nonlinear_iterations(Some(output)).unwrap();
                assert!(iterations > 1);
                assert!(
                    result
                        .nonlinear_initial_residual_norm(Some(output))
                        .unwrap()
                        > 0.
                );
                assert!(
                    result.nonlinear_true_residual_norm(Some(output)).unwrap()
                        <= result.nonlinear_residual_target(Some(output)).unwrap()
                );
                for iteration in 0..iterations {
                    assert!(
                        result
                            .solve_evidence(Some(output), Some(iteration))
                            .map(|solve| solve.solver().id())
                            .is_some()
                    );
                    assert!(
                        result
                            .solve_true_residual_norm(Some(output), Some(iteration))
                            .unwrap()
                            <= result
                                .solve_evidence(Some(output), Some(iteration))
                                .map(|solve| solve.residual_target())
                                .unwrap()
                    );
                }
                assert!(
                    result
                        .solve_evidence(Some(output), Some(iterations))
                        .map(|solve| solve.solver().id())
                        .is_none()
                );
                assert!(
                    result
                        .solve_true_residual_norm(Some(output), None)
                        .is_none()
                );
            }
            assert!(result.nonlinear_iterations(None).is_none());
            assert!(result.nonlinear_iterations(Some(2)).is_none());
            assert!(
                result
                    .solve_evidence(Some(0), None)
                    .map(|solve| solve.solver().id())
                    .is_none(),
                "a Newton step has no fabricated primal linear receipt"
            );
            let values = states.last().unwrap().1.state().spatial_values().unwrap();
            let (fields, remainder) = values.as_chunks::<32>();
            assert!(remainder.is_empty());
            let mut field_norms = fields
                .iter()
                .map(|field| field.iter().fold(0_f64, |n, value| n.hypot(*value)))
                .collect::<Vec<_>>();
            assert_eq!(field_norms.len(), count);
            assert!(
                field_norms.iter().all(|norm| *norm > 1e-6),
                "unforced Fields must respond through feedback"
            );
            field_norms.sort_by(f64::total_cmp);
            norms.push(field_norms);
            let mut changed = states.clone();
            if let CommonSolveHistory::Newton {
                iterations,
                linear_solves,
                ..
            } = &mut changed[0].1.execution.as_mut().unwrap().solve
            {
                *iterations = 0;
                linear_solves.clear();
            }
            assert!(
                crate::CommonTrajectory::accept_spatial_transient(run.clone(), changed).is_err()
            );
        }
        for (a, b) in norms[0].iter().zip(&norms[1]) {
            assert!((a - b).abs() < 1e-9);
        }
    }
    let resolved = resolve(3, false, 3., 1);
    let plan = resolved.as_spatial().unwrap();
    let initial = plan.initial_state(Some(0.), vec![]).unwrap();
    let before = initial.to_bytes().unwrap();
    assert!(
        plan.advance_scalar(&initial, &REFERENCE_LINEAR_SOLVER)
            .is_err()
    );
    assert_eq!(initial.to_bytes().unwrap(), before);

    let resolved = resolve(3, false, 0., 12);
    let plan = resolved.as_spatial().unwrap();
    let initial = plan.initial_state(Some(0.), vec![]).unwrap();
    let mut accepted = plan
        .advance_scalar(&initial, &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    accepted.reaccept_spatial(plan).unwrap();
    let CommonSolveHistory::Newton {
        iterations,
        linear_solves,
        ..
    } = &accepted.execution.as_ref().unwrap().solve
    else {
        panic!("Newton history");
    };
    assert_eq!(*iterations, 0);
    assert!(linear_solves.is_empty());
}

#[test]
fn spatial_newton_replay_rejects_foreign_solver_and_execution_metadata() {
    use crate::common_result::evidence::{CommonExecutionTopology, CommonSolveEvidence};

    let resolved = resolve(2, false, 3., 12);
    let plan = resolved.as_spatial().unwrap();
    let initial = plan.initial_state(Some(0.), vec![]).unwrap();
    let accepted = plan
        .advance_scalar(&initial, &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let run = CommonTransientRunRequest::from_steps(resolved.clone(), initial, 1, vec![1]).unwrap();
    crate::CommonTrajectory::accept_spatial_transient(run.clone(), vec![(1, accepted.clone())])
        .unwrap();
    let mutations: [fn(&mut CommonSolveEvidence); 9] = [
        |solve| solve.solver.id.push_str(".foreign"),
        |solve| solve.solver.implementation_version.push_str(".foreign"),
        |solve| solve.solver.libraries.push(("foreign".into(), "1".into())),
        |solve| solve.execution.provider.id.push_str(".foreign"),
        |solve| solve.execution.adapter.push_str(".foreign"),
        |solve| solve.execution.topology = CommonExecutionTopology::Host { workers: 2 },
        |solve| {
            solve
                .verification
                .provider
                .implementation_version
                .push_str(".foreign")
        },
        |solve| solve.verification.adapter.push_str(".foreign"),
        |solve| solve.verification.topology = CommonExecutionTopology::Host { workers: 2 },
    ];
    for mutate in mutations {
        let mut changed = accepted.clone();
        let CommonSolveHistory::Newton { linear_solves, .. } =
            &mut changed.execution.as_mut().unwrap().solve
        else {
            panic!("Newton history");
        };
        mutate(&mut linear_solves[0]);
        assert!(
            crate::CommonTrajectory::accept_spatial_transient(run.clone(), vec![(1, changed)])
                .is_err()
        );
    }
    let mut changed = accepted;
    changed
        .execution
        .as_mut()
        .unwrap()
        .assembly
        .adapter
        .push_str(".foreign");
    assert!(crate::CommonTrajectory::accept_spatial_transient(run, vec![(1, changed)]).is_err());
}

#[test]
fn coupled_natural_outlet_preserves_constant_transport_and_replays_newton_updates() {
    for count in [2, 3] {
        for force in [0., 3.] {
            let resolved = resolve_with_boundary(count, true, force, 12, true);
            let plan = resolved.as_spatial().unwrap();
            let initial = plan.initial_state(Some(0.), vec![]).unwrap();
            let before = initial.to_bytes().unwrap();
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
                panic!("uninterrupted Run");
            };
            assert_eq!(initial.to_bytes().unwrap(), before);
            for (_, accepted) in &states {
                let CommonSolveHistory::Newton {
                    iterations,
                    linear_solves,
                    ..
                } = &accepted.execution.as_ref().unwrap().solve
                else {
                    panic!("Newton history");
                };
                assert_eq!(*iterations, linear_solves.len());
                if force == 0. {
                    // Constant transport has zero divergence and diffusion. Equal
                    // Fields also cancel every pairwise volume interaction. The
                    // integrated volume flux must cancel its natural outlet flux.
                    assert_eq!(*iterations, 0);
                    for pair in accepted
                        .state()
                        .spatial_values()
                        .unwrap()
                        .as_chunks::<2>()
                        .0
                    {
                        assert!((pair[0] - 1.).abs() < 1e-12);
                        assert!((pair[1] - 2.).abs() < 1e-12);
                    }
                } else {
                    assert!(*iterations > 0);
                }
            }
            let trajectory =
                crate::CommonTrajectory::accept_spatial_transient(run, states).unwrap();
            let result = crate::CommonResult::accept_trajectory(0., trajectory).unwrap();
            let bytes = result.to_bytes().unwrap();
            let replayed = crate::CommonResult::from_bytes(&bytes, &resolved).unwrap();
            assert_eq!(replayed.to_bytes().unwrap(), bytes);
            for output in [0, 1] {
                assert!(
                    replayed.nonlinear_true_residual_norm(Some(output)).unwrap()
                        <= replayed.nonlinear_residual_target(Some(output)).unwrap()
                );
            }
        }
    }
}
