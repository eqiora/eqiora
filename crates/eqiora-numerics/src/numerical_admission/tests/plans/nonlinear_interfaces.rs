//! Distinct transport coefficients prevent missing interface fluxes from cancelling.
use super::*;
use crate::common_result::evidence::CommonSolveHistory;

fn source(connections: bool, force: f64) -> String {
    let mut source = String::from(
        "public operator dyad(input left:spatial[1],input right:spatial[1]):spatial[2]=component(left,0)*component(right,1);\nmodel TransportPartition() { parameter mu:kg/(m*s)=0.1; parameter rate:kg/(m^3*s)=2;",
    );
    source += &format!("parameter force:kg/(m^2*s^2)={force};");
    for (index, density) in [1, 2, 4].iter().enumerate() {
        source += &format!("domain body{index}=box({index},{end},0,1);
            domain left{index}=boundary(body{index},axis=0,side=lower);
            domain right{index}=boundary(body{index},axis=0,side=upper);
            domain bottom{index}=boundary(body{index},axis=1,side=lower);
            domain top{index}=boundary(body{index},axis=1,side=upper);
            parameter rho{index}:kg/m^3={density};
            variable g{index}:m^2/s on body{index} in smooth;
            relation potential{index} on body{index} {{g{index}=1[m/s]*coordinate(0)+2[m/s]*coordinate(1);}}", end=index+1);
        for (field, other) in [("u", "v"), ("v", "u")] {
            source += &format!(
                "state {field}{index}:vector<m/s,2> on body{index} in smooth;
                initial {{ {field}{index}=grad(g{index}); }}
                relation balance_{field}{index} on body{index} {{
                    rho{index}*derivative({field}{index})
                    +div(rho{index}*dyad(left={field}{index},right={field}{index}))
                    -div(mu*grad({field}{index}))+rate*({field}{index}-{other}{index})"
            );
            if index == 0 && field == "u" {
                source += "-force*grad(coordinate(0))";
            }
            source += "=0;}";
            for side in ["left", "right", "bottom", "top"] {
                if (side == "left" && index > 0) || (side == "right" && index < 2) {
                    continue;
                }
                if side == "right" {
                    source += &format!(
                        "relation outlet_{field}{index} on right{index} {{normal(mu*grad({field}{index}))=0;}}"
                    );
                } else {
                    source += &format!(
                        "relation fixed_{side}_{field}{index} on {side}{index} {{trace({field}{index})=trace(grad(g{index}));}}"
                    );
                }
            }
        }
    }
    for index in 0..2 {
        let next = index + 1;
        if connections {
            for field in ["u", "v"] {
                source += &format!("instance outgoing_{field}{index}: VectorInterface(body=body{index},face=right{index},value={field}{index},viscosity=mu);
                    instance incoming_{field}{index}: VectorInterface(body=body{next},face=left{next},value={field}{next},viscosity=mu);
                    connect outgoing_{field}{index}.edge,incoming_{field}{index}.edge;");
            }
        } else {
            source += &format!("domain contact{index}=interface(right{index},left{next});");
            for field in ["u", "v"] {
                source += &format!("relation trace_{field}{index} on contact{index} {{trace({field}{index})=trace({field}{next});}}
                    relation flux_{field}{index} on contact{index} {{normal(mu*grad({field}{index}))=normal(mu*grad({field}{next}));}}");
            }
        }
    }
    if connections {
        source = String::from(
            r#"
public connector VectorBoundary {
    trace value:m/s;
    flux outward_flux:kg/(m*s^2);
    shape spatial_vector;
    frame spatial;
    pairing euclidean_boundary_duality;
    orientation parent_outward;
}
public component VectorInterface(
    support body:volume(ambient_dimension=2),
    support face:boundary(parent=body),
    variable value:vector<m/s,2> on body in smooth,
    parameter viscosity:kg/(m*s),
    port edge:VectorBoundary over face
) {
    relation carrier on face {
        trace(value)=edge.value;
        normal(viscosity*grad(value))=edge.outward_flux;
    }
}
"#,
        ) + &source;
    }
    source + "}"
}

fn resolve(source: &str) -> Result<ResolvedCommonPlan, Diagnostic> {
    let (transaction, model, _) = eqiora_compiler::compile("nonlinear-interfaces.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    let geometry = CanonicalGeometryV1::decode_cartesian_box_v1_canonical(
        br#"{"schema":"eqiora.cartesian-box-envelope/v1","encoding":"eqiora.canonical-json/v1","length_unit":"metre","bounds":[[0.0,3.0],[0.0,1.0]],"entity_sets":[{"name":"bottom","dimension":1,"members":[2]},{"name":"left","dimension":1,"members":[0]},{"name":"right","dimension":1,"members":[1]},{"name":"top","dimension":1,"members":[3]},{"name":"body","dimension":2,"members":[0]}]}"#,
        eqiora_geometry::CanonicalGeometryLimits::default(),
    ).unwrap();
    ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[6, 2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Newton {
            linear: exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-12,
                1e-14,
                NonZeroUsize::new(1000).unwrap(),
            ),
            nonlinear: NonlinearSolvePlan::new(1e-10, 1e-12, NonZeroUsize::new(12).unwrap(), 12)
                .unwrap(),
        },
        None,
        Some(CommonBackwardEuler::from_seconds(0.125).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
}

#[test]
fn nonlinear_interfaces_complete_both_owned_fluxes_and_replay_coupled_updates() {
    for connections in [false, true] {
        for force in [0., 3.] {
            let resolved = resolve(
                &source(connections, force).replace("TransportPartition", "RenamedPartition"),
            )
            .unwrap();
            let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
            let plan = resolved.as_spatial().unwrap();
            assert_eq!(plan.fields().len(), 6);
            assert_eq!(plan.portable_realization().domains().len(), 3);
            assert_eq!(plan.portable_realization().transformations().len(), 4);
            let initial = plan.initial_state(0., vec![]).unwrap();
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
                let (fields, remainder) =
                    accepted.state().spatial_values().unwrap().as_chunks::<18>();
                assert!(remainder.is_empty());
                assert_eq!(fields.len(), 6);
                if force == 0. {
                    // Every open Region has zero divergence, diffusion, storage
                    // change and feedback. Unequal rho prevents the two missing
                    // interface transport fluxes from cancelling after quotienting.
                    assert_eq!(*iterations, 0);
                    for field in fields {
                        for pair in field.as_chunks::<2>().0 {
                            assert!((pair[0] - 1.).abs() < 1e-12);
                            assert!((pair[1] - 2.).abs() < 1e-12);
                        }
                    }
                } else {
                    assert!(*iterations > 0);
                    for field in fields {
                        let change = field
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|p| (p[0] - 1.).powi(2) + (p[1] - 2.).powi(2))
                            .sum::<f64>();
                        assert!(
                            change > 1e-18,
                            "every unforced Field must receive feedback through the interfaces"
                        );
                    }
                }
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
