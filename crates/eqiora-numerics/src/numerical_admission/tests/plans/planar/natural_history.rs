use super::*;
use crate::numerical_admission::scalar::balance::{evaluations, reset_evaluations};

#[test]
fn nonzero_natural_flux_preserves_affine_history() {
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
    for slope in [-2., 1.] {
        let source = format!(
            r#"
public component Heat(
    support body: volume(ambient_dimension = 2),
    support left: boundary(parent = body),
    support right: boundary(parent = body),
    support bottom: boundary(parent = body),
    support top: boundary(parent = body)
) {{
    coordinate x: m on body from body[0];
    state u: m on body in smooth;
    initial {{ u = {slope} * x; }}
    relation balance on body {{ 3[s/m^2]*derivative(u)-div(grad(u))=0[1/m]; }}
    relation prescribed on left {{ trace(u)=0[m]; }}
    relation outlet on right {{ normal(grad(u))={slope}; }}
    relation bottom_flux on bottom {{ normal(grad(u))=0; }}
    relation top_flux on top {{ normal(grad(u))=0; }}
}}
"#
        );
        let model = compile_model(
            "natural-history.eqi",
            &source,
            &geometry,
            "Heat",
            &supports,
            &[],
        );
        for permuted in [false, true] {
            let owner = resources(&geometry, permuted);
            // u=slope*x has zero Laplacian and time derivative. At x=1 the
            // outward normal is +e_x, so grad(u).n=slope. This nonzero load
            // exactly balances stiffness at the free outlet and center nodes.
            let expected = owner
                .simplicial_mesh()
                .unwrap()
                .mesh()
                .vertices()
                .iter()
                .map(|point| slope * point[0])
                .collect::<Vec<_>>();
            let resolved = ResolvedCommonPlan::resolve(
                &model,
                owner,
                CommonSpatialPolicy::P1,
                CommonSolvePolicy::Linear(exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-12,
                    1e-14,
                    NonZeroUsize::new(100).unwrap(),
                )),
                None,
                Some(CommonTemporalPolicy::backward_euler(0.25).unwrap()),
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap();
            let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
            let initial = resolved
                .as_spatial()
                .unwrap()
                .initial_state(Some(0.), Vec::new())
                .unwrap();
            assert_eq!(initial.spatial_values().unwrap(), expected);
            // Source evaluation retains -0 from a negative slope at x=0.
            // A physically equal essential trace must not rewrite State words.
            for (actual, expected) in initial.spatial_values().unwrap().iter().zip(&expected) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
            let run =
                CommonTransientRunRequest::from_steps(resolved.clone(), initial, 3, vec![1, 2, 3])
                    .unwrap();
            reset_evaluations();
            let std::ops::ControlFlow::Continue(outputs) = run
                .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
                .unwrap()
            else {
                panic!("affine natural-flux equilibrium must complete")
            };
            assert_eq!(outputs.len(), 3);
            let trajectory =
                crate::CommonTrajectory::accept_spatial_transient(run, outputs.clone()).unwrap();
            assert_eq!(
                evaluations(),
                0,
                "retaining outputs must not request work observations"
            );
            let bytes = trajectory.to_bytes().unwrap();
            let restored = crate::CommonTrajectory::from_bytes(&bytes, &resolved).unwrap();
            assert_eq!(evaluations(), 0, "replay must only revalidate acceptance");
            assert!(trajectory.spatial_step_balance(3).is_err());
            assert_eq!(evaluations(), 0, "an absent output must not evaluate work");
            for output_index in 0..3 {
                let balance = trajectory.spatial_step_balance(output_index).unwrap();
                assert_eq!(
                    evaluations(),
                    output_index + 1,
                    "only the requested output is observed"
                );
                // Exact outward work on x=1: dt*integral(u*du/dn)=slope^2/4.
                // The left essential trace is zero and supplies no work.
                let work = balance
                    .boundary_work()
                    .iter()
                    .map(|term| term.work().value())
                    .sum::<f64>();
                assert!((work - 0.25 * slope * slope).abs() < 1e-11);
                // On x=0, the outward derivative is -slope. Each of the two
                // endpoint hats integrates to 1/2 along the unit edge. The
                // trace is zero, but each physical weak action is -slope/2 m.
                let prescribed = balance
                    .boundary_work()
                    .iter()
                    .filter(|term| term.action().is_some())
                    .collect::<Vec<_>>();
                assert_eq!(prescribed.len(), 2);
                for term in prescribed {
                    let action = term.action().unwrap();
                    assert_eq!(
                        action.dim(),
                        eqiora_core::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap()
                    );
                    assert!((action.value() + slope / 2.).abs() < 1e-11);
                    assert_eq!(term.work().value(), 0.);
                    let crate::SpatialBoundarySource::Prescribed {
                        entity,
                        component,
                        slot,
                        ..
                    } = term.source()
                    else {
                        panic!("zero-trace reaction must retain its prescribed coordinate");
                    };
                    assert_eq!(entity.dimension(), 0);
                    assert_eq!((*component, *slot), (0, 0));
                }
                assert!(
                    balance
                        .defects()
                        .unwrap()
                        .iter()
                        .all(|(_, defect)| defect.value().abs() < 1e-11)
                );
            }

            assert_eq!(trajectory.to_bytes().unwrap(), bytes);
            assert_eq!(restored.spatial_steps(), trajectory.spatial_steps());
            for output_index in 0..3 {
                assert_eq!(
                    restored.spatial_step_balance(output_index).unwrap(),
                    trajectory.spatial_step_balance(output_index).unwrap(),
                );
            }

            for (_, state) in outputs {
                let state = state.state();
                let state = CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap();
                for (actual, expected) in state.spatial_values().unwrap().iter().zip(&expected) {
                    assert!((actual - expected).abs() < 1e-11);
                }
            }
        }
    }
}
