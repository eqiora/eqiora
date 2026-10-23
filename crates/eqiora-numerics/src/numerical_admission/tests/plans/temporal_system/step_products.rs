//! Step products distinguish fixed committed history from trajectory sensitivity.
use super::*;
use eqiora_ir::{LinearizedOutput, LinearizedRelation};
use eqiora_schema::kernel::KernelNode;
use eqiora_solver::LinearOperatorOrientation;

#[test]
fn accepted_step_parameter_products_hold_physical_history_fixed() {
    for complex in [false, true] {
        for spatial in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            for storage_parameter in [false, true] {
                let geometry = cartesian_interval();
                let body = geometry.entity_set("body").unwrap();
                let scalar = if complex { "complex<1>" } else { "1" };
                let phase = if complex { "math.complex(1,2)" } else { "1" };
                let initial_value = if storage_parameter { phase } else { "0" };
                let balance = if storage_parameter {
                    "p*1[s]*derivative(h)+h=0".to_owned()
                } else {
                    format!("1[s]*derivative(h)+h={phase}*p*(if time()<1.5[s] then 1 else 0)")
                };
                let source = format!(
                    "model Heat(support body:volume(ambient_dimension=1),
                    support left:boundary(parent=body),support right:boundary(parent=body)) {{
                    parameter p:1=1;
                    state h:{scalar} on body in smooth;
                    initial {{h={initial_value};}}
                    relation evolution on body {{{balance};}}
                }}"
                );
                let model = compile_model(
                    "step-products.eqi",
                    &source,
                    &geometry,
                    "Heat",
                    &[
                        ("body", body, None),
                        (
                            "left",
                            geometry.entity_set("left").unwrap(),
                            Some(("body", body)),
                        ),
                        (
                            "right",
                            geometry.entity_set("right").unwrap(),
                            Some(("body", body)),
                        ),
                    ],
                    &[],
                );
                let resolved = ResolvedCommonPlan::resolve(
                    &model,
                    cartesian_box_resources(&geometry, &[2]),
                    spatial,
                    CommonSolvePolicy::Linear(exact_reference_linear(
                        LinearSolver::BiConjugateGradientStabilized,
                        1e-12,
                        1e-14,
                        NonZeroUsize::new(100).unwrap(),
                    )),
                    None,
                    Some(CommonTemporalPolicy::backward_euler(1.).unwrap()),
                    &REFERENCE_LINEAR_SOLVER,
                    None,
                    None,
                )
                .unwrap();
                let plan = resolved.as_spatial().unwrap();
                let parameter = plan
                    .kernel()
                    .nodes()
                    .find_map(|node| match node {
                        KernelNode::Parameter(parameter) => Some(parameter.id()),
                        _ => None,
                    })
                    .expect("retained Model Parameter");
                let field = plan.fields().next().unwrap().0.erase();
                let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
                let before = initial.to_bytes().unwrap();
                let request = CommonTransientRunRequest::from_steps(
                    resolved.clone(),
                    initial.clone(),
                    2,
                    vec![1, 2],
                )
                .unwrap();
                let std::ops::ControlFlow::Continue(steps) = request
                    .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
                    .unwrap()
                else {
                    panic!("unexpected stop")
                };
                for (index, (_, step)) in steps.iter().enumerate() {
                    let linearization = plan.linearize_step(step, &[parameter], &[field]).unwrap();
                    let output = output_tangent(plan, &linearization);
                    let nodes = if spatial == CommonSpatialPolicy::Q1 {
                        3
                    } else {
                        2
                    };
                    let phase = C::new(1., if complex { 2. } else { 0. });
                    // h_next=(h_committed+phase*p*loading)/2. On unloading,
                    // fixed-history dh/dp=0, while a trajectory derivative is phase/4.
                    // For p*h_dot+h=0, dh_next/dp=h_committed/(p+1)^2:
                    // phase/4 at the first step and phase/8 at the second.
                    let expected = if storage_parameter {
                        phase / if index == 0 { 4. } else { 8. }
                    } else if index == 0 {
                        phase / 2.
                    } else {
                        C::new(0., 0.)
                    };
                    close(&output, &coordinates(&vec![expected; nodes], complex));
                    close(
                        linearization.output_values(),
                        &coordinates(
                            &vec![phase / if index == 0 { 2. } else { 4. }; nodes],
                            complex,
                        ),
                    );
                    let evaluated = plan
                        .differentiate_step(
                            step,
                            &[parameter],
                            Some(&[3.]),
                            &[field],
                            &REFERENCE_LINEAR_SOLVER,
                        )
                        .unwrap();
                    let restored = plan
                        .validate_differentiated_step_values(
                            step,
                            &[parameter],
                            Some(&[3.]),
                            &[field],
                            evaluated.linearization().relation().accepted_unknowns(),
                        )
                        .unwrap();
                    assert_eq!(&restored, evaluated.linearization());
                    let mut altered = restored.relation().accepted_unknowns().to_vec();
                    altered[0] += 0.125;
                    let error = plan
                        .validate_differentiated_step_values(
                            step,
                            &[parameter],
                            Some(&[3.]),
                            &[field],
                            &altered,
                        )
                        .unwrap_err();
                    assert!(error.message().contains("residual"), "{error:?}");
                    assert_eq!(evaluated.linear_solves().len(), 1);
                    assert_eq!(evaluated.linear_solves()[0].solver_plan(), plan.linear());
                    assert_eq!(evaluated.linearization().relation().design_values(), &[3.]);
                    let old = if storage_parameter {
                        phase / if index == 0 { 1. } else { 2. }
                    } else if index == 0 {
                        C::new(0., 0.)
                    } else {
                        phase / 2.
                    };
                    // p=3 is a new solve with the original physical predecessor:
                    // storage: h=3*old/4, dh/dp=old/16;
                    // forcing: h=(old+3*phase*loading)/2, dh/dp=phase*loading/2.
                    let (value, tangent) = if storage_parameter {
                        (old * 0.75, old / 16.)
                    } else if index == 0 {
                        (phase * 1.5, phase / 2.)
                    } else {
                        (old / 2., C::new(0., 0.))
                    };
                    close(
                        evaluated.linearization().output_values(),
                        &coordinates(&vec![value; nodes], complex),
                    );
                    close(
                        &output_tangent(plan, evaluated.linearization()),
                        &coordinates(&vec![tangent; nodes], complex),
                    );
                    assert_eq!(
                        plan.linearize_step(step, &[parameter], &[field]).unwrap(),
                        linearization
                    );
                }
                assert_eq!(initial.to_bytes().unwrap(), before);
            }
        }
    }
}

fn output_tangent(
    plan: &CommonSpatialPlan,
    point: &crate::CommonDifferentiationLinearization,
) -> Vec<f64> {
    let relation = point.relation();
    assert_eq!(relation.parameter_dimension(), 1);
    let rhs = relation
        .design_jacobian()
        .iter()
        .map(|value| -*value)
        .collect::<Vec<_>>();
    let tangent = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan.linear())
        .solve_canonical_oriented(
            relation.state_jacobian(),
            &rhs,
            LinearOperatorOrientation::Normal,
        )
        .unwrap();
    let mut output = vec![0.; point.output_dimension()];
    point.jvp(tangent.values(), &[1.], &mut output).unwrap();
    output
}
