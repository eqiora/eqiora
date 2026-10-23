//! Ordinary accepted steps share residual and output products with static solves.
use super::*;
use eqiora::differentiation::{
    AcceptedOutputLinearization, adjoint_output_gradient, forward_output_sensitivity,
};
use eqiora::solver::LinearSolveRequest;
use eqiora_numerics::CommonDifferentiationLinearization;

fn check_products(
    point: &CommonDifferentiationLinearization,
    solver: SolverPlan,
    expected: &[f64],
) {
    let relation = point.relation();
    let request = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, solver);
    let accepted = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
        relation,
        point,
        relation.state_jacobian(),
        point.state_scalar_domain(),
        point.residual_target(),
    )
    .unwrap();
    let properties = relation.state_jacobian().properties();
    let forward = forward_output_sensitivity(&accepted, &[1.], properties, request).unwrap();
    assert_eq!(forward.output_tangent().len(), expected.len());
    for (actual, expected) in forward.output_tangent().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }
    let weights = (1..=expected.len())
        .map(|index| index as f64)
        .collect::<Vec<_>>();
    let adjoint = adjoint_output_gradient(&accepted, &weights, properties, request).unwrap();
    let expected: f64 = expected
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum();
    assert!((adjoint.gradient()[0] - expected).abs() < 1e-10);
}

fn source(equation: &str, boundary: &str) -> String {
    format!(
        "model Heat(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {{
        parameter p:1=2;
        state h:1 on body in h1;
        initial {{h=0;}}
        relation evolution on body {{{equation}}}
        {boundary}
    }}"
    )
}

#[test]
fn accepted_step_products_include_current_boundary_lifts_without_history_feedback() {
    let source = source(
        "1[s/m^2]*derivative(h)-div(grad(h))=0;",
        "relation lower on left {trace(h)=p*time()/1[s];}
         relation upper on right {trace(h)=p*time()/1[s];}",
    );
    let (document, plan) =
        resolve_source_document(&source, 1., CommonSolvePolicy::Linear(linear())).unwrap();
    let plan = replay_plan(plan);
    let spatial = plan.as_spatial().unwrap();
    let initial = spatial.initial_state(Some(0.), Vec::new()).unwrap();
    let request =
        CommonTransientRunRequest::from_steps(plan.clone(), initial, 2, vec![1, 2]).unwrap();
    let std::ops::ControlFlow::Continue(steps) = request
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("unexpected stop")
    };
    let parameter = document.parameter_ref("p").unwrap().id();
    let field = document.field_ref("h").unwrap().id().erase();
    for (index, (_, step)) in steps.iter().enumerate() {
        let before = step.state().to_bytes().unwrap();
        let product = spatial
            .linearize_step(step, &[parameter], &[field])
            .unwrap();
        // Two Q1 cells: M_ii=1/3, sum M_if=1/6, K_ii=4, sum K_if=-4.
        // Fixed-history interior derivative is t*(4-1/6)/(4+1/3)=23*t/26.
        // At step two the preceding boundary and interior sensitivities stay zero.
        let time = (index + 1) as f64;
        check_products(&product, spatial.linear(), &[time, 23. * time / 26., time]);
        let changed = spatial
            .differentiate_step(
                step,
                &[parameter],
                Some(&[3.]),
                &[field],
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
        check_products(
            changed.linearization(),
            spatial.linear(),
            &[time, 23. * time / 26., time],
        );
        let reopened = spatial
            .validate_differentiated_step_values(
                step,
                &[parameter],
                Some(&[3.]),
                &[field],
                changed.linearization().relation().accepted_unknowns(),
            )
            .unwrap();
        check_products(&reopened, spatial.linear(), &[time, 23. * time / 26., time]);
        // Independent mass/stiffness recurrence with the p=2 predecessor held:
        // y_1=23*3/26; y_2=((1/3)*(23/13)+(1/6)*2+(23/6)*6)/(13/3).
        let expected = [
            3. * time,
            if index == 0 { 69. / 26. } else { 933. / 169. },
            3. * time,
        ];
        for (actual, expected) in changed.linearization().output_values().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
        assert_eq!(step.state().to_bytes().unwrap(), before);
    }
    let (foreign_document, foreign) = resolve_source_document(
        &source.replace("p:1=2", "p:1=3"),
        1.,
        CommonSolvePolicy::Linear(linear()),
    )
    .unwrap();
    let error = foreign
        .as_spatial()
        .unwrap()
        .linearize_step(
            &steps[0].1,
            &[foreign_document.parameter_ref("p").unwrap().id()],
            &[foreign_document.field_ref("h").unwrap().id().erase()],
        )
        .unwrap_err();
    assert!(error.message().contains("checkpoint"), "{error:?}");
}

#[test]
fn nonlinear_accepted_step_products_use_the_original_residual_jacobian() {
    let source = source("1[s]*derivative(h)+h^3=p;", "");
    let (document, plan) = resolve_source_document(&source, 1., newton(20)).unwrap();
    let spatial = plan.as_spatial().unwrap();
    let initial = spatial.initial_state(Some(0.), Vec::new()).unwrap();
    let request = CommonTransientRunRequest::from_steps(plan.clone(), initial, 1, vec![1]).unwrap();
    let std::ops::ControlFlow::Continue(steps) = request
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("unexpected stop")
    };
    let product = spatial
        .linearize_step(
            &steps[0].1,
            &[document.parameter_ref("p").unwrap().id()],
            &[document.field_ref("h").unwrap().id().erase()],
        )
        .unwrap();
    // h+h^3=p=2 gives h=1 and dh/dp=1/(1+3h^2)=1/4.
    for value in product.output_values() {
        assert!((value - 1.).abs() < 1e-10);
    }
    check_products(&product, spatial.linear(), &[0.25; 3]);
    let before = steps[0].1.state().to_bytes().unwrap();
    let parameter = document.parameter_ref("p").unwrap().id();
    let field = document.field_ref("h").unwrap().id().erase();
    let changed = spatial
        .differentiate_step(
            &steps[0].1,
            &[parameter],
            Some(&[10.]),
            &[field],
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
    // h+h^3=10 gives h=2, so dh/dp=1/13 at the reexecuted point.
    for value in changed.linearization().output_values() {
        assert!((value - 2.).abs() < 1e-10);
    }
    check_products(changed.linearization(), spatial.linear(), &[1. / 13.; 3]);
    let reopened = spatial
        .validate_differentiated_step_values(
            &steps[0].1,
            &[parameter],
            Some(&[10.]),
            &[field],
            changed.linearization().relation().accepted_unknowns(),
        )
        .unwrap();
    check_products(&reopened, spatial.linear(), &[1. / 13.; 3]);
    reopened
        .validate_nonlinear_update_count(changed.nonlinear_iterations().unwrap())
        .unwrap();
    assert!(reopened.validate_nonlinear_update_count(0).is_err());
    let error = spatial
        .validate_differentiated_step_values(
            &steps[0].1,
            &[parameter],
            Some(&[10.]),
            &[field],
            &[1.; 3],
        )
        .unwrap_err();
    assert!(error.message().contains("residual"), "{error:?}");
    assert!(changed.nonlinear_iterations().unwrap() > 0);
    assert!(!changed.linear_solves().is_empty());
    assert!(
        spatial
            .differentiate_step(
                &steps[0].1,
                &[parameter],
                Some(&[f64::NAN]),
                &[field],
                &REFERENCE_LINEAR_SOLVER,
            )
            .is_err()
    );
    assert_eq!(steps[0].1.state().to_bytes().unwrap(), before);
}
