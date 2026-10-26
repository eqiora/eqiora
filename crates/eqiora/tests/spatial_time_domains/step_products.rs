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

#[derive(Debug, Default)]
struct CountSolves(std::sync::atomic::AtomicUsize);

impl eqiora::solver::LinearSolverBackend for CountSolves {
    fn provider(&self) -> eqiora::solver::SolverProvider {
        REFERENCE_LINEAR_SOLVER.provider()
    }
    fn capabilities(&self) -> eqiora::solver::SolverCapabilities {
        REFERENCE_LINEAR_SOLVER.capabilities()
    }
    fn solve_with_execution(
        &self,
        problem: &eqiora::solver::LinearProblem<'_>,
        policy: SolverPlan,
        execution: &dyn eqiora::solver::ReplicatedLinearExecution,
    ) -> Result<eqiora::solver::LinearSolution, eqiora::Diagnostic> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        REFERENCE_LINEAR_SOLVER.solve_with_execution(problem, policy, execution)
    }
}

#[test]
fn step_request_products_need_no_previously_solved_canonical_step() {
    let source = source("1[s]*derivative(h)+h=p;", "");
    let (document, plan) =
        resolve_source_document(&source, 1., CommonSolvePolicy::Linear(linear())).unwrap();
    let spatial = plan.as_spatial().unwrap();
    let initial = spatial.initial_state(Some(0.), Vec::new()).unwrap();
    let before = initial.to_bytes().unwrap();
    let request =
        CommonTransientRunRequest::from_steps(plan.clone(), initial.clone(), 1, vec![1]).unwrap();
    let parameter = document.parameter_ref("p").unwrap().id();
    let field = document.field_ref("h").unwrap().id().erase();
    let backend = CountSolves::default();
    let point = spatial
        .differentiate_step(
            &request,
            &[parameter],
            Some(&[3.]),
            &[field],
            &Default::default(),
            &backend,
        )
        .unwrap();
    // h_next=(h_previous+p)/2 with h_previous=0: h=3/2, dh/dp=1/2.
    for value in point.linearization().output_values() {
        assert!((value - 1.5).abs() < 1e-11);
    }
    assert_eq!(backend.0.load(std::sync::atomic::Ordering::Relaxed), 1);
    check_products(point.linearization(), spatial.linear(), &[0.5; 3]);
    let restored = spatial
        .validate_differentiated_step_values(
            &request,
            &[parameter],
            Some(&[3.]),
            &[field],
            &Default::default(),
            point.linearization().relation().accepted_unknowns(),
        )
        .unwrap();
    assert_eq!(&restored, point.linearization());
    let longer =
        CommonTransientRunRequest::from_steps(plan.clone(), initial.clone(), 2, vec![2]).unwrap();
    let error = spatial
        .differentiate_step(
            &longer,
            &[parameter],
            None,
            &[field],
            &Default::default(),
            &backend,
        )
        .unwrap_err();
    assert!(
        error.message().contains("exactly one requested step"),
        "{error:?}"
    );
    let foreign = resolve_source(
        &source.replace("p:1=2", "p:1=4"),
        1.,
        CommonSolvePolicy::Linear(linear()),
    )
    .unwrap();
    let foreign_initial = foreign
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
        .unwrap();
    let foreign_request =
        CommonTransientRunRequest::from_steps(foreign, foreign_initial, 1, vec![1]).unwrap();
    let error = spatial
        .differentiate_step(
            &foreign_request,
            &[parameter],
            None,
            &[field],
            &Default::default(),
            &backend,
        )
        .unwrap_err();
    assert!(error.message().contains("foreign Plan"), "{error:?}");
    assert_eq!(backend.0.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(initial.to_bytes().unwrap(), before);
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
    let mut products = Vec::new();
    for (index, (_, step)) in steps.iter().enumerate() {
        let step_request = CommonTransientRunRequest::from_steps(
            plan.clone(),
            step.previous_state().clone(),
            1,
            vec![1],
        )
        .unwrap();
        let before = step.state().to_bytes().unwrap();
        let product = spatial
            .linearize_step(step, &[parameter], &[field], &Default::default())
            .unwrap();
        // Two Q1 cells: M_ii=1/3, sum M_if=1/6, K_ii=4, sum K_if=-4.
        // Fixed-history interior derivative is t*(4-1/6)/(4+1/3)=23*t/26.
        // At step two the preceding boundary and interior sensitivities stay zero.
        let time = (index + 1) as f64;
        check_products(&product, spatial.linear(), &[time, 23. * time / 26., time]);
        let changed = spatial
            .differentiate_step(
                &step_request,
                &[parameter],
                Some(&[3.]),
                &[field],
                &Default::default(),
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
                &step_request,
                &[parameter],
                Some(&[3.]),
                &[field],
                &Default::default(),
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
        products.push(product);
    }
    // Full trajectory: (13/3)y_n=(1/3)y_prev+(1/6)g_prev+(23/6)g_n,
    // g_n=n*p. Thus d_0=0, d_1=23/26, d_2=317/169.
    // The initial physical state is independent even at prescribed nodes.
    let relations = products
        .iter()
        .map(|point| point.discrete_step().unwrap())
        .collect::<Vec<_>>();
    let accepted = relations
        .iter()
        .zip(&products)
        .map(|(relation, point)| {
            eqiora::differentiation::AcceptedOutputLinearization::new_with_canonical_state_jacobian(
                relation,
                relation,
                point.relation().state_jacobian(),
                point.state_scalar_domain(),
                point.residual_target(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    // Independent partial-action control: keep y1 and y2 independent here.
    // R1=(13/3)y1-(23/6)p; R2=(13/3)y2-(1/3)y1-(47/6)p.
    // O=[p,y1,p,2p,y2,2p]. These actions must perform no implicit solve.
    {
        use eqiora::differentiation::DiscreteTrajectoryLinearization;
        use eqiora::ir::{
            LinearizedOutput, LinearizedRelation, RelationCotangent, RelationTangent,
        };
        let trajectory = DiscreteTrajectoryLinearization::new(&accepted, &[]).unwrap();
        let mut residual = [0.; 2];
        LinearizedRelation::jvp(
            &trajectory,
            RelationTangent::Both {
                unknown: &[2., 3.],
                parameter: &[1.],
            },
            &mut residual,
        )
        .unwrap();
        for (actual, expected) in residual.into_iter().zip([29. / 6., 9. / 2.]) {
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }
        let mut unknown = [0.; 2];
        let mut parameter = [0.];
        LinearizedRelation::vjp(
            &trajectory,
            &[5., -2.],
            RelationCotangent::Both {
                unknown: &mut unknown,
                parameter: &mut parameter,
            },
        )
        .unwrap();
        for (actual, expected) in
            unknown
                .into_iter()
                .chain(parameter)
                .zip([67. / 3., -26. / 3., -7. / 2.])
        {
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }
        let mut output = [0.; 6];
        LinearizedOutput::jvp(&trajectory, &[2., 3.], &[1.], &mut output).unwrap();
        assert_eq!(output, [1., 2., 1., 2., 3., 2.]);
        LinearizedOutput::vjp(
            &trajectory,
            &[1., 2., 3., 4., 5., 6.],
            &mut unknown,
            &mut parameter,
        )
        .unwrap();
        assert_eq!(unknown, [2., 5.]);
        assert_eq!(parameter, [24.]);
    }
    for (coordinate, expected) in [2., 317. / 169., 2.].into_iter().enumerate() {
        let mut terminal = vec![0.; 3];
        terminal[coordinate] = 1.;
        let derivative = eqiora::differentiation::discrete_trajectory_adjoint(
            &accepted,
            &[],
            &[&[0.; 3], &terminal],
            &[0.],
            products[0].relation().state_jacobian().properties(),
            LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, spatial.linear()),
        )
        .unwrap();
        assert!((derivative.parameter_gradient()[0] - expected).abs() < 1e-10);
        let initial = if coordinate == 1 {
            [1. / 676., 1. / 169., 1. / 676.]
        } else {
            [0.; 3]
        };
        for (actual, expected) in derivative.initial_state_cotangent().iter().zip(initial) {
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }
        assert_eq!(derivative.step_adjoints().len(), 2);
    }
    let properties = products[0].relation().state_jacobian().properties();
    let solver = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, spatial.linear());
    let tangent = eqiora::differentiation::discrete_trajectory_sensitivity(
        &accepted,
        &[],
        &[0.; 3],
        &[1.],
        properties,
        solver,
    )
    .unwrap();
    for (step, expected) in tangent
        .iter()
        .zip([[1., 23. / 26., 1.], [2., 317. / 169., 2.]])
    {
        for (actual, expected) in step.output_tangent().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
        assert_eq!(
            step.state().report().orientation(),
            eqiora::solver::LinearOperatorOrientation::Normal
        );
    }
    // J=3*y_1-2*y_2+p/2. Both time samples contribute independently.
    let mixed = eqiora::differentiation::discrete_trajectory_adjoint(
        &accepted,
        &[],
        &[&[0., 3., 0.], &[0., -2., 0.]],
        &[0.5],
        properties,
        solver,
    )
    .unwrap();
    assert!(
        (mixed.parameter_gradient()[0] - (3. * 23. / 26. - 2. * 317. / 169. + 0.5)).abs() < 1e-10
    );
    let initial = [
        3. / 52. - 2. / 676.,
        3. / 13. - 2. / 169.,
        3. / 52. - 2. / 676.,
    ];
    for (actual, expected) in mixed.initial_state_cotangent().iter().zip(initial) {
        assert!((actual - expected).abs() < 1e-10);
    }
    let seeded = eqiora::differentiation::discrete_trajectory_sensitivity(
        &accepted,
        &[],
        &[1., 2., -1.],
        &[0.],
        properties,
        solver,
    )
    .unwrap();
    for (step, expected) in seeded.iter().zip([2. / 13., 2. / 169.]) {
        assert!((step.output_tangent()[1] - expected).abs() < 1e-10);
        assert_eq!(step.output_tangent()[0], 0.);
        assert_eq!(step.output_tangent()[2], 0.);
    }
    assert!(
        eqiora::differentiation::discrete_trajectory_adjoint(
            &accepted,
            &[],
            &[&[0.; 3]],
            &[0.],
            properties,
            solver,
        )
        .is_err()
    );
    assert!(
        eqiora::differentiation::discrete_trajectory_sensitivity(
            &accepted,
            &[],
            &[0., f64::NAN, 0.],
            &[1.],
            properties,
            solver,
        )
        .is_err()
    );
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
            &Default::default(),
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
            &Default::default(),
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
            &request,
            &[parameter],
            Some(&[10.]),
            &[field],
            &Default::default(),
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
            &request,
            &[parameter],
            Some(&[10.]),
            &[field],
            &Default::default(),
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
            &request,
            &[parameter],
            Some(&[10.]),
            &[field],
            &Default::default(),
            &[1.; 3],
        )
        .unwrap_err();
    assert!(error.message().contains("residual"), "{error:?}");
    assert!(changed.nonlinear_iterations().unwrap() > 0);
    assert!(!changed.linear_solves().is_empty());
    assert!(
        spatial
            .differentiate_step(
                &request,
                &[parameter],
                Some(&[f64::NAN]),
                &[field],
                &Default::default(),
                &REFERENCE_LINEAR_SOLVER,
            )
            .is_err()
    );
    assert_eq!(steps[0].1.state().to_bytes().unwrap(), before);
}

#[test]
fn changed_parameter_trajectory_owns_history_and_actual_nonlinear_solves() {
    // At p=10: h1+h1^3=10 -> h1=2;
    // h2-h1+h2^3=(33/5)*10 -> h2=4.
    let source = source(
        "1[s]*derivative(h)+h^3=p*(if time()<1.5[s] then 1 else 33/5);",
        "",
    );
    let (document, plan) = resolve_source_document(&source, 1., newton(20)).unwrap();
    let spatial = plan.as_spatial().unwrap();
    let initial = spatial.initial_state(Some(0.), Vec::new()).unwrap();
    let before = initial.to_bytes().unwrap();
    let request =
        CommonTransientRunRequest::from_steps(plan.clone(), initial.clone(), 2, vec![2]).unwrap();
    let parameter = document.parameter_ref("p").unwrap().id();
    let field = document.field_ref("h").unwrap().id().erase();
    let trajectory = spatial
        .differentiate_trajectory(
            &request,
            &[parameter],
            Some(&[10.]),
            &[field],
            &Default::default(),
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
    assert_eq!(trajectory.request().output_steps().unwrap(), vec![2]);
    assert_eq!(trajectory.steps().len(), 2);
    for (index, point) in trajectory.steps().iter().enumerate() {
        for value in point.linearization().output_values() {
            assert!((value - (index + 1) as f64 * 2.).abs() < 1e-10);
        }
        assert!(point.nonlinear_iterations().unwrap() > 0);
        assert!(!point.linear_solves().is_empty());
        assert_eq!(point.linearization().relation().design_values(), &[10.]);
        assert_eq!(point.nonlinear_initial_state().is_some(), index == 0);
        point
            .linearization()
            .validate_nonlinear_update_count(point.nonlinear_iterations().unwrap())
            .unwrap();
    }
    let candidates = trajectory
        .steps()
        .iter()
        .map(|p| p.linearization().relation().accepted_unknowns().to_vec())
        .collect::<Vec<_>>();
    let replay = spatial
        .validate_differentiated_trajectory_values(
            &request,
            &[parameter],
            Some(&[10.]),
            &[field],
            &Default::default(),
            &candidates,
        )
        .unwrap();
    for (reopened, point) in replay.iter().zip(trajectory.steps()) {
        assert_eq!(reopened, point.linearization());
    }
    assert!(
        spatial
            .validate_differentiated_trajectory_values(
                &request,
                &[parameter],
                Some(&[10.]),
                &[field],
                &Default::default(),
                &candidates[..1],
            )
            .is_err()
    );
    assert!(
        spatial
            .validate_differentiated_trajectory_values(
                &request,
                &[parameter],
                Some(&[2.]),
                &[field],
                &Default::default(),
                &candidates,
            )
            .is_err()
    );
    let mut altered = candidates.clone();
    altered[1][0] += 0.125;
    let error = spatial
        .validate_differentiated_trajectory_values(
            &request,
            &[parameter],
            Some(&[10.]),
            &[field],
            &Default::default(),
            &altered,
        )
        .unwrap_err();
    assert!(error.message().contains("residual"), "{error:?}");
    let views = trajectory
        .steps()
        .iter()
        .map(|p| p.linearization().discrete_step().unwrap())
        .collect::<Vec<_>>();
    let accepted = views
        .iter()
        .zip(trajectory.steps())
        .map(|(view, point)| {
            let point = point.linearization();
            eqiora::differentiation::AcceptedOutputLinearization::new_with_canonical_state_jacobian(
                view,
                view,
                point.relation().state_jacobian(),
                point.state_scalar_domain(),
                point.residual_target(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let properties = trajectory.steps()[0]
        .linearization()
        .relation()
        .state_jacobian()
        .properties();
    let solver = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, spatial.linear());
    let tangent = eqiora::differentiation::discrete_trajectory_sensitivity(
        &accepted,
        &[],
        &[0.; 3],
        &[1.],
        properties,
        solver,
    )
    .unwrap();
    // d1=1/13, d2=(d1+33/5)/49=62/455, independently of Newton updates.
    for (step, expected) in tangent.iter().zip([1. / 13., 62. / 455.]) {
        for actual in step.output_tangent() {
            assert!((actual - expected).abs() < 1e-10);
        }
    }
    let adjoint = eqiora::differentiation::discrete_trajectory_adjoint(
        &accepted,
        &[],
        &[&[0.; 3], &[1.; 3]],
        &[0.],
        properties,
        solver,
    )
    .unwrap();
    assert!((adjoint.parameter_gradient()[0] - 3. * 62. / 455.).abs() < 1e-10);
    assert_eq!(initial.to_bytes().unwrap(), before);
}

#[test]
fn trajectory_selected_outputs_keep_unobserved_history_and_direct_boundary_terms() {
    let source = "model Heat(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {
        parameter p:1=2;
        state h:1 on body in h1;
        state q:1 on body in h1;
        initial {h=0; q=0;}
        relation evolution on body {
            1[s/m^2]*derivative(h)-div(grad(h))=0;
        }
        relation relaxation on body {
            1[s]*derivative(q)+q=p;
        }
        relation lower on left {trace(h)=p*time()/1[s];}
        relation upper on right {trace(h)=p*time()/1[s];}
    }";
    let (document, plan) =
        resolve_source_document(source, 1., CommonSolvePolicy::Linear(linear())).unwrap();
    let spatial = plan.as_spatial().unwrap();
    let initial = spatial.initial_state(Some(0.), Vec::new()).unwrap();
    let request =
        CommonTransientRunRequest::from_steps(plan.clone(), initial, 2, vec![1, 2]).unwrap();
    let parameter = document.parameter_ref("p").unwrap().id();
    let h = document.field_ref("h").unwrap().id().erase();
    let trajectory = spatial
        .differentiate_trajectory(
            &request,
            &[parameter],
            Some(&[3.]),
            &[h],
            &Default::default(),
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
    let relations = trajectory
        .steps()
        .iter()
        .map(|point| point.linearization().discrete_step().unwrap())
        .collect::<Vec<_>>();
    let outputs = relations
        .iter()
        .map(|step| step.selected_output())
        .collect::<Vec<_>>();
    let accepted = relations
        .iter()
        .zip(&outputs)
        .zip(trajectory.steps())
        .map(|((relation, output), point)| {
            let point = point.linearization();
            AcceptedOutputLinearization::new_with_canonical_state_jacobian(
                relation,
                output,
                point.relation().state_jacobian(),
                point.state_scalar_domain(),
                point.residual_target(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let states = eqiora::ir::DiscreteStepLinearization::state_coordinates(&relations[0]).len();
    assert_eq!(states, 6); // Both three-node Fields remain in history.
    let properties = trajectory.steps()[0]
        .linearization()
        .relation()
        .state_jacobian()
        .properties();
    let solver = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, spatial.linear());
    let forward = eqiora::differentiation::discrete_trajectory_sensitivity(
        &accepted,
        &[],
        &vec![0.; states],
        &[1.],
        properties,
        solver,
    )
    .unwrap();
    // Only h is selected. Its boundary terms are direct Parameter output
    // contributions; q still evolves but contributes nothing to this objective.
    for (step, expected) in forward
        .iter()
        .zip([[1., 23. / 26., 1.], [2., 317. / 169., 2.]])
    {
        assert_eq!(step.output_tangent().len(), 3);
        for (actual, expected) in step.output_tangent().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }
    let reverse = eqiora::differentiation::discrete_trajectory_adjoint(
        &accepted,
        &[],
        &[&[1.; 3], &[1.; 3]],
        &[0.],
        properties,
        solver,
    )
    .unwrap();
    assert!((reverse.parameter_gradient()[0] - (6. + 23. / 26. + 317. / 169.)).abs() < 1e-10);
}

#[test]
fn trajectory_observables_share_endpoint_time_quadrature_and_direct_partials() {
    let source = source(
        "1[s]*derivative(h)+h=p;",
        "observable amount:m=integral(h*time()/1[s],measure(body));
         observable total:1=amount/1[m]+p*time()/1[s];",
    );
    let (document, plan) =
        resolve_source_document(&source, 1., CommonSolvePolicy::Linear(linear())).unwrap();
    let spatial = plan.as_spatial().unwrap();
    let parameter = document.parameter_ref("p").unwrap().id();
    let total = document.observable_ref("total").unwrap().id();
    let field = document.field_ref("h").unwrap().id().erase();
    let outputs = [total.erase(), field];
    let rules = plan.observation_quadratures(&[total], 2).unwrap();
    let request = CommonTransientRunRequest::from_steps(
        plan.clone(),
        spatial.initial_state(Some(0.), Vec::new()).unwrap(),
        2,
        vec![2],
    )
    .unwrap();
    let trajectory = spatial
        .differentiate_trajectory(
            &request,
            &[parameter],
            Some(&[3.]),
            &outputs,
            &rules,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
    // h1=3/2, h2=9/4, O_n=t_n*(h_n+p). These products hold the
    // preceding physical State fixed; through-history products are separate.
    for (index, point) in trajectory.steps().iter().enumerate() {
        let time = (index + 1) as f64;
        let value = [1.5, 2.25][index];
        let expected = [time * (value + 3.), value, value, value];
        for (actual, expected) in point.linearization().output_values().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
        check_products(
            point.linearization(),
            spatial.linear(),
            &[1.5 * time, 0.5, 0.5, 0.5],
        );
    }
    let candidates = trajectory
        .steps()
        .iter()
        .map(|point| {
            point
                .linearization()
                .relation()
                .accepted_unknowns()
                .to_vec()
        })
        .collect::<Vec<_>>();
    let replay = spatial
        .validate_differentiated_trajectory_values(
            &request,
            &[parameter],
            Some(&[3.]),
            &outputs,
            &rules,
            &candidates,
        )
        .unwrap();
    assert!((replay[1].output_values()[0] - 10.5).abs() < 1e-10);
    assert!(
        spatial
            .differentiate_trajectory(
                &request,
                &[parameter],
                Some(&[3.]),
                &outputs,
                &Default::default(),
                &REFERENCE_LINEAR_SOLVER,
            )
            .is_err()
    );
}
