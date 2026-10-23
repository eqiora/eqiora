//! Explicit internal variables use the existing typed State and occurrence owners.
use super::*;

const SOURCE: &str = r#"
component Material(support body:volume(ambient_dimension=1),
    parameter initial_value:1, parameter amplitude:1) {
    coordinate x:m on body from body[0];
    state h:1 on body in h1;
    variable stress:Pa on body in h1;
    initial {
        h=initial_value+x/1[m];
        stress=2[Pa]*(amplitude-initial_value-x/1[m]);
    }
    relation update on body {
        1[s]*derivative(h)+h=amplitude*(if time()<1.5[s] then 1 else 0);
    }
    relation response on body {
        stress=2[Pa]*(amplitude*(if time()<1.5[s] then 1 else 0)-h);
    }
}
public component Heat(support body:volume(ambient_dimension=1),
    support left:boundary(parent=body),support right:boundary(parent=body)) {
    instance first:Material(body=body,initial_value=0,amplitude=1);
    instance second:Material(body=body,initial_value=4,amplitude=0);
}
"#;

fn check_fields(
    document: &ModelDocument,
    plan: &ResolvedCommonPlan,
    state: &CommonState,
    expected: Vec<Vec<f64>>,
) {
    let fields = plan
        .as_spatial()
        .unwrap()
        .state_fields(state)
        .unwrap()
        .collect::<Vec<_>>();
    let names = [
        "definition.first.h",
        "definition.first.stress",
        "definition.second.h",
        "definition.second.stress",
    ];
    assert_eq!(fields.len(), names.len());
    assert_eq!(expected.len(), names.len());
    for (name, expected) in names.into_iter().zip(expected) {
        let id = document.field_ref(name).unwrap().id();
        let actual = fields
            .iter()
            .find(|(field, _, _, _)| *field == id)
            .unwrap()
            .3;
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-11,
                "{name}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn independent_material_occurrences_reuse_accepted_state_without_trial_feedback() {
    for solve in [CommonSolvePolicy::Linear(linear()), newton(20)] {
        let (document, plan) = resolve_source_document(SOURCE, 1., solve).unwrap();
        let plan = replay_plan(plan);
        let initial = plan
            .as_spatial()
            .unwrap()
            .initial_state(Some(0.), Vec::new())
            .unwrap();
        let before = initial.to_bytes().unwrap();
        // At x=0,1/2,1, h_initial=offset+x. BE gives
        // h_next=(h_committed+forcing)/2 and stress=2*(forcing-h_next).
        let first =
            CommonTransientRunRequest::from_steps(plan.clone(), initial.clone(), 1, vec![1])
                .unwrap();
        let run = || {
            let std::ops::ControlFlow::Continue(steps) = first
                .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
                .unwrap()
            else {
                panic!("unexpected stop")
            };
            steps[0].1.state().clone()
        };
        let rejected = RejectCandidate::default();
        let Err(error) = first.advance_accepted_actions(&rejected, |_, _| false) else {
            panic!("the provider must reject this candidate")
        };
        assert!(
            error.message().contains("injected candidate failure"),
            "{error:?}"
        );
        assert!(rejected.calls.load(std::sync::atomic::Ordering::Relaxed) > 0);
        assert_eq!(initial.to_bytes().unwrap(), before);
        let accepted = run();
        assert_eq!(run(), accepted);
        assert_eq!(initial.to_bytes().unwrap(), before);
        check_fields(
            &document,
            &plan,
            &accepted,
            vec![
                vec![0.5, 0.75, 1.],
                vec![1., 0.5, 0.],
                vec![2., 2.25, 2.5],
                vec![-4., -4.5, -5.],
            ],
        );
        let restart = CommonState::from_bytes(&accepted.to_bytes().unwrap(), &plan).unwrap();
        let next =
            CommonTransientRunRequest::from_steps(plan.clone(), restart, 1, vec![1]).unwrap();
        let std::ops::ControlFlow::Continue(steps) = next
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("unexpected stop")
        };
        check_fields(
            &document,
            &plan,
            steps[0].1.state(),
            vec![
                vec![0.25, 0.375, 0.5],
                vec![-0.5, -0.75, -1.],
                vec![1., 1.125, 1.25],
                vec![-2., -2.25, -2.5],
            ],
        );
        let full =
            CommonTransientRunRequest::from_steps(plan.clone(), initial.clone(), 2, vec![1, 2])
                .unwrap();
        let std::ops::ControlFlow::Continue(uninterrupted) = full
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("unexpected stop")
        };
        assert_eq!(uninterrupted[0].1.state(), &accepted);
        assert_eq!(uninterrupted[1].1.state(), steps[0].1.state());
        let foreign = replay_plan(
            resolve_source(
                &SOURCE.replace("amplitude=1", "amplitude=2"),
                1.,
                CommonSolvePolicy::Linear(linear()),
            )
            .unwrap(),
        );
        assert!(CommonTransientRunRequest::from_steps(foreign, accepted, 1, vec![1]).is_err());
        assert_eq!(initial.to_bytes().unwrap(), before);
    }
}

#[derive(Debug, Default)]
struct RejectCandidate {
    calls: std::sync::atomic::AtomicUsize,
}
impl eqiora::solver::LinearSolverBackend for RejectCandidate {
    fn provider(&self) -> eqiora::solver::SolverProvider {
        REFERENCE_LINEAR_SOLVER.provider()
    }
    fn capabilities(&self) -> eqiora::solver::SolverCapabilities {
        REFERENCE_LINEAR_SOLVER.capabilities()
    }
    fn solve_with_execution(
        &self,
        _: &eqiora::solver::LinearProblem<'_>,
        _: SolverPlan,
        _: &dyn eqiora::solver::ReplicatedLinearExecution,
    ) -> Result<eqiora::solver::LinearSolution, eqiora::Diagnostic> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Err(eqiora::Diagnostic::error(
            eqiora::diagnostic::codes::INVALID_REALIZATION,
            "injected candidate failure",
        ))
    }
}
