//! Observe native RHS trials without changing the authored physical system.
use super::*;
use std::cell::RefCell;

struct Traced<'a> {
    system: &'a dyn TimeSystem,
    times: &'a RefCell<Vec<f64>>,
}

impl TimeSystem for Traced<'_> {
    fn dimension(&self) -> usize {
        self.system.dimension()
    }
    fn rhs(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        self.times.borrow_mut().push(time);
        self.system.rhs(time, state, output)
    }
    fn rhs_jvp(
        &self,
        time: f64,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.system.rhs_jvp(time, state, direction, output)
    }
}

fn execute(request: &CommonTimeRunRequest) -> (CommonTrajectory, Vec<f64>) {
    let times = RefCell::new(Vec::new());
    let token = Arc::new(AtomicBool::new(false));
    let trajectory = request
        .advance(None, |problem, roots, plan, saved, _| {
            assert!(roots.is_none());
            assert!(saved.is_none());
            let system = Traced {
                system: problem.system(),
                times: &times,
            };
            let problem = TimeProblem::new(
                &system,
                problem.equation_class(),
                InitialConditionPolicy::Provided,
                problem.initial_state().to_vec(),
            )?;
            let mut derivative = vec![0.; system.dimension()];
            system.rhs(plan.start_time(), problem.initial_state(), &mut derivative)?;
            // The native checkpoint explicitly chooses the Plan's oversized
            // initial trial, bypassing only the provider's initial-step heuristic.
            let initial = eqiora::time::TimeCheckpoint::initial(
                &problem,
                plan,
                DiffsolTimeBackend::IDENTITY,
                None,
                plan.initial_step(),
                derivative,
            )?;
            DiffsolTimeBackend::new().advance(&problem, plan, roots, Some(&initial), &token, None)
        })
        .unwrap()
        .continue_value()
        .unwrap();
    (trajectory, times.into_inner())
}

#[test]
fn rejected_trials_and_accepted_histories_remain_occurrence_local() {
    use eqiora::api::{EvaluationMapExecutionPolicy, EvaluationMapPlan, EvaluationMapRetention};
    use eqiora_numerics::CommonResult;

    let template = fixture(None, 1e-9, 1.);
    let plan = template.plan().clone();
    let preparation = CommonTimeRunRequest::new(
        plan.clone(),
        plan.initial_state(0.).unwrap(),
        0.5,
        vec![0.5],
    )
    .unwrap();
    let prepared = finish(&preparation, None).unwrap();
    let requests = [
        prepared.time_states().unwrap()[0].clone(),
        plan.initial_state(0.5).unwrap(),
    ]
    .into_iter()
    .map(|state| CommonTimeRunRequest::new(plan.clone(), state, 2., vec![0.75, 1., 2.]).unwrap())
    .collect::<Vec<_>>();
    let expected = requests.iter().map(execute).collect::<Vec<_>>();
    for (_, trace) in &expected {
        // Tsit45 stage times advance within a trial. Returning from the first
        // oversized trial to before t=1 proves rejection, with no events here.
        assert!(trace.windows(2).any(|pair| pair[0] >= 1.4 && pair[1] < 1.));
    }
    assert_ne!(expected[0].1, expected[1].1);
    assert_ne!(
        expected[0].0.ode_history().unwrap().steps(),
        expected[1].0.ode_history().unwrap().steps()
    );
    let expected = Arc::new(expected);
    let reference = expected.clone();
    let originals = requests.clone();
    let map = EvaluationMapPlan::from_time_requests(
        Arc::new(plan),
        &requests,
        None,
        EvaluationMapExecutionPolicy::new(
            2,
            eqiora::realization::Target::HostCpu {
                threads: std::num::NonZeroUsize::new(2).unwrap(),
            },
            EvaluationMapRetention::Retain,
            1 << 28,
        )
        .unwrap(),
        1 << 20,
        move |request, saved| {
            assert!(saved.is_none());
            let index = originals.iter().position(|value| value == request).unwrap();
            let (trajectory, trace) = execute(request);
            assert_eq!(trace, reference[index].1);
            Ok(ControlFlow::Continue(
                CommonResult::accept_trajectory(0., trajectory).unwrap(),
            ))
        },
    )
    .unwrap();
    let complete = map.execute().unwrap();
    for (index, (trajectory, _)) in expected.iter().enumerate() {
        assert_eq!(
            complete
                .time_result(index)
                .unwrap()
                .trajectory()
                .unwrap()
                .to_bytes()
                .unwrap(),
            trajectory.to_bytes().unwrap()
        );
    }
}
