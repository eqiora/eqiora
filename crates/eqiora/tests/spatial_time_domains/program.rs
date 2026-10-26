//! Run-bound Programs use ordinary products, records and mapped execution.
use super::*;
use crate::support::derivative_products::{FixtureCoordinates, cotangent, tangent};
use eqiora::api::{
    DifferentiableProgram, DifferentiableProgramIdentity, EvaluationCollection,
    EvaluationMapExecutionPolicy, EvaluationMapPlan, EvaluationMapProducts, EvaluationMapRetention,
    RecordedDifferentiableEvaluation, RunRequest,
};
use eqiora::solver::LinearSolverBackend;
use std::sync::Arc;

#[derive(Debug)]
struct NoSolve;
impl LinearSolverBackend for NoSolve {
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
        panic!("Run-bound collection reopening must not solve")
    }
}

fn close(values: &[f64], expected: &[f64]) {
    assert_eq!(values.len(), expected.len());
    for (actual, expected) in values.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }
}

#[test]
fn run_program_preserves_fixed_history_products_and_solve_free_records() {
    for nonlinear in [false, true] {
        let reaction = if nonlinear { "h^3" } else { "h" };
        let source = format!(
            "model Heat(support body:volume(ambient_dimension=1),
            support left:boundary(parent=body),support right:boundary(parent=body)) {{
                parameter p:1=2; state h:1 on body in h1; initial {{h=0;}}
                relation evolution on body {{1[s]*derivative(h)+{reaction}=p;}}
            }}"
        );
        let solve = if nonlinear {
            newton(20)
        } else {
            CommonSolvePolicy::Linear(linear())
        };
        let (document, plan) = resolve_source_document(&source, 1., solve).unwrap();
        let spatial = plan.as_spatial().unwrap();
        let zero_fields = || {
            vec![
                eqiora_numerics::CommonInitialField::new(
                    document.artifact_reference().unwrap().artifact().clone(),
                    spatial.fields().next().unwrap().0,
                    Some(
                        eqiora_numerics::CommonInitialValues::new(
                            eqiora::ValueShape::scalar(),
                            vec![0.; 3],
                        )
                        .unwrap(),
                    ),
                    None,
                )
                .unwrap(),
            ]
        };
        let initial = spatial.initial_state(Some(1.), zero_fields()).unwrap();
        let before = initial.to_bytes().unwrap();
        let request = RunRequest::from_steps(plan.clone(), initial.clone(), 1, vec![1]).unwrap();
        let parameter = document.parameter_ref("p").unwrap();
        let field = document.field_ref("h").unwrap().erase();
        let program = Arc::new(
            DifferentiableProgram::compile_run(
                request.clone(),
                std::slice::from_ref(&parameter),
                &[field],
                Default::default(),
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap(),
        );
        assert_eq!(
            program.identity().run_request_identity(),
            Some(request.identity())
        );
        assert_eq!(
            program.identity().initial_state_identity(),
            Some(initial.identity())
        );
        let identity_bytes = program.identity().to_bytes().unwrap();
        assert_eq!(
            DifferentiableProgramIdentity::from_run_bytes(&identity_bytes, &request).unwrap(),
            *program.identity()
        );
        assert!(
            DifferentiableProgramIdentity::from_bytes(
                &identity_bytes,
                &plan,
                Some(&(&initial).into())
            )
            .is_err()
        );
        // h_previous=0, dt=1: h+h^3=p or 2h=p. Choosing h=2 gives
        // p=10, dh/dp=1/13 or p=4, dh/dp=1/2, respectively.
        let p = if nonlinear { 10. } else { 4. };
        let derivative = if nonlinear { 1. / 13. } else { 0.5 };
        let point = crate::support::real_point(&program, &[p]).unwrap();
        let live = program.evaluate(&point).unwrap();
        close(&live.primal().output().fixture_coordinates(), &[2.; 3]);
        close(
            &live
                .jvp(&tangent(program.identity(), &[1.]).unwrap())
                .unwrap()
                .derivative()
                .fixture_coordinates(),
            &[derivative; 3],
        );
        close(
            &live
                .vjp(&cotangent(program.identity(), &[1.; 3]).unwrap())
                .unwrap()
                .derivative()
                .fixture_coordinates(),
            &[3. * derivative],
        );
        let record = live.record().unwrap();
        let bytes = record.to_bytes().unwrap();
        assert_eq!(
            RecordedDifferentiableEvaluation::from_run_bytes(&bytes, &request).unwrap(),
            record
        );
        assert!(
            RecordedDifferentiableEvaluation::from_bytes(&bytes, &plan, Some(&(&initial).into()))
                .is_err()
        );
        let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        forged["steps"][0]["unknown"][0] =
            serde_json::json!(forged["steps"][0]["unknown"][0].as_f64().unwrap() + 0.125);
        assert!(
            RecordedDifferentiableEvaluation::from_run_bytes(
                &serde_json::to_vec(&forged).unwrap(),
                &request
            )
            .is_err()
        );

        // An independently accepted predecessor at the same physical start time
        // changes request meaning even though the Plan and endpoint are unchanged.
        let warmup = CommonTransientRunRequest::from_steps(
            plan.clone(),
            spatial.initial_state(Some(0.), Vec::new()).unwrap(),
            1,
            vec![1],
        )
        .unwrap();
        let std::ops::ControlFlow::Continue(history) = warmup
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("unexpected stop")
        };
        let changed_history =
            RunRequest::from_steps(plan.clone(), history[0].1.state().clone(), 1, vec![1]).unwrap();
        assert_eq!(
            request.native().until_s(),
            changed_history.native().until_s()
        );
        assert_ne!(request.identity(), changed_history.identity());
        assert!(
            RecordedDifferentiableEvaluation::from_run_bytes(&bytes, &changed_history).is_err()
        );
        let changed_time = RunRequest::from_steps(
            plan.clone(),
            spatial.initial_state(Some(2.), zero_fields()).unwrap(),
            1,
            vec![1],
        )
        .unwrap();
        assert!(RecordedDifferentiableEvaluation::from_run_bytes(&bytes, &changed_time).is_err());

        for retention in [
            EvaluationMapRetention::Retain,
            EvaluationMapRetention::Recompute,
        ] {
            let policy = EvaluationMapExecutionPolicy::new(
                2,
                eqiora::realization::Target::HostCpu {
                    threads: NonZeroUsize::new(2).unwrap(),
                },
                retention,
                1 << 28,
            )
            .unwrap();
            let points = crate::support::real_points(&program, &[&[2.], &[p], &[p]]);
            let map = EvaluationMapPlan::new(program.clone(), &points, &[], policy).unwrap();
            let collection = map.execute_recorded(|| false).unwrap();
            let encoded = collection.to_bytes().unwrap();
            let reopened = EvaluationCollection::from_bytes(
                &encoded,
                &NoSolve,
                &[eqiora::time::ImplicitMidpointTimeBackend::CAPABILITIES],
            )
            .unwrap();
            assert_eq!(reopened, collection);
            assert_eq!(reopened.to_bytes().unwrap(), encoded);

            let shared_points = vec![point.clone(); 3];
            let shared =
                EvaluationMapPlan::new(program.clone(), &shared_points, &[parameter.id()], policy)
                    .unwrap()
                    .execute()
                    .unwrap();
            let products = EvaluationMapProducts::new(&shared, &[3], &[], &[0], 1 << 28).unwrap();
            let seed = cotangent(program.identity(), &[1.; 3]).unwrap();
            let reverse = products.vjp(&vec![seed; 3]).unwrap();
            close(
                &reverse.shared_cotangents()[0].fixture_coordinates(),
                &[9. * derivative],
            );
        }
        assert_eq!(initial.to_bytes().unwrap(), before);
    }
}

#[test]
fn multi_step_program_preserves_temporal_axes_all_acceptances_and_unsampled_history() {
    let source = "model Heat(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {
        parameter p:1=2; state h:1 on body in h1; state q:1 on body in h1;
        initial {h=0; q=0;}
        relation h_evolution on body {1[s]*derivative(h)+h=p;}
        relation q_evolution on body {1[s]*derivative(q)+q=2*p;}
    }";
    let (document, plan) =
        resolve_source_document(source, 1., CommonSolvePolicy::Linear(linear())).unwrap();
    let initial = plan
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
        .unwrap();
    let request = RunRequest::from_steps(plan.clone(), initial, 3, vec![1, 3]).unwrap();
    let program = Arc::new(
        DifferentiableProgram::compile_run(
            request.clone(),
            &[document.parameter_ref("p").unwrap()],
            &[
                document.field_ref("q").unwrap().erase(),
                document.field_ref("h").unwrap().erase(),
            ],
            Default::default(),
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap(),
    );
    for output in program.identity().outputs() {
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(output.sample_times_s(), Some(&[1., 3.][..]));
        assert_eq!(output.coefficient_entities().len(), 3);
        assert!(
            output
                .mathematical_type()
                .value_type
                .shape()
                .extents()
                .is_empty()
        );
    }
    assert_eq!(program.identity().inputs()[0].sample_times_s(), None);
    let live = program
        .evaluate(&crate::support::real_point(&program, &[4.]).unwrap())
        .unwrap();
    // Backward Euler: h_n=(h_(n-1)+p)/2, q_n=2*h_n. Observe steps 1 and 3.
    let expected = [4., 4., 4., 7., 7., 7., 2., 2., 2., 3.5, 3.5, 3.5];
    let derivative = [
        1., 1., 1., 1.75, 1.75, 1.75, 0.5, 0.5, 0.5, 0.875, 0.875, 0.875,
    ];
    close(&live.primal().output().fixture_coordinates(), &expected);
    let primal = live.primal();
    let q = primal.output().leaf(0).unwrap();
    assert_eq!(
        q.coefficient(0, Some(1)).unwrap().component(0).unwrap(),
        (7., 0.)
    );
    assert!(q.coefficient(0, None).is_err());
    assert!(q.coefficient(0, Some(2)).is_err());
    let forward = live
        .jvp(&tangent(program.identity(), &[1.]).unwrap())
        .unwrap();
    close(&forward.derivative().fixture_coordinates(), &derivative);
    let reverse = live
        .vjp(&cotangent(program.identity(), &[1.; 12]).unwrap())
        .unwrap();
    close(&reverse.derivative().fixture_coordinates(), &[12.375]);
    for result in [
        live.primal().evidence(),
        forward.evidence(),
        reverse.evidence(),
    ] {
        assert_eq!(result.steps().len(), 3);
        assert!(result.steps().iter().all(|step| {
            step.primal_residual_norm()
                <= step
                    .residual_tolerance()
                    .expect("backward-Euler residual criterion")
        }));
    }
    assert!(
        forward
            .evidence()
            .steps()
            .iter()
            .all(|step| step.derivative_solve().is_some())
    );
    assert_eq!(live.accepted_unknowns().len(), 18);
    let record = live.record().unwrap();
    let bytes = record.to_bytes().unwrap();
    assert_eq!(
        RecordedDifferentiableEvaluation::from_run_bytes(&bytes, &request).unwrap(),
        record
    );
    let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // The unobserved second step must still be reaccepted, not dropped.
    forged["steps"][1]["unknown"][0] =
        serde_json::json!(forged["steps"][1]["unknown"][0].as_f64().unwrap() + 0.125);
    assert!(
        RecordedDifferentiableEvaluation::from_run_bytes(
            &serde_json::to_vec(&forged).unwrap(),
            &request
        )
        .is_err()
    );
    let mut missing: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    missing["steps"].as_array_mut().unwrap().remove(1);
    assert!(
        RecordedDifferentiableEvaluation::from_run_bytes(
            &serde_json::to_vec(&missing).unwrap(),
            &request
        )
        .is_err()
    );
}

#[test]
fn nonlinear_trajectory_record_retains_each_actual_update_and_seed_boundary() {
    let source = "model Heat(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {
        parameter p:1=2; state h:1 on body in h1; initial {h=0;}
        relation evolution on body {1[s]*derivative(h)+h^3=p*(if time()<1.5[s] then 1 else 6.6);}
    }";
    let (document, plan) = resolve_source_document(source, 1., newton(30)).unwrap();
    let initial = plan
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
        .unwrap();
    let request = RunRequest::from_steps(plan, initial.clone(), 2, vec![2]).unwrap();
    let program = DifferentiableProgram::compile_run(
        request.clone(),
        &[document.parameter_ref("p").unwrap()],
        &[document.field_ref("h").unwrap().erase()],
        Default::default(),
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    let live = program
        .evaluate(&crate::support::real_point(&program, &[10.]).unwrap())
        .unwrap();
    close(&live.primal().output().fixture_coordinates(), &[4.; 3]);
    close(
        &live
            .jvp(&tangent(program.identity(), &[1.]).unwrap())
            .unwrap()
            .derivative()
            .fixture_coordinates(),
        &[62. / 455.; 3],
    );
    let primal = live.primal();
    assert_eq!(primal.evidence().steps().len(), 2);
    for step in primal.evidence().steps() {
        assert!(step.nonlinear_iterations(0).unwrap() > 0);
        assert_eq!(
            step.primal_solves(0).unwrap().len(),
            step.nonlinear_iterations(0).unwrap()
        );
    }
    assert_eq!(
        primal.evidence().steps()[0].nonlinear_initial_state_identity(0),
        Some(initial.identity())
    );
    assert_eq!(
        primal.evidence().steps()[1].nonlinear_initial_state_identity(0),
        None
    );
    let record = live.record().unwrap();
    assert_eq!(
        RecordedDifferentiableEvaluation::from_run_bytes(&record.to_bytes().unwrap(), &request)
            .unwrap(),
        record
    );
}
