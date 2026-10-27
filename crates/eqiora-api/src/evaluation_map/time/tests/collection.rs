use super::*;
use crate::{EvaluationCollection, RecordedEvaluationOutcome};

fn reopen(collection: &EvaluationCollection) -> EvaluationCollection {
    let bytes = collection.to_bytes().unwrap();
    let reopened = EvaluationCollection::from_bytes(
        &bytes,
        &eqiora_solver::REFERENCE_LINEAR_SOLVER,
        &[eqiora_time::ImplicitMidpointTimeBackend::CAPABILITIES],
    )
    .unwrap();
    assert_eq!(reopened.to_bytes().unwrap(), bytes);
    assert_eq!(&reopened, collection);
    reopened
}

#[test]
fn complete_time_collection_preserves_repeated_samples_and_original_checkpoint_references() {
    let (plan, requests) = fixture();
    let checkpoint = advance(&requests[0], None, 2)
        .unwrap()
        .break_value()
        .unwrap();
    let checkpoints = [Some(checkpoint.clone()), None, None, None];
    for retention in [
        EvaluationMapRetention::Retain,
        EvaluationMapRetention::Recompute,
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let called = calls.clone();
        let map = EvaluationMapPlan::from_time_requests(
            plan.clone(),
            &requests,
            Some(&checkpoints),
            policy(2, 1, retention),
            MEMBER_LIMIT,
            move |request, checkpoint| {
                called.fetch_add(1, Ordering::SeqCst);
                run(request, checkpoint)
            },
        )
        .unwrap();
        let collection = map.execute_recorded(|| false).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        let restored = reopen(&collection);
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        assert!(restored.is_complete());
        let group = &restored.groups()[0];
        assert_eq!(group.time_requests().unwrap(), requests);
        let recorded = group.time_checkpoints().unwrap()[0].as_ref().unwrap();
        assert_eq!(recorded.as_bytes(), checkpoint.to_bytes().unwrap());
        assert_eq!(recorded.restore().unwrap(), checkpoint);
        let (first_id, RecordedEvaluationOutcome::RunAccepted(first)) =
            group.occurrence(0).unwrap()
        else {
            panic!("accepted time Result")
        };
        let (repeat_id, RecordedEvaluationOutcome::RunAccepted(repeated)) =
            group.occurrence(2).unwrap()
        else {
            panic!("accepted time Result")
        };
        assert_ne!(first_id, repeat_id);
        assert!(Arc::ptr_eq(first, repeated));
        assert_eq!(requests[0].sampling(), requests[2].sampling());
    }
}

#[test]
fn interrupted_time_collection_retains_individual_results_and_requires_explicit_restore() {
    let (plan, requests) = fixture();
    let calls = Arc::new(AtomicUsize::new(0));
    let called = calls.clone();
    let map = EvaluationMapPlan::from_time_requests(
        plan,
        &requests,
        None,
        policy(2, 1, EvaluationMapRetention::Recompute),
        MEMBER_LIMIT,
        move |request, checkpoint| {
            let index = called.fetch_add(1, Ordering::SeqCst);
            advance(request, checkpoint, if index == 1 { 2 } else { usize::MAX })
        },
    )
    .unwrap();
    let saved = map.execute_recorded(|| false).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let restored = reopen(&saved);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(!restored.is_complete());
    let group = &restored.groups()[0];
    assert!(matches!(
        group.occurrence(0).unwrap().1,
        RecordedEvaluationOutcome::RunAccepted(_)
    ));
    assert!(matches!(
        group.occurrence(2).unwrap().1,
        RecordedEvaluationOutcome::NotStarted
    ));
    let (_, RecordedEvaluationOutcome::Cancelled(Some(record))) = group.occurrence(1).unwrap()
    else {
        panic!("recorded checkpoint")
    };
    assert_eq!(record.request(), &requests[1]);
    let checkpoint = record.restore().unwrap();
    assert_eq!(checkpoint.accepted_steps(), 2);
    let resumed = run(record.request(), Some(&checkpoint))
        .unwrap()
        .continue_value()
        .unwrap();
    let baseline = run(&requests[1], None).unwrap().continue_value().unwrap();
    assert_eq!(resumed.to_bytes().unwrap(), baseline.to_bytes().unwrap());
    let mut false_complete: serde_json::Value =
        serde_json::from_slice(&saved.to_bytes().unwrap()).unwrap();
    false_complete["complete"] = serde_json::json!(true);
    let error = EvaluationCollection::from_bytes(
        &serde_json::to_vec(&false_complete).unwrap(),
        &eqiora_solver::REFERENCE_LINEAR_SOLVER,
        &[eqiora_time::ImplicitMidpointTimeBackend::CAPABILITIES],
    )
    .unwrap_err();
    assert!(
        error
            .iter()
            .any(|error| error.message().contains("completion label"))
    );
}

#[test]
fn saved_time_estimator_binds_every_occurrence_without_reducing_on_read() {
    use crate::{MomentEstimator, MomentReduction};
    use eqiora_artifact::ReplayableCanonicalModelArtifact;
    use eqiora_schema::kernel::{KernelNode, ObservableReduction, ProbabilityMoment};
    let (plan, requests) = fixture();
    let model = plan.model_artifact().replay_model().unwrap();
    let output = model
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Observable(value) if value.reduction() == ObservableReduction::Value => {
                Some(value.id())
            }
            _ => None,
        })
        .unwrap();
    let output = crate::ModelEntityRef::from_model_artifact(plan.model_artifact(), output).unwrap();
    let map = EvaluationMapPlan::from_time_requests(
        plan,
        &requests,
        None,
        policy(1, 1, EvaluationMapRetention::Retain),
        MEMBER_LIMIT,
        run,
    )
    .unwrap();
    let complete = map.execute().unwrap();
    let results: Arc<[_]> = (0..complete.len())
        .map(|i| complete.run_result(i).unwrap().clone())
        .collect();
    let reduction = MomentReduction::Empirical(ProbabilityMoment::Expectation);
    let estimator = MomentEstimator::from_time_results(
        results.clone(),
        1.,
        reduction,
        std::slice::from_ref(&output),
        vec![1., 2., 0., 3.],
    )
    .unwrap();
    let saved = complete
        .record()
        .unwrap()
        .with_estimator(0, &estimator)
        .unwrap();
    let restored = crate::moment_estimator::without_reduction(|| reopen(&saved));
    let record = &restored.groups()[0].estimators()[0];
    assert_eq!(record.value(), estimator.value());
    assert_eq!(record.reduction(), reduction);
    assert_eq!(record.weights(), [1., 2., 0., 3.]);
    assert_eq!(
        record.time_observation(),
        Some((1., [output.id()].as_slice()))
    );
    assert_eq!(record.occurrences().len(), 4);
    assert_eq!(record.references()[0], record.references()[2]);
    assert_ne!(record.occurrences()[0], record.occurrences()[2]);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::moment_estimator::without_reduction(|| {
                MomentEstimator::from_time_results(
                    results.clone(),
                    1.,
                    reduction,
                    &[output],
                    vec![1.; 4],
                )
                .unwrap()
            })
        }))
        .is_err()
    );
    let mut reversed = results.to_vec();
    reversed.swap(0, 1);
    let selection = estimator.time_observation().unwrap().1;
    let foreign =
        MomentEstimator::from_time_results(reversed.into(), 1., reduction, selection, vec![1.; 4])
            .unwrap();
    assert!(
        complete
            .record()
            .unwrap()
            .with_estimator(0, &foreign)
            .unwrap_err()
            .message()
            .contains("ordered membership")
    );
    let partial = map
        .execute_with_cancellation(|| true)
        .unwrap_err()
        .record()
        .unwrap();
    assert!(
        partial
            .with_estimator(0, &estimator)
            .unwrap_err()
            .message()
            .contains("complete execution group")
    );

    let wire: serde_json::Value = serde_json::from_slice(&saved.to_bytes().unwrap()).unwrap();
    let mut mutants = Vec::new();
    let mut missing = wire.clone();
    missing["groups"][0]["estimators"][0]["content"]["occurrences"]
        .as_array_mut()
        .unwrap()
        .pop();
    mutants.push((missing, "attribution"));
    let mut foreign = wire.clone();
    foreign["groups"][0]["estimators"][0]["content"]["references"][0] =
        serde_json::json!("foreign");
    mutants.push((foreign, "attribution"));
    let mut time = wire.clone();
    time["groups"][0]["estimators"][0]["content"]["selection"]["time_s"] = serde_json::json!(0.75);
    mutants.push((time, "retained time"));
    let mut iid = wire.clone();
    iid["groups"][0]["estimators"][0]["content"]["reduction"]["kind"] =
        serde_json::json!("unbiased-iid");
    iid["groups"][0]["estimators"][0]["content"]["weights"] = serde_json::json!([1., 1., 1., 1.]);
    mutants.push((iid, "distinct uncoupled"));
    for (mut mutant, message) in mutants {
        let estimate = &mut mutant["groups"][0]["estimators"][0];
        let typed: crate::moment_estimator::record::artifact::WireEstimator =
            serde_json::from_value(estimate.clone()).unwrap();
        estimate["identity"] = serde_json::json!(typed.content_identity().unwrap());
        let errors = crate::moment_estimator::without_reduction(|| {
            EvaluationCollection::from_bytes(
                &serde_json::to_vec(&mutant).unwrap(),
                &eqiora_solver::REFERENCE_LINEAR_SOLVER,
                &[eqiora_time::ImplicitMidpointTimeBackend::CAPABILITIES],
            )
        })
        .unwrap_err();
        assert!(
            errors.iter().any(|error| error.message().contains(message)),
            "{errors:?}"
        );
    }
}

#[test]
fn time_collection_has_one_canonical_shared_plan_and_rejects_member_plan_payloads() {
    use base64::Engine as _;
    let (plan, requests) = fixture();
    let collection = EvaluationMapPlan::from_time_requests(
        plan,
        &requests,
        None,
        policy(1, 1, EvaluationMapRetention::Retain),
        MEMBER_LIMIT,
        run,
    )
    .unwrap()
    .execute_recorded(|| false)
    .unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&collection.to_bytes().unwrap()).unwrap();
    assert!(
        wire["groups"][0]["members"]
            .as_array()
            .unwrap()
            .iter()
            .all(|member| member["outcome"]["plan"].is_null())
    );
    for index in [0, 2] {
        let mut bad = wire.clone();
        bad["groups"][0]["members"][index]["outcome"]["plan"] = serde_json::json!(
            base64::engine::general_purpose::STANDARD
                .encode(collection.groups()[0].plan().to_bytes().unwrap())
        );
        let errors = EvaluationCollection::from_bytes(
            &serde_json::to_vec(&bad).unwrap(),
            &eqiora_solver::REFERENCE_LINEAR_SOLVER,
            &[eqiora_time::ImplicitMidpointTimeBackend::CAPABILITIES],
        )
        .unwrap_err();
        assert!(errors[0].message().contains("Plan"), "{errors:?}");
    }
}
