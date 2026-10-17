use super::super::tests::{POISSON, WIENER, request};
use super::*;

fn recreated(run: &ProcessRun) -> Arc<ProcessRun> {
    ProcessRun::new(run.request.clone()).unwrap()
}

#[test]
fn checkpoint_and_rejected_trials_preserve_both_methods_exactly() {
    for (source, method) in [
        (WIENER, ProcessMethod::EulerMaruyama { depth: 8 }),
        (POISSON, ProcessMethod::PoissonEvents { max_events: 64 }),
    ] {
        let run = request(source, method);
        let mut uninterrupted = run.start();
        let mut restarting = run.start();
        assert_eq!(
            run.resume_bytes(&restarting.checkpoint_bytes().unwrap())
                .unwrap(),
            restarting
        );
        for elapsed in [0.25, 0.5, 0.75, 1.] {
            // Try the whole remaining interval, then discard it. Only the
            // agreed accepted partition is persisted or replayed.
            let before = restarting.checkpoint_bytes().unwrap();
            drop(restarting.trial(&exact(1.)).unwrap());
            assert_eq!(restarting.checkpoint_bytes().unwrap(), before);
            uninterrupted
                .accept(uninterrupted.trial(&exact(elapsed)).unwrap())
                .unwrap();
            restarting
                .accept(restarting.trial(&exact(elapsed)).unwrap())
                .unwrap();
            let bytes = restarting.checkpoint_bytes().unwrap();
            // Recreate the immutable Run so this proves persistence, not just
            // cloning an in-memory cursor or sharing its Arc allocation.
            restarting = recreated(&run).resume_bytes(&bytes).unwrap();
            assert_eq!(restarting, uninterrupted);
            assert_eq!(restarting.checkpoint_bytes().unwrap(), bytes);
        }
    }
}

#[test]
fn poisson_checkpoint_retains_pending_threshold_and_right_continuous_counter() {
    let run = request(POISSON, ProcessMethod::PoissonEvents { max_events: 64 });
    let RealizedNoise::Poisson(paths) = run.paths.values().next().unwrap() else {
        panic!("Poisson")
    };
    let arrivals = paths[0].arrivals().collect::<Vec<_>>();
    assert!(arrivals.len() > 1);
    let mut cursor = run.start();
    cursor.accept(cursor.trial(&arrivals[0]).unwrap()).unwrap();
    let pending = paths[0]
        .pending_threshold(&arrivals[0])
        .unwrap()
        .unwrap()
        .clone();
    let bytes = cursor.checkpoint_bytes().unwrap();
    let mut reopened = recreated(&run).resume_bytes(&bytes).unwrap();
    assert_eq!(reopened.state().values(), [2.]);
    let RealizedNoise::Poisson(retained) = reopened.run.paths.values().next().unwrap() else {
        panic!("Poisson")
    };
    assert_eq!(retained[0].count(&reopened.elapsed).unwrap(), 1);
    assert_eq!(
        retained[0].pending_threshold(&reopened.elapsed).unwrap(),
        Some(&pending)
    );
    reopened
        .accept(reopened.trial(&arrivals[1]).unwrap())
        .unwrap();
    assert_eq!(reopened.state().values(), [4.]);
    let accepted = reopened.checkpoint_bytes().unwrap();
    drop(reopened.trial(&exact(1.)).unwrap());
    assert_eq!(reopened.checkpoint_bytes().unwrap(), accepted);
}

fn saved(run: &Arc<ProcessRun>) -> Vec<u8> {
    let mut cursor = run.start();
    cursor.accept(cursor.trial(&exact(0.5)).unwrap()).unwrap();
    cursor.checkpoint_bytes().unwrap()
}

#[test]
fn reidentified_mutants_reach_the_physical_path_and_request_boundaries() {
    let run = request(WIENER, ProcessMethod::EulerMaruyama { depth: 8 });
    let bytes = saved(&run);
    let reject = |payload: &Payload, message: &str| {
        // Recompute the content digest with the normal codec: these probes
        // must reach semantic admission, not merely a stale-hash rejection.
        let bytes = checkpoint::encode(PROFILE, payload).unwrap();
        let error = run.resume_bytes(&bytes).unwrap_err();
        assert!(error.message().contains(message), "{error:?}");
    };
    let mut payload: Payload = checkpoint::decode(&bytes, PROFILE, Ok).unwrap();
    let mut wire = serde_json::to_value(&payload.request.run).unwrap();
    wire["sampling"][1]["lineage"] = serde_json::json!([2, 5]);
    payload.request.run = serde_json::from_value(wire).unwrap();
    reject(&payload, "exact Run");
    let mut payload: Payload = checkpoint::decode(&bytes, PROFILE, Ok).unwrap();
    payload.request.sampler.push_str("-foreign");
    reject(&payload, "exact Run");
    let mut payload: Payload = checkpoint::decode(&bytes, PROFILE, Ok).unwrap();
    payload.request.target_arch.push_str("-foreign");
    reject(&payload, "exact Run");
    let mut payload: Payload = checkpoint::decode(&bytes, PROFILE, Ok).unwrap();
    payload.paths.clear();
    reject(&payload, "retained paths");
    let mut payload: Payload = checkpoint::decode(&bytes, PROFILE, Ok).unwrap();
    payload.accepted_positions.push(exact(0.5));
    reject(&payload, "advance its accepted position");
    let mut payload: Payload = checkpoint::decode(&bytes, PROFILE, Ok).unwrap();
    payload.accepted_positions[0] = BigRational::new(1.into(), 3.into());
    reject(&payload, "dyadic grid");
    let mut payload: Payload = checkpoint::decode(&bytes, PROFILE, Ok).unwrap();
    payload.state_base64 = STANDARD.encode(
        run.program()
            .accept_state(0.5, vec![123.], "result")
            .unwrap()
            .to_bytes()
            .unwrap(),
    );
    reject(&payload, "physical State differs");
}

#[test]
fn checkpoint_rejects_truncation_unknown_profiles_and_foreign_original_horizons() {
    let run = request(WIENER, ProcessMethod::EulerMaruyama { depth: 8 });
    let bytes = saved(&run);
    for end in [0, bytes.len() / 2, bytes.len() - 1] {
        assert!(run.resume_bytes(&bytes[..end]).is_err());
    }
    let mut noncanonical = bytes.clone();
    noncanonical.push(b'\n');
    assert!(run.resume_bytes(&noncanonical).is_err());
    let payload: Payload = checkpoint::decode(&bytes, PROFILE, Ok).unwrap();
    assert!(
        run.resume_bytes(&checkpoint::encode("adaptive-ode", &payload).unwrap())
            .is_err()
    );
    let (master, identity) = run.request.sampling().unwrap();
    let foreign = ProcessRun::new(
        crate::CommonTimeRunRequest::sampled(
            run.request.plan().clone(),
            run.request.state().clone(),
            2.,
            vec![2.],
            *master,
            identity.clone(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(foreign.resume_bytes(&bytes).is_err());
    let changed_outputs = ProcessRun::new(
        crate::CommonTimeRunRequest::sampled(
            run.request.plan().clone(),
            run.request.state().clone(),
            run.request.until_s(),
            vec![0.5, 1.],
            *master,
            identity.clone(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(run.paths, changed_outputs.paths);
    assert!(changed_outputs.resume_bytes(&bytes).is_err());
}

#[test]
fn poisson_checkpoint_rejects_missing_pending_threshold_even_after_reidentification() {
    let run = request(POISSON, ProcessMethod::PoissonEvents { max_events: 64 });
    let bytes = saved(&run);
    let payload: Payload = checkpoint::decode(&bytes, PROFILE, Ok).unwrap();
    let mut wire = serde_json::to_value(payload).unwrap();
    let thresholds = wire["paths"][0][1]["path"][0]["thresholds"]
        .as_array_mut()
        .unwrap();
    assert!(thresholds.len() > 1);
    // Drop the next arrival beyond the original horizon, keeping the accepted
    // past untouched. It is still necessary stochastic continuation data.
    thresholds.pop();
    let payload: Payload = serde_json::from_value(wire).unwrap();
    let forged = checkpoint::encode(PROFILE, &payload).unwrap();
    let error = run.resume_bytes(&forged).unwrap_err();
    assert!(error.message().contains("retained paths"), "{error:?}");
}

#[test]
fn accepted_partition_cannot_skip_a_requested_output_even_after_reidentification() {
    let original = request(WIENER, ProcessMethod::EulerMaruyama { depth: 8 });
    let (master, sampling) = original.request.sampling().unwrap();
    let run = ProcessRun::new(
        crate::CommonTimeRunRequest::sampled(
            original.request.plan().clone(),
            original.request.state().clone(),
            1.,
            vec![0.25, 0.5, 1.],
            *master,
            sampling.clone(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut cursor = run.start();
    let unchanged = cursor.checkpoint_bytes().unwrap();
    // Large candidates may be evaluated and rejected without changing the path.
    drop(cursor.trial(&exact(1.)).unwrap());
    assert_eq!(cursor.checkpoint_bytes().unwrap(), unchanged);
    let error = cursor
        .accept(cursor.trial(&exact(0.5)).unwrap())
        .unwrap_err();
    assert!(error.message().contains("skips a requested output"));
    assert_eq!(cursor.checkpoint_bytes().unwrap(), unchanged);
    cursor.accept(cursor.trial(&exact(0.25)).unwrap()).unwrap();
    let bytes = cursor.checkpoint_bytes().unwrap();
    let mut payload: Payload = checkpoint::decode(&bytes, PROFILE, Ok).unwrap();
    payload.accepted_positions[0] = exact(0.5);
    let error = ProcessCursor::from_bytes(
        &checkpoint::encode(PROFILE, &payload).unwrap(),
        run.request.plan(),
    )
    .unwrap_err();
    assert!(error.message().contains("skips a requested output"));
}

#[test]
fn an_unfinished_checkpoint_cannot_be_relabelled_as_a_completed_trajectory() {
    for (source, method) in [
        (WIENER, ProcessMethod::EulerMaruyama { depth: 8 }),
        (POISSON, ProcessMethod::PoissonEvents { max_events: 64 }),
    ] {
        let run = request(source, method);
        let mut partial = run.start();
        partial.accept(partial.trial(&exact(0.5)).unwrap()).unwrap();
        let complete = run
            .request
            .advance_process(None, |_, _| std::ops::ControlFlow::Continue(()))
            .unwrap()
            .continue_value()
            .unwrap();
        let mut forged: serde_json::Value =
            serde_json::from_slice(&complete.to_bytes().unwrap()).unwrap();
        forged["payload"]["history"] = serde_json::to_value(
            history::artifact::WireProcessHistory::capture(
                &history::ProcessHistory::from_cursor(partial).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let plan = crate::ResolvedCommonPlan::Time(Box::new(run.request.plan().clone()));
        let error =
            crate::CommonTrajectory::from_bytes(&serde_json::to_vec(&forged).unwrap(), &plan)
                .unwrap_err();
        assert!(error.message().contains("complete original horizon"));
    }
}

/// Registered claim: discarded trials and persisted continuations retain one accepted path.
#[test]
fn retained_stochastic_continuation_claim() {
    checkpoint_and_rejected_trials_preserve_both_methods_exactly();
    poisson_checkpoint_retains_pending_threshold_and_right_continuous_counter();
    reidentified_mutants_reach_the_physical_path_and_request_boundaries();
    checkpoint_rejects_truncation_unknown_profiles_and_foreign_original_horizons();
    poisson_checkpoint_rejects_missing_pending_threshold_even_after_reidentification();
    accepted_partition_cannot_skip_a_requested_output_even_after_reidentification();
    an_unfinished_checkpoint_cannot_be_relabelled_as_a_completed_trajectory();
}
