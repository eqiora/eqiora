use super::*;

fn cancellation(id: i32) -> Message {
    Notification::new("$/cancelRequest".to_owned(), serde_json::json!({"id": id})).into()
}

#[test]
fn pending_request_cancellation_preserves_unrelated_messages() {
    for already_buffered in [false, true] {
        let (connection, client) = Connection::memory();
        let mut state = ServerState::new(vec![]);
        state.pending.insert(
            "group".to_owned(),
            PendingAnalysis {
                version: 1,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
        );
        // No worker can complete analysis. A missed cancellation fails rather than hanging.
        let (sender, receiver) = crossbeam_channel::unbounded();
        drop(sender);
        let mut buffered = VecDeque::new();
        for message in [cancellation(3), cancellation(2)] {
            if already_buffered {
                buffered.push_back(message);
            } else {
                client.sender.send(message).unwrap();
            }
        }
        let mut input_closed = false;
        assert!(
            settle_group(
                &connection,
                &mut state,
                &receiver,
                "group",
                &RequestId::from(2),
                &mut buffered,
                &mut input_closed,
            )
            .expect("cancel without waiting for analysis")
        );
        assert_eq!(buffered.len(), 1);
        assert!(cancels_request(&buffered[0], &RequestId::from(3)));
        assert!(state.pending.contains_key("group"));
        assert!(!input_closed);
    }
}

#[test]
fn completed_analysis_does_not_wait_for_a_future_cancellation() {
    let (connection, _client) = Connection::memory();
    let mut state = ServerState::new(vec![]);
    let (_sender, receiver) = crossbeam_channel::unbounded();
    assert!(
        !settle_group(
            &connection,
            &mut state,
            &receiver,
            "group",
            &RequestId::from(2),
            &mut VecDeque::new(),
            &mut false,
        )
        .unwrap()
    );
}

#[test]
fn superseded_in_flight_analysis_cannot_replace_current_revision() {
    let mut state = ServerState::new(vec![]);
    let (wake, _receiver) = crossbeam_channel::unbounded();
    let scheduler = AnalysisScheduler {
        queued: Arc::new(Mutex::new(BTreeMap::new())),
        wake,
    };
    state.schedule_group("group", &scheduler).unwrap();
    // Dequeue exactly as the worker does, then supersede before delivering completion.
    let old = scheduler.queued.lock().unwrap().remove("group").unwrap();
    state.schedule_group("group", &scheduler).unwrap();
    let current_version = state.pending["group"].version;
    assert_ne!(old.version, current_version);
    assert!(old.cancelled.load(Ordering::Acquire));
    assert_eq!(
        state.apply_completed(CompletedAnalysis {
            group: old.group,
            version: old.version,
            outcome: AnalysisOutcome::Empty,
        }),
        None
    );
    assert_eq!(state.pending["group"].version, current_version);
    let current = scheduler.queued.lock().unwrap().remove("group").unwrap();
    assert!(!current.cancelled.load(Ordering::Acquire));
    assert_eq!(
        state.apply_completed(CompletedAnalysis {
            group: current.group,
            version: current.version,
            outcome: AnalysisOutcome::Empty,
        }),
        Some("group".to_owned())
    );
    assert!(!state.pending.contains_key("group"));
}
