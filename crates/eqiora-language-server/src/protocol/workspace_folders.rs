//! Explicit client folder changes replace the source-root authority and regroup open buffers.
use super::*;
use std::collections::BTreeSet;

pub(super) fn root_key(uri: &Uri) -> String {
    let uri = uri.as_str();
    if uri.ends_with('/') {
        uri.to_owned()
    } else {
        format!("{uri}/")
    }
}

pub(super) fn change(
    connection: &Connection,
    notification: Notification,
    state: &mut ServerState,
    scheduler: &AnalysisScheduler,
) -> ServerResult<()> {
    let Some(params) =
        decode_notification::<lsp_types::DidChangeWorkspaceFoldersParams>(notification.params)
    else {
        return Ok(());
    };
    let mut root_set: BTreeSet<_> = state.roots.iter().cloned().collect();
    for folder in params.event.removed {
        root_set.remove(&root_key(&folder.uri));
    }
    for folder in params.event.added {
        root_set.insert(root_key(&folder.uri));
    }
    let mut roots: Vec<_> = root_set.iter().cloned().collect();
    roots.sort_by_key(|root| std::cmp::Reverse(root.len()));
    if roots == state.roots {
        return Ok(());
    }

    // Folder changes are infrequent global partition changes. Invalidate the old
    // partition atomically rather than retain facts under changed root authority.
    for (_, pending) in std::mem::take(&mut state.pending) {
        pending.cancelled.store(true, Ordering::Release);
    }
    scheduler
        .queued
        .lock()
        .map_err(|_| "analysis queue failed")?
        .clear();
    state.workspaces.clear();
    state.roots = roots;
    state.projects.retain(|root, _| root_set.contains(root));
    for root in state.roots.clone() {
        synchronization::stage_project(state, &root);
    }

    let groups: BTreeSet<_> = state
        .documents
        .keys()
        .map(|uri| state.group_for_uri(uri))
        .collect();
    for group in groups {
        state.schedule_group(&group, scheduler)?;
    }
    synchronization::register_pending_watchers(connection, state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn folders(added: &[&str], removed: &[&str]) -> Notification {
        let values = |items: &[&str]| {
            items
                .iter()
                .map(|uri| json!({"uri":uri,"name":"folder"}))
                .collect::<Vec<_>>()
        };
        Notification::new(
            "workspace/didChangeWorkspaceFolders".into(),
            json!({"event":{"added":values(added),"removed":values(removed)}}),
        )
    }

    #[test]
    fn repartition_cancels_old_results_preserves_buffers_and_registers_watchers_once() {
        let root = "file:///workspace/";
        let nested = "file:///workspace/nested/";
        let outside = "file:///workspace2/other.eqi";
        let mut state = ServerState::new(vec![root.into()]);
        for (uri, source) in [
            ("file:///workspace/main.eqi", "model Main(){}"),
            ("file:///workspace/nested/child.eqi", "model Child(){}"),
            (outside, "model Other(){}"),
        ] {
            state.documents.insert(
                uri.into(),
                OpenDocument::new(Uri::from_str(uri).unwrap(), 7, source.into()),
            );
        }
        let (wake, _receiver) = crossbeam_channel::unbounded();
        let scheduler = AnalysisScheduler {
            queued: Arc::new(Mutex::new(BTreeMap::new())),
            wake,
        };
        let (connection, client) = Connection::memory();
        state.schedule_group(root, &scheduler).unwrap();
        let job = scheduler.queued.lock().unwrap().remove(root).unwrap();
        let outcome = analyze_group(
            &job.group,
            job.version,
            job.documents,
            job.project,
            &job.cancelled,
        );
        state.apply_completed(CompletedAnalysis {
            group: job.group,
            version: job.version,
            outcome,
        });
        assert!(state.workspaces.contains_key(root));
        // Keep a completed old-generation result outside the queue, like an in-flight worker.
        state.schedule_group(root, &scheduler).unwrap();
        let job = scheduler.queued.lock().unwrap().remove(root).unwrap();
        let cancelled = Arc::clone(&job.cancelled);
        let outcome = analyze_group(
            &job.group,
            job.version,
            job.documents,
            job.project,
            &job.cancelled,
        );
        let stale = CompletedAnalysis {
            group: job.group,
            version: job.version,
            outcome,
        };
        state.watch_registration_pending = true;
        change(&connection, folders(&[nested], &[]), &mut state, &scheduler).unwrap();
        assert!(cancelled.load(Ordering::Acquire));
        assert!(state.workspaces.is_empty());
        assert!(state.apply_completed(stale).is_none());
        assert_eq!(
            state.group_for_uri("file:///workspace/nested/child.eqi"),
            nested
        );
        assert_eq!(state.group_for_uri(outside), outside);
        {
            let queued = scheduler.queued.lock().unwrap();
            assert_eq!(queued.len(), 3);
            assert_eq!(queued[root].documents.len(), 1);
            assert_eq!(queued[nested].documents[0].source, "model Child(){}");
        }
        assert!(state.documents.values().all(|doc| doc.version == 7));
        let Message::Request(request) = client.receiver.try_recv().unwrap() else {
            panic!("registration request");
        };
        assert_eq!(request.method, "client/registerCapability");
        assert!(!state.watch_registration_pending);
        let version = state.next_analysis_version;
        change(
            &connection,
            folders(
                &[nested, "file:///workspace/nested"],
                &["file:///unrelated"],
            ),
            &mut state,
            &scheduler,
        )
        .unwrap();
        assert_eq!(state.next_analysis_version, version, "normalized no-op");
        change(
            &connection,
            Notification::new(
                "workspace/didChangeWorkspaceFolders".into(),
                json!({"event":{"added":[{}],"removed":[]}}),
            ),
            &mut state,
            &scheduler,
        )
        .unwrap();
        assert_eq!(state.next_analysis_version, version, "malformed input");
        let old_nested = scheduler.queued.lock().unwrap().remove(nested).unwrap();
        change(
            &connection,
            folders(&[], &[root, nested]),
            &mut state,
            &scheduler,
        )
        .unwrap();
        assert!(old_nested.cancelled.load(Ordering::Acquire));
        assert!(state.roots.is_empty());
        assert!(state.projects.is_empty());
        assert_eq!(
            state.group_for_uri("file:///workspace/main.eqi"),
            "file:///workspace/main.eqi"
        );
        assert!(
            state
                .apply_completed(CompletedAnalysis {
                    group: old_nested.group,
                    version: old_nested.version,
                    outcome: AnalysisOutcome::Empty
                })
                .is_none()
        );
        assert_eq!(scheduler.queued.lock().unwrap().len(), 3);
        assert!(
            client.receiver.try_recv().is_err(),
            "no repeated registration"
        );
    }
}
