//! Client-owned file watching refreshes the existing package analysis queue.
use super::*;
use std::collections::BTreeSet;

const WATCH_REGISTRATION: &str = "eqiora.workspace-files";

pub(super) fn register_watchers(
    connection: &Connection,
    params: &InitializeParams,
    state: &mut ServerState,
) -> ServerResult<()> {
    state.watch_registration_pending = params
        .capabilities
        .workspace
        .as_ref()
        .and_then(|workspace| workspace.did_change_watched_files.as_ref())
        .and_then(|watching| watching.dynamic_registration)
        .unwrap_or(false);
    register_pending_watchers(connection, state)
}

pub(super) fn register_pending_watchers(
    connection: &Connection,
    state: &mut ServerState,
) -> ServerResult<()> {
    if state.watch_registration_pending && !state.roots.is_empty() {
        connection.sender.send(
            Request::new(
                RequestId::from(WATCH_REGISTRATION.to_owned()),
                "client/registerCapability".to_owned(),
                lsp_types::RegistrationParams {
                    registrations: vec![lsp_types::Registration {
                        id: WATCH_REGISTRATION.to_owned(),
                        method: "workspace/didChangeWatchedFiles".to_owned(),
                        register_options: Some(serde_json::json!({
                            "watchers": [
                                {"globPattern": "**/*.eqi"},
                                {"globPattern": "**/eqiora.toml"},
                                {"globPattern": "**/eqiora.lock"}
                            ]
                        })),
                    }],
                },
            )
            .into(),
        )?;
        state.watch_registration_pending = false;
    }
    Ok(())
}

pub(super) fn registration_result(response: Response) {
    if response.id == RequestId::from(WATCH_REGISTRATION.to_owned())
        && response.response_result.is_err()
    {
        crate::log_event("warn", "workspace_watch_registration_rejected");
    }
}

pub(super) fn refresh(
    notification: Notification,
    state: &mut ServerState,
    scheduler: &AnalysisScheduler,
) -> ServerResult<()> {
    let groups: BTreeSet<String> = if notification.method == "textDocument/didSave" {
        let Some(params) =
            decode_notification::<lsp_types::DidSaveTextDocumentParams>(notification.params)
        else {
            return Ok(());
        };
        // Full-document didChange is authoritative, including any unsaved overlays.
        if !state
            .documents
            .contains_key(params.text_document.uri.as_str())
        {
            return Ok(());
        }
        state
            .roots
            .iter()
            .filter(|root| params.text_document.uri.as_str().starts_with(root.as_str()))
            .cloned()
            .collect()
    } else {
        let Some(params) =
            decode_notification::<lsp_types::DidChangeWatchedFilesParams>(notification.params)
        else {
            return Ok(());
        };
        params
            .changes
            .iter()
            .filter(|event| {
                file_uri_path(&event.uri).is_some_and(|path| {
                    path.extension().is_some_and(|extension| extension == "eqi")
                        || path
                            .file_name()
                            .is_some_and(|name| name == "eqiora.toml" || name == "eqiora.lock")
                })
            })
            // A client-declared nested workspace can also be a dependency source
            // of its parent project, so refresh every containing project.
            .flat_map(|event| {
                state
                    .roots
                    .iter()
                    .filter(|root| event.uri.as_str().starts_with(root.as_str()))
                    .cloned()
            })
            .collect()
    };
    for group in groups {
        if !state
            .documents
            .keys()
            .any(|uri| state.group_for_uri(uri) == group)
        {
            continue;
        }
        if !stage_project(state, &group) {
            continue;
        }
        state.schedule_group(&group, scheduler)?;
    }
    Ok(())
}

// Callers select this group from the current client-declared root set. Native
// admission must still supply every path before current buffers can override it.
pub(super) fn stage_project(state: &mut ServerState, group: &str) -> bool {
    if state.projects.contains_key(group) {
        return true;
    }
    let Some(root_path) = Uri::from_str(group)
        .ok()
        .and_then(|uri| file_uri_path(&uri))
        .filter(|path| path.join("eqiora.toml").is_file())
    else {
        return false;
    };
    state.projects.insert(
        group.to_owned(),
        PackageProject {
            root_path,
            relative_by_uri: BTreeMap::new(),
        },
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn watchers_require_client_opt_in_and_a_workspace() {
        for (support, folders, expected) in [
            (false, true, false),
            (true, false, false),
            (true, true, true),
        ] {
            let (connection, client) = Connection::memory();
            let params = serde_json::from_value(json!({
                "capabilities":{"workspace":{"didChangeWatchedFiles":{"dynamicRegistration":support}}},
                "workspaceFolders": if folders { json!([{"uri":"file:///project","name":"project"}]) } else { json!([]) }
            })).unwrap();
            let mut state = ServerState::new(workspace_roots(&params));
            register_watchers(&connection, &params, &mut state).unwrap();
            assert_eq!(state.watch_registration_pending, support && !folders);
            let response = client.receiver.try_recv();
            assert_eq!(response.is_ok(), expected);
            if let Ok(Message::Request(request)) = response {
                assert_eq!(request.method, "client/registerCapability");
                let watchers = &request.params["registrations"][0]["registerOptions"]["watchers"];
                assert_eq!(
                    watchers,
                    &json!([
                        {"globPattern":"**/*.eqi"}, {"globPattern":"**/eqiora.toml"},
                        {"globPattern":"**/eqiora.lock"}
                    ])
                );
            }
        }
    }

    #[test]
    fn nested_workspace_event_refreshes_both_owning_package_graphs() {
        let mut state = ServerState::new(vec![]);
        let groups = ["file:///project/library/", "file:///project/"];
        for group in groups {
            state.roots.push(group.into());
            state.projects.insert(
                group.into(),
                PackageProject {
                    root_path: PathBuf::from("/project"),
                    relative_by_uri: BTreeMap::new(),
                },
            );
            let uri = Uri::from_str(&format!("{group}main.eqi")).unwrap();
            state.documents.insert(
                uri.as_str().into(),
                OpenDocument::new(uri, 7, "model Open(){}".into()),
            );
        }
        let (wake, _receiver) = crossbeam_channel::bounded(1);
        let scheduler = AnalysisScheduler {
            queued: Arc::new(Mutex::new(BTreeMap::new())),
            wake,
        };
        refresh(
            Notification::new(
                "workspace/didChangeWatchedFiles".into(),
                json!({
                    "changes":[{"uri":"file:///project/library/part.eqi","type":2}]
                }),
            ),
            &mut state,
            &scheduler,
        )
        .unwrap();
        let queued = scheduler.queued.lock().unwrap();
        assert_eq!(queued.len(), 2);
        for group in groups {
            assert_eq!(queued[group].documents.len(), 1);
            assert_eq!(
                queued[group].documents[0].uri.as_str(),
                format!("{group}main.eqi")
            );
        }
        drop(queued);
        refresh(
            Notification::new(
                "textDocument/didSave".into(),
                json!({
                    "textDocument":{"uri":"file:///project/library/main.eqi"}
                }),
            ),
            &mut state,
            &scheduler,
        )
        .unwrap();
        assert_eq!(state.next_analysis_version, 4);
    }

    #[test]
    fn duplicate_events_coalesce_and_unrelated_paths_cannot_supersede_analysis() {
        let group = "file:///project/";
        let uri = Uri::from_str("file:///project/main.eqi").unwrap();
        let mut state = ServerState::new(vec![]);
        state.roots.push(group.into());
        state.projects.insert(
            group.into(),
            PackageProject {
                root_path: PathBuf::from("/project"),
                relative_by_uri: BTreeMap::new(),
            },
        );
        state.documents.insert(
            uri.as_str().into(),
            OpenDocument::new(uri, 7, "model Unsaved(){}".into()),
        );
        let (wake, _receiver) = crossbeam_channel::bounded(1);
        let scheduler = AnalysisScheduler {
            queued: Arc::new(Mutex::new(BTreeMap::new())),
            wake,
        };
        let event = |changes| {
            Notification::new(
                "workspace/didChangeWatchedFiles".into(),
                json!({"changes":changes}),
            )
        };
        refresh(
            event(json!([
                {"uri":"file:///project/dep.eqi","type":2},
                {"uri":"file:///project/dep.eqi","type":2},
                {"uri":"file:///project/eqiora.toml","type":2},
                {"uri":"file:///project/eqiora.lock","type":2}
            ])),
            &mut state,
            &scheduler,
        )
        .unwrap();
        assert_eq!(state.next_analysis_version, 1);
        assert_eq!(scheduler.queued.lock().unwrap().len(), 1);
        refresh(
            event(json!([
                {"uri":"file:///project-other/dep.eqi","type":2},
                {"uri":"file:///project/readme.md","type":2},
                {"uri":"untitled:dep.eqi","type":2}
            ])),
            &mut state,
            &scheduler,
        )
        .unwrap();
        assert_eq!(state.next_analysis_version, 1);
        let first = scheduler.queued.lock().unwrap().remove(group).unwrap();
        refresh(
            event(json!([{"uri":"file:///project/dep.eqi","type":3}])),
            &mut state,
            &scheduler,
        )
        .unwrap();
        assert!(first.cancelled.load(Ordering::Acquire));
        let latest = scheduler.queued.lock().unwrap().remove(group).unwrap();
        assert_eq!(latest.documents[0].source, "model Unsaved(){}");
        assert_eq!(state.documents.values().next().unwrap().version, 7);
    }
}
