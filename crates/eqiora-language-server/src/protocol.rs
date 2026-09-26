use std::{
    collections::{BTreeMap, VecDeque},
    error::Error,
    path::PathBuf,
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use crossbeam_channel::{Receiver, Sender};
use eqiora::api::{EditorService, EditorSnapshot, EditorSymbol, EditorWorkspaceSnapshot};
use eqiora::compiler::{CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit};
use lsp_server::{Connection, ErrorCode, Message, Notification, Request, RequestId, Response};
use lsp_types::{
    CancelParams, DidChangeTextDocumentParams, DidCloseTextDocumentParams,
    DidOpenTextDocumentParams, DocumentFormattingParams, DocumentSymbol, DocumentSymbolParams,
    DocumentSymbolResponse, FoldingRange, FoldingRangeParams, FoldingRangeProviderCapability,
    HoverProviderCapability, InitializeParams, NumberOrString, OneOf, PositionEncodingKind,
    PublishDiagnosticsParams, ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind,
    TextDocumentSyncOptions, TextEdit, Uri, WorkspaceFoldersServerCapabilities,
    WorkspaceServerCapabilities,
};
use serde::de::DeserializeOwned;

use crate::{
    lsp_projection::{lsp_diagnostic, source_range, symbol_kind, symbol_label, symbol_range},
    workspace_uri::{file_uri_path, project_file_uri},
};

type ServerResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

mod assistance;
mod inspection;
mod navigation;
mod package_analysis;
mod synchronization;
#[cfg(test)]
mod tests;

struct OpenDocument {
    uri: Uri,
    source: String,
    version: i32,
    analysis: EditorService,
}

impl OpenDocument {
    fn new(uri: Uri, version: i32, source: String) -> Self {
        let analysis_version = analysis_version(version);
        let analysis = EditorService::new(uri.as_str(), analysis_version, source.clone());
        Self {
            uri,
            source,
            version,
            analysis,
        }
    }

    fn replace(&mut self, version: i32, source: String) -> bool {
        let analysis_version = analysis_version(version);
        if version <= self.version
            || self
                .analysis
                .replace(analysis_version, source.clone())
                .is_err()
        {
            return false;
        }
        self.source = source;
        self.version = version;
        true
    }

    fn snapshot(&self) -> &EditorSnapshot {
        self.analysis.current()
    }
}

struct WorkspaceAnalysis {
    snapshot: EditorWorkspaceSnapshot,
    file_by_uri: BTreeMap<String, String>,
    uri_by_file: BTreeMap<String, Uri>,
}

#[derive(Clone)]
struct PackageProject {
    root_path: PathBuf,
    relative_by_uri: BTreeMap<String, PathBuf>,
}

struct ServerState {
    documents: BTreeMap<String, OpenDocument>,
    roots: Vec<String>,
    projects: BTreeMap<String, PackageProject>,
    workspaces: BTreeMap<String, WorkspaceAnalysis>,
    pending: BTreeMap<String, PendingAnalysis>,
    next_analysis_version: u64,
}

struct PendingAnalysis {
    version: u64,
    cancelled: Arc<AtomicBool>,
}

struct AnalysisDocument {
    key: String,
    uri: Uri,
    source: String,
}

struct AnalysisJob {
    group: String,
    version: u64,
    documents: Vec<AnalysisDocument>,
    project: Option<PackageProject>,
    cancelled: Arc<AtomicBool>,
}

#[derive(Clone)]
struct AnalysisScheduler {
    queued: Arc<Mutex<BTreeMap<String, AnalysisJob>>>,
    wake: Sender<()>,
}

struct CompletedAnalysis {
    group: String,
    version: u64,
    outcome: AnalysisOutcome,
}

enum AnalysisOutcome {
    Workspace {
        analysis: WorkspaceAnalysis,
        relative_by_uri: Option<BTreeMap<String, PathBuf>>,
    },
    Empty,
    Cancelled,
}

impl ServerState {
    fn new(mut roots: Vec<String>) -> Self {
        roots.sort();
        roots.dedup();
        roots.sort_by_key(|root| std::cmp::Reverse(root.len()));
        let mut state = Self {
            documents: BTreeMap::new(),
            roots,
            projects: BTreeMap::new(),
            workspaces: BTreeMap::new(),
            pending: BTreeMap::new(),
            next_analysis_version: 0,
        };
        for root in state.roots.clone() {
            let Ok(uri) = Uri::from_str(&root) else {
                continue;
            };
            let Some(root_path) = file_uri_path(&uri) else {
                continue;
            };
            if !root_path.join("eqiora.toml").is_file() {
                continue;
            }
            state.next_analysis_version = state.next_analysis_version.saturating_add(1);
            let Ok((snapshot, paths)) = EditorWorkspaceSnapshot::analyze_local_package_project_v1(
                state.next_analysis_version,
                &root_path,
                &BTreeMap::new(),
            ) else {
                continue;
            };
            let Some((workspace, relative_by_uri)) = package_workspace(&root, snapshot, paths)
            else {
                continue;
            };
            state.projects.insert(
                root.clone(),
                PackageProject {
                    root_path,
                    relative_by_uri,
                },
            );
            state.workspaces.insert(root, workspace);
        }
        state
    }

    fn group_for_uri(&self, uri: &str) -> String {
        self.roots
            .iter()
            .find(|root| uri.starts_with(root.as_str()))
            .cloned()
            .unwrap_or_else(|| uri.to_owned())
    }

    fn schedule_group(&mut self, group: &str, scheduler: &AnalysisScheduler) -> ServerResult<()> {
        if let Some(pending) = self.pending.remove(group) {
            pending.cancelled.store(true, Ordering::Release);
        }
        self.next_analysis_version = self.next_analysis_version.saturating_add(1);
        let version = self.next_analysis_version;
        let cancelled = Arc::new(AtomicBool::new(false));
        self.pending.insert(
            group.to_owned(),
            PendingAnalysis {
                version,
                cancelled: Arc::clone(&cancelled),
            },
        );
        let documents = self
            .documents
            .iter()
            .filter(|(uri, _)| self.group_for_uri(uri) == group)
            .map(|(uri, document)| AnalysisDocument {
                key: uri.clone(),
                uri: document.uri.clone(),
                source: document.source.clone(),
            })
            .collect();
        let project = self.projects.get(group).cloned();
        scheduler
            .queued
            .lock()
            .map_err(|_| "analysis queue failed")?
            .insert(
                group.to_owned(),
                AnalysisJob {
                    group: group.to_owned(),
                    version,
                    documents,
                    project,
                    cancelled,
                },
            );
        match scheduler.wake.try_send(()) {
            Ok(()) | Err(crossbeam_channel::TrySendError::Full(())) => Ok(()),
            Err(crossbeam_channel::TrySendError::Disconnected(())) => {
                Err("analysis worker stopped".into())
            }
        }
    }

    fn apply_completed(&mut self, completed: CompletedAnalysis) -> Option<String> {
        let current = self.pending.get(&completed.group)?;
        if current.version != completed.version {
            return None;
        }
        self.pending.remove(&completed.group);
        match completed.outcome {
            AnalysisOutcome::Workspace {
                analysis,
                relative_by_uri,
            } => {
                if let Some(relative_by_uri) = relative_by_uri {
                    self.projects
                        .get_mut(&completed.group)
                        .expect("package project remains indexed")
                        .relative_by_uri = relative_by_uri;
                }
                self.workspaces.insert(completed.group.clone(), analysis);
            }
            AnalysisOutcome::Empty => {
                self.workspaces.remove(&completed.group);
            }
            AnalysisOutcome::Cancelled => return None,
        }
        Some(completed.group)
    }

    fn resolved(&self, uri: &Uri) -> Option<(&EditorWorkspaceSnapshot, &str)> {
        let group = self.group_for_uri(uri.as_str());
        if self.pending.contains_key(&group) {
            return None;
        }
        let workspace = self.workspaces.get(&group)?;
        let file = workspace.file_by_uri.get(uri.as_str())?;
        Some((&workspace.snapshot, file))
    }

    fn snapshot(&self, uri: &Uri) -> Result<&EditorSnapshot, String> {
        let document = self
            .documents
            .get(uri.as_str())
            .ok_or_else(|| format!("document `{}` is not open", uri.as_str()))?;
        Ok(self
            .resolved(uri)
            .and_then(|(workspace, file)| workspace.document(file))
            .unwrap_or_else(|| document.snapshot()))
    }

    fn uri_for_file(&self, source_uri: &Uri, file: &str) -> Option<Uri> {
        let group = self.group_for_uri(source_uri.as_str());
        self.workspaces.get(&group)?.uri_by_file.get(file).cloned()
    }
}

fn analyze_group(
    group: &str,
    version: u64,
    documents: Vec<AnalysisDocument>,
    project: Option<PackageProject>,
    cancelled: &AtomicBool,
) -> AnalysisOutcome {
    if cancelled.load(Ordering::Acquire) {
        return AnalysisOutcome::Cancelled;
    }
    if let Some(project) = project
        && let Some(outcome) =
            package_analysis::analyze(group, version, &documents, project, cancelled)
    {
        return outcome;
    }
    let owner = CompilationNamespaceId::new(["editor.workspace"])
        .expect("fixed editor workspace namespace is valid");
    let mut units = Vec::new();
    let mut file_by_uri = BTreeMap::new();
    let mut uri_by_file = BTreeMap::new();
    for document in documents {
        if document.source.len() > EditorSnapshot::MAX_SOURCE_BYTES {
            continue;
        }
        let relative = document
            .key
            .strip_prefix(group)
            .unwrap_or(&document.key)
            .trim_start_matches('/');
        let virtual_path = if relative.is_empty() {
            "src/main.eqi".to_owned()
        } else {
            format!("src/{relative}")
        };
        let Ok(unit) = ResolvedSourceUnit::new(owner.clone(), virtual_path, document.source) else {
            continue;
        };
        let file = unit.diagnostic_file();
        file_by_uri.insert(document.key, file.clone());
        uri_by_file.insert(file, document.uri);
        units.push(unit);
    }
    if units.is_empty() {
        return if cancelled.load(Ordering::Acquire) {
            AnalysisOutcome::Cancelled
        } else {
            AnalysisOutcome::Empty
        };
    }
    let root_module = units
        .iter()
        .find(|unit| unit.module_segments() == ["main"])
        .unwrap_or(&units[0])
        .module_segments()
        .to_vec();
    let input = ResolvedHierarchyInput::with_root_module(owner, root_module, units, vec![])
        .expect("source-derived root module is valid");
    let Some(snapshot) =
        EditorWorkspaceSnapshot::analyze_modules_with_cancellation(version, input, || {
            cancelled.load(Ordering::Acquire)
        })
    else {
        return AnalysisOutcome::Cancelled;
    };
    AnalysisOutcome::Workspace {
        analysis: WorkspaceAnalysis {
            snapshot,
            file_by_uri,
            uri_by_file,
        },
        relative_by_uri: None,
    }
}

fn package_workspace(
    root: &str,
    snapshot: EditorWorkspaceSnapshot,
    paths: BTreeMap<String, PathBuf>,
) -> Option<(WorkspaceAnalysis, BTreeMap<String, PathBuf>)> {
    let mut file_by_uri = BTreeMap::new();
    let mut uri_by_file = BTreeMap::new();
    let mut relative_by_uri = BTreeMap::new();
    for (file, path) in paths {
        let uri = project_file_uri(root, &path)?;
        file_by_uri.insert(uri.as_str().to_owned(), file.clone());
        relative_by_uri.insert(uri.as_str().to_owned(), path);
        uri_by_file.insert(file, uri);
    }
    Some((
        WorkspaceAnalysis {
            snapshot,
            file_by_uri,
            uri_by_file,
        },
        relative_by_uri,
    ))
}

fn analysis_version(version: i32) -> u64 {
    u64::try_from(i64::from(version) - i64::from(i32::MIN))
        .expect("every LSP document version maps to u64")
}

pub fn run(connection: Connection, version: &str) -> ServerResult<()> {
    let capabilities = ServerCapabilities {
        position_encoding: Some(PositionEncodingKind::UTF16),
        text_document_sync: Some(TextDocumentSyncCapability::Options(
            TextDocumentSyncOptions {
                open_close: Some(true),
                change: Some(TextDocumentSyncKind::FULL),
                save: Some(lsp_types::TextDocumentSyncSaveOptions::Supported(true)),
                ..TextDocumentSyncOptions::default()
            },
        )),
        experimental: Some(serde_json::json!({"eqioraInspection": 1, "eqioraPlanInspection": 1})),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        completion_provider: Some(lsp_types::CompletionOptions {
            trigger_characters: Some(vec![".".into()]),
            ..Default::default()
        }),
        signature_help_provider: Some(lsp_types::SignatureHelpOptions {
            trigger_characters: Some(vec!["(".into(), ",".into(), "=".into()]),
            retrigger_characters: Some(vec![")".into()]),
            ..Default::default()
        }),
        definition_provider: Some(OneOf::Left(true)),
        references_provider: Some(OneOf::Left(true)),
        document_symbol_provider: Some(OneOf::Left(true)),
        document_formatting_provider: Some(OneOf::Left(true)),
        folding_range_provider: Some(FoldingRangeProviderCapability::Simple(true)),
        workspace: Some(WorkspaceServerCapabilities {
            workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                supported: Some(true),
                change_notifications: Some(OneOf::Left(false)),
            }),
            file_operations: None,
        }),
        ..ServerCapabilities::default()
    };
    let (initialize_id, initialize_params) = connection.initialize_start()?;
    let initialize_params: InitializeParams = serde_json::from_value(initialize_params)?;
    let roots = workspace_roots(&initialize_params);
    connection.initialize_finish(
        initialize_id,
        serde_json::json!({
            "capabilities": capabilities,
            "serverInfo": { "name": "eqiora-language-server", "version": version },
        }),
    )?;

    synchronization::register_watchers(&connection, &initialize_params)?;
    let mut state = ServerState::new(roots);
    let queued = Arc::new(Mutex::new(BTreeMap::<String, AnalysisJob>::new()));
    let (wake, wake_receiver) = crossbeam_channel::bounded(1);
    let scheduler = AnalysisScheduler {
        queued: Arc::clone(&queued),
        wake,
    };
    let (analysis_sender, analysis_receiver) = crossbeam_channel::unbounded();
    thread::spawn(move || {
        while wake_receiver.recv().is_ok() {
            let Ok(mut queued_jobs) = queued.lock() else {
                return;
            };
            let jobs = std::mem::take(&mut *queued_jobs);
            drop(queued_jobs);
            for (_group, job) in jobs {
                let outcome = analyze_group(
                    &job.group,
                    job.version,
                    job.documents,
                    job.project,
                    &job.cancelled,
                );
                if analysis_sender
                    .send(CompletedAnalysis {
                        group: job.group,
                        version: job.version,
                        outcome,
                    })
                    .is_err()
                {
                    return;
                }
            }
        }
    });
    let mut buffered = VecDeque::new();
    let mut input_closed = false;
    loop {
        let message = if let Some(message) = buffered.pop_front() {
            Some(message)
        } else if input_closed {
            break;
        } else {
            crossbeam_channel::select_biased! {
                recv(connection.receiver) -> message => match message {
                    Ok(message) => Some(message),
                    Err(_) => {
                        input_closed = true;
                        None
                    },
                },
                recv(analysis_receiver) -> completed => {
                    apply_completed(&connection, &mut state, completed?)?;
                    None
                },
            }
        };
        let Some(message) = message else {
            continue;
        };
        match message {
            Message::Request(request) => {
                if request.method == "shutdown" {
                    settle_pending(&connection, &mut state, &analysis_receiver)?;
                } else if let Some(group) = request_group(&request, &state)
                    && settle_group(
                        &connection,
                        &mut state,
                        &analysis_receiver,
                        &group,
                        &request.id,
                        &mut buffered,
                        &mut input_closed,
                    )?
                {
                    connection.sender.send(
                        Response::new_err(
                            request.id,
                            ErrorCode::RequestCanceled as i32,
                            "canceled by client".to_owned(),
                        )
                        .into(),
                    )?;
                    continue;
                }
                if handle_shutdown(&connection, &request, &mut buffered)? {
                    break;
                }
                handle_request(&connection, request, &state)?;
            }
            Message::Notification(notification) => {
                if notification.method == "exit" {
                    break;
                }
                handle_notification(&connection, notification, &mut state, &scheduler)?;
            }
            Message::Response(response) => synchronization::registration_result(response),
        }
    }
    Ok(())
}

#[allow(deprecated)]
fn workspace_roots(params: &InitializeParams) -> Vec<String> {
    let roots = params.workspace_folders.as_ref().map_or_else(
        || params.root_uri.iter().collect::<Vec<_>>(),
        |folders| folders.iter().map(|folder| &folder.uri).collect(),
    );
    roots
        .into_iter()
        .map(|uri| {
            let uri = uri.as_str();
            if uri.ends_with('/') {
                uri.to_owned()
            } else {
                format!("{uri}/")
            }
        })
        .collect()
}

fn handle_notification(
    connection: &Connection,
    notification: Notification,
    state: &mut ServerState,
    scheduler: &AnalysisScheduler,
) -> ServerResult<()> {
    match notification.method.as_str() {
        "textDocument/didOpen" => {
            let Some(params) = decode_notification(notification.params) else {
                return Ok(());
            };
            let params: DidOpenTextDocumentParams = params;
            let item = params.text_document;
            let document = OpenDocument::new(item.uri.clone(), item.version, item.text);
            let group = state.group_for_uri(item.uri.as_str());
            state
                .documents
                .insert(item.uri.as_str().to_owned(), document);
            state.schedule_group(&group, scheduler)?;
        }
        "textDocument/didChange" => {
            let Some(params) = decode_notification(notification.params) else {
                return Ok(());
            };
            let params: DidChangeTextDocumentParams = params;
            let identifier = params.text_document;
            if params.content_changes.len() != 1 {
                return Ok(());
            }
            let Some(change) = params.content_changes.into_iter().next() else {
                return Ok(());
            };
            if change.range.is_some() {
                return Ok(());
            }
            let group = state.group_for_uri(identifier.uri.as_str());
            let accepted = state
                .documents
                .get_mut(identifier.uri.as_str())
                .is_some_and(|document| document.replace(identifier.version, change.text));
            if accepted {
                state.schedule_group(&group, scheduler)?;
            }
        }
        "textDocument/didClose" => {
            let Some(params) = decode_notification(notification.params) else {
                return Ok(());
            };
            let params: DidCloseTextDocumentParams = params;
            let group = state.group_for_uri(params.text_document.uri.as_str());
            if state
                .documents
                .remove(params.text_document.uri.as_str())
                .is_some()
            {
                state.schedule_group(&group, scheduler)?;
                let notification = Notification::new(
                    "textDocument/publishDiagnostics".to_owned(),
                    PublishDiagnosticsParams::new(params.text_document.uri, Vec::new(), None),
                );
                connection.sender.send(notification.into())?;
            }
        }
        "workspace/didChangeWatchedFiles" | "textDocument/didSave" => {
            synchronization::refresh(notification, state, scheduler)?;
        }
        "initialized" | "$/cancelRequest" => {}
        _ => {}
    }
    Ok(())
}

fn publish_group_diagnostics(
    connection: &Connection,
    state: &ServerState,
    group: &str,
) -> ServerResult<()> {
    let uris = state
        .documents
        .values()
        .filter(|document| state.group_for_uri(document.uri.as_str()) == group)
        .map(|document| document.uri.clone())
        .collect::<Vec<_>>();
    for uri in uris {
        publish_diagnostics(connection, state, &uri)?;
    }
    Ok(())
}

fn apply_completed(
    connection: &Connection,
    state: &mut ServerState,
    completed: CompletedAnalysis,
) -> ServerResult<()> {
    if let Some(group) = state.apply_completed(completed) {
        publish_group_diagnostics(connection, state, &group)?;
    }
    Ok(())
}

fn settle_pending(
    connection: &Connection,
    state: &mut ServerState,
    receiver: &Receiver<CompletedAnalysis>,
) -> ServerResult<()> {
    while !state.pending.is_empty() {
        apply_completed(connection, state, receiver.recv()?)?;
    }
    Ok(())
}

fn settle_group(
    connection: &Connection,
    state: &mut ServerState,
    receiver: &Receiver<CompletedAnalysis>,
    group: &str,
    request_id: &RequestId,
    buffered: &mut VecDeque<Message>,
    input_closed: &mut bool,
) -> ServerResult<bool> {
    loop {
        if remove_cancellation(buffered, request_id) {
            return Ok(true);
        }
        if !state.pending.contains_key(group) {
            return Ok(false);
        }
        if *input_closed {
            apply_completed(connection, state, receiver.recv()?)?;
            continue;
        }
        crossbeam_channel::select_biased! {
            recv(connection.receiver) -> message => {
                match message {
                    Ok(message) => {
                        if cancels_request(&message, request_id) {
                            return Ok(true);
                        }
                        buffered.push_back(message);
                    }
                    Err(_) => {
                        *input_closed = true;
                    }
                }
            },
            recv(receiver) -> completed => {
                apply_completed(connection, state, completed?)?;
            },
        }
    }
}

fn remove_cancellation(buffered: &mut VecDeque<Message>, request_id: &RequestId) -> bool {
    let Some(index) = buffered
        .iter()
        .position(|message| cancels_request(message, request_id))
    else {
        return false;
    };
    buffered.remove(index);
    true
}

fn cancels_request(message: &Message, request_id: &RequestId) -> bool {
    let Message::Notification(notification) = message else {
        return false;
    };
    if notification.method != "$/cancelRequest" {
        return false;
    }
    let Ok(params) = serde_json::from_value::<CancelParams>(notification.params.clone()) else {
        return false;
    };
    let cancelled_id = match params.id {
        NumberOrString::Number(id) => RequestId::from(id),
        NumberOrString::String(id) => RequestId::from(id),
    };
    cancelled_id == *request_id
}

fn request_group(request: &Request, state: &ServerState) -> Option<String> {
    let uri = request.params.get("textDocument")?.get("uri")?.as_str()?;
    Some(state.group_for_uri(uri))
}

fn handle_shutdown(
    connection: &Connection,
    request: &Request,
    buffered: &mut VecDeque<Message>,
) -> ServerResult<bool> {
    if request.method != "shutdown" {
        return Ok(false);
    }
    connection
        .sender
        .send(Response::new_ok(request.id.clone(), ()).into())?;
    let message = if let Some(message) = buffered.pop_front() {
        message
    } else {
        connection
            .receiver
            .recv_timeout(Duration::from_secs(30))
            .map_err(|error| format!("failed waiting for exit notification: {error}"))?
    };
    match message {
        Message::Notification(notification) if notification.method == "exit" => Ok(true),
        message => Err(format!("unexpected message during shutdown: {message:?}").into()),
    }
}

fn decode_notification<T: DeserializeOwned>(params: serde_json::Value) -> Option<T> {
    serde_json::from_value(params).ok()
}

fn handle_request(
    connection: &Connection,
    request: Request,
    state: &ServerState,
) -> ServerResult<()> {
    let id = request.id.clone();
    let response = match request.method.as_str() {
        "eqiora/inspect" => response_from(
            id,
            decode(request.params).and_then(|params| inspection::inspect(params, state)),
        ),
        "textDocument/formatting" => response_from(
            id,
            decode(request.params).and_then(|params| formatting(params, state)),
        ),
        "textDocument/documentSymbol" => response_from(
            id,
            decode(request.params).and_then(|params| document_symbols(params, state)),
        ),
        "textDocument/foldingRange" => response_from(
            id,
            decode(request.params).and_then(|params| folding_ranges(params, state)),
        ),
        "textDocument/hover" => response_from(
            id,
            decode(request.params).and_then(|params| navigation::hover(params, state)),
        ),
        "textDocument/completion" => response_from(
            id,
            decode(request.params).and_then(|params| assistance::completion(params, state)),
        ),
        "textDocument/signatureHelp" => response_from(
            id,
            decode(request.params).and_then(|params| assistance::signature_help(params, state)),
        ),
        "textDocument/definition" => response_from(
            id,
            decode(request.params).and_then(|params| navigation::definition(params, state)),
        ),
        "textDocument/references" => response_from(
            id,
            decode(request.params).and_then(|params| navigation::references(params, state)),
        ),
        _ => Response::new_err(
            id,
            ErrorCode::MethodNotFound as i32,
            format!("unsupported request method `{}`", request.method),
        ),
    };
    connection.sender.send(response.into())?;
    Ok(())
}

fn decode<T: DeserializeOwned>(params: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(params).map_err(|error| format!("invalid request parameters: {error}"))
}

fn response_from<T: serde::Serialize>(
    id: lsp_server::RequestId,
    result: Result<T, String>,
) -> Response {
    match result {
        Ok(value) => Response::new_ok(id, value),
        Err(message) => Response::new_err(id, ErrorCode::InvalidParams as i32, message),
    }
}

fn document<'a>(state: &'a ServerState, uri: &Uri) -> Result<&'a OpenDocument, String> {
    state
        .documents
        .get(uri.as_str())
        .ok_or_else(|| format!("document `{}` is not open", uri.as_str()))
}

fn formatting(
    params: DocumentFormattingParams,
    state: &ServerState,
) -> Result<Vec<TextEdit>, String> {
    let document = document(state, &params.text_document.uri)?;
    let snapshot = state.snapshot(&params.text_document.uri)?;
    let Some(formatted) = snapshot.formatted() else {
        return Ok(Vec::new());
    };
    if formatted == document.source {
        return Ok(Vec::new());
    }
    Ok(vec![TextEdit {
        range: source_range(snapshot, 0, document.source.len())?,
        new_text: formatted.to_owned(),
    }])
}

fn document_symbols(
    params: DocumentSymbolParams,
    state: &ServerState,
) -> Result<DocumentSymbolResponse, String> {
    let snapshot = state.snapshot(&params.text_document.uri)?;
    let symbols = snapshot
        .symbols()
        .iter()
        .map(|symbol| lsp_symbol(snapshot, symbol))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DocumentSymbolResponse::Nested(symbols))
}

#[allow(deprecated)]
fn lsp_symbol(snapshot: &EditorSnapshot, symbol: &EditorSymbol) -> Result<DocumentSymbol, String> {
    let range = symbol_range(snapshot, symbol)?;
    let children = symbol
        .children()
        .iter()
        .map(|child| lsp_symbol(snapshot, child))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DocumentSymbol {
        name: symbol.name().to_owned(),
        detail: Some(
            symbol
                .detail()
                .unwrap_or_else(|| symbol_label(symbol.kind()))
                .to_owned(),
        ),
        kind: symbol_kind(symbol.kind()),
        tags: None,
        deprecated: None,
        range,
        selection_range: range,
        children: (!children.is_empty()).then_some(children),
    })
}

fn folding_ranges(
    params: FoldingRangeParams,
    state: &ServerState,
) -> Result<Vec<FoldingRange>, String> {
    let snapshot = state.snapshot(&params.text_document.uri)?;
    let mut ranges = Vec::new();
    collect_folding_ranges(snapshot, snapshot.symbols(), &mut ranges)?;
    Ok(ranges)
}

fn collect_folding_ranges(
    snapshot: &EditorSnapshot,
    symbols: &[EditorSymbol],
    output: &mut Vec<FoldingRange>,
) -> Result<(), String> {
    for symbol in symbols {
        let range = symbol_range(snapshot, symbol)?;
        if range.start.line < range.end.line {
            output.push(FoldingRange {
                start_line: range.start.line,
                start_character: Some(range.start.character),
                end_line: range.end.line,
                end_character: Some(range.end.character),
                kind: None,
                collapsed_text: Some(symbol.name().to_owned()),
            });
        }
        collect_folding_ranges(snapshot, symbol.children(), output)?;
    }
    Ok(())
}

fn publish_diagnostics(
    connection: &Connection,
    state: &ServerState,
    uri: &Uri,
) -> ServerResult<()> {
    let document = document(state, uri)?;
    let snapshot = state.snapshot(uri)?;
    let diagnostics = snapshot
        .diagnostics()
        .iter()
        .map(|diagnostic| lsp_diagnostic(snapshot, diagnostic))
        .collect();
    let params =
        PublishDiagnosticsParams::new(document.uri.clone(), diagnostics, Some(document.version));
    connection
        .sender
        .send(Notification::new("textDocument/publishDiagnostics".to_owned(), params).into())?;
    Ok(())
}
