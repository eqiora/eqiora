import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import iconUrl from "../assets/icon.svg";
import {
  COMMAND_REGISTRY,
  type FocusTarget,
  resolveApplication,
  resolveCommandAvailability,
  resolveElementFocusId,
  type WorkspaceId,
} from "./application";
import { type StudioExample, studioBridge } from "./bridge";
import { useCadSession } from "./cad-session";
import { CadWorkspace } from "./cad-workspace";
import { type CommandAvailability, CommandPalette } from "./command-palette";
import type { CommandId } from "./commands";
import { Icon, Inspector, ModelOutline, SourceEditor } from "./components";
import { studioCompileRequest } from "./control-protocol";
import { type DiagnosticPresentation, Diagnostics } from "./diagnostics";
import { CAD_EXAMPLE_SOURCE, EXAMPLE_SOURCE } from "./example";
import { formatMessage } from "./messages";
import { ModelCanvas } from "./projection";
import { BRIDGE_PROTOCOL, type SourceSpan, type StudioDiagnostic } from "./protocol";
import { sourceLocationLabel, sourceSelection } from "./source-span";
import { currentLayout, initialStudioState, type NodeLayout, studioReducer } from "./state";
import { validateValueEditInput } from "./value-edit";
import { readWorkspace, writeWorkspace } from "./workspace";

export function App() {
  const [state, dispatch] = useReducer(studioReducer, EXAMPLE_SOURCE, initialStudioState);
  const compileSequence = useRef(0),
    valuePreviewSequence = useRef(0),
    valueCommitSequence = useRef(0);
  const compiledOnce = useRef(false),
    sourceEditor = useRef<HTMLTextAreaElement>(null),
    relationView = useRef<HTMLElement>(null);
  const latestSource = useRef(state.source);
  const [requestedWorkspace, setRequestedWorkspace] = useState<WorkspaceId>("relations");
  latestSource.current = state.source;
  const sourceEdited = state.compiledSource !== state.source;
  const {
    projection: cadProjection,
    selection: cadSelectionState,
    status: cadStatus,
    error: cadError,
    requestSelection: requestCadSelection,
  } = useCadSession(state.document === null || sourceEdited ? null : state.document.digest);
  const application = useMemo(
    () =>
      resolveApplication(
        {
          acceptedProjection: state.document,
          cad: { status: cadStatus, acceptedModelDigest: cadProjection?.modelDigest ?? null },
        },
        requestedWorkspace,
      ),
    [cadProjection?.modelDigest, cadStatus, requestedWorkspace, state.document],
  );
  const activeWorkspace = application.workspace;
  const sourceAncestor =
    state.document !== null &&
    state.sourceDigest !== null &&
    state.document.digest !== state.sourceDigest;
  const layout = currentLayout(state);
  const selectedNode = useMemo(
    () => state.document?.nodes.find((node) => node.id === state.selectedNodeId) ?? null,
    [state.document, state.selectedNodeId],
  );
  const valueValidation = useMemo(
    () =>
      selectedNode?.kind !== "parameter" || selectedNode.value === null
        ? { value: null, error: null }
        : validateValueEditInput(state.valueEditInput, selectedNode.value),
    [selectedNode?.kind, selectedNode?.value, state.valueEditInput],
  );
  const valueEditDisabledReason = sourceEdited ? formatMessage("command.reason.edit-source") : null;
  const valueEditExpanded = valueValidation.value !== null || state.valueEditStatus.kind !== "idle";
  const present = useCallback(
    (diagnostic: StudioDiagnostic, source: string | null): DiagnosticPresentation => ({
      diagnostic,
      location:
        diagnostic.span === null || source === null
          ? null
          : sourceLocationLabel(source, diagnostic.span),
      navigable: diagnostic.span !== null && source === state.source,
    }),
    [state.source],
  );
  const diagnostics = useMemo(
    () => [
      ...state.compileDiagnostics.map((d) => present(d, state.compileDiagnosticSource)),
      ...state.valueEditDiagnostics.map((d) => present(d, state.compiledSource)),
    ],
    [
      present,
      state.compileDiagnostics,
      state.compileDiagnosticSource,
      state.compiledSource,
      state.valueEditDiagnostics,
    ],
  );
  const revealSourceSpan = useCallback((source: string, span: SourceSpan) => {
    const editor = sourceEditor.current;
    if (editor === null) return;
    const selection = sourceSelection(source, span);
    editor.focus();
    editor.setSelectionRange(selection.start, selection.end, "forward");
  }, []);
  const submitSource = useCallback(
    async (source: string, example: StudioExample | null) => {
      const requestId = ++compileSequence.current;
      if (source !== state.source) {
        latestSource.current = source;
        dispatch({ type: "source-edited", source });
      }
      dispatch({ type: "compile-started", requestId });
      const request = studioCompileRequest(`studio.compile:${requestId}`, "untitled.eqi", source);
      const response =
        example === null
          ? await studioBridge.compile(request)
          : await studioBridge.loadReadOnlyExample(example, request);
      let workspaceLayout: NodeLayout | null = null;
      if (response.result !== null) {
        try {
          workspaceLayout = readWorkspace(window.localStorage, response.result.digest);
        } catch {
          workspaceLayout = null;
        }
      }
      dispatch({
        type: "compile-finished",
        requestId,
        compiledSource: source,
        document: response.result,
        workspaceLayout,
        diagnostics: response.diagnostics,
      });
      const span = response.diagnostics.find((d) => d.span !== null)?.span;
      if (response.result === null && span != null && latestSource.current === source)
        window.requestAnimationFrame(() => revealSourceSpan(source, span));
    },
    [revealSourceSpan, state.source],
  );
  const compile = useCallback(() => submitSource(state.source, null), [state.source, submitSource]);
  const openCadExample = useCallback(async () => {
    await submitSource(CAD_EXAMPLE_SOURCE, "cad");
    setRequestedWorkspace("geometry");
  }, [submitSource]);
  useEffect(() => {
    if (!compiledOnce.current) {
      compiledOnce.current = true;
      void submitSource(EXAMPLE_SOURCE, "decay");
    }
  }, [submitSource]);
  useEffect(() => {
    const document = state.document,
      target = selectedNode,
      value = valueValidation.value;
    if (
      document === null ||
      target === null ||
      value === null ||
      sourceEdited ||
      target.kind !== "parameter"
    )
      return;
    const requestId = ++valuePreviewSequence.current,
      digest = document.digest,
      targetId = target.id,
      input = state.valueEditInput;
    dispatch({ type: "value-edit-preview-started", requestId });
    const timer = window.setTimeout(() => {
      void studioBridge
        .previewValueEdit({ protocol: BRIDGE_PROTOCOL, digest, targetId, value })
        .then((response) =>
          dispatch({
            type: "value-edit-preview-finished",
            requestId,
            digest,
            targetId,
            input,
            plan: response.result,
            diagnostics: response.diagnostics,
          }),
        );
    }, 180);
    return () => window.clearTimeout(timer);
  }, [selectedNode, sourceEdited, state.document, state.valueEditInput, valueValidation.value]);
  useEffect(() => {
    const document = state.document;
    if (document === null) return;
    const saved = state.layoutsByDigest[document.digest];
    if (saved === undefined) return;
    const timer = window.setTimeout(() => {
      try {
        writeWorkspace(window.localStorage, document.digest, saved);
      } catch {}
    }, 250);
    return () => window.clearTimeout(timer);
  }, [state.document, state.layoutsByDigest]);
  const commitValueEdit = useCallback(async () => {
    if (
      state.document === null ||
      selectedNode === null ||
      selectedNode.kind !== "parameter" ||
      valueValidation.value === null ||
      state.valueEditStatus.kind !== "ready" ||
      sourceEdited ||
      state.valueEditStatus.digest !== state.document.digest ||
      state.valueEditStatus.targetId !== selectedNode.id ||
      state.valueEditStatus.input !== state.valueEditInput
    )
      return;
    const plan = state.valueEditStatus.plan,
      requestId = ++valueCommitSequence.current;
    dispatch({ type: "value-edit-commit-started", requestId, plan });
    const response = await studioBridge.commitValueEdit({
      protocol: BRIDGE_PROTOCOL,
      digest: state.document.digest,
      targetId: selectedNode.id,
      value: valueValidation.value,
      planKey: plan.key,
    });
    dispatch({
      type: "value-edit-commit-finished",
      requestId,
      result: response.result,
      diagnostics: response.diagnostics,
    });
  }, [
    selectedNode,
    sourceEdited,
    state.document,
    state.valueEditInput,
    state.valueEditStatus,
    valueValidation.value,
  ]);
  const cadAvailability = application.workflows.find(
    (workflow) => workflow.definition.id === "cad-box",
  )?.availability;
  if (cadAvailability === undefined) throw new Error("Closed Studio registry omitted CAD");
  const canNavigate = state.valueEditStatus.kind !== "committing";
  const rawAvailability = resolveCommandAvailability({
    activeWorkflow: application.activeWorkflow,
    compiling: state.compileStatus.kind === "compiling",
    documentAccepted: state.document !== null,
    valueEditReady: state.valueEditStatus.kind === "ready",
    valueEditBlock: sourceEdited ? "source" : null,
    revisionNavigationBlocked: !canNavigate,
    canUndo: state.revisionIndex > 0 && canNavigate,
    canRedo:
      state.revisionIndex >= 0 &&
      state.revisionIndex < state.revisionLineage.length - 1 &&
      canNavigate,
    selectedEntity:
      application.activeWorkflow === "cad-box"
        ? cadSelectionState.accepted !== null
        : selectedNode !== null,
    cadAvailability,
  });
  const commandAvailability = Object.fromEntries(
    Object.entries(rawAvailability).map(([id, item]) => [
      id,
      { enabled: item.enabled, reason: item.reason === null ? null : formatMessage(item.reason) },
    ]),
  ) as CommandAvailability;
  const focusTarget = useCallback((target: FocusTarget) => {
    if (target === "source-editor") sourceEditor.current?.focus();
    else if (target === "relation-view") relationView.current?.focus();
    else document.getElementById(resolveElementFocusId(target))?.focus();
  }, []);
  const focusCommand = useCallback(
    (command: CommandId) => {
      const target = COMMAND_REGISTRY.find((item) => item.id === command)?.focusTarget;
      if (target != null) window.setTimeout(() => focusTarget(target), 0);
    },
    [focusTarget],
  );
  const executeCommand = useCallback(
    (command: CommandId) => {
      if (!commandAvailability[command].enabled) return;
      switch (command) {
        case "model.compile":
          void compile();
          return;
        case "edit.commit":
          void commitValueEdit();
          return;
        case "history.undo":
          dispatch({ type: "revision-undo" });
          return;
        case "history.redo":
          dispatch({ type: "revision-redo" });
          return;
        case "view.reflow":
          dispatch({ type: "layout-reset" });
          return;
        case "workspace.relations":
          setRequestedWorkspace("relations");
          break;
        case "workspace.geometry":
          setRequestedWorkspace("geometry");
          break;
        case "example.cad":
          void openCadExample();
          return;
        default:
          break;
      }
      focusCommand(command);
    },
    [commandAvailability, commitValueEdit, compile, focusCommand, openCadExample],
  );
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        dispatch({ type: "command-palette-opened" });
        return;
      }
      const target = event.target;
      const editing =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        (target instanceof HTMLElement && target.isContentEditable);
      if (!editing && (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        executeCommand(event.shiftKey ? "history.redo" : "history.undo");
      }
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [executeCommand]);
  return (
    <>
      <CommandPalette
        availability={commandAvailability}
        onClose={() => dispatch({ type: "command-palette-closed" })}
        onExecute={executeCommand}
        onQuery={(query) => dispatch({ type: "command-query-edited", query })}
        open={state.commandPalette.kind === "open"}
        query={state.commandPalette.kind === "open" ? state.commandPalette.query : ""}
      />
      <div className="studio-shell">
        <header className="app-bar">
          <div className="brand">
            <img alt="" className="brand-mark" height={28} src={iconUrl} width={28} />
            <span>
              <strong>Eqiora</strong>
              <small>Studio</small>
            </span>
          </div>
          <div className="app-bar__context">
            <span className="document-name">untitled.eqi</span>
            <fieldset aria-label="Revision lineage" className="history-actions">
              <button
                aria-label="Previous revision"
                disabled={!commandAvailability["history.undo"].enabled}
                onClick={() => executeCommand("history.undo")}
                title={
                  commandAvailability["history.undo"].reason ??
                  "Previous revision (Ctrl/⌘ Z outside an editor)"
                }
                type="button"
              >
                ←
              </button>
              <span>
                {state.revisionIndex < 0 ? 0 : state.revisionIndex + 1}/
                {state.revisionLineage.length}
              </span>
              <button
                aria-label="Next revision"
                disabled={!commandAvailability["history.redo"].enabled}
                onClick={() => executeCommand("history.redo")}
                title={
                  commandAvailability["history.redo"].reason ??
                  "Next revision (Ctrl/⌘ Shift Z outside an editor)"
                }
                type="button"
              >
                →
              </button>
            </fieldset>
            <nav aria-label="Workspace view" className="workspace-switcher">
              <button
                aria-current={activeWorkspace === "relations" ? "page" : undefined}
                onClick={() => executeCommand("workspace.relations")}
                type="button"
              >
                Relations
              </button>
              {commandAvailability["workspace.geometry"].enabled ? (
                <button
                  aria-current={activeWorkspace === "geometry" ? "page" : undefined}
                  onClick={() => executeCommand("workspace.geometry")}
                  type="button"
                >
                  Geometry
                </button>
              ) : null}
            </nav>
          </div>
          <div className="app-bar__actions">
            <button
              className="secondary-action command-trigger"
              onClick={() => dispatch({ type: "command-palette-opened" })}
              type="button"
            >
              <Icon name="command" />
              Commands
            </button>
            <button
              className="secondary-action"
              disabled={!commandAvailability["model.compile"].enabled}
              onClick={() => executeCommand("model.compile")}
              type="button"
            >
              <Icon name="compile" />
              {state.compileStatus.kind === "compiling" ? "Compiling…" : "Compile model"}
            </button>
          </div>
        </header>
        <div className="preview-banner" role="status">
          The browser Studio demonstrates interaction and layout with fixed example projections. It
          does not compile or execute models.
        </div>
        <main
          className="geometry-workspace-shell"
          hidden={activeWorkspace !== "geometry"}
          id={activeWorkspace === "geometry" ? "workspace" : undefined}
          tabIndex={-1}
        >
          {cadProjection === null ? (
            <section className="geometry-workspace-empty">
              <h1>{cadStatus === "loading" ? "Replaying CAD evidence…" : "No CAD plan"}</h1>
              <button
                className="primary-action"
                onClick={() => executeCommand("example.cad")}
                type="button"
              >
                Open CAD example
              </button>
            </section>
          ) : (
            <>
              {cadError === null ? null : (
                <div className="cad-error" role="alert">
                  {cadError}
                </div>
              )}
              <CadWorkspace
                onRequestSelection={(request) => void requestCadSelection(request)}
                projection={cadProjection}
                selection={cadSelectionState.accepted}
                selectionPending={cadSelectionState.status.kind === "resolving"}
              />
            </>
          )}
        </main>
        <main
          className="workspace"
          hidden={activeWorkspace !== "relations"}
          id={activeWorkspace === "relations" ? "workspace" : undefined}
          tabIndex={-1}
        >
          <aside className="workspace__left">
            <SourceEditor
              ref={sourceEditor}
              onChange={(source) => dispatch({ type: "source-edited", source })}
              onCompile={() => executeCommand("model.compile")}
              source={state.source}
              ancestor={sourceAncestor}
              edited={sourceEdited}
            />
            <ModelOutline
              document={state.document}
              onSelect={(nodeId) => dispatch({ type: "node-selected", nodeId })}
              selectedNodeId={state.selectedNodeId}
            />
          </aside>
          <section ref={relationView} className="canvas-panel" tabIndex={-1}>
            <div className="canvas-panel__heading">
              <div>
                <span className="eyebrow">Semantic projection</span>
                <h1>Relation view</h1>
              </div>
              <button
                className="icon-action"
                disabled={!commandAvailability["view.reflow"].enabled}
                onClick={() => executeCommand("view.reflow")}
                type="button"
              >
                <Icon name="reset" />
                Reflow view
              </button>
            </div>
            <div className="canvas-stage">
              {state.document === null ? (
                <div className="canvas-empty">
                  <h2>A validated model appears here</h2>
                </div>
              ) : (
                <ModelCanvas
                  document={state.document}
                  layout={layout}
                  onMove={(nodeId, position) => dispatch({ type: "node-moved", nodeId, position })}
                  onSelect={(nodeId) => dispatch({ type: "node-selected", nodeId })}
                  selectedNodeId={state.selectedNodeId}
                />
              )}
            </div>
          </section>
          <aside
            className={
              valueEditExpanded
                ? "workspace__right workspace__right--value-edit"
                : "workspace__right"
            }
          >
            <Inspector
              node={selectedNode}
              onNudge={(delta) => {
                if (selectedNode === null) return;
                const position = layout[selectedNode.id] ?? { x: 0, y: 0 };
                dispatch({
                  type: "node-moved",
                  nodeId: selectedNode.id,
                  position: { x: position.x + delta.x, y: position.y + delta.y },
                });
              }}
              position={selectedNode === null ? null : (layout[selectedNode.id] ?? null)}
              valueEdit={{
                input: state.valueEditInput,
                validation: valueValidation,
                status: state.valueEditStatus,
                disabledReason: valueEditDisabledReason,
                onChange: (value) => dispatch({ type: "value-edit-input-edited", value }),
                onCommit: () => executeCommand("edit.commit"),
              }}
            />
          </aside>
          <section className="workspace__bottom">
            <Diagnostics
              diagnostics={diagnostics}
              onNavigate={(diagnostic) => {
                if (diagnostic.span !== null) revealSourceSpan(state.source, diagnostic.span);
              }}
            />
          </section>
        </main>
      </div>
    </>
  );
}
