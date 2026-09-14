import { COMMAND_REGISTRY, WORKFLOW_REGISTRY } from "./application-registry";
import type { MessageKey } from "./messages";
import type { DocumentProjection } from "./protocol";

export { COMMAND_REGISTRY, WORKFLOW_REGISTRY } from "./application-registry";
export type WorkflowId = "relations" | "cad-box";
export type WorkspaceId = "relations" | "geometry";
export type CommandId =
  | "model.compile"
  | "edit.commit"
  | "history.undo"
  | "history.redo"
  | "view.reflow"
  | "workspace.relations"
  | "workspace.geometry"
  | "example.cad"
  | "focus.source"
  | "focus.relation"
  | "focus.inspector";
export type CommandGroup = "model" | "view" | "navigate";
export type FocusTarget =
  | "source-editor"
  | "relation-view"
  | "selection-inspector"
  | "cad-viewport"
  | "cad-domain-table";
export type ElementFocusTarget = Exclude<FocusTarget, "source-editor" | "relation-view">;
export function resolveElementFocusId(target: ElementFocusTarget): string {
  switch (target) {
    case "selection-inspector":
      return "inspector-panel";
    case "cad-viewport":
      return "cad-viewport";
    case "cad-domain-table":
      return "cad-domain-table";
  }
}
export interface CommandDefinition {
  readonly id: CommandId;
  readonly group: CommandGroup;
  readonly label: MessageKey;
  readonly description: MessageKey;
  readonly shortcut: string | null;
  readonly focusTarget: FocusTarget | null;
  readonly workflows: readonly WorkflowId[];
}
export type WorkflowDefinition = Readonly<{
  id: WorkflowId;
  workspace: WorkspaceId;
  label: MessageKey;
  description: MessageKey;
  primaryFocus: FocusTarget;
}>;
export type WorkflowAvailability =
  | Readonly<{ kind: "available"; reason: null }>
  | Readonly<{ kind: "loading" | "unavailable"; reason: MessageKey }>;
export type ResolvedWorkflow = Readonly<{
  definition: WorkflowDefinition;
  availability: WorkflowAvailability;
  commands: readonly CommandDefinition[];
}>;
export type CadApplicationInput = Readonly<{
  status: "idle" | "loading" | "ready" | "unavailable";
  acceptedModelDigest: string | null;
}>;
export type ApplicationInputs = Readonly<{
  acceptedProjection: DocumentProjection | null;
  cad: CadApplicationInput;
}>;
export function resolveApplicationWorkflows(
  inputs: ApplicationInputs,
): readonly ResolvedWorkflow[] {
  return WORKFLOW_REGISTRY.map((definition) => {
    let availability: WorkflowAvailability;
    if (inputs.acceptedProjection === null)
      availability = { kind: "unavailable", reason: "workflow.reason.compile-first" };
    else if (definition.id === "cad-box")
      availability =
        inputs.cad.status === "loading" || inputs.cad.status === "idle"
          ? { kind: "loading", reason: "workflow.reason.cad-loading" }
          : inputs.cad.status === "ready" &&
              inputs.cad.acceptedModelDigest === inputs.acceptedProjection.digest
            ? { kind: "available", reason: null }
            : {
                kind: "unavailable",
                reason:
                  inputs.cad.status === "ready"
                    ? "workflow.reason.cad-stale"
                    : "workflow.reason.cad-unavailable",
              };
    else availability = { kind: "available", reason: null };
    return {
      definition,
      availability,
      commands: COMMAND_REGISTRY.filter((command) =>
        (command.workflows as readonly WorkflowId[]).includes(definition.id),
      ),
    };
  });
}
export function resolveApplication(inputs: ApplicationInputs, requestedWorkspace: WorkspaceId) {
  const workflows = resolveApplicationWorkflows(inputs);
  const cadKind = workflows.find((item) => item.definition.id === "cad-box")?.availability.kind;
  const workspace =
    requestedWorkspace === "geometry" && !["available", "loading"].includes(cadKind ?? "")
      ? "relations"
      : requestedWorkspace;
  return {
    requestedWorkspace,
    workspace,
    activeWorkflow: workspace === "geometry" ? "cad-box" : "relations",
    workflows,
    fellBack: workspace !== requestedWorkspace,
  } as const;
}
export type ValueEditBlock = "source" | null;
export type CommandFacts = Readonly<{
  activeWorkflow: WorkflowId;
  compiling: boolean;
  documentAccepted: boolean;
  valueEditReady: boolean;
  valueEditBlock: ValueEditBlock;
  revisionNavigationBlocked: boolean;
  canUndo: boolean;
  canRedo: boolean;
  selectedEntity: boolean;
  cadAvailability: WorkflowAvailability;
}>;
export type CommandAvailability = Readonly<
  Record<CommandId, Readonly<{ enabled: boolean; reason: MessageKey | null }>>
>;
export function resolveCommandAvailability(facts: CommandFacts): CommandAvailability {
  const state = (enabled: boolean, reason: MessageKey) => ({
    enabled,
    reason: enabled ? null : reason,
  });
  return {
    "model.compile": state(!facts.compiling, "command.reason.compiling"),
    "edit.commit": state(
      facts.valueEditReady && facts.valueEditBlock === null,
      facts.valueEditBlock === "source"
        ? "command.reason.edit-source"
        : "command.reason.edit-preview",
    ),
    "history.undo": state(facts.canUndo, "command.reason.first-revision"),
    "history.redo": state(facts.canRedo, "command.reason.no-child-revision"),
    "view.reflow": state(facts.documentAccepted, "command.reason.compile-first"),
    "workspace.relations": { enabled: true, reason: null },
    "workspace.geometry": state(
      ["available", "loading"].includes(facts.cadAvailability.kind),
      facts.cadAvailability.reason ?? "workflow.reason.cad-unavailable",
    ),
    "example.cad": state(!facts.compiling, "command.reason.compiling"),
    "focus.source": { enabled: true, reason: null },
    "focus.relation": { enabled: true, reason: null },
    "focus.inspector": state(facts.selectedEntity, "command.reason.select-entity"),
  };
}
