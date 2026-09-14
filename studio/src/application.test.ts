import { describe, expect, test } from "vitest";
import {
  COMMAND_REGISTRY,
  resolveApplication,
  resolveCommandAvailability,
  WORKFLOW_REGISTRY,
} from "./application";
import { BRIDGE_PROTOCOL, type DocumentProjection } from "./protocol";

const document: DocumentProjection = {
  protocol: BRIDGE_PROTOCOL,
  digest: "0123456789abcdef",
  revision: 1,
  modelId: "Model:test",
  nodes: [],
  edges: [],
};
const inputs = {
  acceptedProjection: document,
  cad: { status: "ready" as const, acceptedModelDigest: document.digest },
};
describe("browser Studio application registry", () => {
  test("contains only fixed relation and CAD projections", () =>
    expect(WORKFLOW_REGISTRY.map((item) => item.id)).toEqual(["relations", "cad-box"]));

  test("contains no execution or authored-CAD commands", () => {
    const commands: readonly string[] = COMMAND_REGISTRY.map((item) => item.id);
    expect(commands).not.toContain("run.execute");
    expect(commands).not.toContain("example.dc-drive");
    expect(commands).not.toContain("workspace.cad-authoring");
  });

  test("binds fixed CAD availability to the preview digest", () => {
    expect(resolveApplication(inputs, "geometry").workspace).toBe("geometry");
    expect(
      resolveApplication(
        { ...inputs, cad: { status: "ready", acceptedModelDigest: "foreign-digest-00" } },
        "geometry",
      ).workspace,
    ).toBe("relations");
  });

  test("retains browser interaction commands", () => {
    const availability = resolveCommandAvailability({
      activeWorkflow: "relations",
      compiling: false,
      documentAccepted: true,
      valueEditReady: false,
      valueEditBlock: null,
      revisionNavigationBlocked: false,
      canUndo: false,
      canRedo: false,
      selectedEntity: false,
      cadAvailability: { kind: "available", reason: null },
    });
    expect(availability["model.compile"].enabled).toBe(true);
    expect(availability["workspace.geometry"].enabled).toBe(true);
  });
});
