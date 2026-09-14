import { checkedRequest, protocolFailure } from "./bridge-contract";
import { type CompileRequestV2, compileRequestV2Schema } from "./control-protocol";
import { CAD_EXAMPLE_SOURCE, CAD_PREVIEW_MODEL_DIGEST, EXAMPLE_SOURCE } from "./example";
import { BRIDGE_PROTOCOL, type BridgeEnvelope, type DocumentProjection } from "./protocol";
import {
  type ValueEditCommitRequest,
  type ValueEditPlan,
  type ValueEditPreviewRequest,
  type ValueEditResult,
  valueEditCommitRequestSchema,
  valueEditPreviewRequestSchema,
} from "./value-edit-protocol";

export type StudioExample = "decay" | "cad";

export interface StudioBridge {
  compile(request: CompileRequestV2): Promise<BridgeEnvelope<DocumentProjection>>;
  loadReadOnlyExample(
    example: StudioExample,
    request: CompileRequestV2,
  ): Promise<BridgeEnvelope<DocumentProjection>>;
  previewValueEdit(request: ValueEditPreviewRequest): Promise<BridgeEnvelope<ValueEditPlan>>;
  commitValueEdit(request: ValueEditCommitRequest): Promise<BridgeEnvelope<ValueEditResult>>;
}

function exampleSource(example: StudioExample): string {
  switch (example) {
    case "decay":
      return EXAMPLE_SOURCE;
    case "cad":
      return CAD_EXAMPLE_SOURCE;
  }
}

function exampleRequestMatchesSource(example: StudioExample, request: CompileRequestV2): boolean {
  return request.source === exampleSource(example);
}

const PREVIEW_DIGEST = "preview-4b6ec236856d4bf394168dbac7f5851b";

const previewDocument: DocumentProjection = {
  protocol: BRIDGE_PROTOCOL,
  digest: PREVIEW_DIGEST,
  revision: 1,
  modelId: "Model:01J8EQIORASTUDIOPREVIEW000",
  nodes: [
    {
      id: "Field:state",
      name: "state",
      kind: "field",
      summary: "Continuous state constrained by an initial equation",
      dimension: "1",
      value: null,
    },
    {
      id: "Parameter:rate",
      name: "rate",
      kind: "parameter",
      summary: "Canonical model parameter",
      dimension: "T^-1",
      value: 0.8,
    },
    {
      id: "Relation:state_initial",
      name: "initial state",
      kind: "relation",
      summary: "1 initial equation",
      dimension: null,
      value: null,
    },
    {
      id: "Relation:decay",
      name: "decay",
      kind: "relation",
      summary: "1 implicit residual · 5 expression operations",
      dimension: null,
      value: null,
    },
    {
      id: "Activation:decay",
      name: "decay activation",
      kind: "activation",
      summary: "Continuous activation",
      dimension: null,
      value: null,
    },
  ],
  edges: [
    {
      id: "Relation:state_initial→Field:state:depends-on",
      source: "Relation:state_initial",
      target: "Field:state",
      kind: "depends-on",
      label: "depends on",
    },
    {
      id: "Relation:decay→Field:state:depends-on",
      source: "Relation:decay",
      target: "Field:state",
      kind: "depends-on",
      label: "depends on",
    },
    {
      id: "Relation:decay→Parameter:rate:depends-on",
      source: "Relation:decay",
      target: "Parameter:rate",
      kind: "depends-on",
      label: "depends on",
    },
    {
      id: "Activation:decay→Relation:decay:activates",
      source: "Activation:decay",
      target: "Relation:decay",
      kind: "activates",
      label: "activates",
    },
  ],
};

const previewCadDocument: DocumentProjection = {
  protocol: BRIDGE_PROTOCOL,
  digest: CAD_PREVIEW_MODEL_DIGEST,
  revision: 1,
  modelId: "Model:01J8EQIORACADPREVIEW00000",
  nodes: [
    {
      id: "Domain:body",
      name: "body",
      kind: "domain",
      summary: "3D Cartesian body realized by the exact CAD plan",
      dimension: null,
      value: null,
    },
    ...(["x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper"] as const).map(
      (name) => ({
        id: `Domain:${name}`,
        name,
        kind: "domain" as const,
        summary: "Semantic boundary retained independently of CAD face order",
        dimension: null,
        value: null,
      }),
    ),
    {
      id: "Representation:geometry_space",
      name: "body continuum",
      kind: "representation",
      summary: "Continuous field representation",
      dimension: null,
      value: null,
    },
    {
      id: "Field:marker",
      name: "marker",
      kind: "field",
      summary: "Scalar variable projected through the selected physical boundary",
      dimension: "1",
      value: null,
    },
    {
      id: "Relation:marker_initial",
      name: "initial marker",
      kind: "relation",
      summary: "1 initial equation",
      dimension: null,
      value: null,
    },
    {
      id: "Relation:selected_boundary",
      name: "selected_boundary",
      kind: "relation",
      summary: "Physical boundary relation on x_upper",
      dimension: null,
      value: null,
    },
  ],
  edges: [
    ...(["x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper"] as const).map(
      (name) => ({
        id: `Domain:${name}→Domain:body:boundary-of`,
        source: `Domain:${name}`,
        target: "Domain:body",
        kind: "boundary-of",
        label: "boundary of",
      }),
    ),
    {
      id: "Field:marker→Domain:body:defined-on",
      source: "Field:marker",
      target: "Domain:body",
      kind: "defined-on",
      label: "defined on",
    },
    {
      id: "Field:marker→Representation:geometry_space:represented-by",
      source: "Field:marker",
      target: "Representation:geometry_space",
      kind: "represented-by",
      label: "represented by",
    },
    {
      id: "Relation:marker_initial→Field:marker:depends-on",
      source: "Relation:marker_initial",
      target: "Field:marker",
      kind: "depends-on",
      label: "depends on",
    },
    {
      id: "Relation:selected_boundary→Domain:x_upper:applies-on",
      source: "Relation:selected_boundary",
      target: "Domain:x_upper",
      kind: "applies-on",
      label: "applies on",
    },
  ],
};

const previewDocuments = new Map<string, DocumentProjection>([
  [previewDocument.digest, previewDocument],
]);
const MAX_PREVIEW_DOCUMENTS = 32;
const previewLineage: string[] = [previewDocument.digest];

function resetPreviewLineage(document: DocumentProjection) {
  previewDocuments.clear();
  previewDocuments.set(document.digest, document);
  previewLineage.splice(0, previewLineage.length, document.digest);
}

function retainPreviewChild(baseDigest: string, child: DocumentProjection): boolean {
  const baseIndex = previewLineage.indexOf(baseDigest);
  if (baseIndex < 0) return false;
  for (const abandoned of previewLineage.splice(baseIndex + 1)) {
    previewDocuments.delete(abandoned);
  }
  previewLineage.push(child.digest);
  previewDocuments.set(child.digest, child);
  while (previewLineage.length > MAX_PREVIEW_DOCUMENTS) {
    const oldest = previewLineage.shift();
    if (oldest !== undefined) previewDocuments.delete(oldest);
  }
  return true;
}

function previewFingerprint(input: string): string {
  let state = 2_166_136_261;
  for (const byte of new TextEncoder().encode(input)) {
    state ^= byte;
    state = Math.imul(state, 16_777_619) >>> 0;
  }
  return state.toString(16).padStart(8, "0").repeat(8);
}

function previewValuePlan(
  document: DocumentProjection,
  request: ValueEditPreviewRequest,
): ValueEditPlan | null {
  const node = document.nodes.find((candidate) => candidate.id === request.targetId);
  if (
    node === undefined ||
    node.kind !== "parameter" ||
    node.value === null ||
    node.dimension === null ||
    node.value === request.value
  ) {
    return null;
  }
  const transactionDigest = previewFingerprint(
    `${request.digest}\0${request.targetId}\0${request.value.toString()}`,
  );
  return {
    protocol: BRIDGE_PROTOCOL,
    key: `eqiora.preview-value-edit-plan/v1:${transactionDigest}`,
    baseDigest: request.digest,
    baseRevision: document.revision,
    targetId: request.targetId,
    before: { value: node.value, dimension: node.dimension },
    after: { value: request.value, dimension: node.dimension },
    transactionDigest,
  };
}

export const studioBridge: StudioBridge = {
  async compile(request) {
    const checked = checkedRequest(compileRequestV2Schema, request, "Compile/check");
    if (!checked.ok) {
      return checked.failure;
    }
    await Promise.resolve();
    return {
      protocol: BRIDGE_PROTOCOL,
      result: null,
      diagnostics: [
        {
          source: "studio",
          severity: "error",
          code: "STPREVIEW",
          message:
            "Browser preview cannot compile source. Open a read-only example to inspect the browser projection.",
          graphPath: null,
          span: null,
        },
      ],
    };
  },
  async loadReadOnlyExample(example, request) {
    const checked = checkedRequest(compileRequestV2Schema, request, "Read-only example");
    if (!checked.ok) {
      return checked.failure;
    }
    if (!exampleRequestMatchesSource(example, checked.value)) {
      return protocolFailure("Read-only example identity does not match its immutable source.");
    }
    await Promise.resolve();
    const document = example === "cad" ? previewCadDocument : previewDocument;
    resetPreviewLineage(document);
    return { protocol: BRIDGE_PROTOCOL, result: document, diagnostics: [] };
  },
  async previewValueEdit(request) {
    const checked = checkedRequest(valueEditPreviewRequestSchema, request, "Value edit preview");
    if (!checked.ok) {
      return checked.failure;
    }
    const document = previewDocuments.get(checked.value.digest);
    if (document === undefined) {
      return protocolFailure("Value-edit base revision is not available in the browser preview.");
    }
    const plan = previewValuePlan(document, checked.value);
    if (plan === null) {
      return protocolFailure("Select a Parameter and enter a different finite value.");
    }
    return { protocol: BRIDGE_PROTOCOL, result: plan, diagnostics: [] };
  },
  async commitValueEdit(request) {
    const checked = checkedRequest(valueEditCommitRequestSchema, request, "Value edit commit");
    if (!checked.ok) {
      return checked.failure;
    }
    const document = previewDocuments.get(checked.value.digest);
    if (document === undefined) {
      return protocolFailure("Value-edit base revision is not available in the browser preview.");
    }
    const plan = previewValuePlan(document, checked.value);
    if (plan === null || plan.key !== checked.value.planKey) {
      return protocolFailure("Value edit no longer matches the browser preview.");
    }
    const resultDigest = `preview-${previewFingerprint(`${plan.key}\0child`)}`;
    const child: DocumentProjection = {
      ...document,
      digest: resultDigest,
      revision: document.revision + 1,
      nodes: document.nodes.map((node) =>
        node.id === plan.targetId ? { ...node, value: plan.after.value } : node,
      ),
    };
    if (!retainPreviewChild(document.digest, child)) {
      return protocolFailure("Value-edit base revision left the browser preview lineage.");
    }
    return {
      protocol: BRIDGE_PROTOCOL,
      result: {
        protocol: BRIDGE_PROTOCOL,
        document: child,
        evidence: {
          plan,
          resultDigest,
          resultRevision: child.revision,
        },
      },
      diagnostics: [],
    };
  },
};
