import { protocolFailure } from "./bridge-contract";
import {
  CAD_VIEW_PROTOCOL,
  type CadProjection,
  type CadProjectionRequest,
  type CadSelectionRequest,
  type CadSelectionResult,
  cadProjectionRequestSchema,
  cadProjectionSchema,
  cadSelectionRequestSchema,
} from "./cad-protocol";
import { CAD_PREVIEW_MODEL_DIGEST } from "./example";
import { BRIDGE_PROTOCOL, type BridgeEnvelope } from "./protocol";

export interface CadBridge {
  preview(request: CadProjectionRequest): Promise<BridgeEnvelope<CadProjection>>;
  select(request: CadSelectionRequest): Promise<BridgeEnvelope<CadSelectionResult>>;
}

export function cadProjectionMatchesRequest(
  request: Pick<CadProjectionRequest, "modelDigest">,
  projection: Pick<CadProjection, "modelDigest">,
): boolean {
  return projection.modelDigest === request.modelDigest;
}

const roles = [
  [0, "lower"],
  [0, "upper"],
  [1, "lower"],
  [1, "upper"],
  [2, "lower"],
  [2, "upper"],
] as const;
const faceTriangles = [
  [
    [0, 2, 6],
    [0, 6, 4],
  ],
  [
    [1, 5, 7],
    [1, 7, 3],
  ],
  [
    [0, 4, 5],
    [0, 5, 1],
  ],
  [
    [2, 3, 7],
    [2, 7, 6],
  ],
  [
    [0, 1, 3],
    [0, 3, 2],
  ],
  [
    [4, 6, 7],
    [4, 7, 5],
  ],
] as const;

const boundaryDomains = roles.map(([axis, side]) => `Domain:${axis}-${side}`);
const previewProjection = cadProjectionSchema.parse({
  protocol: CAD_VIEW_PROTOCOL,
  planKey: "1".repeat(64),
  modelDigest: CAD_PREVIEW_MODEL_DIGEST,
  geometryDigest: "3".repeat(64),
  meshDigest: "4".repeat(64),
  design: {
    sourceUnit: "millimetre",
    importedStockBoundsM: [
      [-1, 1],
      [-1, 1],
      [-1, 1],
    ],
    sketch: {
      xBoundsM: [-0.5, 0.5],
      yBoundsM: [-0.5, 0.5],
      planeZM: -0.5,
      remainingDegreesOfFreedom: 0,
    },
    extrusion: { direction: "positive-z", depthM: 1 },
    boolean: "intersection",
    resultBoundsM: [
      [-0.5, 0.5],
      [-0.5, 0.5],
      [-0.5, 0.5],
    ],
  },
  build: {
    adapter: "eqiora.cad.truck-box-v1",
    adapterVersion: "0.1.0",
    kernel: "truck-stepio-modeling-topology",
    kernelVersion: "stepio-0.3.0+modeling-0.6.0+topology-0.6.0",
    repair: "none",
    importedStock: { solidCount: 1, closedShellCount: 1, planarFaceCount: 6 },
    extrudedTool: { solidCount: 1, closedShellCount: 1, planarFaceCount: 6 },
    intersection: { solidCount: 1, closedShellCount: 1, planarFaceCount: 6 },
  },
  verticesM: [
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
  ],
  triangles: boundaryDomains.flatMap((domainId, face) =>
    (faceTriangles[face] ?? []).map((vertexIndices) => ({
      domainId,
      vertexIndices: [...vertexIndices],
    })),
  ),
  entities: [
    {
      domainId: "Domain:body",
      name: "body",
      kind: "body",
      parentDomainId: null,
      axis: null,
      side: null,
      meshEntityCount: 6,
      relationIds: [],
      portIds: [],
    },
    ...roles.map(([axis, side]) => ({
      domainId: `Domain:${axis}-${side}`,
      name: `${["x", "y", "z"][axis]}_${side}`,
      kind: "boundary" as const,
      parentDomainId: "Domain:body",
      axis,
      side,
      meshEntityCount: 2,
      relationIds: axis === 0 && side === "upper" ? ["Relation:selected_boundary"] : [],
      portIds: [],
    })),
  ],
});

export const cadBridge: CadBridge = {
  async preview(request) {
    const checked = cadProjectionRequestSchema.safeParse(request);
    if (!checked.success || checked.data.modelDigest !== previewProjection.modelDigest) {
      return protocolFailure("The bounded CAD workflow is unavailable for this preview Model.");
    }
    await Promise.resolve();
    return { protocol: BRIDGE_PROTOCOL, result: previewProjection, diagnostics: [] };
  },

  async select(request) {
    const checked = cadSelectionRequestSchema.safeParse(request);
    if (!checked.success) return protocolFailure("CAD selection request is not canonical.");
    if (
      checked.data.modelDigest !== previewProjection.modelDigest ||
      checked.data.planKey !== previewProjection.planKey ||
      checked.data.geometryDigest !== previewProjection.geometryDigest
    ) {
      return protocolFailure("CAD selection references a stale browser-preview revision.");
    }
    const entity = previewProjection.entities.find(
      (candidate) => candidate.domainId === checked.data.domainId,
    );
    if (entity === undefined) return protocolFailure("CAD selection Domain is unavailable.");
    const result: CadSelectionResult = {
      protocol: CAD_VIEW_PROTOCOL,
      modelDigest: checked.data.modelDigest,
      planKey: checked.data.planKey,
      geometryDigest: checked.data.geometryDigest,
      domainId: checked.data.domainId,
      entity,
    };
    return { protocol: BRIDGE_PROTOCOL, result, diagnostics: [] };
  },
};
