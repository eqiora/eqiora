import type { BinaryInput, SceneMetadata } from "./contract";

function f64(values: readonly number[]): Uint8Array {
	const bytes = new Uint8Array(values.length * 8);
	const view = new DataView(bytes.buffer);
	values.forEach((value, index) => {
		view.setFloat64(index * 8, value, true);
	});
	return bytes;
}

function u32(values: readonly number[]): Uint8Array {
	const bytes = new Uint8Array(values.length * 4);
	const view = new DataView(bytes.buffer);
	values.forEach((value, index) => {
		view.setUint32(index * 4, value, true);
	});
	return bytes;
}

export interface ViewerFixture {
	readonly metadataJson: string;
	readonly buffers: readonly BinaryInput[];
}

export function viewerFixture(): ViewerFixture {
	const buffers = [
		f64([0, 0, 0, 1, 1, 0, 1, 1]),
		u32([0, 1, 2, 3, 0, 2, 1, 3]),
		u32([0, 1, 2, 3]),
		f64([0, 0, 0, 1, 1, 0, 1, 1]),
		u32([0, 2, 3, 0, 3, 1]),
		u32([0]),
		u32([0, 1]),
		f64([0, 1, 2, 3]),
		f64([10, 20]),
	];
	const shapes = [[4, 2], [4, 2], [4], [4, 2], [2, 3], [1], [1, 2], [4], [2]];
	const scalarTypes = [
		"float64-le",
		"uint32-le",
		"uint32-le",
		"float64-le",
		"uint32-le",
		"uint32-le",
		"uint32-le",
		"float64-le",
		"float64-le",
	] as const;
	const metadata: SceneMetadata = {
		schema: "eqiora.viewer.scene/v0-private",
		buffers: buffers.map((buffer, index) => ({
			index,
			role: `fixture:${index}`,
			scalar_type: scalarTypes[index],
			shape: shapes[index],
			byte_length: buffer.byteLength,
			sha256: "0".repeat(64),
		})),
		layers: [
			{
				kind: "geometry",
				id: "geometry:g",
				owner_digest: "g".repeat(64),
				dimension: 2,
				projection: "exact-axis-aligned-segments/v0",
				positions: { buffer: 0 },
				segments: { buffer: 1 },
				source_entities: { buffer: 2 },
			},
			{
				kind: "mesh",
				id: "mesh:m",
				owner_digest: "m".repeat(64),
				source_digest: "g".repeat(64),
				correspondence_digest: "c".repeat(64),
				dimension: 2,
				cell_kind: "triangle",
				presentation_policy: "exact-triangle-connectivity/v0",
				vertex_count: 4,
				cell_count: 2,
				coordinates: { buffer: 3 },
				connectivity: { buffer: 4 },
			},
			{
				kind: "selection",
				id: "selection:mesh:m:left",
				target_layer: "mesh:m",
				owner_digest: "m".repeat(64),
				correspondence_digest: "c".repeat(64),
				name: "left",
				dimension: 1,
				available: true,
				unavailable_reason: null,
				entity_indices: { buffer: 5 },
				connectivity: { buffer: 6 },
			},
			{
				kind: "scalar-field",
				id: "scalar-field:vertex",
				target_layer: "mesh:m",
				mesh_digest: "m".repeat(64),
				model_digest: "d".repeat(64),
				field_id: "pressure",
				observation_digest: null,
				operator: null,
				association: "vertex",
				component_shape: [],
				unit: "coherent-si",
				dimension: [
					[1, 1],
					[-1, 1],
					[-2, 1],
					[0, 1],
					[0, 1],
					[0, 1],
					[0, 1],
				],
				frame: "scalar",
				space: "continuous-lagrange-p1",
				values: { buffer: 7 },
				scale: {
					provenance: "presentation-linear-range-from-accepted-values/v0",
					minimum: 0,
					maximum: 3,
				},
			},
			{
				kind: "scalar-field",
				id: "scalar-field:cell",
				target_layer: "mesh:m",
				mesh_digest: "m".repeat(64),
				model_digest: "d".repeat(64),
				field_id: "temperature",
				observation_digest: null,
				operator: null,
				association: "cell",
				component_shape: [],
				unit: "coherent-si",
				dimension: [
					[0, 1],
					[0, 1],
					[0, 1],
					[0, 1],
					[1, 1],
					[0, 1],
					[0, 1],
				],
				frame: "scalar",
				space: "cell-constant",
				values: { buffer: 8 },
				scale: {
					provenance: "presentation-linear-range-from-accepted-values/v0",
					minimum: 10,
					maximum: 20,
				},
			},
		],
		presentation: { camera: "disposable", state_is_scientific: false },
		reserved_layer_kinds: ["vector-field", "tensor-field", "trajectory"],
	};
	return { metadataJson: JSON.stringify(metadata), buffers };
}

export function encodeU32(values: readonly number[]): Uint8Array {
	return u32(values);
}

export function encodeF64(values: readonly number[]): Uint8Array {
	return f64(values);
}
