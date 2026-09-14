export const PRIVATE_SCENE_SCHEMA = "eqiora.viewer.scene/v0-private";

export type ScalarType = "float64-le" | "uint32-le";

export interface BufferDescriptor {
	readonly index: number;
	readonly role: string;
	readonly scalar_type: ScalarType;
	readonly shape: readonly number[];
	readonly byte_length: number;
	readonly sha256: string;
}

interface BufferReference {
	readonly buffer: number;
}

export interface GeometryLayer {
	readonly kind: "geometry";
	readonly id: string;
	readonly owner_digest: string;
	readonly dimension: 2;
	readonly projection: string;
	readonly positions: BufferReference;
	readonly segments: BufferReference;
	readonly source_entities: BufferReference;
}

export interface MeshLayer {
	readonly kind: "mesh";
	readonly id: string;
	readonly owner_digest: string;
	readonly source_digest: string;
	readonly correspondence_digest: string;
	readonly dimension: 2;
	readonly cell_kind: "triangle" | "quadrilateral";
	readonly presentation_policy: string;
	readonly vertex_count: number;
	readonly cell_count: number;
	readonly coordinates: BufferReference;
	readonly connectivity: BufferReference;
}

export interface SelectionLayer {
	readonly kind: "selection";
	readonly id: string;
	readonly target_layer: string;
	readonly owner_digest: string;
	readonly correspondence_digest: string | null;
	readonly name: string;
	readonly dimension: number;
	readonly available: boolean;
	readonly unavailable_reason: string | null;
	readonly entity_indices: BufferReference | null;
	readonly connectivity: BufferReference | null;
}

export interface ScalarFieldLayer {
	readonly kind: "scalar-field";
	readonly id: string;
	readonly target_layer: string;
	readonly mesh_digest: string;
	readonly model_digest: string;
	readonly field_id: string;
	readonly observation_digest: string | null;
	readonly operator: "curl" | null;
	readonly association: "vertex" | "cell";
	readonly component_shape: readonly [];
	readonly unit: "coherent-si";
	readonly dimension: readonly [
		readonly [number, number],
		readonly [number, number],
		readonly [number, number],
		readonly [number, number],
		readonly [number, number],
		readonly [number, number],
		readonly [number, number],
	];
	readonly frame: "scalar" | "spatial-axial";
	readonly space: string;
	readonly values: BufferReference;
	readonly scale: {
		readonly provenance: string;
		readonly minimum: number;
		readonly maximum: number;
	};
}

export type SceneLayer = GeometryLayer | MeshLayer | SelectionLayer | ScalarFieldLayer;

export interface SceneMetadata {
	readonly schema: typeof PRIVATE_SCENE_SCHEMA;
	readonly layers: readonly SceneLayer[];
	readonly buffers: readonly BufferDescriptor[];
	readonly presentation: {
		readonly camera: "disposable";
		readonly state_is_scientific: false;
	};
	readonly reserved_layer_kinds: readonly [
		"vector-field",
		"tensor-field",
		"trajectory",
	];
}

export type BinaryInput = ArrayBuffer | ArrayBufferView;

export interface DecodedScene {
	readonly metadata: SceneMetadata;
	readonly buffers: readonly (Float64Array | Uint32Array)[];
}

function record(value: unknown, label: string): Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new Error(`${label} must be an object`);
	}
	return value as Record<string, unknown>;
}

function text(value: unknown, label: string): string {
	if (typeof value !== "string" || value.length === 0) {
		throw new Error(`${label} must be a non-empty string`);
	}
	return value;
}

function integer(value: unknown, label: string): number {
	if (!Number.isSafeInteger(value) || (value as number) < 0) {
		throw new Error(`${label} must be a non-negative safe integer`);
	}
	return value as number;
}

function binaryBytes(value: BinaryInput): Uint8Array {
	if (value instanceof ArrayBuffer) {
		return new Uint8Array(value.slice(0));
	}
	return new Uint8Array(
		value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength),
	);
}

function decodeBuffer(
	descriptor: BufferDescriptor,
	input: BinaryInput,
): Float64Array | Uint32Array {
	const bytes = binaryBytes(input);
	if (bytes.byteLength !== descriptor.byte_length) {
		throw new Error(`buffer ${descriptor.index} byte length differs from metadata`);
	}
	const scalarBytes = descriptor.scalar_type === "float64-le" ? 8 : 4;
	const expected =
		descriptor.shape.reduce((total, size) => total * size, 1) * scalarBytes;
	if (expected !== bytes.byteLength) {
		throw new Error(`buffer ${descriptor.index} shape differs from byte length`);
	}
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	if (descriptor.scalar_type === "float64-le") {
		const values = new Float64Array(bytes.byteLength / 8);
		for (let index = 0; index < values.length; index += 1) {
			values[index] = view.getFloat64(index * 8, true);
			if (!Number.isFinite(values[index])) {
				throw new Error(`buffer ${descriptor.index} contains a non-finite float`);
			}
		}
		return values;
	}
	const values = new Uint32Array(bytes.byteLength / 4);
	for (let index = 0; index < values.length; index += 1) {
		values[index] = view.getUint32(index * 4, true);
	}
	return values;
}

function shapeOf(metadata: SceneMetadata, value: BufferReference): readonly number[] {
	const descriptor = metadata.buffers[value.buffer];
	if (descriptor === undefined || descriptor.index !== value.buffer) {
		throw new Error(`layer references absent buffer ${value.buffer}`);
	}
	return descriptor.shape;
}

function expectShape(
	metadata: SceneMetadata,
	value: BufferReference,
	expected: readonly number[],
	label: string,
): void {
	const observed = shapeOf(metadata, value);
	if (
		observed.length !== expected.length ||
		observed.some((size, index) => size !== expected[index])
	) {
		throw new Error(
			`${label} has shape [${observed.join(",")}], expected [${expected.join(",")}]`,
		);
	}
}

function parseMetadata(metadataJson: string): SceneMetadata {
	const root = record(JSON.parse(metadataJson), "scene metadata");
	if (
		root.schema !== PRIVATE_SCENE_SCHEMA ||
		!Array.isArray(root.layers) ||
		!Array.isArray(root.buffers)
	) {
		throw new Error("unsupported private Eqiora viewer scene");
	}
	const descriptors = root.buffers.map((value, index): BufferDescriptor => {
		const item = record(value, `buffers[${index}]`);
		const scalarType = item.scalar_type;
		if (scalarType !== "float64-le" && scalarType !== "uint32-le") {
			throw new Error(`buffers[${index}] has unsupported scalar type`);
		}
		if (!Array.isArray(item.shape) || item.shape.length === 0) {
			throw new Error(`buffers[${index}] has invalid shape`);
		}
		const shape = item.shape.map((size, axis) => {
			const parsed = integer(size, `buffers[${index}].shape[${axis}]`);
			if (parsed === 0) throw new Error(`buffers[${index}] has an empty axis`);
			return parsed;
		});
		const digest = text(item.sha256, `buffers[${index}].sha256`);
		if (!/^[0-9a-f]{64}$/.test(digest))
			throw new Error(`buffers[${index}] has invalid sha256`);
		return {
			index: integer(item.index, `buffers[${index}].index`),
			role: text(item.role, `buffers[${index}].role`),
			scalar_type: scalarType,
			shape,
			byte_length: integer(item.byte_length, `buffers[${index}].byte_length`),
			sha256: digest,
		};
	});
	if (descriptors.some((descriptor, index) => descriptor.index !== index)) {
		throw new Error("scene buffer indices are not canonical");
	}
	return { ...root, buffers: descriptors } as unknown as SceneMetadata;
}

function validateLayerReferences(scene: DecodedScene): void {
	const { metadata, buffers } = scene;
	const targets = new Map<string, GeometryLayer | MeshLayer>();
	const ids = new Set<string>();
	for (const unknownLayer of metadata.layers) {
		const layer = record(unknownLayer, "layer") as unknown as SceneLayer;
		if (!text(layer.id, "layer.id") || ids.has(layer.id))
			throw new Error("scene repeats a layer id");
		ids.add(layer.id);
		if (layer.kind === "geometry") {
			if (layer.dimension !== 2)
				throw new Error("v0 GeometryLayer must be two-dimensional");
			expectShape(
				metadata,
				layer.positions,
				[shapeOf(metadata, layer.positions)[0], 2],
				"Geometry positions",
			);
			const count = shapeOf(metadata, layer.segments)[0];
			expectShape(metadata, layer.segments, [count, 2], "Geometry segments");
			expectShape(metadata, layer.source_entities, [count], "Geometry source entities");
			const positions = buffers[layer.positions.buffer];
			const segments = buffers[layer.segments.buffer];
			const sourceEntities = buffers[layer.source_entities.buffer];
			if (
				!(positions instanceof Float64Array) ||
				!(segments instanceof Uint32Array) ||
				!(sourceEntities instanceof Uint32Array)
			)
				throw new Error("Geometry buffer types disagree");
			const vertices = positions.length / 2;
			if (segments.some((value) => value >= vertices))
				throw new Error("Geometry segment is out of range");
			targets.set(layer.id, layer);
		} else if (layer.kind === "mesh") {
			const width =
				layer.cell_kind === "triangle"
					? 3
					: layer.cell_kind === "quadrilateral"
						? 4
						: 0;
			if (layer.dimension !== 2 || width === 0)
				throw new Error("v0 MeshLayer has unsupported topology");
			expectShape(
				metadata,
				layer.coordinates,
				[layer.vertex_count, 2],
				"Mesh coordinates",
			);
			expectShape(
				metadata,
				layer.connectivity,
				[layer.cell_count, width],
				"Mesh connectivity",
			);
			const cells = buffers[layer.connectivity.buffer];
			if (
				!(buffers[layer.coordinates.buffer] instanceof Float64Array) ||
				!(cells instanceof Uint32Array)
			)
				throw new Error("Mesh buffer types disagree");
			if (cells.some((value) => value >= layer.vertex_count))
				throw new Error("Mesh connectivity is out of range");
			targets.set(layer.id, layer);
		} else if (layer.kind !== "selection" && layer.kind !== "scalar-field") {
			const unsupported = layer as unknown as Record<string, unknown>;
			throw new Error(
				`private v0 scene has unsupported layer kind ${String(unsupported.kind)}`,
			);
		}
	}
	for (const layer of metadata.layers) {
		if (layer.kind === "selection") {
			const target = targets.get(layer.target_layer);
			if (target === undefined || target.owner_digest !== layer.owner_digest)
				throw new Error("SelectionLayer has a foreign owner");
			if (
				target.kind === "mesh" &&
				target.correspondence_digest !== layer.correspondence_digest
			)
				throw new Error("SelectionLayer has a foreign correspondence");
			if (layer.available) {
				if (layer.unavailable_reason !== null)
					throw new Error("available SelectionLayer has an unavailable reason");
				if (layer.entity_indices === null)
					throw new Error("available SelectionLayer omits membership");
				const indices = buffers[layer.entity_indices.buffer];
				if (!(indices instanceof Uint32Array))
					throw new Error("Selection membership is not uint32");
				const membershipShape = shapeOf(metadata, layer.entity_indices);
				if (membershipShape.length !== 1)
					throw new Error("Selection membership is not one-dimensional");
				const memberCount = membershipShape[0];
				if (target.kind === "geometry") {
					if (layer.dimension !== 1)
						throw new Error("Geometry selection has unsupported dimension");
					if (layer.connectivity !== null)
						throw new Error("Geometry selection fabricates connectivity");
					const primitiveCount = shapeOf(metadata, target.segments)[0];
					if (indices.some((value) => value >= primitiveCount))
						throw new Error("Geometry selection primitive is out of range");
				} else {
					if (layer.dimension !== 1 && layer.dimension !== 2)
						throw new Error("Mesh selection has unsupported dimension");
					if (layer.connectivity === null)
						throw new Error("Mesh selection omits exact connectivity");
					const selected = buffers[layer.connectivity.buffer];
					const width =
						layer.dimension === 1 ? 2 : target.cell_kind === "triangle" ? 3 : 4;
					expectShape(
						metadata,
						layer.connectivity,
						[memberCount, width],
						"Mesh selection connectivity",
					);
					if (
						!(selected instanceof Uint32Array) ||
						selected.some((value) => value >= target.vertex_count)
					)
						throw new Error("Mesh selection connectivity is invalid");
				}
			} else if (
				layer.unavailable_reason === null ||
				layer.entity_indices !== null ||
				layer.connectivity !== null
			) {
				throw new Error("unavailable SelectionLayer is not explicit");
			}
		} else if (layer.kind === "scalar-field") {
			const target = targets.get(layer.target_layer);
			if (target?.kind !== "mesh" || target.owner_digest !== layer.mesh_digest)
				throw new Error("ScalarFieldLayer has a foreign Mesh owner");
			if (layer.component_shape.length !== 0 || layer.unit !== "coherent-si")
				throw new Error("v0 ScalarFieldLayer is not scalar");
			const expected =
				layer.association === "vertex"
					? target.vertex_count
					: layer.association === "cell"
						? target.cell_count
						: 0;
			if (expected === 0)
				throw new Error("ScalarFieldLayer has unsupported association");
			if (layer.observation_digest === null || layer.operator === null) {
				if (
					layer.observation_digest !== null ||
					layer.operator !== null ||
					layer.frame !== "scalar"
				)
					throw new Error("static ScalarFieldLayer has derived identity");
			} else if (
				!/^[0-9a-f]{64}$/.test(layer.observation_digest) ||
				layer.operator !== "curl" ||
				layer.frame !== "spatial-axial" ||
				layer.association !== "cell" ||
				layer.space !== "cell-average"
			) {
				throw new Error("derived ScalarFieldLayer is unsupported or inconsistent");
			}
			expectShape(metadata, layer.values, [expected], "Scalar field values");
			if (!(buffers[layer.values.buffer] instanceof Float64Array))
				throw new Error("Scalar field values are not float64");
		}
	}
}

export function decodeScene(
	metadataJson: string,
	inputBuffers: readonly BinaryInput[],
): DecodedScene {
	const metadata = parseMetadata(metadataJson);
	if (inputBuffers.length !== metadata.buffers.length)
		throw new Error("scene binary buffer count differs from metadata");
	const buffers = metadata.buffers.map((descriptor, index) =>
		decodeBuffer(descriptor, inputBuffers[index]),
	);
	const scene = { metadata, buffers };
	validateLayerReferences(scene);
	return scene;
}
