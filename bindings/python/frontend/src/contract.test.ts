import { describe, expect, it } from "vitest";
import { decodeScene } from "./contract";
import { encodeF64, encodeU32, viewerFixture } from "./fixture";

function metadata(fixture = viewerFixture()): Record<string, unknown> {
	return JSON.parse(fixture.metadataJson) as Record<string, unknown>;
}

describe("private typed scene admission", () => {
	it("admits composed Geometry, Mesh, Selection, and vertex/cell scalar layers", () => {
		const fixture = viewerFixture();
		const scene = decodeScene(fixture.metadataJson, fixture.buffers);
		expect(scene.metadata.layers.map((layer) => layer.kind)).toEqual([
			"geometry",
			"mesh",
			"selection",
			"scalar-field",
			"scalar-field",
		]);
		expect(scene.buffers[3]).toBeInstanceOf(Float64Array);
		expect(scene.buffers[4]).toBeInstanceOf(Uint32Array);
	});

	it("rejects foreign selection and scalar owners even when shapes match", () => {
		const fixture = viewerFixture();
		const document = metadata(fixture);
		const layers = document.layers as Array<Record<string, unknown>>;
		layers[2].owner_digest = "foreign";
		expect(() => decodeScene(JSON.stringify(document), fixture.buffers)).toThrow(
			"foreign owner",
		);

		layers[2].owner_digest = "m".repeat(64);
		layers[2].correspondence_digest = "foreign";
		expect(() => decodeScene(JSON.stringify(document), fixture.buffers)).toThrow(
			"foreign correspondence",
		);

		layers[2].correspondence_digest = "c".repeat(64);
		layers[3].mesh_digest = "foreign";
		expect(() => decodeScene(JSON.stringify(document), fixture.buffers)).toThrow(
			"foreign Mesh owner",
		);
	});

	it("rejects out-of-range topology and non-finite scalar buffers", () => {
		const fixture = viewerFixture();
		const topology = fixture.buffers.slice();
		topology[4] = encodeU32([0, 2, 9, 0, 3, 1]);
		expect(() => decodeScene(fixture.metadataJson, topology)).toThrow("out of range");

		const incomplete = fixture.buffers.slice();
		incomplete[4] = encodeU32([0, 2, 3]);
		expect(() => decodeScene(fixture.metadataJson, incomplete)).toThrow("byte length");

		const values = fixture.buffers.slice();
		values[7] = encodeF64([0, 1, Number.NaN, 3]);
		expect(() => decodeScene(fixture.metadataJson, values)).toThrow("non-finite");
	});

	it("rejects unsupported associations and fabricated unavailable mappings", () => {
		const fixture = viewerFixture();
		const document = metadata(fixture);
		const layers = document.layers as Array<Record<string, unknown>>;
		layers[3].association = "cell-bubble";
		expect(() => decodeScene(JSON.stringify(document), fixture.buffers)).toThrow(
			"unsupported association",
		);

		layers[3].association = "vertex";
		layers[2].available = false;
		layers[2].unavailable_reason = "not mapped";
		expect(() => decodeScene(JSON.stringify(document), fixture.buffers)).toThrow(
			"not explicit",
		);
	});

	it("admits exact cell-average curl identity and rejects inconsistent derivation", () => {
		const fixture = viewerFixture();
		const document = metadata(fixture);
		const layers = document.layers as Array<Record<string, unknown>>;
		const field = layers[4];
		field.observation_digest = "a".repeat(64);
		field.operator = "curl";
		field.frame = "spatial-axial";
		field.space = "cell-average";
		expect(
			decodeScene(JSON.stringify(document), fixture.buffers).metadata.layers[4],
		).toMatchObject({
			observation_digest: "a".repeat(64),
			operator: "curl",
		});

		field.frame = "scalar";
		expect(() => decodeScene(JSON.stringify(document), fixture.buffers)).toThrow(
			"unsupported or inconsistent",
		);
	});

	it("rejects unknown layers and unsupported available selection dimensions", () => {
		const fixture = viewerFixture();
		const document = metadata(fixture);
		const layers = document.layers as Array<Record<string, unknown>>;
		layers[2].dimension = 0;
		expect(() => decodeScene(JSON.stringify(document), fixture.buffers)).toThrow(
			"unsupported dimension",
		);

		layers[2].dimension = 1;
		layers.push({ kind: "trajectory", id: "not-implemented" });
		expect(() => decodeScene(JSON.stringify(document), fixture.buffers)).toThrow(
			"unsupported layer kind",
		);
	});
});
