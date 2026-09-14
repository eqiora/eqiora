use std::collections::{BTreeMap, BTreeSet};

use eqiora::Diagnostic;
use eqiora::diagnostic::codes;
use serde::Serialize;
use sha2::{Digest, Sha256};

pub(super) const PRIVATE_SCENE_SCHEMA: &str = "eqiora.viewer.scene/v0-private";
const MAX_SCENE_LAYERS: usize = 256;
const MAX_SCENE_BUFFER_BYTES: usize = 512 * 1024 * 1024;

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum ScalarType {
    Float64Le,
    Uint32Le,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct BufferRef {
    pub(super) buffer: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct BufferDescriptor {
    index: usize,
    role: String,
    scalar_type: ScalarType,
    shape: Vec<usize>,
    byte_length: usize,
    sha256: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub(super) enum LayerMetadata {
    Geometry(GeometryLayer),
    Mesh(MeshLayer),
    Selection(SelectionLayer),
    ScalarField(ScalarFieldLayer),
}

impl LayerMetadata {
    fn id(&self) -> &str {
        match self {
            Self::Geometry(layer) => &layer.id,
            Self::Mesh(layer) => &layer.id,
            Self::Selection(layer) => &layer.id,
            Self::ScalarField(layer) => &layer.id,
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct GeometryLayer {
    pub(super) id: String,
    pub(super) owner_digest: String,
    pub(super) dimension: usize,
    pub(super) projection: String,
    pub(super) positions: BufferRef,
    pub(super) segments: BufferRef,
    pub(super) source_entities: BufferRef,
}

#[derive(Debug, Serialize)]
pub(super) struct MeshLayer {
    pub(super) id: String,
    pub(super) owner_digest: String,
    pub(super) source_digest: String,
    pub(super) correspondence_digest: String,
    pub(super) dimension: usize,
    pub(super) cell_kind: String,
    pub(super) presentation_policy: String,
    pub(super) vertex_count: usize,
    pub(super) cell_count: usize,
    pub(super) coordinates: BufferRef,
    pub(super) connectivity: BufferRef,
}

#[derive(Debug, Serialize)]
pub(super) struct SelectionLayer {
    pub(super) id: String,
    pub(super) target_layer: String,
    pub(super) owner_digest: String,
    pub(super) correspondence_digest: Option<String>,
    pub(super) name: String,
    pub(super) dimension: usize,
    pub(super) available: bool,
    pub(super) unavailable_reason: Option<String>,
    pub(super) entity_indices: Option<BufferRef>,
    pub(super) connectivity: Option<BufferRef>,
}

#[derive(Debug, Serialize)]
pub(super) struct ScalarFieldLayer {
    pub(super) id: String,
    pub(super) target_layer: String,
    pub(super) mesh_digest: String,
    pub(super) model_digest: String,
    pub(super) field_id: String,
    pub(super) observation_digest: Option<String>,
    pub(super) operator: Option<String>,
    pub(super) association: String,
    pub(super) component_shape: Vec<usize>,
    pub(super) unit: String,
    pub(super) dimension: [(i32, i32); 7],
    pub(super) frame: String,
    pub(super) space: String,
    pub(super) values: BufferRef,
    pub(super) scale: PresentationScale,
}

#[derive(Debug, Serialize)]
pub(super) struct PresentationScale {
    pub(super) provenance: String,
    pub(super) minimum: f64,
    pub(super) maximum: f64,
}

#[derive(Debug, Serialize)]
struct SceneMetadata<'a> {
    schema: &'static str,
    layers: &'a [LayerMetadata],
    buffers: &'a [BufferDescriptor],
    presentation: ScenePresentation,
    reserved_layer_kinds: [&'static str; 3],
}

#[derive(Debug, Serialize)]
struct ScenePresentation {
    camera: &'static str,
    state_is_scientific: bool,
}

#[derive(Clone, Debug)]
pub(super) struct MeshTarget {
    pub(super) layer_id: String,
    pub(super) vertex_count: usize,
    pub(super) cell_count: usize,
}

#[derive(Debug)]
pub(super) struct FinishedScene {
    pub(super) metadata_json: String,
    pub(super) buffers: Vec<Vec<u8>>,
    pub(super) layer_count: usize,
}

#[derive(Debug, Default)]
pub(super) struct SceneBuilder {
    layers: Vec<LayerMetadata>,
    layer_ids: BTreeSet<String>,
    descriptors: Vec<BufferDescriptor>,
    buffers: Vec<Vec<u8>>,
    mesh_targets: BTreeMap<String, MeshTarget>,
    total_buffer_bytes: usize,
}

impl SceneBuilder {
    pub(super) fn push_f64(
        &mut self,
        role: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<BufferRef, Diagnostic> {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid("viewer float buffer contains a non-finite value"));
        }
        let bytes = values
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>();
        self.push_buffer(role.into(), ScalarType::Float64Le, shape, bytes, 8)
    }

    pub(super) fn push_u32(
        &mut self,
        role: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<u32>,
    ) -> Result<BufferRef, Diagnostic> {
        let bytes = values
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        self.push_buffer(role.into(), ScalarType::Uint32Le, shape, bytes, 4)
    }

    fn push_buffer(
        &mut self,
        role: String,
        scalar_type: ScalarType,
        shape: Vec<usize>,
        bytes: Vec<u8>,
        scalar_bytes: usize,
    ) -> Result<BufferRef, Diagnostic> {
        if role.is_empty() || shape.is_empty() || shape.contains(&0) {
            return Err(invalid("viewer buffer role and shape must be non-empty"));
        }
        let scalars = shape
            .iter()
            .try_fold(1_usize, |total, value| total.checked_mul(*value));
        let expected = scalars.and_then(|count| count.checked_mul(scalar_bytes));
        if expected != Some(bytes.len()) {
            return Err(invalid(
                "viewer buffer shape differs from its immutable bytes",
            ));
        }
        self.total_buffer_bytes = self
            .total_buffer_bytes
            .checked_add(bytes.len())
            .ok_or_else(|| invalid("viewer scene buffer bytes overflow usize"))?;
        if self.total_buffer_bytes > MAX_SCENE_BUFFER_BYTES {
            return Err(invalid("viewer scene exceeds the private v0 buffer budget"));
        }
        let index = self.buffers.len();
        let sha256 = Sha256::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        self.descriptors.push(BufferDescriptor {
            index,
            role,
            scalar_type,
            shape,
            byte_length: bytes.len(),
            sha256,
        });
        self.buffers.push(bytes);
        Ok(BufferRef { buffer: index })
    }

    pub(super) fn push_layer(&mut self, layer: LayerMetadata) -> Result<(), Diagnostic> {
        if self.layers.len() >= MAX_SCENE_LAYERS {
            return Err(invalid("viewer scene exceeds the private v0 layer budget"));
        }
        if layer.id().is_empty() || !self.layer_ids.insert(layer.id().to_owned()) {
            return Err(invalid("viewer layer identity is empty or duplicated"));
        }
        self.layers.push(layer);
        Ok(())
    }

    pub(super) fn register_mesh_target(
        &mut self,
        mesh_digest: String,
        target: MeshTarget,
    ) -> Result<(), Diagnostic> {
        if self.mesh_targets.insert(mesh_digest, target).is_some() {
            return Err(invalid("viewer scene repeats one exact Mesh"));
        }
        Ok(())
    }

    pub(super) fn mesh_target(&self, mesh_digest: &str) -> Option<&MeshTarget> {
        self.mesh_targets.get(mesh_digest)
    }

    pub(super) fn finish(self) -> Result<FinishedScene, Diagnostic> {
        if self.layers.is_empty() {
            return Err(invalid("viewer scene requires at least one typed layer"));
        }
        let metadata = SceneMetadata {
            schema: PRIVATE_SCENE_SCHEMA,
            layers: &self.layers,
            buffers: &self.descriptors,
            presentation: ScenePresentation {
                camera: "disposable",
                state_is_scientific: false,
            },
            reserved_layer_kinds: ["vector-field", "tensor-field", "trajectory"],
        };
        let metadata_json = serde_json::to_string(&metadata).map_err(|error| {
            invalid(format!(
                "cannot serialize private viewer scene metadata: {error}"
            ))
        })?;
        Ok(FinishedScene {
            metadata_json,
            buffers: self.buffers,
            layer_count: self.layers.len(),
        })
    }
}
