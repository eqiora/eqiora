//! Shared Model-to-Geometry selection binding after common Mesh authentication.

use std::collections::BTreeMap;

use eqiora_artifact::{GeometryMeshCorrespondenceEnvelopeV1, SimplicialMeshEnvelopeV1};
use eqiora_core::{Diagnostic, RawId};
use eqiora_geometry::CanonicalGeometryV1;
use eqiora_meshing::{MeshEntity, MeshTopology};
use eqiora_schema::kernel::{DomainKind, KernelNode};
use eqiora_sem::KernelProgram;

use super::navier_stokes_realization::invalid_realization;

/// Retains the exact resources and selected entities shared by both flow models.
/// Provider-output authentication remains owned by `AuthenticatedCommonMesh`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct FlowGeometryBinding2d {
    pub(super) program: KernelProgram,
    pub(super) source: CanonicalGeometryV1,
    pub(super) mesh: SimplicialMeshEnvelopeV1,
    pub(super) correspondence: GeometryMeshCorrespondenceEnvelopeV1,
    pub(super) entity_sets: BTreeMap<String, Vec<MeshEntity>>,
}

impl FlowGeometryBinding2d {
    /// Preserve the exact-source diagnostic before model-specific recognition.
    pub(super) fn require_source(
        program: &KernelProgram,
        source: &CanonicalGeometryV1,
    ) -> Result<(), Diagnostic> {
        let mut digests = program.nodes().filter_map(|node| match node {
            KernelNode::Domain(domain) => match domain.kind() {
                DomainKind::GeometryRegion { geometry, .. } => Some(geometry.bytes()),
                _ => None,
            },
            _ => None,
        });
        if let Some(digest) = digests.next()
            && digests.next().is_none()
            && digest != source.digest_bytes()
        {
            return Err(invalid_realization(
                "Model belongs to another exact source revision",
            ));
        }
        Ok(())
    }

    pub(super) fn new<'a>(
        program: &'a KernelProgram,
        source: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        domain: RawId,
        model_source_digest: Option<[u8; 32]>,
        boundary_names: impl Iterator<Item = &'a String>,
    ) -> Result<Self, Diagnostic> {
        let Some(KernelNode::Domain(domain)) = program.node(domain) else {
            unreachable!("recognized flow model owns a Domain")
        };
        let DomainKind::GeometryRegion {
            entity_set: region_set,
            ..
        } = domain.kind()
        else {
            unreachable!("recognized geometry flow model owns a GeometryRegion")
        };
        let mut entity_sets = BTreeMap::new();
        for name in boundary_names.chain(std::iter::once(region_set)) {
            entity_sets.insert(
                name.clone(),
                correspondence.planar_circular_hole_v2_entity_set_entities(source, name)?,
            );
        }
        let expected_cells = (0..mesh.mesh().entity_count(2).expect("2D mesh cells"))
            .map(|index| MeshEntity::new(2, index))
            .collect::<Vec<_>>();
        if entity_sets[region_set] != expected_cells {
            return Err(invalid_realization(
                "the authored GeometryRegion must realize every mesh cell exactly once",
            ));
        }
        if model_source_digest != Some(source.digest_bytes()) {
            return Err(invalid_realization(
                "Model GeometryRegion digest differs from the accepted exact source revision",
            ));
        }
        Ok(Self {
            program: program.clone(),
            source: source.clone(),
            mesh: mesh.clone(),
            correspondence: correspondence.clone(),
            entity_sets,
        })
    }

    pub(super) fn entities(&self, name: &str) -> Result<&[MeshEntity], Diagnostic> {
        self.entity_sets
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                invalid_realization(format!("geometry binding has no entity set named `{name}`"))
            })
    }
}
