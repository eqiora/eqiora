//! One production and replay authority for source-owned affine triangles.

use eqiora_core::Diagnostic;
use eqiora_geometry::CanonicalGeometryV1;
use eqiora_meshing::{
    CellId, FacetId, MeshEntity, MeshTopology, RetainedFacetSide2d, RevisionFacetFragment2d,
    SimplicialMesh, SimplicialRevisionOverlap2d,
};
use std::collections::{BTreeMap, BTreeSet};

use super::WireCorrespondenceV1;

use crate::{
    AffineTriangleMeshPolicy, GeometryMeshCorrespondenceEnvelopeV1, SimplicialMeshEnvelopeV1,
    invalid_artifact,
};

impl GeometryMeshCorrespondenceEnvelopeV1 {
    /// Generate a source-owned affine-triangle Mesh and its exact correspondence.
    ///
    /// Geometry selects the supported seed construction; all consumers use this
    /// same production path. The policy explicitly supplies initial cell counts.
    pub fn from_affine_triangles(
        geometry: &CanonicalGeometryV1,
        policy: &AffineTriangleMeshPolicy,
    ) -> Result<(SimplicialMeshEnvelopeV1, Self), Diagnostic> {
        let (mut mesh, mut correspondence) = if geometry.planar_rectangle_bounds().is_some() {
            Self::from_planar_rectangle_v2_affine_triangles(geometry, policy.cells())
        } else {
            Self::from_adjacent_rectangle_partition_affine_triangles(geometry, policy.cells())
        }?;
        for (marked, maximum_cells) in policy.refinements() {
            let marked = marked.iter().copied().map(CellId::new).collect::<Vec<_>>();
            let (target, overlap) = mesh.mesh().bisect_marked(
                eqiora_meshing::OverlapCoordinateChart2d::Material,
                &marked,
                *maximum_cells,
            )?;
            let target = SimplicialMeshEnvelopeV1::from_mesh(&target)?;
            correspondence = correspondence.refined(&mesh, &target, &overlap)?;
            mesh = target;
        }
        Ok((mesh, correspondence))
    }

    pub(crate) fn validate_affine_triangle_policy(
        &self,
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        policy: &AffineTriangleMeshPolicy,
    ) -> Result<(), Diagnostic> {
        let (expected_mesh, expected) = Self::from_affine_triangles(geometry, policy)?;
        if mesh != &expected_mesh || self != &expected {
            return Err(invalid_artifact(
                "affine-triangle resources differ from exact policy replay",
            ));
        }
        Ok(())
    }
}

/// Refine memberships through the admitted cell and retained-facet relations.
/// Both supported Geometry sources share this proof; their wire layouts only
/// project the resulting memberships and orientation.
pub(super) struct RefinedFace {
    overlap: SimplicialRevisionOverlap2d,
}

impl RefinedFace {
    pub(super) fn new(
        source: &SimplicialMesh,
        target: &SimplicialMesh,
        overlap: &SimplicialRevisionOverlap2d,
        source_cells: &[u64],
        source_facets: impl Iterator<Item = u64>,
    ) -> Result<Self, Diagnostic> {
        let source_cells = source_cells
            .iter()
            .map(|&index| local(index).map(CellId::new))
            .collect::<Result<Vec<_>, _>>()?;
        let selected = source_cells.iter().copied().collect::<BTreeSet<_>>();
        let target_cells = overlap
            .cell_fragments()
            .iter()
            .filter(|fragment| selected.contains(&fragment.source_cell()))
            .map(|fragment| fragment.target_cell())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let region = overlap.select_cells(&source_cells, &target_cells)?;
        let source_sides = frontier_sides(source, &source_cells);
        let declared = source_facets
            .map(|facet| local(facet).map(FacetId::new))
            .collect::<Result<BTreeSet<_>, _>>()?;
        if declared != source_sides.iter().map(|side| side.facet()).collect() {
            return Err(invalid_artifact(
                "source correspondence does not cover its complete region frontier",
            ));
        }
        let target_sides = frontier_sides(target, &target_cells);
        Ok(Self {
            overlap: region.with_retained_facets(source, &source_sides, target, &target_sides)?,
        })
    }

    pub(super) fn cells(&self) -> Result<Vec<u64>, Diagnostic> {
        self.overlap
            .target_cells()
            .iter()
            .map(|cell| portable(cell.index()))
            .collect()
    }

    pub(super) fn facets(
        &self,
        source_facets: &[u64],
    ) -> Result<Vec<&RevisionFacetFragment2d>, Diagnostic> {
        let source_facets = source_facets
            .iter()
            .map(|&facet| local(facet).map(FacetId::new))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut targets = BTreeMap::new();
        for fragment in self.overlap.retained_facet_fragments() {
            if source_facets.contains(&fragment.source_facet()) {
                targets.insert(fragment.target_facet(), fragment);
            }
        }
        if targets.is_empty() {
            return Err(invalid_artifact("refined semantic frontier is empty"));
        }
        Ok(targets.into_values().collect())
    }
}

fn frontier_sides(mesh: &SimplicialMesh, cells: &[CellId]) -> Vec<RetainedFacetSide2d> {
    let selected = cells.iter().copied().collect::<BTreeSet<_>>();
    (0..mesh.entity_count(1).expect("triangle edges"))
        .filter_map(|index| {
            let parents = mesh
                .incidence(MeshEntity::new(1, index), 2)
                .expect("edge incidence")
                .iter()
                .map(|entry| CellId::new(entry.entity.index()))
                .filter(|parent| selected.contains(parent))
                .collect::<Vec<_>>();
            match parents.as_slice() {
                &[parent] => Some(RetainedFacetSide2d::new(FacetId::new(index), parent)),
                _ => None,
            }
        })
        .collect()
}

pub(super) fn portable(index: usize) -> Result<u64, Diagnostic> {
    u64::try_from(index).map_err(|_| invalid_artifact("refined membership exceeds portable u64"))
}

fn local(index: u64) -> Result<usize, Diagnostic> {
    usize::try_from(index).map_err(|_| invalid_artifact("source membership exceeds local usize"))
}

impl GeometryMeshCorrespondenceEnvelopeV1 {
    fn refined(
        &self,
        source: &SimplicialMeshEnvelopeV1,
        target: &SimplicialMeshEnvelopeV1,
        overlap: &SimplicialRevisionOverlap2d,
    ) -> Result<Self, Diagnostic> {
        if !target
            .mesh()
            .vertices()
            .starts_with(source.mesh().vertices())
        {
            return Err(invalid_artifact(
                "refinement must retain source geometry vertices",
            ));
        }
        let mut refined = self.clone();
        match &mut refined.wire {
            WireCorrespondenceV1::PlanarRectangleV2(wire) => {
                wire.refine(source.mesh(), target.mesh(), overlap)?
            }
            WireCorrespondenceV1::AuthoredRegion(wire) => {
                wire.refine(source.mesh(), target.mesh(), overlap)?
            }
            _ => {
                return Err(invalid_artifact(
                    "unsupported affine-triangle source correspondence",
                ));
            }
        }
        let digest = target.digest()?.to_string();
        match &mut refined.wire {
            WireCorrespondenceV1::PlanarRectangleV2(wire) => wire.mesh_sha256 = digest,
            WireCorrespondenceV1::AuthoredRegion(wire) => wire.mesh_sha256 = digest,
            _ => unreachable!(),
        }
        refined.validate_local(crate::GeometryDecoderLimits::default())?;
        Ok(refined)
    }
}
