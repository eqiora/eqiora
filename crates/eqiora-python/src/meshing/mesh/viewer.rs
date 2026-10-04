//! Viewer-only semantic projections of accepted Mesh resources.

use eqiora::Diagnostic;
use eqiora::diagnostic::codes;
use eqiora::meshing::MeshEntity;
use pyo3::prelude::*;

use super::{AcceptedMeshSource, PyMesh, SourceOwnedProviderObservation};

impl PyMesh {
    pub(crate) fn viewer_coordinates(&self, py: Python<'_>) -> PyResult<(Vec<f64>, [usize; 2])> {
        let (rows, columns) = self.coordinates.shape();
        Ok((self.coordinates.snapshot(py)?, [rows, columns]))
    }

    pub(crate) fn viewer_cells(&self, py: Python<'_>) -> PyResult<(Vec<u32>, [usize; 2])> {
        let (rows, columns) = self.cells.shape();
        Ok((self.cells.snapshot(py)?, [rows, columns]))
    }

    pub(crate) fn viewer_selection_names(&self) -> impl Iterator<Item = (&str, usize)> {
        let selections = match &self.source {
            AcceptedMeshSource::CoordinateFactors { .. } => &[][..],
            AcceptedMeshSource::SourceOwned { geometry, .. }
            | AcceptedMeshSource::SourceOwnedCartesian { geometry, .. } => geometry.entity_sets(),
        };
        selections
            .iter()
            .map(|selection| (selection.name(), selection.dimension()))
    }

    pub(crate) fn viewer_selection_entities(
        &self,
        name: &str,
    ) -> Result<Vec<MeshEntity>, Diagnostic> {
        match &self.source {
            AcceptedMeshSource::CoordinateFactors { .. } => Err(Diagnostic::error(
                codes::NOT_IMPLEMENTED,
                "coordinate-factor grids have no Geometry selections",
            )),
            AcceptedMeshSource::SourceOwned {
                geometry,
                correspondence,
                provider_observation: SourceOwnedProviderObservation::AffineTriangle,
                ..
            } => {
                if geometry.planar_rectangle_bounds().is_some() {
                    correspondence.planar_rectangle_v2_entity_set_entities(geometry, name)
                } else {
                    correspondence.adjacent_rectangle_partition_entity_set_entities(geometry, name)
                }
            }
            AcceptedMeshSource::SourceOwned {
                geometry,
                correspondence,
                ..
            } => correspondence.planar_circular_hole_v2_entity_set_entities(geometry, name),
            AcceptedMeshSource::SourceOwnedCartesian {
                geometry,
                correspondence,
                ..
            } => correspondence.planar_rectangle_v2_entity_set_entities(geometry, name),
        }
    }

    pub(crate) fn viewer_entity_vertices(
        &self,
        entity: MeshEntity,
    ) -> Result<Vec<usize>, Diagnostic> {
        let vertices = match &self.source {
            AcceptedMeshSource::CoordinateFactors { owner } => owner
                .cartesian_mesh()
                .and_then(|mesh| mesh.mesh().entity_vertices(entity)),
            AcceptedMeshSource::SourceOwned { mesh, .. } => mesh.mesh().entity_vertices(entity),
            AcceptedMeshSource::SourceOwnedCartesian { mesh, .. } => {
                mesh.mesh().entity_vertices(entity)
            }
        }
        .ok_or_else(|| {
            Diagnostic::error(
                codes::INVALID_ARTIFACT,
                "accepted Mesh omitted correspondence-selected entity connectivity",
            )
        })?;
        Ok(vertices.iter().map(|vertex| vertex.index()).collect())
    }
}
