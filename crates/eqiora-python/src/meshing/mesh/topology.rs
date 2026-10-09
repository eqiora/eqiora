//! Exact entity connectivity shared by inspection and the viewer.
use super::*;

impl PyMesh {
    pub(crate) fn entity_vertex_indices(
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
                "entity is absent from the accepted Mesh topology",
            )
        })?;
        Ok(vertices.iter().map(|vertex| vertex.index()).collect())
    }
}
