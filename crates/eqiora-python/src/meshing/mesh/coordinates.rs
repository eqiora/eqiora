//! Python projection of the same authenticated dimensioned-factor grid used by native Plans.
use super::*;
use crate::model::{PyModel, PyModelDomainRef};

impl PyMesh {
    /// Build a tensor grid over an exact Model-bound coordinate Domain.
    pub(super) fn from_coordinate_domain(
        py: Python<'_>,
        model: &PyModel,
        domain: &PyModelDomainRef,
        cells_per_factor: Vec<usize>,
    ) -> PyResult<Self> {
        let artifact = model.artifact();
        let digest = artifact
            .digest()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        if domain.exact_model_digest() != digest.to_string() {
            return Err(request_error(
                py,
                "coordinate Domain belongs to a foreign or stale Model",
            ));
        }
        let id = ulid::Ulid::from_string(domain.exact_id())
            .map(eqiora::Id::<eqiora::kinds::Domain>::from_ulid)
            .map_err(|_| request_error(py, "coordinate Domain has an invalid identity"))?;
        let owner = AuthenticatedCommonMesh::coordinate_factors(artifact, id, &cells_per_factor)
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        Self::from_authenticated(py, owner)
    }
}

impl PyMesh {
    pub(super) fn from_coordinate_factors(
        py: Python<'_>,
        owner: AuthenticatedCommonMesh,
    ) -> PyResult<Self> {
        let mesh = owner
            .cartesian_mesh()
            .ok_or_else(|| request_error(py, "coordinate grid omitted its Cartesian topology"))?;
        let dimension = mesh.dimension();
        let native = mesh.mesh();
        let vertex_count = native.entity_count(0).expect("authenticated grid vertices");
        let cell_count = native
            .entity_count(dimension)
            .expect("authenticated grid cells");
        let (coordinates, cells) =
            project_cartesian_mesh(py, native, dimension, vertex_count, cell_count)?;
        let source_digest = owner
            .source_digest()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?
            .to_string();
        let mesh_digest = mesh
            .digest()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?
            .to_string();
        let canonical_bytes = mesh
            .canonical_json()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        Ok(Self {
            source: AcceptedMeshSource::CoordinateFactors {
                owner: Box::new(owner),
            },
            lineage: MeshLineage {
                source_digest,
                mesh_digest,
                realized_geometry_digest: None,
                correspondence_digest: None,
                dimension,
                vertex_count,
                cell_count,
            },
            canonical_bytes,
            coordinates,
            cells,
        })
    }
}
