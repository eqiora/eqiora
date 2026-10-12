//! Exact entity connectivity shared by inspection and the viewer.
use super::*;
use crate::geometry::PyGeometrySelection;
use pyo3::exceptions::PyTypeError;

impl PyMesh {
    pub(super) fn selected_entities(
        &self,
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<MeshEntity>> {
        let selection;
        let (name, expected_dimension) = if let Ok(name) = name.extract::<&str>() {
            (name, None)
        } else if let Ok(value) = name.extract::<PyRef<'_, PyGeometrySelection>>() {
            selection = value;
            if selection.bound_source_digest() != self.lineage.source_digest {
                let diagnostic = Diagnostic::error(
                    codes::INVALID_ARTIFACT,
                    "GeometrySelection belongs to a foreign or stale Geometry revision",
                );
                return Err(validation_error(py, std::slice::from_ref(&diagnostic)));
            }
            (
                selection.canonical_name(),
                Some(selection.canonical_dimension()),
            )
        } else {
            return Err(PyTypeError::new_err(
                "name must be a str or GeometrySelection",
            ));
        };
        match &self.source {
            AcceptedMeshSource::CoordinateFactors { .. } => Err(capability_error(
                py,
                "coordinate-factor grids have no Geometry selections",
            )),
            AcceptedMeshSource::SourceOwned {
                geometry,
                correspondence,
                provider_observation: SourceOwnedProviderObservation::AffineTriangle,
                ..
            } => (if geometry.planar_rectangle_bounds().is_some() {
                correspondence.planar_rectangle_v2_entity_set_entities(geometry, name)
            } else {
                correspondence.adjacent_rectangle_partition_entity_set_entities(geometry, name)
            })
            .and_then(|entities| validated_entities(entities, expected_dimension))
            .map_err(|diagnostic| validation_error(py, &[diagnostic])),
            AcceptedMeshSource::SourceOwned {
                geometry,
                correspondence,
                ..
            } => correspondence
                .planar_circular_hole_v2_entity_set_entities(geometry, name)
                .and_then(|entities| validated_entities(entities, expected_dimension))
                .map_err(|diagnostic| validation_error(py, std::slice::from_ref(&diagnostic))),
            AcceptedMeshSource::SourceOwnedCartesian {
                geometry,
                correspondence,
                ..
            } => (if geometry.cartesian_box_bounds().is_some() {
                correspondence.cartesian_box_v1_entity_set_entities(geometry, name)
            } else {
                correspondence.planar_rectangle_v2_entity_set_entities(geometry, name)
            })
            .and_then(|entities| validated_entities(entities, expected_dimension))
            .map_err(|diagnostic| validation_error(py, &[diagnostic])),
        }
    }

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

fn validated_entities(
    entities: Vec<MeshEntity>,
    expected_dimension: Option<usize>,
) -> Result<Vec<MeshEntity>, Diagnostic> {
    if expected_dimension.is_some_and(|dimension| {
        entities
            .iter()
            .any(|entity| entity.dimension() != dimension)
    }) {
        return Err(Diagnostic::error(
            codes::INVALID_ARTIFACT,
            "GeometrySelection dimension differs from correspondence-owned membership",
        ));
    }
    Ok(entities)
}

#[cfg(test)]
#[test]
fn revision_bound_selection_dimension_must_match_correspondence_membership() {
    assert!(
        validated_entities(vec![MeshEntity::new(1, 0)], Some(2)).is_err(),
        "dimension-wrong correspondence membership must reject"
    );
    assert_eq!(
        validated_entities(vec![MeshEntity::new(1, 0)], Some(1)).unwrap(),
        vec![MeshEntity::new(1, 0)]
    );
}
