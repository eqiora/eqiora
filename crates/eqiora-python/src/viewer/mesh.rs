use eqiora::Diagnostic;
use eqiora::diagnostic::codes;
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;

use crate::error::{diagnostic_error, validation_error};
use crate::meshing::PyMesh;

use super::scene::{LayerMetadata, MeshLayer, MeshTarget, SceneBuilder, SelectionLayer};

pub(super) fn add_mesh(py: Python<'_>, builder: &mut SceneBuilder, mesh: &PyMesh) -> PyResult<()> {
    let (coordinates, [vertex_count, dimension]) = mesh.viewer_coordinates(py)?;
    let (connectivity, [cell_count, cell_width]) = mesh.viewer_cells(py)?;
    let (cell_kind, presentation_policy) = match (dimension, cell_width) {
        (2, 3) => ("triangle", "exact-triangle-connectivity/v0"),
        (2, 4) => (
            "quadrilateral",
            "quad-fixed-diagonal-0-2/presentation-only/v0",
        ),
        _ => {
            return Err(diagnostic_error(
                py,
                &[Diagnostic::error(
                    codes::NOT_IMPLEMENTED,
                    format!(
                        "private v0 viewer supports only 2D triangle or quadrilateral Mesh cells, received dimension {dimension} and arity {cell_width}"
                    ),
                )],
            ));
        }
    };
    if connectivity
        .iter()
        .any(|index| (*index as usize) >= vertex_count)
    {
        return Err(validation_error(
            py,
            &[Diagnostic::error(
                codes::INVALID_ARTIFACT,
                "Mesh viewer connectivity contains an out-of-range vertex",
            )],
        ));
    }
    let correspondence_digest = mesh.correspondence_digest_value().ok_or_else(|| {
        diagnostic_error(py, &[Diagnostic::error(codes::NOT_IMPLEMENTED,
            "physical Mesh viewer requires Geometry correspondence; coordinate-factor plots are not yet supported")])
    })?;
    let mesh_digest = mesh.exact_mesh_digest().to_owned();
    let layer_id = format!("mesh:{mesh_digest}");
    let coordinates = builder
        .push_f64(
            format!("{layer_id}:coordinates"),
            vec![vertex_count, dimension],
            coordinates,
        )
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
    let connectivity = builder
        .push_u32(
            format!("{layer_id}:connectivity"),
            vec![cell_count, cell_width],
            connectivity,
        )
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
    builder
        .push_layer(LayerMetadata::Mesh(MeshLayer {
            id: layer_id.clone(),
            owner_digest: mesh_digest.clone(),
            source_digest: mesh.source_digest_value().to_owned(),
            correspondence_digest: correspondence_digest.to_owned(),
            dimension,
            cell_kind: cell_kind.to_owned(),
            presentation_policy: presentation_policy.to_owned(),
            vertex_count,
            cell_count,
            coordinates,
            connectivity,
        }))
        .and_then(|()| {
            builder.register_mesh_target(
                mesh_digest.clone(),
                MeshTarget {
                    layer_id: layer_id.clone(),
                    vertex_count,
                    cell_count,
                },
            )
        })
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;

    for (name, expected_dimension) in mesh.viewer_selection_names() {
        let selection_id = format!("selection:{layer_id}:{name}");
        if !matches!(expected_dimension, 1 | 2) {
            builder
                .push_layer(LayerMetadata::Selection(SelectionLayer {
                    id: selection_id,
                    target_layer: layer_id.clone(),
                    owner_digest: mesh_digest.clone(),
                    correspondence_digest: Some(correspondence_digest.to_owned()),
                    name: name.to_owned(),
                    dimension: expected_dimension,
                    available: false,
                    unavailable_reason: Some(
                        "private v0 viewer does not expose Mesh vertex selection interaction"
                            .to_owned(),
                    ),
                    entity_indices: None,
                    connectivity: None,
                }))
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            continue;
        }
        let entities = mesh
            .viewer_selection_entities(name)
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        if entities.is_empty()
            || entities
                .iter()
                .any(|entity| entity.dimension() != expected_dimension)
        {
            return Err(validation_error(
                py,
                &[Diagnostic::error(
                    codes::INVALID_ARTIFACT,
                    "correspondence-owned selection is empty or dimension-inconsistent",
                )],
            ));
        }
        let mut indices = Vec::with_capacity(entities.len());
        let mut selected_connectivity = Vec::new();
        let mut width = None;
        for entity in entities {
            indices.push(u32::try_from(entity.index()).map_err(|_| {
                PyOverflowError::new_err("viewer selection entity index exceeds uint32")
            })?);
            let vertices = mesh
                .entity_vertex_indices(entity)
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            if width
                .replace(vertices.len())
                .is_some_and(|value| value != vertices.len())
            {
                return Err(validation_error(
                    py,
                    &[Diagnostic::error(
                        codes::INVALID_ARTIFACT,
                        "viewer selection connectivity has mixed arity",
                    )],
                ));
            }
            selected_connectivity.extend(
                vertices
                    .into_iter()
                    .map(|vertex| {
                        u32::try_from(vertex).map_err(|_| {
                            PyOverflowError::new_err("viewer selection vertex index exceeds uint32")
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            );
        }
        let width = width.expect("non-empty exact selection");
        let entity_count = indices.len();
        let indices = builder
            .push_u32(
                format!("{selection_id}:entity-indices"),
                vec![entity_count],
                indices,
            )
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        let selected_connectivity = builder
            .push_u32(
                format!("{selection_id}:connectivity"),
                vec![entity_count, width],
                selected_connectivity,
            )
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        builder
            .push_layer(LayerMetadata::Selection(SelectionLayer {
                id: selection_id,
                target_layer: layer_id.clone(),
                owner_digest: mesh_digest.clone(),
                correspondence_digest: Some(correspondence_digest.to_owned()),
                name: name.to_owned(),
                dimension: expected_dimension,
                available: true,
                unavailable_reason: None,
                entity_indices: Some(indices),
                connectivity: Some(selected_connectivity),
            }))
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
    }
    Ok(())
}
