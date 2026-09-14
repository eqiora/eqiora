use eqiora::diagnostic::codes;
use eqiora::{Diagnostic, DimExponents};
use pyo3::prelude::*;

use crate::error::{diagnostic_error, validation_error};
use crate::result::PyFieldOutput;
use crate::trajectory::PyDerivedFieldSnapshot;

use super::scene::{LayerMetadata, PresentationScale, ScalarFieldLayer, SceneBuilder};

struct ScalarBlock {
    association: &'static str,
    values: Vec<f64>,
    coefficient_count: usize,
    logical_count: usize,
    support_indices: Option<Vec<u32>>,
}

struct ScalarProjection {
    identity: String,
    mesh_digest: String,
    model_digest: String,
    field_id: String,
    observation_digest: Option<String>,
    operator: Option<String>,
    dimension: DimExponents,
    frame: String,
    space: String,
    blocks: Vec<ScalarBlock>,
}

pub(super) fn add_field_output(
    py: Python<'_>,
    builder: &mut SceneBuilder,
    output: &PyFieldOutput,
) -> PyResult<()> {
    if !output.value_shape_value().is_empty() {
        return Err(unsupported(
            py,
            "private v0 viewer accepts scalar FieldOutput values only",
        ));
    }
    let mesh = output.mesh_handle(py);
    let mesh_digest = mesh.borrow(py).exact_mesh_digest().to_owned();
    let field = output.field_handle(py);
    let field = field.borrow(py);
    add_scalar_projection(
        py,
        builder,
        ScalarProjection {
            identity: format!("{}:{}", field.exact_model_digest(), field.exact_id()),
            mesh_digest,
            model_digest: field.exact_model_digest().to_owned(),
            field_id: field.exact_id().to_owned(),
            observation_digest: None,
            operator: None,
            dimension: output.dimension_value(),
            frame: "scalar".to_owned(),
            space: output.space_value().to_owned(),
            blocks: output
                .blocks()
                .iter()
                .map(|block| {
                    Ok(ScalarBlock {
                        association: block.association(),
                        values: block.snapshot(py)?,
                        coefficient_count: block.coefficient_count(),
                        logical_count: block.logical_shape().iter().product(),
                        support_indices: None,
                    })
                })
                .collect::<PyResult<_>>()?,
        },
    )
}

pub(super) fn add_derived_field(
    py: Python<'_>,
    builder: &mut SceneBuilder,
    output: &PyDerivedFieldSnapshot,
) -> PyResult<()> {
    if !output.value_shape_value().is_empty() {
        return Err(unsupported(
            py,
            "private v0 viewer accepts scalar DerivedFieldSnapshot values only",
        ));
    }
    let source_field = output.source_field_handle(py);
    let source_field = source_field.borrow(py);
    let blocks = output
        .blocks()
        .iter()
        .map(|block| {
            let values = block.scalar_snapshot(py)?.ok_or_else(|| {
                unsupported(
                    py,
                    "private v0 viewer accepts scalar DerivedFieldSnapshot values only",
                )
            })?;
            let coefficient_count = values.len();
            Ok(ScalarBlock {
                association: block.association(),
                values,
                coefficient_count,
                logical_count: coefficient_count,
                support_indices: Some(block.support_indices_snapshot(py)?),
            })
        })
        .collect::<PyResult<_>>()?;
    add_scalar_projection(
        py,
        builder,
        ScalarProjection {
            identity: output.exact_digest().to_owned(),
            mesh_digest: output.exact_mesh_digest().to_owned(),
            model_digest: source_field.exact_model_digest().to_owned(),
            field_id: source_field.exact_id().to_owned(),
            observation_digest: Some(output.exact_digest().to_owned()),
            operator: Some(output.operator_value().to_owned()),
            dimension: output.dimension_value(),
            frame: output.frame_value().to_owned(),
            space: "cell-average".to_owned(),
            blocks,
        },
    )
}

fn add_scalar_projection(
    py: Python<'_>,
    builder: &mut SceneBuilder,
    projection: ScalarProjection,
) -> PyResult<()> {
    let target = builder
        .mesh_target(&projection.mesh_digest)
        .cloned()
        .ok_or_else(|| {
            validation_error(
                py,
                &[Diagnostic::error(
                    codes::INVALID_ARTIFACT,
                    "ScalarFieldLayer requires its exact MeshLayer in the same scene",
                )],
            )
        })?;
    let dimension = projection.dimension.exponents();
    for block in projection.blocks {
        let expected = match block.association {
            "vertex" => target.vertex_count,
            "cell" => target.cell_count,
            association => {
                return Err(unsupported(
                    py,
                    &format!(
                        "private v0 viewer supports only vertex or cell scalar association, received {association:?}"
                    ),
                ));
            }
        };
        let dense_support = block.support_indices.as_ref().is_none_or(|indices| {
            indices.len() == expected
                && indices
                    .iter()
                    .enumerate()
                    .all(|(index, value)| usize::try_from(*value) == Ok(index))
        });
        if block.coefficient_count != expected
            || block.logical_count != expected
            || block.values.len() != expected
            || !dense_support
            || block.values.iter().any(|value| !value.is_finite())
        {
            return Err(validation_error(
                py,
                &[Diagnostic::error(
                    codes::INVALID_ARTIFACT,
                    "ScalarFieldLayer values disagree with exact dense association support or contain a non-finite value",
                )],
            ));
        }
        let minimum = block.values.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = block
            .values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let layer_id = format!(
            "scalar-field:{}:{}:{}",
            projection.mesh_digest, projection.identity, block.association
        );
        let values = builder
            .push_f64(format!("{layer_id}:values"), vec![expected], block.values)
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        builder
            .push_layer(LayerMetadata::ScalarField(ScalarFieldLayer {
                id: layer_id,
                target_layer: target.layer_id.clone(),
                mesh_digest: projection.mesh_digest.clone(),
                model_digest: projection.model_digest.clone(),
                field_id: projection.field_id.clone(),
                observation_digest: projection.observation_digest.clone(),
                operator: projection.operator.clone(),
                association: block.association.to_owned(),
                component_shape: Vec::new(),
                unit: "coherent-si".to_owned(),
                dimension,
                frame: projection.frame.clone(),
                space: projection.space.clone(),
                values,
                scale: PresentationScale {
                    provenance: "presentation-linear-range-from-accepted-values/v0".to_owned(),
                    minimum,
                    maximum,
                },
            }))
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
    }
    Ok(())
}

fn unsupported(py: Python<'_>, message: &str) -> PyErr {
    diagnostic_error(py, &[Diagnostic::error(codes::NOT_IMPLEMENTED, message)])
}
