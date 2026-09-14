//! Private semantic viewer projection for the optional installed Notebook host.
//!
//! This is deliberately not a persisted artifact or public renderer protocol.

mod field;
mod geometry;
mod mesh;
mod scene;

use std::collections::BTreeSet;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule, PyTuple};

use crate::error::{diagnostic_error, validation_error};
use crate::geometry::PyGeometry;
use crate::meshing::PyMesh;
use crate::result::PyFieldOutput;
use crate::trajectory::PyDerivedFieldSnapshot;

use scene::{FinishedScene, SceneBuilder};

#[pyclass(name = "_ViewerScene", module = "eqiora._eqiora", frozen)]
struct PyViewerScene {
    metadata_json: String,
    buffers: Vec<Vec<u8>>,
    layer_count: usize,
}

#[pymethods]
impl PyViewerScene {
    #[getter]
    fn metadata_json(&self) -> &str {
        &self.metadata_json
    }

    #[getter]
    fn buffers(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(
            py,
            self.buffers.iter().map(|buffer| PyBytes::new(py, buffer)),
        )?
        .unbind())
    }

    #[getter]
    const fn layer_count(&self) -> usize {
        self.layer_count
    }

    fn __repr__(&self) -> String {
        format!(
            "_ViewerScene(layers={}, buffers={}, schema={:?})",
            self.layer_count,
            self.buffers.len(),
            scene::PRIVATE_SCENE_SCHEMA,
        )
    }
}

#[pyfunction(name = "_compose_view")]
#[pyo3(signature = (values, /))]
fn compose_view(py: Python<'_>, values: &Bound<'_, PyTuple>) -> PyResult<PyViewerScene> {
    if values.is_empty() {
        return Err(PyTypeError::new_err(
            "View requires at least one Geometry, Mesh, FieldOutput, or DerivedFieldSnapshot",
        ));
    }
    let mut geometries = Vec::new();
    let mut meshes = Vec::new();
    let mut fields = Vec::new();
    let mut derived_fields = Vec::new();
    for value in values.iter() {
        if let Ok(value) = value.extract::<Py<PyGeometry>>() {
            geometries.push(value);
        } else if let Ok(value) = value.extract::<Py<PyMesh>>() {
            meshes.push(value);
        } else if let Ok(value) = value.extract::<Py<PyFieldOutput>>() {
            fields.push(value);
        } else if let Ok(value) = value.extract::<Py<PyDerivedFieldSnapshot>>() {
            derived_fields.push(value);
        } else {
            return Err(PyTypeError::new_err(
                "View.add accepts only accepted Geometry, Mesh, scalar FieldOutput, or scalar DerivedFieldSnapshot values",
            ));
        }
    }

    let mut builder = SceneBuilder::default();
    let mut geometry_digests = BTreeSet::new();
    for geometry in geometries {
        let geometry = geometry.borrow(py);
        let digest = crate::geometry::digest_to_hex(&geometry.geometry().digest_bytes());
        if !geometry_digests.insert(digest) {
            return Err(PyTypeError::new_err("View repeats one exact Geometry"));
        }
        geometry::add_geometry(&mut builder, geometry.geometry())
            .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?;
    }
    for mesh in meshes {
        mesh::add_mesh(py, &mut builder, &mesh.borrow(py))?;
    }
    for field in fields {
        field::add_field_output(py, &mut builder, &field.borrow(py))?;
    }
    for field in derived_fields {
        field::add_derived_field(py, &mut builder, &field.borrow(py))?;
    }
    let FinishedScene {
        metadata_json,
        buffers,
        layer_count,
    } = builder
        .finish()
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
    if buffers.is_empty() {
        return Err(PyRuntimeError::new_err(
            "private viewer scene unexpectedly contains no binary buffers",
        ));
    }
    Ok(PyViewerScene {
        metadata_json,
        buffers,
        layer_count,
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyViewerScene>()?;
    module.add_function(wrap_pyfunction!(compose_view, module)?)?;
    Ok(())
}
