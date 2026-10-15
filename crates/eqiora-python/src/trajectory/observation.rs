use std::hash::{Hash, Hasher};

use eqiora::DimExponents;
use pyo3::prelude::*;
use sha2::{Digest, Sha256};

use crate::geometry::PyGeometrySelection;
use crate::model::PyModelFieldRef;

use super::hex_sha256;

const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
const INTRINSIC_2D_FORCE: DimExponents =
    DimExponents::from_integers([1, 0, -2, 0, 0, 0, 0]).expect("bounded dimension");
const INTRINSIC_2D_FLUX: DimExponents =
    DimExponents::from_integers([0, 2, -1, 0, 0, 0, 0]).expect("bounded dimension");

fn hash_text(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

/// One typed continuous-Field sample bound to an accepted State and point.
#[pyclass(
    name = "FieldSample",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
pub(crate) struct PyFieldSample {
    digest: String,
    source_state_digest: String,
    field: Py<PyModelFieldRef>,
    field_id: String,
    mesh_digest: String,
    support_domain_id: String,
    point_m: [f64; 2],
    value: f64,
}

impl PartialEq for PyFieldSample {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}

impl Eq for PyFieldSample {}

impl Hash for PyFieldSample {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.digest.hash(state);
    }
}

impl PyFieldSample {
    pub(crate) fn pressure(
        field: Py<PyModelFieldRef>,
        field_id: String,
        source_state_digest: &str,
        mesh_digest: &str,
        support_domain_id: String,
        point_m: [f64; 2],
        value: f64,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"eqiora.field-sample/v1\0");
        hash_text(&mut hasher, source_state_digest);
        hash_text(&mut hasher, &field_id);
        hash_text(&mut hasher, mesh_digest);
        hash_text(&mut hasher, &support_domain_id);
        for coordinate in point_m {
            hasher.update(coordinate.to_bits().to_be_bytes());
        }
        hasher.update(value.to_bits().to_be_bytes());
        Self {
            digest: hex_sha256(hasher.finalize().as_slice()),
            source_state_digest: source_state_digest.to_owned(),
            field,
            field_id,
            mesh_digest: mesh_digest.to_owned(),
            support_domain_id,
            point_m,
            value,
        }
    }
}

#[pymethods]
impl PyFieldSample {
    #[getter]
    fn digest(&self) -> &str {
        &self.digest
    }

    #[getter]
    fn source_state_digest(&self) -> &str {
        &self.source_state_digest
    }

    #[getter]
    fn field(&self, py: Python<'_>) -> Py<PyModelFieldRef> {
        self.field.clone_ref(py)
    }

    #[getter]
    fn mesh_digest(&self) -> &str {
        &self.mesh_digest
    }

    #[getter]
    fn support_domain_id(&self) -> &str {
        &self.support_domain_id
    }

    #[getter]
    const fn point_m(&self) -> (f64, f64) {
        (self.point_m[0], self.point_m[1])
    }

    #[getter]
    const fn value(&self) -> f64 {
        self.value
    }

    #[getter]
    fn dimension(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::modeling::dimension::exponents(py, PRESSURE)
    }

    #[getter]
    const fn frame(&self) -> &'static str {
        "invariant"
    }

    fn __repr__(&self) -> String {
        format!(
            "FieldSample(field={:?}, point_m={:?}, value={:?}, digest={:?})",
            self.field_id, self.point_m, self.value, self.digest,
        )
    }
}

/// Signed action pair for one authenticated boundary of an accepted State.
#[pyclass(
    name = "BoundaryForce",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
pub(crate) struct PyBoundaryForce {
    digest: String,
    source_digest: String,
    output_index: Option<usize>,
    selection: Py<PyGeometrySelection>,
    selection_name: String,
    geometry_digest: String,
    mesh_digest: String,
    on_domain: [f64; 2],
}

impl PartialEq for PyBoundaryForce {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}

impl Eq for PyBoundaryForce {}

impl Hash for PyBoundaryForce {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.digest.hash(state);
    }
}

impl PyBoundaryForce {
    pub(crate) fn new(
        selection: Py<PyGeometrySelection>,
        selection_name: String,
        geometry_digest: String,
        source_digest: &str,
        output_index: Option<usize>,
        mesh_digest: &str,
        on_domain: [f64; 2],
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"eqiora.boundary-force/v3\0");
        hash_text(&mut hasher, source_digest);
        match output_index {
            None => hasher.update([0]),
            Some(index) => {
                hasher.update([1]);
                hasher.update((index as u64).to_be_bytes());
            }
        }
        hash_text(&mut hasher, &geometry_digest);
        hash_text(&mut hasher, &selection_name);
        hash_text(&mut hasher, mesh_digest);
        for component in on_domain {
            hasher.update(component.to_bits().to_be_bytes());
        }
        Self {
            digest: hex_sha256(hasher.finalize().as_slice()),
            source_digest: source_digest.to_owned(),
            output_index,
            selection,
            selection_name,
            geometry_digest,
            mesh_digest: mesh_digest.to_owned(),
            on_domain,
        }
    }
}

#[pymethods]
impl PyBoundaryForce {
    #[getter]
    fn digest(&self) -> &str {
        &self.digest
    }

    #[getter]
    fn source_digest(&self) -> &str {
        &self.source_digest
    }

    #[getter]
    const fn source_kind(&self) -> &'static str {
        "result"
    }

    #[getter]
    const fn output_index(&self) -> Option<usize> {
        self.output_index
    }

    #[getter]
    fn selection(&self, py: Python<'_>) -> Py<PyGeometrySelection> {
        self.selection.clone_ref(py)
    }

    #[getter]
    fn geometry_digest(&self) -> &str {
        &self.geometry_digest
    }

    #[getter]
    fn mesh_digest(&self) -> &str {
        &self.mesh_digest
    }

    /// Force exerted by the selected boundary on the fluid domain, in N/m.
    #[getter]
    const fn on_domain(&self) -> (f64, f64) {
        (self.on_domain[0], self.on_domain[1])
    }

    /// Equal-and-opposite force exerted by the fluid on the selected boundary, in N/m.
    #[getter]
    const fn on_selection(&self) -> (f64, f64) {
        (-self.on_domain[0], -self.on_domain[1])
    }

    #[getter]
    fn dimension(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::modeling::dimension::exponents(py, INTRINSIC_2D_FORCE)
    }

    #[getter]
    const fn frame(&self) -> &'static str {
        "spatial-cartesian"
    }

    fn __repr__(&self) -> String {
        format!(
            "BoundaryForce(selection={:?}, on_selection={:?}, digest={:?})",
            self.selection_name,
            [-self.on_domain[0], -self.on_domain[1]],
            self.digest,
        )
    }
}

/// Signed intrinsic-2D volume flux on one authenticated boundary.
#[pyclass(
    name = "BoundaryFlux",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
pub(crate) struct PyBoundaryFlux {
    digest: String,
    source_digest: String,
    selection: Py<PyGeometrySelection>,
    selection_name: String,
    geometry_digest: String,
    mesh_digest: String,
    value: f64,
}

impl PartialEq for PyBoundaryFlux {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}

impl Eq for PyBoundaryFlux {}

impl Hash for PyBoundaryFlux {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.digest.hash(state);
    }
}

impl PyBoundaryFlux {
    pub(crate) fn new(
        selection: Py<PyGeometrySelection>,
        selection_name: String,
        geometry_digest: String,
        source_digest: &str,
        mesh_digest: &str,
        value: f64,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"eqiora.boundary-flux/v1\0");
        for text in [
            source_digest,
            &geometry_digest,
            &selection_name,
            mesh_digest,
        ] {
            hash_text(&mut hasher, text);
        }
        hasher.update(value.to_bits().to_be_bytes());
        Self {
            digest: hex_sha256(hasher.finalize().as_slice()),
            source_digest: source_digest.to_owned(),
            selection,
            selection_name,
            geometry_digest,
            mesh_digest: mesh_digest.to_owned(),
            value,
        }
    }
}

#[pymethods]
impl PyBoundaryFlux {
    #[getter]
    fn digest(&self) -> &str {
        &self.digest
    }
    #[getter]
    fn source_digest(&self) -> &str {
        &self.source_digest
    }
    #[getter]
    fn selection(&self, py: Python<'_>) -> Py<PyGeometrySelection> {
        self.selection.clone_ref(py)
    }
    #[getter]
    fn geometry_digest(&self) -> &str {
        &self.geometry_digest
    }
    #[getter]
    fn mesh_digest(&self) -> &str {
        &self.mesh_digest
    }
    #[getter]
    const fn value(&self) -> f64 {
        self.value
    }
    #[getter]
    fn dimension(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::modeling::dimension::exponents(py, INTRINSIC_2D_FLUX)
    }
    #[getter]
    const fn frame(&self) -> &'static str {
        "invariant"
    }
    fn __repr__(&self) -> String {
        format!(
            "BoundaryFlux(selection={:?}, value={:?}, digest={:?})",
            self.selection_name, self.value, self.digest,
        )
    }
}
