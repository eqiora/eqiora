//! Shaped immutable projections of complete native first-order actions.

use super::*;
use eqiora::api::{DifferentiableCotangent, DifferentiableTangent, EvaluationMapResult};

/// Complete batched JVP values with unchanged per-action accepted evidence.
#[pyclass(name = "EvaluationMapJvp", module = "eqiora._eqiora", frozen)]
pub(super) struct PyEvaluationMapJvp {
    pub(super) native: EvaluationMapResult<DifferentiableTangent>,
    pub(super) plan: PyEvaluationMapPlan,
    pub(super) seeds: Vec<usize>,
    pub(super) positions: Vec<usize>,
}

#[pymethods]
impl PyEvaluationMapJvp {
    fn __repr__(&self) -> String {
        format!(
            "EvaluationMapJvp(shape={:?}, point_axes={:?})",
            self.native.shape(),
            self.positions
        )
    }
    #[getter]
    fn plan(&self) -> PyEvaluationMapPlan {
        self.plan.clone()
    }
    #[getter]
    fn seed_shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        tuple(py, &self.seeds)
    }
    #[getter]
    fn point_axes(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        tuple(py, &self.positions)
    }
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        tuple(py, self.native.shape())
    }
    #[getter]
    fn output(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        values::project_collection(
            py,
            self.plan.evaluation_program()?.identity().outputs(),
            self.native.shape(),
            self.native.products().len(),
            |row, leaf| {
                self.native.products()[row]
                    .output()
                    .leaf(leaf)
                    .expect("admitted leaf")
                    .components()
            },
        )
    }
    #[getter]
    fn tangent(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        values::project_collection(
            py,
            self.plan.evaluation_program()?.identity().outputs(),
            self.native.shape(),
            self.native.products().len(),
            |row, leaf| {
                self.native.products()[row]
                    .derivative()
                    .components(leaf)
                    .expect("admitted leaf")
            },
        )
    }
    fn member(&self, py: Python<'_>, index: usize) -> PyResult<PyDifferentiableJvp> {
        let value = self
            .native
            .products()
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("product grid occurrence is out of range"))?;
        jvp_result(py, value.clone())
    }
}

/// Complete batched VJP with shared sums, mapped covectors and accepted evidence.
#[pyclass(name = "EvaluationMapVjp", module = "eqiora._eqiora", frozen)]
pub(super) struct PyEvaluationMapVjp {
    pub(super) native: EvaluationMapResult<DifferentiableCotangent>,
    pub(super) plan: PyEvaluationMapPlan,
    pub(super) seeds: Vec<usize>,
    pub(super) positions: Vec<usize>,
}

#[pymethods]
impl PyEvaluationMapVjp {
    fn __repr__(&self) -> String {
        format!(
            "EvaluationMapVjp(shared_shape={:?}, mapped_shape={:?})",
            self.native.shared_shape(),
            self.native.mapped_shape()
        )
    }
    #[getter]
    fn plan(&self) -> PyEvaluationMapPlan {
        self.plan.clone()
    }
    #[getter]
    fn seed_shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        tuple(py, &self.seeds)
    }
    #[getter]
    fn point_axes(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        tuple(py, &self.positions)
    }
    #[getter]
    fn shared_shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        tuple(py, self.native.shared_shape())
    }
    #[getter]
    fn mapped_shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        tuple(py, self.native.mapped_shape())
    }
    #[getter]
    fn shared_input_ids(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.plan.shared_input_ids(py)
    }
    #[getter]
    fn mapped_input_ids(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.plan.mapped_input_ids(py)
    }
    #[getter]
    fn shared_cotangents(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        values::project_collection(
            py,
            self.native.shared_signatures(),
            self.native.shared_shape(),
            self.native.shared_cotangents().len(),
            |row, leaf| {
                self.native.shared_cotangents()[row]
                    .components(leaf)
                    .expect("admitted leaf")
            },
        )
    }
    #[getter]
    fn mapped_cotangents(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        values::project_collection(
            py,
            self.native.mapped_signatures(),
            self.native.mapped_shape(),
            self.native.mapped_cotangents().len(),
            |row, leaf| {
                self.native.mapped_cotangents()[row]
                    .components(leaf)
                    .expect("admitted leaf")
            },
        )
    }
    fn member(&self, py: Python<'_>, index: usize) -> PyResult<PyDifferentiableVjp> {
        let value = self
            .native
            .products()
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("product grid occurrence is out of range"))?;
        vjp_result(py, value.clone())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEvaluationMapJvp>()?;
    module.add_class::<PyEvaluationMapVjp>()
}
