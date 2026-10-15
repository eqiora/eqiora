//! Shaped immutable projections of complete native first-order actions.

use super::*;
use eqiora::api::{EvaluationMapJvp, EvaluationMapVjp};

/// Complete batched JVP values with unchanged per-action accepted evidence.
#[pyclass(name = "EvaluationMapJvp", module = "eqiora._eqiora", frozen)]
pub(super) struct PyEvaluationMapJvp {
    pub(super) native: EvaluationMapJvp,
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
    fn output(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        readonly_array(
            py,
            self.native
                .products()
                .iter()
                .flat_map(|p| p.output().coordinates().iter().copied())
                .collect(),
            self.native.shape(),
        )
    }
    #[getter]
    fn tangent(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        readonly_array(
            py,
            self.native
                .products()
                .iter()
                .flat_map(|p| {
                    values::coordinate_projection(p.derivative().base_signatures(), |index| {
                        p.derivative().components(index).expect("admitted leaf")
                    })
                })
                .collect(),
            self.native.shape(),
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
    pub(super) native: EvaluationMapVjp,
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
    fn shared_input_ids(&self) -> Vec<String> {
        ids(self.native.shared_inputs())
    }
    #[getter]
    fn mapped_input_ids(&self) -> Vec<String> {
        ids(self.native.mapped_inputs())
    }
    #[getter]
    fn shared_cotangents(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        readonly_array(
            py,
            self.native.shared_cotangents().to_vec(),
            self.native.shared_shape(),
        )
    }
    #[getter]
    fn mapped_cotangents(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        readonly_array(
            py,
            self.native.mapped_cotangents().to_vec(),
            self.native.mapped_shape(),
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
