//! Immutable point operations and staged partial actions.
use super::*;

#[pymethods]
impl PyDifferentiableEvaluation {
    /// The Parameter point this evaluation is bound to.
    fn __repr__(&self, py: Python<'_>) -> String {
        let point = self.point.bind(py).as_any().len().unwrap_or(0);
        format!("DifferentiableEvaluation(point={point} values)")
    }
    /// Complete accepted Parameter point in exact program input order.
    #[getter]
    fn point(&self, py: Python<'_>) -> Py<PyArrayBuffer> {
        self.point.clone_ref(py)
    }

    fn primal(&self, py: Python<'_>) -> PyResult<PyDifferentiablePrimal> {
        panic_boundary(py, || {
            let evaluation = Arc::clone(&self.value);
            let result = py.detach(move || evaluation.primal());
            primal_result(py, result)
        })
    }

    fn jvp(&self, py: Python<'_>, tangent: &Bound<'_, PyAny>) -> PyResult<PyDifferentiableJvp> {
        panic_boundary(py, || {
            let tangent = stage_f64_input(
                py,
                tangent,
                self.value.identity().input_dimension(),
                "tangent",
            )?;
            let evaluation = Arc::clone(&self.value);
            let result = py
                .detach(move || evaluation.jvp(&tangent))
                .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?;
            jvp_result(py, result)
        })
    }

    fn vjp(&self, py: Python<'_>, cotangent: &Bound<'_, PyAny>) -> PyResult<PyDifferentiableVjp> {
        panic_boundary(py, || {
            let cotangent = stage_f64_input(
                py,
                cotangent,
                self.value.identity().output_dimension(),
                "cotangent",
            )?;
            let evaluation = Arc::clone(&self.value);
            let result = py
                .detach(move || evaluation.vjp(&cotangent))
                .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?;
            vjp_result(py, result)
        })
    }

    /// Accepted unknown coordinates in native relation order.
    #[getter]
    fn accepted_unknowns(&self, py: Python<'_>) -> PyResult<Py<PyArrayBuffer>> {
        PyArrayBuffer::from_owned_result(py, self.value.accepted_unknowns().to_vec())
    }

    fn residual_jvp(
        &self,
        py: Python<'_>,
        unknown: &Bound<'_, PyAny>,
        parameter: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyArrayBuffer>> {
        self.partial_jvp(
            py,
            unknown,
            parameter,
            DifferentiableEvaluation::residual_jvp,
        )
    }
    fn output_partial_jvp(
        &self,
        py: Python<'_>,
        unknown: &Bound<'_, PyAny>,
        parameter: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyArrayBuffer>> {
        self.partial_jvp(
            py,
            unknown,
            parameter,
            DifferentiableEvaluation::output_partial_jvp,
        )
    }
    fn residual_vjp(
        &self,
        py: Python<'_>,
        cotangent: &Bound<'_, PyAny>,
    ) -> PyResult<(Py<PyArrayBuffer>, Py<PyArrayBuffer>)> {
        self.partial_vjp(
            py,
            cotangent,
            self.value.accepted_unknowns().len(),
            DifferentiableEvaluation::residual_vjp,
        )
    }
    fn output_partial_vjp(
        &self,
        py: Python<'_>,
        cotangent: &Bound<'_, PyAny>,
    ) -> PyResult<(Py<PyArrayBuffer>, Py<PyArrayBuffer>)> {
        self.partial_vjp(
            py,
            cotangent,
            self.value.identity().output_dimension(),
            DifferentiableEvaluation::output_partial_vjp,
        )
    }
}

type PartialJvp =
    fn(&DifferentiableEvaluation, &[f64], &[f64]) -> Result<Vec<f64>, eqiora::Diagnostic>;
type PartialVjp =
    fn(&DifferentiableEvaluation, &[f64]) -> Result<(Vec<f64>, Vec<f64>), eqiora::Diagnostic>;

impl PyDifferentiableEvaluation {
    fn partial_jvp(
        &self,
        py: Python<'_>,
        unknown: &Bound<'_, PyAny>,
        parameter: &Bound<'_, PyAny>,
        action: PartialJvp,
    ) -> PyResult<Py<PyArrayBuffer>> {
        panic_boundary(py, || {
            let unknown = stage_f64_input(
                py,
                unknown,
                self.value.accepted_unknowns().len(),
                "unknown tangent",
            )?;
            let parameter = stage_f64_input(
                py,
                parameter,
                self.value.identity().input_dimension(),
                "Parameter tangent",
            )?;
            let value = Arc::clone(&self.value);
            let result = py
                .detach(move || action(&value, &unknown, &parameter))
                .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?;
            PyArrayBuffer::from_owned_result(py, result)
        })
    }
    fn partial_vjp(
        &self,
        py: Python<'_>,
        cotangent: &Bound<'_, PyAny>,
        dimension: usize,
        action: PartialVjp,
    ) -> PyResult<(Py<PyArrayBuffer>, Py<PyArrayBuffer>)> {
        panic_boundary(py, || {
            let cotangent = stage_f64_input(py, cotangent, dimension, "cotangent")?;
            let value = Arc::clone(&self.value);
            let (unknown, parameter) = py
                .detach(move || action(&value, &cotangent))
                .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?;
            Ok((
                PyArrayBuffer::from_owned_result(py, unknown)?,
                PyArrayBuffer::from_owned_result(py, parameter)?,
            ))
        })
    }
}
