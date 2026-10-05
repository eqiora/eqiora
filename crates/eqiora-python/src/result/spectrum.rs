//! Finite-window observations delegate to the native accepted trajectory owner.
use super::*;
use crate::model::PyObservableRef;
use crate::modeling::PyValueType;
use eqiora_numerics::{FiniteSpectrum, UniformDft};

/// Normalized sampled Fourier coefficients bound to exact trajectory and sampling semantics.
#[pyclass(name = "FiniteSpectrum", module = "eqiora._eqiora", frozen)]
pub(crate) struct PyFiniteSpectrum {
    value: FiniteSpectrum,
    #[pyo3(get)]
    result_identity: String,
}
#[pymethods]
impl PyFiniteSpectrum {
    #[getter]
    fn trajectory_identity(&self) -> &str {
        self.value.trajectory_identity()
    }
    #[getter]
    fn observable_id(&self) -> String {
        self.value.observable().ulid().to_string()
    }
    #[getter]
    fn interval_s(&self) -> (f64, f64) {
        let g = self.value.sampling();
        (g.start_s(), g.end_s())
    }
    #[getter]
    fn phase_reference_s(&self) -> f64 {
        self.value.sampling().start_s()
    }
    #[getter]
    fn spacing_s(&self) -> f64 {
        self.value.sampling().spacing_s()
    }
    #[getter]
    fn sample_count(&self) -> usize {
        self.value.sampling().count()
    }
    #[getter]
    fn window(&self) -> &'static str {
        match self.value.sampling() {
            UniformDft::Rectangular { .. } => "rectangular",
            UniformDft::PeriodicHann { .. } => "periodic-hann",
        }
    }
    #[getter]
    fn endpoint_convention(&self) -> &'static str {
        "half-open-uniform-samples"
    }
    #[getter]
    fn convention(&self) -> &'static str {
        "forward:+i,1/N;inverse:-i,1;phase:first-sample"
    }
    #[getter]
    fn input_type(&self) -> PyValueType {
        PyValueType {
            value: self.value.input_type().clone(),
        }
    }
    #[getter]
    fn coefficient_type(&self) -> PyValueType {
        PyValueType {
            value: self.value.coefficients()[0].value_type().clone(),
        }
    }
    #[getter]
    fn coefficients(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.value
            .coefficients()
            .iter()
            .map(|v| crate::modeling::value_literal::to_python(py, v))
            .collect()
    }
    fn frequency_hz(&self, py: Python<'_>, bin: usize) -> PyResult<f64> {
        self.value
            .frequency_hz(bin)
            .map_err(|e| diagnostic_error(py, &[e]))
    }
    fn angular_frequency_rad_s(&self, py: Python<'_>, bin: usize) -> PyResult<f64> {
        self.value
            .angular_frequency_rad_s(bin)
            .map_err(|e| diagnostic_error(py, &[e]))
    }
    #[pyo3(signature = (bin, *, component=None))]
    fn phase_rad(&self, py: Python<'_>, bin: usize, component: Option<usize>) -> PyResult<f64> {
        self.value
            .phase_rad(bin, self.component(component)?)
            .map_err(|e| diagnostic_error(py, &[e]))
    }
    #[pyo3(signature = (bin, *, component=None))]
    fn amplitude(
        &self,
        py: Python<'_>,
        bin: usize,
        component: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        literal(py, self.value.amplitude(bin, self.component(component)?))
    }
    #[pyo3(signature = (bin, *, component=None))]
    fn one_sided_amplitude(
        &self,
        py: Python<'_>,
        bin: usize,
        component: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        literal(
            py,
            self.value
                .one_sided_amplitude(bin, self.component(component)?),
        )
    }
    #[pyo3(signature = (bin, *, component=None))]
    fn power(&self, py: Python<'_>, bin: usize, component: Option<usize>) -> PyResult<Py<PyAny>> {
        literal(py, self.value.power(bin, self.component(component)?))
    }
    #[pyo3(signature = (bin, *, component=None))]
    fn power_density_per_hz(
        &self,
        py: Python<'_>,
        bin: usize,
        component: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        literal(
            py,
            self.value
                .power_density_per_hz(bin, self.component(component)?),
        )
    }
    #[pyo3(signature = (bin, *, component=None))]
    fn one_sided_power_density_per_hz(
        &self,
        py: Python<'_>,
        bin: usize,
        component: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        literal(
            py,
            self.value
                .one_sided_power_density_per_hz(bin, self.component(component)?),
        )
    }
    fn rectangle_transform_estimate(&self, py: Python<'_>, bin: usize) -> PyResult<Py<PyAny>> {
        literal(py, self.value.rectangle_transform_estimate(bin))
    }
    fn reconstruct_sample(&self, py: Python<'_>, sample: usize) -> PyResult<Py<PyAny>> {
        literal(py, self.value.reconstruct_sample(sample))
    }
    /// Exact physical type of a named projection, independent of bin magnitude.
    fn projection_type(&self, py: Python<'_>, projection: &str) -> PyResult<PyValueType> {
        let value = match projection {
            "amplitude" | "one-sided-amplitude" => self.value.amplitude(0, 0),
            "power" => self.value.power(0, 0),
            "power-density-per-hz" | "one-sided-power-density-per-hz" => {
                self.value.power_density_per_hz(0, 0)
            }
            "rectangle-transform-estimate" => self.value.rectangle_transform_estimate(0),
            _ => return Err(PyValueError::new_err("unknown finite spectrum projection")),
        }
        .map_err(|e| diagnostic_error(py, &[e]))?;
        Ok(PyValueType {
            value: value.value_type().clone(),
        })
    }
}
impl PyFiniteSpectrum {
    fn component(&self, selected: Option<usize>) -> PyResult<usize> {
        match selected {
            Some(index) => Ok(index),
            None if self.value.input_type().shape().is_scalar() => Ok(0),
            None => Err(PyValueError::new_err(
                "shaped spectrum projections require an explicit ordered component index",
            )),
        }
    }
}
fn literal(py: Python<'_>, value: Result<eqiora::ValueLiteral, Diagnostic>) -> PyResult<Py<PyAny>> {
    crate::modeling::value_literal::to_python(py, &value.map_err(|e| diagnostic_error(py, &[e]))?)
}
impl PyRunResult {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn spectrum(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        start_s: f64,
        spacing_s: f64,
        count: usize,
        window: &str,
        max_products: usize,
    ) -> PyResult<PyFiniteSpectrum> {
        if observable.model_digest != self.identity.model_digest() {
            return Err(PyValueError::new_err(
                "ObservableRef belongs to a different exact Model artifact",
            ));
        }
        let window = match window {
            "rectangular" => UniformDft::Rectangular {
                start_s,
                spacing_s,
                count,
            },
            "periodic-hann" => UniformDft::PeriodicHann {
                start_s,
                spacing_s,
                count,
            },
            _ => {
                return Err(PyValueError::new_err(
                    "window must be rectangular or periodic-hann",
                ));
            }
        };
        let grid = window.validate().map_err(|e| diagnostic_error(py, &[e]))?;
        let trajectory = self.native.trajectory().ok_or_else(|| {
            PyValueError::new_err("finite spectrum requires an accepted Trajectory")
        })?;
        let value = trajectory
            .observe_spectrum(
                self.native.plan().model_artifact(),
                observable.id,
                grid,
                max_products,
            )
            .map_err(|e| diagnostic_error(py, &[e]))?;
        Ok(PyFiniteSpectrum {
            value,
            result_identity: self.native.identity().to_owned(),
        })
    }
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyFiniteSpectrum>()
}
