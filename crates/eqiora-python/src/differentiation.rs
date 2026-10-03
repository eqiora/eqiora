//! Thin Python adapter for exact accepted-point differentiation programs.

use std::sync::Arc;

use eqiora::api::{
    DerivativeImplementation, DifferentiableEvaluation, DifferentiableJvp, DifferentiablePrimal,
    DifferentiableProgram, DifferentiableVjp, DifferentiationEvidence, DifferentiationMode,
    LinearizationState,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PySequence};

use crate::array::{PyArrayBuffer, stage_f64_input};
use crate::common_plan::PyPlan;
use crate::error::{diagnostic_error, panic_boundary, validation_error};
use crate::model::{PyModelFieldRef, PyModelParameterRef};
use crate::realization::PyLinearSolveSummary;

mod batch;

/// Primal, JVP, or VJP occurrence.
#[pyclass(
    name = "DifferentiationMode",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyDifferentiationMode {
    Primal,
    Jvp,
    Vjp,
}

impl From<DifferentiationMode> for PyDifferentiationMode {
    fn from(value: DifferentiationMode) -> Self {
        match value {
            DifferentiationMode::Primal => Self::Primal,
            DifferentiationMode::Jvp => Self::Jvp,
            DifferentiationMode::Vjp => Self::Vjp,
        }
    }
}

/// Derivative action source.
#[pyclass(
    name = "DerivativeImplementation",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyDerivativeImplementation {
    AnalyticAssembled,
    OperatorIr,
}

impl From<DerivativeImplementation> for PyDerivativeImplementation {
    fn from(value: DerivativeImplementation) -> Self {
        match value {
            DerivativeImplementation::AnalyticAssembled => Self::AnalyticAssembled,
            DerivativeImplementation::OperatorIr => Self::OperatorIr,
        }
    }
}

/// Accepted-linearization reuse disposition.
#[pyclass(
    name = "LinearizationState",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyLinearizationState {
    Established,
    Reused,
}

impl From<LinearizationState> for PyLinearizationState {
    fn from(value: LinearizationState) -> Self {
        match value {
            LinearizationState::Established => Self::Established,
            LinearizationState::Reused => Self::Reused,
        }
    }
}

/// Typed in-memory provenance for one differentiation occurrence.
#[pyclass(
    name = "DifferentiationEvidence",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyDifferentiationEvidence {
    model_digest: String,
    plan_identity: String,
    input_ids: Vec<String>,
    output_id: String,
    mode: PyDifferentiationMode,
    implementation: PyDerivativeImplementation,
    linearization_state: PyLinearizationState,
    state_system_fingerprint: String,
    primal_residual_norm: f64,
    residual_tolerance: f64,
    primal_solve: Option<PyLinearSolveSummary>,
    derivative_solve: Option<PyLinearSolveSummary>,
}

impl PyDifferentiationEvidence {
    fn from_value(value: &DifferentiationEvidence) -> Self {
        let identity = value.identity();
        Self {
            model_digest: identity.model().artifact().to_string(),
            plan_identity: identity.plan_identity().to_owned(),
            input_ids: identity
                .inputs()
                .iter()
                .map(|id| id.ulid().to_string())
                .collect(),
            output_id: identity.output().ulid().to_string(),
            mode: value.mode().into(),
            implementation: value.implementation().into(),
            linearization_state: value.linearization_state().into(),
            state_system_fingerprint: hex(value.state_system().as_bytes()),
            primal_residual_norm: value.primal_residual_norm(),
            residual_tolerance: value.residual_tolerance(),
            primal_solve: value.primal_solve().map(PyLinearSolveSummary::from_report),
            derivative_solve: value
                .derivative_solve()
                .map(PyLinearSolveSummary::from_report),
        }
    }
}

/// The leading twelve characters of a content digest.
///
/// A repr exists to be read. A full 64-character digest pushes every other
/// field off the line and identifies nothing a reader could not get from the
/// first few bytes, so the prefix is shown and the getter remains authoritative.
fn short_digest(digest: &str) -> &str {
    digest.get(..12).unwrap_or(digest)
}

#[pymethods]
impl PyDifferentiationEvidence {
    /// The mode, the implementation, and whether the primal met its tolerance.
    fn __repr__(&self) -> String {
        format!(
            "DifferentiationEvidence(mode={:?}, implementation={:?}, output_id={:?}, primal_residual_norm={:e}, residual_tolerance={:e})",
            self.mode,
            self.implementation,
            self.output_id,
            self.primal_residual_norm,
            self.residual_tolerance
        )
    }
    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }

    #[getter]
    fn plan_identity(&self) -> &str {
        &self.plan_identity
    }

    #[getter]
    fn input_ids(&self) -> Vec<String> {
        self.input_ids.clone()
    }

    #[getter]
    fn output_id(&self) -> &str {
        &self.output_id
    }

    #[getter]
    const fn mode(&self) -> PyDifferentiationMode {
        self.mode
    }

    #[getter]
    const fn implementation(&self) -> PyDerivativeImplementation {
        self.implementation
    }

    #[getter]
    const fn linearization_state(&self) -> PyLinearizationState {
        self.linearization_state
    }

    #[getter]
    fn state_system_fingerprint(&self) -> &str {
        &self.state_system_fingerprint
    }

    #[getter]
    const fn primal_residual_norm(&self) -> f64 {
        self.primal_residual_norm
    }

    #[getter]
    const fn residual_tolerance(&self) -> f64 {
        self.residual_tolerance
    }

    #[getter]
    fn primal_solve(&self) -> Option<PyLinearSolveSummary> {
        self.primal_solve.clone()
    }

    #[getter]
    fn derivative_solve(&self) -> Option<PyLinearSolveSummary> {
        self.derivative_solve.clone()
    }
}

/// Accepted complete primary Field.
#[pyclass(
    name = "DifferentiablePrimal",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyDifferentiablePrimal {
    output: Py<PyArrayBuffer>,
    evidence: PyDifferentiationEvidence,
}

#[pymethods]
impl PyDifferentiablePrimal {
    /// The output it accepted and the residual that accepted it.
    fn __repr__(&self) -> String {
        format!(
            "DifferentiablePrimal(output_id={:?}, primal_residual_norm={:e})",
            self.evidence.output_id, self.evidence.primal_residual_norm
        )
    }
    #[getter]
    fn output(&self, py: Python<'_>) -> Py<PyArrayBuffer> {
        self.output.clone_ref(py)
    }

    #[getter]
    fn evidence(&self) -> PyDifferentiationEvidence {
        self.evidence.clone()
    }
}

/// Accepted complete primary Field and its forward tangent.
#[pyclass(
    name = "DifferentiableJvp",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyDifferentiableJvp {
    output: Py<PyArrayBuffer>,
    tangent: Py<PyArrayBuffer>,
    evidence: PyDifferentiationEvidence,
}

#[pymethods]
impl PyDifferentiableJvp {
    /// The output it accepted, in forward mode.
    fn __repr__(&self) -> String {
        format!(
            "DifferentiableJvp(output_id={:?}, mode={:?}, primal_residual_norm={:e})",
            self.evidence.output_id, self.evidence.mode, self.evidence.primal_residual_norm
        )
    }
    #[getter]
    fn output(&self, py: Python<'_>) -> Py<PyArrayBuffer> {
        self.output.clone_ref(py)
    }

    #[getter]
    fn tangent(&self, py: Python<'_>) -> Py<PyArrayBuffer> {
        self.tangent.clone_ref(py)
    }

    #[getter]
    fn evidence(&self) -> PyDifferentiationEvidence {
        self.evidence.clone()
    }
}

/// Accepted complete primary Field and its reverse input cotangent.
#[pyclass(
    name = "DifferentiableVjp",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyDifferentiableVjp {
    output: Py<PyArrayBuffer>,
    input_cotangent: Py<PyArrayBuffer>,
    evidence: PyDifferentiationEvidence,
}

#[pymethods]
impl PyDifferentiableVjp {
    /// The output it accepted, in reverse mode.
    fn __repr__(&self) -> String {
        format!(
            "DifferentiableVjp(output_id={:?}, mode={:?}, primal_residual_norm={:e})",
            self.evidence.output_id, self.evidence.mode, self.evidence.primal_residual_norm
        )
    }
    #[getter]
    fn output(&self, py: Python<'_>) -> Py<PyArrayBuffer> {
        self.output.clone_ref(py)
    }

    #[getter]
    fn input_cotangent(&self, py: Python<'_>) -> Py<PyArrayBuffer> {
        self.input_cotangent.clone_ref(py)
    }

    #[getter]
    fn evidence(&self) -> PyDifferentiationEvidence {
        self.evidence.clone()
    }
}

/// Opaque immutable accepted evaluation at one numerical Parameter point.
#[pyclass(
    name = "DifferentiableEvaluation",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyDifferentiableEvaluation {
    value: Arc<DifferentiableEvaluation>,
    point: Py<PyArrayBuffer>,
}

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
}

/// Opaque immutable program over one fixed input coordinate set.
#[pyclass(
    name = "DifferentiableProgram",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyDifferentiableProgram {
    value: Arc<DifferentiableProgram>,
}

#[pymethods]
impl PyDifferentiableProgram {
    /// The model and exact common Plan this program was lowered from.
    fn __repr__(&self) -> String {
        format!(
            "DifferentiableProgram(model_digest={:?}, plan_identity={:?}, output_id={:?})",
            short_digest(&self.model_digest()),
            short_digest(self.plan_identity()),
            self.output_id()
        )
    }
    #[getter]
    fn model_digest(&self) -> String {
        self.value.identity().model().artifact().to_string()
    }

    #[getter]
    fn plan_identity(&self) -> &str {
        self.value.identity().plan_identity()
    }

    #[getter]
    fn input_ids(&self) -> Vec<String> {
        self.value
            .identity()
            .inputs()
            .iter()
            .map(|id| id.ulid().to_string())
            .collect()
    }

    #[getter]
    fn output_id(&self) -> String {
        self.value.identity().output().ulid().to_string()
    }

    #[getter]
    fn input_shape(&self) -> (usize,) {
        (self.value.identity().input_dimension(),)
    }

    #[getter]
    fn output_shape(&self) -> (usize,) {
        (self.value.identity().output_dimension(),)
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        "float64"
    }

    #[getter]
    fn device(&self) -> &'static str {
        "cpu:0"
    }

    #[getter]
    fn derivative_contract(&self) -> &'static str {
        "implicit-first-order"
    }

    /// Retain this immutable program behind its deterministic native FFI key.
    ///
    /// This is a private adapter seam used only by ``eqiora.jax``. The key is
    /// static program identity, never a pointer or registration-order token.
    fn _jax_ffi_register(&self, py: Python<'_>) -> PyResult<String> {
        panic_boundary(py, || {
            crate::jax_ffi::register_program(Arc::clone(&self.value))
                .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))
        })
    }

    fn evaluate(
        &self,
        py: Python<'_>,
        parameters: &Bound<'_, PyAny>,
    ) -> PyResult<PyDifferentiableEvaluation> {
        panic_boundary(py, || {
            let parameters = stage_f64_input(
                py,
                parameters,
                self.value.identity().input_dimension(),
                "parameters",
            )?;
            let program = Arc::clone(&self.value);
            let result = py
                .detach(move || program.evaluate(&parameters))
                .map_err(|diagnostics| diagnostic_error(py, &diagnostics))?;
            evaluation_result(py, result)
        })
    }

    /// Freeze an ordered tensor of mapped inputs and explicit global sharing.
    #[pyo3(signature = (mapped, *, shared_inputs=None, shared=None, retained_bytes_limit=67_108_864))]
    fn map(
        &self,
        py: Python<'_>,
        mapped: &Bound<'_, PyAny>,
        shared_inputs: Option<&Bound<'_, PyAny>>,
        shared: Option<&Bound<'_, PyAny>>,
        retained_bytes_limit: usize,
    ) -> PyResult<batch::PyEvaluationMapPlan> {
        panic_boundary(py, || {
            batch::plan(
                py,
                self,
                mapped,
                shared_inputs,
                shared,
                retained_bytes_limit,
            )
        })
    }

    fn primal(&self, py: Python<'_>) -> PyResult<PyDifferentiablePrimal> {
        panic_boundary(py, || {
            let program = Arc::clone(&self.value);
            let result = py.detach(move || program.primal());
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
            let program = Arc::clone(&self.value);
            let result = py
                .detach(move || program.jvp(&tangent))
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
            let program = Arc::clone(&self.value);
            let result = py
                .detach(move || program.vjp(&cotangent))
                .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?;
            vjp_result(py, result)
        })
    }
}

/// Compile one exact accepted-point differentiable program.
#[pyfunction(name = "_compile_differentiable")]
#[pyo3(signature = (plan, /, *, inputs, output))]
pub(crate) fn compile_differentiable(
    py: Python<'_>,
    plan: &PyPlan,
    inputs: &Bound<'_, PyAny>,
    output: &PyModelFieldRef,
) -> PyResult<PyDifferentiableProgram> {
    panic_boundary(py, || {
        let sequence = inputs.cast::<PySequence>().map_err(|_| {
            validation_error(
                py,
                &[eqiora::Diagnostic::error(
                    eqiora::diagnostic::codes::INVALID_LINEARIZATION,
                    "differentiable inputs must be an ordered sequence of ParameterRef values",
                )],
            )
        })?;
        let mut selected = Vec::with_capacity(sequence.len()?);
        for index in 0..sequence.len()? {
            let item = sequence.get_item(index)?;
            let parameter = item.extract::<PyRef<'_, PyModelParameterRef>>()?;
            selected.push(parameter.value.clone());
        }
        let native_plan = plan.scalar_native().cloned().ok_or_else(|| {
            validation_error(
                py,
                &[eqiora::Diagnostic::error(
                    eqiora::diagnostic::codes::INVALID_LINEARIZATION,
                    "differentiation requires a 2D scalar Q1 or cell-centered TPFA Plan",
                )],
            )
        })?;
        let model = plan.model_handle(py);
        let document = model
            .bind(py)
            .borrow()
            .document()
            .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?
            .clone();
        let output_model_digest = output.exact_model_digest().to_owned();
        let output = document
            .field_ref(output.exact_id())
            .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?;
        if output.model().artifact().to_string() != output_model_digest {
            return Err(validation_error(
                py,
                &[eqiora::Diagnostic::error(
                    eqiora::diagnostic::codes::INVALID_LINEARIZATION,
                    "differentiable output FieldRef belongs to another exact Model artifact",
                )],
            ));
        }
        let value = py
            .detach(move || {
                DifferentiableProgram::compile(
                    eqiora_numerics::ResolvedCommonPlan::Scalar(Box::new(native_plan)),
                    &selected,
                    &output,
                    None,
                    &eqiora::solver::REFERENCE_LINEAR_SOLVER,
                )
            })
            .map_err(|diagnostics| diagnostic_error(py, &diagnostics))?;
        Ok(PyDifferentiableProgram {
            value: Arc::new(value),
        })
    })
}

fn primal_result(py: Python<'_>, value: DifferentiablePrimal) -> PyResult<PyDifferentiablePrimal> {
    let (output, evidence) = value.into_parts();
    Ok(PyDifferentiablePrimal {
        output: PyArrayBuffer::from_owned_result(py, output)?,
        evidence: PyDifferentiationEvidence::from_value(&evidence),
    })
}

fn jvp_result(py: Python<'_>, value: DifferentiableJvp) -> PyResult<PyDifferentiableJvp> {
    let (output, tangent, evidence) = value.into_parts();
    Ok(PyDifferentiableJvp {
        output: PyArrayBuffer::from_owned_result(py, output)?,
        tangent: PyArrayBuffer::from_owned_result(py, tangent)?,
        evidence: PyDifferentiationEvidence::from_value(&evidence),
    })
}

fn vjp_result(py: Python<'_>, value: DifferentiableVjp) -> PyResult<PyDifferentiableVjp> {
    let (output, input_cotangent, evidence) = value.into_parts();
    Ok(PyDifferentiableVjp {
        output: PyArrayBuffer::from_owned_result(py, output)?,
        input_cotangent: PyArrayBuffer::from_owned_result(py, input_cotangent)?,
        evidence: PyDifferentiationEvidence::from_value(&evidence),
    })
}

fn evaluation_result(
    py: Python<'_>,
    value: DifferentiableEvaluation,
) -> PyResult<PyDifferentiableEvaluation> {
    let point = PyArrayBuffer::from_owned_result(py, value.point().values().to_vec())?;
    Ok(PyDifferentiableEvaluation {
        value: Arc::new(value),
        point,
    })
}

fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    batch::register(module)?;
    module.add_class::<PyDifferentiationMode>()?;
    module.add_class::<PyDerivativeImplementation>()?;
    module.add_class::<PyLinearizationState>()?;
    module.add_class::<PyDifferentiationEvidence>()?;
    module.add_class::<PyDifferentiablePrimal>()?;
    module.add_class::<PyDifferentiableJvp>()?;
    module.add_class::<PyDifferentiableVjp>()?;
    module.add_class::<PyDifferentiableEvaluation>()?;
    module.add_class::<PyDifferentiableProgram>()?;
    module.add_function(wrap_pyfunction!(compile_differentiable, module)?)?;
    Ok(())
}
