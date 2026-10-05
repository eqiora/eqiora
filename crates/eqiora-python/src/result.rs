//! Common installed-Python ownership for accepted execution results.

use std::collections::BTreeMap;

use eqiora::diagnostic::codes;
use eqiora::{Diagnostic, DimExponents};
use eqiora_numerics::ResolvedCommonPlan;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyModule};

use crate::array::PyArrayBuffer;
use crate::common_plan::PyPlan;
use crate::diagnostic_error;
use crate::elasticity::PyLinearElasticityEvidence;
use crate::execution::RunIdentity;
use crate::fsi_evidence::PyFsiEvidence;
use crate::geometry::PyGeometrySelection;
use crate::meshing::PyMesh;
use crate::model::PyModelFieldRef;
use crate::model_io::{
    ArtifactFileSpec, read_artifact_bytes, unicode_artifact_path, write_artifact_bytes,
};
use crate::realization::PyLinearSolveSummary;
use crate::steady_stokes::PySteadyStokesEvidence;
use crate::trajectory::{PyBoundaryFlux, PyBoundaryForce, PyState, PyTrajectory};

mod constraints;
mod field_output;
mod nonlinear;
pub(crate) use nonlinear::PyNonlinearSolveSummary;
mod eigen;
mod materialize;
pub(crate) use materialize::materialize_common_result;
mod observe;
mod time_observe;

use field_output::FieldOutputBlock;
pub(crate) use field_output::PyFieldOutput;

const RESULT_FILE_SPEC: ArtifactFileSpec = ArtifactFileSpec {
    artifact_name: "complete Result",
    extension: "eqresult",
    staging_name: "result",
    // This is the pre-read counterpart of the canonical Result decoder's bound.
    max_bytes: 512 * 1024 * 1024,
};

/// One read-only, field-local sampled series in SI units.
#[pyclass(name = "Series", module = "eqiora._eqiora", frozen)]
pub(crate) struct PySeries {
    field: Option<Py<PyModelFieldRef>>,
    derivative_order: u32,
    component: usize,
    imaginary: bool,
    id: String,
    name: Option<String>,
    dimension: DimExponents,
    time: Py<PyArrayBuffer>,
    values: Py<PyArrayBuffer>,
}

#[pymethods]
impl PySeries {
    #[getter]
    fn field(&self, py: Python<'_>) -> Option<Py<PyModelFieldRef>> {
        self.field.as_ref().map(|field| field.clone_ref(py))
    }
    #[getter]
    fn component(&self) -> usize {
        self.component
    }
    #[getter]
    fn imaginary(&self) -> bool {
        self.imaginary
    }
    #[getter]
    fn derivative_order(&self) -> u32 {
        self.derivative_order
    }
    #[getter]
    fn id(&self) -> &str {
        &self.id
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    #[getter]
    fn dimension(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::modeling::dimension::exponents(py, self.dimension)
    }

    #[getter]
    fn time(&self, py: Python<'_>) -> Py<PyArrayBuffer> {
        self.time.clone_ref(py)
    }

    #[getter]
    fn values(&self, py: Python<'_>) -> Py<PyArrayBuffer> {
        self.values.clone_ref(py)
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.values.borrow(py).len())
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let time = self.time.borrow(py).snapshot(py)?;
        let values = self.values.borrow(py).snapshot(py)?;
        if time.len() != values.len() {
            return Err(PyRuntimeError::new_err(
                "Series time and value buffers report different lengths",
            ));
        }
        let samples = PyList::new(py, time.into_iter().zip(values))?;
        Ok(samples.as_any().try_iter()?.into_any().unbind())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let label = self.name.as_deref().unwrap_or(&self.id);
        Ok(format!("Series({label:?}, samples={})", self.__len__(py)?))
    }
}

enum StaticScientificEvidence {
    SteadyStokes(Py<PySteadyStokesEvidence>),
    LinearElasticity(Py<PyLinearElasticityEvidence>),
}

struct CommonFieldResultPayload {
    outputs: Vec<Py<PyFieldOutput>>,
    lookup: BTreeMap<String, usize>,
    solve: Py<PyAny>,
    evidence: Option<StaticScientificEvidence>,
}

struct CommonTrajectoryResultPayload {
    trajectory: Py<PyTrajectory>,
    fsi_evidence: Option<Py<PyFsiEvidence>>,
}

struct CommonOdeResultPayload {
    fields: Vec<Py<PySeries>>,
    lookup: BTreeMap<(String, u32, usize, bool), usize>,
    states: Vec<eqiora_numerics::CommonOdeState>,
}

enum ResultPayload {
    Eigen,
    Fields(Box<CommonFieldResultPayload>),
    Trajectory(CommonTrajectoryResultPayload),
    Ode(CommonOdeResultPayload),
}

/// One accepted execution occurrence with typed output relationships.
#[pyclass(
    name = "Result",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyRunResult {
    native: eqiora_numerics::CommonResult,
    identity: RunIdentity,
    elapsed_seconds: f64,
    payload: ResultPayload,
    profile: Option<Py<crate::profile::PyProfile>>,
}

impl PyRunResult {
    pub(crate) fn common_state_at(
        &self,
        py: Python<'_>,
        state_space_identity: &str,
        time_s: f64,
    ) -> Option<Py<PyState>> {
        let ResultPayload::Trajectory(payload) = &self.payload else {
            return None;
        };
        payload
            .trajectory
            .borrow(py)
            .state_handles(py)
            .into_iter()
            .find(|state| {
                let state = state.borrow(py);
                state
                    .common_native()
                    .is_some_and(|native| native.state_space_identity() == state_space_identity)
                    && state.time_s_value().to_bits() == time_s.to_bits()
            })
    }
}

#[pymethods]
impl PyRunResult {
    #[getter(eigenpair_count)]
    fn eigenpair_count_python(&self) -> usize {
        self.eigenpair_count()
    }
    #[getter(eigen_convergence)]
    fn eigen_convergence_python(&self) -> Option<&'static str> {
        self.eigen_convergence()
    }
    #[getter(eigen_candidate_counts)]
    fn eigen_candidate_counts_python(&self) -> Option<(usize, usize)> {
        self.eigen_candidate_counts()
    }
    fn eigenpair(&self, index: usize) -> PyResult<eigen::PyEigenpair> {
        self.eigenpair_value(index)
    }
    fn eigenprojector(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
    ) -> PyResult<(Py<PyAny>, crate::modeling::PyValueType)> {
        self.eigenprojector_value(py, indices)
    }

    #[getter]
    fn original_residual_norm(&self) -> Option<f64> {
        self.native.original_residual_norm()
    }
    #[getter]
    fn compatibility_residual(&self) -> Option<f64> {
        self.native.compatibility_residual()
    }
    #[getter]
    fn gauge_residual(&self) -> Option<f64> {
        self.native.gauge_residual()
    }
    #[getter]
    fn gauge_multiplier(&self) -> Option<f64> {
        self.native.gauge_multiplier()
    }

    /// Independently evaluated original mathematical conditions with operand units.
    #[getter]
    fn constraints(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        constraints::measurements(self, py)
    }

    /// Evaluate at the accepted terminal State, independently of output cadence.
    fn observe_terminal(
        &self,
        py: Python<'_>,
        observable: &crate::model::PyObservableRef,
    ) -> PyResult<time_observe::PyTrajectoryObservation> {
        self.observe_trajectory(py, observable, None)
    }

    /// Integrate accepted history with an explicitly selected quadrature policy.
    #[pyo3(signature = (observable, *, quadrature))]
    fn observe_time_integral(
        &self,
        py: Python<'_>,
        observable: &crate::model::PyObservableRef,
        quadrature: time_observe::PyTimeFunctionalQuadrature,
    ) -> PyResult<time_observe::PyTrajectoryObservation> {
        self.observe_trajectory(py, observable, Some(quadrature))
    }

    /// Apply one exact unit-bearing Parameter direction at the fixed terminal time.
    fn observe_terminal_parameter_jvp(
        &self,
        py: Python<'_>,
        observable: &crate::model::PyObservableRef,
        direction: &Bound<'_, pyo3::types::PyDict>,
    ) -> PyResult<time_observe::PyTrajectoryObservation> {
        self.observe_trajectory_parameter_jvp(py, observable, None, direction)
    }

    /// Apply one exact Parameter direction to the accepted-step time integral.
    #[pyo3(signature = (observable, direction, *, quadrature))]
    fn observe_time_integral_parameter_jvp(
        &self,
        py: Python<'_>,
        observable: &crate::model::PyObservableRef,
        direction: &Bound<'_, pyo3::types::PyDict>,
        quadrature: time_observe::PyTimeFunctionalQuadrature,
    ) -> PyResult<time_observe::PyTrajectoryObservation> {
        self.observe_trajectory_parameter_jvp(py, observable, Some(quadrature), direction)
    }

    /// Evaluate one exact derived output; spatial reductions require points per axis.
    #[pyo3(signature = (observable, *, quadrature_points=None))]
    fn observe(
        &self,
        py: Python<'_>,
        observable: &crate::model::PyObservableRef,
        quadrature_points: Option<usize>,
    ) -> PyResult<observe::PyObservation> {
        self.observe_value(py, observable, quadrature_points)
    }

    /// Sample a field-valued Observable on its exact output support.
    #[pyo3(signature = (observable, coordinates, *, quadrature_points=None))]
    fn observe_at(
        &self,
        py: Python<'_>,
        observable: &crate::model::PyObservableRef,
        coordinates: Vec<(Py<PyAny>, Py<crate::modeling::PyDimension>)>,
        quadrature_points: Option<usize>,
    ) -> PyResult<observe::PyObservation> {
        self.observe_point(py, observable, coordinates, quadrature_points)
    }

    /// Bind explicit SI dimensions and vertex coefficients to this exact Result.
    fn observable_state_tangent(
        &self,
        py: Python<'_>,
        directions: &Bound<'_, pyo3::types::PyDict>,
    ) -> PyResult<observe::PyObservableStateTangent> {
        self.bind_observable_tangent(py, directions)
    }

    /// Apply a bound field-state direction with Model parameters and geometry fixed.
    #[pyo3(signature = (observable, tangent, *, quadrature_points))]
    fn observe_state_jvp(
        &self,
        py: Python<'_>,
        observable: &crate::model::PyObservableRef,
        tangent: &observe::PyObservableStateTangent,
        quadrature_points: usize,
    ) -> PyResult<observe::PyObservation> {
        self.observe_jvp(py, observable, tangent, quadrature_points)
    }

    /// Apply two ordered directions to the same fixed polynomial energy.
    #[pyo3(signature = (observable, first, second, *, wrt, quadrature_points))]
    fn observe_state_second_variation(
        &self,
        py: Python<'_>,
        observable: &crate::model::PyObservableRef,
        first: &observe::PyObservableStateTangent,
        second: &observe::PyObservableStateTangent,
        wrt: &PyModelFieldRef,
        quadrature_points: usize,
    ) -> PyResult<observe::PyObservation> {
        self.observe_second_variation(py, observable, [first, second], wrt, quadrature_points)
    }

    #[getter]
    fn model_id(&self) -> &str {
        self.identity.model_id()
    }

    #[getter]
    fn model_digest(&self) -> &str {
        self.identity.model_digest()
    }

    #[getter]
    const fn model_revision(&self) -> u64 {
        self.identity.model_revision()
    }

    #[getter]
    fn plan_key(&self) -> &str {
        self.identity.plan_key()
    }

    #[getter]
    fn adapter(&self) -> &'static str {
        self.identity.adapter()
    }

    #[getter]
    fn adapter_version(&self) -> &'static str {
        self.identity.adapter_version()
    }

    #[getter]
    const fn elapsed_seconds(&self) -> f64 {
        self.elapsed_seconds
    }

    #[getter]
    fn profile(&self, py: Python<'_>) -> Option<Py<crate::profile::PyProfile>> {
        self.profile.as_ref().map(|profile| profile.clone_ref(py))
    }

    /// Canonical complete Result bytes, including fields and accepted evidence.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.native
            .to_bytes()
            .map(|bytes| PyBytes::new(py, &bytes))
            .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))
    }

    /// Decode one complete Result against its exact owning Plan.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, plan: PyRef<'_, PyPlan>, data: &[u8]) -> PyResult<Self> {
        let native = eqiora_numerics::CommonResult::from_bytes(data, plan.native())
            .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
        let identity = RunIdentity::from_common_result(&native).ok_or_else(|| {
            PyRuntimeError::new_err("Result artifact has no valid execution occurrence")
        })?;
        materialize_common_result(py, plan, identity, native, None)
    }

    /// Atomically write this exact complete Result to an `.eqresult` file.
    fn write(&self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = unicode_artifact_path(py, path)?;
        let bytes = self
            .native
            .to_bytes()
            .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
        py.detach(move || write_artifact_bytes(&path, &bytes, RESULT_FILE_SPEC))
            .map_err(|diagnostic| crate::error::compatibility_error(py, &[diagnostic]))
    }

    /// Read one complete Result against its exact owning Plan.
    #[staticmethod]
    fn read(py: Python<'_>, plan: PyRef<'_, PyPlan>, path: &Bound<'_, PyAny>) -> PyResult<Self> {
        let path = unicode_artifact_path(py, path)?;
        let bytes = py
            .detach(move || read_artifact_bytes(&path, RESULT_FILE_SPEC))
            .map_err(|diagnostic| crate::error::compatibility_error(py, &[diagnostic]))?;
        let native = eqiora_numerics::CommonResult::from_bytes(&bytes, plan.native())
            .map_err(|diagnostic| crate::error::compatibility_error(py, &[diagnostic]))?;
        let identity = RunIdentity::from_common_result(&native).ok_or_else(|| {
            PyRuntimeError::new_err("Result artifact has no valid execution occurrence")
        })?;
        materialize_common_result(py, plan, identity, native, None)
    }

    /// Independently sampled common-ODE series in canonical Field order.
    #[getter]
    fn fields(&self, py: Python<'_>) -> Vec<Py<PySeries>> {
        match &self.payload {
            ResultPayload::Ode(payload) => payload
                .fields
                .iter()
                .map(|field| field.clone_ref(py))
                .collect(),
            ResultPayload::Eigen | ResultPayload::Fields(_) | ResultPayload::Trajectory(_) => {
                Vec::new()
            }
        }
    }

    /// Select one no-Mesh scalar series by exact Model-bound Field identity.
    #[pyo3(signature = (field, /, *, derivative_order=0, component=0, imaginary=false))]
    fn series(
        &self,
        py: Python<'_>,
        field: &PyModelFieldRef,
        derivative_order: u32,
        component: usize,
        imaginary: bool,
    ) -> PyResult<Py<PySeries>> {
        if field.exact_model_digest() != self.identity.model_digest() {
            return Err(PyValueError::new_err(
                "FieldRef belongs to a different exact Model artifact",
            ));
        }
        let ResultPayload::Ode(payload) = &self.payload else {
            return Err(PyKeyError::new_err(field.exact_id().to_owned()));
        };
        let index = payload
            .lookup
            .get(&(
                field.exact_id().to_owned(),
                derivative_order,
                component,
                imaginary,
            ))
            .copied()
            .ok_or_else(|| PyKeyError::new_err(field.exact_id().to_owned()))?;
        Ok(payload.fields[index].clone_ref(py))
    }

    /// Exact durable spatial trajectory when this Result owns one.
    #[getter]
    fn trajectory(&self, py: Python<'_>) -> PyResult<Py<PyTrajectory>> {
        match &self.payload {
            ResultPayload::Trajectory(payload) => Ok(payload.trajectory.clone_ref(py)),
            ResultPayload::Eigen | ResultPayload::Fields(_) | ResultPayload::Ode(_) => Err(
                capability_error(py, "this Result occurrence has no spatial Trajectory"),
            ),
        }
    }

    /// Select the exact accepted Mesh paired with one static Field.
    #[pyo3(signature = (field, /))]
    fn mesh(&self, py: Python<'_>, field: &PyModelFieldRef) -> PyResult<Py<PyMesh>> {
        if let ResultPayload::Fields(_) = &self.payload {
            return self
                .common_output(py, field)
                .map(|output| output.borrow(py).mesh_handle(py));
        }
        Err(PyKeyError::new_err(field.exact_id().to_owned()))
    }

    #[getter]
    fn solve(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.payload {
            ResultPayload::Fields(payload) => Ok(payload.solve.clone_ref(py)),
            _ => Err(capability_error(
                py,
                "this Result occurrence has no algebraic solve summary",
            )),
        }
    }

    /// Select one common multi-field output by exact FieldRef identity.
    #[pyo3(signature = (field, /))]
    fn output(&self, py: Python<'_>, field: &PyModelFieldRef) -> PyResult<Py<PyFieldOutput>> {
        self.common_output(py, field)
    }

    /// Signed force pair on one exact Geometry boundary of this Result.
    #[pyo3(signature = (selection, /))]
    fn boundary_force(
        &self,
        py: Python<'_>,
        selection: Py<PyGeometrySelection>,
    ) -> PyResult<Py<PyBoundaryForce>> {
        let payload = self.steady_stokes_payload(py, "boundary-force")?;
        let selected = selection.borrow(py);
        self.validate_observable_selection(py, payload, &selected)?;
        let name = selected.canonical_name().to_owned();
        let force = self
            .native
            .steady_stokes_boundary_reaction(&name)
            .ok_or_else(|| {
                PyKeyError::new_err(format!(
                    "this Result has no boundary-force observable for {name:?}"
                ))
            })?;
        let geometry_digest = selected.bound_source_digest().to_owned();
        drop(selected);
        let mesh = payload.outputs[0].borrow(py).mesh_handle(py);
        let mesh_digest = mesh.borrow(py).exact_mesh_digest().to_owned();
        Py::new(
            py,
            PyBoundaryForce::new(
                selection,
                name,
                geometry_digest,
                self.identity.plan_key(),
                "result",
                &mesh_digest,
                force,
            ),
        )
    }

    /// Signed volume flux on one exact Geometry boundary of this Result.
    #[pyo3(signature = (selection, /))]
    fn boundary_flux(
        &self,
        py: Python<'_>,
        selection: Py<PyGeometrySelection>,
    ) -> PyResult<Py<PyBoundaryFlux>> {
        let payload = self.steady_stokes_payload(py, "boundary-flux")?;
        let selected = selection.borrow(py);
        self.validate_observable_selection(py, payload, &selected)?;
        let name = selected.canonical_name().to_owned();
        let value = self
            .native
            .steady_stokes_boundary_flux(&name)
            .ok_or_else(|| {
                PyKeyError::new_err(format!(
                    "this Result has no boundary-flux observable for {name:?}"
                ))
            })?;
        let geometry_digest = selected.bound_source_digest().to_owned();
        drop(selected);
        let mesh = payload.outputs[0].borrow(py).mesh_handle(py);
        let mesh_digest = mesh.borrow(py).exact_mesh_digest().to_owned();
        Py::new(
            py,
            PyBoundaryFlux::new(
                selection,
                name,
                geometry_digest,
                self.identity.plan_key(),
                &mesh_digest,
                value,
            ),
        )
    }

    fn __repr__(&self) -> String {
        let fields = match &self.payload {
            ResultPayload::Fields(payload) => payload.outputs.len(),
            ResultPayload::Eigen | ResultPayload::Trajectory(_) => 0,
            ResultPayload::Ode(payload) => payload.fields.len(),
        };
        format!(
            "Result(fields={}, model_digest={:?}, plan_key={:?})",
            fields,
            self.model_digest(),
            self.plan_key(),
        )
    }
}

impl PyRunResult {
    pub(crate) fn plan_key_value(&self) -> &str {
        self.identity.plan_key()
    }
    pub(crate) fn common_ode_state_at(
        &self,
        state_space_identity: &str,
        time_s: f64,
    ) -> Option<eqiora_numerics::CommonOdeState> {
        let ResultPayload::Ode(payload) = &self.payload else {
            return None;
        };
        payload
            .states
            .iter()
            .find(|state| {
                state.state_space_identity() == state_space_identity
                    && state.time_s().to_bits() == time_s.to_bits()
            })
            .cloned()
    }

    fn common_output(
        &self,
        py: Python<'_>,
        field: &PyModelFieldRef,
    ) -> PyResult<Py<PyFieldOutput>> {
        if field.exact_model_digest() != self.identity.model_digest() {
            return Err(PyValueError::new_err(
                "FieldRef belongs to a different exact Model artifact",
            ));
        }
        let ResultPayload::Fields(payload) = &self.payload else {
            return Err(PyKeyError::new_err(field.exact_id().to_owned()));
        };
        let index = payload
            .lookup
            .get(field.exact_id())
            .copied()
            .ok_or_else(|| PyKeyError::new_err(field.exact_id().to_owned()))?;
        Ok(payload.outputs[index].clone_ref(py))
    }

    fn steady_stokes_payload(
        &self,
        py: Python<'_>,
        observable: &str,
    ) -> PyResult<&CommonFieldResultPayload> {
        let ResultPayload::Fields(payload) = &self.payload else {
            return Err(capability_error(
                py,
                &format!("this Result occurrence has no {observable} observable"),
            ));
        };
        if self.native.family_name() != "steady-stokes" {
            return Err(capability_error(
                py,
                &format!("this Result occurrence has no {observable} observable"),
            ));
        }
        Ok(payload)
    }

    fn validate_observable_selection(
        &self,
        py: Python<'_>,
        payload: &CommonFieldResultPayload,
        selection: &PyGeometrySelection,
    ) -> PyResult<()> {
        let mesh = payload.outputs[0].borrow(py).mesh_handle(py);
        let source_digest = mesh.borrow(py).source_digest_value().to_owned();
        if selection.bound_source_digest() != source_digest {
            return Err(PyValueError::new_err(
                "GeometrySelection belongs to a foreign or stale Geometry revision",
            ));
        }
        if selection.canonical_dimension() != 1 {
            return Err(PyValueError::new_err(
                "Result boundary observables require a codimension-one GeometrySelection",
            ));
        }
        Ok(())
    }

    pub(crate) fn steady_stokes_evidence(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PySteadyStokesEvidence>> {
        match &self.payload {
            ResultPayload::Fields(payload) => match &payload.evidence {
                Some(StaticScientificEvidence::SteadyStokes(evidence)) => {
                    Ok(evidence.clone_ref(py))
                }
                _ => Err(capability_error(
                    py,
                    "this Result occurrence has no steady-Stokes evidence",
                )),
            },
            ResultPayload::Eigen | ResultPayload::Trajectory(_) | ResultPayload::Ode(_) => Err(
                capability_error(py, "this transient Result has no steady-Stokes evidence"),
            ),
        }
    }

    pub(crate) fn linear_elasticity_evidence(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PyLinearElasticityEvidence>> {
        match &self.payload {
            ResultPayload::Fields(payload) => match &payload.evidence {
                Some(StaticScientificEvidence::LinearElasticity(evidence)) => {
                    Ok(evidence.clone_ref(py))
                }
                _ => Err(capability_error(
                    py,
                    "this Result occurrence has no linear-elasticity evidence",
                )),
            },
            ResultPayload::Eigen | ResultPayload::Trajectory(_) | ResultPayload::Ode(_) => {
                Err(capability_error(
                    py,
                    "this Result occurrence has no linear-elasticity evidence",
                ))
            }
        }
    }

    pub(crate) fn fsi_evidence(&self, py: Python<'_>) -> PyResult<Py<PyFsiEvidence>> {
        match &self.payload {
            ResultPayload::Trajectory(payload) => payload
                .fsi_evidence
                .as_ref()
                .map(|evidence| evidence.clone_ref(py))
                .ok_or_else(|| capability_error(py, "this transient Result has no FSI evidence")),
            ResultPayload::Eigen | ResultPayload::Fields(_) | ResultPayload::Ode(_) => Err(
                capability_error(py, "this Result occurrence has no FSI evidence"),
            ),
        }
    }
}

fn capability_error(py: Python<'_>, message: &str) -> PyErr {
    diagnostic_error(py, &[Diagnostic::error(codes::NOT_IMPLEMENTED, message)])
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    observe::register(module)?;
    module.add_class::<eigen::PyEigenpair>()?;
    time_observe::register(module)?;
    module.add_class::<PySeries>()?;
    module.add_class::<PyFieldOutput>()?;
    module.add_class::<PyRunResult>()?;
    module.add_class::<nonlinear::PyNonlinearSolveSummary>()?;
    module.add_class::<constraints::PyConstraintMeasurement>()?;
    Ok(())
}
