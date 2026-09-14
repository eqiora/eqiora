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
    solve: Py<PyLinearSolveSummary>,
    evidence: Option<StaticScientificEvidence>,
    steady_stokes_observation: Option<([f64; 6], [[f64; 2]; 7])>,
}

struct CommonTrajectoryResultPayload {
    trajectory: Py<PyTrajectory>,
    fsi_evidence: Option<Py<PyFsiEvidence>>,
}

struct CommonOdeResultPayload {
    fields: Vec<Py<PySeries>>,
    lookup: BTreeMap<String, usize>,
    states: Vec<eqiora_numerics::CommonOdeState>,
}

enum ResultPayload {
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
            ResultPayload::Fields(_) | ResultPayload::Trajectory(_) => Vec::new(),
        }
    }

    /// Select one no-Mesh scalar series by exact Model-bound Field identity.
    #[pyo3(signature = (field, /))]
    fn series(&self, py: Python<'_>, field: &PyModelFieldRef) -> PyResult<Py<PySeries>> {
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
            .get(field.exact_id())
            .copied()
            .ok_or_else(|| PyKeyError::new_err(field.exact_id().to_owned()))?;
        Ok(payload.fields[index].clone_ref(py))
    }

    /// Exact durable spatial trajectory when this Result owns one.
    #[getter]
    fn trajectory(&self, py: Python<'_>) -> PyResult<Py<PyTrajectory>> {
        match &self.payload {
            ResultPayload::Trajectory(payload) => Ok(payload.trajectory.clone_ref(py)),
            ResultPayload::Fields(_) | ResultPayload::Ode(_) => Err(capability_error(
                py,
                "this Result occurrence has no spatial Trajectory",
            )),
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
    fn solve(&self, py: Python<'_>) -> PyResult<Py<PyLinearSolveSummary>> {
        match &self.payload {
            ResultPayload::Fields(payload) => Ok(payload.solve.clone_ref(py)),
            _ => Err(capability_error(
                py,
                "this Result occurrence has no linear solve summary",
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
        if name != "cylinder" {
            return Err(PyKeyError::new_err(format!(
                "this Result has no boundary-force observable for {name:?}"
            )));
        }
        let geometry_digest = selected.bound_source_digest().to_owned();
        drop(selected);
        let mesh = payload.outputs[0].borrow(py).mesh_handle(py);
        let mesh_digest = mesh.borrow(py).exact_mesh_digest().to_owned();
        let force = payload
            .steady_stokes_observation
            .as_ref()
            .expect("validated steady-Stokes observation")
            .1[2];
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
        let observation = payload
            .steady_stokes_observation
            .as_ref()
            .expect("validated steady-Stokes observation");
        let value = match name.as_str() {
            "inlet" => observation.0[2],
            "outlet" => observation.0[3],
            _ => {
                return Err(PyKeyError::new_err(format!(
                    "this Result has no boundary-flux observable for {name:?}"
                )));
            }
        };
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
            ResultPayload::Trajectory(_) => 0,
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
        if payload.steady_stokes_observation.is_none() {
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
            ResultPayload::Trajectory(_) | ResultPayload::Ode(_) => Err(capability_error(
                py,
                "this transient Result has no steady-Stokes evidence",
            )),
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
            ResultPayload::Trajectory(_) | ResultPayload::Ode(_) => Err(capability_error(
                py,
                "this Result occurrence has no linear-elasticity evidence",
            )),
        }
    }

    pub(crate) fn fsi_evidence(&self, py: Python<'_>) -> PyResult<Py<PyFsiEvidence>> {
        match &self.payload {
            ResultPayload::Trajectory(payload) => payload
                .fsi_evidence
                .as_ref()
                .map(|evidence| evidence.clone_ref(py))
                .ok_or_else(|| capability_error(py, "this transient Result has no FSI evidence")),
            ResultPayload::Fields(_) | ResultPayload::Ode(_) => Err(capability_error(
                py,
                "this Result occurrence has no FSI evidence",
            )),
        }
    }
}

fn materialize_common_spatial_trajectory(
    py: Python<'_>,
    plan: PyRef<'_, PyPlan>,
    identity: RunIdentity,
    elapsed_seconds: f64,
    native_trajectory: eqiora_numerics::CommonTrajectory,
    native_result: eqiora_numerics::CommonResult,
) -> PyResult<PyRunResult> {
    if !matches!(
        plan.native(),
        ResolvedCommonPlan::TransientFlow(_)
            | ResolvedCommonPlan::Scalar(_)
            | ResolvedCommonPlan::Fsi(_)
    ) {
        return Err(PyRuntimeError::new_err(
            "common transient output crossed a different Plan",
        ));
    }
    let trajectory = Py::new(py, PyTrajectory::from_common(py, &plan, native_trajectory)?)?;
    let fsi_evidence = if matches!(plan.native(), ResolvedCommonPlan::Fsi(_)) {
        Some(Py::new(
            py,
            PyFsiEvidence::from_common(
                py,
                &plan,
                &trajectory.borrow(py),
                identity.plan_key(),
                &native_result,
            )?,
        )?)
    } else {
        None
    };
    Ok(PyRunResult {
        native: native_result,
        identity,
        elapsed_seconds,
        payload: ResultPayload::Trajectory(CommonTrajectoryResultPayload {
            trajectory,
            fsi_evidence,
        }),
        profile: None,
    })
}

fn materialize_common_ode_trajectory(
    py: Python<'_>,
    plan: PyRef<'_, PyPlan>,
    identity: RunIdentity,
    elapsed_seconds: f64,
    trajectory: eqiora_numerics::CommonTrajectory,
    native_result: eqiora_numerics::CommonResult,
) -> PyResult<PyRunResult> {
    let ResolvedCommonPlan::Ode(native) = plan.native() else {
        return Err(PyRuntimeError::new_err(
            "common ODE output crossed a different Plan",
        ));
    };
    let states = trajectory
        .ode_states()
        .expect("ODE materialization requires an ODE Trajectory")
        .to_vec();
    let times = states
        .iter()
        .map(eqiora_numerics::CommonOdeState::time_s)
        .collect::<Vec<_>>();
    let mut fields = Vec::new();
    let mut lookup = BTreeMap::new();
    for (column, (field, dimension)) in native
        .field_ids()
        .zip(native.field_dimensions())
        .enumerate()
    {
        let id = field.to_string();
        let values = states.iter().map(|state| state.values()[column]).collect();
        let series = Py::new(
            py,
            PySeries {
                field: Some(Py::new(
                    py,
                    PyModelFieldRef::from_exact(native.model_digest().to_owned(), id.clone()),
                )?),
                id: id.clone(),
                name: None,
                dimension: *dimension,
                time: PyArrayBuffer::from_owned_result(py, times.clone())?,
                values: PyArrayBuffer::from_owned_result(py, values)?,
            },
        )?;
        lookup.insert(id, fields.len());
        fields.push(series);
    }
    Ok(PyRunResult {
        native: native_result,
        identity,
        elapsed_seconds,
        payload: ResultPayload::Ode(CommonOdeResultPayload {
            fields,
            lookup,
            states,
        }),
        profile: None,
    })
}

fn materialize_common_result_unprofiled(
    py: Python<'_>,
    plan: PyRef<'_, PyPlan>,
    identity: RunIdentity,
    result: eqiora_numerics::CommonResult,
) -> PyResult<PyRunResult> {
    if result.plan().identity() != plan.native().identity() {
        return Err(PyRuntimeError::new_err(
            "common Result crossed a different exact Plan",
        ));
    }
    if let Some(trajectory) = result.trajectory() {
        if identity.plan_key() != trajectory.request_identity() {
            return Err(PyRuntimeError::new_err(
                "common Result crossed a different Run request occurrence",
            ));
        }
        let elapsed_seconds = result.elapsed_seconds();
        let trajectory = trajectory.clone();
        if trajectory.ode_states().is_some() {
            return materialize_common_ode_trajectory(
                py,
                plan,
                identity,
                elapsed_seconds,
                trajectory,
                result,
            );
        }
        return materialize_common_spatial_trajectory(
            py,
            plan,
            identity,
            elapsed_seconds,
            trajectory,
            result,
        );
    }
    if identity.plan_key() != result.plan().identity() {
        return Err(PyRuntimeError::new_err(
            "static common Result crossed a different Run Plan occurrence",
        ));
    }
    let mesh = (result.field_count() > 0).then(|| plan.mesh_handle(py));
    let mut outputs = Vec::with_capacity(result.field_count());
    let mut lookup = BTreeMap::new();
    for field_index in 0..result.field_count() {
        let (field_id, dimension, value_shape, native_space) = result
            .field(field_index)
            .ok_or_else(|| PyRuntimeError::new_err("common Result omitted Field metadata"))?;
        let field_id = field_id.to_owned();
        let value_shape = value_shape.to_vec();
        let space = match native_space {
            "continuous-lagrange-p1" => "continuous-lagrange-p1",
            "cell-constant" => "cell-constant",
            "simplex-p1-bubble" => "simplex-p1-bubble",
            _ => {
                return Err(PyRuntimeError::new_err(
                    "common Result Field declared an unknown exact space",
                ));
            }
        };
        let field = Py::new(
            py,
            PyModelFieldRef::from_exact(identity.model_digest().to_owned(), field_id.clone()),
        )?;
        let value_width = value_shape.iter().product::<usize>().max(1);
        let blocks = (0..result.field_block_count(field_index))
            .map(|block_index| {
                let (association, values, logical_shape) = result
                    .field_block(field_index, block_index)
                    .ok_or_else(|| PyRuntimeError::new_err("common Result omitted Field block"))?;
                if !values.len().is_multiple_of(value_width) {
                    return Err(PyRuntimeError::new_err(
                        "common Result Field block contradicts its value shape",
                    ));
                }
                Ok(FieldOutputBlock::new(
                    association,
                    PyArrayBuffer::from_owned_result(py, values.to_vec())?,
                    values.len() / value_width,
                    logical_shape.to_vec(),
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let output = Py::new(
            py,
            PyFieldOutput::new(
                field,
                mesh.as_ref()
                    .expect("spatial field owns mesh")
                    .clone_ref(py),
                dimension,
                value_shape,
                space,
                blocks,
            ),
        )?;
        lookup.insert(field_id, outputs.len());
        outputs.push(output);
    }
    let solve = PyLinearSolveSummary::from_common_result(&result, None)
        .ok_or_else(|| PyRuntimeError::new_err("static common Result omitted solve evidence"))?;
    let solve = Py::new(py, solve)?;
    let (evidence, steady_stokes_observation) = match result.family_name() {
        "algebraic" | "scalar" => (None, None),
        "elasticity" => (
            Some(StaticScientificEvidence::LinearElasticity(Py::new(
                py,
                PyLinearElasticityEvidence::from_result(py, identity.plan_key(), &result)?,
            )?)),
            None,
        ),
        "steady-stokes" => {
            let observation = result.steady_stokes_observation().ok_or_else(|| {
                PyRuntimeError::new_err("steady-Stokes Result omitted its observation")
            })?;
            (
                Some(StaticScientificEvidence::SteadyStokes(Py::new(
                    py,
                    PySteadyStokesEvidence::from_result(py, identity.plan_key(), &result)?,
                )?)),
                Some(observation),
            )
        }
        _ => {
            return Err(PyRuntimeError::new_err(
                "dynamic common Result reached static materialization",
            ));
        }
    };
    let elapsed_seconds = result.elapsed_seconds();
    Ok(PyRunResult {
        native: result,
        identity,
        elapsed_seconds,
        payload: ResultPayload::Fields(Box::new(CommonFieldResultPayload {
            outputs,
            lookup,
            solve,
            evidence,
            steady_stokes_observation,
        })),
        profile: None,
    })
}

pub(crate) fn materialize_common_result(
    py: Python<'_>,
    plan: PyRef<'_, PyPlan>,
    identity: RunIdentity,
    result: eqiora_numerics::CommonResult,
    profile: Option<crate::profile::ProfileData>,
) -> PyResult<PyRunResult> {
    let mut result = materialize_common_result_unprofiled(py, plan, identity, result)?;
    result.profile = profile
        .map(|profile| Py::new(py, crate::profile::PyProfile::new(profile)))
        .transpose()?;
    Ok(result)
}

fn capability_error(py: Python<'_>, message: &str) -> PyErr {
    diagnostic_error(py, &[Diagnostic::error(codes::NOT_IMPLEMENTED, message)])
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    observe::register(module)?;
    time_observe::register(module)?;
    module.add_class::<PySeries>()?;
    module.add_class::<PyFieldOutput>()?;
    module.add_class::<PyRunResult>()?;
    module.add_class::<constraints::PyConstraintMeasurement>()?;
    Ok(())
}
