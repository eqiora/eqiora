//! Immutable Python projection of one accepted spatial trajectory.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use eqiora::DimExponents;
use eqiora::artifact::ArtifactDigest;
use eqiora_numerics::{
    CommonInitialField, CommonInitialValues, CommonState, CommonTransientFlowPlan,
};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyModule, PySequence, PyTuple};
use sha2::{Digest, Sha256};

use crate::geometry::PyGeometrySelection;
use crate::matrix::{ReadOnlyMatrix, ReadOnlyVector};
use crate::meshing::PyMesh;
use crate::model::{PyModel, PyModelFieldRef};
use crate::model_io::{
    ArtifactFileSpec, read_artifact_bytes, unicode_artifact_path, write_artifact_bytes,
};
mod field;
pub(crate) use field::PyDerivedFieldSnapshot;
use field::{PyFieldSnapshot, PyInitialField};
mod observation;
mod projection;
mod scalar;
use observation::PyFieldSample;
pub(crate) use observation::{PyBoundaryFlux, PyBoundaryForce};

const TRAJECTORY_FILE_SPEC: ArtifactFileSpec = ArtifactFileSpec {
    artifact_name: "spatial Trajectory",
    extension: "eqtrajectory",
    staging_name: "trajectory",
    // This is the pre-read counterpart of the canonical Trajectory decoder's bound.
    max_bytes: 512 * 1024 * 1024,
};

/// Immutable installed-Python projection of one state in a common execution.
#[pyclass(
    name = "State",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
pub(crate) struct PyState {
    digest: String,
    model_digest: String,
    step: u64,
    time_s: f64,
    fields: Vec<Py<PyFieldSnapshot>>,
    field_lookup: BTreeMap<String, usize>,
    model: Option<Py<PyModel>>,
    mesh: Option<Py<PyMesh>>,
    native: Option<CommonState>,
    transient_plan: Option<CommonTransientFlowPlan>,
    ode_native: Option<eqiora_numerics::CommonOdeState>,
    pub(crate) algebraic_native: Option<eqiora_numerics::CommonAlgebraicState>,
    plan_identity: Option<String>,
    source_request_identity: Option<String>,
    source_trajectory_identity: Option<String>,
    source_kind: Option<&'static str>,
}

impl PartialEq for PyState {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}

impl Eq for PyState {}

impl Hash for PyState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.digest.hash(state);
    }
}

impl PyState {
    pub(crate) fn digest_value(&self) -> &str {
        &self.digest
    }

    pub(crate) fn model_digest_value(&self) -> &str {
        &self.model_digest
    }

    pub(crate) fn from_common(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        native: CommonState,
        step: u64,
        source_request_identity: Option<&str>,
        source_trajectory_identity: Option<&str>,
    ) -> PyResult<Self> {
        if plan.scalar_native().is_some() {
            return Self::from_scalar(
                py,
                plan,
                native,
                step,
                source_request_identity,
                source_trajectory_identity,
            );
        }
        let native_plan = plan
            .transient_native()
            .expect("common State requires a transient Plan");
        let mesh = plan.mesh_handle(py);
        let mesh_digest = mesh.borrow(py).exact_mesh_digest().to_owned();
        let velocity =
            PyFieldSnapshot::from_common_velocity(py, native_plan, &native, &mesh_digest)?;
        let pressure =
            PyFieldSnapshot::from_common_pressure(py, native_plan, &native, &mesh_digest)?;
        let mut field_lookup = BTreeMap::new();
        field_lookup.insert(native_plan.velocity_field_id().to_owned(), 0);
        field_lookup.insert(native_plan.pressure_field_id().to_owned(), 1);
        Ok(Self {
            digest: native.identity().to_owned(),
            model_digest: native_plan.model_digest().to_owned(),
            step,
            time_s: native.time_s(),
            fields: vec![Py::new(py, velocity)?, Py::new(py, pressure)?],
            field_lookup,
            model: Some(plan.model_handle(py)),
            mesh: Some(mesh),
            native: Some(native),
            transient_plan: Some(native_plan.clone()),
            algebraic_native: None,
            ode_native: None,
            plan_identity: Some(native_plan.identity().to_owned()),
            source_request_identity: source_request_identity.map(str::to_owned),
            source_trajectory_identity: source_trajectory_identity.map(str::to_owned),
            source_kind: Some(if source_request_identity.is_some() {
                "result"
            } else {
                "zero"
            }),
        })
    }

    pub(crate) fn common_native(&self) -> Option<&CommonState> {
        self.native.as_ref()
    }

    pub(crate) fn from_common_cancellation(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        native: CommonState,
        accepted_steps: u64,
        request_identity: &str,
    ) -> PyResult<Self> {
        let mut state = match plan.native() {
            eqiora_numerics::ResolvedCommonPlan::TransientFlow(_)
            | eqiora_numerics::ResolvedCommonPlan::Scalar(_) => Self::from_common(
                py,
                plan,
                native,
                accepted_steps,
                Some(request_identity),
                None,
            )?,
            eqiora_numerics::ResolvedCommonPlan::Fsi(_) => {
                Self::from_common_fsi(py, plan, native, accepted_steps, Some(request_identity))?
            }
            _ => {
                return Err(PyValueError::new_err(
                    "cancelled spatial State requires a transient-flow or fixed-reference FSI Plan",
                ));
            }
        };
        state.source_kind = Some("cancellation");
        Ok(state)
    }

    pub(crate) fn common_ode_native(&self) -> Option<&eqiora_numerics::CommonOdeState> {
        self.ode_native.as_ref()
    }

    fn from_common_ode(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        native: eqiora_numerics::CommonOdeState,
        source_request_identity: Option<&str>,
    ) -> Self {
        Self {
            digest: native.identity().to_owned(),
            model_digest: native.model_digest().to_owned(),
            step: 0,
            time_s: native.time_s(),
            fields: Vec::new(),
            field_lookup: native
                .field_ids()
                .iter()
                .enumerate()
                .map(|(index, field)| (field.to_string(), index))
                .collect(),
            model: Some(plan.model_handle(py)),
            mesh: None,
            native: None,
            transient_plan: None,
            algebraic_native: None,
            ode_native: Some(native.clone()),
            plan_identity: plan.ode_native().map(|plan| plan.identity().to_owned()),
            source_request_identity: source_request_identity.map(str::to_owned),
            source_trajectory_identity: None,
            source_kind: Some(native.source_kind()),
        }
    }

    fn from_common_fsi(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        native: CommonState,
        step: u64,
        source_request_identity: Option<&str>,
    ) -> PyResult<Self> {
        let native_plan = plan.fsi_native().expect("FSI State requires FSI Plan");
        let mesh = plan.mesh_handle(py);
        let mesh_digest = mesh.borrow(py).exact_mesh_digest().to_owned();
        let fields = PyFieldSnapshot::from_common_fsi(py, native_plan, &native, &mesh_digest)?
            .into_iter()
            .map(|field| Py::new(py, field))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            digest: native.identity().to_owned(),
            model_digest: native_plan.model_digest().to_owned(),
            step,
            time_s: native.time_s(),
            fields,
            field_lookup: native_plan
                .field_ids()
                .iter()
                .enumerate()
                .map(|(index, field)| (field.clone(), index))
                .collect(),
            model: Some(plan.model_handle(py)),
            mesh: Some(mesh),
            native: Some(native),
            transient_plan: None,
            algebraic_native: None,
            ode_native: None,
            plan_identity: Some(native_plan.identity().to_owned()),
            source_request_identity: source_request_identity.map(str::to_owned),
            source_trajectory_identity: None,
            source_kind: Some(if source_request_identity.is_some() {
                "result"
            } else {
                "initial"
            }),
        })
    }

    pub(crate) const fn time_s_value(&self) -> f64 {
        self.time_s
    }

    fn from_artifact(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        data: &[u8],
    ) -> PyResult<Self> {
        let mut state = match plan.native() {
            eqiora_numerics::ResolvedCommonPlan::Algebraic(native_plan) => {
                eqiora_numerics::CommonAlgebraicState::from_bytes(data, native_plan)
                    .map_err(|d| crate::error::validation_error(py, &[d]))?;
                Self::initial(py, plan, None, None)?
            }
            eqiora_numerics::ResolvedCommonPlan::Ode(native_plan) => {
                let native = eqiora_numerics::CommonOdeState::from_bytes(data, native_plan)
                    .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
                Self::from_common_ode(py, plan, native, None)
            }
            eqiora_numerics::ResolvedCommonPlan::TransientFlow(_)
            | eqiora_numerics::ResolvedCommonPlan::Scalar(_) => {
                let native = CommonState::from_bytes(data, plan.native())
                    .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
                Self::from_common(py, plan, native, 0, None, None)?
            }
            eqiora_numerics::ResolvedCommonPlan::Fsi(_) => {
                let native = CommonState::from_bytes(data, plan.native())
                    .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
                Self::from_common_fsi(py, plan, native, 0, None)?
            }
            eqiora_numerics::ResolvedCommonPlan::Elasticity(_)
            | eqiora_numerics::ResolvedCommonPlan::SteadyStokes(_) => {
                return Err(PyValueError::new_err(
                    "State.from_bytes requires an ODE, transient-flow, or fixed-reference FSI Plan",
                ));
            }
        };
        state.source_kind = Some("artifact");
        state.source_request_identity = None;
        state.source_trajectory_identity = None;
        Ok(state)
    }
}

#[pymethods]
impl PyState {
    /// Canonical State bytes, reauthenticated against the owning Plan on load.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if let Some(state) = &self.algebraic_native {
            return state
                .to_bytes()
                .map(|bytes| PyBytes::new(py, &bytes))
                .map_err(|d| crate::error::validation_error(py, &[d]));
        }
        let bytes = match (&self.ode_native, &self.native) {
            (Some(state), None) => state.to_bytes(),
            (None, Some(state)) => state.to_bytes(),
            _ => {
                return Err(PyRuntimeError::new_err(
                    "State lost its exact native representation",
                ));
            }
        }
        .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Decode one exact State against its state-space-owning Plan.
    #[staticmethod]
    fn from_bytes(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        data: &[u8],
    ) -> PyResult<Self> {
        Self::from_artifact(py, plan, data)
    }

    #[staticmethod]
    #[pyo3(signature = (plan, /, *, fields=None, time_s=None))]
    fn initial(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        fields: Option<&Bound<'_, PyTuple>>,
        time_s: Option<f64>,
    ) -> PyResult<Self> {
        if let Some(native_plan) = plan.native().as_algebraic() {
            if fields.is_some() || time_s.is_some() {
                return Err(PyValueError::new_err(
                    "finite State.initial accepts Plan alone",
                ));
            }
            let state = native_plan
                .initial_state()
                .map_err(|d| crate::error::validation_error(py, &[d]))?;
            return Ok(Self {
                digest: state.identity().to_owned(),
                model_digest: native_plan.model_digest().to_owned(),
                step: 0,
                time_s: 0.0,
                fields: Vec::new(),
                field_lookup: BTreeMap::new(),
                model: Some(plan.model_handle(py)),
                mesh: None,
                native: None,
                transient_plan: None,
                ode_native: None,
                algebraic_native: Some(state),
                plan_identity: Some(native_plan.identity().to_owned()),
                source_request_identity: None,
                source_trajectory_identity: None,
                source_kind: Some("initial"),
            });
        }
        if let Some(scalar) = plan.scalar_native() {
            if fields.is_some() || time_s.is_some() {
                return Err(PyValueError::new_err(
                    "scalar State.initial consumes its exact source initial condition",
                ));
            }
            let native = scalar
                .initial_state()
                .map_err(|d| crate::error::validation_error(py, &[d]))?;
            return Self::from_common(py, plan, native, 0, None, None);
        }
        if let Some(native_plan) = plan.ode_native() {
            if fields.is_some() || time_s.is_some() {
                return Err(PyValueError::new_err(
                    "ODE State.initial(plan) accepts no FSI field/time arguments",
                ));
            }
            let state = native_plan
                .initial_state()
                .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
            return Ok(Self::from_common_ode(py, plan, state, None));
        }
        if let Some(native_plan) = plan.transient_native() {
            let fields = fields.ok_or_else(|| {
                PyValueError::new_err(
                    "transient State.initial requires fields=(InitialField(...), ...)",
                )
            })?;
            let time_s = time_s.ok_or_else(|| {
                PyValueError::new_err("transient State.initial requires explicit time_s=")
            })?;
            let fields = fields
                .iter()
                .map(|value| {
                    value
                        .extract::<PyRef<'_, PyInitialField>>()
                        .map(|field| field.native.clone())
                        .map_err(|_| {
                            PyValueError::new_err("fields must contain only InitialField values")
                        })
                })
                .collect::<PyResult<Vec<_>>>()?;
            let state = native_plan
                .initial_state(time_s, fields)
                .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
            return Self::from_common(py, plan, state, 0, None, None);
        }
        let native_plan = plan.fsi_native().ok_or_else(|| {
            PyValueError::new_err(
                "State.initial requires an ODE, transient-flow, or fixed-reference FSI Plan",
            )
        })?;
        let fields = fields.ok_or_else(|| {
            PyValueError::new_err("FSI State.initial requires fields=(InitialField(...), ...)")
        })?;
        let time_s = time_s
            .ok_or_else(|| PyValueError::new_err("FSI State.initial requires explicit time_s="))?;
        let fields = fields
            .iter()
            .map(|value| {
                value
                    .extract::<PyRef<'_, PyInitialField>>()
                    .map(|field| field.native.clone())
                    .map_err(|_| {
                        PyValueError::new_err("fields must contain only InitialField values")
                    })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let state = native_plan
            .initial_state(time_s, fields)
            .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
        Self::from_common_fsi(py, plan, state, 0, None)
    }

    #[staticmethod]
    #[pyo3(signature = (plan, /, *, time_s=0.0))]
    fn zero(py: Python<'_>, plan: &crate::common_plan::PyPlan, time_s: f64) -> PyResult<Self> {
        let native_plan = plan.transient_native().ok_or_else(|| {
            PyValueError::new_err("State.zero requires an admitted transient Plan")
        })?;
        let state = native_plan
            .zero_state(time_s)
            .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
        Self::from_common(py, plan, state, 0, None, None)
    }

    #[staticmethod]
    #[pyo3(signature = (plan, result, /, *, time_s))]
    fn from_result(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        result: &crate::result::PyRunResult,
        time_s: f64,
    ) -> PyResult<Py<PyState>> {
        if !time_s.is_finite() || time_s < 0.0 {
            return Err(PyValueError::new_err(
                "State.from_result time_s must be finite and non-negative",
            ));
        }
        if let Some(native) = plan.ode_native() {
            let state = result.common_ode_state_at(native.state_space_identity(), time_s)
                .ok_or_else(|| PyValueError::new_err(
                    "Result contains no State at time_s compatible with the exact ODE Plan state space"
                ))?;
            return Py::new(
                py,
                Self::from_common_ode(py, plan, state, Some(result.plan_key_value())),
            );
        }
        if let Some(scalar) = plan.scalar_native() {
            return result
                .common_state_at(py, scalar.identity(), time_s)
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "Result contains no scalar State for this exact Plan and time",
                    )
                });
        }
        let native = plan
            .transient_native()
            .ok_or_else(|| PyValueError::new_err("State.from_result requires a transient Plan"))?;
        result
            .common_state_at(py, &native.state_space_identity(), time_s)
            .ok_or_else(|| {
                PyValueError::new_err(
                    "Result contains no State at time_s compatible with the exact Plan state space",
                )
            })
    }

    #[getter]
    fn digest(&self) -> &str {
        &self.digest
    }

    #[getter]
    const fn step(&self) -> u64 {
        self.step
    }

    #[getter]
    const fn time_s(&self) -> f64 {
        self.time_s
    }

    #[getter]
    fn state_space_identity(&self) -> &str {
        self.ode_native.as_ref().map_or_else(
            || {
                self.native
                    .as_ref()
                    .map_or(self.digest.as_str(), CommonState::state_space_identity)
            },
            eqiora_numerics::CommonOdeState::state_space_identity,
        )
    }

    #[getter]
    fn mesh(&self, py: Python<'_>) -> Option<Py<PyMesh>> {
        self.mesh.as_ref().map(|mesh| mesh.clone_ref(py))
    }

    #[getter]
    fn model(&self, py: Python<'_>) -> Option<Py<PyModel>> {
        self.model.as_ref().map(|model| model.clone_ref(py))
    }

    #[getter]
    fn source_plan_identity(&self) -> Option<&str> {
        self.plan_identity.as_deref()
    }

    #[getter]
    fn source_request_identity(&self) -> Option<&str> {
        self.source_request_identity.as_deref()
    }

    #[getter]
    fn source_trajectory_identity(&self) -> Option<&str> {
        self.source_trajectory_identity.as_deref()
    }

    #[getter]
    const fn source_kind(&self) -> Option<&'static str> {
        self.source_kind
    }

    #[getter]
    fn fields(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.fields.iter().map(|field| field.clone_ref(py)))?.unbind())
    }

    #[getter]
    fn field_refs(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let Some(native) = &self.ode_native else {
            return Ok(PyTuple::empty(py).unbind());
        };
        PyTuple::new(
            py,
            native.field_ids().iter().map(|field| {
                PyModelFieldRef::from_exact(self.model_digest.clone(), field.to_string())
            }),
        )
        .map(|value| value.unbind())
    }

    #[pyo3(signature = (field, /))]
    fn value(&self, field: &PyModelFieldRef) -> PyResult<f64> {
        if field.exact_model_digest() != self.model_digest {
            return Err(PyValueError::new_err(
                "FieldRef belongs to a different exact Model artifact",
            ));
        }
        let native = self.ode_native.as_ref().ok_or_else(|| {
            PyValueError::new_err("State.value is available only for no-Mesh scalar ODE States")
        })?;
        let index = native
            .field_ids()
            .iter()
            .position(|id| id.to_string() == field.exact_id())
            .ok_or_else(|| PyKeyError::new_err(field.exact_id().to_owned()))?;
        Ok(native.values()[index])
    }

    /// Select one complete Field observation by exact Model-bound identity.
    #[pyo3(signature = (field, /))]
    fn field(&self, py: Python<'_>, field: &PyModelFieldRef) -> PyResult<Py<PyFieldSnapshot>> {
        if field.exact_model_digest() != self.model_digest {
            return Err(PyValueError::new_err(
                "FieldRef belongs to a different exact Model artifact",
            ));
        }
        let index = self
            .field_lookup
            .get(field.exact_id())
            .copied()
            .ok_or_else(|| PyKeyError::new_err(field.exact_id().to_owned()))?;
        Ok(self.fields[index].clone_ref(py))
    }

    /// Derive the cell-average two-dimensional curl of one exact velocity Field.
    #[pyo3(signature = (field, /))]
    fn curl(
        &self,
        py: Python<'_>,
        field: &PyModelFieldRef,
    ) -> PyResult<Py<PyDerivedFieldSnapshot>> {
        if field.exact_model_digest() != self.model_digest {
            return Err(PyValueError::new_err(
                "FieldRef belongs to a different exact Model artifact",
            ));
        }
        let plan = self.transient_plan.as_ref().ok_or_else(|| {
            PyValueError::new_err("State.curl requires a two-dimensional transient-flow State")
        })?;
        if field.exact_id() != plan.velocity_field_id() {
            return Err(PyValueError::new_err(
                "State.curl currently requires the Plan velocity FieldRef",
            ));
        }
        let state = self.native.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("transient State lost its native accepted coefficients")
        })?;
        let values = plan
            .cell_average_velocity_curl_2d(state)
            .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
        Py::new(
            py,
            PyDerivedFieldSnapshot::from_cell_average_curl(
                py,
                &self.model_digest,
                state.identity(),
                field,
                plan.mesh_digest(),
                &plan.domain_id(),
                &values,
            )?,
        )
    }

    /// Sample the exact continuous pressure Field at one physical point.
    #[pyo3(signature = (field, /, *, at))]
    fn sample(
        &self,
        py: Python<'_>,
        field: &PyModelFieldRef,
        at: &Bound<'_, PyTuple>,
    ) -> PyResult<Py<PyFieldSample>> {
        if field.exact_model_digest() != self.model_digest {
            return Err(PyValueError::new_err(
                "FieldRef belongs to a different exact Model artifact",
            ));
        }
        let plan = self.transient_plan.as_ref().ok_or_else(|| {
            PyValueError::new_err("State.sample requires a two-dimensional transient-flow State")
        })?;
        if field.exact_id() != plan.pressure_field_id() {
            return Err(PyValueError::new_err(
                "State.sample currently requires the Plan pressure FieldRef",
            ));
        }
        if at.len() != 2 {
            return Err(PyValueError::new_err(
                "State.sample at= must contain exactly two physical coordinates",
            ));
        }
        let mut point = [0.0; 2];
        for (axis, coordinate) in point.iter_mut().enumerate() {
            let value = at.get_item(axis)?;
            if value.is_instance_of::<PyBool>() {
                return Err(PyValueError::new_err(
                    "State.sample coordinates must be finite real numbers, not booleans",
                ));
            }
            *coordinate = value.extract::<f64>().map_err(|_| {
                PyValueError::new_err("State.sample coordinates must be finite real numbers")
            })?;
        }
        let state = self.native.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("transient State lost its native accepted coefficients")
        })?;
        let value = plan
            .sample_pressure_2d(state, point)
            .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
        let field_id = field.exact_id().to_owned();
        let field = Py::new(
            py,
            PyModelFieldRef::from_exact(self.model_digest.clone(), field_id.clone()),
        )?;
        Py::new(
            py,
            PyFieldSample::pressure(
                field,
                field_id,
                state.identity(),
                plan.mesh_digest(),
                plan.domain_id(),
                point,
                value,
            ),
        )
    }

    /// Observe the signed force pair on one authenticated constrained boundary.
    #[pyo3(signature = (selection, /))]
    fn boundary_force(
        &self,
        py: Python<'_>,
        selection: Py<PyGeometrySelection>,
    ) -> PyResult<Py<PyBoundaryForce>> {
        let plan = self.transient_plan.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "State.boundary_force requires a two-dimensional transient-flow State",
            )
        })?;
        let selected = selection.borrow(py);
        if selected.bound_source_digest() != plan.geometry_digest() {
            return Err(PyValueError::new_err(
                "GeometrySelection belongs to a foreign or stale Geometry revision",
            ));
        }
        if selected.canonical_dimension() != 1 {
            return Err(PyValueError::new_err(
                "State.boundary_force requires a codimension-one GeometrySelection",
            ));
        }
        let selection_name = selected.canonical_name().to_owned();
        let geometry_digest = selected.bound_source_digest().to_owned();
        drop(selected);
        let state = self.native.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("transient State lost its native accepted coefficients")
        })?;
        let force = state
            .named_boundary_force_on_domain(&selection_name)
            .ok_or_else(|| {
                PyKeyError::new_err(format!(
                    "accepted State has no boundary-force observation for {selection_name:?}"
                ))
            })?;
        Py::new(
            py,
            PyBoundaryForce::new(
                selection,
                selection_name,
                geometry_digest,
                state.identity(),
                "state",
                plan.mesh_digest(),
                force,
            ),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "State(step={}, time_s={}, digest={:?})",
            self.step, self.time_s, self.digest,
        )
    }
}

/// Immutable installed-Python projection of one accepted trajectory.
///
/// Common transient execution retains its exact owning lineage without
/// fabricating a separate application-shaped result artifact.
#[pyclass(
    name = "Trajectory",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
pub(crate) struct PyTrajectory {
    model_digest: String,
    geometry_digest: String,
    correspondence_digest: String,
    mesh_digest: String,
    realization_digest: Option<String>,
    plan_identity: Option<String>,
    run_digest: Option<String>,
    request_identity: Option<String>,
    trajectory_digest: String,
    coordinates: ReadOnlyMatrix<f64>,
    cells: ReadOnlyMatrix<u32>,
    states: Vec<Py<PyState>>,
    state_lookup: BTreeMap<u64, usize>,
    common_mesh: Option<Py<PyMesh>>,
    native: eqiora_numerics::CommonTrajectory,
}

impl PartialEq for PyTrajectory {
    fn eq(&self, other: &Self) -> bool {
        self.trajectory_digest == other.trajectory_digest
    }
}

impl Eq for PyTrajectory {}

impl Hash for PyTrajectory {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.trajectory_digest.hash(state);
    }
}

#[pymethods]
impl PyTrajectory {
    /// Canonical immutable Run request and requested output States.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.native
            .to_bytes()
            .map(|bytes| PyBytes::new(py, &bytes))
            .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))
    }

    /// Decode one producer-independent spatial Trajectory against its Plan.
    #[staticmethod]
    fn from_bytes(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        data: &[u8],
    ) -> PyResult<Self> {
        let native = eqiora_numerics::CommonTrajectory::from_bytes(data, plan.native())
            .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
        if native.spatial_states().is_none() {
            return Err(PyValueError::new_err(
                "Trajectory.from_bytes requires a transient-flow or fixed-reference FSI artifact",
            ));
        }
        Self::from_common(py, plan, native)
    }

    /// Atomically write this exact spatial Trajectory to an `.eqtrajectory` file.
    fn write(&self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = unicode_artifact_path(py, path)?;
        let bytes = self
            .native
            .to_bytes()
            .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
        py.detach(move || write_artifact_bytes(&path, &bytes, TRAJECTORY_FILE_SPEC))
            .map_err(|diagnostic| crate::error::compatibility_error(py, &[diagnostic]))
    }

    /// Read one producer-independent spatial Trajectory against its exact Plan.
    #[staticmethod]
    fn read(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        path: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let path = unicode_artifact_path(py, path)?;
        let bytes = py
            .detach(move || read_artifact_bytes(&path, TRAJECTORY_FILE_SPEC))
            .map_err(|diagnostic| crate::error::compatibility_error(py, &[diagnostic]))?;
        let native = eqiora_numerics::CommonTrajectory::from_bytes(&bytes, plan.native())
            .map_err(|diagnostic| crate::error::compatibility_error(py, &[diagnostic]))?;
        if native.spatial_states().is_none() {
            return Err(PyValueError::new_err(
                "Trajectory.read requires a transient-flow or fixed-reference FSI artifact",
            ));
        }
        Self::from_common(py, plan, native)
    }

    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }

    #[getter]
    fn geometry_digest(&self) -> &str {
        &self.geometry_digest
    }

    #[getter]
    fn correspondence_digest(&self) -> &str {
        &self.correspondence_digest
    }

    #[getter]
    fn mesh_digest(&self) -> &str {
        &self.mesh_digest
    }

    #[getter]
    fn realization_digest(&self) -> Option<&str> {
        self.realization_digest.as_deref()
    }

    #[getter]
    fn plan_identity(&self) -> Option<&str> {
        self.plan_identity.as_deref()
    }

    #[getter]
    fn run_digest(&self) -> Option<&str> {
        self.run_digest.as_deref()
    }

    #[getter]
    fn request_identity(&self) -> Option<&str> {
        self.request_identity.as_deref()
    }

    #[getter]
    fn digest(&self) -> &str {
        &self.trajectory_digest
    }

    #[getter]
    const fn dimension(&self) -> usize {
        2
    }

    #[getter]
    fn coordinates(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        match &self.common_mesh {
            Some(mesh) => mesh.borrow(py).coordinate_array(py),
            None => self.coordinates.numpy(py),
        }
    }

    #[getter]
    fn cells(&self, py: Python<'_>) -> PyResult<Py<PyArray2<u32>>> {
        match &self.common_mesh {
            Some(mesh) => mesh.borrow(py).cell_array(py),
            None => self.cells.numpy(py),
        }
    }

    #[getter]
    fn states(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.states.iter().map(|state| state.clone_ref(py)))?.unbind())
    }

    /// Select one accepted state by its exact step ordinal.
    #[pyo3(signature = (step, /))]
    fn state(&self, py: Python<'_>, step: u64) -> PyResult<Py<PyState>> {
        let index = self.state_lookup.get(&step).copied().ok_or_else(|| {
            PyIndexError::new_err(format!("trajectory has no accepted step {step}"))
        })?;
        Ok(self.states[index].clone_ref(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "Trajectory(digest={:?}, states={})",
            self.digest(),
            self.states.len()
        )
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBoundaryForce>()?;
    module.add_class::<PyBoundaryFlux>()?;
    module.add_class::<PyDerivedFieldSnapshot>()?;
    module.add_class::<PyFieldSample>()?;
    module.add_class::<PyFieldSnapshot>()?;
    module.add_class::<PyInitialField>()?;
    module.add_class::<PyState>()?;
    module.add_class::<PyTrajectory>()?;
    Ok(())
}
