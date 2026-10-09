//! Root common Plan resolution over an exact Model and applicable caller resources.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use eqiora::artifact::CanonicalModelArtifact;
use eqiora::backends::faer::FaerLinearSolver;
use eqiora_numerics::{
    CommonFsiPlan, CommonLinearPlan, CommonMethodRequest, CommonOdePlan, CommonScopedSpatialPolicy,
    CommonSolvePolicy, CommonSpatialPolicy, CommonTransientFlowPlan, ResolvedCommonPlan,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyModule, PyTuple};

use crate::error::{compatibility_error, validation_error};
use crate::meshing::PyMesh;
use crate::model::{PyModel, PyModelFieldRef};
use crate::model_io::{self, ArtifactFileSpec};

const PLAN_FILE_SPEC: ArtifactFileSpec = ArtifactFileSpec {
    artifact_name: "resolved Plan",
    extension: "eqplan",
    staging_name: "plan",
    // This is the pre-read counterpart of the canonical Plan decoder's bound.
    max_bytes: 256 * 1024 * 1024,
};

mod algebraic;
mod capability_view;
mod compatible;
mod eigen;
mod enforcement;
pub(crate) mod harmonic;
use capability_view::{
    PyElasticityPlanView, PyFixedReferenceFsiPlanView, PyFormulationKind,
    PyFormulationSelectionMode, PyFormulationView, PyIncompressibleFlowPlanView, PyLinearPlanView,
    PyOdePlanView, space_name,
};
mod event_policy;
mod forward_policy;
mod policy;
mod registration;
mod solver_request;
use policy::{
    PyBackwardEuler, PyCellCentered, PyCellCenteredTpfa, PyImplicitMidpoint, PyLinear, PyMiniP1,
    PyNewton, PyP1, PyPressureGauge2d, PyQ1, PyScopedSpatialBinding, PySolverPlanningObjective,
    PyTetrahedralEdge, PyTetrahedralFace, PyTsitouras45, ScopedSpatialKind,
    spatial_handle_from_request,
};
pub(crate) use registration::register;
mod scaling;
use scaling::{PyIncompressibleScales, PyIncompressibleScaling, PyIncompressibleScalingReceipt2d};
mod resolved_solve;
use resolved_solve::{
    PyResolvedLinear, PyResolvedNewton, SolverPlanningAudit, solve_handles_from_native,
};
mod resolved_execution;
use resolved_execution::PyResolvedExecution;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpatialPolicy {
    Q1,
    TetrahedralEdge,
    TetrahedralFace,
    CellCenteredTpfa,
    MiniP1,
    CellCentered,
}

#[derive(Debug)]
enum SpatialHandle {
    Uniform(SpatialPolicy),
    Scoped(Vec<Py<PyScopedSpatialBinding>>),
}

#[derive(Debug)]
enum RequestedSolveHandle {
    Eigen(Py<eigen::PyHermitianEigen>),
    Linear(Py<PyLinear>),
    Newton(Py<PyNewton>),
}

#[derive(Debug)]
enum ResolvedSolveHandle {
    Eigen(Py<eigen::PyHermitianEigen>),
    Linear(Py<PyResolvedLinear>),
    Newton(Py<PyResolvedNewton>),
}

#[derive(Debug)]
enum TemporalHandle {
    BackwardEuler(Py<PyBackwardEuler>),
    Tsitouras45(Py<PyTsitouras45>),
    ImplicitMidpoint(Py<PyImplicitMidpoint>),
}

/// Immutable common Plan owning one exact Model, Mesh, and effective policy set.
#[pyclass(name = "Plan", module = "eqiora._eqiora", frozen, skip_from_py_object)]
#[derive(Debug)]
pub(crate) struct PyPlan {
    native: ResolvedCommonPlan,
    model: Py<PyModel>,
    mesh: Option<Py<PyMesh>>,
    spatial: Option<SpatialHandle>,
    requested_solve: Option<RequestedSolveHandle>,
    solve: Option<ResolvedSolveHandle>,
    temporal: Option<TemporalHandle>,
}

impl PyPlan {
    fn from_native_artifact(py: Python<'_>, native: ResolvedCommonPlan) -> PyResult<Self> {
        let model_bytes = native
            .model_artifact()
            .canonical_json()
            .map_err(|diagnostic| crate::error::internal_diagnostic_error(py, &[diagnostic]))?;
        let model = match model_io::decode_model(&model_bytes)
            .map_err(|diagnostics| crate::error::internal_diagnostic_error(py, &diagnostics))?
        {
            model_io::DecodedModel::Document(document) => PyModel::from_document(py, *document),
            model_io::DecodedModel::Deferred(artifact) => PyModel::from_artifact(py, artifact),
        }?;
        let model = Py::new(py, model)?;
        let mesh = native
            .authenticated_mesh()
            .map(|owner| PyMesh::from_authenticated(py, owner).and_then(|mesh| Py::new(py, mesh)))
            .transpose()?;
        let spatial = native
            .canonical_method_request()
            .map(|request| spatial_handle_from_request(py, native.model_digest(), request))
            .transpose()?;
        let (requested_solve, solve) = solve_handles_from_native(py, &native)?;
        let temporal = if let Some(temporal) = native.backward_euler() {
            Some(TemporalHandle::BackwardEuler(Py::new(
                py,
                PyBackwardEuler::from_native(temporal),
            )?))
        } else if let Some(temporal) = native.ode_temporal() {
            let data = policy::OdePolicyData::from_native(
                py,
                native.model_digest(),
                temporal.clone(),
                model
                    .borrow(py)
                    .document()
                    .map_err(|error| validation_error(py, &[error]))?,
            )?;
            Some(match temporal.method() {
                eqiora::time::TimeMethod::Tsitouras45 => {
                    TemporalHandle::Tsitouras45(Py::new(py, PyTsitouras45 { data })?)
                }
                eqiora::time::TimeMethod::ImplicitMidpoint => {
                    TemporalHandle::ImplicitMidpoint(Py::new(py, PyImplicitMidpoint { data })?)
                }
                _ => unreachable!("validated ODE policy"),
            })
        } else {
            None
        };
        Ok(Self {
            native,
            model,
            mesh,
            spatial,
            requested_solve,
            solve,
            temporal,
        })
    }

    pub(crate) fn model_handle(&self, py: Python<'_>) -> Py<PyModel> {
        self.model.clone_ref(py)
    }

    pub(crate) fn mesh_handle(&self, py: Python<'_>) -> Py<PyMesh> {
        self.mesh
            .as_ref()
            .expect("spatial common Plan owns an exact Mesh")
            .clone_ref(py)
    }

    pub(crate) fn ode_native(&self) -> Option<&CommonOdePlan> {
        match &self.native {
            ResolvedCommonPlan::Ode(plan) => Some(plan),
            _ => None,
        }
    }

    pub(crate) fn transient_native(&self) -> Option<&CommonTransientFlowPlan> {
        match &self.native {
            ResolvedCommonPlan::TransientFlow(plan) => Some(plan),
            _ => None,
        }
    }

    pub(crate) fn fsi_native(&self) -> Option<&CommonFsiPlan> {
        match &self.native {
            ResolvedCommonPlan::Fsi(plan) => Some(plan),
            _ => None,
        }
    }

    pub(crate) fn linear_native(&self) -> Option<&CommonLinearPlan> {
        match &self.native {
            ResolvedCommonPlan::Linear(plan) => Some(plan),
            _ => None,
        }
    }

    pub(crate) fn native(&self) -> &ResolvedCommonPlan {
        &self.native
    }

    pub(crate) fn package_compilation_digest_value(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<String>> {
        self.model
            .borrow(py)
            .package_compilation_digest_value()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }
}

#[pymethods]
impl PyPlan {
    /// Original real Model retained by a harmonic response restriction.
    #[getter]
    fn harmonic_original_model(&self, py: Python<'_>) -> PyResult<Option<PyModel>> {
        harmonic::original_model(&self.native)
            .map(|model| PyModel::from_artifact(py, model.clone()))
            .transpose()
    }

    /// Named exact original/derived Field pairs in the declared amplitude order.
    #[getter]
    fn harmonic_amplitudes(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        harmonic::amplitudes(py, &self.native)
    }

    /// Positive angular frequency in inverse seconds, distinct from cyclic frequency.
    #[getter]
    fn harmonic_angular_frequency(&self) -> Option<f64> {
        match &self.native {
            ResolvedCommonPlan::Algebraic(plan) => plan.harmonic_angular_frequency(),
            ResolvedCommonPlan::Linear(plan) => plan.harmonic_angular_frequency(),
            _ => None,
        }
    }

    /// Explicit finite mathematical enforcement, separate from the Model.
    #[getter]
    fn enforcement(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        enforcement::from_plan(py, self)
    }

    /// Canonical self-contained bytes of this complete resolved Plan.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.native
            .to_bytes()
            .map(|bytes| PyBytes::new(py, &bytes))
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    /// Decode and independently resolve one exact self-contained Plan.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        ResolvedCommonPlan::from_bytes(
            data,
            &FaerLinearSolver,
            eqiora::backends::diffsol::DiffsolTimeBackend::CAPABILITIES,
        )
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))
        .and_then(|native| Self::from_native_artifact(py, native))
    }

    /// Atomically write this exact resolved Plan to an `.eqplan` file.
    fn write(&self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = model_io::unicode_artifact_path(py, path)?;
        let bytes = self
            .native
            .to_bytes()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        py.detach(move || model_io::write_artifact_bytes(&path, &bytes, PLAN_FILE_SPEC))
            .map_err(|diagnostic| compatibility_error(py, &[diagnostic]))
    }

    /// Read and independently resolve one exact canonical `.eqplan` file.
    #[staticmethod]
    fn read(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Self> {
        let path = model_io::unicode_artifact_path(py, path)?;
        let native = py
            .detach(move || {
                let bytes = model_io::read_artifact_bytes(&path, PLAN_FILE_SPEC)?;
                ResolvedCommonPlan::from_bytes(
                    &bytes,
                    &FaerLinearSolver,
                    eqiora::backends::diffsol::DiffsolTimeBackend::CAPABILITIES,
                )
            })
            .map_err(|diagnostic| compatibility_error(py, &[diagnostic]))?;
        Self::from_native_artifact(py, native)
    }

    #[getter]
    fn identity(&self) -> &str {
        self.native.identity()
    }
    #[getter]
    fn model_id(&self) -> &str {
        self.native.model_id()
    }
    #[getter]
    fn model_digest(&self) -> &str {
        self.native.model_digest()
    }
    #[getter]
    fn model_revision(&self) -> u64 {
        self.native.model_revision()
    }
    #[getter]
    fn package_compilation_digest(&self, py: Python<'_>) -> PyResult<Option<String>> {
        self.package_compilation_digest_value(py)
    }
    #[getter]
    fn geometry_digest(&self) -> Option<&str> {
        self.native.geometry_digest()
    }
    #[getter]
    fn mesh_digest(&self) -> Option<&str> {
        self.native.mesh_digest()
    }
    #[getter]
    fn correspondence_digest(&self) -> Option<&str> {
        self.native.correspondence_digest()
    }
    #[getter]
    fn production_digest(&self) -> Option<&str> {
        self.native.production_digest()
    }
    #[getter]
    fn realization_digest(&self) -> Option<&str> {
        self.native.realization_digest()
    }
    #[getter]
    fn model(&self, py: Python<'_>) -> Py<PyModel> {
        self.model.clone_ref(py)
    }
    #[getter]
    fn mesh(&self, py: Python<'_>) -> Option<Py<PyMesh>> {
        self.mesh.as_ref().map(|mesh| mesh.clone_ref(py))
    }
    #[getter]
    fn formulation(&self, py: Python<'_>) -> PyResult<Option<Py<PyFormulationView>>> {
        let description = self.native.formulation();
        description
            .map(|description| {
                Py::new(
                    py,
                    PyFormulationView::from_native(
                        description,
                        self.native.model_digest().to_owned(),
                    ),
                )
            })
            .transpose()
    }
    #[getter]
    fn capability(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.native {
            ResolvedCommonPlan::Eigen(plan) => eigen::view(py, plan),
            ResolvedCommonPlan::Algebraic(plan) => algebraic::view(py, plan),
            ResolvedCommonPlan::Ode(plan) => Py::new(
                py,
                PyOdePlanView {
                    backend: plan.backend().id(),
                    backend_version: plan.backend().version(),
                },
            )
            .map(Py::into_any),
            ResolvedCommonPlan::Linear(plan) => Py::new(
                py,
                PyLinearPlanView {
                    fields: plan
                        .fields()
                        .map(|(field, _)| {
                            PyModelFieldRef::from_exact(
                                plan.model_digest().to_owned(),
                                field.ulid().to_string(),
                            )
                        })
                        .collect(),
                    coefficient_sampling: match plan.spatial() {
                        eqiora_numerics::CommonSpatialPolicy::Q1
                        | eqiora_numerics::CommonSpatialPolicy::TetrahedralEdge
                        | eqiora_numerics::CommonSpatialPolicy::TetrahedralFace
                        | eqiora_numerics::CommonSpatialPolicy::CellCentered => "quadrature-point",
                        eqiora_numerics::CommonSpatialPolicy::CellCenteredTpfa => "facet-centroid",
                        _ => unreachable!("common linear Plan cannot own this policy"),
                    },
                    face_coefficient_policy: match plan.spatial() {
                        eqiora_numerics::CommonSpatialPolicy::Q1
                        | eqiora_numerics::CommonSpatialPolicy::TetrahedralEdge
                        | eqiora_numerics::CommonSpatialPolicy::TetrahedralFace
                        | eqiora_numerics::CommonSpatialPolicy::CellCentered => "not-applicable",
                        eqiora_numerics::CommonSpatialPolicy::CellCenteredTpfa => {
                            "direct-centroid-evaluation"
                        }
                        _ => unreachable!("common linear Plan cannot own this policy"),
                    },
                },
            )
            .map(Py::into_any),
            ResolvedCommonPlan::Elasticity(plan) => Py::new(
                py,
                PyElasticityPlanView {
                    displacement: PyModelFieldRef::from_exact(
                        plan.model_digest().to_owned(),
                        plan.displacement_field_id().to_owned(),
                    ),
                },
            )
            .map(Py::into_any),
            ResolvedCommonPlan::SteadyStokes(plan) => Py::new(
                py,
                PyIncompressibleFlowPlanView {
                    kind: "steady-stokes",
                    velocity: PyModelFieldRef::from_exact(
                        plan.model_digest().to_owned(),
                        plan.velocity_field_id().to_owned(),
                    ),
                    pressure: PyModelFieldRef::from_exact(
                        plan.model_digest().to_owned(),
                        plan.pressure_field_id().to_owned(),
                    ),
                    velocity_space: space_name(plan.velocity_space()),
                    pressure_space: space_name(plan.pressure_space()),
                    pressure_gauge: None,
                    scaling: Py::new(py, PyIncompressibleScales::from_native(plan.scales()))?,
                    scaling_receipt: Py::new(
                        py,
                        PyIncompressibleScalingReceipt2d::from_native(
                            plan.scaling_receipt().clone(),
                        ),
                    )?,
                },
            )
            .map(Py::into_any),
            ResolvedCommonPlan::TransientFlow(plan) => Py::new(
                py,
                PyIncompressibleFlowPlanView {
                    kind: "transient-incompressible-flow",
                    velocity: PyModelFieldRef::from_exact(
                        plan.model_digest().to_owned(),
                        plan.velocity_field_id().to_owned(),
                    ),
                    pressure: PyModelFieldRef::from_exact(
                        plan.model_digest().to_owned(),
                        plan.pressure_field_id().to_owned(),
                    ),
                    velocity_space: space_name(plan.velocity_space()),
                    pressure_space: space_name(plan.pressure_space()),
                    pressure_gauge: Some(plan.gauge().into()),
                    scaling: Py::new(py, PyIncompressibleScales::from_native(plan.scales()))?,
                    scaling_receipt: Py::new(
                        py,
                        PyIncompressibleScalingReceipt2d::from_native(
                            plan.scaling_receipt().clone(),
                        ),
                    )?,
                },
            )
            .map(Py::into_any),
            ResolvedCommonPlan::Fsi(plan) => Py::new(
                py,
                PyFixedReferenceFsiPlanView {
                    scaling: Py::new(py, PyIncompressibleScales::from_fsi(plan.scaling()))?,
                    scaling_receipt: Py::new(
                        py,
                        PyIncompressibleScalingReceipt2d::from_native(
                            plan.scaling_receipt().clone(),
                        ),
                    )?,
                },
            )
            .map(Py::into_any),
        }
    }
    #[getter]
    fn fields(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let model_digest = self.native.model_digest().to_owned();
        let fields = match &self.native {
            ResolvedCommonPlan::Eigen(plan) => [plan.mode_field(), plan.eigenvalue_field()]
                .into_iter()
                .chain(
                    plan.coordinate_embeddings()
                        .map(|(_, _, coordinate, _)| coordinate),
                )
                .map(|field| {
                    PyModelFieldRef::from_exact(model_digest.clone(), field.ulid().to_string())
                })
                .collect(),
            ResolvedCommonPlan::Algebraic(plan) => plan
                .symbols()
                .iter()
                .filter_map(|symbol| {
                    if let eqiora::kernel::SymbolRef::Field(field) = symbol {
                        Some(PyModelFieldRef::from_exact(
                            model_digest.clone(),
                            field.ulid().to_string(),
                        ))
                    } else {
                        None
                    }
                })
                .collect(),
            ResolvedCommonPlan::Ode(plan) => plan
                .state_coordinates()
                .filter_map(|coordinate| {
                    (coordinate.derivative_order() == 0
                        && coordinate.component() == 0
                        && !coordinate.is_imaginary())
                    .then_some(coordinate.field())
                })
                .map(|field| PyModelFieldRef::from_exact(model_digest.clone(), field.to_string()))
                .collect(),
            ResolvedCommonPlan::Linear(plan) => plan
                .fields()
                .map(|(field, _)| {
                    PyModelFieldRef::from_exact(model_digest.clone(), field.ulid().to_string())
                })
                .collect(),
            ResolvedCommonPlan::Elasticity(plan) => vec![PyModelFieldRef::from_exact(
                model_digest,
                plan.displacement_field_id().to_owned(),
            )],
            ResolvedCommonPlan::SteadyStokes(plan) => vec![
                PyModelFieldRef::from_exact(
                    model_digest.clone(),
                    plan.velocity_field_id().to_owned(),
                ),
                PyModelFieldRef::from_exact(model_digest, plan.pressure_field_id().to_owned()),
            ],
            ResolvedCommonPlan::TransientFlow(plan) => vec![
                PyModelFieldRef::from_exact(
                    model_digest.clone(),
                    plan.velocity_field_id().to_owned(),
                ),
                PyModelFieldRef::from_exact(model_digest, plan.pressure_field_id().to_owned()),
            ],
            ResolvedCommonPlan::Fsi(plan) => plan
                .field_ids()
                .iter()
                .map(|field| PyModelFieldRef::from_exact(model_digest.clone(), field.clone()))
                .collect(),
        };
        Ok(PyTuple::new(py, fields)?.unbind())
    }
    /// Exact Mesh (dimension, index) entities in one linear Field's coefficient order.
    #[pyo3(signature = (field, /))]
    fn field_coefficient_entities(
        &self,
        py: Python<'_>,
        field: PyRef<'_, PyModelFieldRef>,
    ) -> PyResult<Py<PyTuple>> {
        compatible::entities(py, self, &field)
    }

    /// Independent vertex-gradient columns with oriented edge entries; not a whole-operator nullspace claim.
    #[pyo3(signature = (field, /))]
    fn field_gradient_modes(
        &self,
        py: Python<'_>,
        field: PyRef<'_, PyModelFieldRef>,
    ) -> PyResult<Py<PyDict>> {
        compatible::gradient_modes(py, self, &field)
    }

    /// Exact face-integrated curl or cell-integrated divergence rows on this Field's support.
    #[pyo3(signature = (field, /))]
    fn field_exterior_derivative(
        &self,
        py: Python<'_>,
        field: PyRef<'_, PyModelFieldRef>,
    ) -> PyResult<Py<PyDict>> {
        compatible::exterior_derivative(py, self, &field)
    }

    #[getter]
    fn spatial(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.spatial
            .as_ref()
            .map(|spatial| match spatial {
                SpatialHandle::Uniform(SpatialPolicy::TetrahedralEdge) => {
                    Py::new(py, PyTetrahedralEdge).map(Py::into_any)
                }
                SpatialHandle::Uniform(SpatialPolicy::TetrahedralFace) => {
                    Py::new(py, PyTetrahedralFace).map(Py::into_any)
                }
                SpatialHandle::Uniform(SpatialPolicy::Q1) => Py::new(py, PyQ1).map(Py::into_any),
                SpatialHandle::Uniform(SpatialPolicy::CellCenteredTpfa) => {
                    Py::new(py, PyCellCenteredTpfa).map(Py::into_any)
                }
                SpatialHandle::Uniform(SpatialPolicy::MiniP1) => {
                    Py::new(py, PyMiniP1).map(Py::into_any)
                }
                SpatialHandle::Uniform(SpatialPolicy::CellCentered) => {
                    Py::new(py, PyCellCentered).map(Py::into_any)
                }
                SpatialHandle::Scoped(values) => {
                    PyTuple::new(py, values.iter().map(|value| value.clone_ref(py)))
                        .map(|value| value.unbind().into_any())
                }
            })
            .transpose()
    }
    #[getter]
    fn solve(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.solve.as_ref().map(|solve| match solve {
            ResolvedSolveHandle::Eigen(value) => value.clone_ref(py).into_any(),
            ResolvedSolveHandle::Linear(value) => value.clone_ref(py).into_any(),
            ResolvedSolveHandle::Newton(value) => value.clone_ref(py).into_any(),
        })
    }
    #[getter]
    fn requested_solve(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.requested_solve.as_ref().map(|solve| match solve {
            RequestedSolveHandle::Eigen(value) => value.clone_ref(py).into_any(),
            RequestedSolveHandle::Linear(value) => value.clone_ref(py).into_any(),
            RequestedSolveHandle::Newton(value) => value.clone_ref(py).into_any(),
        })
    }
    #[getter]
    fn temporal(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.temporal.as_ref().map(|value| match value {
            TemporalHandle::BackwardEuler(value) => value.clone_ref(py).into_any(),
            TemporalHandle::Tsitouras45(value) => value.clone_ref(py).into_any(),
            TemporalHandle::ImplicitMidpoint(value) => value.clone_ref(py).into_any(),
        })
    }
    #[getter]
    fn execution(&self, py: Python<'_>) -> PyResult<Py<PyResolvedExecution>> {
        Py::new(py, PyResolvedExecution)
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|other| self.identity() == other.identity())
    }
    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.identity().hash(&mut hasher);
        hasher.finish() as isize
    }
    fn __repr__(&self) -> String {
        format!(
            "Plan(identity={:?}, model_digest={:?}, mesh_digest={:?})",
            self.identity(),
            self.model_digest(),
            self.mesh_digest()
        )
    }
}

mod resolution;
use resolution::resolve_plan;

#[cfg(test)]
#[path = "common_plan/tests.rs"]
mod tests;
