//! Root common Plan resolution over an exact Model and applicable caller resources.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use eqiora::artifact::CanonicalModelArtifact;
use eqiora::backends::faer::FaerLinearSolver;
use eqiora_numerics::{
    CommonFsiPlan, CommonMethodRequest, CommonOdePlan, CommonScalarPlan, CommonScopedSpatialPolicy,
    CommonSolvePolicy, CommonSpatialPolicy, CommonTransientFlowPlan, ResolvedCommonPlan,
    resolve_common_ode_plan, resolve_common_plan,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyModule, PyTuple};

use crate::error::{compatibility_error, validation_error};
use crate::meshing::PyMesh;
use crate::model::{PyModel, PyModelDomainRef, PyModelFieldRef};
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
mod enforcement;
use capability_view::{
    PyElasticityPlanView, PyFixedReferenceFsiPlanView, PyFormulationKind,
    PyFormulationSelectionMode, PyFormulationView, PyIncompressibleFlowPlanView, PyOdePlanView,
    PyScalarPlanView, space_name,
};
mod event_policy;
mod forward_policy;
mod policy;
mod registration;
mod solver_request;
use policy::{
    PyBackwardEuler, PyCellCentered, PyCellCenteredTpfa, PyLinear, PyMiniP1, PyNewton, PyP1,
    PyPressureGauge2d, PyQ1, PyScopedSpatialBinding, PySolverPlanningObjective, PyTsitouras45,
    ScopedSpatialKind,
};
pub(crate) use registration::register;
mod scaling;
use scaling::{PyIncompressibleScales, PyIncompressibleScaling, PyIncompressibleScalingReceipt2d};
mod resolved_solve;
use resolved_solve::{PyResolvedLinear, PyResolvedNewton, SolverPlanningAudit};
mod resolved_execution;
use resolved_execution::PyResolvedExecution;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpatialPolicy {
    Q1,
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
    Linear(Py<PyLinear>),
    Newton(Py<PyNewton>),
}

#[derive(Debug)]
enum ResolvedSolveHandle {
    Linear(Py<PyResolvedLinear>),
    Newton(Py<PyResolvedNewton>),
}

#[derive(Debug)]
enum TemporalHandle {
    BackwardEuler(Py<PyBackwardEuler>),
    Tsitouras45(Py<PyTsitouras45>),
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
        } else if let Some(temporal) = native.tsitouras45() {
            Some(TemporalHandle::Tsitouras45(Py::new(
                py,
                PyTsitouras45::from_native(
                    py,
                    native.model_digest(),
                    temporal.clone(),
                    model
                        .borrow(py)
                        .document()
                        .map_err(|error| validation_error(py, &[error]))?,
                )?,
            )?))
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

    pub(crate) fn scalar_native(&self) -> Option<&CommonScalarPlan> {
        match &self.native {
            ResolvedCommonPlan::Scalar(plan) => Some(plan),
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

fn spatial_handle_from_request(
    py: Python<'_>,
    model_digest: &str,
    request: CommonMethodRequest,
) -> PyResult<SpatialHandle> {
    match request {
        CommonMethodRequest::Uniform(policy)
        | CommonMethodRequest::Exact {
            spatial: policy, ..
        } => {
            let policy = match policy {
                CommonSpatialPolicy::Q1 => SpatialPolicy::Q1,
                CommonSpatialPolicy::CellCenteredTpfa => SpatialPolicy::CellCenteredTpfa,
                CommonSpatialPolicy::MiniP1 => SpatialPolicy::MiniP1,
                CommonSpatialPolicy::CellCentered => SpatialPolicy::CellCentered,
                CommonSpatialPolicy::P1 => {
                    return Err(PyTypeError::new_err(
                        "uniform P1 is not an admitted common Plan policy",
                    ));
                }
            };
            Ok(SpatialHandle::Uniform(policy))
        }
        CommonMethodRequest::Scoped(bindings) => bindings
            .into_iter()
            .map(|binding| {
                let policy = match binding.policy() {
                    CommonSpatialPolicy::MiniP1 => ScopedSpatialKind::MiniP1,
                    CommonSpatialPolicy::P1 => ScopedSpatialKind::P1,
                    CommonSpatialPolicy::Q1
                    | CommonSpatialPolicy::CellCenteredTpfa
                    | CommonSpatialPolicy::CellCentered => {
                        return Err(PyTypeError::new_err(
                            "persisted scoped Plan contains an unsupported spatial policy",
                        ));
                    }
                };
                Py::new(
                    py,
                    PyScopedSpatialBinding {
                        domain: PyModelDomainRef::from_exact(
                            model_digest.to_owned(),
                            binding.domain().ulid().to_string(),
                        ),
                        policy,
                    },
                )
            })
            .collect::<PyResult<Vec<_>>>()
            .map(SpatialHandle::Scoped),
    }
}

fn solve_handles_from_native(
    py: Python<'_>,
    native: &ResolvedCommonPlan,
) -> PyResult<(Option<RequestedSolveHandle>, Option<ResolvedSolveHandle>)> {
    let Some(request) = native.canonical_solve_request() else {
        return Ok((None, None));
    };
    let requested = match request {
        CommonSolvePolicy::Linear(linear) => {
            RequestedSolveHandle::Linear(Py::new(py, PyLinear::from_native(linear))?)
        }
        CommonSolvePolicy::Newton { nonlinear, linear } => {
            let linear = Py::new(py, PyLinear::from_native(linear))?;
            RequestedSolveHandle::Newton(Py::new(py, PyNewton::from_native(linear, nonlinear))?)
        }
    };
    let solver_planning_audit = native.solver_planning_objective().map(|objective| {
        SolverPlanningAudit::new(
            objective.into(),
            native
                .solver_planning_policy_id()
                .expect("planned solver retains its policy identity"),
            native
                .selected_solver_candidate_id()
                .expect("planned solver retains its selected candidate"),
            native
                .selected_solver_evidence_case()
                .expect("planned solver retains its evidence identity"),
            native.solver_planning_reasons().to_vec(),
        )
    });
    let linear = Py::new(
        py,
        PyResolvedLinear::new(
            native
                .effective_solver()
                .expect("spatial common Plan owns an effective linear solver"),
            native
                .operator_properties()
                .expect("spatial common Plan owns operator properties"),
            native
                .linear_solver_provider()
                .expect("spatial Plan owns its exact provider"),
            solver_planning_audit,
        ),
    )?;
    let resolved = match native {
        ResolvedCommonPlan::TransientFlow(plan) => ResolvedSolveHandle::Newton(Py::new(
            py,
            PyResolvedNewton::new(linear, plan.nonlinear()),
        )?),
        ResolvedCommonPlan::Ode(_) => unreachable!("ODE Plan has no common solve request"),
        ResolvedCommonPlan::Algebraic(_)
        | ResolvedCommonPlan::Scalar(_)
        | ResolvedCommonPlan::Elasticity(_)
        | ResolvedCommonPlan::SteadyStokes(_)
        | ResolvedCommonPlan::Fsi(_) => ResolvedSolveHandle::Linear(linear),
    };
    Ok((Some(requested), Some(resolved)))
}

#[pymethods]
impl PyPlan {
    /// Explicit finite mathematical enforcement, separate from the Model.
    #[getter]
    fn enforcement(&self) -> Option<enforcement::PyActiveSet> {
        enforcement::from_plan(self)
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
            eqiora::backends::diffsol::DIFFSOL_TIME_BACKEND,
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
                    eqiora::backends::diffsol::DIFFSOL_TIME_BACKEND,
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
            .map(|description| Py::new(py, PyFormulationView::from_native(description)))
            .transpose()
    }
    #[getter]
    fn capability(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.native {
            ResolvedCommonPlan::Algebraic(plan) => algebraic::view(py, plan),
            ResolvedCommonPlan::Ode(plan) => Py::new(
                py,
                PyOdePlanView {
                    backend: plan.backend().id(),
                    backend_version: plan.backend().version(),
                },
            )
            .map(Py::into_any),
            ResolvedCommonPlan::Scalar(plan) => Py::new(
                py,
                PyScalarPlanView {
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
                        eqiora_numerics::CommonSpatialPolicy::Q1 => "quadrature-point",
                        eqiora_numerics::CommonSpatialPolicy::CellCenteredTpfa => "facet-centroid",
                        _ => unreachable!("common scalar Plan cannot own a non-scalar policy"),
                    },
                    face_coefficient_policy: match plan.spatial() {
                        eqiora_numerics::CommonSpatialPolicy::Q1 => "not-applicable",
                        eqiora_numerics::CommonSpatialPolicy::CellCenteredTpfa => {
                            "direct-centroid-evaluation"
                        }
                        _ => unreachable!("common scalar Plan cannot own a non-scalar policy"),
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
            ResolvedCommonPlan::Fsi(plan) => {
                let fields = plan.field_ids();
                let model = plan.model_digest().to_owned();
                Py::new(
                    py,
                    PyFixedReferenceFsiPlanView {
                        fluid_velocity: PyModelFieldRef::from_exact(
                            model.clone(),
                            fields[0].clone(),
                        ),
                        pressure: PyModelFieldRef::from_exact(model.clone(), fields[1].clone()),
                        solid_velocity: PyModelFieldRef::from_exact(
                            model.clone(),
                            fields[2].clone(),
                        ),
                        displacement: PyModelFieldRef::from_exact(model, fields[3].clone()),
                        scaling: Py::new(py, PyIncompressibleScales::from_fsi(plan.scaling()))?,
                        scaling_receipt: Py::new(
                            py,
                            PyIncompressibleScalingReceipt2d::from_native(
                                plan.scaling_receipt().clone(),
                            ),
                        )?,
                    },
                )
                .map(Py::into_any)
            }
        }
    }
    #[getter]
    fn fields(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let model_digest = self.native.model_digest().to_owned();
        let fields = match &self.native {
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
                .field_ids()
                .map(|field| PyModelFieldRef::from_exact(model_digest.clone(), field.to_string()))
                .collect(),
            ResolvedCommonPlan::Scalar(plan) => plan
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
    #[getter]
    fn spatial(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.spatial
            .as_ref()
            .map(|spatial| match spatial {
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
            ResolvedSolveHandle::Linear(value) => value.clone_ref(py).into_any(),
            ResolvedSolveHandle::Newton(value) => value.clone_ref(py).into_any(),
        })
    }
    #[getter]
    fn requested_solve(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.requested_solve.as_ref().map(|solve| match solve {
            RequestedSolveHandle::Linear(value) => value.clone_ref(py).into_any(),
            RequestedSolveHandle::Newton(value) => value.clone_ref(py).into_any(),
        })
    }
    #[getter]
    fn temporal(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.temporal.as_ref().map(|value| match value {
            TemporalHandle::BackwardEuler(value) => value.clone_ref(py).into_any(),
            TemporalHandle::Tsitouras45(value) => value.clone_ref(py).into_any(),
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

#[pyfunction(name = "_resolve_plan")]
#[pyo3(signature = (model, /, *, mesh=None, spatial=None, formulation=None, solve=None, scaling=None, temporal=None, enforcement=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "PyO3 counts its injected Python token beside the seven-field public resolve boundary"
)]
fn resolve_plan(
    py: Python<'_>,
    model: Py<PyModel>,
    mesh: Option<Py<PyMesh>>,
    spatial: Option<&Bound<'_, PyAny>>,
    formulation: Option<&Bound<'_, PyAny>>,
    solve: Option<&Bound<'_, PyAny>>,
    scaling: Option<&Bound<'_, PyAny>>,
    temporal: Option<&Bound<'_, PyAny>>,
    enforcement: Option<PyRef<'_, enforcement::PyActiveSet>>,
) -> PyResult<PyPlan> {
    if enforcement.is_some() && (mesh.is_some() || spatial.is_some() || temporal.is_some()) {
        return Err(PyTypeError::new_err(
            "finite enforcement cannot accompany spatial or temporal resolution",
        ));
    }
    let ode_temporal = temporal.and_then(|value| value.extract::<Py<PyTsitouras45>>().ok());
    if let Some(temporal_handle) = ode_temporal {
        if mesh.is_some()
            || spatial.is_some()
            || formulation.is_some_and(|value| !value.is_none())
            || solve.is_some()
            || scaling.is_some_and(|value| !value.is_none())
        {
            return Err(PyTypeError::new_err(
                "no-Mesh explicit ODE resolve accepts only model and temporal=eqiora.time.Tsitouras45(...)",
            ));
        }
        let model_ref = model.borrow(py);
        let artifact = model_ref.artifact();
        let reference = artifact
            .artifact_reference()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        let temporal_ref = temporal_handle.borrow(py);
        if !temporal_ref.belongs_to_model(py, &reference.artifact().to_string()) {
            return Err(PyTypeError::new_err(
                "Tsitouras45 absolute_tolerances must use exact FieldRefs from this Model",
            ));
        }
        let program = artifact
            .to_program()
            .map_err(|diagnostics| validation_error(py, &diagnostics))?;
        let native = resolve_common_ode_plan(
            artifact,
            &program,
            temporal_ref.native.clone(),
            eqiora::backends::diffsol::DIFFSOL_TIME_BACKEND,
        )
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        drop(temporal_ref);
        drop(model_ref);
        return Ok(PyPlan {
            native,
            model,
            mesh: None,
            spatial: None,
            requested_solve: None,
            solve: None,
            temporal: Some(TemporalHandle::Tsitouras45(temporal_handle)),
        });
    }

    if mesh.is_none() && spatial.is_none() && temporal.is_none() {
        return algebraic::resolve(
            py,
            model,
            solve,
            formulation,
            scaling,
            enforcement.as_deref(),
        );
    }

    let mesh = mesh.ok_or_else(|| PyTypeError::new_err("spatial resolve requires mesh=Mesh"))?;
    let spatial_value =
        spatial.ok_or_else(|| PyTypeError::new_err("spatial resolve requires spatial policy"))?;
    let solve =
        solve.ok_or_else(|| PyTypeError::new_err("spatial resolve requires solve policy"))?;
    let (spatial_request, spatial_handle) = if spatial_value.extract::<PyRef<'_, PyQ1>>().is_ok() {
        (
            CommonMethodRequest::Uniform(CommonSpatialPolicy::Q1),
            SpatialHandle::Uniform(SpatialPolicy::Q1),
        )
    } else if spatial_value
        .extract::<PyRef<'_, PyCellCenteredTpfa>>()
        .is_ok()
    {
        (
            CommonMethodRequest::Uniform(CommonSpatialPolicy::CellCenteredTpfa),
            SpatialHandle::Uniform(SpatialPolicy::CellCenteredTpfa),
        )
    } else if spatial_value.extract::<PyRef<'_, PyMiniP1>>().is_ok() {
        (
            CommonMethodRequest::Uniform(CommonSpatialPolicy::MiniP1),
            SpatialHandle::Uniform(SpatialPolicy::MiniP1),
        )
    } else if spatial_value.extract::<PyRef<'_, PyCellCentered>>().is_ok() {
        (
            CommonMethodRequest::Uniform(CommonSpatialPolicy::CellCentered),
            SpatialHandle::Uniform(SpatialPolicy::CellCentered),
        )
    } else if let Ok(tuple) = spatial_value.cast::<PyTuple>() {
        if tuple.is_empty() {
            return Err(PyTypeError::new_err(
                "scoped spatial policy tuple must be nonempty",
            ));
        }
        let mut native = Vec::with_capacity(tuple.len());
        let mut handles = Vec::with_capacity(tuple.len());
        for value in tuple.iter() {
            let handle = value.extract::<Py<PyScopedSpatialBinding>>().map_err(|_| {
                PyTypeError::new_err("scoped spatial tuples must contain only MiniP1.at(DomainRef) or P1.at(DomainRef)")
            })?;
            let binding = handle.borrow(py);
            let model_digest = eqiora::artifact::ArtifactDigest::from_hex(
                binding.domain.exact_model_digest().to_owned(),
            )
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            let domain = ulid::Ulid::from_string(binding.domain.exact_id())
                .map(eqiora::Id::<eqiora::kinds::Domain>::from_ulid)
                .map_err(|_| {
                    PyTypeError::new_err("DomainRef contains an invalid exact Domain ULID")
                })?;
            let policy = match binding.policy {
                ScopedSpatialKind::MiniP1 => CommonSpatialPolicy::MiniP1,
                ScopedSpatialKind::P1 => CommonSpatialPolicy::P1,
            };
            native.push(CommonScopedSpatialPolicy::new(model_digest, domain, policy));
            drop(binding);
            handles.push(handle);
        }
        (
            CommonMethodRequest::Scoped(native),
            SpatialHandle::Scoped(handles),
        )
    } else {
        return Err(PyTypeError::new_err(
            "spatial must be a supported uniform policy or an exact tuple of Domain-scoped policies",
        ));
    };
    let scaling = match scaling {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) => Some(
            value
                .extract::<PyRef<'_, PyIncompressibleScaling>>()
                .map_err(|_| {
                    PyTypeError::new_err(
                        "scaling must be eqiora.fluid.IncompressibleScaling or None",
                    )
                })?
                .native(),
        ),
    };
    let formulation = match formulation {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) => Some(
            (*value
                .extract::<PyRef<'_, PyFormulationKind>>()
                .map_err(|_| {
                    PyTypeError::new_err("formulation must be eqiora.FormulationKind or None")
                })?)
            .into(),
        ),
    };
    let method_request = match (spatial_request, formulation) {
        (CommonMethodRequest::Uniform(spatial), Some(formulation)) => CommonMethodRequest::Exact {
            spatial,
            formulation,
        },
        (request, None) => request,
        (CommonMethodRequest::Scoped(_), Some(_)) => {
            return Err(PyTypeError::new_err(
                "exact formulation requests require one supported uniform spatial policy",
            ));
        }
        (CommonMethodRequest::Exact { .. }, Some(_)) => {
            unreachable!("Python constructs exact method requests only in this match")
        }
    };
    let (solve_native, requested_solve_handle) = if let Ok(linear) = solve.extract::<Py<PyLinear>>()
    {
        let native = linear.borrow(py).native;
        (
            CommonSolvePolicy::Linear(native),
            RequestedSolveHandle::Linear(linear),
        )
    } else if let Ok(newton) = solve.extract::<Py<PyNewton>>() {
        let newton_ref = newton.borrow(py);
        let linear = newton_ref.linear.borrow(py).native;
        let native = CommonSolvePolicy::Newton {
            nonlinear: newton_ref.native,
            linear,
        };
        drop(newton_ref);
        (native, RequestedSolveHandle::Newton(newton))
    } else {
        return Err(PyTypeError::new_err(
            "solve must be eqiora.solve.Linear or eqiora.solve.Newton",
        ));
    };
    let (temporal_native, temporal_handle) = match temporal {
        None => (None, None),
        Some(value) if value.is_none() => (None, None),
        Some(value) => {
            let handle = value.extract::<Py<PyBackwardEuler>>().map_err(|_| {
                PyTypeError::new_err("temporal must be eqiora.time.BackwardEuler or None")
            })?;
            let native = handle.borrow(py).native;
            (Some(native), Some(TemporalHandle::BackwardEuler(handle)))
        }
    };
    let model_ref = model.borrow(py);
    let mesh_ref = mesh.borrow(py);
    let owner = mesh_ref
        .authenticated_common_mesh()
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?
        .ok_or_else(|| {
            PyTypeError::new_err("mesh must be an authenticated caller-owned common Mesh")
        })?;
    let native = resolve_common_plan(
        model_ref.artifact(),
        owner,
        method_request,
        solve_native,
        scaling,
        temporal_native,
        &FaerLinearSolver,
        model_ref
            .authored_formulation_projection()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?,
    )
    .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
    if native.mesh_digest() != Some(mesh_ref.exact_mesh_digest()) {
        return Err(PyTypeError::new_err(
            "resolved Plan did not retain the exact caller Mesh occurrence",
        ));
    }
    let solver_planning_audit = native.solver_planning_objective().map(|objective| {
        SolverPlanningAudit::new(
            objective.into(),
            native
                .solver_planning_policy_id()
                .expect("planned solver retains its policy identity"),
            native
                .selected_solver_candidate_id()
                .expect("planned solver retains its selected candidate"),
            native
                .selected_solver_evidence_case()
                .expect("planned solver retains its evidence identity"),
            native.solver_planning_reasons().to_vec(),
        )
    });
    let linear = Py::new(
        py,
        PyResolvedLinear::new(
            native
                .effective_solver()
                .expect("spatial common Plan owns an effective linear solver"),
            native
                .operator_properties()
                .expect("spatial common Plan owns operator properties"),
            native
                .linear_solver_provider()
                .expect("spatial Plan owns its exact provider"),
            solver_planning_audit,
        ),
    )?;
    let solve_handle = match &native {
        ResolvedCommonPlan::TransientFlow(plan) => ResolvedSolveHandle::Newton(Py::new(
            py,
            PyResolvedNewton::new(linear, plan.nonlinear()),
        )?),
        ResolvedCommonPlan::Ode(_) => unreachable!("spatial resolver cannot return an ODE Plan"),
        ResolvedCommonPlan::Algebraic(_)
        | ResolvedCommonPlan::Scalar(_)
        | ResolvedCommonPlan::Elasticity(_)
        | ResolvedCommonPlan::SteadyStokes(_)
        | ResolvedCommonPlan::Fsi(_) => ResolvedSolveHandle::Linear(linear),
    };
    drop(mesh_ref);
    drop(model_ref);
    Ok(PyPlan {
        native,
        model,
        mesh: Some(mesh),
        spatial: Some(spatial_handle),
        requested_solve: Some(requested_solve_handle),
        solve: Some(solve_handle),
        temporal: temporal_handle,
    })
}

#[cfg(test)]
#[path = "common_plan/tests.rs"]
mod tests;
