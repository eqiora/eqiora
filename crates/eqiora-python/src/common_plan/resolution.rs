//! Resolve the public mathematical and numerical request into an exact Plan.
use super::*;

#[pyfunction(name = "_resolve_plan")]
#[pyo3(signature = (model, /, *, mesh=None, spatial=None, formulation=None, solve=None, scaling=None, temporal=None, enforcement=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "PyO3 counts its injected Python token beside the seven-field public resolve boundary"
)]
pub(super) fn resolve_plan(
    py: Python<'_>,
    model: Py<PyModel>,
    mesh: Option<Py<PyMesh>>,
    spatial: Option<&Bound<'_, PyAny>>,
    formulation: Option<&Bound<'_, PyAny>>,
    solve: Option<&Bound<'_, PyAny>>,
    scaling: Option<&Bound<'_, PyAny>>,
    temporal: Option<&Bound<'_, PyAny>>,
    enforcement: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyPlan> {
    if enforcement.is_some() && (mesh.is_some() || spatial.is_some() || temporal.is_some()) {
        return Err(PyTypeError::new_err(
            "finite enforcement cannot accompany spatial or temporal resolution",
        ));
    }
    if let Some(policy) =
        solve.and_then(|value| value.extract::<PyRef<'_, eigen::PyHermitianEigen>>().ok())
    {
        if mesh.is_some()
            || spatial.is_some()
            || temporal.is_some()
            || formulation.is_some()
            || scaling.is_some()
            || enforcement.is_some()
        {
            return Err(PyTypeError::new_err(
                "Hermitian eigen resolution accepts only model and its spectral solve policy",
            ));
        }
        return eigen::resolve(py, model, &policy);
    }
    let ode_temporal = temporal.and_then(|value| {
        value
            .extract::<Py<PyTsitouras45>>()
            .ok()
            .map(TemporalHandle::Tsitouras45)
            .or_else(|| {
                value
                    .extract::<Py<PyImplicitMidpoint>>()
                    .ok()
                    .map(TemporalHandle::ImplicitMidpoint)
            })
    });
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
        let digest = reference.artifact().to_string();
        let (policy, owned) = match &temporal_handle {
            TemporalHandle::Tsitouras45(handle) => {
                let value = handle.borrow(py);
                (
                    value.data.native.clone(),
                    value.data.belongs_to_model(py, &digest),
                )
            }
            TemporalHandle::ImplicitMidpoint(handle) => {
                let value = handle.borrow(py);
                (
                    value.data.native.clone(),
                    value.data.belongs_to_model(py, &digest),
                )
            }
            _ => unreachable!("ODE handle"),
        };
        if !owned {
            return Err(PyTypeError::new_err(
                "ODE tolerances must use exact FieldRefs from this Model",
            ));
        }
        let backend = if policy.method() == eqiora::time::TimeMethod::ImplicitMidpoint {
            eqiora::time::ImplicitMidpointTimeBackend::CAPABILITIES
        } else {
            eqiora::backends::diffsol::DiffsolTimeBackend::CAPABILITIES
        };
        let program = artifact
            .to_program()
            .map_err(|diagnostics| validation_error(py, &diagnostics))?;
        let native =
            eqiora_numerics::ResolvedCommonPlan::resolve_ode(artifact, &program, policy, backend)
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        drop(model_ref);
        return Ok(PyPlan {
            native,
            model,
            mesh: None,
            spatial: None,
            requested_solve: None,
            solve: None,
            temporal: Some(temporal_handle),
        });
    }

    if mesh.is_none() && spatial.is_none() && temporal.is_none() {
        return algebraic::resolve(py, model, solve, formulation, scaling, enforcement);
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
        .extract::<PyRef<'_, PyTetrahedralEdge>>()
        .is_ok()
    {
        (
            CommonMethodRequest::Uniform(CommonSpatialPolicy::TetrahedralEdge),
            SpatialHandle::Uniform(SpatialPolicy::TetrahedralEdge),
        )
    } else if spatial_value
        .extract::<PyRef<'_, PyTetrahedralFace>>()
        .is_ok()
    {
        (
            CommonMethodRequest::Uniform(CommonSpatialPolicy::TetrahedralFace),
            SpatialHandle::Uniform(SpatialPolicy::TetrahedralFace),
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
    let native = eqiora_numerics::ResolvedCommonPlan::resolve(
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
        ResolvedCommonPlan::Eigen(_) | ResolvedCommonPlan::Ode(_) => {
            unreachable!("spatial resolver cannot return an ODE Plan")
        }
        ResolvedCommonPlan::Algebraic(_)
        | ResolvedCommonPlan::Linear(_)
        | ResolvedCommonPlan::Elasticity(_)
        | ResolvedCommonPlan::SteadyStokes(_)
        | ResolvedCommonPlan::Fsi(_) => ResolvedSolveHandle::Linear(linear),
    };
    drop(mesh_ref);
    drop(model_ref);
    let model = if harmonic::original_model(&native).is_some() {
        Py::new(
            py,
            PyModel::from_artifact(py, native.model_artifact().clone())?,
        )?
    } else {
        model
    };
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
