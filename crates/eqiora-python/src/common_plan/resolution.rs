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
    let mut temporal_handle = temporal
        .filter(|value| !value.is_none())
        .map(|value| {
            value
                .extract::<Py<PyBackwardEuler>>()
                .ok()
                .map(TemporalHandle::BackwardEuler)
                .or_else(|| {
                    value
                        .extract::<Py<PyTsitouras45>>()
                        .ok()
                        .map(TemporalHandle::Tsitouras45)
                })
                .or_else(|| {
                    value
                        .extract::<Py<PyImplicitMidpoint>>()
                        .ok()
                        .map(TemporalHandle::ImplicitMidpoint)
                })
                .or_else(|| value.extract::<Py<PyBdf>>().ok().map(TemporalHandle::Bdf))
                .or_else(|| {
                    value
                        .extract::<Py<PyEulerMaruyama>>()
                        .ok()
                        .map(TemporalHandle::EulerMaruyama)
                })
                .or_else(|| {
                    value
                        .extract::<Py<PyPoissonEvents>>()
                        .ok()
                        .map(TemporalHandle::PoissonEvents)
                })
                .ok_or_else(|| {
                    PyTypeError::new_err("temporal must be an eqiora.time policy or None")
                })
        })
        .transpose()?;
    if mesh.is_none() && spatial.is_none() && temporal_handle.is_some() {
        let temporal_handle = temporal_handle.take().expect("present temporal policy");
        if formulation.is_some_and(|value| !value.is_none())
            || scaling.is_some_and(|value| !value.is_none())
        {
            return Err(PyTypeError::new_err(
                "no-Mesh time resolve accepts model, temporal and its consumed solve policy",
            ));
        }
        let solve_policy = solve.map(|value| solve_policy(py, value)).transpose()?;
        if matches!(
            temporal_handle,
            TemporalHandle::EulerMaruyama(_) | TemporalHandle::PoissonEvents(_)
        ) && solve_policy.is_some()
        {
            return Err(PyTypeError::new_err(
                "stochastic time policies do not consume solve",
            ));
        }
        let model_ref = model.borrow(py);
        let artifact = model_ref.artifact();
        let reference = artifact
            .artifact_reference()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        let digest = reference.artifact().to_string();
        let program = artifact
            .to_program()
            .map_err(|diagnostics| validation_error(py, &diagnostics))?;
        let native = match &temporal_handle {
            TemporalHandle::EulerMaruyama(handle) => {
                eqiora_numerics::CommonTimePlan::resolve_euler_maruyama(
                    artifact,
                    &program,
                    handle.borrow(py).depth,
                )
                .map(|plan| ResolvedCommonPlan::Time(Box::new(plan)))
            }
            TemporalHandle::PoissonEvents(handle) => {
                eqiora_numerics::CommonTimePlan::resolve_poisson_events(
                    artifact,
                    &program,
                    handle.borrow(py).max_events,
                )
                .map(|plan| ResolvedCommonPlan::Time(Box::new(plan)))
            }
            _ => {
                let policy = ode_policy(py, &temporal_handle, &digest)?;
                let backend = time_backend(&policy);
                eqiora_numerics::ResolvedCommonPlan::resolve_ode(
                    artifact,
                    &program,
                    policy,
                    backend,
                    solve_policy.as_ref().map(|(native, _)| *native),
                    &FaerLinearSolver,
                )
            }
        }
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        let resolved_solve = native
            .effective_solver()
            .map(|_| resolved_solve::resolved_solve_from_native(py, &native))
            .transpose()?;
        drop(model_ref);
        return Ok(PyPlan {
            native,
            model,
            mesh: None,
            spatial: None,
            requested_solve: solve_policy.map(|(_, handle)| handle),
            solve: resolved_solve,
            temporal: Some(temporal_handle),
        });
    }

    if mesh.is_none() && spatial.is_none() && temporal_handle.is_none() {
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
    let (solve_native, requested_solve_handle) = solve_policy(py, solve)?;
    let model_ref = model.borrow(py);
    let temporal_native = temporal_handle
        .as_ref()
        .map(|handle| match handle {
            TemporalHandle::BackwardEuler(value) => {
                eqiora_numerics::CommonTemporalPolicy::backward_euler(
                    value.borrow(py).native.value(),
                )
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))
            }
            TemporalHandle::Tsitouras45(_)
            | TemporalHandle::ImplicitMidpoint(_)
            | TemporalHandle::Bdf(_) => {
                let digest = model_ref
                    .artifact()
                    .digest()
                    .map_err(|diagnostic| validation_error(py, &[diagnostic]))?
                    .to_string();
                ode_policy(py, handle, &digest).map(Into::into)
            }
            TemporalHandle::EulerMaruyama(_) | TemporalHandle::PoissonEvents(_) => {
                Err(PyTypeError::new_err(
                    "spatial temporal resolution requires a deterministic time policy",
                ))
            }
        })
        .transpose()?;
    let mesh_ref = mesh.borrow(py);
    let owner = mesh_ref
        .authenticated_common_mesh()
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?
        .ok_or_else(|| {
            PyTypeError::new_err("mesh must be an authenticated caller-owned common Mesh")
        })?;
    let available_time_backend = temporal_native
        .as_ref()
        .and_then(eqiora_numerics::CommonTemporalPolicy::as_ode)
        .map(time_backend);
    let native = eqiora_numerics::ResolvedCommonPlan::resolve(
        model_ref.artifact(),
        owner,
        method_request,
        solve_native,
        scaling.map(eqiora_numerics::CommonScalingRequest::Incompressible),
        temporal_native,
        &FaerLinearSolver,
        available_time_backend,
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
    let solve_handle = resolved_solve::resolved_solve_from_native(py, &native)?;
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

fn ode_policy(
    py: Python<'_>,
    handle: &TemporalHandle,
    model_digest: &str,
) -> PyResult<eqiora_numerics::CommonOdePolicy> {
    let (policy, owned) = match handle {
        TemporalHandle::Tsitouras45(handle) => {
            let value = handle.borrow(py);
            (
                value.data.native.clone(),
                value.data.belongs_to_model(py, model_digest),
            )
        }
        TemporalHandle::ImplicitMidpoint(handle) => {
            let value = handle.borrow(py);
            (
                value.data.native.clone(),
                value.data.belongs_to_model(py, model_digest),
            )
        }
        TemporalHandle::Bdf(handle) => {
            let value = handle.borrow(py);
            (
                value.data.native.clone(),
                value.data.belongs_to_model(py, model_digest),
            )
        }
        _ => {
            return Err(PyTypeError::new_err(
                "continuous deterministic time resolution requires Tsitouras45, ImplicitMidpoint or Bdf",
            ));
        }
    };
    if !owned {
        return Err(PyTypeError::new_err(
            "ODE tolerances must use exact FieldRefs from this Model",
        ));
    }
    Ok(policy)
}

fn time_backend(
    policy: &eqiora_numerics::CommonOdePolicy,
) -> eqiora::time::TimeBackendCapabilities {
    match policy.method() {
        eqiora::time::TimeMethod::ImplicitMidpoint => {
            eqiora::time::ImplicitMidpointTimeBackend::CAPABILITIES
        }
        eqiora::time::TimeMethod::Bdf => eqiora::backends::sundials::IdaTimeBackend::CAPABILITIES,
        _ => eqiora::backends::diffsol::DiffsolTimeBackend::CAPABILITIES,
    }
}

fn solve_policy(
    py: Python<'_>,
    solve: &Bound<'_, PyAny>,
) -> PyResult<(CommonSolvePolicy, RequestedSolveHandle)> {
    Ok(if let Ok(linear) = solve.extract::<Py<PyLinear>>() {
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
    })
}
