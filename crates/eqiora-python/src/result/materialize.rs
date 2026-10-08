//! Materialize exact common Results into their Python output families.
use super::*;

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
    for (column, (coordinate, dimension)) in native
        .state_coordinates()
        .zip(native.state_dimensions())
        .enumerate()
    {
        let (field, order) = (coordinate.field(), coordinate.derivative_order());
        let id = field.to_string();
        let values = states.iter().map(|state| state.values()[column]).collect();
        let series = Py::new(
            py,
            PySeries {
                derivative_order: order,
                component: coordinate.component(),
                imaginary: coordinate.is_imaginary(),
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
        lookup.insert(
            (id, order, coordinate.component(), coordinate.is_imaginary()),
            fields.len(),
        );
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
    if result.plan().as_eigen().is_some() {
        return Ok(PyRunResult {
            elapsed_seconds: result.elapsed_seconds(),
            native: result,
            identity,
            payload: ResultPayload::Eigen,
            profile: None,
        });
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
        if result.field_scalar_domain(field_index) != Some(eqiora::ScalarDomain::Real) {
            return Err(PyRuntimeError::new_err(
                "complex spatial Field materialization is not yet admitted by the Python coefficient buffer",
            ));
        }
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
    let solve = if result.nonlinear_iterations().is_some() {
        Py::new(
            py,
            nonlinear::PyNonlinearSolveSummary::from_result(&result)?,
        )?
        .into_any()
    } else {
        let solve = PyLinearSolveSummary::from_common_result(&result, None).ok_or_else(|| {
            PyRuntimeError::new_err("static common Result omitted solve evidence")
        })?;
        Py::new(py, solve)?.into_any()
    };
    let evidence = match result.family_name() {
        "algebraic" | "scalar" => None,
        "elasticity" => Some(StaticScientificEvidence::LinearElasticity(Py::new(
            py,
            PyLinearElasticityEvidence::from_result(py, identity.plan_key(), &result)?,
        )?)),
        "steady-stokes" => Some(StaticScientificEvidence::SteadyStokes(Py::new(
            py,
            PySteadyStokesEvidence::from_result(py, identity.plan_key(), &result)?,
        )?)),
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
