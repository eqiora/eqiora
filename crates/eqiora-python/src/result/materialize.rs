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
        ResolvedCommonPlan::TransientFlow(_) | ResolvedCommonPlan::Spatial(_)
    ) {
        return Err(PyRuntimeError::new_err(
            "common transient output crossed a different Plan",
        ));
    }
    let trajectory = Py::new(py, PyTrajectory::from_common(py, &plan, native_trajectory)?)?;
    Ok(PyRunResult {
        native: native_result,
        identity,
        elapsed_seconds,
        payload: ResultPayload::Trajectory(CommonTrajectoryResultPayload { trajectory }),
        profile: None,
    })
}

fn materialize_common_time_trajectory(
    py: Python<'_>,
    plan: PyRef<'_, PyPlan>,
    identity: RunIdentity,
    elapsed_seconds: f64,
    trajectory: eqiora_numerics::CommonTrajectory,
    native_result: eqiora_numerics::CommonResult,
) -> PyResult<PyRunResult> {
    let ResolvedCommonPlan::Time(native) = plan.native() else {
        return Err(PyRuntimeError::new_err(
            "common time output crossed a different Plan",
        ));
    };
    let states = trajectory
        .time_states()
        .expect("time materialization requires a time Trajectory")
        .to_vec();
    let times = states
        .iter()
        .map(eqiora_numerics::CommonTimeState::time_s)
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
        payload: ResultPayload::Time(CommonTimeResultPayload {
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
    if RunIdentity::from_common_result(&result).as_ref() != Some(&identity) {
        return Err(PyRuntimeError::new_err(
            "common Result crossed a different Run request occurrence",
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
        let elapsed_seconds = result.elapsed_seconds();
        let trajectory = trajectory.clone();
        if trajectory.time_states().is_some() {
            return materialize_common_time_trajectory(
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
    let mesh = (result.field_count() > 0).then(|| plan.mesh_handle(py));
    let mut outputs = Vec::with_capacity(result.field_count());
    let mut lookup = BTreeMap::new();
    for field_index in 0..result.field_count() {
        let (field_id, value_type, typed_space) = result
            .field(field_index)
            .ok_or_else(|| PyRuntimeError::new_err("common Result omitted Field metadata"))?;
        let field_id = field_id.to_owned();
        let value_type = value_type.clone();
        let complex = match value_type.scalar_domain() {
            eqiora::ScalarDomain::Real => false,
            eqiora::ScalarDomain::Complex => true,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "common Result has an unsupported Field scalar domain",
                ));
            }
        };
        let value_shape = value_type
            .shape()
            .extents()
            .iter()
            .map(|extent| extent.get() as usize)
            .collect::<Vec<_>>();
        use eqiora::realization::SpaceFamily;
        let space = match typed_space.family() {
            SpaceFamily::ContinuousLagrange { order } if order == std::num::NonZeroU16::MIN => {
                "continuous-lagrange-p1"
            }
            SpaceFamily::CellConstant => "cell-constant",
            SpaceFamily::SimplexP1Bubble => "simplex-p1-bubble",
            SpaceFamily::TetrahedralEdge => "tetrahedral-edge",
            SpaceFamily::TetrahedralFace => "tetrahedral-face",
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
        let coefficient_dimension = typed_space
            .coefficient_dimension(value_type.dimension())
            .ok_or_else(|| PyRuntimeError::new_err("common Result coefficient units overflow"))?;
        let blocks = (0..result.field_block_count(field_index))
            .map(|block_index| {
                let (association, values, logical_shape) = result
                    .field_block(field_index, block_index)
                    .ok_or_else(|| PyRuntimeError::new_err("common Result omitted Field block"))?;
                let coefficient_count = field_output::coefficient_count(
                    typed_space,
                    &value_shape,
                    complex,
                    values.len(),
                    logical_shape,
                )?;
                Ok(FieldOutputBlock::new(
                    association,
                    if complex {
                        PyArrayBuffer::from_owned_complex_result(
                            py,
                            values
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .map(|pair| numpy::Complex64::new(pair[0], pair[1]))
                                .collect(),
                        )?
                    } else {
                        PyArrayBuffer::from_owned_result(py, values.to_vec())?
                    },
                    coefficient_count,
                    logical_shape.to_vec(),
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let output = Py::new(
            py,
            PyFieldOutput::new(
                field,
                Some(
                    mesh.as_ref()
                        .expect("spatial field owns mesh")
                        .clone_ref(py),
                ),
                value_type,
                coefficient_dimension,
                space,
                blocks,
            ),
        )?;
        lookup.insert(field_id, outputs.len());
        outputs.push(output);
    }
    if result.plan().as_algebraic().is_some() {
        for (id, value) in result
            .finite_field_values()
            .map_err(|error| crate::error::validation_error(py, &[error]))?
        {
            let field_id = id.ulid().to_string();
            let field = Py::new(
                py,
                PyModelFieldRef::from_exact(identity.model_digest().to_owned(), field_id.clone()),
            )?;
            let components = value.components().ok_or_else(|| {
                PyRuntimeError::new_err("finite Field has no floating coefficients")
            })?;
            let buffer = if value.value_type().scalar_domain() == eqiora::ScalarDomain::Complex {
                PyArrayBuffer::from_owned_complex_result(
                    py,
                    components
                        .map(|(real, imaginary)| numpy::Complex64::new(real, imaginary))
                        .collect(),
                )?
            } else {
                PyArrayBuffer::from_owned_result(py, components.map(|(real, _)| real).collect())?
            };
            let shape = value
                .value_type()
                .shape()
                .extents()
                .iter()
                .map(|extent| extent.get() as usize)
                .collect();
            let output = Py::new(
                py,
                PyFieldOutput::new(
                    field,
                    None,
                    value.value_type().clone(),
                    value.value_type().dimension(),
                    "finite",
                    vec![FieldOutputBlock::new("global", buffer, 1, shape)],
                ),
            )?;
            lookup.insert(field_id, outputs.len());
            outputs.push(output);
        }
    }
    let solve = if result.nonlinear_iterations(None).is_some() {
        Py::new(
            py,
            nonlinear::PyNonlinearSolveSummary::from_result(&result, None)?,
        )?
        .into_any()
    } else {
        let solve =
            PyLinearSolveSummary::from_common_result(&result, None, None).ok_or_else(|| {
                PyRuntimeError::new_err("static common Result omitted solve evidence")
            })?;
        Py::new(py, solve)?.into_any()
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
