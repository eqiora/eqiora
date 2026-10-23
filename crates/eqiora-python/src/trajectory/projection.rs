//! Accepted common trajectory projection into Python-owned States.
use super::*;

impl PyTrajectory {
    pub(crate) fn state_handles(&self, py: Python<'_>) -> Vec<Py<PyState>> {
        self.states
            .iter()
            .map(|state| state.clone_ref(py))
            .collect()
    }

    pub(crate) fn from_common(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        native: eqiora_numerics::CommonTrajectory,
    ) -> PyResult<Self> {
        let request_identity = native.request_identity().to_owned();
        let trajectory_digest = native.identity().to_owned();
        let request = native.spatial_request().ok_or_else(|| {
            PyRuntimeError::new_err("Python spatial Trajectory requires spatial native States")
        })?;
        let output_steps = if request.plan().backward_euler().is_some() {
            Some(
                request
                    .output_steps()
                    .map_err(|d| crate::error::validation_error(py, &[d]))?,
            )
        } else {
            None
        };
        let (model_digest, plan_identity, realization_digest) =
            if let Some(native) = plan.transient_native() {
                (
                    native.model_digest().to_owned(),
                    native.identity().to_owned(),
                    Some(native.realization_digest().to_owned()),
                )
            } else if let Some(native) = plan.linear_native() {
                (
                    native.model_digest().to_owned(),
                    native.identity().to_owned(),
                    Some(native.realization_digest().to_owned()),
                )
            } else {
                return Err(PyRuntimeError::new_err(
                    "common Trajectory requires a spatial transient Plan",
                ));
            };
        let mesh = plan.mesh_handle(py);
        let mesh_ref = mesh.borrow(py);
        let mut state_lookup = BTreeMap::new();
        let mut projected = Vec::with_capacity(request.output_times_s().len());
        for index in 0..request.output_times_s().len() {
            let state = native
                .spatial_state(index)
                .ok_or_else(|| {
                    PyRuntimeError::new_err("common Trajectory omitted a requested spatial State")
                })?
                .clone();
            let step = output_steps
                .as_ref()
                .map(|steps| {
                    u64::try_from(steps[index])
                        .map_err(|_| PyOverflowError::new_err("accepted step exceeds Python u64"))
                })
                .transpose()?;
            if let Some(step) = step
                && state_lookup.insert(step, projected.len()).is_some()
            {
                return Err(PyRuntimeError::new_err(
                    "common Trajectory contains a duplicate output step",
                ));
            }
            let projected_state = PyState::from_common(
                py,
                plan,
                state,
                step,
                Some(&request_identity),
                Some(&trajectory_digest),
            )?;
            projected.push(Py::new(py, projected_state)?);
        }
        let geometry_digest = mesh_ref.source_digest_value().to_owned();
        let correspondence_digest = mesh_ref
            .correspondence_digest_value()
            .ok_or_else(|| {
                PyRuntimeError::new_err("spatial trajectory requires Geometry correspondence")
            })?
            .to_owned();
        let mesh_digest = mesh_ref.exact_mesh_digest().to_owned();
        drop(mesh_ref);
        Ok(Self {
            model_digest,
            geometry_digest,
            correspondence_digest,
            mesh_digest,
            realization_digest,
            plan_identity: Some(plan_identity),
            run_digest: Some(request_identity.clone()),
            request_identity: Some(request_identity),
            trajectory_digest,
            states: projected,
            state_lookup,
            mesh,
            native,
        })
    }
}

impl PyState {
    pub(super) fn from_common_algebraic(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        state: eqiora_numerics::CommonAlgebraicState,
    ) -> Self {
        Self {
            digest: state.identity().to_owned(),
            model_digest: plan.native().model_digest().to_owned(),
            step: None,
            time_s: None,
            fields: Vec::new(),
            field_lookup: BTreeMap::new(),
            model: Some(plan.model_handle(py)),
            mesh: None,
            native: None,
            transient_plan: None,
            time_native: None,
            plan_identity: Some(state.plan_identity().to_owned()),
            algebraic_native: Some(state),
            source_request_identity: None,
            source_trajectory_identity: None,
            source_kind: Some("initial"),
        }
    }
}
