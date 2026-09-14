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
        let states = native
            .spatial_states()
            .expect("Python spatial Trajectory requires spatial native States")
            .to_vec();
        let (model_digest, plan_identity, realization_digest, is_fsi) =
            if let Some(native) = plan.transient_native() {
                (
                    native.model_digest().to_owned(),
                    native.identity().to_owned(),
                    Some(native.realization_digest().to_owned()),
                    false,
                )
            } else if let Some(native) = plan.scalar_native() {
                (
                    native.model_digest().to_owned(),
                    native.identity().to_owned(),
                    Some(native.realization_digest().to_owned()),
                    false,
                )
            } else if let Some(native) = plan.fsi_native() {
                (
                    native.model_digest().to_owned(),
                    native.identity().to_owned(),
                    Some(native.realization_digest().to_owned()),
                    true,
                )
            } else {
                return Err(PyRuntimeError::new_err(
                    "common Trajectory requires a transient or FSI Plan",
                ));
            };
        let mesh = plan.mesh_handle(py);
        let mesh_ref = mesh.borrow(py);
        let mut state_lookup = BTreeMap::new();
        let mut projected = Vec::with_capacity(states.len());
        for (step, state) in states {
            let step = u64::try_from(step)
                .map_err(|_| PyOverflowError::new_err("accepted step exceeds Python u64"))?;
            if state_lookup.insert(step, projected.len()).is_some() {
                return Err(PyRuntimeError::new_err(
                    "common Trajectory contains a duplicate output step",
                ));
            }
            let projected_state = if is_fsi {
                let mut value =
                    PyState::from_common_fsi(py, plan, state, step, Some(&request_identity))?;
                value.source_trajectory_identity = Some(trajectory_digest.clone());
                value
            } else {
                PyState::from_common(
                    py,
                    plan,
                    state,
                    step,
                    Some(&request_identity),
                    Some(&trajectory_digest),
                )?
            };
            projected.push(Py::new(py, projected_state)?);
        }
        let geometry_digest = mesh_ref.source_digest_value().to_owned();
        let correspondence_digest = mesh_ref.correspondence_digest_value().to_owned();
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
            coordinates: ReadOnlyMatrix::new(0, 2, Vec::new()),
            cells: ReadOnlyMatrix::new(0, 0, Vec::new()),
            states: projected,
            state_lookup,
            common_mesh: Some(mesh),
            native,
        })
    }
}
