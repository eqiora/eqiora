use super::*;

impl PyState {
    pub(super) fn from_scalar(
        py: Python<'_>,
        plan: &crate::common_plan::PyPlan,
        native: CommonState,
        step: u64,
        source_request_identity: Option<&str>,
        source_trajectory_identity: Option<&str>,
    ) -> PyResult<Self> {
        let scalar = plan.scalar_native().expect("scalar State Plan");
        let mesh = plan.mesh_handle(py);
        let mesh_digest = mesh.borrow(py).exact_mesh_digest().to_owned();
        let snapshot = PyFieldSnapshot::from_common_scalar(py, scalar, &native, &mesh_digest)?;
        let field_lookup = BTreeMap::from([(
            scalar
                .fields()
                .next()
                .expect("scalar storage Field")
                .0
                .ulid()
                .to_string(),
            0,
        )]);
        Ok(Self {
            digest: native.identity().to_owned(),
            model_digest: scalar.model_digest().to_owned(),
            step,
            time_s: native.time_s(),
            fields: vec![Py::new(py, snapshot)?],
            field_lookup,
            model: Some(plan.model_handle(py)),
            mesh: Some(mesh),
            native: Some(native),
            transient_plan: None,
            ode_native: None,
            algebraic_native: None,
            plan_identity: Some(scalar.identity().to_owned()),
            source_request_identity: source_request_identity.map(str::to_owned),
            source_trajectory_identity: source_trajectory_identity.map(str::to_owned),
            source_kind: Some(if source_request_identity.is_some() {
                "result"
            } else {
                "initial"
            }),
        })
    }
}
