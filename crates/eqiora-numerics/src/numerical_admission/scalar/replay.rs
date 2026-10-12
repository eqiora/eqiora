use super::*;

impl CommonSpatialPlan {
    pub(crate) fn reaccept_spatial_step(
        &self,
        previous: &CommonState,
        current: &CommonState,
        nonlinear_iterations: Option<usize>,
    ) -> Result<(f64, f64, usize, usize, Option<f64>), Diagnostic> {
        if current.state_space_identity() != self.identity() {
            return Err(invalid(
                "step replay requires its exact current Plan checkpoint",
            ));
        }
        let (mapping, input) = self.scalar_step_assembly(previous, current.time_s())?;
        let values = current
            .spatial_values()
            .ok_or_else(|| invalid("step replay requires physical linear coefficients"))?;
        let fields = self.history_fields(&mapping, values)?;
        let solver = self.admission.linear.solver;
        let workers = self.admission.linear.workers;
        if self.admission.nonlinear.is_some() != nonlinear_iterations.is_some() {
            return Err(invalid(
                "spatial iteration history differs from its Plan solve strategy",
            ));
        }
        let validate = |mesh: &eqiora_meshing::SimplicialMesh| match self.admission.nonlinear {
            Some(nonlinear) => mapping.validate_nonlinear_step_coefficients(
                mesh,
                &input,
                &fields,
                nonlinear,
                nonlinear_iterations.expect("validated history"),
            ),
            None => mapping
                .validate_step_coefficients(mesh, &input, &fields, solver, workers)
                .map(|(n, t, p, a)| (n, t, p, a, None)),
        };
        match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => match self.admission.nonlinear {
                Some(nonlinear) => mapping.validate_nonlinear_step_coefficients(
                    mesh.mesh(),
                    &input,
                    &fields,
                    nonlinear,
                    nonlinear_iterations.expect("validated history"),
                ),
                None => mapping
                    .validate_step_coefficients(mesh.mesh(), &input, &fields, solver, workers)
                    .map(|(n, t, p, a)| (n, t, p, a, None)),
            },
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. } => {
                let current_mesh = input
                    .geometry_action
                    .as_ref()
                    .map(|action| action.current_mesh());
                validate(current_mesh.unwrap_or_else(|| mesh.mesh()))
            }
            _ => Err(invalid("step replay requires its exact affine mesh")),
        }
    }
}
