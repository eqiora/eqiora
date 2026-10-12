use super::*;

impl CommonLinearPlan {
    pub(crate) fn reaccept_spatial_step(
        &self,
        previous: &CommonState,
        current: &CommonState,
    ) -> Result<(f64, f64, usize, usize), Diagnostic> {
        if current.state_space_identity() != self.identity() {
            return Err(invalid(
                "step replay requires its exact current Plan checkpoint",
            ));
        }
        let (mapping, input) = self.scalar_step_assembly(previous, current.time_s())?;
        let values = current
            .linear_values()
            .ok_or_else(|| invalid("step replay requires physical linear coefficients"))?;
        let fields = self.history_fields(&mapping, values)?;
        let solver = self.admission.linear.solver;
        let workers = self.admission.linear.workers;
        match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => {
                mapping.validate_step_coefficients(mesh.mesh(), &input, &fields, solver, workers)
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. } => {
                let current_mesh = input
                    .geometry_action
                    .as_ref()
                    .map(|action| action.current_mesh());
                mapping.validate_step_coefficients(
                    current_mesh.unwrap_or_else(|| mesh.mesh()),
                    &input,
                    &fields,
                    solver,
                    workers,
                )
            }
            _ => Err(invalid("step replay requires its exact affine mesh")),
        }
    }
}
