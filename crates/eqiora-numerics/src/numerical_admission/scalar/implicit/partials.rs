//! Input actions at independent State/rate points retain analytic trace lifts.
use super::*;
use eqiora_solver::LinearOperator;

impl<S: Coefficient + Send + Sync> SpatialImplicitSystem<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    /// Hold free State/rate coordinates fixed while varying one declared input.
    /// The original residual supplies both State and rate boundary-chain actions.
    pub(in crate::numerical_admission) fn parameter_partial(
        &self,
        time: f64,
        state: &[f64],
        rate: &[f64],
        coordinate: &eqiora_ir::ScalarSymbolCoordinate,
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        self.validate(state)?;
        self.validate(rate)?;
        let (mapping, input) = self.plan.scalar_assembly_at(self.equations, time)?;
        let partial = self.plan.scalar_assembly_with_history(
            self.equations,
            time,
            None,
            None,
            &[(coordinate, InputPartialScope::Full)],
        )?;
        let (_, rates) = self.prepare(time, rate)?;
        // Reuse the original independent-rate residual and integration owner.
        // Its prescribed-data lift applies to any explicit input coordinate.
        let (lift, mut residual) = match self.plan.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => {
                mapping.initial_rate_data(mesh.mesh(), &input, state, &rates, partial)?
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                mapping.initial_rate_data(mesh.mesh(), &input, state, &rates, partial)?
            }
            _ => {
                return Err(invalid(
                    "implicit input partial requires the admitted affine Mesh",
                ));
            }
        };
        let mut differentiated = self.equations.clone();
        for region in &mut differentiated.regions {
            region.form = region
                .form
                .input_partial(coordinate, InputPartialScope::EssentialTraces)?;
        }
        let (rate_mapping, _) = self.plan.scalar_time_partial(
            &differentiated,
            time,
            &[InputPartialScope::EssentialTraces],
        )?;
        let rate_lift = mapping.prescribed_rates(&rate_mapping)?;
        let layout = RegionRealCoordinates::new(&mapping)?;
        for (weights, direction) in [
            (
                FieldPullback::Uniform {
                    current: Some(1.),
                    rate: None,
                },
                &lift,
            ),
            (
                FieldPullback::Uniform {
                    current: None,
                    rate: Some(1.),
                },
                &rate_lift,
            ),
        ] {
            if direction.iter().all(|value| *value == 0.) {
                continue;
            }
            let assembly = self.linearize(time, state, rate, weights)?;
            let matrix = assembly.systems[1].matrix();
            let mut full = vec![0.; matrix.rows()];
            matrix.apply(direction, &mut full)?;
            for (value, action) in residual.iter_mut().zip(layout.restrict_action(&full)?) {
                *value += action;
            }
        }
        Ok((residual, lift))
    }
}
