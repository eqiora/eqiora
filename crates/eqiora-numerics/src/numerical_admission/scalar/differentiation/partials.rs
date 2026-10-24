//! Parameter residuals retain primal quadrature and the accepted physical Jacobian.
use super::*;
use crate::region_assembly::mapping::RegionSolveInput;
use eqiora_solver::LinearOperator;

impl<S: DifferentiationScalar> PreparedPoint<'_, S> {
    pub(in crate::numerical_admission) fn affine_parameter_partial<A: DifferentiationScalar>(
        &self,
        system: &StaticRegionSystem<A, S>,
        coordinate: &ScalarSymbolCoordinate,
        values: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        self.mapped_parameter_partial(
            &system.mapping,
            &system.assembly.full,
            &system.assembly.input,
            coordinate,
            values,
        )
    }

    fn partial_equations(
        &self,
        coordinate: &ScalarSymbolCoordinate,
    ) -> Result<ExecutableSpatialEquations<S>, Diagnostic> {
        let mut partial = self.equations.clone();
        for region in &mut partial.regions {
            region.form = region.form.input_partial(
                coordinate,
                crate::form_compiler::spatial::InputPartialScope::Full,
            )?;
        }
        Ok(partial)
    }

    pub(super) fn mapped_parameter_partial<A: DifferentiationScalar>(
        &self,
        mapping: &RegionDofMap<S>,
        full: &eqiora_assembly::LinearSystem<A>,
        input: &RegionSolveInput<S>,
        coordinate: &ScalarSymbolCoordinate,
        values: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        let (partial_mapping, mut differentiated) =
            if let Some([previous, current]) = self.time_interval {
                self.plan.scalar_assembly_with_history(
                    &self.equations,
                    current,
                    Some(previous),
                    input.previous.as_ref(),
                    &[(
                        coordinate,
                        crate::form_compiler::spatial::InputPartialScope::Full,
                    )],
                )?
            } else {
                let NativeMeshResources::Cartesian { mesh, .. } = self.plan.admission.resources()
                else {
                    return Err(invalid(
                        "spatial Parameter partials require Cartesian resources",
                    ));
                };
                let partial = self.partial_equations(coordinate)?;
                let forms = input
                    .forms
                    .iter()
                    .map(|(form, _)| form.input_partial(coordinate))
                    .collect::<Result<Vec<_>, _>>()?;
                partial.cartesian_assembly_bound(
                    mesh.mesh(),
                    &self.plan.admission.discretizations,
                    forms,
                    |_, _, value| Ok(Some(value)),
                )?
            };
        if !partial_mapping.has_same_coordinate_layout(mapping) {
            return Err(invalid(
                "Parameter partial changed the exact Field constraint layout",
            ));
        }
        // Derivatives use the primal integration rule and fixed primal trace.
        differentiated.retain_primal_integration(input)?;
        differentiated.previous = input.previous.clone();
        let mut residual = match self.plan.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => {
                S::parameter_residual(mapping, mesh.mesh(), differentiated, values)?
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                S::parameter_residual(mapping, mesh.mesh(), differentiated, values)?
            }
            _ => {
                return Err(invalid(
                    "Parameter partials require the exact admitted affine Mesh",
                ));
            }
        };
        let lift = partial_mapping.lift(&vec![S::zero(); mapping.free_count()], false)?;
        let layout = RegionRealCoordinates::new(mapping)?;
        let lift = layout.encode_full(&lift)?;
        // The full accepted Jacobian supplies J*g_p for affine and nonlinear laws.
        let numerical_lift = A::from_coordinates(&lift)?;
        let mut action = vec![A::zero(); full.matrix().rows()];
        full.matrix().apply(&numerical_lift, &mut action)?;
        let action = layout.restrict_action(&coordinates(action).collect::<Vec<_>>())?;
        for (value, action) in residual.iter_mut().zip(action) {
            *value += action;
        }
        Ok((residual, lift))
    }
}
