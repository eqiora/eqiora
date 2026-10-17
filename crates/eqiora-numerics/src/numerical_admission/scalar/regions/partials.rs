//! Parameter residuals use the primal basis, quadrature, constraints and assembly kernels.
use super::execute::{StaticAssembly, StaticRegionSystem};
use super::*;
use crate::region_assembly::residual::evaluate_affine_work;
use eqiora_ir::ScalarSymbolCoordinate;
use eqiora_solver::LinearOperator;

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send> StaticRegionSystem<S> {
    /// Partial of the reduced residual at fixed free coordinates, and of its lift.
    pub(in crate::numerical_admission) fn parameter_partial(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        coordinate: &ScalarSymbolCoordinate,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        values: &[S],
    ) -> Result<(Vec<S>, Vec<S>), Diagnostic> {
        let mut partial = equations.clone();
        for region in &mut partial.regions {
            region.form = region.form.parameter_partial(coordinate)?;
        }
        match &self.assembly {
            StaticAssembly::FiniteElement { full, input } => {
                let NativeMeshResources::Cartesian { mesh, .. } = resources else {
                    return Err(invalid(
                        "spatial Parameter partials require Cartesian resources",
                    ));
                };
                let forms = input
                    .forms
                    .iter()
                    .map(|(form, _)| form.parameter_partial(coordinate))
                    .collect::<Result<Vec<_>, _>>()?;
                let (mapping, mut differentiated) = partial.cartesian_assembly_bound(
                    mesh.mesh(),
                    selected,
                    forms,
                    |_, _, value| Ok(Some(value)),
                )?;
                if mapping.full_count() != self.mapping.full_count()
                    || mapping.free_count() != self.mapping.free_count()
                    || !mapping.keys().eq(self.mapping.keys())
                    || mapping.keys().any(|key| {
                        mapping.global_dof(key) != self.mapping.global_dof(key)
                            || mapping.free_dof(key) != self.mapping.free_dof(key)
                    })
                {
                    return Err(invalid(
                        "Parameter partial changed the exact Field constraint layout",
                    ));
                }
                // A derivative's degree never changes the primal integration rule.
                for ((_, quadrature), (_, primal)) in
                    differentiated.forms.iter_mut().zip(&input.forms)
                {
                    *quadrature = primal.clone();
                }
                let prepared = self.mapping.prepare_assembly(
                    eqiora_assembly::AssemblyPacketSetIdentityV1::Unbound,
                    mesh.mesh(),
                    differentiated.forms,
                    differentiated.natural,
                    None,
                    None,
                )?;
                let mut residual = evaluate_affine_work(
                    &prepared.plan,
                    prepared.plan.target_id(0).expect("reduced target"),
                    &prepared.work,
                    values,
                )?;
                // K_p acts with the primal trace; the retained K supplies K*g_p.
                let lift = mapping.lift(&vec![S::zero(); mapping.free_count()], false)?;
                let mut action = vec![S::zero(); self.mapping.full_count()];
                full.matrix().apply(&lift, &mut action)?;
                for (value, action) in residual
                    .iter_mut()
                    .zip(self.mapping.restrict_action(&action)?)
                {
                    *value += action;
                }
                Ok((residual, lift))
            }
            StaticAssembly::FiniteVolume(primal) => {
                let trace_action = primal.essential_boundary_action(
                    |component, axis, side, point| match partial
                        .tpfa_boundary(component, axis, side, point)?
                    {
                        CartesianBoundaryValue::Essential(value) => Ok(value),
                        _ => Err(invalid("Parameter partial changed the TPFA boundary kind")),
                    },
                )?;
                for (region, primal) in partial.regions.iter_mut().zip(&equations.regions) {
                    region.form.hold_essential_data(&primal.form)?;
                }
                let (_, mut residual) = partial.with_tpfa_work(
                    resources,
                    LinearOperatorProperties::General,
                    |plan, work| {
                        let target = plan.target_id(0).expect("TPFA target");
                        evaluate_affine_work(plan, target, work, values)
                    },
                )?;
                for (value, trace) in residual.iter_mut().zip(trace_action) {
                    *value -= trace;
                }
                Ok((residual, vec![S::zero(); self.mapping.full_count()]))
            }
        }
    }
}
