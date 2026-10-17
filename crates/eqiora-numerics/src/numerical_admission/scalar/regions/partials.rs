//! Parameter residuals use the primal basis, quadrature, constraints and assembly kernels.
use super::execute::{StaticAssembly, StaticRegionSystem};
use super::*;
use eqiora_ir::ScalarSymbolCoordinate;
use eqiora_solver::LinearOperator;

impl StaticRegionSystem<f64> {
    /// Partial of the reduced residual at fixed free coordinates, and of its lift.
    pub(in crate::numerical_admission) fn parameter_partial(
        &self,
        equations: &ExecutableSpatialEquations<f64>,
        coordinate: &ScalarSymbolCoordinate,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        values: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
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
                let mut residual = prepared.work.residual(
                    &prepared.plan,
                    prepared.plan.target_id(0).expect("reduced target"),
                    values,
                )?;
                // K_p acts with the primal trace; the retained K supplies K*g_p.
                let lift = mapping.lift(&vec![0.; mapping.free_count()], false)?;
                let mut action = vec![0.; self.mapping.full_count()];
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
                        let mut result = vec![0.; values.len()];
                        for index in 0..work.packet_count() {
                            let packet = work.evaluate(index)?;
                            let map = packet
                                .mappings()
                                .iter()
                                .find(|map| map.target() == target)
                                .ok_or_else(|| invalid("TPFA packet omits its residual target"))?
                                .map();
                            let local =
                                crate::region_assembly::nonlinear::evaluate_linear_residual(
                                    packet.local(),
                                    map,
                                    values,
                                )?;
                            crate::region_assembly::nonlinear::scatter_residual(
                                &mut result,
                                map,
                                &local,
                            )?;
                        }
                        Ok(result)
                    },
                )?;
                for (value, trace) in residual.iter_mut().zip(trace_action) {
                    *value -= trace;
                }
                Ok((residual, vec![0.; self.mapping.full_count()]))
            }
        }
    }
}
