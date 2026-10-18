//! Parameter residuals retain primal basis, quadrature and the physical boundary lift.
use super::super::regions::StaticAssembly;
use super::*;
use crate::region_assembly::residual::evaluate_affine_work;
use eqiora_solver::LinearOperator;

impl<A: DifferentiationScalar, S: DifferentiationScalar> StaticRegionSystem<A, S> {
    pub(in crate::numerical_admission) fn parameter_partial(
        &self,
        equations: &ExecutableSpatialEquations<S>,
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
            StaticAssembly::FiniteElement(retained) => {
                let (full, input) = retained.as_ref();
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
                // Real projection also admits a conjugate-dependent partial at a
                // complex-linear primal point (for example b=0 in a*u+b*conj(u)).
                let mut residual =
                    S::parameter_residual(&self.mapping, mesh.mesh(), differentiated, values)?;
                // K_p acts with the primal trace; the retained K supplies K*g_p.
                let lift = mapping.lift(&vec![S::zero(); mapping.free_count()], false)?;
                let lift = coordinates(lift).collect::<Vec<_>>();
                let numerical_lift = A::from_coordinates(&lift)?;
                let mut action = vec![A::zero(); full.matrix().rows()];
                full.matrix().apply(&numerical_lift, &mut action)?;
                let physical_action =
                    S::from_coordinates(&coordinates(action).collect::<Vec<_>>())?;
                for (value, action) in residual
                    .iter_mut()
                    .zip(coordinates(self.mapping.restrict_action(&physical_action)?))
                {
                    *value += action;
                }
                Ok((residual, lift))
            }
            StaticAssembly::FiniteVolume => {
                let (_, residual) = equations.with_tpfa_work::<S::Projection, _>(
                    resources,
                    &self.mapping,
                    LinearOperatorProperties::General,
                    Some(&partial),
                    |plan, work| {
                        let target = plan.target_id(0).expect("TPFA target");
                        evaluate_affine_work(plan, target, work, values)
                    },
                )?;
                Ok((residual, vec![0.; self.mapping.full_count() * S::WIDTH]))
            }
        }
    }
}
