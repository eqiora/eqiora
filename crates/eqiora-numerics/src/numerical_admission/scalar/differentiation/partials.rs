//! Parameter residuals retain primal quadrature and the accepted physical Jacobian.
use super::super::regions::StaticAssembly;
use super::*;
use crate::region_assembly::{mapping::RegionSolveInput, residual::evaluate_affine_work};
use eqiora_solver::LinearOperator;

impl<S: DifferentiationScalar> PreparedPoint<'_, S> {
    pub(in crate::numerical_admission) fn affine_parameter_partial<A: DifferentiationScalar>(
        &self,
        system: &StaticRegionSystem<A, S>,
        coordinate: &ScalarSymbolCoordinate,
        values: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        match &system.assembly {
            StaticAssembly::FiniteElement(retained) => {
                let (full, input) = retained.as_ref();
                self.finite_element_partial(&system.mapping, full, input, coordinate, values)
            }
            StaticAssembly::FiniteVolume => {
                let partial = self.partial_equations(coordinate)?;
                let (_, residual) = self.equations.with_tpfa_work::<S::Projection, _>(
                    self.plan.admission.resources(),
                    &system.mapping,
                    LinearOperatorProperties::General,
                    Some(&partial),
                    |plan, work| {
                        evaluate_affine_work(
                            plan,
                            plan.target_id(0).expect("TPFA target"),
                            work,
                            values,
                        )
                    },
                )?;
                Ok((residual, vec![0.; system.mapping.full_count() * S::WIDTH]))
            }
        }
    }

    fn partial_equations(
        &self,
        coordinate: &ScalarSymbolCoordinate,
    ) -> Result<ExecutableSpatialEquations<S>, Diagnostic> {
        let mut partial = self.equations.clone();
        for region in &mut partial.regions {
            region.form = region.form.parameter_partial(coordinate)?;
        }
        Ok(partial)
    }

    pub(super) fn finite_element_partial<A: DifferentiationScalar>(
        &self,
        mapping: &RegionDofMap<S>,
        full: &eqiora_assembly::LinearSystem<A>,
        input: &RegionSolveInput<S>,
        coordinate: &ScalarSymbolCoordinate,
        values: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        let NativeMeshResources::Cartesian { mesh, .. } = self.plan.admission.resources() else {
            return Err(invalid(
                "spatial Parameter partials require Cartesian resources",
            ));
        };
        let partial = self.partial_equations(coordinate)?;
        let forms = input
            .forms
            .iter()
            .map(|(form, _)| form.parameter_partial(coordinate))
            .collect::<Result<Vec<_>, _>>()?;
        let (partial_mapping, mut differentiated) = partial.cartesian_assembly_bound(
            mesh.mesh(),
            &self.plan.admission.discretizations,
            forms,
            |_, _, value| Ok(Some(value)),
        )?;
        if partial_mapping.full_count() != mapping.full_count()
            || partial_mapping.free_count() != mapping.free_count()
            || !partial_mapping.keys().eq(mapping.keys())
            || partial_mapping.keys().any(|key| {
                partial_mapping.global_dof(key) != mapping.global_dof(key)
                    || partial_mapping.free_dof(key) != mapping.free_dof(key)
            })
        {
            return Err(invalid(
                "Parameter partial changed the exact Field constraint layout",
            ));
        }
        // Derivatives use the primal integration rule and fixed primal trace.
        for ((_, rule), (_, primal)) in differentiated.forms.iter_mut().zip(&input.forms) {
            *rule = primal.clone();
        }
        let mut residual = S::parameter_residual(mapping, mesh.mesh(), differentiated, values)?;
        let lift = partial_mapping.lift(&vec![S::zero(); mapping.free_count()], false)?;
        let lift = coordinates(lift).collect::<Vec<_>>();
        // The full accepted Jacobian supplies J*g_p for affine and nonlinear laws.
        let numerical_lift = A::from_coordinates(&lift)?;
        let mut action = vec![A::zero(); full.matrix().rows()];
        full.matrix().apply(&numerical_lift, &mut action)?;
        let physical_action = S::from_coordinates(&coordinates(action).collect::<Vec<_>>())?;
        for (value, action) in residual
            .iter_mut()
            .zip(coordinates(mapping.restrict_action(&physical_action)?))
        {
            *value += action;
        }
        Ok((residual, lift))
    }
}
