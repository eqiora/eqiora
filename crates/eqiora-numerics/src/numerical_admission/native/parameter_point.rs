//! Reauthenticate a selected policy against a Parameter point's actual system.
use super::*;

impl NativeLinearPolicy {
    /// Parameter binding can change complex linearity without changing caller
    /// intent. Derive the numerical domain from the typed canonical system and
    /// check every retained solver/profile fact before rebinding it. This neither
    /// reranks automatic choices nor changes the provider, algorithm or controls.
    pub(in crate::numerical_admission) fn at_parameter_problem<S>(
        &self,
        system: &eqiora_solver::CanonicalCsrSystemView<S>,
        structure: Option<&eqiora_solver::AlgebraicStructure>,
    ) -> Result<Self, Diagnostic>
    where
        S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + Sync,
    {
        if !self.planning_audit_is_coherent() {
            return Err(invalid(
                "Parameter point has an incoherent solver planning audit",
            ));
        }
        self.capabilities.require_problem(
            self.solver,
            S::DOMAIN,
            ScalarType::F64,
            system.properties(),
        )?;
        if let Some(profile) = &self.planning_profile {
            profile.require_plan(self.solver)?;
            profile.require_structure(structure)?;
            profile.require_problem(&system.linear_problem()?)?;
            profile.require_problem_resources(
                self.solver,
                self.provider,
                &system.linear_problem()?,
            )?;
        } else if structure.is_some() {
            return Err(invalid("typed algebraic structure lacks solver admission"));
        }
        let mut point = self.clone();
        point.scalar_domain = S::DOMAIN;
        Ok(point)
    }
}
