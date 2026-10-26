//! Affine execution retains its genuine scalar-domain solve and original system.
use super::*;

impl<S: DifferentiationScalar> PreparedPoint<'_, S> {
    pub(super) fn solve_affine(
        self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        let system = S::finalize(self.plan, &self.equations)?;
        match &system {
            StaticSpatialSystem::Real(system) => self.solve_system(system, backend),
            StaticSpatialSystem::Complex(system) => self.solve_system(system, backend),
        }
    }

    fn solve_system<A: DifferentiationScalar>(
        &self,
        system: &StaticRegionSystem<A, S>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        let structure = self.equations.algebraic_structure(None)?;
        let linear = self
            .plan
            .admission
            .linear
            .at_parameter_problem(system.core.canonical_csr_system_view(), Some(&structure))?;
        let (values, primal) =
            A::solve(self.plan, &linear, &self.equations, &system.core, backend)?;
        let target = match &primal {
            Primal::Linear(receipt) => receipt.report().residual_target(),
            Primal::Affine(report) => report.residual_target(),
            Primal::Nonlinear { .. } | Primal::Midpoint(_) => unreachable!("static linear solve"),
        };
        let linearization = self.linearize(system, values, target)?;
        if let Primal::Linear(receipt) = &primal
            && receipt.operator()
                != linearization
                    .relation()
                    .state_jacobian()
                    .agreement_fingerprint()
        {
            return Err(invalid(
                "common Plan solve receipt differs from its differentiated state system",
            ));
        }
        Ok(CommonDifferentiationPoint::from_primal(
            linearization,
            primal,
        ))
    }

    pub(super) fn accept_affine(
        self,
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let system = S::finalize(self.plan, &self.equations)?;
        match &system {
            StaticSpatialSystem::Real(system) => self.accept_system(system, candidate),
            StaticSpatialSystem::Complex(system) => self.accept_system(system, candidate),
        }
    }

    fn accept_system<A: DifferentiationScalar>(
        &self,
        system: &StaticRegionSystem<A, S>,
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let structure = self.equations.algebraic_structure(None)?;
        self.plan
            .admission
            .linear
            .at_parameter_problem(system.core.canonical_csr_system_view(), Some(&structure))?;
        let values = A::from_coordinates(candidate)?;
        let (_, target) = system.core.validate_coefficients(&values)?;
        self.linearize(system, values, target)
    }

    fn linearize<A: DifferentiationScalar>(
        &self,
        system: &StaticRegionSystem<A, S>,
        values: Vec<A>,
        target: f64,
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let state_coordinates = coordinates(values).collect::<Vec<_>>();
        let (residual, output) = self.output_partials(
            &system.mapping,
            &state_coordinates,
            state_coordinates.len(),
            |coordinate| self.affine_parameter_partial(system, coordinate, &state_coordinates),
        )?;
        let relation = AssembledLinearizedRelation::from_shared_canonical(
            A::real_system(&system.core)?,
            state_coordinates,
            self.design.clone(),
            self.values.clone(),
            residual,
        )?;
        Ok(CommonDifferentiationLinearization::from_spatial(
            relation,
            A::DOMAIN,
            output,
            eqiora_differentiation::PrimalAcceptance::Residual(target),
            None,
        ))
    }
}
