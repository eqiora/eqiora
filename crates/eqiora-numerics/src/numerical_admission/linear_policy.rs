//! Admitted linear solver identity and execution checks shared by numerical Plans.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NativeLinearPolicy {
    pub(crate) scalar_domain: eqiora_core::ScalarDomain,
    pub(crate) solver: SolverPlan,
    pub(crate) provider: SolverProvider,
    pub(crate) capabilities: SolverCapabilities,
    pub(crate) execution: ExecutionProvider,
    pub(crate) workers: NonZeroUsize,
    pub(crate) planning_objective: Option<SolverPlanningObjective>,
    pub(crate) planning_profile: Option<eqiora_solver::HostSerialSolverProfile>,
    pub(crate) planning_policy_id: Option<&'static str>,
    pub(crate) selected_candidate_id: Option<&'static str>,
    pub(crate) selected_evidence_case: Option<&'static str>,
    pub(crate) planning_reasons: Vec<(&'static str, &'static str)>,
}

impl NativeLinearPolicy {
    pub(crate) fn exact<S: eqiora_core::Scalar>(
        solver: SolverPlan,
        backend: &dyn LinearSolverBackend<S>,
    ) -> Result<Self, Diagnostic> {
        if solver.relative_tolerance().to_bits() == (-0.0_f64).to_bits()
            || solver.absolute_tolerance().to_bits() == (-0.0_f64).to_bits()
        {
            return Err(invalid(
                "linear policy contains signed-zero tolerance ambiguity",
            ));
        }
        let provider = backend.provider();
        provider.validate()?;
        SERIAL_EXECUTION_PROVIDER.validate()?;
        Ok(Self {
            scalar_domain: S::DOMAIN,
            solver,
            provider,
            capabilities: backend.capabilities(),
            execution: SERIAL_EXECUTION_PROVIDER,
            workers: NonZeroUsize::MIN,
            planning_objective: None,
            planning_profile: None,
            planning_policy_id: None,
            selected_candidate_id: None,
            selected_evidence_case: None,
            planning_reasons: Vec::new(),
        })
    }

    pub(crate) fn with_planning(
        mut self,
        decision: &ResolvedHostSerialSolverPlan<'_>,
    ) -> Result<Self, Diagnostic> {
        if self.solver != decision.solver_plan()
            || self.provider != decision.solver_provider()
            || self.execution != decision.execution_provider()
        {
            return Err(invalid(
                "solver planning audit does not authenticate the effective linear policy",
            ));
        }
        self.planning_objective = Some(decision.objective());
        self.planning_profile = Some(decision.profile());
        self.planning_policy_id = Some(decision.policy_id());
        self.selected_candidate_id = Some(decision.selected_candidate_id());
        self.selected_evidence_case = Some(decision.selected_evidence_case());
        self.planning_reasons = decision.reasons().collect();
        Ok(self)
    }

    pub(crate) fn checked_backend<
        'a,
        S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + Sync + 'static,
    >(
        &self,
        backend: &'a dyn LinearSolverBackend<S>,
        structure: Option<&eqiora_solver::AlgebraicStructure>,
    ) -> Result<ProfileCheckedBackend<'a, S>, Diagnostic> {
        if S::DOMAIN != self.scalar_domain
            || backend.provider() != self.provider
            || backend.capabilities() != self.capabilities
        {
            return Err(invalid(
                "execution backend differs from admitted exact provider or capabilities",
            ));
        }
        if let Some(profile) = &self.planning_profile {
            profile.require_structure(structure)?;
        } else if structure.is_some() {
            return Err(invalid("typed algebraic structure lacks solver admission"));
        }
        Ok(ProfileCheckedBackend {
            backend,
            plan: self.solver,
            profile: self.planning_profile.clone(),
        })
    }

    pub(crate) fn checked_complex_backend<'a>(
        &self,
        backend: &'a dyn LinearSolverBackend,
        structure: Option<&eqiora_solver::AlgebraicStructure>,
    ) -> Result<ProfileCheckedBackend<'a, num_complex::Complex64>, Diagnostic> {
        let complex = backend.complex_backend().ok_or_else(|| {
            invalid("selected provider has no typed complex linear implementation")
        })?;
        if backend.provider() != self.provider {
            return Err(invalid(
                "complex execution differs from admitted exact provider or capabilities",
            ));
        }
        if !self.planning_audit_is_coherent() {
            return Err(invalid(
                "complex execution requires a coherent solver planning audit",
            ));
        }
        self.checked_backend(complex, structure)
    }

    pub(crate) fn planning_audit_is_coherent(&self) -> bool {
        match self.planning_objective {
            None => {
                self.planning_policy_id.is_none()
                    && self.selected_candidate_id.is_none()
                    && self.selected_evidence_case.is_none()
                    && self.planning_reasons.is_empty()
            }
            Some(_) => {
                self.planning_profile.is_some()
                    && self.planning_policy_id.is_some()
                    && self.selected_candidate_id.is_some()
                    && self.selected_evidence_case.is_some()
                    && !self.planning_reasons.is_empty()
            }
        }
    }
}

/// The existing execution boundary rechecks admitted facts before forwarding
/// exactly one numerical attempt to the already selected backend.
#[derive(Debug)]
pub(crate) struct ProfileCheckedBackend<'a, S> {
    backend: &'a dyn LinearSolverBackend<S>,
    plan: SolverPlan,
    profile: Option<eqiora_solver::HostSerialSolverProfile>,
}

impl<S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + Sync + 'static>
    LinearSolverBackend<S> for ProfileCheckedBackend<'_, S>
{
    fn default_execution(&self) -> eqiora_solver::ExecutionReport {
        self.backend.default_execution()
    }
    fn provider(&self) -> SolverProvider {
        self.backend.provider()
    }
    fn capabilities(&self) -> SolverCapabilities {
        self.backend.capabilities()
    }
    fn prepare_linear(
        &self,
        plan: SolverPlan,
    ) -> Result<Option<Box<dyn eqiora_solver::PreparedLinearSolver<S>>>, Diagnostic> {
        if plan != self.plan {
            return Err(invalid("execution changed the admitted exact solver plan"));
        }
        let Some(prepared) = self.backend.prepare_linear(plan)? else {
            return Ok(None);
        };
        Ok(Some(Box::new(ProfileCheckedPreparedLinear {
            prepared,
            profile: self.profile.clone(),
        })))
    }
    fn solve_with_execution(
        &self,
        problem: &eqiora_solver::LinearProblem<'_, S>,
        plan: SolverPlan,
        execution: &dyn eqiora_solver::ReplicatedLinearExecution<S>,
    ) -> Result<eqiora_solver::LinearSolution<S>, Diagnostic> {
        if plan != self.plan {
            return Err(invalid("execution changed the admitted exact solver plan"));
        }
        if let Some(profile) = &self.profile {
            profile.require_problem(problem)?;
        }
        self.backend.solve_with_execution(problem, plan, execution)
    }
}

#[derive(Debug)]
pub(super) struct ProfileCheckedPreparedLinear<S> {
    pub(super) prepared: Box<dyn eqiora_solver::PreparedLinearSolver<S>>,
    pub(super) profile: Option<eqiora_solver::HostSerialSolverProfile>,
}

impl<S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + Sync>
    eqiora_solver::PreparedLinearSolver<S> for ProfileCheckedPreparedLinear<S>
{
    fn solve(
        &mut self,
        structure: &eqiora_solver::PreparedLinearStructureIdentity,
        problem: &eqiora_solver::LinearProblem<'_, S>,
    ) -> Result<eqiora_solver::LinearSolution<S>, Diagnostic> {
        if let Some(profile) = &self.profile {
            profile.require_problem(problem)?;
        }
        self.prepared.solve(structure, problem)
    }
}
