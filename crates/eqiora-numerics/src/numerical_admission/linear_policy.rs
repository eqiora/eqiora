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
    pub(crate) planning_profile: Option<eqiora_solver::SolverPlanningProfile>,
    pub(crate) planning_policy_id: Option<&'static str>,
    pub(crate) selected_candidate_id: Option<&'static str>,
    pub(crate) selected_evidence_case: Option<&'static str>,
    pub(crate) planning_costs: Vec<eqiora_solver::SolverCandidateCost>,
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
        let execution = backend.default_execution_provider();
        execution.validate()?;
        let report = backend.default_execution();
        let eqiora_solver::ExecutionTopology::Host { workers } = report.topology() else {
            return Err(invalid(
                "common linear policies require host-local execution",
            ));
        };
        if report.adapter() != execution.id() {
            return Err(invalid(
                "execution report differs from its provider identity",
            ));
        }
        Ok(Self {
            scalar_domain: S::DOMAIN,
            solver,
            provider,
            capabilities: backend.capabilities(),
            execution,
            workers,
            planning_objective: None,
            planning_profile: None,
            planning_policy_id: None,
            selected_candidate_id: None,
            selected_evidence_case: None,
            planning_reasons: Vec::new(),
            planning_costs: Vec::new(),
        })
    }

    pub(crate) fn with_planning<
        S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + Sync,
    >(
        mut self,
        decision: &ResolvedHostSolverPlan<'_, S>,
    ) -> Result<Self, Diagnostic> {
        if self.scalar_domain != S::DOMAIN
            || self.solver != decision.solver_plan()
            || self.provider != decision.solver_provider()
            || self.execution != decision.execution_provider()
            || eqiora_solver::ExecutionReport::host(self.execution.id(), self.workers)
                != decision.execution()
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
        self.planning_costs = decision.costs().to_vec();
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
            || backend.default_execution_provider() != self.execution
            || backend.default_execution()
                != eqiora_solver::ExecutionReport::host(self.execution.id(), self.workers)
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
        if backend.provider() != self.provider
            || backend.default_execution_provider() != complex.default_execution_provider()
            || backend.default_execution() != complex.default_execution()
        {
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
    profile: Option<eqiora_solver::SolverPlanningProfile>,
}

impl<S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + Sync + 'static>
    LinearSolverBackend<S> for ProfileCheckedBackend<'_, S>
{
    fn default_execution_provider(&self) -> ExecutionProvider {
        self.backend.default_execution_provider()
    }
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
            plan: self.plan,
            provider: self.backend.provider(),
            action: PreparedLinearAction::Idle,
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
            profile.require_problem_resources(plan, self.backend.provider(), problem)?;
        }
        self.backend.solve_with_execution(problem, plan, execution)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PreparedLinearAction {
    Idle,
    Active,
    Failed,
}

#[derive(Debug)]
pub(super) struct ProfileCheckedPreparedLinear<S> {
    pub(super) prepared: Box<dyn eqiora_solver::PreparedLinearSolver<S>>,
    pub(super) profile: Option<eqiora_solver::SolverPlanningProfile>,
    pub(super) plan: SolverPlan,
    pub(super) provider: SolverProvider,
    pub(super) action: PreparedLinearAction,
}

impl<S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + Sync>
    eqiora_solver::PreparedLinearSolver<S> for ProfileCheckedPreparedLinear<S>
{
    fn begin_action(&mut self) -> Result<(), Diagnostic> {
        if self.action != PreparedLinearAction::Idle {
            return Err(invalid("prepared action must be idle before beginning"));
        }
        self.prepared.begin_action()?;
        self.action = PreparedLinearAction::Active;
        Ok(())
    }

    fn solve(
        &mut self,
        structure: &eqiora_solver::PreparedLinearStructureIdentity,
        problem: &eqiora_solver::LinearProblem<'_, S>,
    ) -> Result<eqiora_solver::LinearSolution<S>, Diagnostic> {
        if self.action != PreparedLinearAction::Active {
            return Err(invalid(
                "prepared solve requires an active, nonfailed action",
            ));
        }
        let result = (|| {
            if let Some(profile) = &self.profile {
                profile.require_problem(problem)?;
                profile.require_problem_resources(self.plan, self.provider, problem)?;
            }
            self.prepared.solve(structure, problem)
        })();
        // Profile rejection happens before the provider sees this solve. It must
        // still poison the enclosing action so earlier tentative factors cannot
        // be committed after a later rejected correction.
        if result.is_err() {
            self.action = PreparedLinearAction::Failed;
        }
        result
    }

    fn commit_action(&mut self) -> Result<(), Diagnostic> {
        if self.action != PreparedLinearAction::Active {
            return Err(invalid(
                "prepared commit requires an active, nonfailed action",
            ));
        }
        match self.prepared.commit_action() {
            Ok(()) => {
                self.action = PreparedLinearAction::Idle;
                Ok(())
            }
            Err(error) => {
                self.action = PreparedLinearAction::Failed;
                Err(error)
            }
        }
    }

    fn discard_action(&mut self) -> Result<(), Diagnostic> {
        if self.action == PreparedLinearAction::Idle {
            return Err(invalid(
                "prepared discard requires an active or failed action",
            ));
        }
        self.prepared.discard_action()?;
        self.action = PreparedLinearAction::Idle;
        Ok(())
    }
}
