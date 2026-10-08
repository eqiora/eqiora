//! Exact no-Mesh lifecycle for admitted finite algebraic mathematics.

use super::*;
use eqiora_realization::{NonlinearSolvePlan, PositivePhysicalScale};
mod differentiation;
mod problem;
use crate::finite_constraints::{ConstraintAssessment, ConstraintRef, FiniteConstraintEnforcement};
use crate::physical_network::{
    ScalarPhysicalAffineProblem, lower_scalar_physical_affine, solve_scalar_physical_affine,
};
use eqiora_schema::kernel::{KernelNode, SymbolRef};
use eqiora_sem::PhysicalUnknown;
use eqiora_solver::{FixedOrderInnerProduct, ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION};
use problem::AlgebraicProblem;

/// One exact finite algebraic Model and its explicit numerical realization.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonAlgebraicPlan {
    model: Arc<ModelEnvelope>,
    kernel: KernelProgram,
    problem: AlgebraicProblem,
    authored: Option<eqiora_compiler::AuthoredFormulationProjection>,
    gauge: Option<crate::finite_constraints::FiniteGauge>,
    complex_system: Option<eqiora_solver::CanonicalCsrSystemView<num_complex::Complex64>>,
    pub(super) linear: NativeLinearPolicy,
    nonlinear: Option<NonlinearSolvePlan>,
    symbols: Vec<SymbolRef>,
    dimensions: Vec<DimExponents>,
    identity: String,
    model_id: String,
    model_digest: String,
    model_revision: u64,
}

mod state;
pub use state::CommonAlgebraicState;

impl CommonAlgebraicPlan {
    pub fn resolve(
        model: &ModelEnvelope,
        solve: CommonSolvePolicy,
        enforcement: Option<FiniteConstraintEnforcement>,
        residual_scales: &[(ConstraintRef, PositivePhysicalScale)],
        authored: Option<&eqiora_compiler::AuthoredFormulationProjection>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<Self, Diagnostic> {
        let (request, nonlinear) = match solve {
            CommonSolvePolicy::Linear(request) => (request, None),
            CommonSolvePolicy::Newton { linear, nonlinear } => (linear, Some(nonlinear)),
        };
        if enforcement
            .as_ref()
            .is_some_and(|policy| policy.is_strict_interior() != nonlinear.is_some())
        {
            return Err(invalid(
                "finite Newton requires strict-interior inequalities; active-set Plans require Linear",
            ));
        }
        let kernel = model.to_program().map_err(|errors| {
            errors
                .into_iter()
                .next()
                .unwrap_or_else(|| invalid("finite Model replay failed"))
        })?;
        let problem = AlgebraicProblem::admit(&kernel, enforcement, nonlinear.is_some())?;
        let problem = match problem {
            AlgebraicProblem::Constrained(problem) => {
                AlgebraicProblem::Constrained(problem.with_residual_scales(residual_scales)?)
            }
            other if residual_scales.is_empty() => other,
            _ => {
                return Err(invalid(
                    "residual scales require a finite Field Newton Plan",
                ));
            }
        };
        let effective_scales = match &problem {
            AlgebraicProblem::Constrained(problem) if nonlinear.is_some() => {
                problem.residual_scales()
            }
            _ => Vec::new(),
        };
        let gauge = authored.map(|form| problem.gauge(form)).transpose()?;
        let symbols = problem.symbols();
        let dimensions = problem.dimensions();
        if request.objective().is_some() {
            return Err(invalid(
                "finite algebraic Plan requires exact linear controls",
            ));
        }
        let complex_system = if nonlinear.is_none() {
            problem.complex_linear_system()?
        } else {
            None
        };
        let linear = if complex_system.is_some() {
            solver_planning::resolve_complex_linear(
                request,
                LinearOperatorProperties::General,
                backend,
            )?
        } else {
            solver_planning::resolve_linear(
                request,
                if gauge.is_some() {
                    LinearOperatorProperties::SymmetricIndefinite
                } else {
                    LinearOperatorProperties::General
                },
                None,
                None,
                None,
                backend,
            )?
        };
        if complex_system.is_none()
            && (linear.solver.algorithm() != LinearSolver::SparseLu
                || linear.solver.preconditioner() != PreconditionerPolicy::Identity
                || linear.solver.reduction() != ReductionPolicy::Fast)
        {
            return Err(invalid(
                "finite algebraic Plan admits only exact SparseLU/Identity/Fast execution",
            ));
        }
        let reference = model.artifact_reference()?;
        let model_digest = reference.artifact().to_string();
        let mut bytes = model_digest.as_bytes().to_vec();
        push_framed(&mut bytes, &plan_artifact::solve_intent_bytes(solve)?);
        push_framed(
            &mut bytes,
            &super::plan_artifact::finite_enforcement_bytes(problem.enforcement())?,
        );
        push_framed(
            &mut bytes,
            authored.map_or(&[], |form| form.canonical_bytes()),
        );
        push_framed(
            &mut bytes,
            &plan_artifact::residual_scaling_bytes(&effective_scales)?,
        );
        let identity = finite_digest(b"eqiora.common-algebraic-plan/v6\0", &bytes);
        Ok(Self {
            model: Arc::new(model.clone()),
            kernel,
            problem,
            authored: authored.cloned(),
            gauge,
            complex_system,
            linear,
            nonlinear,
            symbols,
            dimensions,
            identity,
            model_id: reference.model().ulid().to_string(),
            model_digest,
            model_revision: reference.semantic_revision().get(),
        })
    }
    /// Complete equality scales in canonical condition order. Each divides every
    /// real/imaginary row of its original equality. Empty for linear Plans.
    #[must_use]
    pub fn residual_scales(&self) -> Vec<(ConstraintRef, PositivePhysicalScale)> {
        match &self.problem {
            AlgebraicProblem::Constrained(problem) if self.nonlinear.is_some() => {
                problem.residual_scales()
            }
            _ => Vec::new(),
        }
    }
    /// Exact authored finite reference retained for Plan replay.
    #[must_use]
    pub fn authored_formulation_bytes(&self) -> Option<&[u8]> {
        self.authored.as_ref().map(|form| form.canonical_bytes())
    }
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
    #[must_use]
    pub fn model_digest(&self) -> &str {
        &self.model_digest
    }
    #[must_use]
    pub const fn model_revision(&self) -> u64 {
        self.model_revision
    }
    #[must_use]
    pub fn model_artifact(&self) -> &ModelEnvelope {
        &self.model
    }
    #[must_use]
    pub fn kernel(&self) -> &KernelProgram {
        &self.kernel
    }
    /// Distinct original Model unknowns; a shaped or complex value has multiple
    /// numerical coordinates without creating additional semantic symbols.
    #[must_use]
    pub fn symbols(&self) -> &[SymbolRef] {
        &self.symbols
    }
    /// Number of real numerical coordinates, including both parts of complex values.
    #[must_use]
    pub fn coordinate_count(&self) -> usize {
        self.problem.coordinate_count()
    }

    pub(crate) fn field_values(
        &self,
        values: &[f64],
    ) -> Result<
        Vec<(
            eqiora_core::Id<eqiora_core::entity::kinds::Field>,
            eqiora_core::ValueLiteral,
        )>,
        Diagnostic,
    > {
        self.problem.field_values(values)
    }

    #[must_use]
    pub fn dimensions(&self) -> &[DimExponents] {
        &self.dimensions
    }
    #[must_use]
    pub const fn solver_provider(&self) -> SolverProvider {
        self.linear.provider
    }
    #[must_use]
    pub const fn linear(&self) -> SolverPlan {
        self.linear.solver
    }
    /// Nonlinear residual/globalization controls, absent for affine execution.
    #[must_use]
    pub const fn nonlinear(&self) -> Option<NonlinearSolvePlan> {
        self.nonlinear
    }

    /// Explicit mathematical enforcement retained by this numerical Plan.
    #[must_use]
    pub fn enforcement(&self) -> Option<&FiniteConstraintEnforcement> {
        self.problem.enforcement()
    }
    pub fn run_result(
        &self,
        state: &CommonAlgebraicState,
        backend: &dyn LinearSolverBackend,
    ) -> Result<crate::CommonResult, Diagnostic> {
        if state != &self.state_from_values(state.values.clone())?
            || backend.provider() != self.linear.provider
        {
            return Err(invalid(
                "finite Run requires its exact Plan-bound State and admitted provider",
            ));
        }
        if let Some(gauge) = &self.gauge {
            let checked = self.linear.checked_backend(backend, None)?;
            let solution = gauge.solve(LinearSolveRequest::new(&checked, self.linear.solver))?;
            return crate::CommonResult::from_algebraic(
                self,
                state,
                solution.values,
                solution.report,
                None,
                Some(solution.evidence),
            );
        }
        if let Some(system) = &self.complex_system {
            let initial = state
                .values
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| num_complex::Complex64::new(pair[0], pair[1]))
                .collect::<Vec<_>>();
            let problem = system.linear_problem()?.with_initial_guess(&initial)?;
            let complex = self.linear.checked_complex_backend(backend, None)?;
            let solution = LinearSolveRequest::new(&complex, self.linear.solver).solve(&problem)?;
            let values = solution
                .values()
                .iter()
                .flat_map(|value| [value.re, value.im])
                .collect();
            return crate::CommonResult::from_algebraic(
                self,
                state,
                values,
                solution.report().clone(),
                None,
                None,
            );
        }
        let checked = self.linear.checked_backend(backend, None)?;
        if let Some(nonlinear) = self.nonlinear {
            let solution = self.problem.nonlinear()?.solve_nonlinear(
                &state.values,
                &[],
                &[],
                nonlinear,
                LinearSolveRequest::new(&checked, self.linear.solver),
            )?;
            return crate::CommonResult::from_nonlinear_algebraic(self, state, solution);
        }
        let solution = self.problem.solve(
            &state.values,
            LinearSolveRequest::new(&checked, self.linear.solver),
        )?;
        crate::CommonResult::from_algebraic(
            self,
            state,
            solution.values,
            solution.report,
            solution.active_set_mask,
            None,
        )
    }
    pub(crate) fn validate_nonlinear_values(
        &self,
        state: &CommonAlgebraicState,
        values: &[f64],
    ) -> Result<(f64, ConstraintAssessment), Diagnostic> {
        if state != &self.state_from_values(state.values.clone())? {
            return Err(invalid("nonlinear Result has a foreign initial State"));
        }
        self.problem.nonlinear()?.validate_nonlinear_values(
            state.values(),
            values,
            self.nonlinear
                .ok_or_else(|| invalid("nonlinear Result requires a Newton Plan"))?,
        )
    }
    pub(crate) fn validate_values(
        &self,
        values: &[f64],
        target: f64,
        mask: Option<u32>,
        evidence: Option<&crate::nullspace::NullspaceEvidence>,
    ) -> Result<(f64, Option<ConstraintAssessment>), Diagnostic> {
        match (&self.gauge, evidence) {
            (Some(gauge), Some(evidence)) => {
                if mask.is_some()
                    || gauge.residual_target(self.linear.solver)?.to_bits() != target.to_bits()
                    || &gauge.assess(values, evidence.multiplier, self.linear.solver)? != evidence
                {
                    return Err(invalid(
                        "finite gauge Result evidence differs from its original equations or explicit reference",
                    ));
                }
                let assessment = self.problem.gauge_assessment(values, self.linear.solver)?;
                return Ok((assessment.equality_residual_norm(), Some(assessment)));
            }
            (None, None) => {}
            _ => {
                return Err(invalid(
                    "finite Result gauge evidence presence differs from its Plan",
                ));
            }
        }
        self.problem
            .validate_values(values, self.linear.solver, target, mask)
    }
}

fn finite_digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(bytes);
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
