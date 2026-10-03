//! Exact no-Mesh lifecycle for admitted finite algebraic mathematics.

use super::*;
use eqiora_realization::NonlinearSolvePlan;
mod problem;
use crate::finite_constraints::{ConstraintAssessment, FiniteConstraintEnforcement};
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
        backend: &dyn LinearSolverBackend,
    ) -> Result<Self, Diagnostic> {
        let (request, nonlinear) = match solve {
            CommonSolvePolicy::Linear(request) => (request, None),
            CommonSolvePolicy::Newton { linear, nonlinear } => (linear, Some(nonlinear)),
        };
        if nonlinear.is_some()
            != enforcement
                .as_ref()
                .is_some_and(FiniteConstraintEnforcement::is_strict_interior)
        {
            return Err(invalid(
                "finite Newton requires strict-interior enforcement; active-set and conserving Plans require Linear",
            ));
        }
        let kernel = model.to_program().map_err(|errors| {
            errors
                .into_iter()
                .next()
                .unwrap_or_else(|| invalid("finite Model replay failed"))
        })?;
        let problem = AlgebraicProblem::admit(&kernel, enforcement)?;
        let symbols = problem.symbols();
        let dimensions = problem.dimensions();
        if request.objective().is_some() {
            return Err(invalid(
                "finite algebraic Plan requires exact linear controls",
            ));
        }
        let linear = solver_planning::resolve_linear(
            request,
            LinearOperatorProperties::General,
            None,
            None,
            None,
            backend,
        )?;
        if linear.solver.algorithm() != LinearSolver::SparseLu
            || linear.solver.preconditioner() != PreconditionerPolicy::Identity
            || linear.solver.reduction() != ReductionPolicy::Fast
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
        let identity = finite_digest(b"eqiora.common-algebraic-plan/v4\0", &bytes);
        Ok(Self {
            model: Arc::new(model.clone()),
            kernel,
            problem,
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
    #[must_use]
    pub fn symbols(&self) -> &[SymbolRef] {
        &self.symbols
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
    ) -> Result<(f64, Option<ConstraintAssessment>), Diagnostic> {
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
