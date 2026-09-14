//! Exact no-Mesh lifecycle for admitted finite algebraic mathematics.

use super::*;
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
    symbols: Vec<SymbolRef>,
    dimensions: Vec<DimExponents>,
    identity: String,
    model_id: String,
    model_digest: String,
    model_revision: u64,
}

/// Initial numerical values bound to one exact finite Plan.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonAlgebraicState {
    plan_identity: String,
    identity: String,
    values: Vec<f64>,
}

impl CommonAlgebraicState {
    /// Canonical exact initial-State encoding.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&(
            "eqiora.common-algebraic-state/v1",
            &self.plan_identity,
            &self.identity,
            &self.values,
        ))
        .map_err(|e| invalid(format!("cannot encode finite State: {e}")))
    }
    /// Reauthenticate the complete initial State against its exact Plan.
    pub fn from_bytes(bytes: &[u8], plan: &CommonAlgebraicPlan) -> Result<Self, Diagnostic> {
        let expected = plan.initial_state()?;
        if bytes != expected.to_bytes()? {
            return Err(invalid(
                "finite State bytes differ from exact Plan initial State",
            ));
        }
        Ok(expected)
    }

    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

impl CommonAlgebraicPlan {
    pub fn resolve(
        model: &ModelEnvelope,
        solve: CommonSolvePolicy,
        enforcement: Option<FiniteConstraintEnforcement>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<Self, Diagnostic> {
        let CommonSolvePolicy::Linear(request) = solve else {
            return Err(invalid("finite affine Plan requires a linear solve policy"));
        };
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
            return Err(invalid("finite affine Plan requires exact linear controls"));
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
                "finite affine Plan admits only exact SparseLU/Identity/Fast execution",
            ));
        }
        let reference = model.artifact_reference()?;
        let model_digest = reference.artifact().to_string();
        let mut bytes = model_digest.as_bytes().to_vec();
        push_framed(&mut bytes, &plan_artifact::linear_intent_bytes(request)?);
        push_framed(
            &mut bytes,
            &super::plan_artifact::finite_enforcement_bytes(problem.enforcement())?,
        );
        let identity = finite_digest(b"eqiora.common-algebraic-plan/v3\0", &bytes);
        Ok(Self {
            model: Arc::new(model.clone()),
            kernel,
            problem,
            linear,
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
    /// Explicit mathematical enforcement retained by this numerical Plan.
    #[must_use]
    pub fn enforcement(&self) -> Option<&FiniteConstraintEnforcement> {
        self.problem.enforcement()
    }
    pub fn initial_state(&self) -> Result<CommonAlgebraicState, Diagnostic> {
        let values = vec![0.0; self.symbols.len()];
        Ok(CommonAlgebraicState {
            identity: finite_digest(
                b"eqiora.common-algebraic-initial-state/v1\0",
                self.identity.as_bytes(),
            ),
            plan_identity: self.identity.clone(),
            values,
        })
    }
    pub fn run_result(
        &self,
        state: &CommonAlgebraicState,
        backend: &dyn LinearSolverBackend,
    ) -> Result<crate::CommonResult, Diagnostic> {
        if state != &self.initial_state()? || backend.provider() != self.linear.provider {
            return Err(invalid(
                "finite Run requires its exact Plan-bound State and admitted provider",
            ));
        }
        let checked = self.linear.checked_backend(backend, None)?;
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
