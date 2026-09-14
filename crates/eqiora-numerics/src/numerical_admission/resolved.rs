//! Shared inspection of the one native common Plan sum.

use eqiora_solver::{
    ExecutionProvider, LinearOperatorProperties, SolverPlan, SolverPlanningObjective,
    SolverProvider,
};

use super::{
    CommonElasticityPlan, CommonFsiPlan, CommonOdePlan, CommonScalarPlan, CommonSteadyStokesPlan,
    CommonTransientFlowPlan, ResolvedCommonPlan,
};

impl ResolvedCommonPlan {
    #[must_use]
    pub fn as_algebraic(&self) -> Option<&super::CommonAlgebraicPlan> {
        match self {
            Self::Algebraic(plan) => Some(plan),
            _ => None,
        }
    }
    /// Exact identity of this complete resolved Plan.
    #[must_use]
    pub fn identity(&self) -> &str {
        match self {
            Self::Algebraic(plan) => plan.identity(),
            Self::Ode(plan) => plan.identity(),
            Self::Scalar(plan) => plan.identity(),
            Self::Elasticity(plan) => plan.identity(),
            Self::SteadyStokes(plan) => plan.identity(),
            Self::TransientFlow(plan) => plan.identity(),
            Self::Fsi(plan) => plan.identity(),
        }
    }

    /// Exact semantic Model identifier.
    #[must_use]
    pub fn model_id(&self) -> &str {
        match self {
            Self::Algebraic(plan) => plan.model_id(),
            Self::Ode(plan) => plan.model_id(),
            Self::Scalar(plan) => plan.model_id(),
            Self::Elasticity(plan) => plan.model_id(),
            Self::SteadyStokes(plan) => plan.model_id(),
            Self::TransientFlow(plan) => plan.model_id(),
            Self::Fsi(plan) => plan.model_id(),
        }
    }

    /// Exact canonical Model artifact digest.
    #[must_use]
    pub fn model_digest(&self) -> &str {
        match self {
            Self::Algebraic(plan) => plan.model_digest(),
            Self::Ode(plan) => plan.model_digest(),
            Self::Scalar(plan) => plan.model_digest(),
            Self::Elasticity(plan) => plan.model_digest(),
            Self::SteadyStokes(plan) => plan.model_digest(),
            Self::TransientFlow(plan) => plan.model_digest(),
            Self::Fsi(plan) => plan.model_digest(),
        }
    }

    /// Exact authored Model revision.
    #[must_use]
    pub const fn model_revision(&self) -> u64 {
        match self {
            Self::Algebraic(plan) => plan.model_revision(),
            Self::Ode(plan) => plan.model_revision(),
            Self::Scalar(plan) => plan.model_revision(),
            Self::Elasticity(plan) => plan.model_revision(),
            Self::SteadyStokes(plan) => plan.model_revision(),
            Self::TransientFlow(plan) => plan.model_revision(),
            Self::Fsi(plan) => plan.model_revision(),
        }
    }

    /// Exact Geometry digest for a spatial Plan.
    #[must_use]
    pub fn geometry_digest(&self) -> Option<&str> {
        match self {
            Self::Algebraic(_) | Self::Ode(_) => None,
            Self::Scalar(plan) => Some(plan.geometry_digest()),
            Self::Elasticity(plan) => Some(plan.geometry_digest()),
            Self::SteadyStokes(plan) => Some(plan.geometry_digest()),
            Self::TransientFlow(plan) => Some(plan.geometry_digest()),
            Self::Fsi(plan) => Some(plan.geometry_digest()),
        }
    }

    /// Exact Mesh digest for a spatial Plan.
    #[must_use]
    pub fn mesh_digest(&self) -> Option<&str> {
        match self {
            Self::Algebraic(_) | Self::Ode(_) => None,
            Self::Scalar(plan) => Some(plan.mesh_digest()),
            Self::Elasticity(plan) => Some(plan.mesh_digest()),
            Self::SteadyStokes(plan) => Some(plan.mesh_digest()),
            Self::TransientFlow(plan) => Some(plan.mesh_digest()),
            Self::Fsi(plan) => Some(plan.mesh_digest()),
        }
    }

    /// Exact Geometry-to-Mesh correspondence digest for a spatial Plan.
    #[must_use]
    pub fn correspondence_digest(&self) -> Option<&str> {
        match self {
            Self::Algebraic(_) | Self::Ode(_) => None,
            Self::Scalar(plan) => Some(plan.correspondence_digest()),
            Self::Elasticity(plan) => Some(plan.correspondence_digest()),
            Self::SteadyStokes(plan) => Some(plan.correspondence_digest()),
            Self::TransientFlow(plan) => Some(plan.correspondence_digest()),
            Self::Fsi(plan) => Some(plan.correspondence_digest()),
        }
    }

    /// Exact Mesh production occurrence digest for a spatial Plan.
    #[must_use]
    pub fn production_digest(&self) -> Option<&str> {
        match self {
            Self::Algebraic(_) | Self::Ode(_) => None,
            Self::Scalar(plan) => Some(plan.production_digest()),
            Self::Elasticity(plan) => Some(plan.production_digest()),
            Self::SteadyStokes(plan) => Some(plan.production_digest()),
            Self::TransientFlow(plan) => Some(plan.production_digest()),
            Self::Fsi(plan) => Some(plan.production_digest()),
        }
    }

    /// Exact portable realization-graph digest for a spatial Plan.
    #[must_use]
    pub fn realization_digest(&self) -> Option<&str> {
        match self {
            Self::Algebraic(_) | Self::Ode(_) => None,
            Self::Scalar(plan) => Some(plan.realization_digest()),
            Self::Elasticity(plan) => Some(plan.realization_digest()),
            Self::SteadyStokes(plan) => Some(plan.realization_digest()),
            Self::TransientFlow(plan) => Some(plan.realization_digest()),
            Self::Fsi(plan) => Some(plan.realization_digest()),
        }
    }

    /// Effective linear solve for a spatial Plan.
    #[must_use]
    pub const fn effective_solver(&self) -> Option<SolverPlan> {
        match self {
            Self::Algebraic(plan) => Some(plan.linear()),
            Self::Ode(_) => None,
            Self::Scalar(plan) => Some(plan.linear()),
            Self::Elasticity(plan) => Some(plan.linear()),
            Self::SteadyStokes(plan) => Some(plan.linear()),
            Self::TransientFlow(plan) => Some(plan.linear()),
            Self::Fsi(plan) => Some(plan.linear()),
        }
    }

    /// Exact spatial solver release, including its declared library inventory.
    #[must_use]
    pub const fn linear_solver_provider(&self) -> Option<SolverProvider> {
        match self {
            Self::Algebraic(plan) => Some(plan.solver_provider()),
            Self::Ode(_) => None,
            Self::Scalar(plan) => Some(plan.admission.linear.provider),
            Self::Elasticity(plan) => Some(plan.admission.linear.provider),
            Self::SteadyStokes(plan) => Some(plan.admission.linear.provider),
            Self::TransientFlow(plan) => Some(plan.admission.linear.provider),
            Self::Fsi(plan) => Some(plan.solver_provider()),
        }
    }

    pub(crate) const fn linear_execution_provider(&self) -> Option<(ExecutionProvider, usize)> {
        match self {
            Self::Algebraic(plan) => Some((plan.linear.execution, 1)),
            Self::Ode(_) => None,
            Self::Scalar(plan) => Some((
                plan.admission.linear.execution,
                plan.admission.linear.workers.get(),
            )),
            Self::Elasticity(plan) => Some((
                plan.admission.linear.execution,
                plan.admission.linear.workers.get(),
            )),
            Self::SteadyStokes(plan) => Some((
                plan.admission.linear.execution,
                plan.admission.linear.workers.get(),
            )),
            Self::TransientFlow(plan) => Some((
                plan.admission.linear.execution,
                plan.admission.linear.workers.get(),
            )),
            Self::Fsi(plan) => Some((plan.execution_provider(), plan.workers().get())),
        }
    }

    /// Linear operator class owned by the resolved capability.
    #[must_use]
    pub const fn operator_properties(&self) -> Option<LinearOperatorProperties> {
        match self {
            Self::Algebraic(_) => Some(LinearOperatorProperties::General),
            Self::Ode(_) => None,
            Self::Scalar(plan) => Some(match plan.admission.spatial {
                super::NativeSpatialPolicy::ScalarQ1 => LinearOperatorProperties::General,
                super::NativeSpatialPolicy::ScalarTpfa => {
                    LinearOperatorProperties::SymmetricPositiveDefinite
                }
                _ => unreachable!(),
            }),
            Self::Elasticity(_) => Some(LinearOperatorProperties::SymmetricPositiveDefinite),
            Self::SteadyStokes(_) | Self::Fsi(_) => {
                Some(LinearOperatorProperties::SymmetricIndefinite)
            }
            Self::TransientFlow(_) => Some(LinearOperatorProperties::General),
        }
    }

    /// Effective solver/time backend identity.
    #[must_use]
    pub fn solver_backend(&self) -> &'static str {
        match self {
            Self::Algebraic(plan) => plan.solver_provider().id().as_str(),
            Self::Ode(plan) => plan.backend().id(),
            Self::Scalar(plan) => plan.admission.linear.provider.id().as_str(),
            Self::Elasticity(plan) => plan.admission.linear.provider.id().as_str(),
            Self::SteadyStokes(plan) => plan.admission.linear.provider.id().as_str(),
            Self::TransientFlow(plan) => plan.solver_provider().id().as_str(),
            Self::Fsi(plan) => plan.solver_provider().id().as_str(),
        }
    }

    /// Effective solver/time backend implementation version.
    #[must_use]
    pub fn solver_backend_version(&self) -> &'static str {
        match self {
            Self::Algebraic(plan) => plan.solver_provider().implementation_version(),
            Self::Ode(plan) => plan.backend().version(),
            Self::Scalar(plan) => plan.admission.linear.provider.implementation_version(),
            Self::Elasticity(plan) => plan.admission.linear.provider.implementation_version(),
            Self::SteadyStokes(plan) => plan.admission.linear.provider.implementation_version(),
            Self::TransientFlow(plan) => plan.solver_provider().implementation_version(),
            Self::Fsi(plan) => plan.solver_provider().implementation_version(),
        }
    }

    const fn linear_policy(&self) -> Option<&super::NativeLinearPolicy> {
        match self {
            Self::Algebraic(plan) => Some(&plan.linear),
            Self::Ode(_) => None,
            Self::Scalar(plan) => Some(&plan.admission.linear),
            Self::Elasticity(plan) => Some(&plan.admission.linear),
            Self::SteadyStokes(plan) => Some(&plan.admission.linear),
            Self::TransientFlow(plan) => Some(&plan.admission.linear),
            Self::Fsi(plan) => Some(&plan.linear),
        }
    }

    /// Program-controlled solver-planning objective when one was requested.
    #[must_use]
    pub const fn solver_planning_objective(&self) -> Option<SolverPlanningObjective> {
        match self.linear_policy() {
            Some(linear) => linear.planning_objective,
            None => None,
        }
    }

    /// Versioned solver-planning policy identity, when applicable.
    #[must_use]
    pub const fn solver_planning_policy_id(&self) -> Option<&'static str> {
        match self.linear_policy() {
            Some(linear) => linear.planning_policy_id,
            None => None,
        }
    }

    /// Selected solver candidate identity, when planning was requested.
    #[must_use]
    pub const fn selected_solver_candidate_id(&self) -> Option<&'static str> {
        match self.linear_policy() {
            Some(linear) => linear.selected_candidate_id,
            None => None,
        }
    }

    /// Evidence case attached to the selected solver candidate.
    #[must_use]
    pub const fn selected_solver_evidence_case(&self) -> Option<&'static str> {
        match self.linear_policy() {
            Some(linear) => linear.selected_evidence_case,
            None => None,
        }
    }

    /// Stable reason codes for program-controlled solver planning.
    #[must_use]
    pub fn solver_planning_reasons(&self) -> &[(&'static str, &'static str)] {
        self.linear_policy()
            .map_or(&[], |linear| linear.planning_reasons.as_slice())
    }

    /// Borrow the exact ODE Plan when this is the ODE variant.
    #[must_use]
    pub fn as_ode(&self) -> Option<&CommonOdePlan> {
        match self {
            Self::Ode(plan) => Some(plan),
            _ => None,
        }
    }

    /// Borrow the exact scalar Plan when this is the scalar variant.
    #[must_use]
    pub fn as_scalar(&self) -> Option<&CommonScalarPlan> {
        match self {
            Self::Scalar(plan) => Some(plan),
            _ => None,
        }
    }

    /// Borrow the exact elasticity Plan when this is the elasticity variant.
    #[must_use]
    pub fn as_elasticity(&self) -> Option<&CommonElasticityPlan> {
        match self {
            Self::Elasticity(plan) => Some(plan),
            _ => None,
        }
    }

    /// Borrow the exact steady-Stokes Plan when this is that variant.
    #[must_use]
    pub fn as_steady_stokes(&self) -> Option<&CommonSteadyStokesPlan> {
        match self {
            Self::SteadyStokes(plan) => Some(plan),
            _ => None,
        }
    }

    /// Borrow the exact transient-flow Plan when this is that variant.
    #[must_use]
    pub fn as_transient_flow(&self) -> Option<&CommonTransientFlowPlan> {
        match self {
            Self::TransientFlow(plan) => Some(plan),
            _ => None,
        }
    }

    /// Borrow the exact fixed-reference FSI Plan when this is that variant.
    #[must_use]
    pub fn as_fsi(&self) -> Option<&CommonFsiPlan> {
        match self {
            Self::Fsi(plan) => Some(plan),
            _ => None,
        }
    }
}
