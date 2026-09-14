use eqiora_core::{Diagnostic, OntologyId};
use eqiora_schema::Model;

use crate::{
    BackwardEulerStatePair, ConformingTraceQuotient, CoupledFieldwiseRealizationPlan,
    DomainFieldInventory, QuadraturePolicy, RealizationCapabilities, RealizationRequirements,
    RealizationRevision, SemanticRevision, invalid_realization,
};

/// Exact multi-Domain lowerer facts against which one coupled plan is admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoupledFieldwiseRealizationRequirements {
    domains: Vec<DomainFieldInventory>,
    trace_quotients: Vec<ConformingTraceQuotient>,
    eliminated_states: Vec<BackwardEulerStatePair>,
    execution: RealizationRequirements,
}

impl CoupledFieldwiseRealizationRequirements {
    /// Construct canonical exact Domain, Field, Connection, state, and execution requirements.
    ///
    /// # Errors
    /// Returns `EQ0807` unless at least two Domains are distinct, every Field
    /// occurs exactly once, and every trace endpoint occurs in that inventory.
    /// The quotient collection must be nonempty without duplicate selections.
    pub fn new(
        domains: impl IntoIterator<Item = DomainFieldInventory>,
        trace_quotients: impl AsRef<[ConformingTraceQuotient]>,
        eliminated_states: impl IntoIterator<Item = BackwardEulerStatePair>,
        execution: RealizationRequirements,
    ) -> Result<Self, Diagnostic> {
        let mut domains = domains.into_iter().collect::<Vec<_>>();
        domains.sort_by_key(|domain| domain.domain().ulid());
        if domains.len() < 2 {
            return Err(invalid_realization(
                "coupled field-wise requirements need at least two Semantic Domains",
            ));
        }
        if domains
            .windows(2)
            .any(|pair| pair[0].domain() == pair[1].domain())
        {
            return Err(invalid_realization(
                "coupled field-wise requirements contain a duplicate Semantic Domain",
            ));
        }
        let mut fields = domains
            .iter()
            .flat_map(|domain| domain.fields().iter().copied())
            .collect::<Vec<_>>();
        fields.sort_by_key(|field| field.ulid());
        if fields.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid_realization(
                "coupled field-wise requirements contain a Field in more than one Domain",
            ));
        }
        let trace_quotients = crate::coupled_fieldwise::canonical_trace_quotients(
            trace_quotients.as_ref().iter().copied(),
        )?;
        if trace_quotients
            .iter()
            .flat_map(|quotient| quotient.endpoints())
            .any(|endpoint| {
                !domains.iter().any(|domain| {
                    domain.domain() == endpoint.domain()
                        && domain.fields().contains(&endpoint.field())
                })
            })
        {
            return Err(invalid_realization(
                "coupled field-wise requirements must contain both exact trace endpoint Fields",
            ));
        }
        let eliminated_states = crate::coupled_fieldwise::canonical_state_pairs(eliminated_states)?;
        for eliminated_state in &eliminated_states {
            let state_domain = domains
                .iter()
                .find(|domain| domain.fields().contains(&eliminated_state.state()))
                .map(DomainFieldInventory::domain);
            let rate_domain = domains
                .iter()
                .find(|domain| domain.fields().contains(&eliminated_state.rate()))
                .map(DomainFieldInventory::domain);
            if state_domain.is_none() || state_domain != rate_domain {
                return Err(invalid_realization(
                    "Backward Euler state and rate requirements must occur exactly once on the same Domain",
                ));
            }
        }
        Ok(Self {
            domains,
            trace_quotients,
            eliminated_states,
            execution,
        })
    }

    /// Canonically ordered exact Domain and participating-Field inventory.
    #[must_use]
    pub fn domains(&self) -> &[DomainFieldInventory] {
        &self.domains
    }

    /// Exact Connections and paired Field traces required by the lowerer.
    #[must_use]
    pub fn trace_quotients(&self) -> &[ConformingTraceQuotient] {
        &self.trace_quotients
    }

    /// Exact Relation/state/rate inventory selected for Backward Euler elimination.
    #[must_use]
    pub fn eliminated_states(&self) -> &[BackwardEulerStatePair] {
        &self.eliminated_states
    }

    /// Dimension, scalar, and vector-layout requirements.
    #[must_use]
    pub const fn execution(&self) -> RealizationRequirements {
        self.execution
    }
}

/// Explicit multi-Domain Field-wise request at an independent revision.
#[derive(Debug, Clone, PartialEq)]
pub struct CoupledFieldwiseRealizationRequest {
    model: OntologyId<Model>,
    semantic_revision: SemanticRevision,
    realization_revision: RealizationRevision,
    plan: CoupledFieldwiseRealizationPlan,
}

impl CoupledFieldwiseRealizationRequest {
    /// Bind one explicit plan to exact semantic and realization revisions.
    #[must_use]
    pub const fn explicit(
        model: OntologyId<Model>,
        semantic_revision: SemanticRevision,
        realization_revision: RealizationRevision,
        plan: CoupledFieldwiseRealizationPlan,
    ) -> Self {
        Self {
            model,
            semantic_revision,
            realization_revision,
            plan,
        }
    }

    /// Semantic Model identity.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.model
    }

    /// Semantic Model revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        self.semantic_revision
    }

    /// Independent Realization revision.
    #[must_use]
    pub const fn realization_revision(&self) -> RealizationRevision {
        self.realization_revision
    }

    /// Explicit unresolved plan.
    #[must_use]
    pub const fn plan(&self) -> &CoupledFieldwiseRealizationPlan {
        &self.plan
    }
}

/// Validated coupled plan and its two-layer provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedCoupledFieldwiseRealization {
    model: OntologyId<Model>,
    semantic_revision: SemanticRevision,
    realization_revision: RealizationRevision,
    requirements: CoupledFieldwiseRealizationRequirements,
    plan: CoupledFieldwiseRealizationPlan,
}

impl ResolvedCoupledFieldwiseRealization {
    /// Semantic Model identity.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.model
    }

    /// Semantic Model revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        self.semantic_revision
    }

    /// Independently selected Realization revision.
    #[must_use]
    pub const fn realization_revision(&self) -> RealizationRevision {
        self.realization_revision
    }

    /// Exact lowerer facts used during admission.
    #[must_use]
    pub const fn requirements(&self) -> &CoupledFieldwiseRealizationRequirements {
        &self.requirements
    }

    /// Complete validated plan.
    #[must_use]
    pub const fn plan(&self) -> &CoupledFieldwiseRealizationPlan {
        &self.plan
    }
}

/// Resolve one explicit coupled Field-wise request without fallback.
///
/// # Errors
/// Returns `EQ0807` for exact Domain, Field, or Connection drift, an invalid
/// plan, or an unsupported complete backend capability.
pub fn resolve_coupled_fieldwise(
    request: &CoupledFieldwiseRealizationRequest,
    requirements: CoupledFieldwiseRealizationRequirements,
    capabilities: &RealizationCapabilities,
) -> Result<ResolvedCoupledFieldwiseRealization, Diagnostic> {
    request.plan.validate()?;
    let eliminated = request
        .plan
        .time_step()
        .eliminated_states()
        .iter()
        .map(|state| state.pair())
        .collect::<Vec<_>>();
    let represented = request.plan.represented_physical_fields()?;
    let selected_domains = request
        .plan
        .spatial()
        .domains()
        .iter()
        .map(|domain| {
            DomainFieldInventory::new(
                domain.domain(),
                represented
                    .iter()
                    .filter(|field| field.domain() == domain.domain())
                    .map(|field| field.field()),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if selected_domains != requirements.domains {
        return Err(invalid_realization(
            "coupled field-wise plan must bind the exact lowerer Domain and participating-Field inventory",
        ));
    }
    if request.plan.spatial().trace_quotients() != requirements.trace_quotients {
        return Err(invalid_realization(
            "coupled field-wise plan trace quotient differs from the exact lowerer Connection and Field pair",
        ));
    }
    if eliminated != requirements.eliminated_states {
        return Err(invalid_realization(
            "coupled field-wise plan Backward Euler state/rate pair differs from the exact lowerer requirement",
        ));
    }
    let quadrature = request.plan.spatial().discretization().quadrature();
    let required_dimension = requirements.execution.spatial_dimension();
    let quadrature_dimension = match quadrature {
        QuadraturePolicy::TriangleDuffyGaussLegendre { .. } => 2,
        QuadraturePolicy::SimplexDuffyGaussLegendre {
            spatial_dimension, ..
        } => spatial_dimension.get(),
        _ => {
            return Err(invalid_realization(
                "coupled field-wise plan has no admitted simplex Duffy quadrature",
            ));
        }
    };
    if quadrature_dimension != required_dimension.get() {
        return Err(invalid_realization(format!(
            "coupled field-wise quadrature dimension {quadrature_dimension} differs from required spatial dimension {required_dimension}",
        )));
    }
    capabilities.supports_coupled_fieldwise(requirements.execution, &request.plan)?;
    Ok(ResolvedCoupledFieldwiseRealization {
        model: request.model,
        semantic_revision: request.semantic_revision,
        realization_revision: request.realization_revision,
        requirements,
        plan: request.plan.clone(),
    })
}

#[cfg(test)]
#[path = "coupled_fieldwise_resolution/tests.rs"]
mod tests;
