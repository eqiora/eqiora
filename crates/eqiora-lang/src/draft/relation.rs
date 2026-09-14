//! Native authored equation groups.
use super::{DraftExpression, DraftSpatialDomain};

/// Immutable continuous implicit Relation declaration.
#[derive(Debug, Clone)]
pub struct DraftRelation {
    pub(super) name: String,
    pub(super) domain: Option<DraftSpatialDomain>,
    pub(super) equations: Vec<(DraftExpression, DraftExpression)>,
}

impl DraftRelation {
    /// Declare ordered left and right sides of simultaneous equations.
    #[must_use]
    pub fn continuous(
        name: impl Into<String>,
        equations: impl IntoIterator<Item = (DraftExpression, DraftExpression)>,
    ) -> Self {
        Self {
            name: name.into(),
            domain: None,
            equations: equations.into_iter().collect(),
        }
    }

    /// Declare continuous equations on one exact draft-local spatial Domain.
    #[must_use]
    pub fn continuous_on(
        name: impl Into<String>,
        domain: &DraftSpatialDomain,
        equations: impl IntoIterator<Item = (DraftExpression, DraftExpression)>,
    ) -> Self {
        Self {
            name: name.into(),
            domain: Some(domain.clone()),
            equations: equations.into_iter().collect(),
        }
    }

    /// Declaration name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// RelationCondition sides in declared order.
    #[must_use]
    pub fn equations(&self) -> &[(DraftExpression, DraftExpression)] {
        &self.equations
    }

    /// Exact draft-local support Domain, when spatially scoped.
    #[must_use]
    pub const fn domain(&self) -> Option<&DraftSpatialDomain> {
        self.domain.as_ref()
    }
}
