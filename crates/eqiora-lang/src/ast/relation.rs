//! Ordered source equalities and their activation/support owner.

use super::{Expr, FamilyBinderSyntax, TextRange};
use eqiora_schema::kernel::RelationConditionKind;

/// Simultaneous mathematical conditions used only for fresh initialization.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) equations: Vec<RelationCondition>,
    pub(crate) range: TextRange,
}

impl InitialDecl {
    /// Conditions in authored order; their semantics are simultaneous.
    #[must_use]
    pub fn equations(&self) -> &[RelationCondition] {
        &self.equations
    }

    /// Full initialization-block range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Relation declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) activation: ActivationSyntax,
    pub(crate) domain: Option<String>,
    pub(crate) body: RelationBody,
    pub(crate) range: TextRange,
}

impl RelationDecl {
    /// Source name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Continuous or named clock/event activation.
    #[must_use]
    pub const fn activation(&self) -> &ActivationSyntax {
        &self.activation
    }

    /// Domain on which the equations hold, for a spatial Relation.
    #[must_use]
    pub fn domain(&self) -> Option<&str> {
        self.domain.as_deref()
    }

    /// Exclusive authored mathematical body.
    #[must_use]
    pub const fn body(&self) -> &RelationBody {
        &self.body
    }

    /// Ordered conditions, absent when this declaration is a physical Law.
    #[must_use]
    pub fn conditions(&self) -> Option<&[RelationCondition]> {
        match &self.body {
            RelationBody::Conditions(conditions) => Some(conditions),
            RelationBody::Conservation(_) => None,
        }
    }

    /// Every authored expression, including all retained physical Law terms.
    pub fn expressions(&self) -> impl Iterator<Item = &Expr> {
        let conditions = self
            .conditions()
            .into_iter()
            .flatten()
            .flat_map(|condition| [condition.left(), condition.right()]);
        let law = match &self.body {
            RelationBody::Conservation(law) => Some(law),
            _ => None,
        };
        conditions.chain(law.into_iter().flat_map(|law| [law.flux(), law.source()]))
    }

    /// Full declaration range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Exclusive mathematical syntax of a Relation declaration.
#[derive(Debug, Clone, PartialEq)]
pub enum RelationBody {
    /// Ordered equalities, inequalities and complementarity conditions.
    Conditions(Vec<RelationCondition>),
    /// Physical storage, flux and source terms on one domain.
    Conservation(Box<super::ConservationSyntax>),
}

/// One ordered authored condition, retaining source ranges and explicit predicates.
/// Residual construction and nonnegativity normalization belong to checked lowering.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationCondition {
    pub(crate) kind: RelationConditionKind,
    pub(crate) left: Expr,
    pub(crate) right: Expr,
    pub(crate) range: TextRange,
}

impl RelationCondition {
    /// Mathematical kind, distinct from Boolean expression syntax.
    #[must_use]
    pub const fn kind(&self) -> RelationConditionKind {
        self.kind
    }

    /// Authored left-hand expression.
    #[must_use]
    pub const fn left(&self) -> &Expr {
        &self.left
    }

    /// Authored right-hand expression, including an explicitly written zero.
    #[must_use]
    pub const fn right(&self) -> &Expr {
        &self.right
    }

    /// Equality range excluding the terminating semicolon.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// One Relation expanded once per member of an exact finite set.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationFamilyDecl {
    pub(crate) relation: RelationDecl,
    pub(crate) binder: FamilyBinderSyntax,
}

impl RelationFamilyDecl {
    /// Underlying Relation declaration, including support and activation.
    #[must_use]
    pub const fn relation(&self) -> &RelationDecl {
        &self.relation
    }

    /// Restricted boundary-member binder.
    #[must_use]
    pub const fn binder(&self) -> &FamilyBinderSyntax {
        &self.binder
    }

    /// Full family declaration range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.relation.range
    }
}

/// Source activation syntax.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ActivationSyntax {
    /// Active throughout model time.
    Continuous,
    /// Named clock or event activation, classified by semantic name resolution.
    Named(String),
}
