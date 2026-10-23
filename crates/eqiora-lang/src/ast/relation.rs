//! Ordered source equalities and their activation/support owner.

use super::{Expr, FamilyBinderSyntax, TextRange};
use eqiora_schema::kernel::RelationConditionKind;

/// Simultaneous mathematical conditions used only for fresh initialization.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialDecl {
    pub(crate) domain: Option<String>,
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) equations: Vec<RelationCondition>,
    pub(crate) range: TextRange,
}

impl InitialDecl {
    /// Optional spatial support of every initialization equation in this block.
    #[must_use]
    pub fn domain(&self) -> Option<&str> {
        self.domain.as_deref()
    }

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
    pub(crate) activation_name_range: Option<TextRange>,
    pub(crate) range: TextRange,
}

impl RelationDecl {
    /// Exact authored activation-name token; absent without an assertion or source tokens.
    #[must_use]
    pub const fn activation_name_range(&self) -> Option<TextRange> {
        self.activation_name_range
    }

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
            RelationBody::Conservation(_)
            | RelationBody::Noise { .. }
            | RelationBody::ProbabilityMass { .. }
            | RelationBody::ProbabilityDensity { .. }
            | RelationBody::Stochastic { .. } => None,
        }
    }

    /// Every authored expression, including all retained physical Law terms.
    pub fn expressions(&self) -> impl Iterator<Item = &Expr> {
        let roots: Vec<_> = match &self.body {
            RelationBody::Conditions(conditions) => conditions
                .iter()
                .flat_map(|condition| [condition.left(), condition.right()])
                .collect(),
            RelationBody::Conservation(law) => law
                .storage()
                .into_iter()
                .chain([law.flux(), law.source()])
                .collect(),
            RelationBody::Noise { parameter, .. } => vec![parameter.as_ref()],
            RelationBody::ProbabilityMass { weights, .. } => vec![weights.as_ref()],
            RelationBody::ProbabilityDensity { density, .. } => vec![density.as_ref()],
            RelationBody::Stochastic {
                drift,
                diffusions,
                jumps,
                ..
            } => drift
                .as_deref()
                .into_iter()
                .chain(diffusions.iter().chain(jumps).map(|(_, value)| value))
                .collect(),
        };
        roots.into_iter()
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
    /// Normalized mass on an exact nominal enumeration.
    ProbabilityMass {
        /// Named outcome type; it must resolve to an Enum declaration.
        outcomes: super::ValueTypeSyntax,
        /// Nonnegative proportional weights in declaration order.
        weights: Box<Expr>,
    },
    /// Normalized density relative to an explicit measure.
    ProbabilityDensity {
        /// Exact support of the reference measure.
        domain: String,
        /// Explicit specialized measure, or the Domain's ordinary measure.
        measure: Option<eqiora_schema::kernel::ObservableMeasure>,
        /// Nonnegative proportional density with inverse-measure units.
        density: Box<Expr>,
    },
    /// Ordered equalities, inequalities and complementarity conditions.
    Conditions(Vec<RelationCondition>),
    /// Physical storage, flux and source terms on one domain.
    Conservation(Box<super::ConservationSyntax>),
    /// Nominal process law with no random evaluation operation.
    Noise {
        /// Mathematical increment law.
        process: eqiora_schema::kernel::NoiseProcess,
        /// Covariance or intensity expression; never a sampling operation.
        parameter: Box<Expr>,
        /// Explicit independence from other nominal groups.
        independent: bool,
    },
    /// Owned stochastic evolution through pure coefficient expressions.
    Stochastic {
        /// Exact State selected in the owning lexical scope.
        target: String,
        /// Written continuous integral interpretation, absent for pure jumps.
        calculus: Option<eqiora_schema::kernel::StochasticCalculus>,
        /// Ordinary pure drift expression.
        drift: Option<Box<Expr>>,
        /// Wiener group names with their authored coefficients.
        diffusions: Vec<(String, Expr)>,
        /// Counting group names with their authored State increments.
        jumps: Vec<(String, Expr)>,
    },
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
