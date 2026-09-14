//! Private recovered syntax for authored mathematical formulations.

use super::{ComponentDecl, Expr, TextRange};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FormulationDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) binding: FormulationBinding,
    pub(crate) relation: String,
    pub(crate) left: Expr,
    pub(crate) right: Expr,
    pub(crate) range: TextRange,
}

impl ComponentDecl {
    /// Authored mathematical formulations in source order.
    ///
    /// They are a compiler sidecar and never alter canonical Model identity.
    #[must_use]
    pub fn formulations(
        &self,
    ) -> impl ExactSizeIterator<Item = (&str, &str, &Expr, &Expr, TextRange)> {
        self.formulations.iter().map(|form| {
            (
                form.name.as_str(),
                form.relation.as_str(),
                &form.left,
                &form.right,
                form.range,
            )
        })
    }

    /// Exact mathematical binder owned by one authored form.
    #[must_use]
    pub fn formulation_binding(&self, name: &str) -> Option<&FormulationBinding> {
        self.formulations
            .iter()
            .find(|form| form.name == name)
            .map(|form| &form.binding)
    }

    /// Full component declaration range, including a visibility modifier.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Mathematical variables introduced by an authored scalar formulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormulationBinding {
    /// An admissible test with an exact trial and zero-trace boundary restriction.
    WeakTest {
        name: String,
        trial: String,
        zero_on: Vec<String>,
    },
    /// Every ordered mathematical interval within the named parent support.
    Interval {
        name: String,
        lower: String,
        upper: String,
        domain: String,
    },
}
