//! Private recovered syntax for authored mathematical formulations.

use super::{ComponentDecl, Expr, TextRange};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FormulationDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) test: String,
    pub(crate) trial: String,
    pub(crate) zero_on: Vec<String>,
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

    /// Named test, trial and exact zero-trace support names for one authored form.
    #[must_use]
    pub fn formulation_test(&self, name: &str) -> Option<(&str, &str, &[String])> {
        self.formulations
            .iter()
            .find(|form| form.name == name)
            .map(|form| {
                (
                    form.test.as_str(),
                    form.trial.as_str(),
                    form.zero_on.as_slice(),
                )
            })
    }

    /// Full component declaration range, including a visibility modifier.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}
