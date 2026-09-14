//! Private recovered syntax for authored mathematical formulations.

use super::{ComponentDecl, Expr, TextRange};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FormulationDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) binding: FormulationBinding,
    pub(crate) relations: Vec<String>,
    pub(crate) equations: Vec<(Expr, Expr)>,
    pub(crate) range: TextRange,
}

impl ComponentDecl {
    /// Authored mathematical formulations in source order.
    ///
    /// They are a compiler sidecar and never alter canonical Model identity.
    #[must_use]
    pub fn formulations(
        &self,
    ) -> impl ExactSizeIterator<Item = (&str, &[String], &[(Expr, Expr)], TextRange)> {
        self.formulations.iter().map(|form| {
            (
                form.name.as_str(),
                form.relations.as_slice(),
                form.equations.as_slice(),
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

/// Mathematical variables introduced by an authored formulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormulationBinding {
    /// Ordered tests with exact trials and optional zero-trace boundary restrictions.
    WeakTests {
        tests: Vec<(String, String, Vec<String>)>,
    },
    /// Every ordered mathematical interval within the named parent support.
    Interval {
        name: String,
        lower: String,
        upper: String,
        domain: String,
    },
}
