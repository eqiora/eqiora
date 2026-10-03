//! Private recovered syntax for authored mathematical formulations.

use super::{ComponentDecl, Expr, TextRange};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FormulationDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) binding: FormulationBinding,
    pub(crate) relations: Vec<String>,
    pub(crate) equations: Vec<(Expr, Expr)>,
    pub(crate) gauge: Option<(String, [(Expr, Expr); 2])>,
    pub(crate) range: TextRange,
}

impl ComponentDecl {
    /// Authored mathematical formulations in source order.
    ///
    /// They are a compiler sidecar and never alter canonical Model identity.
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

    /// An explicit constant scalar gauge: its Field, reference equality and
    /// load-compatibility equality. This is mathematical Formulation meaning,
    /// not a numerical basis or an implicit repair of the Model equations.
    #[must_use]
    pub fn formulation_gauge(&self, name: &str) -> Option<(&str, &[(Expr, Expr); 2])> {
        self.formulations
            .iter()
            .find(|form| form.name == name)
            .and_then(|form| form.gauge.as_ref())
            .map(|(field, conditions)| (field.as_str(), conditions))
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
    /// Ordered global scalar Fields sharing one explicitly named finite coordinate space.
    Finite { name: String, trials: Vec<String> },
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
