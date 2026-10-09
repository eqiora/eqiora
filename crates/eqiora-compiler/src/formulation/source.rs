//! Model and Component forms share one read-only compilation interface.
use eqiora_lang::{ComponentDecl, Expr, FormulationBinding, ModelDecl, TextRange};

pub(crate) trait FormulationSource {
    fn formulations(
        &self,
    ) -> impl ExactSizeIterator<Item = (&str, &[String], &[(Expr, Expr)], TextRange)>;
    fn formulation_binding(&self, name: &str) -> Option<&FormulationBinding>;
    fn formulation_gauge(&self, name: &str) -> Option<(&str, &[(Expr, Expr); 2])>;
    fn range(&self) -> TextRange;
    fn name(&self) -> &str;
}

macro_rules! source {
    ($owner:ty) => {
        impl FormulationSource for $owner {
            fn formulations(
                &self,
            ) -> impl ExactSizeIterator<Item = (&str, &[String], &[(Expr, Expr)], TextRange)> {
                <$owner>::formulations(self)
            }
            fn formulation_binding(&self, name: &str) -> Option<&FormulationBinding> {
                <$owner>::formulation_binding(self, name)
            }
            fn formulation_gauge(&self, name: &str) -> Option<(&str, &[(Expr, Expr); 2])> {
                <$owner>::formulation_gauge(self, name)
            }
            fn name(&self) -> &str {
                <$owner>::name(self)
            }
            fn range(&self) -> TextRange {
                <$owner>::range(self)
            }
        }
    };
}
source!(ComponentDecl);
source!(ModelDecl);
