//! The editor's declaration resolver does not yet bind nested finite members.
//! Keep their positions unknown instead of attaching an outer declaration's type.
use eqiora_lang::{Document, ExprKind, Item, SourceAstFactory, TextRange};

pub(super) fn excluded(
    document: &Document,
    is_cancelled: &mut impl FnMut() -> bool,
) -> Option<Vec<TextRange>> {
    if is_cancelled() {
        return None;
    }
    let mut ranges = Vec::new();
    for model in document.models() {
        for item in model.items() {
            match item {
                Item::RelationFamily(value) => ranges.push(value.range()),
                Item::Instance(value) if value.family().is_some() => ranges.push(value.range()),
                Item::Connection(value) if value.binder().is_some() => ranges.push(value.range()),
                Item::BoundaryConnection(value) => ranges.push(value.range()),
                _ => {}
            }
        }
    }
    // Reuse the compiler's complete expression traversal. The temporary syntax
    // copy is made once per source during preparation, never for a query.
    let mut document = document.clone();
    let mut cancelled = false;
    SourceAstFactory::visit_expressions(&mut document, |_, expression| {
        cancelled |= is_cancelled();
        if !cancelled && matches!(expression.kind(), ExprKind::Reduction { .. }) {
            ranges.push(expression.range());
        }
    });
    (!cancelled && !is_cancelled()).then_some(ranges)
}
