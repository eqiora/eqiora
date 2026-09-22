//! The editor's declaration resolver does not yet bind nested finite members.
//! Keep their positions unknown instead of attaching an outer declaration's type.
use eqiora_lang::{Document, ExprKind, Item, SourceAstFactory, TextRange};

#[derive(Clone, Debug)]
pub(super) struct SourceRanges {
    pub excluded: Vec<TextRange>,
    pub references: Vec<(TextRange, String)>,
}

pub(super) fn collect(
    document: &Document,
    is_cancelled: &mut impl FnMut() -> bool,
) -> Option<SourceRanges> {
    if is_cancelled() {
        return None;
    }
    let mut ranges = Vec::new();
    let mut references = Vec::new();
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
        if !cancelled && let ExprKind::Name(name) = expression.kind() {
            references.push((expression.range(), name.clone()));
        }
    });
    if cancelled || is_cancelled() {
        return None;
    }
    references.sort_by_key(|(range, _)| (range.start(), range.end()));
    references.dedup();
    ranges.sort_by_key(|range| (range.start(), range.end()));
    let mut excluded: Vec<TextRange> = Vec::new();
    for range in ranges {
        if let Some(previous) = excluded.last_mut()
            && range.start() <= previous.end()
        {
            *previous = TextRange::new(previous.start(), previous.end().max(range.end()));
        } else {
            excluded.push(range);
        }
    }
    (!is_cancelled()).then_some(SourceRanges {
        excluded,
        references,
    })
}
