//! The editor's declaration resolver does not yet bind nested finite members.
//! Keep their positions unknown instead of attaching an outer declaration's type.
use eqiora_lang::{
    ComponentItem, Document, ExprKind, Item, SignatureItem, SourceAstFactory, TextRange,
};

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
    for component in document.components() {
        for item in component.signature() {
            if let SignatureItem::PortFamily(value) = item {
                ranges.push(value.range());
            }
        }
        for item in component.items() {
            match item {
                ComponentItem::RelationFamily(value) => ranges.push(value.range()),
                ComponentItem::Instance(value) if value.family().is_some() => {
                    ranges.push(value.range())
                }
                ComponentItem::Connection(value) if value.binder().is_some() => {
                    ranges.push(value.range())
                }
                ComponentItem::BoundaryConnection(value) => ranges.push(value.range()),
                ComponentItem::PortFamily(value) => ranges.push(value.range()),
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
        if !cancelled {
            match expression.kind() {
                ExprKind::Name(name) => references.push((expression.range(), name.clone())),
                ExprKind::Path(path) => {
                    references.push((expression.range(), path.as_str().to_owned()))
                }
                _ => {}
            }
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

#[cfg(test)]
mod tests {
    #[test]
    fn component_signature_and_body_port_families_keep_binder_ranges_unknown() {
        let source = "component C(support exterior:complete_exterior(parent=body),support body:volume(ambient_dimension=2),port natural[boundary in exterior]:Scalar over boundary){port local[boundary in exterior]:Scalar over boundary;relation law[boundary in exterior] on boundary{natural[boundary=boundary].flux=0;}connect [boundary in exterior] natural[boundary=boundary],local[boundary=boundary];}";
        let document = eqiora_lang::parse("families.eqi", source)
            .into_document()
            .expect("family syntax");
        let ranges = super::collect(&document, &mut || false).unwrap();
        for needle in [
            "natural[boundary in",
            "local[boundary in",
            "natural[boundary=boundary].flux",
            "natural[boundary=boundary],",
        ] {
            let offset = source.find(needle).unwrap() as u32;
            assert!(
                ranges
                    .excluded
                    .iter()
                    .any(|range| range.start() <= offset && offset < range.end()),
                "{needle}"
            );
        }
        assert!(super::collect(&document, &mut || true).is_none());
    }
}
