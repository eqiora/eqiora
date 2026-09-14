//! Resolve authored test restrictions through already admitted exact support bindings.
use super::*;
use crate::external::ExternalGeometrySupportBinding;
use eqiora_schema::kernel::DomainKind;

pub(super) fn resolve(
    file: &str,
    restriction: (TextRange, &[String]),
    parent: RawId,
    symbols: &ModelSymbols,
    index: &KernelIndex<'_>,
    supports: &[ExternalGeometrySupportBinding],
) -> Result<Vec<String>, Diagnostic> {
    let (range, names) = restriction;
    let mut boundaries = std::collections::BTreeSet::new();
    for name in names {
        let ids = if let Some(ExternalGeometrySupportBinding::CompleteExterior { parent_slot, members, .. }) = supports.iter().find(|binding| {
            matches!(binding, ExternalGeometrySupportBinding::CompleteExterior { slot, .. } if slot == name)
        }) {
            if symbols.get(parent_slot) != Some(parent) {
                return Err(error(file, range, "test boundary set has a foreign parent"));
            }
            members.iter().map(|member| {
                let matches = index.nodes.iter().filter_map(|(id, node)| {
                    matches!(node, KernelNode::Domain(domain) if matches!(domain.kind(), DomainKind::GeometryBoundary { entity_set } if entity_set == &member.entity_set))
                        .then_some(*id)
                        .filter(|id| index.boundary_of.get(id) == Some(&parent))
                }).collect::<Vec<_>>();
                match matches.as_slice() {
                    [id] => Ok(*id),
                    _ => Err(error(file, range, "test boundary member has no unique exact Model identity")),
                }
            }).collect::<Result<Vec<_>, _>>()?
        } else {
            vec![resolve_symbol(file, range, name, symbols)?]
        };
        for id in ids {
            if index.boundary_of.get(&id) != Some(&parent) {
                return Err(error(
                    file,
                    range,
                    "zero_on requires a boundary of the exact trial Domain",
                ));
            }
            if !boundaries.insert(id.ulid().to_string()) {
                return Err(error(file, range, "zero_on repeats an exact boundary"));
            }
        }
    }
    if boundaries.is_empty() {
        return Err(error(
            file,
            range,
            "test requires a nonempty boundary restriction",
        ));
    }
    Ok(boundaries.into_iter().collect())
}
