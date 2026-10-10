use crate::form_compiler::region::CompiledRegionForm;
use crate::spatial_expression::Coefficient;

#[cfg(test)]
mod tests;
use crate::canonical::{boundary_parent, relations_on};
use crate::form_compiler::region::RegionBoundaryLaw;
use eqiora_core::{Diagnostic, RawId, ValueType};
use eqiora_schema::kernel::{BoundarySide, DomainKind, KernelNode};
use eqiora_sem::KernelProgram;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Inventory<S: Coefficient> {
    pub(super) fields: BTreeMap<RawId, BTreeMap<RawId, RegionBoundaryLaw<S>>>,
    pub(super) dependencies: BTreeMap<RawId, BTreeSet<RawId>>,
}

pub(super) fn derive<S: Coefficient>(
    program: &KernelProgram,
    parent: RawId,
    dimension: usize,
    fields: &[(RawId, ValueType)],
    volume: &CompiledRegionForm<S>,
    interface_boundaries: &BTreeSet<RawId>,
    time_s: Option<f64>,
) -> Result<Inventory<S>, Diagnostic> {
    let mut boundaries = fields
        .iter()
        .map(|(field, _)| (*field, BTreeMap::new()))
        .collect::<BTreeMap<_, _>>();
    let mut sides = BTreeSet::new();
    let mut dependencies = BTreeMap::new();
    let geometry_backed = matches!(program.node(parent), Some(KernelNode::Domain(definition)) if matches!(definition.kind(), DomainKind::GeometryRegion { .. }));
    let mut boundary_count = 0;
    for node in program.nodes() {
        let KernelNode::Domain(domain) = node else {
            continue;
        };
        if boundary_parent(program, domain.id().erase()) != Some(parent) {
            continue;
        }
        match domain.kind() {
            DomainKind::CartesianBoundary { axis, side } if !geometry_backed => {
                if *axis >= dimension || !sides.insert((*axis, *side)) {
                    return Err(super::invalid(
                        "duplicate or invalid Cartesian boundary side",
                    ));
                }
            }
            DomainKind::GeometryBoundary { .. } if geometry_backed => {}
            _ => {
                return Err(super::invalid(
                    "linear block boundary has incompatible support",
                ));
            }
        }
        boundary_count += 1;
        if interface_boundaries.contains(&domain.id().erase()) {
            continue;
        }
        let mut covered = BTreeSet::new();
        for relation in relations_on(program, domain.id().erase()) {
            for law in volume.boundary_laws(program, domain.id().erase(), relation, time_s)? {
                let field = law.tested;
                dependencies.insert(relation, law.dependencies.clone());
                if !covered.insert(field) {
                    return Err(super::invalid("duplicate Field boundary law"));
                }
                boundaries
                    .entry(field)
                    .or_insert_with(BTreeMap::new)
                    .insert(domain.id().erase(), law);
            }
        }
        if covered.len() != fields.len() {
            return Err(super::invalid(
                "every unknown Field requires complete boundary law coverage",
            ));
        }
    }
    // Geometry selections may group several facets, and need not describe a box.
    // This owner checks laws on declared supports. The resource binding must
    // additionally prove exact, disjoint coverage of the physical frontier.
    if geometry_backed && boundary_count == 0 {
        return Err(super::invalid(
            "linear block requires declared Geometry boundary supports",
        ));
    }
    if !geometry_backed
        && (boundary_count != 2 * dimension
            || (0..dimension).any(|axis| {
                [BoundarySide::Lower, BoundarySide::Upper]
                    .iter()
                    .any(|side| !sides.contains(&(axis, *side)))
            }))
    {
        return Err(super::invalid(
            "linear block requires every Cartesian boundary side",
        ));
    }
    Ok(Inventory {
        fields: boundaries,
        dependencies,
    })
}
