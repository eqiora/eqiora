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

/// Coverage supplied only after exact interface trace and flux admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct InterfaceBoundary {
    pub(crate) boundary: RawId,
    pub(crate) field: RawId,
    /// Physical-interface laws live on their own support, not this Boundary.
    pub(crate) carrier: Option<RawId>,
}

pub(super) fn derive<S: Coefficient>(
    program: &KernelProgram,
    parent: RawId,
    dimension: usize,
    fields: &[(RawId, ValueType)],
    volume: &CompiledRegionForm<S>,
    interface_boundaries: &BTreeSet<InterfaceBoundary>,
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
        let mut covered = BTreeSet::new();
        let mut carriers = BTreeSet::new();
        let relations = relations_on(program, domain.id().erase());
        for interface in interface_boundaries
            .iter()
            .filter(|interface| interface.boundary == domain.id().erase())
        {
            if !fields.iter().any(|(field, _)| *field == interface.field)
                || !covered.insert(interface.field)
            {
                return Err(super::invalid(
                    "interface requires one exact local Field endpoint",
                ));
            }
            if let Some(carrier) = interface.carrier {
                if !relations.contains(&carrier) {
                    return Err(super::invalid(
                        "interface carrier is outside its exact Boundary",
                    ));
                }
                carriers.insert(carrier);
            }
        }
        for relation in relations {
            if carriers.contains(&relation) {
                continue;
            }
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

impl<S: Coefficient> super::CompiledLinearBlockForm<S> {
    /// Re-evaluate only existing exterior laws; interface coverage stays owned by admission.
    pub(crate) fn bind_boundary_time(
        &mut self,
        program: &KernelProgram,
        time_s: f64,
    ) -> Result<(), Diagnostic> {
        if !time_s.is_finite() {
            return Err(super::invalid(
                "boundary data requires finite physical Time",
            ));
        }
        for (field, laws) in &mut self.boundary_laws {
            for (boundary, law) in laws {
                let mut candidates = self
                    .volume
                    .boundary_laws(program, *boundary, law.binding.relation(), Some(time_s))?
                    .into_iter()
                    .filter(|candidate| candidate.tested == *field);
                let candidate = candidates
                    .next()
                    .ok_or_else(|| super::invalid("time binding lost an exact boundary row"))?;
                if candidates.next().is_some()
                    || candidate.trace_field != law.trace_field
                    || candidate.quantity != law.quantity
                    || candidate.dependencies != law.dependencies
                {
                    return Err(super::invalid("time binding changed an exact boundary row"));
                }
                *law = candidate;
            }
        }
        Ok(())
    }
}
