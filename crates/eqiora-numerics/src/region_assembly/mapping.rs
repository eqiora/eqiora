//! Exact Field/entity ownership between local bases and constrained global vectors.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_assembly::{AssemblyMap, DofId};
use eqiora_core::{Diagnostic, DynQuantity, RawId};
use eqiora_meshing::{EntityIncidence, MeshEntity, MeshTopology, ReferenceCell, ReferenceTopology};
use eqiora_realization::{ConformingTraceQuotient, DomainFieldDiscretization};
use eqiora_schema::kernel::KernelNode;
use eqiora_sem::KernelProgram;

use crate::constrained_dofs::ConstrainedDofLayout;
use crate::form_compiler::region::{RegionFieldLayout, basis, components};

use super::invalid;

mod binding;
mod recovery;
pub(crate) use binding::bind_region_topology;

/// Bind topology without requiring an affine or linear equation compiler.
pub(crate) fn field_layouts(
    program: &KernelProgram,
    domains: &[DomainFieldDiscretization],
    reference: ReferenceCell,
    scales: &BTreeMap<RawId, DynQuantity>,
) -> Result<BTreeMap<RawId, Vec<RegionFieldLayout>>, Diagnostic> {
    let mut result = BTreeMap::new();
    let mut seen_fields = BTreeSet::new();
    for domain in domains {
        let admitted = crate::canonical::continuum_fields_on(program, domain.domain().erase());
        let mut offset = 0usize;
        let mut layouts = Vec::new();
        for binding in domain.field_spaces() {
            let field = binding.field().erase();
            let Some(KernelNode::Field(definition)) = program.node(field) else {
                return Err(invalid("mapping requires an exact Model Field"));
            };
            if !admitted.contains(&field) || !seen_fields.insert(field) {
                return Err(invalid("mapping Field support is absent or duplicated"));
            }
            let value_type = definition.value_type();
            let scale = scales
                .get(&field)
                .ok_or_else(|| invalid("mapping Field has no exact physical scale"))?;
            if scale.dim() != value_type.dimension()
                || !scale.value().is_finite()
                || scale.value() <= 0.0
            {
                return Err(invalid("mapping Field scale differs from exact ValueType"));
            }
            let count = components(value_type, reference.dimension())?;
            let local = basis(binding.space(), reference)?
                .local_dofs()
                .len()
                .checked_mul(count)
                .ok_or_else(|| invalid("mapping local DOF overflow"))?;
            let end = offset
                .checked_add(local)
                .ok_or_else(|| invalid("mapping local DOF overflow"))?;
            layouts.push(RegionFieldLayout {
                field,
                value_type: value_type.clone(),
                space: binding.space(),
                range: offset..end,
                components: count,
                scale: scale.value(),
            });
            offset = end;
        }
        if result.insert(domain.domain().erase(), layouts).is_some() {
            return Err(invalid("mapping requires unique exact Domain bindings"));
        }
    }
    if scales.keys().copied().collect::<BTreeSet<_>>() != seen_fields {
        return Err(invalid(
            "mapping scale inventory differs from algebraic Fields",
        ));
    }
    Ok(result)
}

/// A scalar coordinate of one exact Field on one supporting mesh entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FieldDof {
    pub(crate) field: RawId,
    pub(crate) entity: MeshEntity,
    pub(crate) slot: usize,
    pub(crate) component: usize,
}

/// Oriented facet closure supplied by exact Connection admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TraceFacet {
    pub(crate) facet: MeshEntity,
    /// Ordered like the quotient's exact Domain endpoints.
    pub(crate) sides: [EntityIncidence; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TraceBinding {
    pub(crate) quotient: ConformingTraceQuotient,
    pub(crate) facets: Vec<TraceFacet>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegionDofMap {
    globals: BTreeMap<FieldDof, usize>,
    cells: Vec<Vec<usize>>,
    constraints: ConstrainedDofLayout,
    full_count: usize,
    fields: BTreeMap<RawId, (RawId, RegionFieldLayout)>,
}

impl RegionDofMap {
    /// Membership and Connections are already semantically admitted. This boundary
    /// rechecks exact local layout, complete mesh coverage and oriented trace closure.
    pub(crate) fn new(
        mesh: &dyn MeshTopology,
        layouts: &BTreeMap<RawId, Vec<RegionFieldLayout>>,
        reference: ReferenceCell,
        cell_domains: &[RawId],
        traces: &[TraceBinding],
        prescribed: &BTreeMap<FieldDof, f64>,
    ) -> Result<Self, Diagnostic> {
        let dimension = mesh.topological_dimension();
        if dimension == 0
            || mesh.entity_count(dimension) != Some(cell_domains.len())
            || cell_domains.is_empty()
            || layouts.keys().copied().collect::<BTreeSet<_>>()
                != cell_domains.iter().copied().collect::<BTreeSet<_>>()
        {
            return Err(invalid(
                "global mapping requires exact nonempty Region/cell coverage",
            ));
        }
        let mut fields = BTreeMap::new();
        for (&domain, region_fields) in layouts {
            if reference.dimension() != dimension {
                return Err(invalid(
                    "global mapping Region and reference identities differ",
                ));
            }
            if region_fields.is_empty() {
                return Err(invalid("mapping Region has no algebraic Fields"));
            }
            for layout in region_fields {
                if layout.components != components(&layout.value_type, dimension)?
                    || !layout.scale.is_finite()
                    || layout.scale <= 0.0
                {
                    return Err(invalid("mapping Field type, components or scale is stale"));
                }
                if fields.insert(layout.field, (domain, layout)).is_some() {
                    return Err(invalid("one algebraic Field has multiple Region bindings"));
                }
            }
        }
        let reference_topology = ReferenceTopology::new(reference)?;
        let mut local_keys = Vec::new();
        let mut inventory = BTreeSet::new();
        for (index, domain) in cell_domains.iter().enumerate() {
            let region_fields = &layouts[domain];
            let cell = MeshEntity::new(dimension, index);
            for stratum in 0..dimension {
                if mesh.incidence(cell, stratum).map(|entities| entities.len())
                    != reference_topology.entity_count(stratum)
                {
                    return Err(invalid(
                        "mapping cell closure differs from the bound reference",
                    ));
                }
            }
            let mut keys = Vec::new();
            for layout in region_fields {
                if keys.len() != layout.range.start {
                    return Err(invalid(
                        "bound Field ranges do not match local basis ordering",
                    ));
                }
                for local in basis(layout.space, reference)?.local_dofs() {
                    let entity = if local.entity_dimension() == dimension {
                        if local.entity_ordinal() != 0 {
                            return Err(invalid(
                                "cell-interior basis has an invalid entity ordinal",
                            ));
                        }
                        cell
                    } else {
                        mesh.incidence(cell, local.entity_dimension())
                            .and_then(|entries| entries.get(local.entity_ordinal()).copied())
                            .ok_or_else(|| {
                                invalid("local basis support is absent from mesh incidence")
                            })?
                            .entity
                    };
                    for component in 0..layout.components {
                        keys.push(FieldDof {
                            field: layout.field,
                            entity,
                            slot: local.slot(),
                            component,
                        });
                    }
                }
                if keys.len() != layout.range.end {
                    return Err(invalid(
                        "bound Field layout differs from its topological DOFs",
                    ));
                }
            }
            inventory.extend(keys.iter().copied());
            local_keys.push(keys);
        }
        // Sorted exact identity, rather than declaration or supplied Region order,
        // determines representatives and global algebraic numbering.
        let mut representatives = inventory
            .iter()
            .map(|key| (*key, *key))
            .collect::<BTreeMap<_, _>>();
        let mut connections = BTreeSet::new();
        let mut covered_facets = BTreeSet::new();
        for trace in traces {
            let trace_key = (
                trace.quotient.connection().erase(),
                trace
                    .quotient
                    .endpoints()
                    .map(|endpoint| endpoint.field().erase()),
            );
            if !connections.insert(trace_key) || trace.facets.is_empty() {
                return Err(invalid(
                    "trace quotient requires unique Field pair and nonempty coverage",
                ));
            }
            let endpoints = trace.quotient.endpoints();
            let [(left_domain, left), (right_domain, right)] = endpoints
                .map(|endpoint| fields.get(&endpoint.field().erase()).copied())
                .map(|entry| {
                    entry.ok_or_else(|| invalid("trace quotient references a missing Field"))
                })
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| invalid("trace quotient requires two endpoints"))?;
            if [left_domain, right_domain] != endpoints.map(|endpoint| endpoint.domain().erase())
                || left.value_type != right.value_type
                || left.components != right.components
                || left.scale != right.scale
            {
                return Err(invalid(
                    "trace quotient Field support, type, components or scales differ",
                ));
            }
            let mut seen = BTreeSet::new();
            for witness in &trace.facets {
                if witness.facet.dimension() + 1 != dimension || !seen.insert(witness.facet) {
                    return Err(invalid(
                        "trace quotient facet dimension or duplicate coverage is invalid",
                    ));
                }
                let actual = mesh
                    .incidence(witness.facet, dimension)
                    .ok_or_else(|| invalid("trace facet has no exact mesh incidence"))?;
                if actual.len() != 2
                    || witness.sides.iter().any(|side| !actual.contains(side))
                    || witness.sides[0].entity == witness.sides[1].entity
                    || witness.sides.iter().zip([left_domain, right_domain]).any(
                        |(side, domain)| cell_domains.get(side.entity.index()) != Some(&domain),
                    )
                {
                    return Err(invalid(
                        "trace quotient orientation or Region ownership is stale",
                    ));
                }
                covered_facets.insert(witness.facet);
                // Conforming P1/Q1/MINI traces have vertex-supported DOFs only.
                // Cell bubbles never enter the quotient, even on interface cells.
                let vertices = if dimension == 1 {
                    vec![witness.facet]
                } else {
                    mesh.incidence(witness.facet, 0)
                        .ok_or_else(|| invalid("trace has no vertex closure"))?
                        .into_iter()
                        .map(|entry| entry.entity)
                        .collect()
                };
                for entity in vertices {
                    for component in 0..left.components {
                        let a = FieldDof {
                            field: left.field,
                            entity,
                            slot: 0,
                            component,
                        };
                        let b = FieldDof {
                            field: right.field,
                            entity,
                            slot: 0,
                            component,
                        };
                        let first = representative(&representatives, a)?;
                        let second = representative(&representatives, b)?;
                        representatives.insert(first.max(second), first.min(second));
                    }
                }
            }
        }
        for index in 0..mesh.entity_count(dimension - 1).unwrap_or(0) {
            let facet = MeshEntity::new(dimension - 1, index);
            let sides = mesh
                .incidence(facet, dimension)
                .ok_or_else(|| invalid("mesh facet incidence is missing"))?;
            if sides.len() == 2
                && cell_domains[sides[0].entity.index()] != cell_domains[sides[1].entity.index()]
                && !covered_facets.contains(&facet)
            {
                return Err(invalid(
                    "cross-Region facet lacks complete trace quotient coverage",
                ));
            }
        }
        let roots = inventory
            .iter()
            .map(|key| representative(&representatives, *key))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let indices = roots
            .into_iter()
            .enumerate()
            .map(|(index, key)| (key, index))
            .collect::<BTreeMap<_, _>>();
        let globals = inventory
            .iter()
            .map(|key| Ok((*key, indices[&representative(&representatives, *key)?])))
            .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
        let mut fixed = vec![None; indices.len()];
        for (key, &physical_value) in prescribed {
            let value = physical_value
                / fields
                    .get(&key.field)
                    .ok_or_else(|| invalid("constraint references an absent Field"))?
                    .1
                    .scale;
            let index = *globals
                .get(key)
                .ok_or_else(|| invalid("constraint references a missing exact Field DOF"))?;
            if !value.is_finite() || fixed[index].is_some_and(|old| old != value) {
                return Err(invalid(
                    "quotiented Field constraints are nonfinite or inconsistent",
                ));
            }
            fixed[index] = Some(value);
        }
        let cells = local_keys
            .iter()
            .map(|keys| keys.iter().map(|key| globals[key]).collect())
            .collect();
        Ok(Self {
            globals,
            cells,
            constraints: ConstrainedDofLayout::new(fixed)?,
            full_count: indices.len(),
            fields: fields
                .into_iter()
                .map(|(field, (domain, layout))| (field, (domain, layout.clone())))
                .collect(),
        })
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = FieldDof> + '_ {
        self.globals.keys().copied()
    }

    pub(crate) fn field_scale(&self, field: RawId) -> Result<f64, Diagnostic> {
        self.fields
            .get(&field)
            .map(|(_, layout)| layout.scale)
            .ok_or_else(|| invalid("scale query requires an exact mapped Field"))
    }

    /// Replace essential values without rebuilding topology or quotient identity.
    pub(crate) fn with_prescribed(
        &self,
        prescribed: &BTreeMap<FieldDof, f64>,
    ) -> Result<Self, Diagnostic> {
        let mut fixed = vec![None; self.full_count];
        for (key, physical) in prescribed {
            let index = self
                .global_dof(*key)
                .ok_or_else(|| invalid("constraint references a missing exact Field DOF"))?;
            let value = physical / self.field_scale(key.field)?;
            if !value.is_finite() || fixed[index].is_some_and(|old| old != value) {
                return Err(invalid(
                    "quotiented Field constraints are nonfinite or inconsistent",
                ));
            }
            fixed[index] = Some(value);
        }
        let mut result = self.clone();
        result.constraints = ConstrainedDofLayout::new(fixed)?;
        Ok(result)
    }

    pub(crate) fn map_dofs(
        &self,
        keys: &[FieldDof],
        reduced: bool,
    ) -> Result<AssemblyMap, Diagnostic> {
        let globals = keys
            .iter()
            .map(|key| {
                self.global_dof(*key)
                    .ok_or_else(|| invalid("local map references an absent exact Field DOF"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if reduced {
            self.constraints.reduced_map(&globals)
        } else {
            self.constraints.full_map(&globals)
        }
    }

    pub(crate) fn lift(&self, reduced: &[f64], direction: bool) -> Result<Vec<f64>, Diagnostic> {
        if reduced.iter().any(|value| !value.is_finite()) {
            return Err(invalid("algebraic recovery requires finite values"));
        }
        if direction {
            self.constraints.lift_direction(reduced)
        } else {
            self.constraints.lift(reduced)
        }
    }

    pub(crate) fn restrict(&self, full: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        self.constraints.restrict(full)
    }

    pub(crate) fn cell_map(&self, cell: usize, reduced: bool) -> Result<AssemblyMap, Diagnostic> {
        let globals = self
            .cells
            .get(cell)
            .ok_or_else(|| invalid("cell is outside global Field mapping"))?;
        if reduced {
            self.constraints.reduced_map(globals)
        } else {
            self.constraints.full_map(globals)
        }
    }

    pub(crate) fn full_count(&self) -> usize {
        self.full_count
    }
    pub(crate) fn free_count(&self) -> usize {
        self.constraints.free_count()
    }
    pub(crate) fn global_dof(&self, key: FieldDof) -> Option<usize> {
        self.globals.get(&key).copied()
    }
    pub(crate) fn free_dof(&self, key: FieldDof) -> Option<DofId> {
        self.globals
            .get(&key)
            .and_then(|index| self.constraints.free_index(*index))
    }

    /// Exact Field block membership, including shared quotient coordinates.
    pub(crate) fn field_free_dofs(&self, field: RawId) -> Result<Vec<DofId>, Diagnostic> {
        if !self.fields.contains_key(&field) {
            return Err(invalid("requested algebraic block has no exact Field"));
        }
        let indices = self
            .globals
            .iter()
            .filter(|(key, _)| key.field == field)
            .filter_map(|(_, index)| self.constraints.free_index(*index))
            .map(|dof| dof.index())
            .collect::<BTreeSet<_>>();
        Ok(indices.into_iter().map(DofId::new).collect())
    }
}

fn representative(
    parents: &BTreeMap<FieldDof, FieldDof>,
    mut key: FieldDof,
) -> Result<FieldDof, Diagnostic> {
    loop {
        let parent = *parents
            .get(&key)
            .ok_or_else(|| invalid("trace references an absent topological Field DOF"))?;
        if parent == key {
            return Ok(key);
        }
        key = parent;
    }
}

#[cfg(test)]
mod tests;
