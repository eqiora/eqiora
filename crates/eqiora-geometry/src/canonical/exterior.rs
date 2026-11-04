//! Complete selected-region frontiers from admitted entity incidences.

use std::collections::{BTreeMap, BTreeSet};

use super::{CanonicalGeometryKind, CanonicalGeometryLimits, CanonicalGeometryV1, invalid};
use crate::{NamedEntitySet, PlanarRegion};
use eqiora_core::Diagnostic;

impl CanonicalGeometryV1 {
    /// Check that named selections partition the complete exterior of a region.
    ///
    /// All arguments must borrow selections from this exact object's catalog,
    /// as with [`Self::selection_is_boundary_of`]. Independently replayed or
    /// cloned objects have separate catalog references even when their canonical
    /// geometry identity is equal. The parent must be full-dimensional. Parts
    /// must be nonempty codimension-one sets with disjoint entity memberships;
    /// together they must cover every exterior entity, including unnamed ones.
    /// Part ordering and grouping do not affect acceptance.
    ///
    /// Interior facets shared by selected parents are excluded. Planar incidence
    /// uses exact vertex identities and opposite loop directions, never coordinate
    /// adjacency or Cartesian side labels. No mesh or numerical claim is made.
    ///
    /// `max_entity_sets` bounds parts, `max_entity_set_members` bounds all input
    /// memberships including the parent, `max_faces` bounds inspected parents,
    /// and `max_loop_indices` bounds inspected topology references. These bounds
    /// are checked before allocating the frontier or membership collections;
    /// other decoder limit fields do not apply to this borrowed query.
    ///
    /// # Errors
    /// Returns `EQ0901` for foreign selections, invalid dimensions, missing or
    /// repeated members, interior members, ambiguous incidence, or excess work.
    pub fn validate_complete_exterior_partition(
        &self,
        parent: &NamedEntitySet,
        parts: &[&NamedEntitySet],
        limits: CanonicalGeometryLimits,
    ) -> Result<(), Diagnostic> {
        if !self.owns_selection(parent)
            || parent.dimension() != self.topological_dimension()
            || parent.members().is_empty()
        {
            return Err(invalid(
                "exterior partition requires an owned full-dimensional parent",
            ));
        }
        if parts.is_empty() || parts.len() > limits.max_entity_sets {
            return Err(invalid(
                "exterior partition part count is empty or exceeds its limit",
            ));
        }
        let mut memberships = parent.members().len();
        bounded(memberships, limits.max_entity_set_members)?;
        for part in parts {
            if !self.owns_selection(part)
                || part.dimension().checked_add(1) != Some(parent.dimension())
                || part.members().is_empty()
            {
                return Err(invalid(
                    "exterior partition requires owned nonempty boundary parts",
                ));
            }
            memberships = checked_total(
                memberships,
                part.members().len(),
                limits.max_entity_set_members,
            )?;
        }
        let mut frontier = match &self.kind {
            CanonicalGeometryKind::StraightEdgedPlanarV1 { region, .. } => {
                planar_frontier(region, parent, limits)?
            }
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(geometry) => {
                planar_frontier(geometry.region(), parent, limits)?
            }
            CanonicalGeometryKind::ConvexPolyhedraV1(geometry) => {
                bounded(parent.members().len(), limits.max_faces)?;
                let mut references = 0;
                for &volume in parent.members() {
                    let facets = geometry
                        .volume_facets(volume)
                        .ok_or_else(|| invalid("exterior parent has no volume incidence"))?;
                    references = checked_total(references, facets.len(), limits.max_loop_indices)?;
                }
                // Accepted polyhedra already prove opposite incidences for each
                // shared canonical facet. Count selected incidences, not names.
                let mut counts = BTreeMap::<usize, usize>::new();
                for &volume in parent.members() {
                    let facets = geometry
                        .volume_facets(volume)
                        .ok_or_else(|| invalid("exterior parent has no volume incidence"))?;
                    for &facet in facets {
                        let count = counts.entry(facet).or_default();
                        *count += 1;
                        if *count > 2 {
                            return Err(invalid(
                                "exterior partition has ambiguous facet incidence",
                            ));
                        }
                    }
                }
                counts
                    .into_iter()
                    .filter_map(|(facet, count)| (count == 1).then_some(facet))
                    .collect()
            }
            CanonicalGeometryKind::CartesianBoxV1(_) => {
                fixed_frontier(2 * self.topological_dimension(), limits)?
            }
            CanonicalGeometryKind::PlanarRectangleV2(_) => fixed_frontier(4, limits)?,
            CanonicalGeometryKind::CircularHolePlanar(_) => fixed_frontier(5, limits)?,
        };
        if frontier.is_empty() {
            return Err(invalid("selected region has no unambiguous exterior"));
        }
        for part in parts {
            for member in part.members() {
                if !frontier.remove(member) {
                    return Err(invalid(
                        "exterior partition repeats an entity or names an entity outside the selected frontier",
                    ));
                }
            }
        }
        if !frontier.is_empty() {
            return Err(invalid(
                "exterior partition omits an entity of the selected frontier",
            ));
        }
        Ok(())
    }
}

fn bounded(value: usize, limit: usize) -> Result<(), Diagnostic> {
    if value > limit {
        return Err(invalid(
            "exterior partition exceeds its topology or membership work limit",
        ));
    }
    Ok(())
}

fn checked_total(total: usize, more: usize, limit: usize) -> Result<usize, Diagnostic> {
    let total = total
        .checked_add(more)
        .ok_or_else(|| invalid("exterior partition reference count overflows"))?;
    bounded(total, limit)?;
    Ok(total)
}

fn fixed_frontier(
    count: usize,
    limits: CanonicalGeometryLimits,
) -> Result<BTreeSet<usize>, Diagnostic> {
    bounded(1, limits.max_faces)?;
    bounded(count, limits.max_loop_indices)?;
    Ok((0..count).collect())
}

fn planar_frontier(
    topology: &PlanarRegion,
    parent: &NamedEntitySet,
    limits: CanonicalGeometryLimits,
) -> Result<BTreeSet<usize>, Diagnostic> {
    bounded(topology.faces().len(), limits.max_faces)?;
    let mut references = 0;
    for face in topology.faces() {
        for vertices in std::iter::once(face.outer()).chain(face.holes().iter().map(Vec::as_slice))
        {
            references = checked_total(references, vertices.len(), limits.max_loop_indices)?;
        }
    }
    // A parent-relative edge retains its own entity index. Only two exact,
    // reversed incidences from distinct selected faces cancel BOTH indices.
    let mut incidences = BTreeMap::<(usize, usize), Vec<(usize, usize, usize, usize)>>::new();
    let mut entity = 0;
    for (face_index, face) in topology.faces().iter().enumerate() {
        for vertices in std::iter::once(face.outer()).chain(face.holes().iter().map(Vec::as_slice))
        {
            for (&a, &b) in vertices
                .iter()
                .zip(vertices.iter().cycle().skip(1))
                .take(vertices.len())
            {
                if parent.members().binary_search(&face_index).is_ok() {
                    let uses = incidences.entry((a.min(b), a.max(b))).or_default();
                    if let Some(&(_, previous_parent, previous_a, previous_b)) = uses.first()
                        && (uses.len() != 1
                            || previous_parent == face_index
                            || previous_a != b
                            || previous_b != a)
                    {
                        return Err(invalid(
                            "exterior partition has same-oriented or nonmanifold edge incidence",
                        ));
                    }
                    uses.push((entity, face_index, a, b));
                }
                entity += 1;
            }
        }
    }
    Ok(incidences
        .into_values()
        .filter_map(|uses| (uses.len() == 1).then_some(uses[0].0))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CanonicalCartesianBoxGeometryV1, CanonicalPlanarAdjacentRectanglePartitionV1,
        CanonicalPlanarRectangleGeometryV2,
    };

    fn validate(geometry: &CanonicalGeometryV1) {
        let before = geometry.canonical_bytes().to_vec();
        geometry
            .validate_complete_exterior_partition(
                geometry.entity_set("body").unwrap(),
                &[geometry.entity_set("outer").unwrap()],
                CanonicalGeometryLimits::default(),
            )
            .unwrap();
        assert_eq!(geometry.canonical_bytes(), before);
    }

    #[test]
    fn fixed_families_use_their_complete_admitted_entity_enumeration() {
        for dimension in 1..=3 {
            let geometry = CanonicalCartesianBoxGeometryV1::new(
                vec![[0.0, 1.0]; dimension],
                vec![
                    NamedEntitySet::new("body", dimension, vec![0]),
                    NamedEntitySet::new("outer", dimension - 1, (0..2 * dimension).collect()),
                ],
            )
            .unwrap();
            validate(&CanonicalGeometryV1::from_cartesian_box_v1(geometry));
        }
        let rectangle = CanonicalPlanarRectangleGeometryV2::new(
            [[0.0, 1.0]; 2],
            vec![
                NamedEntitySet::new("body", 2, vec![0]),
                NamedEntitySet::new("outer", 1, vec![0, 1, 2, 3]),
            ],
        )
        .unwrap();
        validate(&CanonicalGeometryV1::from_planar_rectangle_v2(rectangle));
    }

    #[test]
    fn adjacent_partition_excludes_both_parent_relative_interface_entities() {
        let partition = CanonicalPlanarAdjacentRectanglePartitionV1::new(
            [[0.0, 2.0], [0.0, 1.0]],
            1.0,
            vec![
                NamedEntitySet::new("body", 2, vec![0, 1]),
                NamedEntitySet::new("outer", 1, vec![0, 2, 3, 4, 5, 6]),
                NamedEntitySet::new("interior", 1, vec![1, 7]),
            ],
        )
        .unwrap();
        let geometry = CanonicalGeometryV1::from_planar_adjacent_rectangle_partition_v1(partition);
        validate(&geometry);
        assert!(
            geometry
                .validate_complete_exterior_partition(
                    geometry.entity_set("body").unwrap(),
                    &[
                        geometry.entity_set("outer").unwrap(),
                        geometry.entity_set("interior").unwrap()
                    ],
                    CanonicalGeometryLimits::default(),
                )
                .is_err()
        );
    }
}
