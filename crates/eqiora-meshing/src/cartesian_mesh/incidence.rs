//! Enumerate incident anchor tuples and find them in the canonical strata.
use super::*;

impl CartesianMesh {
    fn local_ordinal(&self, lower: &CartesianEntity, higher: &CartesianEntity) -> Option<usize> {
        let topology = self.reference_topologies.get(higher.free_axes.len())?;
        let mut reference_vertices = lower
            .vertices
            .iter()
            .map(|vertex| {
                higher
                    .vertices
                    .iter()
                    .position(|candidate| candidate == vertex)
            })
            .collect::<Option<Vec<_>>>()?;
        reference_vertices.sort_unstable();
        let dimension = lower.free_axes.len();
        let entity_count = topology.entity_count(dimension)?;
        (0..entity_count).find(|&index| {
            topology
                .entity(dimension, index)
                .is_some_and(|entity| entity.vertex_ordinals() == reference_vertices)
        })
    }

    pub(super) fn incident_entities(
        &self,
        entity: MeshEntity,
        target_dimension: usize,
    ) -> Option<Vec<EntityIncidence>> {
        let source = self.entity(entity)?;
        let targets = self.strata.get(target_dimension)?;
        if entity.dimension() == target_dimension {
            return Some(vec![EntityIncidence {
                entity,
                local_ordinal: 0,
                orientation: OrientationCode::identity(),
            }]);
        }
        let descending = target_dimension < entity.dimension();
        let families = if descending {
            combinations(source.free_axes.len(), target_dimension)
                .into_iter()
                .map(|ordinals| ordinals.into_iter().map(|i| source.free_axes[i]).collect())
                .collect::<Vec<Vec<usize>>>()
        } else {
            let fixed_axes = (0..self.axes.len())
                .filter(|axis| source.free_axes.binary_search(axis).is_err())
                .collect::<Vec<_>>();
            combinations(fixed_axes.len(), target_dimension - entity.dimension())
                .into_iter()
                .map(|ordinals| {
                    let mut axes = source.free_axes.clone();
                    axes.extend(ordinals.into_iter().map(|i| fixed_axes[i]));
                    axes.sort_unstable();
                    axes
                })
                .collect()
        };
        let mut incidences = Vec::new();
        for free_axes in families {
            let changed_axes = if descending {
                source
                    .free_axes
                    .iter()
                    .filter(|axis| free_axes.binary_search(axis).is_err())
                    .copied()
                    .collect::<Vec<_>>()
            } else {
                free_axes
                    .iter()
                    .filter(|axis| source.free_axes.binary_search(axis).is_err())
                    .copied()
                    .collect::<Vec<_>>()
            };
            'corner: for bits in 0..checked_power_of_two(changed_axes.len()).ok()? {
                let mut anchors = source.anchors.clone();
                for (bit, &axis) in changed_axes.iter().enumerate() {
                    let side = (bits >> bit) & 1;
                    if descending {
                        // A freed axis becomes either endpoint of its source interval.
                        anchors[axis] += side;
                    } else {
                        // A fixed vertex can belong to the interval on either side.
                        let Some(anchor) = anchors[axis].checked_sub(side) else {
                            continue 'corner;
                        };
                        if anchor >= self.axes[axis].len() - 1 {
                            continue 'corner;
                        }
                        anchors[axis] = anchor;
                    }
                }
                // Construction orders axis combinations lexicographically, then
                // row-major anchors. Reuse that order without another index/cache.
                let target_index = targets
                    .binary_search_by(|target| {
                        target
                            .free_axes
                            .cmp(&free_axes)
                            .then_with(|| target.anchors.cmp(&anchors))
                    })
                    .ok()?;
                let target = &targets[target_index];
                let (lower, higher) = if descending {
                    (target, source)
                } else {
                    (source, target)
                };
                incidences.push(EntityIncidence {
                    entity: MeshEntity::new(target_dimension, target_index),
                    local_ordinal: self.local_ordinal(lower, higher)?,
                    orientation: OrientationCode::identity(),
                });
            }
        }
        // Preserve reference-cell order downward and global entity order upward.
        incidences.sort_by_key(|incidence| {
            if descending {
                incidence.local_ordinal
            } else {
                incidence.entity.index()
            }
        });
        Some(incidences)
    }
}

#[cfg(test)]
mod tests;
