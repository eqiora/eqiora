use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::{ReferenceCell, ReferenceCellFamily};

const MAX_REFERENCE_ENTITIES: usize = 1_000_000;
const MAX_REFERENCE_VERTEX_REFERENCES: usize = 8_000_000;

/// A bijection of local vertex ordinals.
///
/// The permutation stores the image of each canonical ordinal. Keeping the
/// permutation itself, rather than reducing it to a sign, is necessary for
/// vector elements and high-order traces. Backends may intern validated
/// permutations into compact [`crate::OrientationCode`] values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexPermutation {
    images: Vec<usize>,
}

impl VertexPermutation {
    /// Construct a validated vertex permutation.
    ///
    /// # Errors
    /// Returns `EQ0803` unless every ordinal in `0..images.len()` occurs once.
    pub fn new(images: Vec<usize>) -> Result<Self, Diagnostic> {
        let mut seen = vec![false; images.len()];
        for &image in &images {
            let Some(slot) = seen.get_mut(image) else {
                return Err(invalid_topology(
                    "orientation permutation contains an out-of-range ordinal",
                ));
            };
            if *slot {
                return Err(invalid_topology(
                    "orientation permutation contains a duplicate ordinal",
                ));
            }
            *slot = true;
        }
        Ok(Self { images })
    }

    /// Identity permutation of the requested arity.
    #[must_use]
    pub fn identity(arity: usize) -> Self {
        Self {
            images: (0..arity).collect(),
        }
    }

    /// Number of permuted vertices.
    #[must_use]
    pub fn arity(&self) -> usize {
        self.images.len()
    }

    /// Images in canonical ordinal order.
    #[must_use]
    pub fn images(&self) -> &[usize] {
        &self.images
    }

    /// Whether this is the identity permutation.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.images
            .iter()
            .enumerate()
            .all(|(ordinal, &image)| ordinal == image)
    }

    /// Sign of the permutation: `+1` for even and `-1` for odd parity.
    /// This is the orientation multiplier for an oriented simplex, not a
    /// replacement for the full permutation in higher-order trace maps.
    #[must_use]
    pub fn sign(&self) -> i8 {
        let inversions = self
            .images
            .iter()
            .enumerate()
            .map(|(index, image)| {
                self.images[index + 1..]
                    .iter()
                    .filter(|other| image > *other)
                    .count()
            })
            .sum::<usize>();
        if inversions % 2 == 0 { 1 } else { -1 }
    }

    /// Compose `self` followed by `after`.
    ///
    /// # Errors
    /// Returns `EQ0803` when the two permutations have different arity.
    pub fn then(&self, after: &Self) -> Result<Self, Diagnostic> {
        if self.arity() != after.arity() {
            return Err(invalid_topology(
                "cannot compose orientation permutations of different arity",
            ));
        }
        let images = self
            .images
            .iter()
            .map(|&image| after.images[image])
            .collect();
        Ok(Self { images })
    }

    /// Invert this permutation.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let mut images = vec![0; self.arity()];
        for (ordinal, &image) in self.images.iter().enumerate() {
            images[image] = ordinal;
        }
        Self { images }
    }
}

/// One entity of a reference-cell topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceEntity {
    dimension: usize,
    index: usize,
    vertex_ordinals: Vec<usize>,
}

impl ReferenceEntity {
    /// Entity dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Deterministic index within its dimension stratum.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Canonically ordered reference-cell vertices in this entity's closure.
    #[must_use]
    pub fn vertex_ordinals(&self) -> &[usize] {
        &self.vertex_ordinals
    }
}

/// Incidence of a lower-dimensional entity in a containing entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceIncidence {
    entity_dimension: usize,
    entity_index: usize,
    local_ordinal: usize,
    orientation: VertexPermutation,
}

impl ReferenceIncidence {
    /// Incident entity dimension.
    #[must_use]
    pub const fn entity_dimension(&self) -> usize {
        self.entity_dimension
    }

    /// Incident entity index within its stratum.
    #[must_use]
    pub const fn entity_index(&self) -> usize {
        self.entity_index
    }

    /// Lower entity's deterministic ordinal in the containing closure.
    #[must_use]
    pub const fn local_ordinal(&self) -> usize {
        self.local_ordinal
    }

    /// Vertex ordering relative to the incident entity's canonical order.
    #[must_use]
    pub const fn orientation(&self) -> &VertexPermutation {
        &self.orientation
    }
}

/// Combinatorial topology of one runtime-dimensional reference cell.
///
/// Simplex entities are non-empty subsets of `d + 1` vertices. Hypercube
/// entities are generated from fixed-lower, fixed-upper, or free state on
/// every axis. These native combinatorial definitions avoid dimension-specific
/// tables while retaining deterministic entity ordering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceTopology {
    cell: ReferenceCell,
    strata: Vec<Vec<ReferenceEntity>>,
}

impl ReferenceTopology {
    /// Generate a checked topology for a reference cell.
    ///
    /// # Errors
    /// Returns `EQ0803` when combinatorial counts overflow or exceed the
    /// inspectable artifact resource limits.
    pub fn new(cell: ReferenceCell) -> Result<Self, Diagnostic> {
        check_resource_bounds(cell)?;
        let strata = match cell.family() {
            ReferenceCellFamily::Point => vec![vec![ReferenceEntity {
                dimension: 0,
                index: 0,
                vertex_ordinals: vec![0],
            }]],
            ReferenceCellFamily::Simplex => simplex_strata(cell.dimension()),
            ReferenceCellFamily::Hypercube => hypercube_strata(cell.dimension()),
        };
        Ok(Self { cell, strata })
    }

    /// Reference cell described by this topology.
    #[must_use]
    pub const fn reference_cell(&self) -> ReferenceCell {
        self.cell
    }

    /// Number of entities in a dimension stratum.
    #[must_use]
    pub fn entity_count(&self, dimension: usize) -> Option<usize> {
        self.strata.get(dimension).map(Vec::len)
    }

    /// Retrieve an entity by dimension and stratum-local index.
    #[must_use]
    pub fn entity(&self, dimension: usize, index: usize) -> Option<&ReferenceEntity> {
        self.strata.get(dimension)?.get(index)
    }

    /// Incidence in either direction, in deterministic stratum order.
    ///
    /// The same `local_ordinal` is returned when querying a containing entity's
    /// closure or the reverse adjacency. Invalid entities or dimensions return
    /// `None`; valid non-incident strata return an empty vector.
    #[must_use]
    pub fn incidence(
        &self,
        entity_dimension: usize,
        entity_index: usize,
        target_dimension: usize,
    ) -> Option<Vec<ReferenceIncidence>> {
        let source = self.entity(entity_dimension, entity_index)?;
        let targets = self.strata.get(target_dimension)?;
        if entity_dimension == target_dimension {
            return Some(vec![ReferenceIncidence {
                entity_dimension,
                entity_index,
                local_ordinal: 0,
                orientation: VertexPermutation::identity(source.vertex_ordinals.len()),
            }]);
        }

        let (lower_dimension, lower_entities, containing) = if target_dimension < entity_dimension {
            (target_dimension, targets, source)
        } else {
            (entity_dimension, self.strata.get(entity_dimension)?, source)
        };

        if target_dimension < entity_dimension {
            let contained = contained_entities(lower_entities, containing);
            return Some(
                contained
                    .into_iter()
                    .enumerate()
                    .map(|(local_ordinal, target)| ReferenceIncidence {
                        entity_dimension: target.dimension,
                        entity_index: target.index,
                        local_ordinal,
                        orientation: VertexPermutation::identity(target.vertex_ordinals.len()),
                    })
                    .collect(),
            );
        }

        let mut incidences = Vec::new();
        for target in targets {
            let closure = contained_entities(lower_entities, target);
            if let Some(local_ordinal) = closure
                .iter()
                .position(|candidate| candidate.index == source.index)
            {
                incidences.push(ReferenceIncidence {
                    entity_dimension: target.dimension,
                    entity_index: target.index,
                    local_ordinal,
                    orientation: VertexPermutation::identity(
                        self.strata[lower_dimension][source.index]
                            .vertex_ordinals
                            .len(),
                    ),
                });
            }
        }
        Some(incidences)
    }
}

fn contained_entities<'a>(
    candidates: &'a [ReferenceEntity],
    containing: &ReferenceEntity,
) -> Vec<&'a ReferenceEntity> {
    candidates
        .iter()
        .filter(|candidate| {
            candidate
                .vertex_ordinals
                .iter()
                .all(|vertex| containing.vertex_ordinals.binary_search(vertex).is_ok())
        })
        .collect()
}

fn check_resource_bounds(cell: ReferenceCell) -> Result<(), Diagnostic> {
    let dimension = cell.dimension();
    let (entity_count, vertex_references) = match cell.family() {
        ReferenceCellFamily::Point => (1, 1),
        ReferenceCellFamily::Simplex => {
            let vertex_count = dimension.checked_add(1).ok_or_else(|| {
                invalid_topology("simplex reference-topology dimension overflows usize")
            })?;
            let exponent = u32::try_from(vertex_count).map_err(|_| {
                invalid_topology("simplex reference-topology dimension exceeds u32 capacity")
            })?;
            let subsets = 2_usize.checked_pow(exponent).ok_or_else(|| {
                invalid_topology("simplex reference-entity count overflows usize")
            })?;
            let reference_exponent = u32::try_from(vertex_count - 1).map_err(|_| {
                invalid_topology("simplex vertex-reference exponent exceeds u32 capacity")
            })?;
            let references = vertex_count
                .checked_mul(2_usize.checked_pow(reference_exponent).ok_or_else(|| {
                    invalid_topology("simplex vertex-reference count overflows usize")
                })?)
                .ok_or_else(|| {
                    invalid_topology("simplex vertex-reference count overflows usize")
                })?;
            (subsets - 1, references)
        }
        ReferenceCellFamily::Hypercube => {
            let exponent = u32::try_from(dimension).map_err(|_| {
                invalid_topology("hypercube reference-topology dimension exceeds u32 capacity")
            })?;
            let entities = 3_usize.checked_pow(exponent).ok_or_else(|| {
                invalid_topology("hypercube reference-entity count overflows usize")
            })?;
            let references = 4_usize.checked_pow(exponent).ok_or_else(|| {
                invalid_topology("hypercube vertex-reference count overflows usize")
            })?;
            (entities, references)
        }
    };
    if entity_count > MAX_REFERENCE_ENTITIES || vertex_references > MAX_REFERENCE_VERTEX_REFERENCES
    {
        return Err(invalid_topology(format!(
            "reference topology requires {entity_count} entities and {vertex_references} vertex references, exceeding inspectable artifact limits"
        )));
    }
    Ok(())
}

fn simplex_strata(dimension: usize) -> Vec<Vec<ReferenceEntity>> {
    let vertex_count = dimension + 1;
    (0..=dimension)
        .map(|entity_dimension| {
            combinations(vertex_count, entity_dimension + 1)
                .into_iter()
                .enumerate()
                .map(|(index, vertex_ordinals)| ReferenceEntity {
                    dimension: entity_dimension,
                    index,
                    vertex_ordinals,
                })
                .collect()
        })
        .collect()
}

fn hypercube_strata(dimension: usize) -> Vec<Vec<ReferenceEntity>> {
    (0..=dimension)
        .map(|entity_dimension| {
            let mut entities = Vec::new();
            for free_axes in combinations(dimension, entity_dimension) {
                let fixed_axes = (0..dimension)
                    .filter(|axis| free_axes.binary_search(axis).is_err())
                    .collect::<Vec<_>>();
                let fixed_assignments = 1_usize << fixed_axes.len();
                for fixed_bits in 0..fixed_assignments {
                    let free_assignments = 1_usize << free_axes.len();
                    let mut vertices = Vec::with_capacity(free_assignments);
                    for free_bits in 0..free_assignments {
                        let mut vertex = 0_usize;
                        for (ordinal, axis) in fixed_axes.iter().copied().enumerate() {
                            vertex |= ((fixed_bits >> ordinal) & 1) << axis;
                        }
                        for (ordinal, axis) in free_axes.iter().copied().enumerate() {
                            vertex |= ((free_bits >> ordinal) & 1) << axis;
                        }
                        vertices.push(vertex);
                    }
                    vertices.sort_unstable();
                    entities.push(ReferenceEntity {
                        dimension: entity_dimension,
                        index: entities.len(),
                        vertex_ordinals: vertices,
                    });
                }
            }
            entities
        })
        .collect()
}

fn combinations(universe: usize, selection: usize) -> Vec<Vec<usize>> {
    if selection == 0 {
        return vec![Vec::new()];
    }
    if selection > universe {
        return Vec::new();
    }
    let mut indices = (0..selection).collect::<Vec<_>>();
    let mut result = Vec::new();
    loop {
        result.push(indices.clone());
        let Some(position) = (0..selection)
            .rev()
            .find(|&position| indices[position] < universe - selection + position)
        else {
            break;
        };
        indices[position] += 1;
        for next in position + 1..selection {
            indices[next] = indices[next - 1] + 1;
        }
    }
    result
}

fn invalid_topology(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_MESH, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_forms_a_group_for_matching_arity() {
        let permutation = VertexPermutation::new(vec![2, 0, 3, 1]).unwrap();
        let inverse = permutation.inverse();
        assert!(permutation.then(&inverse).unwrap().is_identity());
        assert!(inverse.then(&permutation).unwrap().is_identity());
        assert_eq!(
            permutation.then(&VertexPermutation::identity(4)).unwrap(),
            permutation
        );
        assert_eq!(
            permutation
                .then(&VertexPermutation::identity(3))
                .unwrap_err()
                .code(),
            codes::INVALID_MESH
        );
    }

    #[test]
    fn simplex_entities_are_nonempty_vertex_subsets() {
        let topology = ReferenceTopology::new(ReferenceCell::simplex(3).unwrap()).unwrap();
        assert_eq!(topology.entity_count(0), Some(4));
        assert_eq!(topology.entity_count(1), Some(6));
        assert_eq!(topology.entity_count(2), Some(4));
        assert_eq!(topology.entity_count(3), Some(1));
        assert_eq!(topology.entity(2, 2).unwrap().vertex_ordinals(), &[0, 2, 3]);
    }

    #[test]
    fn hypercube_entity_counts_are_dimension_general() {
        let topology = ReferenceTopology::new(ReferenceCell::hypercube(5).unwrap()).unwrap();
        assert_eq!(topology.entity_count(0), Some(32));
        assert_eq!(topology.entity_count(1), Some(80));
        assert_eq!(topology.entity_count(2), Some(80));
        assert_eq!(topology.entity_count(3), Some(40));
        assert_eq!(topology.entity_count(4), Some(10));
        assert_eq!(topology.entity_count(5), Some(1));
    }

    #[test]
    fn incidence_local_ordinal_is_identical_in_both_directions() {
        let topology = ReferenceTopology::new(ReferenceCell::simplex(2).unwrap()).unwrap();
        let edge_to_vertices = topology.incidence(1, 1, 0).unwrap();
        assert_eq!(edge_to_vertices.len(), 2);
        for incidence in edge_to_vertices {
            let vertex_to_edges = topology.incidence(0, incidence.entity_index(), 1).unwrap();
            let reverse = vertex_to_edges
                .iter()
                .find(|candidate| candidate.entity_index() == 1)
                .unwrap();
            assert_eq!(reverse.local_ordinal(), incidence.local_ordinal());
            assert!(reverse.orientation().is_identity());
        }
    }

    #[test]
    fn excessive_combinatorics_are_rejected_before_allocation() {
        assert_eq!(
            ReferenceTopology::new(ReferenceCell::hypercube(64).unwrap())
                .unwrap_err()
                .code(),
            codes::INVALID_MESH
        );
    }
}
