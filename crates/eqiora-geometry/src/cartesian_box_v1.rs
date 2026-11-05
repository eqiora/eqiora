//! Classification-free canonical identity of one exact Cartesian box.

use std::collections::BTreeSet;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use serde::{Deserialize, Serialize};

use crate::canonical::{CANONICAL_ENCODING, WireEntitySet, WireLengthUnit, digest_with_schema};
use crate::{CanonicalGeometryLimits, NamedEntitySet};

pub(crate) const SCHEMA: &str = "eqiora.cartesian-box-envelope/v1";
const MAX_DIMENSION: usize = 3;

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

/// Exact axis-aligned Cartesian interval, rectangle, or box.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalCartesianBoxGeometryV1 {
    bounds: Vec<[f64; 2]>,
    entity_sets: Vec<NamedEntitySet>,
    bytes: Vec<u8>,
    digest: [u8; 32],
}

impl CanonicalCartesianBoxGeometryV1 {
    pub(crate) fn new(
        mut bounds: Vec<[f64; 2]>,
        entity_sets: Vec<NamedEntitySet>,
    ) -> Result<Self, Diagnostic> {
        Self::validate_bounds(&mut bounds)?;
        let entity_sets = canonical_entity_sets(bounds.len(), entity_sets)?;
        let wire = WireCartesianBoxV1 {
            schema: SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            length_unit: WireLengthUnit::Metre,
            bounds: bounds.clone(),
            entity_sets: entity_sets.iter().map(WireEntitySet::from_set).collect(),
        };
        let bytes = serde_json::to_vec(&wire)
            .map_err(|error| invalid(format!("cannot serialize Cartesian box: {error}")))?;
        Ok(Self {
            bounds,
            entity_sets,
            digest: digest_with_schema(SCHEMA, &bytes),
            bytes,
        })
    }

    /// Normalize and check Cartesian primitive bounds before topology is named.
    pub(crate) fn validate_bounds(bounds: &mut [[f64; 2]]) -> Result<(), Diagnostic> {
        if bounds.is_empty() || bounds.len() > MAX_DIMENSION {
            return Err(invalid(
                "Cartesian box requires between one and three coordinate axes",
            ));
        }
        for axis in bounds {
            for value in axis.iter_mut() {
                if *value == 0.0 {
                    *value = 0.0;
                }
            }
            if !axis[0].is_finite() || !axis[1].is_finite() || axis[0] >= axis[1] {
                return Err(invalid(
                    "Cartesian box bounds must be finite strict intervals",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn decode_canonical(
        bytes: &[u8],
        limits: CanonicalGeometryLimits,
    ) -> Result<Self, Diagnostic> {
        if bytes.len() > limits.max_bytes {
            return Err(invalid("Cartesian box exceeds the decoder byte limit"));
        }
        let wire: WireCartesianBoxV1 = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid Cartesian box JSON: {error}")))?;
        if wire.schema != SCHEMA
            || wire.encoding != CANONICAL_ENCODING
            || wire.length_unit != WireLengthUnit::Metre
            || wire.entity_sets.len() > limits.max_entity_sets
            || wire
                .entity_sets
                .iter()
                .map(|set| set.members.len())
                .sum::<usize>()
                > limits.max_entity_set_members
        {
            return Err(invalid("unsupported or oversized Cartesian box contract"));
        }
        let canonical = Self::new(
            wire.bounds,
            wire.entity_sets
                .into_iter()
                .map(|set| NamedEntitySet::new(set.name, set.dimension, set.members))
                .collect(),
        )?;
        if canonical.canonical_bytes() != bytes {
            return Err(invalid("Cartesian box JSON is not canonical"));
        }
        Ok(canonical)
    }

    pub(crate) fn bounds(&self) -> &[[f64; 2]] {
        &self.bounds
    }

    pub(crate) fn entity_sets(&self) -> &[NamedEntitySet] {
        &self.entity_sets
    }

    pub(crate) fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) const fn digest_bytes(&self) -> [u8; 32] {
        self.digest
    }
}

fn canonical_entity_sets(
    dimension: usize,
    entity_sets: Vec<NamedEntitySet>,
) -> Result<Vec<NamedEntitySet>, Diagnostic> {
    let boundary_dimension = dimension - 1;
    let boundary_count = dimension
        .checked_mul(2)
        .ok_or_else(|| invalid("Cartesian boundary count overflows usize"))?;
    let mut names = BTreeSet::new();
    let mut memberships = Vec::new();
    let mut canonical = Vec::with_capacity(entity_sets.len());
    for set in entity_sets {
        if set.name().trim().is_empty() || !names.insert(set.name().to_owned()) {
            return Err(invalid(
                "Cartesian box entity-set names must be nonempty and unique",
            ));
        }
        let limit = if set.dimension() == dimension {
            1
        } else if set.dimension() == boundary_dimension {
            boundary_count
        } else {
            return Err(invalid(
                "Cartesian box names may contain only its body or codimension-one sides",
            ));
        };
        let mut members = set.members().to_vec();
        members.sort_unstable();
        members.dedup();
        if members.is_empty() || members.iter().any(|member| *member >= limit) {
            return Err(invalid("Cartesian box entity set names an absent entity"));
        }
        memberships.extend(members.iter().map(|member| (set.dimension(), *member)));
        canonical.push(NamedEntitySet::new(set.name(), set.dimension(), members));
    }
    let unique = memberships.iter().copied().collect::<BTreeSet<_>>();
    let expected = (0..boundary_count)
        .map(|side| (boundary_dimension, side))
        .chain([(dimension, 0)])
        .collect::<BTreeSet<_>>();
    if unique != expected || memberships.len() != expected.len() {
        return Err(invalid(
            "named Cartesian box topology must cover its body and every side exactly once",
        ));
    }
    canonical.sort_by(|left, right| {
        (left.dimension(), left.name()).cmp(&(right.dimension(), right.name()))
    });
    Ok(canonical)
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WireCartesianBoxV1 {
    schema: String,
    encoding: String,
    length_unit: WireLengthUnit,
    bounds: Vec<[f64; 2]>,
    entity_sets: Vec<WireEntitySet>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval() -> CanonicalCartesianBoxGeometryV1 {
        CanonicalCartesianBoxGeometryV1::new(
            vec![[-1.0, 2.0]],
            vec![
                NamedEntitySet::new("left", 0, vec![0]),
                NamedEntitySet::new("right", 0, vec![1]),
                NamedEntitySet::new("body", 1, vec![0]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn interval_is_canonical_and_complete() {
        let geometry = interval();
        let replayed = CanonicalCartesianBoxGeometryV1::decode_canonical(
            geometry.canonical_bytes(),
            CanonicalGeometryLimits::default(),
        )
        .unwrap();
        assert_eq!(replayed, geometry);
        assert_eq!(geometry.bounds(), &[[-1.0, 2.0]]);
    }

    #[test]
    fn rejects_incomplete_or_ambiguous_interval_topology() {
        assert!(
            CanonicalCartesianBoxGeometryV1::new(
                vec![[0.0, 1.0]],
                vec![NamedEntitySet::new("body", 1, vec![0])],
            )
            .is_err()
        );
        assert!(
            CanonicalCartesianBoxGeometryV1::new(
                vec![[0.0, 1.0]],
                vec![
                    NamedEntitySet::new("ends", 0, vec![0, 1]),
                    NamedEntitySet::new("ends", 1, vec![0]),
                ],
            )
            .is_err()
        );
    }
}
