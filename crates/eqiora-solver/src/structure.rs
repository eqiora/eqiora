//! Exact algebraic Field and gauge structure retained before solver selection.
use eqiora_core::{Id, entity::kinds};

/// Method-owned algebraic constraint used to select a unique solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgebraicConstraint {
    /// Add one multiplier enforcing an exactly zero spatial integral.
    ZeroIntegral {
        /// Scalar Field whose constant nullspace is fixed.
        field: Id<kinds::Field>,
    },
}

impl AlgebraicConstraint {
    /// Field constrained by this algebraic choice.
    #[must_use]
    pub const fn field(self) -> Id<kinds::Field> {
        match self {
            Self::ZeroIntegral { field } => field,
        }
    }
}

/// One independently scaled block of the realized algebraic unknown vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgebraicBlock {
    /// Coefficients of one Semantic Field.
    Field(Id<kinds::Field>),
    /// Multiplier introduced by that Field's zero-integral constraint.
    ConstraintMultiplier {
        /// Field identifying the unique constraint.
        field: Id<kinds::Field>,
    },
}

/// Exact unknown Fields and explicitly realized gauges for a canonical projection.
/// This retains semantic blocks; it does not infer DOF ranges from matrix entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraicStructure {
    fields: Vec<Id<kinds::Field>>,
    constraints: Vec<AlgebraicConstraint>,
    coordinates: Option<Vec<(AlgebraicBlock, Vec<usize>)>>,
    coordinate_count: usize,
}

impl AlgebraicStructure {
    /// Admit complete unique Field identities and their explicit gauge constraints.
    ///
    /// # Errors
    /// Rejects an empty/repeated Field inventory or repeated/foreign constraints.
    pub fn new(
        fields: impl IntoIterator<Item = Id<kinds::Field>>,
        constraints: impl IntoIterator<Item = AlgebraicConstraint>,
    ) -> Result<Self, eqiora_core::Diagnostic> {
        let mut fields: Vec<_> = fields.into_iter().collect();
        let mut constraints: Vec<_> = constraints.into_iter().collect();
        fields.sort_by_key(Id::ulid);
        constraints.sort_by_key(|constraint| constraint.field().ulid());
        if fields.is_empty()
            || fields.windows(2).any(|pair| pair[0] == pair[1])
            || constraints.windows(2).any(|pair| pair[0] == pair[1])
            || constraints
                .iter()
                .any(|constraint| !fields.contains(&constraint.field()))
        {
            return Err(eqiora_core::Diagnostic::error(
                eqiora_core::diagnostic::codes::INVALID_REALIZATION,
                "algebraic structure requires unique Fields and complete exact gauge ownership",
            ));
        }
        Ok(Self {
            fields,
            constraints,
            coordinates: None,
            coordinate_count: 0,
        })
    }

    /// Retain exact method-owned reduced membership, including quotient sharing.
    ///
    /// # Errors
    /// Rejects missing/repeated blocks, omitted/foreign/repeated coordinates,
    /// and multiplier coordinates shared with another block.
    pub fn with_coordinates(
        mut self,
        count: usize,
        membership: Vec<(AlgebraicBlock, Vec<usize>)>,
    ) -> Result<Self, eqiora_core::Diagnostic> {
        let blocks: Vec<_> = self.blocks().collect();
        if membership.len() != blocks.len() {
            return Err(invalid_coordinates());
        }
        let mut covered = vec![false; count];
        let mut ordered = Vec::new();
        for block in blocks {
            let matches: Vec<_> = membership
                .iter()
                .filter(|(owner, _)| *owner == block)
                .collect();
            if matches.len() != 1 {
                return Err(invalid_coordinates());
            }
            let values = &matches[0].1;
            if matches!(block, AlgebraicBlock::ConstraintMultiplier { .. }) && values.len() != 1 {
                return Err(invalid_coordinates());
            }
            let mut seen = std::collections::BTreeSet::new();
            for &coordinate in values {
                if coordinate >= count || !seen.insert(coordinate) {
                    return Err(invalid_coordinates());
                }
                covered[coordinate] = true;
            }
            ordered.push((block, values.clone()));
        }
        if covered.iter().any(|covered| !covered) {
            return Err(invalid_coordinates());
        }
        for (block, values) in &ordered {
            if matches!(block, AlgebraicBlock::ConstraintMultiplier { .. })
                && ordered.iter().any(|(other, other_values)| {
                    other != block && other_values.contains(&values[0])
                })
            {
                return Err(invalid_coordinates());
            }
        }
        self.coordinates = Some(ordered);
        self.coordinate_count = count;
        Ok(self)
    }

    /// Exact reduced membership; absent before binding the method's discrete layout.
    #[must_use]
    pub fn coordinates(&self) -> Option<&[(AlgebraicBlock, Vec<usize>)]> {
        self.coordinates.as_deref()
    }

    /// Union physical blocks within explicit disjoint solver groups.
    ///
    /// # Errors
    /// Rejects absent ownership, missing/repeated/foreign blocks and cross-group sharing.
    pub fn partition_coordinates(
        &self,
        groups: &[Vec<AlgebraicBlock>],
    ) -> Result<Vec<Vec<usize>>, eqiora_core::Diagnostic> {
        let coordinates = self.coordinates.as_ref().ok_or_else(invalid_coordinates)?;
        let mut blocks = Vec::new();
        let mut owners = vec![None; self.coordinate_count];
        let mut result = Vec::new();
        for (group_index, group) in groups.iter().enumerate() {
            if group.is_empty() {
                return Err(invalid_coordinates());
            }
            let mut union = std::collections::BTreeSet::new();
            for block in group {
                if blocks.contains(block) {
                    return Err(invalid_coordinates());
                }
                blocks.push(*block);
                let (_, values) = coordinates
                    .iter()
                    .find(|(owner, _)| owner == block)
                    .ok_or_else(invalid_coordinates)?;
                for &coordinate in values {
                    if owners[coordinate].is_some_and(|owner| owner != group_index) {
                        return Err(invalid_coordinates());
                    }
                    owners[coordinate] = Some(group_index);
                    union.insert(coordinate);
                }
            }
            result.push(union.into_iter().collect());
        }
        if blocks.len() != coordinates.len() || owners.iter().any(Option::is_none) {
            return Err(invalid_coordinates());
        }
        Ok(result)
    }

    /// Bind exact reduced membership and ordering into the existing prepared identity.
    ///
    /// # Errors
    /// Rejects unbound inventory-only structure.
    pub fn prepared_identity(
        &self,
        base: &crate::PreparedLinearStructureIdentity,
        groups: &[Vec<AlgebraicBlock>],
    ) -> Result<crate::PreparedLinearStructureIdentity, eqiora_core::Diagnostic> {
        let coordinates = self.coordinates.as_ref().ok_or_else(invalid_coordinates)?;
        let mut encoding = b"eqiora.algebraic-coordinate-ownership/v1\0".to_vec();
        encoding.extend_from_slice(&(base.encoding().len() as u64).to_be_bytes());
        encoding.extend_from_slice(base.encoding());
        encoding.extend_from_slice(&(self.coordinate_count as u64).to_be_bytes());
        encoding.extend_from_slice(&(coordinates.len() as u64).to_be_bytes());
        for (block, values) in coordinates {
            let (tag, field) = match block {
                AlgebraicBlock::Field(field) => (0, field),
                AlgebraicBlock::ConstraintMultiplier { field } => (1, field),
            };
            encoding.push(tag);
            encoding.extend_from_slice(&field.ulid().to_bytes());
            encoding.extend_from_slice(&(values.len() as u64).to_be_bytes());
            for &coordinate in values {
                encoding.extend_from_slice(&(coordinate as u64).to_be_bytes());
            }
        }
        self.partition_coordinates(groups)?;
        encoding.extend_from_slice(&(groups.len() as u64).to_be_bytes());
        for group in groups {
            encoding.extend_from_slice(&(group.len() as u64).to_be_bytes());
            for block in group {
                let position = coordinates
                    .iter()
                    .position(|(owner, _)| owner == block)
                    .ok_or_else(invalid_coordinates)?;
                encoding.extend_from_slice(&(position as u64).to_be_bytes());
            }
        }
        crate::PreparedLinearStructureIdentity::new(encoding)
    }

    /// Exact Field and gauge-multiplier blocks of the canonical projection.
    pub fn blocks(&self) -> impl Iterator<Item = AlgebraicBlock> + '_ {
        self.fields
            .iter()
            .copied()
            .map(AlgebraicBlock::Field)
            .chain(
                self.constraints
                    .iter()
                    .map(|constraint| AlgebraicBlock::ConstraintMultiplier {
                        field: constraint.field(),
                    }),
            )
    }

    /// Method-owned gauge conditions, independent from solver/provider choice.
    #[must_use]
    pub fn constraints(&self) -> &[AlgebraicConstraint] {
        &self.constraints
    }
}

fn invalid_coordinates() -> eqiora_core::Diagnostic {
    eqiora_core::Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_REALIZATION,
        "algebraic coordinates require complete exact ownership and disjoint solver groups",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearOperatorProperties, SolverPlanningProfile};

    #[test]
    fn quotient_membership_is_unioned_only_inside_explicit_solver_groups() {
        let a = Id::new();
        let b = Id::new();
        let p = Id::new();
        let (a_block, b_block, p_block, gauge) = (
            AlgebraicBlock::Field(a),
            AlgebraicBlock::Field(b),
            AlgebraicBlock::Field(p),
            AlgebraicBlock::ConstraintMultiplier { field: p },
        );
        let inventory =
            AlgebraicStructure::new([a, b, p], [AlgebraicConstraint::ZeroIntegral { field: p }])
                .unwrap();
        let membership = vec![
            (a_block, vec![2, 0]),
            (b_block, vec![0, 3]),
            (p_block, vec![1]),
            (gauge, vec![4]),
        ];
        let structure = inventory
            .clone()
            .with_coordinates(5, membership.clone())
            .unwrap();
        let groups = vec![vec![a_block, b_block], vec![p_block, gauge]];
        assert_eq!(
            structure.partition_coordinates(&groups).unwrap(),
            vec![vec![0, 2, 3], vec![1, 4]]
        );
        assert!(
            structure
                .partition_coordinates(&[vec![a_block], vec![b_block, p_block, gauge]])
                .is_err()
        );
        assert!(
            structure
                .partition_coordinates(&[vec![a_block, b_block], vec![p_block]])
                .is_err()
        );
        assert!(
            structure
                .partition_coordinates(&[vec![a_block, b_block], vec![p_block, gauge, gauge]])
                .is_err()
        );
        let base = crate::PreparedLinearStructureIdentity::new(&b"retained layout"[..]).unwrap();
        assert!(inventory.prepared_identity(&base, &groups).is_err());
        let identity = structure.prepared_identity(&base, &groups).unwrap();
        let reversed = vec![groups[1].clone(), groups[0].clone()];
        assert_ne!(
            identity,
            structure.prepared_identity(&base, &reversed).unwrap()
        );
        let mut permutation = membership.clone();
        permutation[0].1.reverse();
        assert_ne!(
            identity,
            inventory
                .clone()
                .with_coordinates(5, permutation)
                .unwrap()
                .prepared_identity(&base, &groups)
                .unwrap()
        );
        for (owner, values) in [
            (a_block, vec![2, 2]),
            (b_block, vec![0, 5]),
            (gauge, vec![]),
            (gauge, vec![0]),
            (p_block, vec![]),
        ] {
            let mutated = membership
                .iter()
                .map(|(block, original)| {
                    (
                        *block,
                        if *block == owner {
                            values.clone()
                        } else {
                            original.clone()
                        },
                    )
                })
                .collect();
            assert!(inventory.clone().with_coordinates(5, mutated).is_err());
        }
        assert!(inventory.with_coordinates(6, membership).is_err());
    }

    #[test]
    fn exact_fields_and_gauges_are_complete_before_candidate_admission() {
        let a = Id::new();
        let b = Id::new();
        let gauge = AlgebraicConstraint::ZeroIntegral { field: b };
        let structure = AlgebraicStructure::new([a, b], [gauge]).unwrap();
        assert_eq!(structure, AlgebraicStructure::new([b, a], [gauge]).unwrap());
        assert_eq!(structure.blocks().count(), 3);
        assert!(AlgebraicStructure::new([a, a], []).is_err());
        assert!(AlgebraicStructure::new([a], [gauge]).is_err());
        assert!(AlgebraicStructure::new([a, b], [gauge, gauge]).is_err());
        assert!(AlgebraicStructure::new([], []).is_err());
        assert!(
            SolverPlanningProfile::canonical_csr(
                LinearOperatorProperties::SymmetricPositiveDefinite,
                None,
                None,
            )
            .with_structure(structure.clone())
            .is_err()
        );
        let profile = SolverPlanningProfile::canonical_csr(
            LinearOperatorProperties::SymmetricIndefinite,
            None,
            None,
        )
        .with_structure(structure.clone())
        .unwrap();
        profile.require_structure(Some(&structure)).unwrap();
        assert!(profile.require_structure(None).is_err());
        assert!(
            profile
                .require_structure(Some(&AlgebraicStructure::new([a, b], []).unwrap()))
                .is_err()
        );
        assert!(
            profile
                .require_structure(Some(&AlgebraicStructure::new([a, Id::new()], []).unwrap()))
                .is_err()
        );
    }
}
