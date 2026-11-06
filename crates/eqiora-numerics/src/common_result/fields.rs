use super::{CommonFieldAssociation, invalid};
use eqiora_core::{Diagnostic, ScalarDomain, ValueType};
use eqiora_realization::{Space, SpaceFamily};

pub(super) type BlockLayout = (CommonFieldAssociation, Vec<usize>);

/// Canonical entity-dimension order is shared by mapping, field recovery and
/// the existing multi-block Result codec. A nodal cell center is not a bubble.
pub(super) fn entity_block_layout(
    space: Space,
    dimension: usize,
    entities: &[eqiora_meshing::MeshEntity],
    value_shape: &[usize],
) -> Result<Vec<BlockLayout>, Diagnostic> {
    if entities.is_empty() || entities.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid(
            "coefficient topology must be nonempty, unique and canonically ordered",
        ));
    }
    let mut blocks = Vec::new();
    for stratum in 0..=dimension {
        let count = entities
            .iter()
            .filter(|entity| entity.dimension() == stratum)
            .count();
        if count == 0 {
            continue;
        }
        let association = if stratum == dimension {
            if space.family() == SpaceFamily::SimplexP1Bubble {
                CommonFieldAssociation::CellBubble
            } else {
                CommonFieldAssociation::Cell
            }
        } else {
            match stratum {
                0 => CommonFieldAssociation::Vertex,
                1 => CommonFieldAssociation::Edge,
                2 => CommonFieldAssociation::Face,
                _ => {
                    return Err(invalid(
                        "Result has an unsupported coefficient entity stratum",
                    ));
                }
            }
        };
        let mut shape = vec![count];
        shape.extend_from_slice(value_shape);
        blocks.push((association, shape));
    }
    if blocks.iter().map(|(_, shape)| shape[0]).sum::<usize>() != entities.len() {
        return Err(invalid(
            "Result coefficient topology has entities outside its mesh dimension",
        ));
    }
    Ok(blocks)
}

/// Finite coefficients whose scalar-domain width is checked by the owning Field.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonResultFieldBlock {
    pub(super) association: CommonFieldAssociation,
    pub(super) values: Vec<f64>,
    pub(super) logical_shape: Vec<usize>,
}

impl CommonResultFieldBlock {
    pub(super) fn new(
        association: CommonFieldAssociation,
        values: Vec<f64>,
        logical_shape: Vec<usize>,
    ) -> Result<Self, Diagnostic> {
        let _count = logical_shape.iter().try_fold(1usize, |count, extent| {
            count
                .checked_mul(*extent)
                .ok_or_else(|| invalid("Result Field block shape overflows usize"))
        })?;
        if logical_shape.is_empty() || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "Result Field block requires a nonempty exact shape and finite coefficients",
            ));
        }
        Ok(Self {
            association,
            values,
            logical_shape,
        })
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[must_use]
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }
}

/// One exact semantic Field and its complete accepted coefficient blocks.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonResultField {
    pub(super) field_id: String,
    pub(super) value_type: ValueType,
    pub(super) space: Space,
    pub(super) blocks: Vec<CommonResultFieldBlock>,
}

impl CommonResultField {
    pub(super) fn new(
        field_id: String,
        value_type: ValueType,
        space: Space,
        blocks: Vec<CommonResultFieldBlock>,
    ) -> Result<Self, Diagnostic> {
        common_space_name(space)?;
        space
            .coefficient_dimension(value_type.dimension())
            .ok_or_else(|| invalid("Result coefficient dimension overflows"))?;
        if field_id.is_empty() || blocks.is_empty() {
            return Err(invalid(
                "Result Field requires exact identity, space, and coefficient blocks",
            ));
        }
        let width = match value_type.scalar_domain() {
            ScalarDomain::Real => 1,
            ScalarDomain::Complex => 2,
            _ => {
                return Err(invalid(
                    "Result Field requires real or complex coefficients",
                ));
            }
        };
        for block in &blocks {
            let count = block
                .logical_shape
                .iter()
                .try_fold(width, |count: usize, extent| count.checked_mul(*extent))
                .ok_or_else(|| invalid("Result Field coordinate count overflows usize"))?;
            if count != block.values.len() {
                return Err(invalid(
                    "Result Field coefficient count differs from its scalar domain and logical shape",
                ));
            }
        }
        Ok(Self {
            field_id,
            value_type,
            space,
            blocks,
        })
    }

    #[must_use]
    pub fn field_id(&self) -> &str {
        &self.field_id
    }
    #[must_use]
    pub fn space(&self) -> &str {
        common_space_name(self.space).expect("validated Result Space")
    }
}

/// Canonical codec/presentation spelling; numerical meaning stays in Space.
///
/// # Errors
/// Returns a typed diagnostic for a Space without an admitted canonical codec.
pub fn common_space_name(space: Space) -> Result<&'static str, Diagnostic> {
    match space.family() {
        SpaceFamily::ContinuousLagrange { order } if order == std::num::NonZeroU16::MIN => {
            Ok("continuous-lagrange-p1")
        }
        SpaceFamily::ContinuousLagrange { order } if order.get() == 2 => {
            Ok("continuous-lagrange-p2")
        }
        SpaceFamily::CellConstant => Ok("cell-constant"),
        SpaceFamily::SimplexP1Bubble => Ok("simplex-p1-bubble"),
        SpaceFamily::TetrahedralEdge => Ok("tetrahedral-edge"),
        SpaceFamily::TetrahedralFace => Ok("tetrahedral-face"),
        _ => Err(invalid("unsupported Result coefficient Space")),
    }
}

pub(super) fn space_from_name(name: &str) -> Result<Space, Diagnostic> {
    match name {
        "continuous-lagrange-p1" => Ok(Space::continuous_lagrange(std::num::NonZeroU16::MIN)),
        "continuous-lagrange-p2" => Ok(Space::continuous_lagrange(
            std::num::NonZeroU16::new(2).expect("quadratic order"),
        )),
        "cell-constant" => Ok(Space::cell_constant()),
        "simplex-p1-bubble" => Ok(Space::simplex_p1_bubble()),
        "tetrahedral-edge" => Ok(Space::tetrahedral_edge()),
        "tetrahedral-face" => Ok(Space::tetrahedral_face()),
        _ => Err(invalid("unsupported Result coefficient Space")),
    }
}

pub(super) fn value_shape(value_type: &eqiora_core::ValueType) -> Vec<usize> {
    value_type
        .shape()
        .extents()
        .iter()
        .map(|extent| usize::try_from(extent.get()).expect("portable Field extent"))
        .collect()
}

/// Project complete types from the exact Model without re-admitting external Geometry.
/// The Model remains the sole persisted owner of mathematical Field meaning.
pub(super) fn model_field_types(
    model: &eqiora_artifact::ModelEnvelope,
) -> Result<std::collections::BTreeMap<String, ValueType>, Diagnostic> {
    let (transaction, _) = model.to_transaction().map_err(|diagnostics| {
        invalid(format!(
            "Result Model Field metadata is invalid: {diagnostics:?}"
        ))
    })?;
    Ok(transaction
        .ops()
        .iter()
        .filter_map(|operation| match operation {
            eqiora_graph::Op::DefineKernelNode {
                node: eqiora_schema::kernel::KernelNode::Field(field),
            } => Some((field.id().ulid().to_string(), field.value_type().clone())),
            _ => None,
        })
        .collect())
}

#[cfg(test)]
mod quadratic_tests {
    use super::*;
    use eqiora_meshing::MeshEntity;

    #[test]
    fn full_quadratic_blocks_preserve_entity_order_and_distinguish_nodal_centers() {
        let space = Space::continuous_lagrange(std::num::NonZeroU16::new(2).unwrap());
        for (dimension, counts, expected) in [
            (
                2,
                vec![4, 4, 1],
                vec![
                    CommonFieldAssociation::Vertex,
                    CommonFieldAssociation::Edge,
                    CommonFieldAssociation::Cell,
                ],
            ),
            (
                3,
                vec![8, 12, 6, 1],
                vec![
                    CommonFieldAssociation::Vertex,
                    CommonFieldAssociation::Edge,
                    CommonFieldAssociation::Face,
                    CommonFieldAssociation::Cell,
                ],
            ),
        ] {
            let entities = counts
                .iter()
                .enumerate()
                .flat_map(|(d, &n)| (0..n).map(move |i| MeshEntity::new(d, i)))
                .collect::<Vec<_>>();
            let blocks = entity_block_layout(space, dimension, &entities, &[]).unwrap();
            assert_eq!(
                blocks
                    .iter()
                    .map(|(association, _)| *association)
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                blocks.iter().map(|(_, shape)| shape[0]).collect::<Vec<_>>(),
                counts
            );
            assert!(
                !blocks
                    .iter()
                    .any(|(association, _)| *association == CommonFieldAssociation::CellBubble)
            );
            let mut bad = entities.clone();
            bad.swap(0, 1);
            assert!(entity_block_layout(space, dimension, &bad, &[]).is_err());
            let mut bad = entities.clone();
            bad.push(entities[0]);
            assert!(entity_block_layout(space, dimension, &bad, &[]).is_err());
        }
        assert_eq!(
            space_from_name(common_space_name(space).unwrap()).unwrap(),
            space
        );
    }
}
