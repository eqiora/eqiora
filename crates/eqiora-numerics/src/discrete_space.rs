//! Reference elements selected by the existing Realization space contract.
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_meshing::{ReferenceCell, ReferenceCellFamily, ReferenceTopology, VertexPermutation};
use eqiora_realization::{Space, SpaceFamily};

mod binding;
mod compatible;
mod mapped;
mod scalar;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod vector_tests;
pub use mapped::PhysicalBasisTabulation;

const MAX_LOCAL_DOFS: usize = 1_000_000;
const MAX_BASIS_ENTRIES: usize = 8_000_000;

/// Topological support of one element-local coefficient functional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalDof {
    entity_dimension: usize,
    entity_ordinal: usize,
    slot: usize,
}
impl LocalDof {
    /// Dimension of the supporting reference entity.
    #[must_use]
    pub const fn entity_dimension(self) -> usize {
        self.entity_dimension
    }
    /// Supporting entity's ordinal in the reference-cell stratum.
    #[must_use]
    pub const fn entity_ordinal(self) -> usize {
        self.entity_ordinal
    }
    /// Coefficient slot on the supporting entity.
    #[must_use]
    pub const fn slot(self) -> usize {
        self.slot
    }
}

/// Values and derivatives in reference coordinates. Rows are basis functions;
/// vector components precede derivative axes in each flattened row.
#[derive(Debug, Clone, PartialEq)]
pub struct BasisTabulation {
    reference_dimension: usize,
    value_dimension: usize,
    values: Vec<f64>,
    reference_gradients: Vec<f64>,
}
impl BasisTabulation {
    /// Reference-coordinate dimension.
    #[must_use]
    pub const fn reference_dimension(&self) -> usize {
        self.reference_dimension
    }
    /// Number of value components per basis function (one for scalar bases).
    #[must_use]
    pub const fn value_dimension(&self) -> usize {
        self.value_dimension
    }
    /// Basis-major values, with `value_dimension` components per basis function.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    /// Basis-major, component-major reference gradients.
    #[must_use]
    pub fn reference_gradients(&self) -> &[f64] {
        &self.reference_gradients
    }
    /// Value of one reference basis function.
    #[must_use]
    pub fn value(&self, local_dof: usize) -> Option<&[f64]> {
        let start = local_dof.checked_mul(self.value_dimension)?;
        self.values
            .get(start..start.checked_add(self.value_dimension)?)
    }
    /// Reference gradient of one basis function, with component-major rows.
    #[must_use]
    pub fn gradient(&self, local_dof: usize) -> Option<&[f64]> {
        self.value(local_dof)?;
        let width = self.value_dimension.checked_mul(self.reference_dimension)?;
        let start = local_dof.checked_mul(width)?;
        self.reference_gradients
            .get(start..start.checked_add(width)?)
    }
}

/// One checked reference element. The Realization `Space` determines its basis,
/// coefficient functionals and Piola map; a caller cannot substitute a map label.
/// Global numbering, constraints and assembly remain separate owners.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteSpace {
    space: Space,
    cell: ReferenceCell,
    local_dofs: Vec<LocalDof>,
    bubble_normalization: f64,
}
impl DiscreteSpace {
    /// Construct an admitted scalar element or lowest-order tetrahedral edge/face element.
    ///
    /// # Errors
    /// Returns `EQ0801` for unsupported space/cell/order combinations or resource overflow.
    pub fn new(space: Space, cell: ReferenceCell) -> Result<Self, Diagnostic> {
        let d = cell.dimension();
        let (count, entity_dimension, value_dimension) = match (space.family(), cell.family()) {
            (SpaceFamily::CellConstant, _) => (1, d, 1),
            (
                SpaceFamily::ContinuousLagrange { order },
                ReferenceCellFamily::Simplex | ReferenceCellFamily::Hypercube,
            ) if order.get() == 1 => (reference_vertex_count(cell)?, 0, 1),
            (SpaceFamily::SimplexP1Bubble, ReferenceCellFamily::Simplex) => (
                d.checked_add(2)
                    .ok_or_else(|| invalid_space("simplex bubble coefficient count overflow"))?,
                0,
                1,
            ),
            (SpaceFamily::TetrahedralEdge, ReferenceCellFamily::Simplex) if d == 3 => (6, 1, 3),
            (SpaceFamily::TetrahedralFace, ReferenceCellFamily::Simplex) if d == 3 => (4, 2, 3),
            _ => {
                return Err(invalid_space(
                    "unsupported discrete space, reference cell or order",
                ));
            }
        };
        if count > MAX_LOCAL_DOFS
            || count
                .checked_mul(value_dimension)
                .and_then(|n| n.checked_mul(d.max(1)))
                .is_none_or(|n| n > MAX_BASIS_ENTRIES)
        {
            return Err(invalid_space(
                "basis-tabulation shape exceeds local resource limits",
            ));
        }
        let bubble_normalization = if space.family() == SpaceFamily::SimplexP1Bubble {
            (0..=d).try_fold(1.0_f64, |value, _| {
                let next = value * (d + 1) as f64;
                next.is_finite().then_some(next).ok_or_else(|| {
                    invalid_space("simplex P1-bubble normalization exceeds the scalar range")
                })
            })?
        } else {
            1.0
        };
        let local_dofs = (0..count)
            .map(|ordinal| {
                let bubble = space.family() == SpaceFamily::SimplexP1Bubble && ordinal == d + 1;
                LocalDof {
                    entity_dimension: if bubble { d } else { entity_dimension },
                    entity_ordinal: if bubble { 0 } else { ordinal },
                    slot: 0,
                }
            })
            .collect();
        Ok(Self {
            space,
            cell,
            local_dofs,
            bubble_normalization,
        })
    }
    /// Exact Realization-owned space selection.
    #[must_use]
    pub const fn space(&self) -> Space {
        self.space
    }
    /// Reference cell carrying the basis.
    #[must_use]
    pub const fn reference_cell(&self) -> ReferenceCell {
        self.cell
    }
    /// Coefficient descriptors in canonical local order.
    #[must_use]
    pub fn local_dofs(&self) -> &[LocalDof] {
        &self.local_dofs
    }
    /// Tabulate basis values and local reference derivatives.
    ///
    /// # Errors
    /// Returns `EQ0801` for invalid point dimensions, non-finite coordinates or points outside the cell.
    pub fn tabulate(&self, reference: &[f64]) -> Result<BasisTabulation, Diagnostic> {
        validate_reference_point(self.cell, reference)?;
        if self.vector_element() {
            compatible::tabulate(self, reference)
        } else {
            scalar::tabulate(self, reference)
        }
    }
    /// Map canonical coefficient functionals through a vertex permutation.
    /// Each pair is `(destination ordinal, orientation sign)`; integral moments
    /// retain signs rather than silently becoming unsigned nodal permutations.
    ///
    /// # Errors
    /// Returns `EQ0801` for incompatible vertex arity.
    pub fn oriented_dofs(
        &self,
        permutation: &VertexPermutation,
    ) -> Result<Vec<(usize, i8)>, Diagnostic> {
        let arity = reference_vertex_count(self.cell)?;
        if permutation.arity() != arity {
            return Err(invalid_space(
                "orientation has incompatible reference vertex arity",
            ));
        }
        if self.vector_element() {
            let topology = ReferenceTopology::new(self.cell)?;
            let dimension = self.local_dofs[0].entity_dimension;
            return self
                .local_dofs
                .iter()
                .map(|dof| {
                    let entity = topology
                        .entity(dimension, dof.entity_ordinal)
                        .expect("validated local entity");
                    let mut images = entity
                        .vertex_ordinals()
                        .iter()
                        .map(|&v| permutation.images()[v])
                        .collect::<Vec<_>>();
                    let sign = compatible::orientation_sign(&images);
                    images.sort_unstable();
                    let destination = self
                        .local_dofs
                        .iter()
                        .position(|candidate| {
                            topology
                                .entity(dimension, candidate.entity_ordinal)
                                .expect("validated local entity")
                                .vertex_ordinals()
                                == images
                        })
                        .expect("permutation preserves simplex entities");
                    Ok((destination, sign))
                })
                .collect();
        }
        if self.space.family() == SpaceFamily::CellConstant {
            return Ok(vec![(0, 1)]);
        }
        let mut order = permutation
            .images()
            .iter()
            .map(|&index| (index, 1))
            .collect::<Vec<_>>();
        if self.space.family() == SpaceFamily::SimplexP1Bubble {
            order.push((arity, 1));
        }
        Ok(order)
    }
    fn vector_element(&self) -> bool {
        matches!(
            self.space.family(),
            SpaceFamily::TetrahedralEdge | SpaceFamily::TetrahedralFace
        )
    }
}

fn validate_reference_point(cell: ReferenceCell, reference: &[f64]) -> Result<(), Diagnostic> {
    if !cell.contains(reference) {
        return Err(invalid_space(
            "basis tabulation point is non-finite, dimensionally invalid, or outside the reference cell",
        ));
    }
    Ok(())
}
fn reference_vertex_count(cell: ReferenceCell) -> Result<usize, Diagnostic> {
    match cell.family() {
        ReferenceCellFamily::Point => Ok(1),
        ReferenceCellFamily::Simplex => cell
            .dimension()
            .checked_add(1)
            .ok_or_else(|| invalid_space("simplex reference-vertex count overflows usize")),
        ReferenceCellFamily::Hypercube => u32::try_from(cell.dimension())
            .ok()
            .and_then(|d| 2_usize.checked_pow(d))
            .filter(|&n| n <= MAX_LOCAL_DOFS)
            .ok_or_else(|| {
                invalid_space("hypercube reference-vertex count exceeds resource limits")
            }),
    }
}
fn invalid_space(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}
