use core::fmt;
use core::num::NonZeroU32;

/// Exact, storage-independent shape of one mathematical value.
///
/// Extents are ordered and positive. The empty extent list denotes a scalar,
/// so a one-component vector (`[1]`) remains distinct from a scalar (`[]`).
/// This type describes mathematical components only: mesh entities, basis
/// functions, quadrature points, and memory strides belong to realization.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueShape {
    extents: Box<[NonZeroU32]>,
}

impl ValueShape {
    /// Construct the scalar shape `[]`.
    #[must_use]
    pub fn scalar() -> Self {
        Self {
            extents: Box::default(),
        }
    }

    /// Construct an exact tensor shape from positive portable extents.
    ///
    /// # Errors
    /// Returns [`InvalidValueShape`] at the first zero extent.
    pub fn new(extents: impl IntoIterator<Item = u32>) -> Result<Self, InvalidValueShape> {
        let extents = extents
            .into_iter()
            .enumerate()
            .map(|(axis, extent)| NonZeroU32::new(extent).ok_or(InvalidValueShape { axis }))
            .collect::<Result<Box<[_]>, _>>()?;
        Ok(Self { extents })
    }

    /// Ordered canonical extents. An empty slice denotes a scalar.
    #[must_use]
    pub fn extents(&self) -> &[NonZeroU32] {
        &self.extents
    }

    /// Number of tensor axes.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.extents.len()
    }

    /// Whether this is the exact scalar shape `[]`.
    #[must_use]
    pub const fn is_scalar(&self) -> bool {
        self.extents.is_empty()
    }

    /// Number of scalar components, when representable on this target.
    ///
    /// A scalar has one component. This is value shape only; callers must
    /// separately account for spatial or temporal discretization cardinality.
    #[must_use]
    pub fn component_count(&self) -> Option<usize> {
        self.extents.iter().try_fold(1_usize, |count, extent| {
            count.checked_mul(usize::try_from(extent.get()).ok()?)
        })
    }

    /// Row-major position of one exact component coordinate.
    ///
    /// Returns `None` for a different rank, an out-of-range axis, or an
    /// unrepresentable shape. The scalar coordinate is the empty slice.
    #[must_use]
    pub fn flat_index(&self, coordinate: &[u32]) -> Option<usize> {
        if coordinate.len() != self.rank() || self.component_count().is_none() {
            return None;
        }
        self.extents
            .iter()
            .zip(coordinate)
            .try_fold(0usize, |flat, (extent, &index)| {
                (index < extent.get()).then_some(())?;
                flat.checked_mul(extent.get() as usize)?
                    .checked_add(index as usize)
            })
    }

    /// Append one positive extent while preserving exact axis order.
    ///
    /// # Errors
    /// Returns [`InvalidValueShape`] when `extent` is zero.
    pub fn appended(&self, extent: u32) -> Result<Self, InvalidValueShape> {
        Self::new(self.extents.iter().map(|value| value.get()).chain([extent]))
    }

    /// Remove and return the trailing extent.
    ///
    /// A scalar has no trailing extent and therefore returns `None`.
    #[must_use]
    pub fn remove_last(&self) -> Option<(Self, NonZeroU32)> {
        let (last, leading) = self.extents.split_last()?;
        Some((
            Self {
                extents: leading.into(),
            },
            *last,
        ))
    }
}

impl Default for ValueShape {
    fn default() -> Self {
        Self::scalar()
    }
}

/// Construction failure for an exact mathematical value shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidValueShape {
    axis: usize,
}

impl InvalidValueShape {
    /// Zero-based axis containing the forbidden zero extent.
    #[must_use]
    pub const fn axis(self) -> usize {
        self.axis
    }
}

impl fmt::Display for InvalidValueShape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "value-shape extent at axis {} must be positive",
            self.axis
        )
    }
}

impl std::error::Error for InvalidValueShape {}

#[cfg(test)]
mod tests {
    use super::ValueShape;

    #[test]
    fn scalar_vector_and_tensor_are_exact_and_distinct() {
        let scalar = ValueShape::scalar();
        let one_vector = ValueShape::new([1]).unwrap();
        let vector = ValueShape::new([2]).unwrap();
        let tensor = ValueShape::new([2, 3]).unwrap();

        assert!(scalar.extents().is_empty());
        assert!(scalar.is_scalar());
        assert_ne!(scalar, one_vector);
        assert_eq!(vector.component_count(), Some(2));
        assert_eq!(
            tensor
                .extents()
                .iter()
                .map(|extent| extent.get())
                .collect::<Vec<_>>(),
            [2, 3]
        );
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.component_count(), Some(6));
    }

    #[test]
    fn zero_extent_fails_at_the_exact_axis() {
        let error = ValueShape::new([2, 0, 4]).unwrap_err();
        assert_eq!(error.axis(), 1);
    }

    #[test]
    fn append_and_remove_are_inverse_exact_shape_operations() {
        let vector = ValueShape::scalar().appended(3).unwrap();
        let (scalar, extent) = vector.remove_last().unwrap();
        assert!(scalar.is_scalar());
        assert_eq!(extent.get(), 3);
        assert!(ValueShape::scalar().remove_last().is_none());
    }
}

#[cfg(test)]
mod coordinate_tests {
    use super::*;
    #[test]
    fn exact_component_coordinates_distinguish_rank_order_and_bounds() {
        let shape = ValueShape::new([2, 3, 4]).unwrap();
        assert_eq!(shape.flat_index(&[1, 0, 2]), Some(14));
        assert_eq!(shape.flat_index(&[0, 2, 1]), Some(9));
        assert_eq!(shape.flat_index(&[1, 2, 3]), Some(23));
        for coordinate in [
            &[1, 2][..],
            &[1, 2, 3, 0],
            &[2, 0, 0],
            &[0, 3, 0],
            &[0, 0, 4],
        ] {
            assert_eq!(shape.flat_index(coordinate), None);
        }
        assert_eq!(ValueShape::scalar().flat_index(&[]), Some(0));
        assert_eq!(ValueShape::scalar().flat_index(&[0]), None);
        let huge = ValueShape::new([u32::MAX; 3]).unwrap();
        assert_eq!(huge.flat_index(&[0, 0, 0]), None);
    }
}
