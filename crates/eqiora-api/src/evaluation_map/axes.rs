//! Private dense-axis admission and indexing, shared by native mapped products
//! and the independent affine reference. This module never evaluates a Program.

#[derive(Clone, Copy)]
pub(super) struct Limits {
    pub(super) rank: usize,
    pub(super) values: usize,
    pub(super) bytes: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Error {
    Rank,
    Axis,
    #[cfg(test)]
    Extent,
    Shape,
    Bounds,
    Overflow,
    Resource,
    #[cfg(test)]
    Signature,
}

type Admission<T> = Result<T, Error>;

pub(super) fn product(extents: &[usize]) -> Admission<usize> {
    // Check the nonzero factors even for an empty map: zero must not hide an
    // unrepresentable logical stride in another axis.
    let nonzero = extents.iter().try_fold(1usize, |size, &extent| {
        size.checked_mul(extent.max(1)).ok_or(Error::Overflow)
    })?;
    Ok(if extents.contains(&0) { 0 } else { nonzero })
}

fn budget(values: usize, limits: Limits) -> Admission<()> {
    let bytes = values
        .checked_mul(size_of::<f64>())
        .ok_or(Error::Overflow)?;
    if values > limits.values || bytes > limits.bytes {
        return Err(Error::Resource);
    }
    Ok(())
}

/// Positions always refer to the complete buffer, not to a progressively
/// shortened shape. One position per nesting level; None shares that leaf.
#[cfg(test)]
fn infer_extents(
    requested: &[Option<usize>],
    inputs: &[(&[usize], &[Option<usize>])],
    limits: Limits,
) -> Admission<Vec<usize>> {
    if requested.is_empty() || requested.len() > limits.rank {
        return Err(Error::Rank);
    }
    let mut extents = requested.to_vec();
    for &(shape, axes) in inputs {
        if shape.len() > limits.rank || axes.len() != requested.len() {
            return Err(Error::Rank);
        }
        for (level, &axis) in axes.iter().enumerate() {
            if let Some(axis) = axis {
                if axes[..level].contains(&Some(axis)) {
                    return Err(Error::Axis);
                }
                let extent = *shape.get(axis).ok_or(Error::Axis)?;
                if extents[level].is_some_and(|declared| declared != extent) {
                    return Err(Error::Extent);
                }
                extents[level] = Some(extent);
            }
        }
    }
    let extents = extents
        .into_iter()
        .map(|extent| extent.ok_or(Error::Extent))
        .collect::<Admission<Vec<_>>>()?;
    budget(product(&extents)?, limits)?;
    Ok(extents)
}

/// One dense buffer with declared collection axes around an unbatched shape.
/// Mathematical type and coefficient support belong to the value's signature;
/// indexing needs only its complete shape, including any coefficient axes.
#[derive(Debug)]
pub(super) struct Layout<'a> {
    value_shape: &'a [usize],
    extents: &'a [usize],
    axes: &'a [Option<usize>],
    pub(super) shape: Vec<usize>,
    elements: usize,
}

impl<'a> Layout<'a> {
    pub(super) fn new(
        value_shape: &'a [usize],
        extents: &'a [usize],
        axes: &'a [Option<usize>],
        limits: Limits,
    ) -> Admission<Self> {
        let rank = value_shape.len() + axes.iter().flatten().count();
        if extents.is_empty() || axes.len() != extents.len() || rank > limits.rank {
            return Err(Error::Rank);
        }
        for (level, &axis) in axes.iter().enumerate() {
            if let Some(axis) = axis
                && (axis >= rank || axes[..level].contains(&Some(axis)))
            {
                return Err(Error::Axis);
            }
        }
        let mut component_extents = value_shape.iter();
        let shape = (0..rank)
            .map(|axis| {
                axes.iter()
                    .position(|position| *position == Some(axis))
                    .map_or_else(
                        || *component_extents.next().unwrap(),
                        |level| extents[level],
                    )
            })
            .collect::<Vec<_>>();
        let elements = product(&shape)?;
        budget(elements, limits)?;
        Ok(Self {
            value_shape,
            extents,
            axes,
            shape,
            elements,
        })
    }

    pub(super) fn admit_buffer(&self, shape: &[usize], length: usize) -> Admission<()> {
        if shape != self.shape || length != self.elements {
            return Err(Error::Shape);
        }
        Ok(())
    }

    pub(super) fn offset(&self, occurrence: &[usize], component: &[usize]) -> Admission<usize> {
        if occurrence.len() != self.extents.len() || component.len() != self.value_shape.len() {
            return Err(Error::Rank);
        }
        if occurrence
            .iter()
            .zip(self.extents)
            .any(|(&index, &extent)| index >= extent)
        {
            return Err(Error::Bounds);
        }
        let mut components = component.iter();
        self.shape
            .iter()
            .enumerate()
            .try_fold(0usize, |offset, (axis, &extent)| {
                let index = axes_index(self.axes, axis)
                    .map_or_else(|| *components.next().unwrap(), |level| occurrence[level]);
                if index >= extent {
                    return Err(Error::Bounds);
                }
                offset
                    .checked_mul(extent)
                    .and_then(|offset| offset.checked_add(index))
                    .ok_or(Error::Overflow)
            })
    }

    #[cfg(test)]
    fn select<'b, T>(
        &self,
        values: &'b [T],
        occurrence: &[usize],
        component: &[usize],
    ) -> Admission<&'b T> {
        self.admit_buffer(&self.shape, values.len())?;
        values
            .get(self.offset(occurrence, component)?)
            .ok_or(Error::Bounds)
    }
}

fn axes_index(axes: &[Option<usize>], axis: usize) -> Option<usize> {
    axes.iter().position(|position| *position == Some(axis))
}

/// Dense outputs retain every map level, even when all members have equal
/// numerical values. Reject a missing level instead of treating it as shared.
#[cfg(test)]
fn output_elements(layouts: &[Layout<'_>], limits: Limits) -> Admission<usize> {
    let elements = layouts.iter().try_fold(0usize, |total, layout| {
        if layout.axes.iter().any(Option::is_none) {
            return Err(Error::Axis);
        }
        total.checked_add(layout.elements).ok_or(Error::Overflow)
    })?;
    budget(elements, limits)?;
    Ok(elements)
}

#[cfg(test)]
fn require_signature(
    expected: &crate::DifferentiableProgramIdentity,
    actual: &crate::DifferentiableProgramIdentity,
) -> Admission<()> {
    if expected != actual {
        return Err(Error::Signature);
    }
    Ok(())
}

#[cfg(test)]
#[path = "axes_reference_tests.rs"]
mod tests;
