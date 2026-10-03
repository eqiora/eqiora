//! Bounded tensor operations expressed in the shared exact component calculus.
use super::*;

fn class(rank: u16, extent: u32) -> Result<PureValueClass, PureOperatorError> {
    if rank > 4 || extent == 0 {
        return Err(PureOperatorError::InvalidResultRule);
    }
    if rank == 0 {
        Ok(PureValueClass::invariant_scalar())
    } else {
        PureValueClass::spatial_tensor(rank)?.with_spatial_extent(extent)
    }
}

impl PureOperatorDefinition {
    /// Algebraic diagonal sum of a full rank-two spatial tensor.
    pub fn matrix_trace(extent: u32) -> Result<Self, PureOperatorError> {
        if u64::from(extent) * 2 > MAX_NODES as u64 {
            return Err(PureOperatorError::NodeLimit);
        }
        let mut builder = CalculusBuilder::new([class(2, extent)?], class(0, extent)?)?;
        let mut root = None;
        for coordinate in 0..extent {
            let value = builder.push(CalculusNode::FormalComponent {
                formal: 0,
                axes: [ComponentIndex::Fixed(coordinate); 2].into(),
            })?;
            root = Some(match root {
                None => value,
                Some(sum) => builder.push(CalculusNode::Add(sum, value))?,
            });
        }
        builder.finish(root.ok_or(PureOperatorError::InvalidNode)?)
    }

    /// Multiply matching full tensor coordinates without contraction or broadcasting.
    pub fn componentwise_product(extent: u32, rank: u16) -> Result<Self, PureOperatorError> {
        let tensor = class(rank, extent)?;
        let mut builder = CalculusBuilder::new([tensor, tensor], tensor)?;
        let axes = (0..rank).map(ComponentIndex::Result).collect::<Box<[_]>>();
        let left = builder.push(CalculusNode::FormalComponent {
            formal: 0,
            axes: axes.clone(),
        })?;
        let right = builder.push(CalculusNode::FormalComponent { formal: 1, axes })?;
        let root = builder.push(CalculusNode::Mul(left, right))?;
        builder.finish(root)
    }

    /// Select one scalar from full spatial tensor coordinates.
    pub fn tensor_component(extent: u32, indices: &[u32]) -> Result<Self, PureOperatorError> {
        let rank =
            u16::try_from(indices.len()).map_err(|_| PureOperatorError::InvalidResultRule)?;
        if rank == 0 || indices.iter().any(|index| *index >= extent) {
            return Err(PureOperatorError::FormalComponentRank);
        }
        let mut builder = CalculusBuilder::new([class(rank, extent)?], class(0, extent)?)?;
        let root = builder.push(CalculusNode::FormalComponent {
            formal: 0,
            axes: indices.iter().copied().map(ComponentIndex::Fixed).collect(),
        })?;
        builder.finish(root)
    }

    /// Permute full spatial axes. `order[k]` identifies input axis of output axis `k`.
    pub fn permute_axes(extent: u32, order: &[u16]) -> Result<Self, PureOperatorError> {
        let rank = u16::try_from(order.len()).map_err(|_| PureOperatorError::InvalidResultRule)?;
        let tensor = class(rank, extent)?;
        let mut axes = vec![ComponentIndex::Result(0); order.len()];
        let mut seen = vec![false; order.len()];
        for (output, input) in order.iter().copied().enumerate() {
            let input = usize::from(input);
            if input >= order.len() || seen[input] {
                return Err(PureOperatorError::ResultAxisOutOfRange);
            }
            seen[input] = true;
            axes[input] = ComponentIndex::Result(output as u16);
        }
        let mut builder = CalculusBuilder::new([tensor], tensor)?;
        let root = builder.push(CalculusNode::FormalComponent {
            formal: 0,
            axes: axes.into(),
        })?;
        builder.finish(root)
    }

    /// Explicit bilinear contraction, without conjugation or implicit symmetry.
    /// Uncontracted left axes precede uncontracted right axes, in their original order.
    /// An empty pair list is the outer product, including scalar multiplication.
    pub fn contract(
        extent: u32,
        left_rank: u16,
        right_rank: u16,
        pairs: &[(u16, u16)],
    ) -> Result<Self, PureOperatorError> {
        let left = class(left_rank, extent)?;
        let right = class(right_rank, extent)?;
        let mut left_bound = vec![None; usize::from(left_rank)];
        let mut right_bound = vec![None; usize::from(right_rank)];
        for (slot, &(a, b)) in pairs.iter().enumerate() {
            let a = left_bound
                .get_mut(usize::from(a))
                .ok_or(PureOperatorError::ResultAxisOutOfRange)?;
            let b = right_bound
                .get_mut(usize::from(b))
                .ok_or(PureOperatorError::ResultAxisOutOfRange)?;
            if a.is_some() || b.is_some() {
                return Err(PureOperatorError::ResultAxisOutOfRange);
            }
            *a = Some(slot);
            *b = Some(slot);
        }
        let result_rank = left_rank + right_rank - 2 * pairs.len() as u16;
        let result = class(result_rank, extent)?;
        let terms = extent
            .checked_pow(pairs.len() as u32)
            .filter(|count| u64::from(*count) * 4 <= MAX_NODES as u64)
            .ok_or(PureOperatorError::NodeLimit)?;
        let mut builder = CalculusBuilder::new([left, right], result)?;
        let mut root = None;
        for term in 0..terms {
            let mut remainder = term;
            let mut coordinates = vec![0; pairs.len()];
            for coordinate in coordinates.iter_mut().rev() {
                *coordinate = remainder % extent;
                remainder /= extent;
            }
            let mut output_axis = 0;
            let mut operand = |formal, bindings: &[Option<usize>]| {
                let axes = bindings
                    .iter()
                    .map(|binding| match binding {
                        Some(slot) => ComponentIndex::Fixed(coordinates[*slot]),
                        None => {
                            let axis = ComponentIndex::Result(output_axis);
                            output_axis += 1;
                            axis
                        }
                    })
                    .collect();
                builder.push(CalculusNode::FormalComponent { formal, axes })
            };
            let a = operand(0, &left_bound)?;
            let b = operand(1, &right_bound)?;
            let product = builder.push(CalculusNode::Mul(a, b))?;
            root = Some(match root {
                None => product,
                Some(sum) => builder.push(CalculusNode::Add(sum, product))?,
            });
        }
        builder.finish(root.ok_or(PureOperatorError::InvalidNode)?)
    }
}
