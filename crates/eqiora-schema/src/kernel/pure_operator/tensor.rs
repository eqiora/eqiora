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
    /// Bilinear three-dimensional cross product in the right-handed Cartesian frame.
    /// Complex inputs use the same algebraic products, without conjugation.
    pub fn cross_product() -> Result<Self, PureOperatorError> {
        let vector = class(1, 3)?;
        let mut builder = CalculusBuilder::new([vector, vector], vector)?;
        let mut root = None;
        for axis in 0..3 {
            let next = (axis + 1) % 3;
            let last = (axis + 2) % 3;
            let mut component = |formal, coordinate| {
                builder.push(CalculusNode::FormalComponent {
                    formal,
                    axes: [ComponentIndex::Fixed(coordinate)].into(),
                })
            };
            let a = component(0, next)?;
            let b = component(1, last)?;
            let c = component(0, last)?;
            let d = component(1, next)?;
            let positive = builder.push(CalculusNode::Mul(a, b))?;
            let negative = builder.push(CalculusNode::Mul(c, d))?;
            let negative = builder.push(CalculusNode::Neg(negative))?;
            let difference = builder.push(CalculusNode::Add(positive, negative))?;
            let selector = builder.push(CalculusNode::KroneckerDelta(
                ComponentIndex::Result(0),
                ComponentIndex::Fixed(axis),
            ))?;
            let term = builder.push(CalculusNode::Mul(selector, difference))?;
            root = Some(match root {
                None => term,
                Some(sum) => builder.push(CalculusNode::Add(sum, term))?,
            });
        }
        builder.finish(root.expect("three Cartesian axes"))
    }

    /// Curl of a physical gradient, whose final axis is the derivative axis.
    /// In 3D this maps grad(vector) to a vector; in 2D it maps grad(vector)
    /// to the oriented scalar, or grad(scalar) to (partial_y, -partial_x).
    /// The caller retains the physical gradient node and its exact support.
    pub fn curl_from_gradient(
        dimensions: u32,
        operand_rank: u16,
    ) -> Result<Self, PureOperatorError> {
        match (dimensions, operand_rank) {
            (3, 1) => {
                let mut builder = CalculusBuilder::new([class(2, 3)?], class(1, 3)?)?;
                let mut root = None;
                for axis in 0..3 {
                    let j = (axis + 1) % 3;
                    let k = (axis + 2) % 3;
                    let positive = builder.push(CalculusNode::FormalComponent {
                        formal: 0,
                        axes: [ComponentIndex::Fixed(k), ComponentIndex::Fixed(j)].into(),
                    })?;
                    let negative = builder.push(CalculusNode::FormalComponent {
                        formal: 0,
                        axes: [ComponentIndex::Fixed(j), ComponentIndex::Fixed(k)].into(),
                    })?;
                    let negative = builder.push(CalculusNode::Neg(negative))?;
                    let difference = builder.push(CalculusNode::Add(positive, negative))?;
                    let selector = builder.push(CalculusNode::KroneckerDelta(
                        ComponentIndex::Result(0),
                        ComponentIndex::Fixed(axis),
                    ))?;
                    let term = builder.push(CalculusNode::Mul(selector, difference))?;
                    root = Some(match root {
                        None => term,
                        Some(sum) => builder.push(CalculusNode::Add(sum, term))?,
                    });
                }
                builder.finish(root.expect("three axes"))
            }
            (2, 1) => {
                let mut builder = CalculusBuilder::new([class(2, 2)?], class(0, 2)?)?;
                let positive = builder.push(CalculusNode::FormalComponent {
                    formal: 0,
                    axes: [ComponentIndex::Fixed(1), ComponentIndex::Fixed(0)].into(),
                })?;
                let negative = builder.push(CalculusNode::FormalComponent {
                    formal: 0,
                    axes: [ComponentIndex::Fixed(0), ComponentIndex::Fixed(1)].into(),
                })?;
                let negative = builder.push(CalculusNode::Neg(negative))?;
                let root = builder.push(CalculusNode::Add(positive, negative))?;
                builder.finish(root)
            }
            (2, 0) => planar_rotation(),
            _ => Err(PureOperatorError::FormalTypeMismatch),
        }
    }

    /// Lift a vector so contraction of its final axis with outward n yields n cross u.
    /// In 2D the normal contraction is n_x*u_y - n_y*u_x, an oriented scalar.
    /// Boundary scope and exact parent identity remain owned by the normal node.
    pub fn tangential_lift(dimensions: u32) -> Result<Self, PureOperatorError> {
        match dimensions {
            2 => planar_rotation(),
            3 => {
                let mut builder = CalculusBuilder::new([class(1, 3)?], class(2, 3)?)?;
                let mut root = None;
                for axis in 0..3 {
                    let j = (axis + 1) % 3;
                    let k = (axis + 2) % 3;
                    let value = builder.push(CalculusNode::FormalComponent {
                        formal: 0,
                        axes: [ComponentIndex::Fixed(k)].into(),
                    })?;
                    let mut delta = |output, coordinate| {
                        builder.push(CalculusNode::KroneckerDelta(
                            ComponentIndex::Result(output),
                            ComponentIndex::Fixed(coordinate),
                        ))
                    };
                    let ij = [delta(0, axis)?, delta(1, j)?];
                    let ji = [delta(0, j)?, delta(1, axis)?];
                    let positive = builder.push(CalculusNode::Mul(ij[0], ij[1]))?;
                    let negative = builder.push(CalculusNode::Mul(ji[0], ji[1]))?;
                    let negative = builder.push(CalculusNode::Neg(negative))?;
                    let selector = builder.push(CalculusNode::Add(positive, negative))?;
                    let term = builder.push(CalculusNode::Mul(selector, value))?;
                    root = Some(match root {
                        None => term,
                        Some(sum) => builder.push(CalculusNode::Add(sum, term))?,
                    });
                }
                builder.finish(root.expect("three axes"))
            }
            _ => Err(PureOperatorError::FormalTypeMismatch),
        }
    }

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

fn planar_rotation() -> Result<PureOperatorDefinition, PureOperatorError> {
    let mut builder = CalculusBuilder::new([class(1, 2)?], class(1, 2)?)?;
    let first = builder.push(CalculusNode::FormalComponent {
        formal: 0,
        axes: [ComponentIndex::Fixed(1)].into(),
    })?;
    let second = builder.push(CalculusNode::FormalComponent {
        formal: 0,
        axes: [ComponentIndex::Fixed(0)].into(),
    })?;
    let second = builder.push(CalculusNode::Neg(second))?;
    let zero = builder.push(CalculusNode::KroneckerDelta(
        ComponentIndex::Result(0),
        ComponentIndex::Fixed(0),
    ))?;
    let one = builder.push(CalculusNode::KroneckerDelta(
        ComponentIndex::Result(0),
        ComponentIndex::Fixed(1),
    ))?;
    let first = builder.push(CalculusNode::Mul(zero, first))?;
    let second = builder.push(CalculusNode::Mul(one, second))?;
    let root = builder.push(CalculusNode::Add(first, second))?;
    builder.finish(root)
}
