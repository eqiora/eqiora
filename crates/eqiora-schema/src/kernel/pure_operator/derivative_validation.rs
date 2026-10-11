//! Revalidate retained derivative values with the same bounded ordered transform.
use super::derivative::Selection;
use super::*;

impl CalculusBuilder {
    pub(super) fn validate_differentiated(
        &self,
        source: CalculusNodeId,
        wrt: CalculusNodeId,
        value: CalculusNodeId,
    ) -> Result<(), PureOperatorError> {
        let selected = self.value_type(wrt)?;
        let output = self.value_type(source)?;
        let selection = match self.nodes.get(wrt.index() as usize) {
            Some(CalculusNode::FormalComponent { formal, axes }) if axes.is_empty() => {
                Selection::Formal(*formal)
            }
            Some(CalculusNode::BoundInput(_)) => Selection::Occurrence(wrt),
            _ => return Err(PureOperatorError::FormalTypeMismatch),
        };
        if [selected.clone(), output.clone()].iter().any(|ty| {
            !ty.shape().is_scalar() || ty.scalar_domain() != eqiora_core::ScalarDomain::Real
        }) {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let dimension = output
            .dimension()
            .div(selected.dimension())
            .ok_or(PureOperatorError::ResultDimensionOverflow)?;
        // The source prefix contains every source operand; excluding the already
        // expanded result keeps replay within the same node bound as construction.
        let count = source.index() as usize + 1;
        let prefix = Self {
            formals: self.formals.clone(),
            result: self.result,
            nodes: self.nodes[..count].to_vec(),
            depths: self.depths[..count].to_vec(),
        };
        let (derived, expected) =
            prefix.raw_partial(source, selection, selected.dimension(), dimension)?;
        if equal_value(&self.nodes, value, &derived.nodes, expected, count) {
            Ok(())
        } else {
            Err(PureOperatorError::DerivativeMismatch)
        }
    }
}

fn equal_value(
    left: &[CalculusNode],
    root: CalculusNodeId,
    right: &[CalculusNode],
    expected: CalculusNodeId,
    prefix: usize,
) -> bool {
    let mut pending = vec![(root, expected)];
    let mut seen = std::collections::BTreeSet::new();
    while let Some((a, b)) = pending.pop() {
        if a == b && (a.index() as usize) < prefix {
            continue;
        }
        if !seen.insert((a, b)) {
            continue;
        }
        if seen.len() > MAX_NODES * 4 {
            return false;
        }
        let (a, b) = (&left[a.index() as usize], &right[b.index() as usize]);
        match (a, b) {
            (
                CalculusNode::Rational {
                    value: a,
                    dimension: ad,
                },
                CalculusNode::Rational {
                    value: b,
                    dimension: bd,
                },
            ) if a == b && ad == bd => {}
            (
                CalculusNode::FormalComponent {
                    formal: a,
                    axes: aa,
                },
                CalculusNode::FormalComponent {
                    formal: b,
                    axes: ba,
                },
            ) if a == b && aa == ba => {}
            (CalculusNode::Neg(a), CalculusNode::Neg(b))
            | (CalculusNode::BoundInput(a), CalculusNode::BoundInput(b)) => pending.push((*a, *b)),
            (CalculusNode::Add(a, b), CalculusNode::Add(c, d))
            | (CalculusNode::Mul(a, b), CalculusNode::Mul(c, d)) => {
                pending.extend([(*a, *c), (*b, *d)])
            }
            (
                CalculusNode::Require {
                    condition: a,
                    value: b,
                },
                CalculusNode::Require {
                    condition: c,
                    value: d,
                },
            ) => pending.extend([(*a, *c), (*b, *d)]),
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forged_derivative_value_rejects_before_becoming_a_kernel_node() {
        let class = PureValueClass::invariant_scalar()
            .with_dimension(DimExponents::DIMENSIONLESS)
            .with_scalar_domain(eqiora_core::ScalarDomain::Real)
            .unwrap();
        let mut builder = CalculusBuilder::new([class], class).unwrap();
        let x = builder
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let square = builder.push(CalculusNode::Mul(x, x)).unwrap();
        let zero = builder
            .push(CalculusNode::Rational {
                value: ExactRational::integer(0),
                dimension: DimExponents::DIMENSIONLESS,
            })
            .unwrap();
        let before = builder.nodes.clone();
        assert_eq!(
            builder.push(CalculusNode::Differentiated {
                source: square,
                wrt: x,
                value: zero
            }),
            Err(PureOperatorError::DerivativeMismatch)
        );
        assert_eq!(builder.nodes, before);
        let first = builder.partial(square, 0).unwrap();
        builder.finish(first).unwrap();
    }
}
