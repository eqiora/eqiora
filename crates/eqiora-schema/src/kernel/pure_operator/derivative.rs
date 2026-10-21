//! Formal partials retain only the mathematical source and selected input.
use super::*;
use eqiora_core::ScalarDomain;

impl PureOperatorDefinition {
    /// Differentiate a closed component definition without substituting its
    /// inputs or discarding its result axes, frame or nominal basis.
    pub(crate) fn formal_partial(&self, formal: u16) -> Result<Self, PureOperatorError> {
        let input = self
            .formals
            .get(usize::from(formal))
            .ok_or(PureOperatorError::InvalidFormal(formal))?;
        let (output_dimension, input_dimension) = self
            .result
            .dimension()
            .zip(input.dimension())
            .ok_or(PureOperatorError::FormalTypeMismatch)?;
        let dimension = output_dimension
            .div(input_dimension)
            .ok_or(PureOperatorError::ResultDimensionOverflow)?;
        let mut builder = CalculusBuilder::new(
            self.formals.iter().cloned(),
            self.result.clone().with_dimension(dimension),
        )?;
        for node in &self.nodes {
            builder.push(node.clone())?;
        }
        let partial = builder.partial(self.root, formal)?;
        builder.finish(partial)
    }
}

impl CalculusBuilder {
    /// Append a scalar partial with respect to an independent real input.
    /// Real or complex responses retain the other numeric formal inputs fixed,
    /// including every component of shaped and complex inputs.
    /// The source, selected input, and derivative order remain canonical. Numerical
    /// execution and optional polynomial proof expansion use their respective owners.
    /// Explicit validity predicates are retained, never differentiated.
    ///
    /// # Errors
    /// Rejects invalid formals, a selected input that is not a real scalar,
    /// unresolved or nonnumeric input types, nonsmooth values,
    /// unrepresentable dimensions, and node or depth limits. Errors append nothing.
    pub fn partial(
        &mut self,
        root: CalculusNodeId,
        formal: u16,
    ) -> Result<CalculusNodeId, PureOperatorError> {
        self.formals
            .get(usize::from(formal))
            .ok_or(PureOperatorError::InvalidFormal(formal))?;
        definition_index(root, self.nodes.len())?;
        let mut staged = Self {
            formals: self.formals.clone(),
            result: self.result.clone(),
            nodes: self.nodes.clone(),
            depths: self.depths.clone(),
        };
        let wrt = staged.push(CalculusNode::FormalComponent {
            formal,
            axes: Box::new([]),
        })?;
        let result = staged.push(CalculusNode::Differentiated { source: root, wrt })?;
        *self = staged;
        Ok(result)
    }

    pub(super) fn validate_differentiated(
        &self,
        source: CalculusNodeId,
        wrt: CalculusNodeId,
    ) -> Result<(), PureOperatorError> {
        if self.formals.iter().any(|class| {
            class.dimension().is_none()
                || !matches!(
                    class.scalar_domain(),
                    Some(ScalarDomain::Real | ScalarDomain::Complex)
                )
        }) {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        match &self.nodes[definition_index(wrt, self.nodes.len())?] {
            CalculusNode::FormalComponent { axes, .. } if axes.is_empty() => {}
            CalculusNode::BoundInput(_) => {}
            _ => return Err(PureOperatorError::FormalTypeMismatch),
        }
        let selected = self.value_type(wrt)?;
        let output = self.value_type(source)?;
        if !selected.shape().is_scalar()
            || selected.scalar_domain() != ScalarDomain::Real
            || !output.shape().is_scalar()
            || !matches!(
                output.scalar_domain(),
                ScalarDomain::Real | ScalarDomain::Complex
            )
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        output
            .dimension()
            .div(selected.dimension())
            .ok_or(PureOperatorError::ResultDimensionOverflow)?;
        let mut pending = vec![source];
        let mut seen = vec![false; self.nodes.len()];
        while let Some(id) = pending.pop() {
            let index = definition_index(id, self.nodes.len())?;
            if std::mem::replace(&mut seen[index], true) {
                continue;
            }
            if id == wrt && matches!(self.nodes[index], CalculusNode::BoundInput(_)) {
                continue;
            }
            match &self.nodes[index] {
                // Predicates retain their authored domain, including lazy branches.
                CalculusNode::Require { value, .. } => pending.push(*value),
                CalculusNode::Select { .. }
                | CalculusNode::UnaryMath(crate::kernel::UnaryMathFunction::Abs, _) => {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
                node => pending.extend(node.operands()),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn real_unit() -> PureValueClass {
        PureValueClass::invariant_scalar()
            .with_dimension(DimExponents::DIMENSIONLESS)
            .with_scalar_domain(ScalarDomain::Real)
            .unwrap()
    }

    #[test]
    fn formal_partials_reject_nonsmooth_values_but_retain_predicate_domains() {
        let mut builder = CalculusBuilder::new([real_unit()], real_unit()).unwrap();
        let x = builder
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let absolute = builder
            .push(CalculusNode::UnaryMath(
                crate::kernel::UnaryMathFunction::Abs,
                x,
            ))
            .unwrap();
        // x|x| has different second derivatives on the two sides of zero.
        // Formal admission cannot infer smoothness from a later sampled value.
        let value = builder.push(CalculusNode::Mul(x, absolute)).unwrap();
        let before = builder.nodes.clone();
        assert_eq!(
            builder.partial(value, 0),
            Err(PureOperatorError::FormalTypeMismatch)
        );
        assert_eq!(builder.nodes, before);

        let one = builder
            .push(CalculusNode::Rational {
                value: ExactRational::integer(1),
                dimension: DimExponents::DIMENSIONLESS,
            })
            .unwrap();
        let condition = builder
            .push(CalculusNode::Compare(
                crate::kernel::ComparisonOp::Less,
                absolute,
                one,
            ))
            .unwrap();
        let guarded = builder
            .push(CalculusNode::Require {
                condition,
                value: x,
            })
            .unwrap();
        assert!(builder.partial(guarded, 0).is_ok());
    }

    #[test]
    fn formal_partials_do_not_require_a_polynomial_witness() {
        for unary in [None, Some(crate::kernel::UnaryMathFunction::Log)] {
            let mut builder = CalculusBuilder::new([real_unit()], real_unit()).unwrap();
            let x = builder
                .push(CalculusNode::FormalComponent {
                    formal: 0,
                    axes: Box::new([]),
                })
                .unwrap();
            let one = builder
                .push(CalculusNode::Rational {
                    value: ExactRational::integer(1),
                    dimension: DimExponents::DIMENSIONLESS,
                })
                .unwrap();
            let root = if let Some(function) = unary {
                builder.push(CalculusNode::UnaryMath(function, x)).unwrap()
            } else {
                builder.push(CalculusNode::Div(one, x)).unwrap()
            };
            let before = builder.nodes.len();
            let first = builder.partial(root, 0).unwrap();
            assert_eq!(
                builder.nodes.len(),
                before + 2,
                "only selection and derivative are stored"
            );
            assert!(
                matches!(builder.nodes[first.index() as usize], CalculusNode::Differentiated { source, .. } if source == root)
            );
            let second = builder.partial(first, 0).unwrap();
            let definition = builder.finish(second).unwrap();
            assert!(definition.polynomial_derivative_view().is_err());
        }
    }

    #[test]
    fn invalid_derivative_source_and_selection_reject_atomically() {
        let mut builder = CalculusBuilder::new([real_unit()], real_unit()).unwrap();
        let x = builder
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let one = builder
            .push(CalculusNode::Rational {
                value: ExactRational::integer(1),
                dimension: DimExponents::DIMENSIONLESS,
            })
            .unwrap();
        let before = builder.nodes.clone();
        assert_eq!(
            builder.partial(CalculusNodeId(before.len() as u32), 0),
            Err(PureOperatorError::InvalidNode)
        );
        assert_eq!(
            builder.push(CalculusNode::Differentiated {
                source: x,
                wrt: one
            }),
            Err(PureOperatorError::FormalTypeMismatch)
        );
        assert_eq!(builder.nodes, before);
    }

    #[test]
    fn partial_retains_validity_for_both_variable_and_typed_zero_derivatives() {
        let t = DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).unwrap();
        let class = |d| {
            PureValueClass::invariant_scalar()
                .with_dimension(d)
                .with_scalar_domain(ScalarDomain::Real)
                .unwrap()
        };
        let mut builder = CalculusBuilder::new([class(t), class(t)], class(t)).unwrap();
        let x = builder
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let zero = builder
            .push(CalculusNode::Rational {
                value: ExactRational::integer(0),
                dimension: t,
            })
            .unwrap();
        let predicate = builder
            .push(CalculusNode::Compare(
                crate::kernel::ComparisonOp::Greater,
                x,
                zero,
            ))
            .unwrap();
        let root = builder
            .push(CalculusNode::Require {
                condition: predicate,
                value: x,
            })
            .unwrap();
        for formal in [0, 1] {
            let derived = builder.partial(root, formal).unwrap();
            let CalculusNode::Differentiated { source, .. } =
                builder.nodes[derived.index() as usize]
            else {
                panic!("derivative history retained")
            };
            assert_eq!(source, root);
            let definition = CalculusBuilder {
                formals: builder.formals.clone(),
                result: real_unit(),
                nodes: builder.nodes.clone(),
                depths: builder.depths.clone(),
            }
            .finish(derived)
            .unwrap();
            let view = definition.polynomial_derivative_view().unwrap();
            let CalculusNode::Require { condition, value } = view.nodes[view.root.index() as usize]
            else {
                panic!("guard retained")
            };
            assert_eq!(condition, predicate);
            assert!(
                matches!(view.nodes[value.index() as usize], CalculusNode::Rational { ref value, dimension }
                if *value == ExactRational::integer(if formal == 0 { 1 } else { 0 }) && dimension == DimExponents::DIMENSIONLESS)
            );
        }
    }
    #[test]
    fn ordered_history_survives_zero_results_beyond_second_order() {
        let class = PureValueClass::invariant_scalar()
            .with_dimension(DimExponents::DIMENSIONLESS)
            .with_scalar_domain(ScalarDomain::Real)
            .unwrap();
        let mut builder =
            CalculusBuilder::new([class.clone(), class.clone()], class.clone()).unwrap();
        let x = builder
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let first = builder.partial(x, 1).unwrap();
        let second = builder.partial(first, 0).unwrap();
        let CalculusNode::Differentiated { source, wrt, .. } =
            builder.nodes[second.index() as usize]
        else {
            panic!("second history")
        };
        assert_eq!(source, first);
        assert!(matches!(
            builder.nodes[wrt.index() as usize],
            CalculusNode::FormalComponent { formal: 0, .. }
        ));
        let CalculusNode::Differentiated { source, wrt, .. } =
            builder.nodes[first.index() as usize]
        else {
            panic!("first history")
        };
        assert_eq!(source, x);
        assert!(matches!(
            builder.nodes[wrt.index() as usize],
            CalculusNode::FormalComponent { formal: 1, .. }
        ));
        let mut current = second;
        for _ in 3..=16 {
            let previous = current;
            current = builder.partial(previous, 0).unwrap();
            let CalculusNode::Differentiated { source, .. } =
                builder.nodes[current.index() as usize]
            else {
                panic!("ordered derivative history");
            };
            assert_eq!(source, previous);
        }
        let definition = builder.finish(current).unwrap();
        let view = definition.polynomial_derivative_view().unwrap();
        assert!(
            matches!(&view.nodes[view.root.index() as usize], CalculusNode::Rational { value, dimension } if *value == ExactRational::integer(0) && *dimension == DimExponents::DIMENSIONLESS)
        );
    }
}
