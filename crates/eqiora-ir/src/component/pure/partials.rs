//! Differentiate independent local inputs before binding caller expressions.
use super::*;
use eqiora_core::DimExponents;

impl<I: Clone + Eq> ComponentDagLowering<'_, I> {
    pub(super) fn lower_formal_partial(
        &mut self,
        state: &mut PureComponent<'_>,
        source: CalculusNodeId,
        wrt: CalculusNodeId,
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let dimension = |node| {
            state
                .definition
                .value_type(node)
                .map(|ty| ty.dimension())
                .map_err(|error| invalid_component_ir(error.to_string()))
        };
        let output_dimension = dimension(source)?
            .div(dimension(wrt)?)
            .ok_or_else(|| invalid_component_ir("formal partial dimension overflows"))?;
        let selected_formal = match &state.definition.nodes()[wrt.index() as usize] {
            CalculusNode::FormalComponent { formal, axes } if axes.is_empty() => {
                Some(usize::from(*formal))
            }
            CalculusNode::BoundInput(_) => None,
            _ => {
                return Err(invalid_component_ir(
                    "formal partial input has no local identity",
                ));
            }
        };
        // Capture outer values before replacing the builder. Nested partials bind
        // to their parent's independent inputs, not directly to actual arguments.
        let mut bindings = Vec::new();
        for (index, class) in state.definition.formals().iter().enumerate() {
            let dimension = class
                .dimension()
                .ok_or_else(|| invalid_component_ir("formal input has no concrete dimension"))?;
            let value = if let Some(values) = &state.formal_values {
                values[index]
            } else {
                self.lower_part(state.arguments[index], &[], ScalarPart::Real)?
            };
            bindings.push((value, dimension));
        }
        let mut bound_nodes = Vec::new();
        for &(node, value) in &state.bound_values {
            bound_nodes.push(node);
            bindings.push((value, dimension(node)?));
        }
        if selected_formal.is_none() && !bound_nodes.contains(&wrt) {
            let value = self.lower_calculus_component(state, wrt, ScalarPart::Real)?;
            bound_nodes.push(wrt);
            bindings.push((value, dimension(wrt)?));
        }
        let formal_count = state.definition.formals().len();
        let selected = selected_formal.unwrap_or_else(|| {
            formal_count
                + bound_nodes
                    .iter()
                    .position(|node| *node == wrt)
                    .expect("selected bound input retained")
        });
        let outer = std::mem::replace(&mut self.builder, ScalarInputIrBuilder::new());
        let result = (|| {
            let values = bindings
                .iter()
                .map(|(_, dimension)| self.builder.scoped_input(*dimension))
                .collect::<Result<Vec<_>, _>>()?;
            let mut local = PureComponent {
                definition: state.definition,
                arguments: state.arguments,
                component: state.component,
                component_types: state.component_types,
                remapped: vec![[None; 2]; state.definition.nodes().len()],
                formal_values: Some(values[..formal_count].to_vec()),
                bound_values: bound_nodes
                    .into_iter()
                    .zip(values[formal_count..].iter().copied())
                    .collect(),
            };
            let root = self.lower_calculus_component(&mut local, source, part)?;
            let mut seeds = vec![None; values.len()];
            seeds[selected] = Some(
                self.builder
                    .constant(DynQuantity::new(1., DimExponents::DIMENSIONLESS))?,
            );
            self.builder
                .formal_directional(root, &seeds, output_dimension, |work| {
                    super::super::finite::charge_products(self.finite_products, work)
                })
        })();
        // Restore on every failure as well as success; no local IDs enter the
        // caller's expression caches or input-coordinate inventory.
        let local = std::mem::replace(&mut self.builder, outer);
        let root = result?;
        let roots = self
            .builder
            .compose_scoped(local, &[root], &bindings, |work| {
                super::super::finite::charge_products(self.finite_products, work)
            })?;
        Ok(roots[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::ScalarDomain;
    use eqiora_schema::kernel::ExprDagBuilder;
    use eqiora_schema::kernel::pure_operator::{CalculusBuilder, ExactRational, PureValueClass};
    use eqiora_schema::kernel::typing::{RootContract, SpatialSupport, TypedResidual};

    fn real() -> PureValueClass {
        PureValueClass::invariant_scalar()
            .with_dimension(DimExponents::DIMENSIONLESS)
            .with_scalar_domain(ScalarDomain::Real)
            .unwrap()
    }
    fn formal(builder: &mut CalculusBuilder, formal: u16) -> CalculusNodeId {
        builder
            .push(CalculusNode::FormalComponent {
                formal,
                axes: Box::new([]),
            })
            .unwrap()
    }
    fn evaluate(definition: PureOperatorDefinition, arguments: &[f64]) -> Result<f64, Diagnostic> {
        let mut builder = ExprDagBuilder::new();
        let arguments = arguments
            .iter()
            .map(|value| {
                builder
                    .constant(DynQuantity::new(*value, DimExponents::DIMENSIONLESS))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let root = builder.pure_operator(&definition, arguments).unwrap();
        let typed = TypedResidual::infer(
            builder.finish([root]).unwrap(),
            None::<SpatialSupport<()>>,
            |_| None,
            RootContract::ComponentwiseResidual,
            |_| -> Result<ExpressionType<()>, ()> { unreachable!() },
        )
        .unwrap();
        Ok(ComponentScalarization::lower(&typed)?.evaluate(|_| None)?[0])
    }

    #[test]
    fn complex_response_formal_partials_use_the_existing_component_products() {
        let complex = real().with_scalar_domain(ScalarDomain::Complex).unwrap();
        // f(x)=x/(1+i*x), f'=1/(1+i*x)^2, f''=-2i/(1+i*x)^3.
        // For g=x*f, g'=f+x*f' and g''=2*f'+x*f''.
        for (square, order, expected) in [
            (false, 0, [0.4, -0.8]),
            (false, 1, [-0.12, -0.16]),
            (false, 2, [0.032, 0.176]),
            (true, 0, [0.8, -1.6]),
            (true, 1, [0.16, -1.12]),
            (true, 2, [-0.176, 0.032]),
        ] {
            let mut builder = CalculusBuilder::new([real()], complex).unwrap();
            let x = formal(&mut builder, 0);
            let one = builder
                .push(CalculusNode::Rational {
                    value: ExactRational::integer(1),
                    dimension: DimExponents::DIMENSIONLESS,
                })
                .unwrap();
            let denominator = builder
                .push(CalculusNode::Complex { real: one, imag: x })
                .unwrap();
            let numerator = if square {
                builder.push(CalculusNode::Mul(x, x)).unwrap()
            } else {
                x
            };
            let mut root = builder
                .push(CalculusNode::Div(numerator, denominator))
                .unwrap();
            for _ in 0..order {
                root = builder.partial(root, 0).unwrap();
            }
            let definition = builder.finish(root).unwrap();
            let argument = eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                    .unwrap(),
                2.,
            )
            .unwrap();
            let result =
                ComponentScalarization::evaluate_pure_operator(&definition, &[&argument]).unwrap();
            assert_eq!(result.value_type().scalar_domain(), ScalarDomain::Complex);
            let (re, im) = result.components().unwrap().next().unwrap();
            for (actual, expected) in [re, im].into_iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-14,
                    "order {order}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn a_real_result_retains_complex_intermediates_and_the_body_owns_its_domain() {
        let complex = real().with_scalar_domain(ScalarDomain::Complex).unwrap();
        let mut builder = CalculusBuilder::new([complex], real()).unwrap();
        let z = formal(&mut builder, 0);
        let squared = builder.push(CalculusNode::Mul(z, z)).unwrap();
        let root = builder
            .push(CalculusNode::UnaryMath(
                eqiora_schema::kernel::UnaryMathFunction::Imag,
                squared,
            ))
            .unwrap();
        let definition = builder.finish(root).unwrap();
        let argument = eqiora_core::ValueLiteral::new(
            eqiora_core::ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                .unwrap(),
            [(2., 3.)],
        )
        .unwrap();
        let result =
            ComponentScalarization::evaluate_pure_operator(&definition, &[&argument]).unwrap();
        assert_eq!(result.real_scalar_value().unwrap().value(), 12.);
        // A complex argument which the body does not consume does not change a real output.
        let mut builder = CalculusBuilder::new([real(), complex], real()).unwrap();
        let x = formal(&mut builder, 0);
        let definition = builder.finish(x).unwrap();
        let first = eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .unwrap(),
            7.,
        )
        .unwrap();
        let result =
            ComponentScalarization::evaluate_pure_operator(&definition, &[&first, &argument])
                .unwrap();
        assert_eq!(result.real_scalar_value().unwrap().value(), 7.);
    }

    #[test]
    fn canonical_elementary_first_and_second_partials_reach_shared_numeric_execution() {
        use eqiora_schema::kernel::UnaryMathFunction::*;
        // Analytic values at exact binary inputs, independent of generated instructions.
        for (function, point, derivatives) in [
            (Sin, 0., [1., 0.]),
            (Cos, 0., [0., -1.]),
            (Exp, 0., [1., 1.]),
            (Log, 1., [1., -1.]),
            (Sqrt, 4., [0.25, -0.03125]),
        ] {
            for (order, expected) in derivatives.into_iter().enumerate() {
                let mut builder = CalculusBuilder::new([real()], real()).unwrap();
                let x = formal(&mut builder, 0);
                let mut root = builder.push(CalculusNode::UnaryMath(function, x)).unwrap();
                for _ in 0..=order {
                    root = builder.partial(root, 0).unwrap();
                }
                assert_eq!(
                    evaluate(builder.finish(root).unwrap(), &[point]).unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    fn canonical_quotient_partials_keep_local_identity_range_and_higher_order() {
        let mut builder = CalculusBuilder::new([real(), real()], real()).unwrap();
        let x = formal(&mut builder, 0);
        let y = formal(&mut builder, 1);
        let ratio = builder.push(CalculusNode::Div(x, y)).unwrap();
        let first = builder.partial(ratio, 1).unwrap();
        let definition = builder.finish(first).unwrap();
        assert_eq!(evaluate(definition.clone(), &[2., 4.]).unwrap(), -0.125);
        let actual = evaluate(definition.clone(), &[1e-200, 1e-200]).unwrap();
        // -x/y² = -1e200; a bound of 16 binary64 epsilons covers the finite
        // division/multiplication chain and rounded decimal input/output references.
        assert!((actual / -1e200 - 1.).abs() <= 16. * f64::EPSILON);
        assert!(evaluate(definition.clone(), &[1., 0.]).is_err());
        for (order, expected) in [(1, 0.25), (2, -0.25)] {
            let mut outer = CalculusBuilder::new([real()], real()).unwrap();
            let z = formal(&mut outer, 0);
            let mut root = outer.apply_scalar(&definition, &[z, z]).unwrap();
            // The inner partial is -1/z after binding x=y=z. Differentiating
            // that composed value gives 1/z² and -2/z³, evaluated at z=2.
            for _ in 0..order {
                root = outer.partial(root, 0).unwrap();
            }
            assert_eq!(
                evaluate(outer.finish(root).unwrap(), &[2.]).unwrap(),
                expected
            );
        }
    }

    fn polynomial_partial(order: usize) -> PureOperatorDefinition {
        let mut builder = CalculusBuilder::new([real(), real()], real()).unwrap();
        let x = formal(&mut builder, 0);
        let y = formal(&mut builder, 1);
        let square = builder.push(CalculusNode::Mul(x, x)).unwrap();
        let mut root = builder.push(CalculusNode::Mul(square, y)).unwrap();
        for _ in 0..order {
            root = builder.partial(root, 0).unwrap();
        }
        builder.finish(root).unwrap()
    }

    #[test]
    fn formal_aliases_and_nested_composition_keep_independent_coordinates() {
        let first_definition = polynomial_partial(1);
        assert_eq!(evaluate(first_definition.clone(), &[3., 3.]).unwrap(), 18.);
        assert_eq!(evaluate(polynomial_partial(2), &[3., 3.]).unwrap(), 6.);
        for (order, expected) in [(1, 54.), (2, 36.)] {
            let mut outer = CalculusBuilder::new([real()], real()).unwrap();
            let z = formal(&mut outer, 0);
            let square = outer.push(CalculusNode::Mul(z, z)).unwrap();
            let mut root = outer.apply_scalar(&first_definition, &[z, square]).unwrap();
            // partial_x (x²y) at (z,z²) = 2z³; its first two derivatives are 6z² and 12z.
            for _ in 0..order {
                root = outer.partial(root, 0).unwrap();
            }
            assert_eq!(
                evaluate(outer.finish(root).unwrap(), &[3.]).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn formal_partial_omits_unused_overflow_but_preserves_inclusive_validity() {
        use eqiora_schema::kernel::ComparisonOp;
        for guarded_zero in [None, Some(false), Some(true)] {
            let mut builder = CalculusBuilder::new([real()], real()).unwrap();
            let x = formal(&mut builder, 0);
            let square = builder.push(CalculusNode::Mul(x, x)).unwrap();
            let root = if let Some(is_zero) = guarded_zero {
                let zero = builder
                    .push(CalculusNode::Rational {
                        value: ExactRational::integer(0),
                        dimension: DimExponents::DIMENSIONLESS,
                    })
                    .unwrap();
                let condition = builder
                    .push(CalculusNode::Compare(ComparisonOp::GreaterEqual, x, zero))
                    .unwrap();
                builder
                    .push(CalculusNode::Require {
                        condition,
                        value: if is_zero { zero } else { square },
                    })
                    .unwrap()
            } else {
                square
            };
            let derivative = builder.partial(root, 0).unwrap();
            let definition = builder.finish(derivative).unwrap();
            assert_eq!(evaluate(definition.clone(), &[0.]).unwrap(), 0.);
            if guarded_zero.is_some() {
                assert!(evaluate(definition, &[-1.]).is_err());
            } else {
                assert_eq!(evaluate(definition, &[1e200]).unwrap(), 2e200);
            }
        }
    }
    #[test]
    fn formal_validity_obligations_preserve_lazy_predicates_and_callers() {
        use eqiora_schema::kernel::ComparisonOp;
        for predicate_case in [false, true] {
            let mut builder = CalculusBuilder::new([real()], real()).unwrap();
            let x = formal(&mut builder, 0);
            let zero = builder
                .push(CalculusNode::Rational {
                    value: ExactRational::integer(0),
                    dimension: DimExponents::DIMENSIONLESS,
                })
                .unwrap();
            let nonnegative = builder
                .push(CalculusNode::Compare(ComparisonOp::GreaterEqual, x, zero))
                .unwrap();
            let condition = if predicate_case {
                let denied = builder.push(CalculusNode::Boolean(false)).unwrap();
                let invalid = builder
                    .push(CalculusNode::Require {
                        condition: denied,
                        value: x,
                    })
                    .unwrap();
                let comparison = builder
                    .push(CalculusNode::Compare(
                        ComparisonOp::GreaterEqual,
                        invalid,
                        zero,
                    ))
                    .unwrap();
                builder
                    .push(CalculusNode::Or(nonnegative, comparison))
                    .unwrap()
            } else {
                nonnegative
            };
            let square = builder.push(CalculusNode::Mul(x, x)).unwrap();
            let guarded = builder
                .push(CalculusNode::Require {
                    condition,
                    value: square,
                })
                .unwrap();
            let derivative = builder.partial(guarded, 0).unwrap();
            let root = if predicate_case {
                derivative
            } else {
                builder
                    .push(CalculusNode::Select {
                        condition: nonnegative,
                        then_value: derivative,
                        else_value: zero,
                    })
                    .unwrap()
            };
            let definition = builder.finish(root).unwrap();
            assert_eq!(evaluate(definition.clone(), &[1.]).unwrap(), 2.);
            if !predicate_case {
                assert_eq!(evaluate(definition, &[-1.]).unwrap(), 0.);
            }
        }
    }
}
