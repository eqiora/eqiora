//! Canonical formal derivatives, independently checked in physical units.
use eqiora_core::{DimExponents, DynQuantity, ScalarDomain, ValueLiteral, ValueType};
use eqiora_ir::{
    CalculusBuilder, CalculusNode, ExactRational, PureOperatorDefinition, PureOperatorError,
    PureValueClass, ScalarOperatorIr,
};
use eqiora_schema::kernel::{ExprDag, ExprDagBuilder};

fn temperature() -> DimExponents {
    DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).unwrap()
}
fn real(dimension: DimExponents) -> PureValueClass {
    PureValueClass::invariant_scalar()
        .with_dimension(dimension)
        .with_scalar_domain(ScalarDomain::Real)
        .unwrap()
}
fn formal(builder: &mut CalculusBuilder, formal: u16) -> eqiora_ir::CalculusNodeId {
    builder
        .push(CalculusNode::FormalComponent {
            formal,
            axes: Box::new([]),
        })
        .unwrap()
}
fn derived(
    definition: &PureOperatorDefinition,
    inputs: &[PureValueClass],
    output: DimExponents,
    selected: u16,
    order: usize,
) -> Result<PureOperatorDefinition, PureOperatorError> {
    let mut builder = CalculusBuilder::new(inputs.iter().cloned(), real(output))?;
    let arguments = (0..inputs.len())
        .map(|slot| formal(&mut builder, slot as u16))
        .collect::<Vec<_>>();
    let mut root = builder.apply_inputs(
        definition,
        &arguments
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<_>>(),
    )?;
    for _ in 0..order {
        root = builder.partial(root, selected)?;
    }
    builder.finish(root)
}
fn evaluate(dag: &ExprDag) -> Vec<ValueLiteral> {
    ScalarOperatorIr::lower(dag)
        .unwrap()
        .evaluate_typed(dag.roots(), &mut |_| None)
        .unwrap()
}

#[test]
fn dimensioned_linear_operator_has_typed_zero_higher_partials() {
    let t = temperature();
    let mut definition = CalculusBuilder::new([real(t)], real(t)).unwrap();
    let x = formal(&mut definition, 0);
    let two = definition
        .push(CalculusNode::Rational {
            value: ExactRational::integer(2),
            dimension: DimExponents::DIMENSIONLESS,
        })
        .unwrap();
    let root = definition.push(CalculusNode::Mul(two, x)).unwrap();
    let definition = definition.finish(root).unwrap();
    assert!(derived(&definition, &[real(t)], DimExponents::DIMENSIONLESS, 1, 1).is_err());
    for (order, expected, dimension) in [
        (1, 2., DimExponents::DIMENSIONLESS),
        (2, 0., t.pow(-1, 1).unwrap()),
        (3, 0., t.pow(-2, 1).unwrap()),
    ] {
        let partial = derived(&definition, &[real(t)], dimension, 0, order).unwrap();
        let mut dag = ExprDagBuilder::new();
        let x = dag.constant(DynQuantity::new(20., t)).unwrap();
        let root = dag.pure_operator(&partial, [x]).unwrap();
        let value = evaluate(&dag.finish([root]).unwrap()).remove(0);
        assert_eq!(value.real_scalar_value().unwrap().value(), expected);
        assert_eq!(value.value_type().dimension(), dimension);
    }
}

#[test]
fn retained_scalar_application_uses_typed_execution_and_rejects_unprojected_ad_or_complex() {
    let scalar = PureValueClass::invariant_scalar();
    let mut definition = CalculusBuilder::new([scalar.clone()], scalar).unwrap();
    let x = formal(&mut definition, 0);
    let square = definition.push(CalculusNode::Mul(x, x)).unwrap();
    let definition = definition.finish(square).unwrap();
    let mut builder = ExprDagBuilder::new();
    let x = builder
        .constant(DynQuantity::new(3., temperature()))
        .unwrap();
    let root = builder.pure_operator(&definition, [x]).unwrap();
    let dag = builder.finish([root]).unwrap();
    let ir = ScalarOperatorIr::lower(&dag).unwrap();
    let value = ir.evaluate_typed(&[root], &mut |_| None).unwrap().remove(0);
    assert_eq!(value.real_scalar_value().unwrap().value(), 9.);
    assert_eq!(
        value.value_type().dimension(),
        temperature().pow(2, 1).unwrap()
    );
    assert!(ir.evaluate(&[]).is_err());
    assert!(ir.linearize(&[], &[]).is_err());
    let complex = real(DimExponents::DIMENSIONLESS)
        .with_scalar_domain(ScalarDomain::Complex)
        .unwrap();
    assert!(derived(&definition, &[complex], DimExponents::DIMENSIONLESS, 0, 1).is_err());
    let complex = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap();
    let mut builder = ExprDagBuilder::new();
    let x = builder
        .constant(ValueLiteral::new(complex, [(1., 2.)]).unwrap())
        .unwrap();
    let root = builder.pure_operator(&definition, [x]).unwrap();
    let dag = builder.finish([root]).unwrap();
    assert!(
        ScalarOperatorIr::lower(&dag)
            .unwrap()
            .evaluate_typed(&[root], &mut |_| None)
            .is_err()
    );
}

#[test]
fn finite_partial_does_not_evaluate_an_unused_overflowing_primal() {
    let unit = DimExponents::DIMENSIONLESS;
    let mut definition = CalculusBuilder::new([real(unit)], real(unit)).unwrap();
    let x = formal(&mut definition, 0);
    let square = definition.push(CalculusNode::Mul(x, x)).unwrap();
    let definition = definition.finish(square).unwrap();
    let mut primal = ExprDagBuilder::new();
    let x = primal.constant(DynQuantity::new(1e200, unit)).unwrap();
    let root = primal.pure_operator(&definition, [x]).unwrap();
    let dag = primal.finish([root]).unwrap();
    assert!(
        ScalarOperatorIr::lower(&dag)
            .unwrap()
            .evaluate_typed(&[root], &mut |_| None)
            .is_err()
    );
    for (order, expected) in [(1, 2e200), (2, 2.)] {
        let partial = derived(&definition, &[real(unit)], unit, 0, order).unwrap();
        let mut dag = ExprDagBuilder::new();
        let x = dag.constant(DynQuantity::new(1e200, unit)).unwrap();
        let root = dag.pure_operator(&partial, [x]).unwrap();
        assert_eq!(
            evaluate(&dag.finish([root]).unwrap())[0]
                .real_scalar_value()
                .unwrap()
                .value(),
            expected
        );
    }
}

#[test]
fn partial_keeps_live_product_rule_order_instead_of_floating_reassociation() {
    let unit = DimExponents::DIMENSIONLESS;
    let mut definition = CalculusBuilder::new([real(unit)], real(unit)).unwrap();
    let x = formal(&mut definition, 0);
    let big = definition
        .push(CalculusNode::Rational {
            value: ExactRational::integer(10_000_000_000_000_000),
            dimension: unit,
        })
        .unwrap();
    let product = definition.push(CalculusNode::Mul(big, x)).unwrap();
    let sum = definition.push(CalculusNode::Add(product, x)).unwrap();
    let negative = definition.push(CalculusNode::Neg(product)).unwrap();
    let root = definition.push(CalculusNode::Add(sum, negative)).unwrap();
    let definition = definition.finish(root).unwrap();
    let partial = derived(&definition, &[real(unit)], unit, 0, 1).unwrap();
    let mut dag = ExprDagBuilder::new();
    let x = dag.constant(DynQuantity::new(1., unit)).unwrap();
    let root = dag.pure_operator(&partial, [x]).unwrap();
    // Binary64 evaluates (1e16+1)-1e16 as zero. Exact normalization to one is a proof view.
    assert_eq!(
        evaluate(&dag.finish([root]).unwrap())[0]
            .real_scalar_value()
            .unwrap()
            .value(),
        0.
    );
}

#[test]
fn tensor_applications_still_require_existing_component_expansion() {
    use eqiora_schema::kernel::SymbolRef;
    let mut builder = ExprDagBuilder::new();
    let tensor = builder
        .symbol(SymbolRef::Field(eqiora_core::Id::new()))
        .unwrap();
    let root = builder
        .pure_operator(&PureOperatorDefinition::symmetric_part().unwrap(), [tensor])
        .unwrap();
    let dag = builder.finish([root]).unwrap();
    assert!(
        ScalarOperatorIr::lower(&dag)
            .unwrap_err()
            .to_string()
            .contains("component expansion")
    );
}

#[test]
fn derived_applications_reject_unavailable_unused_arguments_atomically() {
    let unit = DimExponents::DIMENSIONLESS;
    let mut definition = CalculusBuilder::new(vec![real(unit); 2], real(unit)).unwrap();
    let x = formal(&mut definition, 0);
    let definition = definition.finish(x).unwrap();
    let mut foreign = ExprDagBuilder::new();
    foreign.constant(DynQuantity::new(0., unit)).unwrap();
    let unavailable = foreign.constant(DynQuantity::new(1., unit)).unwrap();
    for order in [1, 2] {
        let partial = derived(&definition, &[real(unit), real(unit)], unit, 0, order).unwrap();
        let mut destination = ExprDagBuilder::new();
        let existing = destination.constant(DynQuantity::new(7., unit)).unwrap();
        assert!(
            destination
                .pure_operator(&partial, [existing, unavailable])
                .is_err()
        );
        assert_eq!(destination.finish([existing]).unwrap().nodes().len(), 1);
    }
}

#[test]
fn scalar_jacobian_actions_and_hessian_directions_match_independent_polynomial_values() {
    let unit = DimExponents::DIMENSIONLESS;
    let class = PureValueClass::invariant_scalar()
        .with_dimension(unit)
        .with_scalar_domain(ScalarDomain::Real)
        .unwrap();
    // At x=3,y=5, f=x²y+y³ has gradient (30,84), H=[[10,6],[6,30]].
    // Direction (2,-1) gives Jv=-24 and Hv=(14,-18). Seed 7 gives
    // Jᵀ7=(210,588), and both dual pairings equal -168.
    let expected = [-24., 210., 588., 14., -18., -168., -168.];
    let mut dag = ExprDagBuilder::new();
    let arguments = [3., 5., 2., -1., 7.]
        .into_iter()
        .map(|x| dag.constant(DynQuantity::new(x, unit)).unwrap())
        .collect::<Vec<_>>();
    let mut roots = Vec::new();
    for selection in 0..expected.len() {
        let mut builder = CalculusBuilder::new(vec![class.clone(); 5], class.clone()).unwrap();
        let inputs = (0..5)
            .map(|formal| {
                builder
                    .push(CalculusNode::FormalComponent {
                        formal,
                        axes: Box::new([]),
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let xx = builder
            .push(CalculusNode::Mul(inputs[0], inputs[0]))
            .unwrap();
        let xxy = builder.push(CalculusNode::Mul(xx, inputs[1])).unwrap();
        let yy = builder
            .push(CalculusNode::Mul(inputs[1], inputs[1]))
            .unwrap();
        let yyy = builder.push(CalculusNode::Mul(yy, inputs[1])).unwrap();
        let f = builder.push(CalculusNode::Add(xxy, yyy)).unwrap();
        let directions = [(0, inputs[2]), (1, inputs[3])];
        let jv = builder.jvp(f, &directions).unwrap();
        let vjp = builder.vjp(f, inputs[4], &[0, 1]).unwrap();
        let fx = builder.partial(f, 0).unwrap();
        let fy = builder.partial(f, 1).unwrap();
        let hx = builder.jvp(fx, &directions).unwrap();
        let hy = builder.jvp(fy, &directions).unwrap();
        let forward_pair = builder.push(CalculusNode::Mul(inputs[4], jv)).unwrap();
        let reverse_x = builder.push(CalculusNode::Mul(vjp[0], inputs[2])).unwrap();
        let reverse_y = builder.push(CalculusNode::Mul(vjp[1], inputs[3])).unwrap();
        let reverse_pair = builder
            .push(CalculusNode::Add(reverse_x, reverse_y))
            .unwrap();
        let root = [jv, vjp[0], vjp[1], hx, hy, forward_pair, reverse_pair][selection];
        let definition = builder.finish(root).unwrap();
        roots.push(dag.pure_operator(&definition, arguments.clone()).unwrap());
    }
    let dag = dag.finish(roots).unwrap();
    assert_eq!(
        ScalarOperatorIr::lower(&dag)
            .unwrap()
            .evaluate_typed(dag.roots(), &mut |_| None)
            .unwrap()
            .into_iter()
            .map(|value| value.real_scalar_value().unwrap().value())
            .collect::<Vec<_>>(),
        expected
    );
}
