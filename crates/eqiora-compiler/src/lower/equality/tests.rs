use super::*;
use crate::lower::{
    FreshLoweringIdentities, LoweringExpression, LoweringItem, LoweringModel, lower_typed_model,
};
use eqiora_core::{DimExponents, ScalarDomain, ValueLiteral, ValueType};
use eqiora_graph::Op;
use eqiora_schema::kernel::typing;
use eqiora_schema::kernel::{ExprNode, KernelNode};

#[test]
fn contextual_zero_adopts_complete_type_but_explicit_zero_never_does() {
    let complex = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
        .expect("admitted numeric scalar type");
    let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
        .expect("admitted numeric scalar type");
    let array = complex.clone().array(2).unwrap();
    let support = Some(typing::SpatialSupport::Volume {
        domain: "body",
        dimensions: 2,
    });
    for value_type in [complex.clone(), array.clone()] {
        let left = ExpressionType::new(value_type.clone(), support.clone());
        let checked = check(
            left.clone(),
            ExpressionType::new(real.clone(), None),
            false,
            true,
        )
        .unwrap();
        assert_eq!(checked.left, left);
        assert_eq!(checked.right.value_type, value_type);
        assert_eq!(checked.right.support, None);
        assert_eq!(checked.equation_type, left);
    }
    assert!(
        check(
            ExpressionType::<()>::new(real.clone(), None),
            ExpressionType::new(array, None),
            false,
            false
        )
        .is_err()
    );
    let promoted = check(
        ExpressionType::<()>::new(real, None),
        ExpressionType::new(complex.clone(), None),
        false,
        false,
    )
    .unwrap();
    assert_eq!(promoted.equation_type.value_type, complex);
    assert_ne!(promoted.equation_type, promoted.left);
}

#[test]
fn explicit_complex_rhs_zero_keeps_its_type_in_the_equation_sides() {
    let range = eqiora_lang::TextRange::new(0, 1);
    let right = LoweringExpression::literal(
        ValueLiteral::from_real(
            ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                .expect("admitted numeric scalar type"),
            -0.0,
        )
        .unwrap(),
        range,
    );
    let value_type = eqiora_lang::ValueTypeSyntax::from_checked(
        &ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("admitted numeric scalar type"),
        |_| None,
    )
    .unwrap();
    let model = LoweringModel {
        structural_dependencies: Default::default(),
        name: "M".into(),
        range,
        items: vec![
            LoweringItem::Field {
                name: "x".into(),
                domain: None,
                representation: None,
                value_type,
                role: eqiora_lang::FieldRoleSyntax::Variable,
                activation: eqiora_lang::ActivationSyntax::Continuous,
                range,
            },
            LoweringItem::Relation {
                name: "r".into(),
                domain: None,
                activation: eqiora_lang::ActivationSyntax::Continuous,
                initial: false,
                range,
                body: vec![crate::lower::LoweringEquation {
                    kind: eqiora_schema::kernel::RelationConditionKind::Equality,
                    left: LoweringExpression::name("x".into(), range),
                    right,
                    contextual_left_zero: false,
                    contextual_right_zero: false,
                    range,
                }]
                .into(),
            },
        ],
    };
    let compiled = lower_typed_model("typed.eqi", &model, &mut FreshLoweringIdentities).unwrap();
    let relation = compiled
        .transaction()
        .ops()
        .iter()
        .find_map(|operation| match operation {
            Op::DefineKernelNode {
                node: KernelNode::Relation(relation),
            } if !relation.is_initial() => Some(relation),
            _ => None,
        })
        .unwrap();
    let (_, right) = relation.equation_sides().next().unwrap();
    let ExprNode::Constant(value) = &relation.expression().nodes()[right.index() as usize] else {
        panic!("typed zero")
    };
    assert_eq!(value.value_type().scalar_domain(), ScalarDomain::Complex);
    assert_eq!(value.component(0).unwrap().0.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn substituted_named_zero_retains_an_explicit_right_side() {
    let source = "component C(parameter zero: 1 = 0) {  variable x: 1; initial { x = 1; } relation r { x = zero; } } model M() { instance c: C(); }";
    let compiled = crate::compile("named.eqi", source).unwrap();
    let dag = compiled[0]
        .transaction()
        .ops()
        .iter()
        .find_map(|operation| match operation {
            Op::DefineKernelNode {
                node: KernelNode::Relation(relation),
            } if !relation.is_initial() => Some(relation.expression()),
            _ => None,
        })
        .unwrap();
    assert_eq!(dag.roots().len(), 2);
    assert!(
        matches!(&dag.nodes()[dag.roots()[1].index() as usize], ExprNode::Constant(value) if value.real_scalar_value().unwrap().value() == 0.0)
    );
}

#[test]
fn full_type_support_and_activation_fail_in_the_source_owner() {
    for (source, message) in [
        (
            "model M() { variable x: m; initial { x = 0; } relation r { x = 0[s]; } }",
            "incompatible types",
        ),
        (
            "model M() { variable x: array<1,2>; variable y: array<1,3>; relation r { x = y; } }",
            "incompatible types",
        ),
        (
            "model M() { domain d = box(0,1,0,1); variable x: vector<1,2> on d; variable y: array<1,2> on d; relation r on d { x = y; } }",
            "incompatible types",
        ),
        (
            "model M() { domain a = box(0,1); domain b = box(0,1); variable x: 1 on a; variable y: 1 on b; relation r on a { x = y; } }",
            "incompatible supports",
        ),
        (
            "model M() { domain a = box(0,1); domain b = box(0,1); variable x: 1 on a; relation r on b { x = 0; } }",
            "Relation scope",
        ),
        (
            "model M() { state x: 1; initial { x = 0; } relation r { next(x) = pre(x); } }",
            "state evolution requires its exact clock",
        ),
        (
            "model M() { clock tick = periodic(1[s] / 1, phase = 0[s] / 1); state x: 1; initial { x = 0; } relation r at tick { derivative(x) = 0; } }",
            "clocked Relation cannot use",
        ),
        (
            "component C() { state x: 1; initial { x = 0; } clock tick = periodic(1[s] / 1, phase = 0[s] / 1); relation r at tick { derivative(x) = 0; } } model M() {}",
            "clocked Relation cannot use",
        ),
    ] {
        let diagnostics = crate::compile("denied.eqi", source).unwrap_err();
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.source_span().is_some()
                    && diagnostic.message().contains(message)),
            "{source}: {diagnostics:?}"
        );
    }
    for source in [
        "model M() { variable x: complex<1>; initial { x = 0; } relation r { x = -(-0); } }",
        "model M() { domain d = box(0,1,0,1); variable x: vector<m,2> on d; relation r on d { x = 0; } }",
        "model M() { clock tick = periodic(1[s] / 1, phase = 0[s] / 1); state x: 1 at tick; initial { pre(x) = 0; } relation r at tick { next(x) = pre(x); } }",
    ] {
        crate::compile("positive.eqi", source).unwrap();
    }
}

#[test]
fn unit_normalization_retains_representable_subnormal_and_rejects_nonzero_underflow() {
    assert_eq!(
        crate::units::normalize_value(f64::from_bits(1), 1.0)
            .unwrap()
            .to_bits(),
        1
    );
    assert!(crate::units::normalize_value(f64::from_bits(1), 0.001).is_err());
    assert_eq!(
        crate::units::normalize_value(-0.0, 0.001)
            .unwrap()
            .to_bits(),
        0
    );
}

#[test]
fn checked_residuals_preserve_negative_base_and_signed_power_meaning() {
    for (expression, expected) in [
        ("-x^2", -4.0),
        ("(-x)^2", 4.0),
        ("x^-2", 0.25),
        ("-x^-2", -0.25),
    ] {
        let compiled = crate::compile(
            "precedence.eqi",
            &format!("model M() {{ variable x: 1; initial {{ x = 2; }} relation r {{ {expression} = 0; }} }}"),
        )
        .unwrap();
        let dag = compiled[0]
            .transaction()
            .ops()
            .iter()
            .find_map(|operation| match operation {
                Op::DefineKernelNode {
                    node: KernelNode::Relation(relation),
                } if !relation.is_initial() => Some(relation.expression()),
                _ => None,
            })
            .unwrap();
        let mut values: Vec<f64> = Vec::new();
        for node in dag.nodes() {
            values.push(match node {
                ExprNode::Symbol(_) => 2.0,
                ExprNode::Constant(value) => value.real_scalar_value().unwrap().value(),
                ExprNode::Neg(value) => -values[value.index() as usize],
                ExprNode::PowI(base, exponent) => values[base.index() as usize].powi(*exponent),
                other => panic!("outside fixed power corpus: {other:?}"),
            });
        }
        assert_eq!(
            values[dag.roots()[0].index() as usize],
            expected,
            "{expression}"
        );
    }
}
