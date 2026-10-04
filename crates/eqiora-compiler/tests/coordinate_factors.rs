//! Selected mathematical support bindings retain units without Geometry inputs.
use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_graph::Op;
use eqiora_schema::kernel::{AxisBounds, DomainKind, KernelNode};

#[test]
fn selected_interval_field_uses_dimensioned_bounds_without_cad() {
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let bounds =
        AxisBounds::new(DynQuantity::new(-2.0, speed), DynQuantity::new(2.0, speed)).unwrap();
    for owner in ["model", "public component"] {
        let source = format!(
            "{owner} Distribution(support velocity: interval(m/s)) {{ variable f: s/m on velocity; relation retain on velocity {{ f = 0[s/m]; }} }}"
        );
        let compiled = CompiledModel::compile_selected(
            "interval.eqi",
            &source,
            "Distribution",
            &[("velocity", StaticBindingValue::CoordinateInterval(bounds))],
        )
        .unwrap_or_else(|errors| panic!("{owner}: {errors:?}"));
        let (transaction, _, _) = compiled.into_parts();
        assert!(transaction.ops().iter().any(|op| matches!(op, Op::DefineKernelNode { node: KernelNode::Domain(domain) } if matches!(domain.kind(), DomainKind::CoordinateInterval { bounds: actual } if *actual == bounds))));
        let wrong = AxisBounds::new(
            DynQuantity::new(-2.0, DimExponents::DIMENSIONLESS),
            DynQuantity::new(2.0, DimExponents::DIMENSIONLESS),
        )
        .unwrap();
        let errors = CompiledModel::compile_selected(
            "interval.eqi",
            &source,
            "Distribution",
            &[("velocity", StaticBindingValue::CoordinateInterval(wrong))],
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("coordinate units")),
            "{errors:?}"
        );
    }
}

#[test]
fn selected_product_preserves_order_and_forward_nested_factor_ownership() {
    let interval = |time| {
        StaticBindingValue::CoordinateInterval(
            AxisBounds::new(
                DynQuantity::new(
                    -1.0,
                    DimExponents::from_integers([0, 1, time, 0, 0, 0, 0]).unwrap(),
                ),
                DynQuantity::new(
                    1.0,
                    DimExponents::from_integers([0, 1, time, 0, 0, 0, 0]).unwrap(),
                ),
            )
            .unwrap(),
        )
    };
    for owner in ["model", "public component"] {
        let source = format!(
            "{owner} Distribution(support position:interval(m), support velocity:interval(m/s), support radius:interval(m)) {{ support phase:product(xv,radius); support xv:product(position,velocity); variable f:s/m^3 on phase; relation retain on phase {{ f=0[s/m^3]; }} }}"
        );
        let compiled = CompiledModel::compile_selected(
            "product.eqi",
            &source,
            "Distribution",
            &[
                ("position", interval(0)),
                ("velocity", interval(-1)),
                ("radius", interval(0)),
            ],
        )
        .unwrap_or_else(|errors| panic!("{owner}: {errors:?}"));
        assert_eq!(compiled.transaction().ops().iter().filter(|op| matches!(op, Op::DefineKernelNode { node: KernelNode::Domain(domain) } if matches!(domain.kind(), DomainKind::CoordinateProduct { .. }))).count(), 2);
    }
}

#[test]
fn product_source_rejects_repetition_cycles_and_an_ambient_frame() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let bounds =
        AxisBounds::new(DynQuantity::new(0.0, length), DynQuantity::new(1.0, length)).unwrap();
    for (body, expected) in [
        (
            "support phase:product(position,position); variable f:1 on phase; relation r on phase { f=0; }",
            "repeats an exact factor",
        ),
        (
            "support phase:product(other); support other:product(phase); variable f:1 on phase; relation r on phase { f=0; }",
            "cyclic dependency",
        ),
        (
            "support phase:product(absent); variable f:1 on phase; relation r on phase { f=0; }",
            "unknown factor",
        ),
        (
            "variable f:vector<1,1> on position; relation r on position { f=f; }",
            "ambient spatial dimension",
        ),
        (
            "variable f:1 on position; relation r on position { coordinate(0)=0[m]; }",
            "coordinate",
        ),
        (
            "variable f:1 on position; relation r on position { grad(f)=grad(f); }",
            "gradient",
        ),
    ] {
        let source = format!("model M(support position:interval(m)) {{ {body} }}");
        let errors = CompiledModel::compile_selected(
            "invalid-factor.eqi",
            &source,
            "M",
            &[("position", StaticBindingValue::CoordinateInterval(bounds))],
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains(expected)),
            "{body}: {errors:?}"
        );
    }
}

#[test]
fn abstract_factors_accept_scalar_operators_and_do_not_ambiguate_physical_frames() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let bounds =
        AxisBounds::new(DynQuantity::new(0.0, length), DynQuantity::new(1.0, length)).unwrap();
    for body in [
        "variable f:1 on position; relation r on position { identity(x=f)=f; }",
        "domain body=box(0,1); parameter p:vector<1,1>=tensor_value(frame=body,components=[1]); parameter a:vector<1,1>=p; variable u:vector<1,1> on body; relation r on body {u=a;}",
    ] {
        let source = format!(
            "operator identity(input x:scalar):scalar=x; model M(support position:interval(m)) {{ {body} }}"
        );
        CompiledModel::compile_selected(
            "factor-context.eqi",
            &source,
            "M",
            &[("position", StaticBindingValue::CoordinateInterval(bounds))],
        )
        .unwrap_or_else(|errors| panic!("{body}: {errors:?}"));
    }
}
