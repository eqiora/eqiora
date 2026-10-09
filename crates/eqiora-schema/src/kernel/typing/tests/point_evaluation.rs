use super::*;
use crate::kernel::{BoundarySide, ExprDagBuilder, ExprId};
use eqiora_core::{DynQuantity, RawId, ValueLiteral};

#[derive(Clone, Copy)]
enum Probe {
    Exact,
    Duplicate,
    Missing,
    WrongUnit,
    ForeignField,
    SideOnProduct,
}

type PointInference = Result<TypedResidual<RawId>, Vec<TypedResidualError<RawId, ()>>>;

fn infer(probe: Probe) -> (ExprId, PointInference) {
    let domain = Id::<kinds::Domain>::new();
    let x = Id::<kinds::Domain>::new();
    let v = Id::<kinds::Domain>::new();
    let field = Id::<kinds::Field>::new();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let support = SpatialSupport::Coordinates {
        domain: domain.erase(),
        factors: vec![(x.erase(), length, 1), (v.erase(), speed, 1)],
    };
    let mut builder = ExprDagBuilder::new();
    let value = builder.symbol(SymbolRef::Field(field)).unwrap();
    let cx = builder.coordinate(domain, x, 0).unwrap();
    let cv = builder
        .coordinate(
            domain,
            if matches!(probe, Probe::Duplicate) {
                x
            } else {
                v
            },
            0,
        )
        .unwrap();
    let px = builder
        .constant(ValueLiteral::try_from(DynQuantity::new(0.25, length)).unwrap())
        .unwrap();
    let pv = builder
        .constant(
            ValueLiteral::try_from(DynQuantity::new(
                0.5,
                if matches!(probe, Probe::WrongUnit | Probe::Duplicate) {
                    length
                } else {
                    speed
                },
            ))
            .unwrap(),
        )
        .unwrap();
    let mut at = vec![(cx, px)];
    if !matches!(probe, Probe::Missing) {
        at.push((cv, pv));
    }
    let side = matches!(probe, Probe::SideOnProduct).then_some(BoundarySide::Lower);
    let root = builder.evaluate_at(value, at, side).unwrap();
    let expression = builder.finish([root]).unwrap();
    let typed = TypedResidual::infer(
        expression,
        None,
        |_| None,
        RootContract::ValueRoots,
        |symbol| {
            Ok(match symbol {
                SymbolRef::Field(_) => {
                    let mut support = support.clone();
                    if matches!(probe, Probe::ForeignField) {
                        let SpatialSupport::Coordinates { domain, .. } = &mut support else {
                            unreachable!()
                        };
                        *domain = Id::<kinds::Domain>::new().erase();
                    }
                    ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(support))
                }
                SymbolRef::Coordinate { factor, axis, .. } => {
                    ExpressionType::coordinate(&factor.erase(), axis, Some(&support)).unwrap()
                }
                _ => unreachable!(),
            })
        },
    );
    (root, typed)
}

#[test]
fn point_evaluation_removes_only_the_complete_exact_support() {
    let (root, typed) = infer(Probe::Exact);
    let typed = typed.unwrap();
    let value = typed.node_type(root).unwrap();
    assert!(value.support.is_none());
    assert_eq!(value.dimension(), DimExponents::DIMENSIONLESS);
    for probe in [
        Probe::Duplicate,
        Probe::Missing,
        Probe::WrongUnit,
        Probe::ForeignField,
        Probe::SideOnProduct,
    ] {
        let (root, typed) = infer(probe);
        assert!(typed.unwrap_err().iter().any(|error| matches!(error,
            TypedResidualError::Type { node_index, error: TypeViolation::PointEvaluationRequiresExactCoordinates }
            if *node_index == root.index()
        )));
    }
}
