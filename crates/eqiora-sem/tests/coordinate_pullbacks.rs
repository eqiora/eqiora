//! Independent polynomial values through ordinary Model admission and point binding.
use eqiora_core::{
    DimExponents, DynQuantity, Id, OntologyId, ScalarDomain, ValueType, entity::kinds,
};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::*;
use eqiora_schema::{Model, ModelView};
use eqiora_sem::KernelProgram;

fn model(target_upper: f64) -> (KernelProgram, Id<kinds::Observable>) {
    model_factor(target_upper, None, false, false)
}

fn model_factor(
    target_upper: f64,
    factor: Option<CoordinateMapFactor>,
    swap: bool,
    singular: bool,
) -> (KernelProgram, Id<kinds::Observable>) {
    let model = OntologyId::<Model>::new();
    let source = Id::<kinds::Domain>::new();
    let target = Id::<kinds::Domain>::new();
    let observable = Id::<kinds::Observable>::new();
    let relation = Id::<kinds::Relation>::new();
    let activation = Id::<kinds::Activation>::new();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let area = length.mul(length).unwrap();
    let bounds = |upper| {
        AxisBounds::new(
            DynQuantity::new(0.0, length),
            DynQuantity::new(upper, length),
        )
        .unwrap()
    };
    let mut dag = ExprDagBuilder::new();
    let mut coordinate = |support, axis| {
        dag.symbol(SymbolRef::Coordinate {
            support,
            factor: support,
            axis,
        })
        .unwrap()
    };
    let xi = coordinate(source, 0);
    let eta = coordinate(source, 1);
    let x = coordinate(target, 0);
    let y = coordinate(target, 1);
    let two = dag
        .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let three = dag
        .constant(DynQuantity::new(3.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let twice_xi = dag.mul(two, xi).unwrap();
    let mapped_x = dag.add(twice_xi, eta).unwrap();
    let mapped_y = if singular {
        mapped_x
    } else {
        dag.mul(three, eta).unwrap()
    };
    let x2 = dag.mul(x, x).unwrap();
    let xy = dag.mul(x, y).unwrap();
    let u = dag.add(x2, xy).unwrap();
    let at = if swap {
        vec![(x, mapped_y), (y, mapped_x)]
    } else {
        vec![(x, mapped_x), (y, mapped_y)]
    };
    let pulled = match factor {
        None => dag.pullback(u, vec![xi, eta], at),
        Some(factor) => dag.coordinate_map_factor(factor, vec![xi, eta], at),
    }
    .unwrap();
    let area = if factor.is_some() {
        DimExponents::DIMENSIONLESS
    } else {
        area
    };
    let mut samples = Vec::new();
    for (a, b) in [(0.25, 0.5), (0.75, 0.25)] {
        let a = dag.constant(DynQuantity::new(a, length)).unwrap();
        let b = dag.constant(DynQuantity::new(b, length)).unwrap();
        samples.push(
            dag.evaluate_at(pulled, vec![(xi, a), (eta, b)], None)
                .unwrap(),
        );
    }
    let sum = dag.add(samples[0], samples[1]).unwrap();
    let mut equation = ExprDagBuilder::new();
    let zero = equation
        .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let nodes: Vec<KernelNode> = vec![
        DomainDef::cartesian_box(source, vec![bounds(1.0); 2])
            .unwrap()
            .into(),
        DomainDef::cartesian_box(target, vec![bounds(target_upper); 2])
            .unwrap()
            .into(),
        ObservableDef::new(
            observable,
            ValueType::scalar(ScalarDomain::Real, area).unwrap(),
            dag.finish([sum]).unwrap(),
            ObservableReduction::Value,
        )
        .unwrap()
        .into(),
        RelationDef::new(relation, equation.finish([zero, zero]).unwrap())
            .unwrap()
            .into(),
        ActivationDef::continuous(activation).into(),
    ];
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("exact coordinate pullback samples");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    transaction.push(Op::Connect {
        from: activation.erase(),
        to: relation.erase(),
        edge: EdgeKind::Activates,
    });
    for domain in [source, target] {
        transaction.push(Op::Connect {
            from: observable.erase(),
            to: domain.erase(),
            edge: EdgeKind::DependsOn,
        });
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, []).unwrap().into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        observable,
    )
}

#[test]
fn nonsymmetric_affine_pullback_rebinds_each_sample_in_its_own_context() {
    let (program, observable) = model(3.0);
    let result = program
        .evaluate_finite_observable(observable, &mut |_| None)
        .unwrap();
    // u(2*xi+eta,3*eta)=4*xi²+10*xi*eta+4*eta².
    // At (1/4,1/2): 5/2; at (3/4,1/4): 35/8; sum=55/8.
    let expected_unit = DimExponents::from_integers([0, 2, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(
        result.real_scalar_value().unwrap(),
        DynQuantity::new(55.0 / 8.0, expected_unit)
    );
}

#[test]
fn mapped_points_cannot_escape_the_exact_target_support() {
    let (program, observable) = model(1.0);
    let error = program
        .evaluate_finite_observable(observable, &mut |_| None)
        .unwrap_err();
    assert!(
        error.message().contains("outside its exact support"),
        "{error:?}"
    );
}

#[test]
fn volume_scale_and_orientation_distinguish_reflection_and_singularity() {
    use CoordinateMapFactor::{Orientation, SignedJacobian, VolumeScale};
    for (factor, swap, singular, expected) in [
        (SignedJacobian, false, false, Some(12.0)),
        (SignedJacobian, true, false, Some(-12.0)),
        (VolumeScale, false, false, Some(12.0)),
        (VolumeScale, true, false, Some(12.0)),
        (Orientation, false, false, Some(2.0)),
        (Orientation, true, false, Some(-2.0)),
        (SignedJacobian, false, true, Some(0.0)),
        (VolumeScale, false, true, None),
        (Orientation, false, true, None),
    ] {
        let (program, observable) = model_factor(3.0, Some(factor), swap, singular);
        let result = program.evaluate_finite_observable(observable, &mut |_| None);
        if let Some(expected) = expected {
            // Sum over two samples of the constant Jacobian [[2,1],[0,3]].
            // Swapping target rows reverses orientation but preserves volume.
            let actual = result.unwrap().real_scalar_value().unwrap();
            assert_eq!(actual.dim(), DimExponents::DIMENSIONLESS);
            if factor == Orientation || expected == 0.0 {
                assert_eq!(actual.value(), expected);
            } else {
                // Well-conditioned 2x2 LU plus logarithm/exponential scaling:
                // allow 64 binary64 rounding units, far below an omitted factor.
                assert!((actual.value() - expected).abs() <= 64.0 * f64::EPSILON * expected.abs());
            }
        } else {
            assert!(result.unwrap_err().message().contains("singular"));
        }
    }
}
