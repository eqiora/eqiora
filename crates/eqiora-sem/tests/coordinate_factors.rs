//! Coordinate factors retain nominal identity, order and units without a mesh.
use eqiora_core::{
    DimExponents, DynQuantity, Id, OntologyId, ScalarDomain, ValueType, entity::kinds,
};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::{
    Model, ModelView,
    kernel::{
        ActivationDef, AxisBounds, DomainDef, ExprDagBuilder, FieldDef, FieldRole, KernelNode,
        ObservableMeasure, RelationDef, RepresentationDef, SymbolRef, typing::SpatialSupport,
    },
};
use eqiora_sem::KernelProgram;

#[test]
fn distribution_on_position_velocity_product_retains_exact_measure() {
    let position = Id::<kinds::Domain>::new();
    let velocity = Id::<kinds::Domain>::new();
    let phase = Id::<kinds::Domain>::new();
    let field = Id::<kinds::Field>::new();
    let representation = Id::<kinds::Representation>::new();
    let relation = Id::<kinds::Relation>::new();
    let activation = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let density = DimExponents::from_integers([0, -2, 1, 0, 0, 0, 0]).unwrap();
    let interval = |id, unit| {
        DomainDef::coordinate_interval(
            id,
            AxisBounds::new(DynQuantity::new(-1.0, unit), DynQuantity::new(1.0, unit)).unwrap(),
        )
    };
    let mut dag = ExprDagBuilder::new();
    let value = dag.symbol(SymbolRef::Field(field)).unwrap();
    let zero = dag.constant(DynQuantity::new(0.0, density)).unwrap();
    let nodes = vec![
        KernelNode::from(interval(position, length)),
        KernelNode::from(interval(velocity, speed)),
        DomainDef::coordinate_product(phase, vec![position, velocity])
            .unwrap()
            .into(),
        FieldDef::new(
            field,
            ValueType::scalar(ScalarDomain::Real, density).unwrap(),
            FieldRole::Variable,
        )
        .into(),
        RepresentationDef::continuum(representation).into(),
        RelationDef::new(relation, dag.finish([value, zero]).unwrap())
            .unwrap()
            .into(),
        ActivationDef::continuous(activation).into(),
    ];
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("position velocity factor typing");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in [
        (phase.erase(), position.erase(), EdgeKind::DependsOn),
        (phase.erase(), velocity.erase(), EdgeKind::DependsOn),
        (field.erase(), phase.erase(), EdgeKind::DefinedOn),
        (field.erase(), representation.erase(), EdgeKind::DefinedOn),
        (relation.erase(), phase.erase(), EdgeKind::AppliesOn),
        (relation.erase(), field.erase(), EdgeKind::DependsOn),
        (activation.erase(), relation.erase(), EdgeKind::Activates),
    ] {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, []).unwrap().into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let typed = program.typed_relation_residual(relation).unwrap();
    let root = typed.node_type(value).unwrap();
    let support = SpatialSupport::Coordinates {
        domain: phase.erase(),
        factors: vec![(position.erase(), length, 1), (velocity.erase(), speed, 1)],
    };
    assert_eq!(root.support.as_ref(), Some(&support));
    assert_eq!(support.ambient_dimensions(), None);
    assert_eq!(support.intrinsic_dimensions(), 2);
    assert!(
        eqiora_schema::kernel::typing::ExpressionType::coordinate(
            &phase.erase(),
            0,
            Some(&support)
        )
        .is_err()
    );
    assert!(eqiora_schema::kernel::typing::gradient(root).is_err());
    // (s/m²) × m × (m/s) = 1, independently of a numerical quadrature.
    assert_eq!(
        ObservableMeasure::Volume
            .output_type(root, &support, &support, None)
            .unwrap()
            .value_type,
        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap()
    );
    assert!(
        ObservableMeasure::Boundary
            .output_type(root, &support, &support, None)
            .is_err()
    );
    let foreign = SpatialSupport::Coordinates {
        domain: Id::<kinds::Domain>::new().erase(),
        factors: vec![(position.erase(), length, 1), (velocity.erase(), speed, 1)],
    };
    assert!(
        ObservableMeasure::Volume
            .output_type(root, &foreign, &foreign, None)
            .is_err()
    );
    let reversed = SpatialSupport::Coordinates {
        domain: phase.erase(),
        factors: vec![(velocity.erase(), speed, 1), (position.erase(), length, 1)],
    };
    assert!(
        ObservableMeasure::Volume
            .output_type(root, &reversed, &reversed, None)
            .is_err()
    );
}

#[test]
fn coordinate_product_requires_exact_unique_factor_closure() {
    for fault in [
        "none",
        "missing member",
        "missing dependency",
        "repeated factor",
        "cycle",
        "physical factor",
    ] {
        let factor = Id::<kinds::Domain>::new();
        let product = Id::<kinds::Domain>::new();
        let model = OntologyId::<Model>::new();
        let relation = Id::<kinds::Relation>::new();
        let activation = Id::<kinds::Activation>::new();
        let unit = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
        let bounds =
            AxisBounds::new(DynQuantity::new(0.0, unit), DynQuantity::new(1.0, unit)).unwrap();
        let factor_definition = if fault == "physical factor" {
            let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
            DomainDef::cartesian_box(
                factor,
                vec![
                    AxisBounds::new(DynQuantity::new(0.0, length), DynQuantity::new(1.0, length))
                        .unwrap(),
                ],
            )
            .unwrap()
        } else {
            DomainDef::coordinate_interval(factor, bounds)
        };
        let factors = match fault {
            "repeated factor" => vec![factor, factor],
            "cycle" => vec![product],
            _ => vec![factor],
        };
        let mut dag = ExprDagBuilder::new();
        let zero = dag
            .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
            .unwrap();
        let nodes = vec![
            KernelNode::from(factor_definition),
            DomainDef::coordinate_product(product, factors.clone())
                .unwrap()
                .into(),
            RelationDef::new(relation, dag.finish([zero, zero]).unwrap())
                .unwrap()
                .into(),
            ActivationDef::continuous(activation).into(),
        ];
        let members = nodes
            .iter()
            .map(KernelNode::id)
            .filter(|id| fault != "missing member" || *id != factor.erase())
            .collect::<Vec<_>>();
        let mut transaction = Transaction::new("coordinate factor closure");
        for node in nodes {
            transaction.push(Op::DefineKernelNode { node });
        }
        if fault != "missing dependency" {
            for target in factors
                .into_iter()
                .map(|id| id.erase())
                .collect::<std::collections::BTreeSet<_>>()
            {
                transaction.push(Op::Connect {
                    from: product.erase(),
                    to: target,
                    edge: EdgeKind::DependsOn,
                });
            }
        }
        transaction.push(Op::Connect {
            from: activation.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        });
        transaction.push(Op::DefineOntologyView {
            view: ModelView::new(model, members, []).unwrap().into(),
        });
        let mut store = InMemoryGraphStore::new();
        let result = store
            .commit(transaction)
            .map(|_| ())
            .and_then(|()| KernelProgram::from_snapshot(&store.snapshot(), model).map(|_| ()));
        if matches!(fault, "none" | "physical factor") {
            result.unwrap();
        } else {
            let diagnostics = result.expect_err(fault);
            let expected = match fault {
                "missing dependency" => "DependsOn",
                "missing member" => "closure",
                "repeated factor" => "repeats an exact factor",
                "cycle" => "cycle",
                _ => unreachable!(),
            };
            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message().contains(expected)),
                "{fault}: {diagnostics:?}"
            );
        }
    }
}

#[test]
fn interval_units_do_not_weaken_physical_cartesian_bounds() {
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let bounds =
        AxisBounds::new(DynQuantity::new(-2.0, speed), DynQuantity::new(3.0, speed)).unwrap();
    assert!(DomainDef::cartesian_box(Id::new(), vec![bounds]).is_err());
    assert!(AxisBounds::new(DynQuantity::new(0.0, speed), DynQuantity::new(1.0, length)).is_err());
    for upper in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        assert!(
            AxisBounds::new(DynQuantity::new(0.0, speed), DynQuantity::new(upper, speed)).is_err()
        );
    }
}

#[test]
fn spherical_measure_requires_a_radial_interval_starting_at_the_center() {
    use eqiora_schema::kernel::{ObservableDef, ObservableReduction};
    for lower in [0.0, -1.0, 0.5] {
        let radial = Id::<kinds::Domain>::new();
        let total = Id::<kinds::Observable>::new();
        let model = OntologyId::<Model>::new();
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let concentration = DimExponents::from_integers([0, -3, 0, 0, 0, 0, 0]).unwrap();
        let mut dag = ExprDagBuilder::new();
        let density = dag.constant(DynQuantity::new(2.0, concentration)).unwrap();
        let relation = Id::<kinds::Relation>::new();
        let activation = Id::<kinds::Activation>::new();
        let mut anchor = ExprDagBuilder::new();
        let zero = anchor
            .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
            .unwrap();
        let nodes = vec![
            KernelNode::from(
                RelationDef::new(relation, anchor.finish([zero, zero]).unwrap()).unwrap(),
            ),
            ActivationDef::continuous(activation).into(),
            KernelNode::from(DomainDef::coordinate_interval(
                radial,
                AxisBounds::new(
                    DynQuantity::new(lower, length),
                    DynQuantity::new(2.0, length),
                )
                .unwrap(),
            )),
            ObservableDef::new(
                total,
                ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
                dag.finish([density]).unwrap(),
                ObservableReduction::SpatialIntegral {
                    limits: None,
                    input: radial,
                    domain: radial,
                    measure: ObservableMeasure::SphericalVolume,
                },
            )
            .unwrap()
            .into(),
        ];
        let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
        let mut transaction = Transaction::new("spherical measure admission");
        for node in nodes {
            transaction.push(Op::DefineKernelNode { node });
        }
        transaction.push(Op::Connect {
            from: total.erase(),
            to: radial.erase(),
            edge: EdgeKind::AppliesOn,
        });
        transaction.push(Op::Connect {
            from: activation.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        });
        transaction.push(Op::DefineOntologyView {
            view: ModelView::new(model, members, []).unwrap().into(),
        });
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let admitted = KernelProgram::from_snapshot(&store.snapshot(), model);
        if lower == 0.0 {
            let program = admitted.unwrap();
            program.typed_observable(total).unwrap();
        } else {
            let errors = admitted.unwrap_err();
            assert!(
                errors
                    .iter()
                    .any(|error| error.message().contains("from zero to a positive radius")),
                "{errors:?}"
            );
        }
    }
}
