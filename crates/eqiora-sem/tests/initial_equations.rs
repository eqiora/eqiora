use eqiora_core::entity::kinds;
use eqiora_core::{
    Diagnostic, DimExponents, DynQuantity, Id, OntologyId, RawId, ScalarDomain, ValueType,
};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{
    ActivationDef, ClockDomainDef, ExprDag, ExprDagBuilder, ExprNode, FieldDef, FieldRole,
    KernelNode, RationalTime, RelationDef, SymbolRef,
};
use eqiora_schema::{Model, ModelView};
use eqiora_sem::{Interpreter, KernelProgram, ReferenceConfig};

fn scalar(field: Id<kinds::Field>, role: FieldRole) -> KernelNode {
    FieldDef::new(
        field,
        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("valid scalar type"),
        role,
    )
    .into()
}

fn equation(symbol: SymbolRef, value: f64, dimension: DimExponents) -> ExprDag {
    let mut dag = ExprDagBuilder::new();
    let symbol = dag.symbol(symbol).unwrap();
    let value = dag.constant(DynQuantity::new(value, dimension)).unwrap();

    dag.finish([symbol, value]).unwrap()
}

fn program(
    mut nodes: Vec<KernelNode>,
    mut edges: Vec<(RawId, RawId, EdgeKind)>,
) -> Result<KernelProgram, Vec<Diagnostic>> {
    let mut activations = Vec::new();
    for node in &nodes {
        if let KernelNode::Relation(relation) = node {
            for expression in relation.expression().nodes() {
                let ExprNode::Symbol(symbol) = expression else {
                    continue;
                };
                let field = match *symbol {
                    SymbolRef::Field(field)
                    | SymbolRef::Derivative(field, std::num::NonZeroU32::MIN)
                    | SymbolRef::Pre(field)
                    | SymbolRef::Next(field) => field,
                    _ => continue,
                };
                let edge = (relation.id().erase(), field.erase(), EdgeKind::DependsOn);
                if !edges.contains(&edge) {
                    edges.push(edge);
                }
            }
            if !relation.is_initial() {
                let activation = ActivationDef::continuous(Id::new());
                edges.push((
                    activation.id().erase(),
                    relation.id().erase(),
                    EdgeKind::Activates,
                ));
                activations.push(activation.into());
            }
        }
    }
    nodes.extend(activations);
    let model = OntologyId::<Model>::new();
    let view = ModelView::new(model, nodes.iter().map(KernelNode::id), []).unwrap();
    let mut transaction = Transaction::new("initial equation test");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in edges {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView { view: view.into() });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model)
}

fn decay(field: Id<kinds::Field>) -> KernelNode {
    let mut dag = ExprDagBuilder::new();
    let derivative = dag
        .symbol(SymbolRef::Derivative(field, std::num::NonZeroU32::MIN))
        .unwrap();
    let value = dag.symbol(SymbolRef::Field(field)).unwrap();
    let rate = dag
        .constant(DynQuantity::new(
            1.0,
            DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap(),
        ))
        .unwrap();
    let rate_value = dag.mul(rate, value).unwrap();
    let root = dag.add(derivative, rate_value).unwrap();
    RelationDef::new(
        Id::new(),
        {
            let equation_zero = dag
                .constant(eqiora_core::DynQuantity::new(
                    0.0,
                    eqiora_core::DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap(),
                ))
                .unwrap();
            dag.finish([root, equation_zero])
        }
        .unwrap(),
    )
    .unwrap()
    .into()
}

#[test]
fn initial_algebraic_condition_and_regular_equations_jointly_determine_state() {
    let x = Id::new();
    let y = Id::new();
    let mut dag = ExprDagBuilder::new();
    let xv = dag.symbol(SymbolRef::Field(x)).unwrap();
    let yv = dag.symbol(SymbolRef::Field(y)).unwrap();
    let twice_x = dag.add(xv, xv).unwrap();

    let model = program(
        vec![
            scalar(x, FieldRole::State),
            scalar(y, FieldRole::Variable),
            decay(x),
            RelationDef::new(Id::new(), dag.finish([yv, twice_x]).unwrap())
                .unwrap()
                .into(),
            RelationDef::initial(
                Id::new(),
                equation(SymbolRef::Field(y), 2.0, DimExponents::DIMENSIONLESS),
            )
            .unwrap()
            .into(),
        ],
        vec![],
    )
    .unwrap();
    let config = ReferenceConfig::new(0.1, 0.1).unwrap();
    let initial = Interpreter::new().initialize(&model, 0.0, config).unwrap();
    assert!(
        (initial.fields()[&x.erase()]
            .real_scalar_value()
            .unwrap()
            .value()
            - 1.0)
            .abs()
            < 1e-8
    );
    assert!(
        (initial.fields()[&y.erase()]
            .real_scalar_value()
            .unwrap()
            .value()
            - 2.0)
            .abs()
            < 1e-8
    );
    assert!(
        (initial.derivatives()[&(x.erase(), std::num::NonZeroU32::MIN)]
            .real_scalar_value()
            .unwrap()
            .value()
            + 1.0)
            .abs()
            < 1e-8
    );
    for guess in [-4.0, 3.0] {
        let another = Interpreter::new()
            .initialize(&model, 0.0, config.with_initial_guess(guess).unwrap())
            .unwrap();
        assert!(
            (another.fields()[&x.erase()]
                .real_scalar_value()
                .unwrap()
                .value()
                - 1.0)
                .abs()
                < 1e-8
        );
        assert!(
            (another.fields()[&y.erase()]
                .real_scalar_value()
                .unwrap()
                .value()
                - 2.0)
                .abs()
                < 1e-8
        );
        assert!(
            (another.derivatives()[&(x.erase(), std::num::NonZeroU32::MIN)]
                .real_scalar_value()
                .unwrap()
                .value()
                + 1.0)
                .abs()
                < 1e-8
        );
    }
    assert_eq!(config.initial_guess(), 0.0);
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(config.with_initial_guess(invalid).is_err());
    }
    let run = Interpreter::new().run(&model, config).unwrap();
    // Backward Euler: x1 = x0 / (1 + h); the initial equation y=2 is not reapplied.
    assert!((run.last_value(x.erase()).unwrap().value() - 1.0 / 1.1).abs() < 1e-8);
}

#[test]
fn initial_derivative_condition_can_determine_stationary_state() {
    let x = Id::new();
    let model = program(
        vec![
            scalar(x, FieldRole::State),
            decay(x),
            RelationDef::initial(
                Id::new(),
                equation(
                    SymbolRef::Derivative(x, std::num::NonZeroU32::MIN),
                    0.0,
                    DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap(),
                ),
            )
            .unwrap()
            .into(),
        ],
        vec![],
    )
    .unwrap();
    let initial = Interpreter::new()
        .initialize(&model, 0.0, ReferenceConfig::new(0.0, 0.1).unwrap())
        .unwrap();
    assert_eq!(
        initial.fields()[&x.erase()]
            .real_scalar_value()
            .unwrap()
            .value(),
        0.0
    );
    assert_eq!(
        initial.derivatives()[&(x.erase(), std::num::NonZeroU32::MIN)]
            .real_scalar_value()
            .unwrap()
            .value(),
        0.0
    );
}

#[test]
fn missing_contradictory_and_rank_deficient_zero_residual_initialization_fail() {
    for mode in 0..3 {
        let x = Id::new();
        let mut nodes = vec![scalar(x, FieldRole::State), decay(x)];
        if mode == 1 {
            for value in [1.0, 2.0] {
                nodes.push(
                    RelationDef::initial(
                        Id::new(),
                        equation(SymbolRef::Field(x), value, DimExponents::DIMENSIONLESS),
                    )
                    .unwrap()
                    .into(),
                );
            }
        } else if mode == 2 {
            let mut dag = ExprDagBuilder::new();
            let x = dag.symbol(SymbolRef::Field(x)).unwrap();

            nodes.push(
                RelationDef::initial(Id::new(), dag.finish([x, x]).unwrap())
                    .unwrap()
                    .into(),
            );
        }
        let model = program(nodes, vec![]).unwrap();
        let diagnostics = Interpreter::new()
            .initialize(&model, 0.0, ReferenceConfig::new(0.0, 0.1).unwrap())
            .unwrap_err();
        assert!(
            diagnostics
                .iter()
                .any(|error| error.message().contains(if mode == 2 {
                    "singular"
                } else {
                    "square"
                })),
            "{diagnostics:?}"
        );
    }
}

#[test]
fn initial_pre_is_a_clocked_unknown_and_next_is_not_an_initial_condition() {
    let field = Id::new();
    let clock = ClockDomainDef::periodic(
        Id::new(),
        RationalTime::new(1, 10).unwrap(),
        RationalTime::ZERO,
    )
    .unwrap();
    let edge = (field.erase(), clock.id().erase(), EdgeKind::ClockedBy);
    let nodes = vec![
        scalar(field, FieldRole::State),
        clock.into(),
        RelationDef::initial(
            Id::new(),
            equation(SymbolRef::Pre(field), 3.0, DimExponents::DIMENSIONLESS),
        )
        .unwrap()
        .into(),
    ];
    let model = program(nodes, vec![edge]).unwrap();
    let initial = Interpreter::new()
        .initialize(&model, 0.0, ReferenceConfig::new(0.0, 0.1).unwrap())
        .unwrap();
    assert_eq!(
        initial.fields()[&field.erase()]
            .real_scalar_value()
            .unwrap()
            .value(),
        3.0
    );
    let invalid = program(
        vec![
            scalar(field, FieldRole::State),
            RelationDef::initial(
                Id::new(),
                equation(SymbolRef::Next(field), 0.0, DimExponents::DIMENSIONLESS),
            )
            .unwrap()
            .into(),
        ],
        vec![],
    )
    .unwrap_err();
    assert!(invalid.iter().any(|error| {
        error
            .message()
            .contains("initial Relation cannot read Next")
    }));
}

#[test]
fn unused_fields_of_either_role_with_two_clocks_fail_admission() {
    for role in [FieldRole::Variable, FieldRole::State] {
        let field = Id::new();
        let mut nodes = vec![
            scalar(field, role),
            RelationDef::initial(
                Id::new(),
                equation(SymbolRef::Field(field), 0.0, DimExponents::DIMENSIONLESS),
            )
            .unwrap()
            .into(),
        ];
        // A separate unused Field ensures validation is not triggered only by symbol use.
        let unused = Id::new();
        nodes.push(scalar(unused, role));
        let mut edges = Vec::new();
        for _ in 0..2 {
            let clock = ClockDomainDef::periodic(
                Id::new(),
                RationalTime::new(1, 10).unwrap(),
                RationalTime::ZERO,
            )
            .unwrap();
            edges.push((unused.erase(), clock.id().erase(), EdgeKind::ClockedBy));
            nodes.push(clock.into());
        }
        let errors = program(nodes, edges).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("at most one exact ClockDomain"))
        );
    }
}

#[test]
fn algebraic_unknown_has_no_authored_continuous_derivative() {
    let x = Id::new();
    let errors = program(vec![scalar(x, FieldRole::Variable), decay(x)], vec![]).unwrap_err();
    assert!(errors.iter().any(|error| {
        error
            .message()
            .contains("Derivative requires continuous state")
    }));
}

#[test]
fn affine_rank_one_descriptor_uses_one_independent_initial_condition() {
    for rate in [0.0, 1.0, 2.0] {
        let x = Id::new();
        let y = Id::new();
        let mut nodes = vec![scalar(x, FieldRole::State), scalar(y, FieldRole::State)];
        for field in [x, y] {
            let mut dag = ExprDagBuilder::new();
            let dx = dag
                .symbol(SymbolRef::Derivative(x, std::num::NonZeroU32::MIN))
                .unwrap();
            let dy = dag
                .symbol(SymbolRef::Derivative(y, std::num::NonZeroU32::MIN))
                .unwrap();
            let sum = dag.add(dx, dy).unwrap();
            let value = dag.symbol(SymbolRef::Field(field)).unwrap();
            let coefficient = dag
                .constant(DynQuantity::new(
                    2.0 * rate,
                    DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap(),
                ))
                .unwrap();
            let value = dag.mul(coefficient, value).unwrap();
            let root = dag.add(sum, value).unwrap();
            nodes.push(
                RelationDef::new(
                    Id::new(),
                    {
                        let equation_zero = dag
                            .constant(eqiora_core::DynQuantity::new(
                                0.0,
                                eqiora_core::DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0])
                                    .unwrap(),
                            ))
                            .unwrap();
                        dag.finish([root, equation_zero])
                    }
                    .unwrap(),
                )
                .unwrap()
                .into(),
            );
        }
        let missing = program(nodes.clone(), vec![]).unwrap();
        assert!(
            Interpreter::new()
                .initialize(&missing, 0.0, ReferenceConfig::new(0.0, 0.1).unwrap())
                .is_err()
        );
        nodes.push(
            RelationDef::initial(
                Id::new(),
                equation(SymbolRef::Field(x), 1.0, DimExponents::DIMENSIONLESS),
            )
            .unwrap()
            .into(),
        );
        let mut redundant = nodes.clone();
        redundant.push(
            RelationDef::initial(
                Id::new(),
                equation(SymbolRef::Field(y), 1.0, DimExponents::DIMENSIONLESS),
            )
            .unwrap()
            .into(),
        );
        assert!(
            Interpreter::new()
                .initialize(
                    &program(redundant, vec![]).unwrap(),
                    0.0,
                    ReferenceConfig::new(0.0, 0.1).unwrap()
                )
                .is_err()
        );
        let model = program(nodes, vec![]).unwrap();
        let result =
            Interpreter::new().initialize(&model, 0.0, ReferenceConfig::new(0.0, 0.1).unwrap());
        if rate == 0.0 {
            let errors = result.unwrap_err();
            assert!(
                errors
                    .iter()
                    .any(|error| error.message().contains("singular"))
            );
            continue;
        }
        let initial = result.unwrap();
        for field in [x, y] {
            assert!(
                (initial.fields()[&field.erase()]
                    .real_scalar_value()
                    .unwrap()
                    .value()
                    - 1.0)
                    .abs()
                    < 1e-8
            );
            assert!(
                (initial.derivatives()[&(field.erase(), std::num::NonZeroU32::MIN)]
                    .real_scalar_value()
                    .unwrap()
                    .value()
                    + rate)
                    .abs()
                    < 1e-8
            );
        }
    }
}
