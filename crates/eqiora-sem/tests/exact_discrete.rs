use eqiora_core::{Id, OntologyId, RawId, ValueLiteral, ValueType, entity::kinds};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::*;
use eqiora_schema::{Model, ModelView};
use eqiora_sem::{Interpreter, KernelProgram, ReferenceConfig};

fn fixture(
    initial: ValueLiteral,
    change: ValueLiteral,
    nominal: Option<KernelNode>,
) -> (KernelProgram, RawId, RawId, RawId, RawId) {
    let model = OntologyId::<Model>::new();
    let clock = Id::<kinds::ClockDomain>::new();
    let activation = Id::<kinds::Activation>::new();
    let field = Id::<kinds::Field>::new();
    let output = Id::<kinds::Port>::new();
    let input = Id::<kinds::Port>::new();
    let mut nodes = vec![
        ClockDomainDef::periodic(clock, RationalTime::new(1, 1).unwrap(), RationalTime::ZERO)
            .unwrap()
            .into(),
        ActivationDef::periodic(activation).into(),
        FieldDef::new(field, initial.value_type().clone(), FieldRole::State).into(),
        PortDef::signal(
            output,
            SignalDirection::Output,
            initial.value_type().clone(),
        )
        .into(),
    ];
    nodes.push(PortDef::signal(input, SignalDirection::Input, change.value_type().clone()).into());
    if let Some(node) = nominal {
        nodes.push(node);
    }
    let mut transaction = Transaction::new("exact discrete state");
    for from in [
        activation.erase(),
        field.erase(),
        output.erase(),
        input.erase(),
    ] {
        transaction.push(Op::Connect {
            from,
            to: clock.erase(),
            edge: EdgeKind::ClockedBy,
        });
    }
    for is_initial in [true, false] {
        let relation = Id::<kinds::Relation>::new();
        let mut dag = ExprDagBuilder::new();
        let mut dependencies = vec![field.erase()];
        let roots = if is_initial {
            let target = dag.symbol(SymbolRef::Pre(field)).unwrap();
            let value = dag.constant(initial.clone()).unwrap();
            vec![target, value]
        } else {
            let next = dag.symbol(SymbolRef::Next(field)).unwrap();
            let pre = dag.symbol(SymbolRef::Pre(field)).unwrap();
            let delta = dag.symbol(SymbolRef::Port(input)).unwrap();
            dependencies.push(input.erase());
            let rhs = if initial.as_bool().is_some() {
                let zero = dag
                    .constant(ValueLiteral::from_integer(change.value_type().clone(), 0).unwrap())
                    .unwrap();
                let max = dag
                    .constant(
                        ValueLiteral::from_integer(change.value_type().clone(), i64::MAX).unwrap(),
                    )
                    .unwrap();
                let is_zero = dag.compare(ComparisonOp::Equal, delta, zero).unwrap();
                let shifted = dag.add(max, delta).unwrap();
                let positive = dag.compare(ComparisonOp::Greater, shifted, zero).unwrap();
                dag.or(is_zero, positive).unwrap()
            } else {
                dag.add(pre, delta).unwrap()
            };
            let out = dag.symbol(SymbolRef::Port(output)).unwrap();
            dependencies.push(output.erase());
            vec![next, rhs, out, next]
        };
        let dag = dag.finish(roots).unwrap();
        nodes.push(
            if is_initial {
                RelationDef::initial(relation, dag).unwrap()
            } else {
                RelationDef::new(relation, dag).unwrap()
            }
            .into(),
        );
        for to in dependencies {
            transaction.push(Op::Connect {
                from: relation.erase(),
                to,
                edge: EdgeKind::DependsOn,
            });
        }
        if !is_initial {
            transaction.push(Op::Connect {
                from: activation.erase(),
                to: relation.erase(),
                edge: EdgeKind::Activates,
            });
        }
    }
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let edges = transaction;
    let mut transaction = Transaction::new("exact discrete state");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for op in edges.ops() {
        transaction.push(op.clone());
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, [output.erase(), input.erase()])
            .unwrap()
            .into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        field.erase(),
        output.erase(),
        input.erase(),
        clock.erase(),
    )
}

#[test]
fn exact_species_transfer_is_atomic_and_resumes_after_second_tick() {
    let space = FiniteSpaceDef::new(Id::new(), ["A".into(), "B".into()]).unwrap();
    let initial = ValueLiteral::integer(space.counts(), [2, 9_007_199_254_740_993]).unwrap();
    let change = ValueLiteral::integer(
        space
            .coordinates(
                eqiora_core::ScalarDomain::Integer,
                eqiora_core::DimExponents::DIMENSIONLESS,
            )
            .expect("integer coordinates"),
        [-1, 1],
    )
    .unwrap();
    let (program, field, output, input, clock) =
        fixture(initial.clone(), change.clone(), Some(space.into()));
    let interpreter = Interpreter::new();
    assert!(
        interpreter
            .run(&program, ReferenceConfig::new(2., 1.).unwrap())
            .is_err()
    );
    assert_eq!(
        interpreter
            .initialize(&program, ReferenceConfig::new(2., 1.).unwrap())
            .unwrap()
            .fields()[&field],
        initial
    );
    let mut session = interpreter
        .execution_session(
            &program,
            ReferenceConfig::new(2., 1.).unwrap(),
            [(input, clock, vec![change; 3])],
        )
        .unwrap();
    for (tick, expected) in [[1, 9_007_199_254_740_994], [0, 9_007_199_254_740_995]]
        .into_iter()
        .enumerate()
    {
        assert_eq!(session.advance_ticks(1).unwrap(), 1);
        let value = session.field(field).unwrap();
        assert_eq!(
            [
                value.integer_component(0).unwrap(),
                value.integer_component(1).unwrap()
            ],
            expected
        );
        assert_eq!(session.output(output, tick as u64).unwrap().1, &value);
    }
    let checkpoint = session.checkpoint();
    let mut resumed = interpreter.resume_execution(&program, &checkpoint).unwrap();
    assert!(resumed.advance_ticks(1).is_err());
    assert_eq!(resumed.field(field), checkpoint.field(field));
    assert_eq!(resumed.next_tick(), checkpoint.next_tick());
    assert_eq!(resumed.output(output, 1), checkpoint.output(output, 1));
    assert!(resumed.output(output, 2).is_none());
}

#[test]
fn integer_overflow_keeps_accepted_state_and_output_absence() {
    let ty = ValueType::scalar(
        eqiora_core::ScalarDomain::Integer,
        eqiora_core::DimExponents::DIMENSIONLESS,
    )
    .expect("valid scalar type");
    let initial = ValueLiteral::from_integer(ty.clone(), i64::MAX).unwrap();
    let (program, field, output, input, clock) = fixture(
        initial.clone(),
        ValueLiteral::from_integer(ty.clone(), 1).unwrap(),
        None,
    );
    let mut session = Interpreter::new()
        .execution_session(
            &program,
            ReferenceConfig::new(1., 1.).unwrap(),
            [(
                input,
                clock,
                vec![ValueLiteral::from_integer(ty, 1).unwrap(); 2],
            )],
        )
        .unwrap();
    assert!(session.advance_ticks(1).is_err());
    assert_eq!(session.field(field), Some(initial));
    assert_eq!(session.next_tick(), Some(RationalTime::ZERO));
    assert!(session.output(output, 0).is_none());
}

#[test]
fn boolean_assignment_short_circuits_and_failed_tick_preserves_snapshot() {
    let ty = ValueType::scalar(
        eqiora_core::ScalarDomain::Integer,
        eqiora_core::DimExponents::DIMENSIONLESS,
    )
    .expect("valid scalar type");
    let value = |n| ValueLiteral::from_integer(ty.clone(), n).unwrap();
    let (program, field, output, input, clock) =
        fixture(ValueLiteral::boolean(false), value(0), None);
    let interpreter = Interpreter::new();
    let config = ReferenceConfig::new(2., 1.).unwrap();
    assert_eq!(
        interpreter.initialize(&program, config).unwrap().fields()[&field].as_bool(),
        Some(false)
    );
    assert!(
        program
            .numerical_residuals(
                program
                    .nodes()
                    .find_map(|node| match node {
                        KernelNode::Relation(r) => Some(r.id().erase()),
                        _ => None,
                    })
                    .unwrap()
            )
            .is_err()
    );
    let mut session = interpreter
        .execution_session(
            &program,
            config,
            [(input, clock, vec![value(0), value(-i64::MAX), value(1)])],
        )
        .unwrap();
    for (tick, expected) in [true, false].into_iter().enumerate() {
        session.advance_ticks(1).unwrap();
        assert_eq!(session.field(field).unwrap().as_bool(), Some(expected));
        assert_eq!(
            session.output(output, tick as u64).unwrap().1.as_bool(),
            Some(expected)
        );
    }
    let checkpoint = session.checkpoint();
    let mut resumed = interpreter.resume_execution(&program, &checkpoint).unwrap();
    assert!(resumed.advance_ticks(1).is_err());
    assert_eq!(resumed.field(field), checkpoint.field(field));
    assert_eq!(resumed.next_tick(), checkpoint.next_tick());
    assert_eq!(resumed.output(output, 1), checkpoint.output(output, 1));
    assert!(resumed.output(output, 2).is_none());
}

fn complex_comparisons(
    operations: [ComparisonOp; 2],
) -> Result<(KernelProgram, [RawId; 2]), Vec<eqiora_core::Diagnostic>> {
    let clock = Id::<kinds::ClockDomain>::new();
    let activation = Id::<kinds::Activation>::new();
    let complex = Id::<kinds::Parameter>::new();
    let real = Id::<kinds::Parameter>::new();
    let outputs = [Id::<kinds::Port>::new(), Id::new()];
    let dimension = eqiora_core::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let mut nodes = vec![
        ClockDomainDef::periodic(clock, RationalTime::new(1, 1).unwrap(), RationalTime::ZERO)
            .unwrap()
            .into(),
        ActivationDef::periodic(activation).into(),
        ParameterDef::new(
            complex,
            ValueLiteral::new(
                ValueType::scalar(eqiora_core::ScalarDomain::Complex, dimension)
                    .expect("valid scalar type"),
                [(1., 0.)],
            )
            .unwrap(),
        )
        .into(),
        ParameterDef::new(
            real,
            ValueLiteral::from_real(
                ValueType::scalar(eqiora_core::ScalarDomain::Real, dimension)
                    .expect("valid scalar type"),
                1.,
            )
            .unwrap(),
        )
        .into(),
    ];
    let mut edges = vec![(activation.erase(), clock.erase(), EdgeKind::ClockedBy)];
    for (output, op) in outputs.into_iter().zip(operations) {
        nodes.push(PortDef::signal(output, SignalDirection::Output, ValueType::boolean()).into());
        let relation = Id::<kinds::Relation>::new();
        let mut dag = ExprDagBuilder::new();
        let left = dag.symbol(SymbolRef::Parameter(complex)).unwrap();
        let right = dag.symbol(SymbolRef::Parameter(real)).unwrap();
        let predicate = dag.compare(op, left, right).unwrap();
        let target = dag.symbol(SymbolRef::Port(output)).unwrap();
        nodes.push(
            RelationDef::new(relation, dag.finish([target, predicate]).unwrap())
                .unwrap()
                .into(),
        );
        edges.push((output.erase(), clock.erase(), EdgeKind::ClockedBy));
        edges.push((activation.erase(), relation.erase(), EdgeKind::Activates));
        for id in [complex.erase(), real.erase(), output.erase()] {
            edges.push((relation.erase(), id, EdgeKind::DependsOn));
        }
    }
    let model = OntologyId::<Model>::new();
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("complex parameter equality emits only Boolean samples");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in edges {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, outputs.map(Id::erase))
            .unwrap()
            .into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model)
        .map(|program| (program, outputs.map(Id::erase)))
}

#[test]
fn complex_parameters_support_equality_boolean_outputs_but_not_ordering() {
    let (program, outputs) =
        complex_comparisons([ComparisonOp::Equal, ComparisonOp::NotEqual]).unwrap();
    let mut session = Interpreter::new()
        .execution_session(&program, ReferenceConfig::new(0., 1.).unwrap(), [])
        .unwrap();
    session.advance_ticks(1).unwrap();
    assert_eq!(
        session.output(outputs[0], 0).unwrap().1.as_bool(),
        Some(true)
    );
    assert_eq!(
        session.output(outputs[1], 0).unwrap().1.as_bool(),
        Some(false)
    );
    assert!(complex_comparisons([ComparisonOp::Less, ComparisonOp::NotEqual]).is_err());
}
