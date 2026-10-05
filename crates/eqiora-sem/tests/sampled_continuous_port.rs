use eqiora_core::{
    DimExponents, DynQuantity, Id, OntologyId, ScalarDomain, ValueType, entity::kinds,
};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::{
    Model, ModelView,
    kernel::{
        ActivationDef, ClockDomainDef, ConnectionDef, ConnectionSemantics, ExprDagBuilder,
        KernelNode, PortDef, RationalTime, RelationDef, SignalDirection, SymbolRef,
    },
};
use eqiora_sem::{Interpreter, KernelProgram, ReferenceConfig};

#[test]
fn sampled_continuous_source_remains_an_initial_and_continuous_unknown() {
    let source = Id::<kinds::Port>::new();
    let input = Id::<kinds::Port>::new();
    let output = Id::<kinds::Port>::new();
    let connection = Id::<kinds::Connection>::new();
    let clock = Id::<kinds::ClockDomain>::new();
    let continuous = Id::<kinds::Activation>::new();
    let periodic = Id::<kinds::Activation>::new();
    let measurement = Id::<kinds::Relation>::new();
    let sample = Id::<kinds::Relation>::new();
    let model = OntologyId::<Model>::new();
    let scalar = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
        .expect("valid scalar type");
    let mut law = ExprDagBuilder::new();
    let measured = law.symbol(SymbolRef::Port(source)).unwrap();
    let time = law.symbol(SymbolRef::Time).unwrap();
    let second = law
        .constant(DynQuantity::new(
            1.,
            DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap(),
        ))
        .unwrap();
    let elapsed = law.div(time, second).unwrap();
    let three = law
        .constant(DynQuantity::new(3., DimExponents::DIMENSIONLESS))
        .unwrap();
    let expected = law.add(three, elapsed).unwrap();

    let law = law.finish([measured, expected]).unwrap();
    let mut sampled = ExprDagBuilder::new();
    let ingress = sampled.symbol(SymbolRef::Port(input)).unwrap();
    let sampled_input = sampled.sample(ingress, clock).unwrap();
    let emitted = sampled.symbol(SymbolRef::Port(output)).unwrap();

    let nodes = vec![
        KernelNode::from(PortDef::signal(
            source,
            SignalDirection::Output,
            scalar.clone(),
        )),
        PortDef::signal(input, SignalDirection::Input, scalar.clone()).into(),
        PortDef::signal(output, SignalDirection::Output, scalar).into(),
        ConnectionDef::new(connection, ConnectionSemantics::Signal { driver: source }).into(),
        ClockDomainDef::periodic(
            clock,
            RationalTime::new(1, 1).unwrap(),
            RationalTime::new(1, 1).unwrap(),
        )
        .unwrap()
        .into(),
        ActivationDef::continuous(continuous).into(),
        ActivationDef::periodic(periodic).into(),
        RelationDef::new(measurement, law).unwrap().into(),
        RelationDef::new(sample, sampled.finish([emitted, sampled_input]).unwrap())
            .unwrap()
            .into(),
    ];
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("sample explicit continuous ingress");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in [
        (connection.erase(), source.erase(), EdgeKind::Connects),
        (connection.erase(), input.erase(), EdgeKind::Connects),
        (continuous.erase(), measurement.erase(), EdgeKind::Activates),
        (periodic.erase(), sample.erase(), EdgeKind::Activates),
        (periodic.erase(), clock.erase(), EdgeKind::ClockedBy),
        (output.erase(), clock.erase(), EdgeKind::ClockedBy),
        (measurement.erase(), source.erase(), EdgeKind::DependsOn),
        (sample.erase(), input.erase(), EdgeKind::DependsOn),
        (sample.erase(), output.erase(), EdgeKind::DependsOn),
        (sample.erase(), clock.erase(), EdgeKind::DependsOn),
    ] {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, [output.erase()])
            .unwrap()
            .into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let config = ReferenceConfig::new(2., 0.25).unwrap();
    // Before a tick, source = 3 + t/1s is still one ordinary continuous equation.
    Interpreter::new()
        .initialize(&program, 0.0, config)
        .unwrap();
    let mut session = Interpreter::new()
        .execution_session(&program, config, [])
        .unwrap();
    assert!(session.output(output.erase(), 0).is_none());
    for (tick, expected) in [(0, 4.), (1, 5.)] {
        assert_eq!(session.advance_ticks(1).unwrap(), 1);
        let (time, value) = session.output(output.erase(), tick).unwrap();
        assert_eq!(time, RationalTime::new(tick + 1, 1).unwrap());
        assert_eq!(value.real_scalar_value().unwrap().value(), expected);
    }
}
