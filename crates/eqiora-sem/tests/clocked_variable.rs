use eqiora_core::{
    DimExponents, DynQuantity, Id, OntologyId, RawId, ScalarDomain, ValueLiteral, ValueType,
    entity::kinds,
};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::{
    Model, ModelView,
    kernel::{
        ActivationDef, ClockDomainDef, ExprDagBuilder, FieldDef, FieldRole, KernelNode, PortDef,
        RationalTime, RelationDef, SignalDirection, SymbolRef,
    },
};
use eqiora_sem::{Interpreter, KernelProgram, ReferenceConfig};

struct Fixture {
    program: KernelProgram,
    field: RawId,
    clocks: [RawId; 2],
    inputs: [RawId; 2],
}
fn value_type() -> ValueType {
    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).expect("valid scalar type")
}
fn fixture(mode: &str) -> Result<Fixture, Vec<eqiora_core::Diagnostic>> {
    let field = Id::<kinds::Field>::new();
    let clocks = [Id::<kinds::ClockDomain>::new(), Id::new()];
    let inputs = [Id::<kinds::Port>::new(), Id::new()];
    let relation = Id::<kinds::Relation>::new();
    let activation = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();
    let mut dag = ExprDagBuilder::new();
    let y = dag
        .symbol(match mode {
            "pre" => SymbolRef::Pre(field),
            "next" => SymbolRef::Next(field),
            "derivative" => SymbolRef::Derivative(field, std::num::NonZeroU32::MIN),
            _ => SymbolRef::Field(field),
        })
        .unwrap();
    let residual = match mode {
        "initial" | "continuous" | "other-clock" => y,
        "sample" => dag.sample(y, clocks[0]).unwrap(),
        "hold" => dag.hold(y).unwrap(),
        "singular" => dag.sub(y, y).unwrap(),
        _ => {
            let input = dag.symbol(SymbolRef::Port(inputs[0])).unwrap();
            let two = dag
                .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
                .unwrap();
            let doubled = dag.mul(two, input).unwrap();
            dag.sub(y, doubled).unwrap()
        }
    };
    let residuals = {
        let equation_zero = dag
            .constant(eqiora_core::DynQuantity::new(
                0.0,
                eqiora_core::DimExponents::DIMENSIONLESS,
            ))
            .unwrap();
        dag.finish([residual, equation_zero])
    }
    .unwrap();
    let mut nodes = vec![
        KernelNode::from(FieldDef::new(field, value_type(), FieldRole::Variable)),
        if mode == "initial" {
            RelationDef::initial(relation, residuals).unwrap().into()
        } else {
            RelationDef::new(relation, residuals).unwrap().into()
        },
    ];
    let mut edges = vec![(field.erase(), clocks[0].erase(), EdgeKind::ClockedBy)];
    if mode == "duplicate-clock" {
        edges.push((field.erase(), clocks[1].erase(), EdgeKind::ClockedBy));
    }
    if mode != "initial" {
        nodes.push(if mode == "continuous" {
            ActivationDef::continuous(activation).into()
        } else {
            ActivationDef::periodic(activation).into()
        });
        edges.push((activation.erase(), relation.erase(), EdgeKind::Activates));
        if mode != "continuous" {
            edges.push((
                activation.erase(),
                clocks[usize::from(mode == "other-clock")].erase(),
                EdgeKind::ClockedBy,
            ));
        }
    }
    for i in 0..2 {
        nodes.push(
            ClockDomainDef::periodic(
                clocks[i],
                RationalTime::new(1, 1).unwrap(),
                RationalTime::new(2 + i as u64, 2).unwrap(),
            )
            .unwrap()
            .into(),
        );
        nodes.push(PortDef::signal(inputs[i], SignalDirection::Input, value_type()).into());
        edges.push((inputs[i].erase(), clocks[i].erase(), EdgeKind::ClockedBy));
        if i == 0
            && matches!(
                mode,
                "ordinary" | "pre" | "next" | "derivative" | "duplicate-clock"
            )
        {
            edges.push((relation.erase(), inputs[i].erase(), EdgeKind::DependsOn));
        }
    }
    if mode == "sample" {
        edges.push((relation.erase(), clocks[0].erase(), EdgeKind::DependsOn));
    }
    edges.push((relation.erase(), field.erase(), EdgeKind::DependsOn));
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("clocked algebraic variable");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in edges {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, inputs.map(Id::erase))
            .unwrap()
            .into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).map(|program| Fixture {
        program,
        field: field.erase(),
        clocks: clocks.map(Id::erase),
        inputs: inputs.map(Id::erase),
    })
}
fn inputs(f: &Fixture) -> Vec<(RawId, RawId, Vec<ValueLiteral>)> {
    [[3., 5., 7.].as_slice(), [9., 9.].as_slice()]
        .into_iter()
        .enumerate()
        .map(|(i, values)| {
            (
                f.inputs[i],
                f.clocks[i],
                values
                    .iter()
                    .map(|&value| ValueLiteral::from_real(value_type(), value).unwrap())
                    .collect(),
            )
        })
        .collect()
}
#[test]
fn clocked_variable_has_only_its_current_tick_value_and_restarts_without_initializing_it() {
    let f = fixture("ordinary").unwrap();
    for seed in [0., 11.] {
        let config = ReferenceConfig::new(3., 0.25)
            .unwrap()
            .with_initial_guess(seed)
            .unwrap();
        assert!(
            !Interpreter::default()
                .initialize(&f.program, 0.0, config)
                .unwrap()
                .fields()
                .contains_key(&f.field)
        );
        let mut session = Interpreter::default()
            .execution_session(&f.program, config, inputs(&f))
            .unwrap();
        assert!(session.field(f.field).is_none());
        for expected in [Some(6.), None, Some(10.), None, Some(14.)] {
            assert_eq!(session.advance_ticks(1).unwrap(), 1);
            assert_eq!(
                session
                    .field(f.field)
                    .map(|value| value.real_scalar_value().unwrap().value()),
                expected
            );
            let checkpoint = session.checkpoint();
            session = Interpreter::default()
                .resume_execution(&f.program, &checkpoint)
                .unwrap();
            assert_eq!(
                session
                    .field(f.field)
                    .map(|value| value.real_scalar_value().unwrap().value()),
                expected
            );
        }
    }
}
#[test]
fn clocked_variable_cannot_acquire_state_or_cross_activation_privileges() {
    for mode in [
        "pre",
        "next",
        "derivative",
        "initial",
        "continuous",
        "other-clock",
        "sample",
        "hold",
        "duplicate-clock",
    ] {
        assert!(fixture(mode).is_err(), "{mode}");
    }
}
#[test]
fn zero_residual_does_not_fabricate_a_clocked_algebraic_value() {
    let f = fixture("singular").unwrap();
    let config = ReferenceConfig::new(3., 0.25).unwrap();
    let mut session = Interpreter::default()
        .execution_session(&f.program, config, inputs(&f))
        .unwrap();
    let before = session.next_tick();
    assert!(session.advance_ticks(1).is_err());
    assert!(session.field(f.field).is_none());
    assert_eq!(session.next_tick(), before);
}
