#[path = "support/initial.rs"]
mod initial_support;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id, OntologyId};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{
    ActivationDef, ClockDomainDef, ConnectionDef, ConnectionSemantics, ExprDagBuilder, FieldDef,
    KernelNode, ParameterDef, PortDef, RationalTime, RelationDef, SignalDirection, SymbolRef,
};
use eqiora_schema::{Model, ModelView};
use eqiora_sem::{Interpreter, KernelProgram, ReferenceConfig};
use initial_support::{define_all, initial};

struct ThermalFixture {
    program: KernelProgram,
    temperature: Id<kinds::Field>,
    command: Id<kinds::Field>,
}

#[test]
fn sampled_controller_holds_between_exact_ticks() {
    let fixture = thermal_fixture();
    let trajectory = Interpreter::new()
        .run(
            &fixture.program,
            ReferenceConfig::new(0.5, 0.1).expect("config"),
        )
        .expect("reference trajectory");

    let temperature = trajectory
        .last_value(fixture.temperature.erase())
        .expect("temperature sample")
        .value();
    let command = trajectory
        .last_value(fixture.command.erase())
        .expect("command sample")
        .value();
    let expected_temperature = 307.0 - 14.0 / 1.01_f64.powi(5);

    assert!((command - 0.7).abs() < 1.0e-10);
    assert!((temperature - expected_temperature).abs() < 1.0e-8);
}

#[test]
fn periodic_update_samples_the_continuous_state_at_the_tick() {
    let fixture = thermal_fixture();
    let trajectory = Interpreter::new()
        .run(
            &fixture.program,
            ReferenceConfig::new(1.0, 0.1).expect("config"),
        )
        .expect("reference trajectory");

    let temperature = trajectory
        .last_value(fixture.temperature.erase())
        .expect("temperature sample")
        .value();
    let command = trajectory
        .last_value(fixture.command.erase())
        .expect("command sample")
        .value();

    assert!((command - 0.1 * (300.0 - temperature)).abs() < 1.0e-9);
    assert_eq!(
        trajectory
            .samples()
            .iter()
            .filter(|sample| sample.coordinate() == fixture.command.erase())
            .count(),
        11
    );
}

#[test]
fn backward_euler_reference_error_decreases_with_step_size() {
    let fixture = thermal_fixture();
    let exact = 307.0 - 14.0 * (-0.05_f64).exp();
    let error = |max_step| {
        let trajectory = Interpreter::new()
            .run(
                &fixture.program,
                ReferenceConfig::new(0.5, max_step).expect("config"),
            )
            .expect("reference trajectory");
        (trajectory
            .last_value(fixture.temperature.erase())
            .expect("temperature")
            .value()
            - exact)
            .abs()
    };

    let coarse = error(0.2);
    let medium = error(0.1);
    let fine = error(0.05);
    assert!(medium < coarse, "{medium} !< {coarse}");
    assert!(fine < medium, "{fine} !< {medium}");
}

fn thermal_fixture() -> ThermalFixture {
    let temperature_dimension =
        DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).expect("bounded dimension");
    let time_dimension =
        DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");
    let temperature_rate_dimension =
        DimExponents::from_integers([0, 0, -1, 0, 1, 0, 0]).expect("bounded dimension");
    let inverse_temperature_dimension =
        DimExponents::from_integers([0, 0, 0, 0, -1, 0, 0]).expect("bounded dimension");

    let temperature = Id::<kinds::Field>::new();
    let command = Id::<kinds::Field>::new();
    let ambient = Id::<kinds::Parameter>::new();
    let time_constant = Id::<kinds::Parameter>::new();
    let heating_gain = Id::<kinds::Parameter>::new();
    let setpoint = Id::<kinds::Parameter>::new();
    let controller_gain = Id::<kinds::Parameter>::new();
    let controller_output = Id::<kinds::Port>::new();
    let plant_input = Id::<kinds::Port>::new();
    let plant_relation = Id::<kinds::Relation>::new();
    let controller_relation = Id::<kinds::Relation>::new();
    let held_relation = Id::<kinds::Relation>::new();
    let continuous = Id::<kinds::Activation>::new();
    let periodic = Id::<kinds::Activation>::new();
    let controller_clock = Id::<kinds::ClockDomain>::new();
    let signal = Id::<kinds::Connection>::new();
    let model = OntologyId::<Model>::new();

    let mut plant = ExprDagBuilder::new();
    let derivative = plant
        .symbol(SymbolRef::Derivative(
            temperature,
            std::num::NonZeroU32::MIN,
        ))
        .expect("dT/dt");
    let ambient_value = plant
        .symbol(SymbolRef::Parameter(ambient))
        .expect("ambient");
    let temperature_value = plant
        .symbol(SymbolRef::Field(temperature))
        .expect("temperature");
    let cooling_delta = plant
        .sub(ambient_value, temperature_value)
        .expect("ambient - T");
    let tau = plant
        .symbol(SymbolRef::Parameter(time_constant))
        .expect("tau");
    let cooling = plant.div(cooling_delta, tau).expect("cooling rate");
    let gain = plant
        .symbol(SymbolRef::Parameter(heating_gain))
        .expect("heating gain");
    let input = plant
        .symbol(SymbolRef::Port(plant_input))
        .expect("held control input");
    let heating = plant.mul(gain, input).expect("heating rate");
    let rate = plant.add(cooling, heating).expect("total rate");

    let mut controller = ExprDagBuilder::new();
    let next_command = controller
        .symbol(SymbolRef::Next(command))
        .expect("next command");
    let setpoint_value = controller
        .symbol(SymbolRef::Parameter(setpoint))
        .expect("setpoint");
    let sampled_temperature = controller
        .symbol(SymbolRef::Field(temperature))
        .expect("sampled temperature");
    let sampled_temperature = controller
        .sample(sampled_temperature, controller_clock)
        .unwrap();
    let error = controller
        .sub(setpoint_value, sampled_temperature)
        .expect("control error");
    let proportional_gain = controller
        .symbol(SymbolRef::Parameter(controller_gain))
        .expect("controller gain");
    let control = controller
        .mul(proportional_gain, error)
        .expect("control law");

    let mut held = ExprDagBuilder::new();
    let output = held.symbol(SymbolRef::Port(controller_output)).unwrap();
    let memory = held.symbol(SymbolRef::Field(command)).unwrap();
    let retained = held.hold(memory).unwrap();

    let nodes = [
        KernelNode::from(FieldDef::new(
            temperature,
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, temperature_dimension)
                .expect("valid scalar type"),
            eqiora_schema::kernel::FieldRole::State,
        )),
        initial(temperature, DynQuantity::new(293.0, temperature_dimension)),
        KernelNode::from(FieldDef::new(
            command,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("valid scalar type"),
            eqiora_schema::kernel::FieldRole::State,
        )),
        initial(command, DynQuantity::new(0.0, DimExponents::DIMENSIONLESS)),
        KernelNode::from(ParameterDef::new(
            ambient,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    temperature_dimension,
                )
                .expect("valid scalar type"),
                293.0,
            )
            .expect("valid parameter value"),
        )),
        KernelNode::from(ParameterDef::new(
            time_constant,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, time_dimension)
                    .expect("valid scalar type"),
                10.0,
            )
            .expect("valid parameter value"),
        )),
        KernelNode::from(ParameterDef::new(
            heating_gain,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    temperature_rate_dimension,
                )
                .expect("valid scalar type"),
                2.0,
            )
            .expect("valid parameter value"),
        )),
        KernelNode::from(ParameterDef::new(
            setpoint,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    temperature_dimension,
                )
                .expect("valid scalar type"),
                300.0,
            )
            .expect("valid parameter value"),
        )),
        KernelNode::from(ParameterDef::new(
            controller_gain,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    inverse_temperature_dimension,
                )
                .expect("valid scalar type"),
                0.1,
            )
            .expect("valid parameter value"),
        )),
        KernelNode::from(PortDef::signal(
            controller_output,
            SignalDirection::Output,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("valid scalar type"),
        )),
        KernelNode::from(PortDef::signal(
            plant_input,
            SignalDirection::Input,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("valid scalar type"),
        )),
        KernelNode::from(
            RelationDef::new(
                plant_relation,
                plant.finish([derivative, rate]).expect("plant DAG"),
            )
            .unwrap(),
        ),
        KernelNode::from(
            RelationDef::new(
                controller_relation,
                controller
                    .finish([next_command, control])
                    .expect("controller DAG"),
            )
            .unwrap(),
        ),
        KernelNode::from(
            RelationDef::new(held_relation, held.finish([output, retained]).unwrap()).unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(continuous)),
        KernelNode::from(ActivationDef::periodic(periodic)),
        KernelNode::from(
            ClockDomainDef::periodic(
                controller_clock,
                RationalTime::new(1, 1).expect("one second"),
                RationalTime::ZERO,
            )
            .expect("periodic clock"),
        ),
        KernelNode::from(ConnectionDef::new(
            signal,
            ConnectionSemantics::Signal {
                driver: controller_output,
            },
        )),
    ];

    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("thermal plant with sampled controller");
    define_all(&mut transaction, nodes);
    connect_dependencies(
        &mut transaction,
        plant_relation.erase(),
        [
            temperature.erase(),
            ambient.erase(),
            time_constant.erase(),
            heating_gain.erase(),
            plant_input.erase(),
        ],
    );
    connect_dependencies(
        &mut transaction,
        controller_relation.erase(),
        [
            command.erase(),
            temperature.erase(),
            setpoint.erase(),
            controller_gain.erase(),
            controller_clock.erase(),
        ],
    );
    connect_dependencies(
        &mut transaction,
        held_relation.erase(),
        [command.erase(), controller_output.erase()],
    );
    transaction.push(Op::Connect {
        from: continuous.erase(),
        to: held_relation.erase(),
        edge: EdgeKind::Activates,
    });
    transaction.push(Op::Connect {
        from: command.erase(),
        to: controller_clock.erase(),
        edge: EdgeKind::ClockedBy,
    });
    transaction
        .push(Op::Connect {
            from: plant_relation.erase(),
            to: plant_input.erase(),
            edge: EdgeKind::HasPort,
        })
        .push(Op::Connect {
            from: held_relation.erase(),
            to: controller_output.erase(),
            edge: EdgeKind::HasPort,
        })
        .push(Op::Connect {
            from: continuous.erase(),
            to: plant_relation.erase(),
            edge: EdgeKind::Activates,
        })
        .push(Op::Connect {
            from: periodic.erase(),
            to: controller_relation.erase(),
            edge: EdgeKind::Activates,
        })
        .push(Op::Connect {
            from: periodic.erase(),
            to: controller_clock.erase(),
            edge: EdgeKind::ClockedBy,
        })
        .push(Op::Connect {
            from: signal.erase(),
            to: controller_output.erase(),
            edge: EdgeKind::Connects,
        })
        .push(Op::Connect {
            from: signal.erase(),
            to: plant_input.erase(),
            edge: EdgeKind::Connects,
        })
        .push(Op::DefineOntologyView {
            view: ModelView::new(model, members, [])
                .expect("closed ModelView")
                .into(),
        });

    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("valid graph");
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).expect("valid program");
    ThermalFixture {
        program,
        temperature,
        command,
    }
}

fn connect_dependencies<const N: usize>(
    transaction: &mut Transaction,
    relation: eqiora_core::RawId,
    dependencies: [eqiora_core::RawId; N],
) {
    for dependency in dependencies {
        transaction.push(Op::Connect {
            from: relation,
            to: dependency,
            edge: EdgeKind::DependsOn,
        });
    }
}

#[test]
fn controlled_reference_cancels_only_after_an_accepted_boundary() {
    struct Stop;
    impl eqiora_sem::ExecutionObserver for Stop {
        fn observe(&mut self, _: eqiora_sem::ExecutionProgress) -> std::ops::ControlFlow<()> {
            std::ops::ControlFlow::Break(())
        }
    }
    let fixture = thermal_fixture();
    let outcome = Interpreter::new()
        .run_controlled(
            &fixture.program,
            ReferenceConfig::new(1., 0.1).unwrap(),
            &mut Stop,
        )
        .unwrap();
    let eqiora_sem::ExecutionOutcome::Cancelled(progress) = outcome else {
        panic!("observer requested cancellation");
    };
    assert_eq!(progress.model_time(), 0.);
    assert_eq!(progress.accepted_steps(), 1);
}
