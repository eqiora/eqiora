#[path = "support/initial.rs"]
mod initial_support;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id, OntologyId};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{
    ActivationDef, ActivationKind, EventDirection, ExprDagBuilder, FieldDef, KernelNode,
    ParameterDef, RelationDef, SymbolRef,
};
use eqiora_schema::{Model, ModelView};
use eqiora_sem::{Interpreter, KernelProgram, ReferenceConfig, Sample};
use initial_support::{define_all, initial};

struct BouncingFixture {
    program: KernelProgram,
    height: Id<kinds::Field>,
    velocity: Id<kinds::Field>,
}

#[test]
fn falling_zero_crossing_commits_split_resets_atomically() {
    let fixture = bouncing_fixture(EventDirection::Falling, true);
    let config = ReferenceConfig::new(0.7, 0.01)
        .unwrap()
        .with_event_tolerances(1.0e-11, 1.0e-10)
        .unwrap();
    let trajectory = Interpreter::new()
        .run(&fixture.program, config)
        .expect("bouncing-ball trajectory");
    let heights = field_samples(trajectory.samples(), fixture.height);
    let velocities = field_samples(trajectory.samples(), fixture.velocity);

    let height_reset = equal_time_pair(&heights).expect("pre/post height samples");
    let velocity_reset = equal_time_pair(&velocities).expect("pre/post velocity samples");
    assert!((height_reset.0.0 - velocity_reset.0.0).abs() < 1.0e-12);
    assert!((0.43..0.46).contains(&height_reset.0.0));
    assert!(height_reset.0.1.abs() < 2.0e-9);
    assert_eq!(height_reset.1.1, 0.0);
    assert!(velocity_reset.0.1 < 0.0);
    assert!(velocity_reset.1.1 > 0.0);
    assert!((velocity_reset.1.1 + 0.8 * velocity_reset.0.1).abs() < 1.0e-8);
}

#[test]
fn crossing_direction_and_node_insertion_order_are_semantic() {
    let falling = bouncing_fixture(EventDirection::Falling, false);
    let reversed = bouncing_fixture(EventDirection::Falling, true);
    let rising = bouncing_fixture(EventDirection::Rising, true);
    let config = ReferenceConfig::new(0.55, 0.01).unwrap();

    let falling_trajectory = Interpreter::new().run(&falling.program, config).unwrap();
    let reversed_trajectory = Interpreter::new().run(&reversed.program, config).unwrap();
    let rising_trajectory = Interpreter::new().run(&rising.program, config).unwrap();

    let falling_height = falling_trajectory
        .last_value(falling.height.erase())
        .unwrap()
        .value();
    let reversed_height = reversed_trajectory
        .last_value(reversed.height.erase())
        .unwrap()
        .value();
    let falling_velocity = falling_trajectory
        .last_value(falling.velocity.erase())
        .unwrap()
        .value();
    let reversed_velocity = reversed_trajectory
        .last_value(reversed.velocity.erase())
        .unwrap()
        .value();
    assert!((falling_height - reversed_height).abs() < 1.0e-10);
    assert!((falling_velocity - reversed_velocity).abs() < 1.0e-10);
    assert!(
        equal_time_pair(&field_samples(falling_trajectory.samples(), falling.height)).is_some()
    );
    assert!(equal_time_pair(&field_samples(rising_trajectory.samples(), rising.height)).is_none());
    assert!(
        rising_trajectory
            .last_value(rising.height.erase())
            .unwrap()
            .value()
            < 0.0
    );
}

#[test]
fn coarse_step_localizes_the_backward_euler_impact_time() {
    let fixture = bouncing_fixture(EventDirection::Falling, false);
    let max_step = 0.2;
    let time_tolerance = 1.0e-11;
    let trajectory = Interpreter::new()
        .run(
            &fixture.program,
            ReferenceConfig::new(0.5, max_step)
                .unwrap()
                .with_event_tolerances(time_tolerance, 1.0e-10)
                .unwrap(),
        )
        .expect("coarse-step event trajectory");
    let impact = equal_time_pair(&field_samples(trajectory.samples(), fixture.height))
        .expect("localized pre/post impact");

    // The reference integrator is backward Euler. After the accepted step to
    // t=0.2, solve h + tau * (v - g*tau) = 0 for the positive tau.
    let gravity = 9.81;
    let velocity_at_step_start = -gravity * max_step;
    let height_at_step_start = 1.0 - gravity * max_step * max_step;
    let tau = (velocity_at_step_start
        + (velocity_at_step_start.powi(2) + 4.0 * gravity * height_at_step_start).sqrt())
        / (2.0 * gravity);
    let expected_time = max_step + tau;

    assert!((impact.0.0 - expected_time).abs() <= 2.0 * time_tolerance);
    assert!(impact.0.1.abs() <= 2.0e-9);
    assert_eq!(impact.1.1, 0.0);
}

#[test]
fn zero_time_chatter_terminates_with_a_zeno_diagnostic() {
    let program = chattering_program();
    let config = ReferenceConfig::new(0.01, 0.001)
        .unwrap()
        .with_event_tolerances(1.0e-5, 1.0e-12)
        .unwrap()
        .with_event_limits(80, 2)
        .unwrap();
    let diagnostics = Interpreter::new().run(&program, config).unwrap_err();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].code(),
        eqiora_core::diagnostic::codes::INVALID_EXECUTION_CONFIG
    );
    assert!(diagnostics[0].message().contains("Zeno"));
}

fn field_samples(
    samples: &[Sample<eqiora_core::RawId>],
    field: Id<kinds::Field>,
) -> Vec<(f64, f64)> {
    samples
        .iter()
        .filter(|sample| sample.coordinate() == field.erase())
        .map(|sample| (sample.time(), sample.value().value()))
        .collect()
}

fn equal_time_pair(samples: &[(f64, f64)]) -> Option<((f64, f64), (f64, f64))> {
    samples
        .windows(2)
        .find_map(|pair| ((pair[0].0 - pair[1].0).abs() < 1.0e-13).then_some((pair[0], pair[1])))
}

fn bouncing_fixture(direction: EventDirection, reverse_nodes: bool) -> BouncingFixture {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
    let velocity_dimension =
        DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
    let acceleration_dimension =
        DimExponents::from_integers([0, 1, -2, 0, 0, 0, 0]).expect("bounded dimension");

    let height = Id::<kinds::Field>::new();
    let velocity = Id::<kinds::Field>::new();
    let gravity = Id::<kinds::Parameter>::new();
    let restitution = Id::<kinds::Parameter>::new();
    let flight = Id::<kinds::Relation>::new();
    let reset_height = Id::<kinds::Relation>::new();
    let reset_velocity = Id::<kinds::Relation>::new();
    let continuous = Id::<kinds::Activation>::new();
    let height_event = Id::<kinds::Activation>::new();
    let velocity_event = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();

    let mut flight_expression = ExprDagBuilder::new();
    let height_rate = flight_expression
        .symbol(SymbolRef::Derivative(height))
        .unwrap();
    let velocity_value = flight_expression
        .symbol(SymbolRef::Field(velocity))
        .unwrap();

    let velocity_rate = flight_expression
        .symbol(SymbolRef::Derivative(velocity))
        .unwrap();
    let gravity_value = flight_expression
        .symbol(SymbolRef::Parameter(gravity))
        .unwrap();
    let velocity_residual = flight_expression.add(velocity_rate, gravity_value).unwrap();
    let acceleration_zero = flight_expression
        .constant(DynQuantity::new(0.0, acceleration_dimension))
        .unwrap();

    let mut height_reset_expression = ExprDagBuilder::new();
    let next_height = height_reset_expression
        .symbol(SymbolRef::Next(height))
        .unwrap();
    let zero_height = height_reset_expression
        .constant(DynQuantity::new(0.0, length))
        .unwrap();

    let mut velocity_reset_expression = ExprDagBuilder::new();
    let next_velocity = velocity_reset_expression
        .symbol(SymbolRef::Next(velocity))
        .unwrap();
    let restitution_value = velocity_reset_expression
        .symbol(SymbolRef::Parameter(restitution))
        .unwrap();
    let previous_velocity = velocity_reset_expression
        .symbol(SymbolRef::Pre(velocity))
        .unwrap();
    let reflected_velocity = velocity_reset_expression
        .mul(restitution_value, previous_velocity)
        .unwrap();
    let velocity_reset_residual = velocity_reset_expression
        .add(next_velocity, reflected_velocity)
        .unwrap();
    let velocity_zero = velocity_reset_expression
        .constant(DynQuantity::new(0.0, velocity_dimension))
        .unwrap();

    let event_guard = || {
        let mut guard = ExprDagBuilder::new();
        let height_value = guard.symbol(SymbolRef::Field(height)).unwrap();
        guard.finish([height_value]).unwrap()
    };

    let mut nodes = vec![
        KernelNode::from(FieldDef::new(
            height,
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, length)
                .expect("valid scalar type"),
            eqiora_schema::kernel::FieldRole::State,
        )),
        initial(height, DynQuantity::new(1.0, length)),
        KernelNode::from(FieldDef::new(
            velocity,
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, velocity_dimension)
                .expect("valid scalar type"),
            eqiora_schema::kernel::FieldRole::State,
        )),
        initial(velocity, DynQuantity::new(0.0, velocity_dimension)),
        KernelNode::from(ParameterDef::new(
            gravity,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    acceleration_dimension,
                )
                .expect("valid scalar type"),
                9.81,
            )
            .expect("valid parameter value"),
        )),
        KernelNode::from(ParameterDef::new(
            restitution,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    DimExponents::DIMENSIONLESS,
                )
                .expect("valid scalar type"),
                0.8,
            )
            .expect("valid parameter value"),
        )),
        KernelNode::from(
            RelationDef::new(
                flight,
                flight_expression
                    .finish([
                        height_rate,
                        velocity_value,
                        velocity_residual,
                        acceleration_zero,
                    ])
                    .unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(
            RelationDef::new(
                reset_height,
                height_reset_expression
                    .finish([next_height, zero_height])
                    .unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(
            RelationDef::new(
                reset_velocity,
                velocity_reset_expression
                    .finish([velocity_reset_residual, velocity_zero])
                    .unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(continuous)),
        KernelNode::from(
            ActivationDef::new(
                height_event,
                ActivationKind::Event {
                    guard: event_guard(),
                    direction,
                },
            )
            .unwrap(),
        ),
        KernelNode::from(
            ActivationDef::new(
                velocity_event,
                ActivationKind::Event {
                    guard: event_guard(),
                    direction,
                },
            )
            .unwrap(),
        ),
    ];
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    if reverse_nodes {
        nodes.reverse();
    }
    let mut transaction = Transaction::new("bouncing ball with split atomic reset");
    define_all(&mut transaction, nodes);
    connect_dependencies(
        &mut transaction,
        flight.erase(),
        [height.erase(), velocity.erase(), gravity.erase()],
    );
    connect_dependencies(&mut transaction, reset_height.erase(), [height.erase()]);
    connect_dependencies(
        &mut transaction,
        reset_velocity.erase(),
        [velocity.erase(), restitution.erase()],
    );
    for (activation, relation) in [
        (continuous, flight),
        (height_event, reset_height),
        (velocity_event, reset_velocity),
    ] {
        transaction.push(Op::Connect {
            from: activation.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        });
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, []).unwrap().into(),
    });

    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    BouncingFixture {
        program,
        height,
        velocity,
    }
}

fn chattering_program() -> KernelProgram {
    let inverse_time =
        DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension");
    let state = Id::<kinds::Field>::new();
    let rate = Id::<kinds::Parameter>::new();
    let reset_value = Id::<kinds::Parameter>::new();
    let flow = Id::<kinds::Relation>::new();
    let reset = Id::<kinds::Relation>::new();
    let continuous = Id::<kinds::Activation>::new();
    let event = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();

    let mut flow_expression = ExprDagBuilder::new();
    let derivative = flow_expression
        .symbol(SymbolRef::Derivative(state))
        .unwrap();
    let rate_value = flow_expression.symbol(SymbolRef::Parameter(rate)).unwrap();
    let flow_residual = flow_expression.add(derivative, rate_value).unwrap();

    let mut reset_expression = ExprDagBuilder::new();
    let next = reset_expression.symbol(SymbolRef::Next(state)).unwrap();
    let reset_parameter = reset_expression
        .symbol(SymbolRef::Parameter(reset_value))
        .unwrap();

    let mut guard = ExprDagBuilder::new();
    let guard_state = guard.symbol(SymbolRef::Field(state)).unwrap();
    let event_definition = ActivationDef::new(
        event,
        ActivationKind::Event {
            guard: guard.finish([guard_state]).unwrap(),
            direction: EventDirection::Falling,
        },
    )
    .unwrap();

    let nodes = [
        KernelNode::from(FieldDef::new(
            state,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("valid scalar type"),
            eqiora_schema::kernel::FieldRole::State,
        )),
        initial(state, DynQuantity::new(1.0e-6, DimExponents::DIMENSIONLESS)),
        KernelNode::from(ParameterDef::new(
            rate,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, inverse_time)
                    .expect("valid scalar type"),
                1.0,
            )
            .expect("valid parameter value"),
        )),
        KernelNode::from(ParameterDef::new(
            reset_value,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    DimExponents::DIMENSIONLESS,
                )
                .expect("valid scalar type"),
                1.0e-6,
            )
            .expect("valid parameter value"),
        )),
        KernelNode::from(
            RelationDef::new(
                flow,
                {
                    let equation_zero = flow_expression
                        .constant(eqiora_core::DynQuantity::new(0.0, inverse_time))
                        .unwrap();
                    flow_expression.finish([flow_residual, equation_zero])
                }
                .unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(
            RelationDef::new(
                reset,
                reset_expression.finish([next, reset_parameter]).unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(continuous)),
        KernelNode::from(event_definition),
    ];
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("deliberate zero-time chatter");
    define_all(&mut transaction, nodes);
    connect_dependencies(
        &mut transaction,
        flow.erase(),
        [state.erase(), rate.erase()],
    );
    connect_dependencies(
        &mut transaction,
        reset.erase(),
        [state.erase(), reset_value.erase()],
    );
    for (activation, relation) in [(continuous, flow), (event, reset)] {
        transaction.push(Op::Connect {
            from: activation.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        });
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, []).unwrap().into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
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

// This affine fixture isolates calendar semantics from integration error: quarter
// steps, slopes, roots, and reset values are exactly representable in binary.
struct CoincidenceFixture {
    program: KernelProgram,
    fields: [Id<kinds::Field>; 4],
    events: [Id<kinds::Activation>; 3],
    tick: Id<kinds::Activation>,
}

fn coincidence_fixture(reverse: bool, conflict: bool, root_shift: f64) -> CoincidenceFixture {
    use eqiora_core::{ScalarDomain, ValueType};
    use eqiora_schema::kernel::{ClockDomainDef, FieldRole, RationalTime};
    let fields = [Id::new(), Id::new(), Id::new(), Id::new()];
    let [x, y, z, memory] = fields;
    let events = [Id::new(), Id::new(), Id::new()];
    let tick = Id::new();
    let continuous = Id::new();
    let clock = Id::new();
    let model = OntologyId::<Model>::new();
    let one = DimExponents::DIMENSIONLESS;
    let rate = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let mut nodes = Vec::<KernelNode>::new();
    let mut edges = Vec::new();
    for field in fields {
        nodes.push(
            FieldDef::new(
                field,
                ValueType::scalar(ScalarDomain::Real, one).expect("valid scalar type"),
                FieldRole::State,
            )
            .into(),
        );
        nodes.push(initial(field, DynQuantity::new(0.0, one)));
    }
    nodes.push(
        ClockDomainDef::periodic(
            clock,
            RationalTime::new(1, 1).unwrap(),
            RationalTime::new(1, 1).unwrap(),
        )
        .unwrap()
        .into(),
    );
    edges.push((memory.erase(), clock.erase(), EdgeKind::ClockedBy));
    nodes.push(ActivationDef::continuous(continuous).into());
    nodes.push(ActivationDef::periodic(tick).into());
    edges.push((tick.erase(), clock.erase(), EdgeKind::ClockedBy));
    for (field, slope) in [(x, 1.0), (y, 2.0), (z, 0.0)] {
        let relation = Id::new();
        let mut dag = ExprDagBuilder::new();
        let lhs = dag.symbol(SymbolRef::Derivative(field)).unwrap();
        let rhs = dag.constant(DynQuantity::new(slope, rate)).unwrap();
        nodes.push(
            RelationDef::new(relation, dag.finish([lhs, rhs]).unwrap())
                .unwrap()
                .into(),
        );
        edges.push((continuous.erase(), relation.erase(), EdgeKind::Activates));
        edges.push((relation.erase(), field.erase(), EdgeKind::DependsOn));
    }
    for (activation, guard_field, threshold, target, reset) in [
        (events[0], x, 1.0 + root_shift, x, 10.0),
        (events[1], y, 2.0, if conflict { x } else { y }, 20.0),
        (events[2], x, 5.0, z, 7.0),
    ] {
        let mut guard = ExprDagBuilder::new();
        let value = guard.symbol(SymbolRef::Field(guard_field)).unwrap();
        let threshold = guard.constant(DynQuantity::new(threshold, one)).unwrap();
        let residual = guard.sub(value, threshold).unwrap();
        nodes.push(
            ActivationDef::new(
                activation,
                ActivationKind::Event {
                    guard: guard.finish([residual]).unwrap(),
                    direction: EventDirection::Rising,
                },
            )
            .unwrap()
            .into(),
        );
        let relation = Id::new();
        let mut dag = ExprDagBuilder::new();
        let next = dag.symbol(SymbolRef::Next(target)).unwrap();
        let value = dag.constant(DynQuantity::new(reset, one)).unwrap();
        nodes.push(
            RelationDef::new(relation, dag.finish([next, value]).unwrap())
                .unwrap()
                .into(),
        );
        edges.push((activation.erase(), relation.erase(), EdgeKind::Activates));
        edges.push((relation.erase(), target.erase(), EdgeKind::DependsOn));
    }
    let relation = Id::new();
    let mut dag = ExprDagBuilder::new();
    let next = dag.symbol(SymbolRef::Next(memory)).unwrap();
    let x_value = dag.symbol(SymbolRef::Field(x)).unwrap();
    let y_value = dag.symbol(SymbolRef::Field(y)).unwrap();
    let sum = dag.add(x_value, y_value).unwrap();
    let sample = dag.sample(sum, clock).unwrap();
    nodes.push(
        RelationDef::new(relation, dag.finish([next, sample]).unwrap())
            .unwrap()
            .into(),
    );
    edges.push((tick.erase(), relation.erase(), EdgeKind::Activates));
    for dependency in [x.erase(), y.erase(), memory.erase(), clock.erase()] {
        edges.push((relation.erase(), dependency, EdgeKind::DependsOn));
    }
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    if reverse {
        nodes.reverse();
        edges.reverse();
    }
    let mut transaction = Transaction::new("affine coincidence with left-state sampling");
    define_all(&mut transaction, nodes);
    for (from, to, edge) in edges {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, []).unwrap().into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    CoincidenceFixture {
        program: KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        fields,
        events,
        tick,
    }
}

fn coincidence_config() -> ReferenceConfig {
    ReferenceConfig::new(1.25, 0.25)
        .unwrap()
        .with_event_tolerances(2.0_f64.powi(-30), 2.0_f64.powi(-40))
        .unwrap()
}

#[test]
fn affine_coincidence_samples_left_state_and_restarts_only_stabilized_boundaries() {
    for reverse in [false, true] {
        let fixture = coincidence_fixture(reverse, false, 0.0);
        let interpreter = Interpreter::new();
        let mut session = interpreter
            .execution_session(&fixture.program, coincidence_config(), [])
            .unwrap();
        for _ in 0..3 {
            assert!(session.advance().unwrap());
        }
        assert_eq!(session.progress().model_time(), 0.75);
        let before = session.checkpoint();
        assert!(session.advance().unwrap());
        assert_eq!(session.progress().model_time(), 1.0);
        for (field, expected) in fixture.fields.into_iter().zip([10.0, 20.0, 7.0, 3.0]) {
            assert_eq!(
                session
                    .field(field.erase())
                    .unwrap()
                    .real_scalar_value()
                    .map(|value| value.value()),
                Some(expected)
            );
        }
        let sequence = session.activation_sequence();
        let group = sequence
            .iter()
            .find(|group| group.contains(&fixture.tick.erase()))
            .unwrap();
        assert_eq!(group.len(), 3);
        assert!(group.contains(&fixture.events[0].erase()));
        assert!(group.contains(&fixture.events[1].erase()));
        assert_eq!(sequence.last().unwrap(), &[fixture.events[2].erase()]);
        let after = session.checkpoint();
        assert!(session.advance().unwrap());
        for (field, expected) in fixture.fields.into_iter().zip([10.25, 20.5, 7.0, 3.0]) {
            assert_eq!(
                session
                    .field(field.erase())
                    .unwrap()
                    .real_scalar_value()
                    .map(|value| value.value()),
                Some(expected)
            );
        }
        for checkpoint in [before, after] {
            let mut resumed = interpreter
                .resume_execution(&fixture.program, &checkpoint)
                .unwrap();
            while resumed.advance().unwrap() {}
            assert_eq!(resumed.activation_sequence(), session.activation_sequence());
            for field in fixture.fields {
                assert_eq!(resumed.field(field.erase()), session.field(field.erase()));
            }
        }
    }
}

#[test]
fn conflicting_coincident_resets_reject_without_advancing_the_checkpoint() {
    let fixture = coincidence_fixture(false, true, 0.0);
    let mut session = Interpreter::new()
        .execution_session(&fixture.program, coincidence_config(), [])
        .unwrap();
    for _ in 0..3 {
        session.advance().unwrap();
    }
    let before = session.checkpoint();
    assert!(session.advance().is_err());
    assert_eq!(session.progress(), before.progress());
    assert_eq!(session.activation_sequence(), before.activation_sequence());
    for field in fixture.fields {
        assert_eq!(session.field(field.erase()), before.field(field.erase()));
    }
}

#[test]
fn unresolved_near_tick_crossing_rejects_instead_of_tolerance_snapping() {
    let fixture = coincidence_fixture(false, false, -2.0_f64.powi(-20));
    let config = ReferenceConfig::new(1.25, 0.25)
        .unwrap()
        .with_event_tolerances(2.0_f64.powi(-10), 2.0_f64.powi(-40))
        .unwrap();
    let mut session = Interpreter::new()
        .execution_session(&fixture.program, config, [])
        .unwrap();
    for _ in 0..3 {
        session.advance().unwrap();
    }
    let before = session.checkpoint();
    assert!(session.advance().is_err());
    assert_eq!(session.progress(), before.progress());
    for field in fixture.fields {
        assert_eq!(session.field(field.erase()), before.field(field.erase()));
    }
}

#[test]
fn resolved_near_tick_crossing_precedes_the_tick_without_becoming_coincident() {
    let delta = 2.0_f64.powi(-20);
    let fixture = coincidence_fixture(false, false, -delta);
    let mut session = Interpreter::new()
        .execution_session(&fixture.program, coincidence_config(), [])
        .unwrap();
    let mut boundaries = Vec::new();
    while session.advance().unwrap() {
        boundaries.push((
            session.progress().model_time(),
            session.activation_sequence().to_vec(),
        ));
    }
    // x resets at 1-delta, then advances by delta before the tick. Its
    // left-state tick contribution is 10+delta, while y contributes 2.
    // 2^-26 is sixteen localization tolerances and exceeds the propagated
    // affine time uncertainty (slopes at most two) without masking delta.
    for (field, expected) in
        fixture
            .fields
            .into_iter()
            .zip([10.25 + delta, 20.5, 7.0, 12.0 + delta])
    {
        let value = session
            .field(field.erase())
            .unwrap()
            .real_scalar_value()
            .unwrap()
            .value();
        assert!(
            (value - expected).abs() <= 2.0_f64.powi(-26),
            "{value} != {expected}"
        );
    }
    let event = boundaries
        .iter()
        .find(|(_, groups)| {
            groups
                .iter()
                .any(|group| group.contains(&fixture.events[0].erase()))
        })
        .unwrap();
    let tick = boundaries
        .iter()
        .find(|(_, groups)| {
            groups
                .iter()
                .any(|group| group.contains(&fixture.tick.erase()))
        })
        .unwrap();
    assert!(event.0 < tick.0);
    assert_eq!(tick.0, 1.0);
    assert!(
        tick.1
            .iter()
            .all(|group| !group.contains(&fixture.events[0].erase()))
    );
}

#[test]
fn armed_crossings_survive_band_entry_without_moving_the_numerical_root() {
    let fixture = coincidence_fixture(false, false, 0.0);
    // Before the root, x-1 enters the arming band at -1/8 and then -1/16.
    // Neither point is a zero crossing; retained arming must reach t=1.
    let config = ReferenceConfig::new(1.25, 1.0 / 16.0)
        .unwrap()
        .with_event_tolerances(2.0_f64.powi(-30), 1.0 / 8.0)
        .unwrap();
    let mut session = Interpreter::new()
        .execution_session(&fixture.program, config, [])
        .unwrap();
    for _ in 0..15 {
        assert!(session.advance().unwrap());
    }
    assert_eq!(session.progress().model_time(), 15.0 / 16.0);
    for (field, expected) in fixture
        .fields
        .into_iter()
        .zip([15.0 / 16.0, 15.0 / 8.0, 0.0, 0.0])
    {
        assert_eq!(
            session
                .field(field.erase())
                .unwrap()
                .real_scalar_value()
                .unwrap()
                .value(),
            expected
        );
    }
    let checkpoint = session.checkpoint();
    let mut resumed = Interpreter::new()
        .resume_execution(&fixture.program, &checkpoint)
        .unwrap();
    assert!(resumed.advance().unwrap());
    assert_eq!(resumed.progress().model_time(), 1.0);
    for (field, expected) in fixture.fields.into_iter().zip([10.0, 20.0, 7.0, 3.0]) {
        assert_eq!(
            resumed
                .field(field.erase())
                .unwrap()
                .real_scalar_value()
                .unwrap()
                .value(),
            expected
        );
    }
}
