//! Native event acceptance at the accepted-state transaction boundary.
use super::*;
use eqiora_core::{DimExponents, Id, OntologyId, ValueType, entity::kinds};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{
    ActivationDef, ActivationKind, EventDirection, ExprDagBuilder, FieldDef, FieldRole, RelationDef,
};
use eqiora_schema::{Model, ModelView};

fn program(nodes: Vec<KernelNode>, activations: &[(RawId, RawId)]) -> KernelProgram {
    program_with_edges(nodes, activations, &[])
}

fn program_with_edges(
    nodes: Vec<KernelNode>,
    activations: &[(RawId, RawId)],
    edges: &[(RawId, RawId, EdgeKind)],
) -> KernelProgram {
    let model = OntologyId::<Model>::new();
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut tx = Transaction::new("native event contract");
    for node in &nodes {
        tx.push(Op::DefineKernelNode { node: node.clone() });
    }
    for node in &nodes {
        if let KernelNode::Relation(relation) = node {
            let dependencies = relation
                .expression()
                .nodes()
                .iter()
                .filter_map(|node| match node {
                    ExprNode::Symbol(
                        SymbolRef::Field(id)
                        | SymbolRef::Derivative(id, std::num::NonZeroU32::MIN)
                        | SymbolRef::Pre(id)
                        | SymbolRef::Next(id),
                    ) => Some(id.erase()),
                    _ => None,
                })
                .collect::<BTreeSet<_>>();
            for to in dependencies {
                tx.push(Op::Connect {
                    from: relation.id().erase(),
                    to,
                    edge: EdgeKind::DependsOn,
                });
            }
        }
    }
    for &(from, to) in activations {
        tx.push(Op::Connect {
            from,
            to,
            edge: EdgeKind::Activates,
        });
    }
    for &(from, to, edge) in edges {
        tx.push(Op::Connect { from, to, edge });
    }
    tx.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, []).unwrap().into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(tx).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn initial(field: Id<kinds::Field>, dimension: DimExponents, value: f64) -> KernelNode {
    let mut b = ExprDagBuilder::new();
    let target = b.symbol(SymbolRef::Field(field)).unwrap();
    let value = b.constant(DynQuantity::new(value, dimension)).unwrap();
    RelationDef::initial(Id::new(), b.finish([target, value]).unwrap())
        .unwrap()
        .into()
}

#[test]
fn thermostat_thresholds_rearm_and_omitted_states_remain_continuous() {
    let temperature = Id::<kinds::Field>::new();
    let rate = Id::<kinds::Field>::new();
    let kelvin = DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).unwrap();
    let per_second = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let rate_dim = kelvin.mul(per_second).unwrap();
    let continuous = Id::<kinds::Activation>::new();
    let flow = Id::<kinds::Relation>::new();
    let mut b = ExprDagBuilder::new();
    let dt = b
        .symbol(SymbolRef::Derivative(
            temperature,
            std::num::NonZeroU32::MIN,
        ))
        .unwrap();
    let r = b.symbol(SymbolRef::Field(rate)).unwrap();
    let dr = b
        .symbol(SymbolRef::Derivative(rate, std::num::NonZeroU32::MIN))
        .unwrap();
    let zero = b
        .constant(DynQuantity::new(0., rate_dim.mul(per_second).unwrap()))
        .unwrap();
    let mut nodes = vec![
        FieldDef::new(
            temperature,
            ValueType::scalar(eqiora_core::ScalarDomain::Real, kelvin).expect("valid scalar type"),
            FieldRole::State,
        )
        .into(),
        FieldDef::new(
            rate,
            ValueType::scalar(eqiora_core::ScalarDomain::Real, rate_dim)
                .expect("valid scalar type"),
            FieldRole::State,
        )
        .into(),
        initial(temperature, kelvin, 20.),
        initial(rate, rate_dim, 1.),
        ActivationDef::continuous(continuous).into(),
        RelationDef::new(flow, b.finish([dt, r, dr, zero]).unwrap())
            .unwrap()
            .into(),
    ];
    let mut activations = vec![(continuous.erase(), flow.erase())];
    for (threshold, direction) in [
        (22., EventDirection::Rising),
        (18., EventDirection::Falling),
    ] {
        let event = Id::<kinds::Activation>::new();
        let reset = Id::<kinds::Relation>::new();
        let mut b = ExprDagBuilder::new();
        let t = b.symbol(SymbolRef::Field(temperature)).unwrap();
        let threshold = b.constant(DynQuantity::new(threshold, kelvin)).unwrap();
        let guard = b.sub(t, threshold).unwrap();
        nodes.push(
            ActivationDef::new(
                event,
                ActivationKind::Event {
                    priority: 0,
                    guard: b.finish([guard]).unwrap(),
                    direction,
                },
            )
            .unwrap()
            .into(),
        );
        let mut b = ExprDagBuilder::new();
        let next = b.symbol(SymbolRef::Next(rate)).unwrap();
        let pre = b.symbol(SymbolRef::Pre(rate)).unwrap();
        let sum = b.add(next, pre).unwrap();
        let zero = b.constant(DynQuantity::new(0., rate_dim)).unwrap();
        nodes.push(
            RelationDef::new(reset, b.finish([sum, zero]).unwrap())
                .unwrap()
                .into(),
        );
        activations.push((event.erase(), reset.erase()));
    }
    let program = program(nodes, &activations);
    for max_step in [0.25, 0.01] {
        let config = ReferenceConfig::new(12., max_step)
            .unwrap()
            .with_nonlinear_tolerances(1e-12, 0.)
            .unwrap()
            .with_event_tolerances(1e-11, 1e-10)
            .unwrap();
        let trajectory = Interpreter::new().run(&program, config).unwrap();
        // Piecewise constant rates integrate exactly: heating reaches22 at2s,
        // cooling reaches18 at6s, reheating reaches22 at10s, cooling returns20 at12s.
        assert!((trajectory.last_value(temperature.erase()).unwrap().value() - 20.).abs() < 1e-8);
        assert_eq!(trajectory.last_value(rate.erase()).unwrap().value(), -1.);
        for (field, expected) in [
            (
                temperature,
                [(2., 22., 22.), (6., 18., 18.), (10., 22., 22.)],
            ),
            (rate, [(2., 1., -1.), (6., -1., 1.), (10., 1., -1.)]),
        ] {
            let samples = trajectory
                .samples()
                .iter()
                .filter(|s| s.coordinate() == field.erase())
                .collect::<Vec<_>>();
            let resets = samples
                .windows(2)
                .filter(|p| p[0].time() == p[1].time())
                .collect::<Vec<_>>();
            assert_eq!(
                resets.len(),
                3,
                "reset onto the guard must not immediately retrigger"
            );
            for (pair, (time, before, after)) in resets.iter().zip(expected) {
                assert!((pair[0].time() - time).abs() < 2e-9);
                assert!((pair[0].value().value() - before).abs() < 2e-9);
                assert!((pair[1].value().value() - after).abs() < 2e-9);
            }
        }
    }
}

#[test]
fn grouped_reset_and_post_reset_consistency_failures_leave_accepted_state_unchanged() {
    for conflicting_reset in [false, true] {
        let state = Id::<kinds::Field>::new();
        let algebraic = Id::<kinds::Field>::new();
        let companion = Id::<kinds::Field>::new();
        let continuous = Id::<kinds::Activation>::new();
        let event = Id::<kinds::Activation>::new();
        let flow = Id::<kinds::Relation>::new();
        let reset = Id::<kinds::Relation>::new();
        let d = DimExponents::DIMENSIONLESS;
        let mut b = ExprDagBuilder::new();
        let x = b.symbol(SymbolRef::Field(state)).unwrap();
        let y = b.symbol(SymbolRef::Field(algebraic)).unwrap();
        let product = b.mul(x, y).unwrap();
        let one = b.constant(DynQuantity::new(1., d)).unwrap();
        let dx = b
            .symbol(SymbolRef::Derivative(state, std::num::NonZeroU32::MIN))
            .unwrap();
        let dc = b
            .symbol(SymbolRef::Derivative(companion, std::num::NonZeroU32::MIN))
            .unwrap();
        let zero = b
            .constant(DynQuantity::new(
                0.,
                DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap(),
            ))
            .unwrap();
        let mut nodes = vec![
            FieldDef::new(
                state,
                ValueType::scalar(eqiora_core::ScalarDomain::Real, d).expect("valid scalar type"),
                FieldRole::State,
            )
            .into(),
            FieldDef::new(
                algebraic,
                ValueType::scalar(eqiora_core::ScalarDomain::Real, d).expect("valid scalar type"),
                FieldRole::Variable,
            )
            .into(),
            initial(state, d, 1.),
            FieldDef::new(
                companion,
                ValueType::scalar(eqiora_core::ScalarDomain::Real, d).expect("valid scalar type"),
                FieldRole::State,
            )
            .into(),
            initial(companion, d, 7.),
            ActivationDef::continuous(continuous).into(),
            RelationDef::new(flow, b.finish([dx, zero, dc, zero, product, one]).unwrap())
                .unwrap()
                .into(),
        ];
        let mut b = ExprDagBuilder::new();
        let x = b.symbol(SymbolRef::Field(state)).unwrap();
        nodes.push(
            ActivationDef::new(
                event,
                ActivationKind::Event {
                    priority: 0,
                    guard: b.finish([x]).unwrap(),
                    direction: EventDirection::Falling,
                },
            )
            .unwrap()
            .into(),
        );
        let mut b = ExprDagBuilder::new();
        let next = b.symbol(SymbolRef::Next(state)).unwrap();
        let zero = b.constant(DynQuantity::new(0., d)).unwrap();
        nodes.push(
            RelationDef::new(reset, b.finish([next, zero]).unwrap())
                .unwrap()
                .into(),
        );
        let mut activations = vec![
            (continuous.erase(), flow.erase()),
            (event.erase(), reset.erase()),
        ];
        if conflicting_reset {
            let conflict = Id::<kinds::Relation>::new();
            let mut b = ExprDagBuilder::new();
            let next = b.symbol(SymbolRef::Next(state)).unwrap();
            let other = b.symbol(SymbolRef::Next(companion)).unwrap();
            let zero = b.constant(DynQuantity::new(0., d)).unwrap();
            let cancelled = b.mul(zero, other).unwrap();
            let conflicting = b.add(next, cancelled).unwrap();
            let two = b.constant(DynQuantity::new(2., d)).unwrap();
            nodes.push(
                RelationDef::new(conflict, b.finish([conflicting, two]).unwrap())
                    .unwrap()
                    .into(),
            );
            activations.push((event.erase(), conflict.erase()));
        }
        let program = program(nodes, &activations);
        let plan = ExecutionPlan::new(&program).unwrap();
        let mut accepted = RuntimeState::new(&program, &plan).unwrap();
        // The bilinear consistency law needs a nonsingular numerical seed.
        let config = ReferenceConfig::new(1., 0.1)
            .unwrap()
            .with_initial_guess(1.)
            .unwrap();
        solve_initialization(
            &program,
            &plan,
            &mut accepted,
            0.0,
            config,
            &ReferenceExpressionBackend,
        )
        .unwrap();
        let before = accepted.clone();
        let error = execute_activated_relations(
            &program,
            &plan,
            &mut accepted,
            0.,
            &plan.events[0].relations,
            "event-activation",
            config,
            &ReferenceExpressionBackend,
        )
        .unwrap_err();
        assert_eq!(error.code(), codes::NONLINEAR_SOLVE_FAILED);
        assert_eq!(accepted.fields, before.fields);
        assert_eq!(accepted.derivatives, before.derivatives);
        assert_eq!(accepted.ports, before.ports);
        assert_eq!(accepted.typed_fields, before.typed_fields);
        assert_eq!(accepted.typed_ports, before.typed_ports);
        assert_eq!(accepted.typed_next, before.typed_next);
        assert_eq!(accepted.physical, before.physical);
    }
}

#[test]
fn reference_config_rejects_non_advancing_steps() {
    let diagnostic = ReferenceConfig::new(1.0, 0.0).expect_err("zero step");

    assert_eq!(diagnostic.code(), codes::INVALID_EXECUTION_CONFIG);
}

#[test]
fn short_terminal_step_retains_the_solved_derivative_without_subtractive_cancellation() {
    let field = Id::<kinds::Field>::new();
    let continuous = Id::<kinds::Activation>::new();
    let flow = Id::<kinds::Relation>::new();
    let d = DimExponents::DIMENSIONLESS;
    let per_second = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let mut b = ExprDagBuilder::new();
    let derivative = b
        .symbol(SymbolRef::Derivative(field, std::num::NonZeroU32::MIN))
        .unwrap();
    let one = b.constant(DynQuantity::new(1., per_second)).unwrap();
    let program = program(
        vec![
            FieldDef::new(
                field,
                ValueType::scalar(eqiora_core::ScalarDomain::Real, d).expect("valid scalar type"),
                FieldRole::State,
            )
            .into(),
            initial(field, d, 20.),
            ActivationDef::continuous(continuous).into(),
            RelationDef::new(flow, b.finish([derivative, one]).unwrap())
                .unwrap()
                .into(),
        ],
        &[(continuous.erase(), flow.erase())],
    );
    let plan = ExecutionPlan::new(&program).unwrap();
    let mut state = RuntimeState::new(&program, &plan).unwrap();
    let config = ReferenceConfig::new(2., 1.)
        .unwrap()
        .with_nonlinear_tolerances(1e-12, 0.)
        .unwrap();
    solve_initialization(
        &program,
        &plan,
        &mut state,
        0.0,
        config,
        &ReferenceExpressionBackend,
    )
    .unwrap();
    let start = 1.;
    let end = 1. + 3e-10;
    // x'=1 is exact. Reconstructing x' from rounded x differences instead
    // has a quantization floor near 6e-6 at this point and interval.
    solve_continuous_step(
        &program,
        &plan,
        &mut state,
        start,
        end,
        config,
        &ReferenceExpressionBackend,
    )
    .unwrap();
    assert_eq!(state.fields[&field.erase()], 20. + (end - start));
    assert_eq!(
        state.derivatives[&(field.erase(), std::num::NonZeroU32::MIN)],
        1.
    );

    // Derived fields remain subject to finite-value admission even when the
    // authored flow reads only the derivative, not the candidate field.
    state.fields.insert(field.erase(), f64::MAX);
    let before = state.clone();
    let error = solve_continuous_step(
        &program,
        &plan,
        &mut state,
        0.,
        f64::MAX,
        config,
        &ReferenceExpressionBackend,
    )
    .unwrap_err();
    assert!(error.message().contains("candidate Field is not finite"));
    assert_eq!(state.fields, before.fields);
    assert_eq!(state.derivatives, before.derivatives);
}

#[test]
fn slow_crossing_keeps_arming_through_the_zero_band_and_checkpoint() {
    let field = Id::<kinds::Field>::new();
    let continuous = Id::<kinds::Activation>::new();
    let event = Id::<kinds::Activation>::new();
    let flow = Id::<kinds::Relation>::new();
    let reset = Id::<kinds::Relation>::new();
    let d = DimExponents::DIMENSIONLESS;
    let rate = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let mut b = ExprDagBuilder::new();
    let derivative = b
        .symbol(SymbolRef::Derivative(field, std::num::NonZeroU32::MIN))
        .unwrap();
    let one = b.constant(DynQuantity::new(1., rate)).unwrap();
    let mut nodes = vec![
        FieldDef::new(
            field,
            ValueType::scalar(eqiora_core::ScalarDomain::Real, d).expect("valid scalar type"),
            FieldRole::State,
        )
        .into(),
        initial(field, d, -2.),
        ActivationDef::continuous(continuous).into(),
        RelationDef::new(flow, b.finish([derivative, one]).unwrap())
            .unwrap()
            .into(),
    ];
    let mut b = ExprDagBuilder::new();
    let x = b.symbol(SymbolRef::Field(field)).unwrap();
    nodes.push(
        ActivationDef::new(
            event,
            ActivationKind::Event {
                priority: 0,
                guard: b.finish([x]).unwrap(),
                direction: EventDirection::Rising,
            },
        )
        .unwrap()
        .into(),
    );
    let mut b = ExprDagBuilder::new();
    let next = b.symbol(SymbolRef::Next(field)).unwrap();
    let one = b.constant(DynQuantity::new(1., d)).unwrap();
    nodes.push(
        RelationDef::new(reset, b.finish([next, one]).unwrap())
            .unwrap()
            .into(),
    );
    let program = program(
        nodes,
        &[
            (continuous.erase(), flow.erase()),
            (event.erase(), reset.erase()),
        ],
    );
    let config = ReferenceConfig::new(2.25, 0.25)
        .unwrap()
        .with_event_tolerances(1e-12, 1.)
        .unwrap();
    let interpreter = Interpreter::new();
    let mut session = interpreter.execution_session(&program, config, []).unwrap();
    assert!(session.activation_sequence().is_empty());
    while session.progress().model_time() < 1.25 {
        assert!(session.advance().unwrap());
        assert!(session.activation_sequence().is_empty());
    }
    // x=-0.75 is inside the arming band, but its earlier negative-side
    // arming must survive. The actual zero is still at t=2, not t=1.
    assert_eq!(
        session
            .field(field.erase())
            .unwrap()
            .real_scalar_value()
            .unwrap()
            .value(),
        -0.75
    );
    let mut resumed = interpreter
        .resume_execution(&program, &session.checkpoint())
        .unwrap();
    let mut activations = Vec::new();
    while session.advance().unwrap() {
        assert!(resumed.advance().unwrap());
        assert_eq!(session.progress(), resumed.progress());
        assert_eq!(session.activation_sequence(), resumed.activation_sequence());
        assert_eq!(session.field(field.erase()), resumed.field(field.erase()));
        if !session.activation_sequence().is_empty() {
            activations.push((
                session.progress().model_time(),
                session.activation_sequence().to_vec(),
            ));
        }
    }
    assert!(!resumed.advance().unwrap());
    assert_eq!(activations, vec![(2., vec![vec![event.erase()]])]);
    assert_eq!(
        session
            .field(field.erase())
            .unwrap()
            .real_scalar_value()
            .unwrap()
            .value(),
        1.25
    );
}

#[test]
fn a_clock_only_group_does_not_consume_the_event_microstep_limit() {
    use eqiora_schema::kernel::ClockDomainDef;
    for cascade in [false, true] {
        let memory = Id::<kinds::Field>::new();
        let state = Id::<kinds::Field>::new();
        let clock = Id::<kinds::ClockDomain>::new();
        let tick = Id::<kinds::Activation>::new();
        let continuous = Id::<kinds::Activation>::new();
        let first = Id::<kinds::Activation>::new();
        let second = Id::<kinds::Activation>::new();
        let update = Id::<kinds::Relation>::new();
        let flow = Id::<kinds::Relation>::new();
        let reset = Id::<kinds::Relation>::new();
        let d = DimExponents::DIMENSIONLESS;
        let ty = ValueType::scalar(eqiora_core::ScalarDomain::Real, d).expect("valid scalar type");
        let mut b = ExprDagBuilder::new();
        let next = b.symbol(SymbolRef::Next(memory)).unwrap();
        let previous = b.symbol(SymbolRef::Pre(memory)).unwrap();
        let one = b.neg(previous).unwrap();
        let mut nodes = vec![
            FieldDef::new(memory, ty.clone(), FieldRole::State).into(),
            FieldDef::new(state, ty, FieldRole::State).into(),
            initial(memory, d, -1.),
            initial(state, d, -1.),
            ClockDomainDef::periodic(clock, RationalTime::new(1, 1).unwrap(), RationalTime::ZERO)
                .unwrap()
                .into(),
            ActivationDef::periodic(tick).into(),
            ActivationDef::continuous(continuous).into(),
            RelationDef::new(update, b.finish([next, one]).unwrap())
                .unwrap()
                .into(),
        ];
        let mut b = ExprDagBuilder::new();
        let derivative = b
            .symbol(SymbolRef::Derivative(state, std::num::NonZeroU32::MIN))
            .unwrap();
        let zero = b
            .constant(DynQuantity::new(
                0.,
                DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap(),
            ))
            .unwrap();
        nodes.push(
            RelationDef::new(flow, b.finish([derivative, zero]).unwrap())
                .unwrap()
                .into(),
        );
        let mut b = ExprDagBuilder::new();
        let memory_value = b.symbol(SymbolRef::Field(memory)).unwrap();
        let guard = b.hold(memory_value).unwrap();
        nodes.push(
            ActivationDef::new(
                first,
                ActivationKind::Event {
                    priority: 0,
                    guard: b.finish([guard]).unwrap(),
                    direction: EventDirection::Rising,
                },
            )
            .unwrap()
            .into(),
        );
        let mut b = ExprDagBuilder::new();
        let next = b.symbol(SymbolRef::Next(state)).unwrap();
        let one = b.constant(DynQuantity::new(1., d)).unwrap();
        nodes.push(
            RelationDef::new(reset, b.finish([next, one]).unwrap())
                .unwrap()
                .into(),
        );
        let mut activations = vec![
            (tick.erase(), update.erase()),
            (continuous.erase(), flow.erase()),
            (first.erase(), reset.erase()),
        ];
        if cascade {
            let reset = Id::<kinds::Relation>::new();
            let mut b = ExprDagBuilder::new();
            let guard = b.symbol(SymbolRef::Field(state)).unwrap();
            nodes.push(
                ActivationDef::new(
                    second,
                    ActivationKind::Event {
                        priority: 0,
                        guard: b.finish([guard]).unwrap(),
                        direction: EventDirection::Rising,
                    },
                )
                .unwrap()
                .into(),
            );
            let mut b = ExprDagBuilder::new();
            let next = b.symbol(SymbolRef::Next(state)).unwrap();
            let two = b.constant(DynQuantity::new(2., d)).unwrap();
            nodes.push(
                RelationDef::new(reset, b.finish([next, two]).unwrap())
                    .unwrap()
                    .into(),
            );
            activations.push((second.erase(), reset.erase()));
        }
        let program = program_with_edges(
            nodes,
            &activations,
            &[
                (memory.erase(), clock.erase(), EdgeKind::ClockedBy),
                (tick.erase(), clock.erase(), EdgeKind::ClockedBy),
            ],
        );
        let config = ReferenceConfig::new(2., 1.)
            .unwrap()
            .with_event_limits(80, 1)
            .unwrap();
        let mut session = Interpreter::new()
            .execution_session(&program, config, [])
            .unwrap();
        let before = session.checkpoint();
        if cascade {
            let errors = session.advance().unwrap_err();
            assert_eq!(errors[0].code(), codes::INVALID_EXECUTION_CONFIG);
            assert!(errors[0].message().contains(&first.erase().to_string()));
            assert!(errors[0].message().contains(&second.erase().to_string()));
            assert_eq!(session.progress(), before.progress());
            assert_eq!(session.next_tick(), before.next_tick());
            assert_eq!(session.activation_sequence(), before.activation_sequence());
            assert_eq!(session.state.fields, before.state.fields);
            assert_eq!(session.state.derivatives, before.state.derivatives);
            assert_eq!(session.field(memory.erase()), before.field(memory.erase()));
        } else {
            assert!(session.advance().unwrap());
            assert_eq!(
                session.activation_sequence(),
                [vec![tick.erase()], vec![first.erase()]]
            );
            assert_eq!(
                session
                    .field(state.erase())
                    .unwrap()
                    .real_scalar_value()
                    .unwrap()
                    .value(),
                1.
            );
            assert_eq!(
                session
                    .field(memory.erase())
                    .unwrap()
                    .real_scalar_value()
                    .unwrap()
                    .value(),
                1.
            );
            // A later tick alone does not spend an event commit; the next
            // rising crossing starts a new physical-time event count.
            assert!(session.advance().unwrap());
            assert_eq!(session.progress().model_time(), 1.);
            assert_eq!(session.activation_sequence(), [vec![tick.erase()]]);
            assert!(session.advance().unwrap());
            assert_eq!(session.progress().model_time(), 2.);
            assert_eq!(
                session.activation_sequence(),
                [vec![tick.erase()], vec![first.erase()]]
            );
            assert!(!session.advance().unwrap());
        }
    }
}

#[test]
fn enum_state_selects_live_numeric_flow_and_resets_without_numeric_storage() {
    use eqiora_schema::kernel::{ComparisonOp, EnumDef};
    let definition = EnumDef::new(Id::new(), ["Slow".into(), "Fast".into()]).unwrap();
    let mode = Id::<kinds::Field>::new();
    let position = Id::<kinds::Field>::new();
    let flow = Id::<kinds::Relation>::new();
    let reset = Id::<kinds::Relation>::new();
    let initial_mode = Id::<kinds::Relation>::new();
    let continuous = Id::<kinds::Activation>::new();
    let event = Id::<kinds::Activation>::new();
    let d = DimExponents::DIMENSIONLESS;
    let rate = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let mut b = ExprDagBuilder::new();
    let target = b.symbol(SymbolRef::Field(mode)).unwrap();
    let slow = b.constant(definition.value(0).unwrap()).unwrap();
    let mode_initial =
        RelationDef::initial(initial_mode, b.finish([target, slow]).unwrap()).unwrap();
    let mut nodes = vec![
        definition.clone().into(),
        FieldDef::new(mode, definition.value_type(), FieldRole::State).into(),
        FieldDef::new(
            position,
            ValueType::scalar(eqiora_core::ScalarDomain::Real, d).expect("valid scalar type"),
            FieldRole::State,
        )
        .into(),
        initial(position, d, 0.),
        ActivationDef::continuous(continuous).into(),
    ];
    let mut b = ExprDagBuilder::new();
    let mode_read = b.symbol(SymbolRef::Field(mode)).unwrap();
    let slow = b.constant(definition.value(0).unwrap()).unwrap();
    let condition = b.compare(ComparisonOp::Equal, mode_read, slow).unwrap();
    let one = b.constant(DynQuantity::new(1., rate)).unwrap();
    let two = b.constant(DynQuantity::new(2., rate)).unwrap();
    let selected = b.select(condition, one, two).unwrap();
    let derivative = b
        .symbol(SymbolRef::Derivative(position, std::num::NonZeroU32::MIN))
        .unwrap();
    nodes.push(
        RelationDef::new(flow, b.finish([derivative, selected]).unwrap())
            .unwrap()
            .into(),
    );
    let mut b = ExprDagBuilder::new();
    let x = b.symbol(SymbolRef::Field(position)).unwrap();
    let threshold = b.constant(DynQuantity::new(1., d)).unwrap();
    let guard = b.sub(x, threshold).unwrap();
    nodes.push(
        ActivationDef::new(
            event,
            ActivationKind::Event {
                priority: 0,
                guard: b.finish([guard]).unwrap(),
                direction: EventDirection::Rising,
            },
        )
        .unwrap()
        .into(),
    );
    let mut b = ExprDagBuilder::new();
    let previous = b.symbol(SymbolRef::Pre(mode)).unwrap();
    let slow = b.constant(definition.value(0).unwrap()).unwrap();
    let fast = b.constant(definition.value(1).unwrap()).unwrap();
    let condition = b.compare(ComparisonOp::Equal, previous, slow).unwrap();
    let selected = b.select(condition, fast, previous).unwrap();
    let next = b.symbol(SymbolRef::Next(mode)).unwrap();
    nodes.push(
        RelationDef::new(reset, b.finish([next, selected]).unwrap())
            .unwrap()
            .into(),
    );
    let edges = [
        (continuous.erase(), flow.erase()),
        (event.erase(), reset.erase()),
    ];
    let missing = program(nodes.clone(), &edges);
    let config = ReferenceConfig::new(1.5, 0.5).unwrap();
    assert!(
        Interpreter::new()
            .execution_session(&missing, config, [])
            .unwrap_err()[0]
            .message()
            .contains("explicit initial")
    );
    nodes.push(mode_initial.into());
    let program = program(nodes, &edges);
    let interpreter = Interpreter::new();
    let mut session = interpreter.execution_session(&program, config, []).unwrap();
    assert_eq!(session.field(mode.erase()).unwrap().enum_tag(), Some(0));
    assert!(!session.state.fields.contains_key(&mode.erase()));
    assert!(session.advance().unwrap());
    let checkpoint = session.checkpoint();
    assert!(session.advance().unwrap());
    assert_eq!(session.activation_sequence(), [vec![event.erase()]]);
    assert_eq!(
        session.field(mode.erase()).unwrap(),
        definition.value(1).unwrap()
    );
    let mut resumed = interpreter.resume_execution(&program, &checkpoint).unwrap();
    assert!(resumed.advance().unwrap());
    assert_eq!(resumed.field(mode.erase()), session.field(mode.erase()));
    assert!(session.advance().unwrap());
    assert!(resumed.advance().unwrap());
    // x'=1 to t1, then x'=2 for half a second: x(1.5)=2 exactly.
    assert_eq!(
        session
            .field(position.erase())
            .unwrap()
            .real_scalar_value()
            .unwrap()
            .value(),
        2.
    );
    assert_eq!(
        resumed.field(position.erase()),
        session.field(position.erase())
    );
    assert_eq!(
        session.field(mode.erase()).unwrap().real_scalar_value(),
        None
    );
    assert!(
        interpreter.run(&program, config).unwrap_err()[0]
            .message()
            .contains("exact discrete Fields")
    );
}
