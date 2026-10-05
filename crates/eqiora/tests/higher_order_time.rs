//! A higher-order source state keeps its identity and both initial conditions.
use eqiora::compiler::compile;
use eqiora::graph::Op;
use eqiora::kernel::{ExprNode, FieldRole, KernelNode, SymbolRef};

const OSCILLATOR: &str = r#"
model Oscillator() {
    parameter mass:kg=1;
    parameter stiffness:N/m=4;
    state displacement:m;
    initial {
        displacement=1[m];
        derivative(displacement)=0[m/s];
    }
    relation motion {
        mass*derivative(derivative(displacement))+stiffness*displacement=0[N];
    }
    observable elapsed:s=time();
}
"#;

#[test]
fn implicit_initialization_uses_the_declared_nonzero_time_for_values_and_regularity() {
    use eqiora::graph::{GraphStore, InMemoryGraphStore};
    use eqiora::runtime::{CpuProgram, GeneralImplicitProgram};
    use eqiora::sem::{KernelProgram, ReferenceConfig};
    let source = "model Timed() { state q:m; initial { q=time()*1[m/s]; derivative(q)=3[m/s]; } relation flow { time()*derivative(derivative(q))=1[m/s]; } }";
    let model = compile("timed.eqi", source).unwrap().pop().unwrap();
    let (transaction, model_id, symbols) = model.into_parts();
    let relation = symbols.get("flow").unwrap().downcast().unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
    let cpu = CpuProgram::lower(&kernel).unwrap();
    let system = GeneralImplicitProgram::lower(&cpu, relation).unwrap();
    let config = ReferenceConfig::new(0., 1.).unwrap();
    let initial = system.initialize(2., config).unwrap();
    assert_eq!(initial.state(), &[2., 3.]);
    assert_eq!(initial.derivative(), &[3., 0.5]);
    // At zero the authored equation is inconsistent; a hidden zero-time check
    // would incorrectly reject the perfectly regular initialization at t=2.
    assert!(system.initialize(0., config).is_err());
}

#[test]
fn continuous_observable_admits_a_total_time_derivative() {
    let source = r#"model Motion() {
        state displacement:m;
        initial { displacement=1[m]; }
        relation motion { derivative(displacement)=2[m/s]; }
        observable squared_displacement_rate:m^2/s=derivative(displacement*displacement);
    }"#;
    compile("motion.eqi", source).unwrap();
}

#[test]
fn explicit_first_order_oscillator_establishes_the_existing_execution_owner() {
    use eqiora::graph::{GraphStore, InMemoryGraphStore};
    use eqiora::runtime::{CpuProgram, FirstOrderProgram, GeneralImplicitProgram};
    use eqiora::sem::{KernelProgram, ReferenceConfig};
    use eqiora::time::{
        DaeVariableKind, ImplicitDaeProblem, ImplicitTimeSystem, InitialConditionPolicy,
        ReferenceImplicitTimeBackend, TimeMethod, TimePlan, TimeSystem,
    };

    for mass in ["1[kg]", "mass"] {
        let source = format!(
            "model Oscillator() {{
                parameter mass:kg=1;
                parameter stiffness:N/m=4;
                state displacement:m;
                state velocity:m/s;
                initial {{ displacement=1[m]; velocity=0[m/s]; }}
                relation motion {{
                    derivative(displacement)=velocity;
                    {mass}*derivative(velocity)+stiffness*displacement=0[N];
                }}
            }}"
        );
        let model = compile("first_order.eqi", &source).unwrap().pop().unwrap();
        let (transaction, model_id, symbols) = model.into_parts();
        let relation = symbols.get("motion").unwrap().downcast().unwrap();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let kernel = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
        let cpu = CpuProgram::lower(&kernel).unwrap();
        // Literal mass is a constant IR coefficient. A retained Parameter is
        // differentiated by the existing residual-native owner, even when its
        // captured value happens to be one; classification must not sample it.
        if mass == "1[kg]" {
            let system = FirstOrderProgram::lower(&cpu, relation).unwrap();
            let initial = system
                .initialize(0.0, ReferenceConfig::new(0., 1.).unwrap())
                .unwrap();
            assert_eq!(initial.state(), &[1., 0.]);
            let mut rhs = [f64::NAN; 2];
            system.rhs(0., initial.state(), &mut rhs).unwrap();
            assert_eq!(rhs, [0., -4.]);
        } else {
            let system = GeneralImplicitProgram::lower(&cpu, relation).unwrap();
            let mut residual = [f64::NAN; 2];
            system
                .residual(0., &[1., 0.], &[0., -4.], &mut residual)
                .unwrap();
            assert_eq!(residual, [0., 0.]);
            let initial = system
                .initialize(0.0, ReferenceConfig::new(0., 1.).unwrap())
                .unwrap();
            assert_eq!(initial.state(), &[1., 0.]);
            assert_eq!(initial.derivative(), &[0., -4.]);
            let problem = ImplicitDaeProblem::new(
                &system,
                vec![DaeVariableKind::Differential; 2],
                InitialConditionPolicy::Provided,
                initial.state().to_vec(),
                initial.derivative().to_vec(),
            )
            .unwrap();
            let plan = TimePlan::new(
                TimeMethod::ImplicitEuler,
                0.,
                0.001,
                1e-11,
                vec![1e-13; 2],
                vec![0.25, 0.5, 1.],
            )
            .unwrap();
            let solution = ReferenceImplicitTimeBackend::new()
                .solve(&problem, &plan)
                .unwrap();
            for (sample, time) in [0.25_f64, 0.5, 1.].into_iter().enumerate() {
                let state = solution.state(sample).unwrap();
                // In (x,v/2) coordinates A is skew symmetric with norm 2.
                // Implicit Euler is contractive and its one-step defect is at
                // most 2*h^2, giving a global norm error <= 2*T*h.
                // These bounds leave 0.001 for the declared nonlinear solve.
                assert!((state[0] - (2. * time).cos()).abs() < 0.003);
                assert!((state[1] + 2. * (2. * time).sin()).abs() < 0.006);
            }
        }
    }
}

#[test]
fn oscillator_source_retains_one_authored_state_and_paired_initial_conditions() {
    // x''+4x=0 with x(0)=1 and x'(0)=0 has x=cos(2t), v=-2sin(2t).
    // The source owns x; its first-order velocity coordinate belongs to Formulation.
    let models = compile("oscillator.eqi", OSCILLATOR).unwrap();
    let operations = models[0].transaction().ops();
    let states = operations
        .iter()
        .filter(|operation| {
            matches!(
                operation,
                Op::DefineKernelNode { node: KernelNode::Field(field) }
                    if field.role() == FieldRole::State
            )
        })
        .count();
    assert_eq!(
        states, 1,
        "a numerical auxiliary must not become an authored State"
    );
    let initial_sides: usize = operations
        .iter()
        .filter_map(|operation| match operation {
            Op::DefineKernelNode {
                node: KernelNode::Relation(relation),
            } if relation.is_initial() => Some(relation.expression().roots().len()),
            _ => None,
        })
        .sum();
    assert_eq!(
        initial_sides, 4,
        "retain both sides of displacement and velocity initialization"
    );
}

#[test]
fn derivative_order_and_units_survive_model_replay_without_a_second_order_ceiling() {
    use eqiora::artifact::ModelEnvelope;
    use eqiora::graph::{GraphStore, InMemoryGraphStore};
    use eqiora::sem::KernelProgram;

    for order in [1_u32, 2, 3, 5, 6, 9, 16] {
        let mut expression = "q".to_owned();
        for _ in 0..order {
            expression = format!("derivative({expression})");
        }
        let source =
            format!("model M() {{ state q:m; relation r {{ {expression}=1[m/s^{order}]; }} }}");
        let model = compile("order.eqi", &source).unwrap().pop().unwrap();
        let (transaction, model_id, symbols) = model.into_parts();
        let field = symbols.get("q").unwrap().downcast().unwrap();
        let expected = SymbolRef::Derivative(field, std::num::NonZeroU32::new(order).unwrap());
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let kernel = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
        let envelope = ModelEnvelope::from_program(&kernel).unwrap();
        let replay = envelope.to_program().unwrap();
        assert!(replay.nodes().any(|node| {
            match node {
                KernelNode::Relation(relation) => relation
                    .expression()
                    .nodes()
                    .iter()
                    .any(|node| matches!(node, ExprNode::Symbol(symbol) if *symbol == expected)),
                _ => false,
            }
        }));
    }
}

#[test]
fn differentiating_a_product_with_a_rate_retains_both_chain_terms() {
    for (expression, expected, order) in [
        ("derivative(q*derivative(q))", 19., 2),
        ("derivative(derivative(q*q))", 38., 2),
        ("derivative(derivative(aliased))", 38., 2),
        ("derivative(derivative(square(x=q)))", 38., 2),
        ("derivative(derivative(derivative(q*q)))", 102., 3),
        ("derivative(derivative(q*derivative(q)))", 51., 3),
    ] {
        let source = format!(
            "operator square(input x:m):m^2=x*x; model M() {{ state q:m; let aliased=q*q; relation r {{ {expression}={expected}[m^2/s^{order}]; }} }}"
        );
        let model = compile("rate_product.eqi", &source).unwrap().pop().unwrap();
        let relation = model
            .transaction()
            .ops()
            .iter()
            .find_map(|operation| match operation {
                Op::DefineKernelNode {
                    node: KernelNode::Relation(relation),
                } => Some(relation),
                _ => None,
            })
            .unwrap();
        let ir = eqiora::ir::ScalarOperatorIr::lower(relation.expression()).unwrap();
        let values = ir
            .evaluate_typed(relation.expression().roots(), &mut |symbol| {
                let (value, time_power) = match symbol {
                    SymbolRef::Field(_) => (3., 0),
                    SymbolRef::Derivative(_, order) if order.get() == 1 => (2., -1),
                    SymbolRef::Derivative(_, order) if order.get() == 2 => (5., -2),
                    SymbolRef::Derivative(_, order) if order.get() == 3 => (7., -3),
                    _ => panic!("unexpected rate-product input {symbol:?}"),
                };
                let dimension =
                    eqiora::DimExponents::from_integers([0, 1, time_power, 0, 0, 0, 0]).unwrap();
                eqiora::ValueLiteral::try_from(eqiora::DynQuantity::new(value, dimension)).ok()
            })
            .unwrap();
        // q=3, q'=2, q''=5, q'''=7. Product rule gives 19, 38, 102, 51 respectively.
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], values[1]);
        assert_eq!(values[0].real_scalar_value().unwrap().value(), expected);
    }
}

#[test]
fn higher_order_initialization_retains_velocity_and_solves_acceleration() {
    use eqiora::graph::{GraphStore, InMemoryGraphStore};
    use eqiora::sem::{Interpreter, KernelProgram, ReferenceConfig};

    for velocity in [Some(0.), Some(2.), None] {
        let source = OSCILLATOR.replace(
            "derivative(displacement)=0[m/s];",
            &velocity.map_or_else(String::new, |value| {
                format!("derivative(displacement)={value}[m/s];")
            }),
        );
        let model = compile("initial_velocity.eqi", &source)
            .unwrap()
            .pop()
            .unwrap();
        let (transaction, model_id, symbols) = model.into_parts();
        let field = symbols.get("displacement").unwrap();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let kernel = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
        let result =
            Interpreter::new().initialize(&kernel, 0.0, ReferenceConfig::new(0., 1.).unwrap());
        if let Some(velocity) = velocity {
            let initial = result.unwrap();
            assert_eq!(
                initial.fields()[&field]
                    .real_scalar_value()
                    .unwrap()
                    .value(),
                1.
            );
            assert_eq!(
                initial.derivatives()[&(field, std::num::NonZeroU32::MIN)],
                velocity
            );
            assert_eq!(
                initial.derivatives()[&(field, std::num::NonZeroU32::new(2).unwrap())],
                -4.
            );
        } else {
            let diagnostics = result.expect_err("omitted velocity must not become a hidden zero");
            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message().contains("2 equations and 3 unknowns")),
                "{diagnostics:?}"
            );
        }
    }
}

#[test]
fn reference_time_steps_advance_every_derivative_coordinate() {
    use eqiora::graph::{GraphStore, InMemoryGraphStore};
    use eqiora::sem::{Interpreter, KernelProgram, ReferenceConfig};

    let third_order = "model Polynomial() {
        state displacement:m;
        initial {
            displacement=0[m];
            derivative(displacement)=0[m/s];
            derivative(derivative(displacement))=0[m/s^2];
        }
        relation motion { derivative(derivative(derivative(displacement)))=6[m/s^3]; }
    }";
    for (source, end, step, expected, tolerance) in [
        (OSCILLATOR, 0.25, 0.001, (0.5_f64).cos(), 0.0006),
        // Backward Euler gives a_n=6*n*h, v_n=3*n*(n+1)*h²,
        // x_n=n*(n+1)*(n+2)*h³. At n=10,h=0.1 this is 1.32.
        (third_order, 1., 0.1, 1.32, 1e-9),
    ] {
        let model = compile("evolution.eqi", source).unwrap().pop().unwrap();
        let (transaction, model_id, symbols) = model.into_parts();
        let field = symbols.get("displacement").unwrap();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let kernel = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
        let config = ReferenceConfig::new(end, step)
            .unwrap()
            .with_nonlinear_tolerances(1e-12, 0.)
            .unwrap();
        let trajectory = Interpreter::new().run(&kernel, config).unwrap();
        let actual = trajectory.last_value(field).unwrap().value();
        assert!(
            (actual - expected).abs() < tolerance,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn repeated_derivatives_of_fixed_parameters_do_not_create_evolving_coordinates() {
    let source = "model M() { parameter p:m=3; relation r {
        derivative(derivative(p))=0[m/s^2];
        derivative(derivative(time()))=0[1/s];
    }}";
    let model = compile("fixed_rates.eqi", source).unwrap().pop().unwrap();
    let relation = model
        .transaction()
        .ops()
        .iter()
        .find_map(|operation| match operation {
            Op::DefineKernelNode {
                node: KernelNode::Relation(relation),
            } => Some(relation),
            _ => None,
        })
        .unwrap();
    let ir = eqiora::ir::ScalarOperatorIr::lower(relation.expression()).unwrap();
    let values = ir
        .evaluate_typed(relation.expression().roots(), &mut |symbol| {
            let (value, dimension) = match symbol {
                SymbolRef::Parameter(_) => (3., [0, 1, 0, 0, 0, 0, 0]),
                SymbolRef::Time => (2., [0, 0, 1, 0, 0, 0, 0]),
                _ => panic!("fixed expression introduced {symbol:?}"),
            };
            eqiora::ValueLiteral::try_from(eqiora::DynQuantity::new(
                value,
                eqiora::DimExponents::from_integers(dimension).unwrap(),
            ))
            .ok()
        })
        .unwrap();
    assert_eq!(values.len(), 4);
    for pair in values.chunks_exact(2) {
        assert_eq!(pair[0], pair[1]);
        assert_eq!(pair[0].real_scalar_value().unwrap().value(), 0.);
    }
}
