use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id, OntologyId, ScalarDomain, ValueType};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_runtime::{CpuProgram, FirstOrderProgram};
use eqiora_schema::kernel::{
    ActivationDef, ExprDagBuilder, FieldDef, FieldRole, KernelNode, ParameterDef, RelationDef,
    SymbolRef,
};
use eqiora_schema::{Model, ModelView};
use eqiora_sem::{KernelProgram, ReferenceConfig};
use eqiora_time::{InitialConditionPolicy, ParametricTimeSystem, TimeProblem, TimeSystem};

#[test]
fn dense_descriptor_initial_sensitivity_uses_regular_compatibility_equations() {
    let (kernel, relation) = dense_descriptor(false);
    let cpu = CpuProgram::lower(&kernel).unwrap();
    let system = FirstOrderProgram::lower(&cpu, relation).unwrap();
    let initial = system
        .initialize(0.0, ReferenceConfig::new(0.0, 1.0).unwrap())
        .unwrap();
    assert_eq!(initial.state(), [1.0, 1.0]);
    assert_eq!(initial.derivative(), [-2.0, -2.0]);
    let mut tangent = [f64::NAN; 2];
    system
        .initial_parameter_jvp(0.0, &[1.0], &mut tangent)
        .unwrap();
    assert_eq!(tangent, [0.0, 0.0]);
    system.forward_sensitivity_problem(0.0).unwrap();
}

#[test]
fn parameter_dependent_initial_conditions_cannot_silently_return_zero_sensitivity() {
    let (kernel, relation) = dense_descriptor(true);
    let cpu = CpuProgram::lower(&kernel).unwrap();
    let system = FirstOrderProgram::lower(&cpu, relation).unwrap();
    let mut tangent = [0.0; 2];
    assert!(
        system
            .initial_parameter_jvp(0.0, &[1.0], &mut tangent)
            .is_err()
    );
    assert!(system.forward_sensitivity_problem(0.0).is_err());
}

fn dense_descriptor(parameter_initial: bool) -> (KernelProgram, Id<kinds::Relation>) {
    let x = Id::<kinds::Field>::new();
    let y = Id::<kinds::Field>::new();
    let rate = Id::<kinds::Parameter>::new();
    let relation = Id::<kinds::Relation>::new();
    let initial = Id::<kinds::Relation>::new();
    let activation = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();
    let scalar = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
        .expect("numeric scalar type");
    let inverse_time = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let mut expression = ExprDagBuilder::new();
    let dx = expression
        .symbol(SymbolRef::Derivative(x, std::num::NonZeroU32::MIN))
        .unwrap();
    let dy = expression
        .symbol(SymbolRef::Derivative(y, std::num::NonZeroU32::MIN))
        .unwrap();
    let xv = expression.symbol(SymbolRef::Field(x)).unwrap();
    let yv = expression.symbol(SymbolRef::Field(y)).unwrap();
    let p = expression.symbol(SymbolRef::Parameter(rate)).unwrap();
    let two = expression
        .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let twice_rate = expression.mul(two, p).unwrap();
    let derivative_sum = expression.add(dx, dy).unwrap();
    let rx = expression.mul(twice_rate, xv).unwrap();
    let ry = expression.mul(twice_rate, yv).unwrap();
    let first = expression.add(derivative_sum, rx).unwrap();
    let second = expression.add(derivative_sum, ry).unwrap();
    let mut condition = ExprDagBuilder::new();
    let value = condition.symbol(SymbolRef::Field(x)).unwrap();
    let prescribed = if parameter_initial {
        let p = condition.symbol(SymbolRef::Parameter(rate)).unwrap();
        let seconds = condition
            .constant(DynQuantity::new(
                1.0,
                DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap(),
            ))
            .unwrap();
        condition.mul(p, seconds).unwrap()
    } else {
        condition
            .constant(DynQuantity::new(1.0, DimExponents::DIMENSIONLESS))
            .unwrap()
    };

    let nodes = vec![
        KernelNode::from(FieldDef::new(x, scalar.clone(), FieldRole::State)),
        KernelNode::from(FieldDef::new(y, scalar, FieldRole::State)),
        KernelNode::from(ParameterDef::new(
            rate,
            eqiora_core::ValueLiteral::from_real(
                ValueType::scalar(ScalarDomain::Real, inverse_time).expect("numeric scalar type"),
                2.0,
            )
            .unwrap(),
        )),
        KernelNode::from(
            RelationDef::new(
                relation,
                {
                    let zero = expression
                        .constant(DynQuantity::new(0.0, inverse_time))
                        .unwrap();
                    expression.finish([first, zero, second, zero])
                }
                .unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(
            RelationDef::initial(initial, condition.finish([value, prescribed]).unwrap()).unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(activation)),
    ];
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("descriptor initial sensitivity");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for dependency in [x.erase(), y.erase(), rate.erase()] {
        transaction.push(Op::Connect {
            from: relation.erase(),
            to: dependency,
            edge: EdgeKind::DependsOn,
        });
    }
    transaction.push(Op::Connect {
        from: initial.erase(),
        to: x.erase(),
        edge: EdgeKind::DependsOn,
    });
    if parameter_initial {
        transaction.push(Op::Connect {
            from: initial.erase(),
            to: rate.erase(),
            edge: EdgeKind::DependsOn,
        });
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
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        relation,
    )
}

#[test]
fn fresh_initialization_solves_values_and_derivatives_together() {
    let (kernel, relation) = decay(Some(3.0));
    let cpu = CpuProgram::lower(&kernel).unwrap();
    let system = FirstOrderProgram::lower(&cpu, relation).unwrap();
    let initial = system
        .initialize(0.0, ReferenceConfig::new(0.0, 1.0).unwrap())
        .unwrap();
    // x(0)=3 and x'+2x=0 independently imply x'(0)=-6.
    assert!((initial.state()[0] - 3.0).abs() < 1e-10);
    assert!((initial.derivative()[0] + 6.0).abs() < 1e-10);
}

#[test]
fn structural_lowering_and_accepted_restart_do_not_require_fresh_conditions() {
    let (kernel, relation) = decay(None);
    let cpu = CpuProgram::lower(&kernel).unwrap();
    let system = FirstOrderProgram::lower(&cpu, relation).unwrap();
    assert!(
        system
            .initialize(0.0, ReferenceConfig::new(0.0, 1.0).unwrap())
            .is_err()
    );
    assert!(system.time_problem().is_err());
    // This is the adapter seam: the exact State owner authenticates identity
    // before passing an accepted coordinate vector to the time problem.
    let restart = TimeProblem::new(
        &system,
        system.equation_class(),
        InitialConditionPolicy::Provided,
        vec![1.5],
    )
    .unwrap();
    let mut rhs = [0.0];
    system.rhs(0.5, restart.initial_state(), &mut rhs).unwrap();
    assert_eq!(rhs, [-3.0]);
}

fn decay(initial_value: Option<f64>) -> (KernelProgram, Id<kinds::Relation>) {
    let field = Id::<kinds::Field>::new();
    let rate = Id::<kinds::Parameter>::new();
    let relation = Id::<kinds::Relation>::new();
    let activation = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();
    let scalar = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
        .expect("numeric scalar type");
    let inverse_time = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let mut expression = ExprDagBuilder::new();
    let value = expression.symbol(SymbolRef::Field(field)).unwrap();
    let derivative = expression
        .symbol(SymbolRef::Derivative(field, std::num::NonZeroU32::MIN))
        .unwrap();
    let coefficient = expression.symbol(SymbolRef::Parameter(rate)).unwrap();
    let decay = expression.mul(coefficient, value).unwrap();
    let residual = expression.add(derivative, decay).unwrap();
    let mut nodes = vec![
        KernelNode::from(FieldDef::new(field, scalar, FieldRole::State)),
        KernelNode::from(ParameterDef::new(
            rate,
            eqiora_core::ValueLiteral::from_real(
                ValueType::scalar(ScalarDomain::Real, inverse_time).expect("numeric scalar type"),
                2.0,
            )
            .unwrap(),
        )),
        KernelNode::from(
            RelationDef::new(
                relation,
                {
                    let equation_zero = expression
                        .constant(eqiora_core::DynQuantity::new(0.0, inverse_time))
                        .unwrap();
                    expression.finish([residual, equation_zero])
                }
                .unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(activation)),
    ];
    let mut transaction = Transaction::new("fresh initialization and restart separation");
    let mut initial_relation = None;
    if let Some(value) = initial_value {
        let initial = Id::<kinds::Relation>::new();
        let mut expression = ExprDagBuilder::new();
        let field_value = expression.symbol(SymbolRef::Field(field)).unwrap();
        let value = expression
            .constant(DynQuantity::new(value, DimExponents::DIMENSIONLESS))
            .unwrap();

        nodes.push(
            RelationDef::initial(initial, expression.finish([field_value, value]).unwrap())
                .unwrap()
                .into(),
        );
        initial_relation = Some(initial);
    }
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    if let Some(initial) = initial_relation {
        transaction.push(Op::Connect {
            from: initial.erase(),
            to: field.erase(),
            edge: EdgeKind::DependsOn,
        });
    }
    for dependency in [field.erase(), rate.erase()] {
        transaction.push(Op::Connect {
            from: relation.erase(),
            to: dependency,
            edge: EdgeKind::DependsOn,
        });
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
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        relation,
    )
}

#[test]
fn typed_cpu_backend_preserves_real_reference_trajectory() {
    let (kernel, _) = decay(Some(3.0));
    let cpu = CpuProgram::lower(&kernel).unwrap();
    let config = ReferenceConfig::new(0.2, 0.1).unwrap();
    let reference = eqiora_sem::Interpreter::new().run(&kernel, config).unwrap();
    let actual = eqiora_runtime::CpuExecutor::new()
        .run(&cpu, config)
        .unwrap();
    assert_eq!(actual, reference);
}
