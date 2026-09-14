//! Time chain rules are checked at nonzero rates, not only equilibrium points.
use eqiora::compiler::compile;
use eqiora::graph::Op;
use eqiora::ir::ScalarOperatorIr;
use eqiora::kernel::{KernelNode, SymbolRef};
use eqiora::{DynQuantity, ValueLiteral};

#[test]
fn time_chain_rule_through_aliases_and_composition_retains_every_state_rate() {
    let source = r#"
operator stored(input x:1,input y:1):1=x*x*y;
model M() {
 state q:1; state y:1;
 parameter fixed:1=7;
 let stored_value=stored(x=q,y=y);
 initial { q=3; y=5; }
 relation r {
   derivative(q*q)=12 [1/s];
   derivative(q^2)=12 [1/s];
   derivative(stored_value)=87 [1/s];
   derivative(fixed*fixed)=0 [1/s];
   derivative(time()*q)=11;
 }
}"#;
    let models = compile("chain.eqi", source).unwrap();
    let operations = models[0].transaction().ops();
    let q = models[0].symbols().get("q").unwrap();
    let y = models[0].symbols().get("y").unwrap();
    for operation in operations {
        let Op::DefineKernelNode {
            node: KernelNode::Relation(relation),
        } = operation
        else {
            continue;
        };
        if relation.is_initial() {
            continue;
        }
        let dag = relation.expression();
        let ir = ScalarOperatorIr::lower(dag).unwrap();
        let values = ir
            .evaluate_typed(dag.roots(), &mut |symbol| {
                let (value, dimension) = match symbol {
                    SymbolRef::Field(id) if id.erase() == q => {
                        (3., eqiora::DimExponents::DIMENSIONLESS)
                    }
                    SymbolRef::Field(id) if id.erase() == y => {
                        (5., eqiora::DimExponents::DIMENSIONLESS)
                    }
                    SymbolRef::Derivative(id) if id.erase() == q => (
                        2.,
                        eqiora::DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap(),
                    ),
                    SymbolRef::Derivative(id) if id.erase() == y => (
                        3.,
                        eqiora::DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap(),
                    ),
                    SymbolRef::Time => (
                        4.,
                        eqiora::DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap(),
                    ),
                    SymbolRef::Parameter(_) => (7., eqiora::DimExponents::DIMENSIONLESS),
                    _ => return None,
                };
                ValueLiteral::try_from(DynQuantity::new(value, dimension)).ok()
            })
            .unwrap();
        let (pairs, remainder) = values.as_chunks::<2>();
        assert!(remainder.is_empty());
        for pair in pairs {
            assert_eq!(pair[0], pair[1]);
        }
    }
}

#[test]
fn time_coordinate_and_smooth_derivatives_reject_wrong_context_or_dependencies() {
    for relation in [
        "derivative(v*v)=0 [1/s]",
        "derivative(pre(q))=0 [1/s]",
        "derivative(math.sin(q))=0 [1/s]",
        "time(1)=0 [s]",
        "time=0 [s]",
    ] {
        let source = format!(
            "model M() {{ state q:1; variable v:1; initial {{q=1;}} relation r {{{relation};}} }}"
        );
        assert!(compile("invalid.eqi", &source).is_err(), "{relation}");
    }
    for expression in ["derivative(q*q)", "time()/1[s]"] {
        let source = format!(
            "model M() {{ clock tick=periodic(1[s]); state q:1 at tick; initial {{q=1;}} relation r at tick {{{expression}=0;}} }}"
        );
        assert!(compile("clocked.eqi", &source).is_err(), "{expression}");
    }
}

#[test]
fn nonlinear_stored_quantity_runs_through_the_common_implicit_lifecycle() {
    use eqiora::graph::{GraphStore, InMemoryGraphStore};
    use eqiora::runtime::{CpuProgram, GeneralImplicitProgram};
    use eqiora::sem::KernelProgram;
    use eqiora::time::{
        DaeVariableKind, ImplicitDaeProblem, InitialConditionPolicy, ReferenceImplicitTimeBackend,
        TimeMethod, TimePlan,
    };
    let mut trajectories = Vec::new();
    for expression in ["derivative(q*q)", "2*q*derivative(q)"] {
        let source = format!(
            "model Storage() {{ state q:1; initial {{q=1;}} relation flow {{{expression}=2 [1/s];}} }}"
        );
        let model = compile("storage.eqi", &source).unwrap().pop().unwrap();
        let (transaction, model_id, symbols) = model.into_parts();
        let relation = symbols.get("flow").unwrap().downcast().unwrap();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let kernel = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
        let cpu = CpuProgram::lower(&kernel).unwrap();
        let system = GeneralImplicitProgram::lower(&cpu, relation).unwrap();
        let envelope = eqiora::artifact::ModelEnvelope::from_program(&kernel).unwrap();
        let replay = envelope.to_program().unwrap();
        let replay_system =
            GeneralImplicitProgram::lower(&CpuProgram::lower(&replay).unwrap(), relation).unwrap();
        assert_eq!(system.lowering_proof(), replay_system.lowering_proof());
        let lowering = eqiora::artifact::GeneralImplicitTimeLoweringEnvelopeV1::from_proof(
            &envelope,
            &kernel,
            system.lowering_proof(),
        )
        .unwrap();
        lowering.validate_against(&envelope, &replay).unwrap();

        let initial = system
            .initialize(
                eqiora::sem::ReferenceConfig::new(0., 1.)
                    .unwrap()
                    .with_initial_guess(2.)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(initial.state(), &[1.]);
        assert_eq!(initial.derivative(), &[1.]);
        let problem = ImplicitDaeProblem::new(
            &system,
            vec![DaeVariableKind::Differential],
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
            vec![1e-13],
            vec![0.25, 0.5, 1.],
        )
        .unwrap();
        let solution = ReferenceImplicitTimeBackend::new()
            .solve(&problem, &plan)
            .unwrap();
        let values = [0.25f64, 0.5, 1.]
            .into_iter()
            .enumerate()
            .map(|(index, time)| {
                let value = solution.state(index).unwrap()[0];
                // q'=1/q, q(0)=1 implies q=sqrt(1+2t). Backward Euler error is O(h).
                assert!((value - (1. + 2. * time).sqrt()).abs() < 0.001);
                value
            })
            .collect::<Vec<_>>();
        trajectories.push(values);
    }
    assert_eq!(trajectories[0], trajectories[1]);
}
