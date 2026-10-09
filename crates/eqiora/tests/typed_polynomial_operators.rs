//! Native/source scalar constitutive algebra and inspectable exact formal derivatives.
use eqiora::api::ModelDocument;
use eqiora::entity::kinds;
use eqiora::graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora::ir::ScalarOperatorIr;
use eqiora::kernel::pure_operator::{
    CalculusBuilder, CalculusNode, ExactRational, PureOperatorDefinition, PureValueClass,
};
use eqiora::kernel::typing::{ExpressionType, RootContract, TypedResidual};
use eqiora::kernel::{
    ActivationDef, ExprDagBuilder, ExprNode, FieldDef, FieldRole, KernelNode, ParameterDef,
    RelationDef, SymbolRef,
};
use eqiora::ontology::{Model, ModelView, OntologyId};
use eqiora::runtime::{CpuExecutor, CpuProgram};
use eqiora::sem::{Interpreter, KernelProgram, ReferenceConfig};
use eqiora::{DimExponents, Id, RawId, ScalarDomain, ValueLiteral, ValueType};
use eqiora_ir::OperatorExpansionExt;

const SOURCE: &str = r#"
operator conductivity(input x:K,input k0:W/m/K,input a:1/K):W/m/K = k0*(1+a*x+a*a*x*x);
model Conductivity() {
  parameter temperature:K=20;
  parameter base:W/m/K=10;
  parameter slope:1/K=0.01;
  variable value:W/m/K;
  relation evaluate { value=conductivity(a=slope,x=temperature,k0=base); }
}
"#;
fn t() -> DimExponents {
    DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).unwrap()
}
fn k() -> DimExponents {
    DimExponents::from_integers([1, 1, -3, 0, -1, 0, 0]).unwrap()
}
fn real(dimension: DimExponents, value: f64) -> ValueLiteral {
    ValueLiteral::from_real(
        ValueType::scalar(ScalarDomain::Real, dimension).expect("valid scalar type"),
        value,
    )
    .unwrap()
}
fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 2e-12, "{actual} != {expected}");
}

fn native_conductivity() -> PureOperatorDefinition {
    let scalar = PureValueClass::invariant_scalar();
    let mut b = CalculusBuilder::new(
        [
            scalar.with_dimension(t()),
            scalar.with_dimension(k()),
            scalar.with_dimension(t().pow(-1, 1).unwrap()),
        ],
        scalar.with_dimension(k()),
    )
    .unwrap();
    let mut formal = |formal| {
        b.push(CalculusNode::FormalComponent {
            formal,
            axes: Box::new([]),
        })
        .unwrap()
    };
    let x = formal(0);
    let k0 = formal(1);
    let a = formal(2);
    let one = b
        .push(CalculusNode::Rational {
            value: ExactRational::integer(1),
            dimension: DimExponents::DIMENSIONLESS,
        })
        .unwrap();
    let ax = b.push(CalculusNode::Mul(a, x)).unwrap();
    let aa = b.push(CalculusNode::Mul(a, a)).unwrap();
    let aax = b.push(CalculusNode::Mul(aa, x)).unwrap();
    let aaxx = b.push(CalculusNode::Mul(aax, x)).unwrap();
    let linear = b.push(CalculusNode::Add(one, ax)).unwrap();
    let quadratic = b.push(CalculusNode::Add(linear, aaxx)).unwrap();
    let result = b.push(CalculusNode::Mul(k0, quadratic)).unwrap();
    b.finish(result).unwrap()
}
fn native_program(definition: &PureOperatorDefinition) -> (KernelProgram, RawId) {
    let field = Id::<kinds::Field>::new();
    let relation = Id::<kinds::Relation>::new();
    let activation = Id::<kinds::Activation>::new();
    let parameters = [Id::<kinds::Parameter>::new(), Id::new(), Id::new()];
    let mut builder = ExprDagBuilder::new();
    let lhs = builder.symbol(SymbolRef::Field(field)).unwrap();
    let args = parameters.map(|id| builder.symbol(SymbolRef::Parameter(id)).unwrap());
    let rhs = builder.pure_operator(definition, args).unwrap();
    let mut nodes = vec![
        KernelNode::from(FieldDef::new(
            field,
            ValueType::scalar(ScalarDomain::Real, k()).expect("valid scalar type"),
            FieldRole::Variable,
        )),
        RelationDef::new(relation, builder.finish([lhs, rhs]).unwrap())
            .unwrap()
            .into(),
        ActivationDef::continuous(activation).into(),
    ];
    for (id, value) in parameters.into_iter().zip([
        real(t(), 20.),
        real(k(), 10.),
        real(t().pow(-1, 1).unwrap(), 0.01),
    ]) {
        nodes.push(ParameterDef::new(id, value).into());
    }
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("typed scalar polynomial native fixture");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for dependency in parameters.map(Id::erase).into_iter().chain([field.erase()]) {
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
    let model = OntologyId::<Model>::new();
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, []).unwrap().into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        field.erase(),
    )
}
fn execute(program: &KernelProgram, field: RawId, expected: f64) {
    let config = ReferenceConfig::new(0., 1.)
        .unwrap()
        .with_nonlinear_tolerances(1e-13, 0.)
        .unwrap();
    let reference = Interpreter::new().run(program, config).unwrap();
    let cpu = CpuExecutor::new()
        .run(&CpuProgram::lower(program).unwrap(), config)
        .unwrap();
    for trajectory in [reference, cpu] {
        let value = trajectory.last_value(field).unwrap();
        close(value.value(), expected);
        assert_eq!(value.dim(), k());
    }
}

#[test]
fn conductivity_native_source_cpu_and_parameter_edit_preserve_typed_results() {
    // K0(1+aT+a²T²): 10(1+.2+.04)=12.4, with no temperature solve or fitted oracle.
    let native = native_conductivity();
    let (program, field) = native_program(&native);
    execute(&program, field, 12.4);
    let source = ModelDocument::compile("conductivity.eqi", SOURCE).unwrap();
    execute(source.program(), source.aliases()["value"], 12.4);
    let id = source.aliases()["temperature"];
    let preview = source.preview_value_edit(id, real(t(), 30.)).unwrap();
    let edited = source.commit_value_edit(preview).unwrap().into_document();
    assert_eq!(edited.aliases()["temperature"], id);
    execute(edited.program(), edited.aliases()["value"], 13.9);
    let relation = source
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(r) if !r.is_initial() => Some(r),
            _ => None,
        })
        .unwrap();
    let application = relation
        .expression()
        .nodes()
        .iter()
        .find_map(|node| match node {
            ExprNode::PureOperatorApplication(app) => Some(app),
            _ => None,
        })
        .unwrap();
    let definition = relation
        .expression()
        .definition(application.definition())
        .unwrap();
    // Compare the executable scalar result and complete derivative types.
    for definition in [definition, &native] {
        check_partials(
            definition,
            [
                real(t(), 20.),
                real(k(), 10.),
                real(t().pow(-1, 1).unwrap(), 0.01),
            ],
            0,
            [0.14, 0.002],
            [
                k().div(t()).unwrap(),
                k().div(t().pow(2, 1).unwrap()).unwrap(),
            ],
        );
    }
}

fn check_partials(
    definition: &PureOperatorDefinition,
    values: [ValueLiteral; 3],
    formal: u16,
    expected: [f64; 2],
    dimensions: [DimExponents; 2],
) {
    let types = values
        .iter()
        .map(|value| ExpressionType::<()>::new(value.value_type().clone(), None))
        .collect::<Vec<_>>();
    let scalar = definition
        .instantiate(&types)
        .unwrap()
        .component(&[])
        .unwrap();
    let mut builder = ExprDagBuilder::new();
    let arguments = values
        .into_iter()
        .map(|value| builder.constant(value).unwrap())
        .collect::<Vec<_>>();
    let mut roots = Vec::new();
    for (order, dimension) in [1, 2].into_iter().zip(dimensions) {
        let (root, ty) = scalar
            .partial(&mut builder, &arguments, formal, order)
            .unwrap();
        assert_eq!(ty.dimension(), dimension);
        assert_eq!(ty.support, None);
        roots.push(root);
    }
    let dag = builder.finish(roots.clone()).unwrap();
    assert!(dag.nodes().iter().all(|node| matches!(
        node,
        ExprNode::Constant(_) | ExprNode::Add(..) | ExprNode::Neg(..) | ExprNode::Mul(..)
    )));
    let typed = TypedResidual::<()>::infer(
        dag.clone(),
        None,
        |_| None,
        RootContract::InitialResiduals,
        |_| Err::<ExpressionType<()>, _>(()),
    )
    .unwrap();
    for (root, dimension) in roots.iter().zip(dimensions) {
        assert_eq!(typed.node_type(*root).unwrap().dimension(), dimension);
    }
    let ir = ScalarOperatorIr::lower(&dag).unwrap();
    for (actual, expected) in ir.evaluate(&[]).unwrap().into_iter().zip(expected) {
        close(actual, expected);
    }
}

#[test]
fn cubic_force_first_and_second_derivatives_retain_stiffness_units() {
    let source = r#"operator force(input x:m,input stiffness:N/m,input cubic:N/m^3):N=stiffness*x+cubic*x*x*x;
        model Spring(){ parameter x:m=2;parameter stiffness:N/m=3;parameter cubic:N/m^3=5;variable force_value:N;
        relation balance { force_value=force(x=x,stiffness=stiffness,cubic=cubic); }}"#;
    let document = ModelDocument::compile("spring.eqi", source).unwrap();
    let definition = document
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(r) => r.expression().definitions().values().next(),
            _ => None,
        })
        .unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let force = DimExponents::from_integers([1, 1, -2, 0, 0, 0, 0]).unwrap();
    let stiffness = force.div(length).unwrap();
    let cubic = force.div(length.pow(3, 1).unwrap()).unwrap();
    // k+3cx²=63 N/m and 6cx=60 N/m² at x=2 m, k=3 N/m, c=5 N/m³.
    check_partials(
        definition,
        [real(length, 2.), real(stiffness, 3.), real(cubic, 5.)],
        0,
        [63., 60.],
        [stiffness, force.div(length.pow(2, 1).unwrap()).unwrap()],
    );
}

#[test]
fn exact_imported_polynomial_replay_preserves_values_and_typed_partials() {
    use eqiora::artifact::ModelEnvelope;
    use eqiora::compiler::{
        CompilationNamespaceId, ResolvedDependency, ResolvedHierarchyInput, ResolvedSourceUnit,
        analyze_resolved_hierarchy,
    };
    let root = CompilationNamespaceId::new(["consumer.example", "1.0.0", "root"]).unwrap();
    let provider = CompilationNamespaceId::new(["provider.example", "1.0.0", "provider"]).unwrap();
    let (declaration, body) = SOURCE.trim().split_once('\n').unwrap();
    let input = ResolvedHierarchyInput::with_root_module(
        root.clone(),
        ["main"],
        vec![
            ResolvedSourceUnit::new(
                root.clone(),
                "src/main.eqi",
                format!(
                    "import provider.example.main as laws;\n{}",
                    body.replace("=conductivity(", "=laws.conductivity(")
                ),
            )
            .unwrap(),
            ResolvedSourceUnit::new(
                provider.clone(),
                "src/main.eqi",
                format!("public {declaration}"),
            )
            .unwrap(),
        ],
        vec![ResolvedDependency::new(root, provider)],
    )
    .unwrap();
    let compiled = analyze_resolved_hierarchy(input)
        .unwrap()
        .validate_definitions()
        .unwrap()
        .compile_root("Conductivity")
        .unwrap();
    let (transaction, model, symbols) = compiled.into_parts();
    let field = symbols.get("value").unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let bytes = ModelEnvelope::from_program(&program)
        .unwrap()
        .canonical_json()
        .unwrap();
    let replay = ModelEnvelope::from_json(&bytes, Default::default())
        .unwrap()
        .to_program()
        .unwrap();
    for program in [&program, &replay] {
        // K0(1+aT+a²T²)=12.4; d/dT=K0(a+2a²T)=.14; d²/dT²=2K0a²=.002.
        execute(program, field, 12.4);
        let definition = program
            .nodes()
            .find_map(|node| {
                let KernelNode::Relation(relation) = node else {
                    return None;
                };
                relation.expression().nodes().iter().find_map(|node| {
                    let ExprNode::PureOperatorApplication(app) = node else {
                        return None;
                    };
                    relation.expression().definition(app.definition())
                })
            })
            .unwrap();
        check_partials(
            definition,
            [
                real(t(), 20.),
                real(k(), 10.),
                real(t().pow(-1, 1).unwrap(), 0.01),
            ],
            0,
            [0.14, 0.002],
            [
                k().div(t()).unwrap(),
                k().div(t().pow(2, 1).unwrap()).unwrap(),
            ],
        );
    }
}
