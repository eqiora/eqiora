//! Distribution-derived moments use named outcomes and the ordinary expression interpreter.

use eqiora_core::{
    Diagnostic, DimExponents, DynQuantity, Id, OntologyId, RawId, ScalarDomain, ValueType,
    entity::kinds,
};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::{
    Model, ModelView,
    kernel::{
        ComparisonOp, EnumDef, ExprDagBuilder, KernelNode, ObservableDef, ObservableReduction,
        ProbabilityMoment, RelationDef, SymbolRef,
    },
};
use eqiora_sem::KernelProgram;

fn scalar(exponents: [i32; 7]) -> ValueType {
    ValueType::scalar(
        ScalarDomain::Real,
        DimExponents::from_integers(exponents).unwrap(),
    )
    .unwrap()
}

fn admit(
    nodes: Vec<KernelNode>,
    edges: Vec<(RawId, RawId, EdgeKind)>,
) -> Result<KernelProgram, Vec<Diagnostic>> {
    let model = OntologyId::<Model>::new();
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("typed probability observations");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in edges {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, []).unwrap().into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model)
}

struct Fixture {
    nodes: Vec<KernelNode>,
    edges: Vec<(RawId, RawId, EdgeKind)>,
    distribution: Id<kinds::Relation>,
    variable: Id<kinds::Observable>,
    moments: [Id<kinds::Observable>; 3],
}

fn fixture(weights: [f64; 2]) -> Fixture {
    let distribution = Id::new();
    let outcomes = EnumDef::new(Id::new(), ["Low".into(), "High".into()]).unwrap();
    let x = Id::new();
    let y = Id::new();
    let moments = [Id::new(), Id::new(), Id::new()];
    let length = scalar([0, 1, 0, 0, 0, 0, 0]);
    let time = scalar([0, 0, 1, 0, 0, 0, 0]);
    let mut mass = ExprDagBuilder::new();
    let weights = weights.map(|value| {
        mass.constant(DynQuantity::new(value, DimExponents::DIMENSIONLESS))
            .unwrap()
    });
    let weights = mass.array(weights).unwrap();
    let law = RelationDef::probability_mass(
        distribution,
        mass.finish([weights]).unwrap(),
        outcomes.id(),
        weights,
    )
    .unwrap();
    let mut variable = ExprDagBuilder::new();
    let outcome = variable.symbol(SymbolRef::Outcome(distribution)).unwrap();
    let low = variable.constant(outcomes.value(0).unwrap()).unwrap();
    let is_low = variable.compare(ComparisonOp::Equal, outcome, low).unwrap();
    let zero = variable
        .constant(DynQuantity::new(0.0, length.dimension()))
        .unwrap();
    let two = variable
        .constant(DynQuantity::new(2.0, length.dimension()))
        .unwrap();
    let value = variable.select(is_low, zero, two).unwrap();
    let x_definition = ObservableDef::new(
        x,
        length.clone(),
        variable.finish([value]).unwrap(),
        ObservableReduction::RandomVariable { distribution },
    )
    .unwrap();
    let mut variable = ExprDagBuilder::new();
    let three = variable
        .constant(DynQuantity::new(3.0, time.dimension()))
        .unwrap();
    let scale = variable
        .constant(DynQuantity::new(
            1.0,
            scalar([0, -1, 1, 0, 0, 0, 0]).dimension(),
        ))
        .unwrap();
    let x_value = variable.symbol(SymbolRef::Observable(x)).unwrap();
    let product = variable.mul(scale, x_value).unwrap();
    let value = variable.sub(three, product).unwrap();
    let y_definition = ObservableDef::new(
        y,
        time.clone(),
        variable.finish([value]).unwrap(),
        ObservableReduction::RandomVariable { distribution },
    )
    .unwrap();
    let mut edges = vec![
        (
            distribution.erase(),
            outcomes.id().erase(),
            EdgeKind::DependsOn,
        ),
        (x.erase(), distribution.erase(), EdgeKind::DependsOn),
        (y.erase(), distribution.erase(), EdgeKind::DependsOn),
        (y.erase(), x.erase(), EdgeKind::DependsOn),
    ];
    let mut nodes = vec![
        outcomes.into(),
        law.into(),
        x_definition.into(),
        y_definition.into(),
    ];
    for (id, moment, result_type) in [
        (moments[0], ProbabilityMoment::Expectation, length),
        (
            moments[1],
            ProbabilityMoment::Variance,
            scalar([0, 2, 0, 0, 0, 0, 0]),
        ),
        (
            moments[2],
            ProbabilityMoment::Covariance,
            scalar([0, 1, 1, 0, 0, 0, 0]),
        ),
    ] {
        let mut expression = ExprDagBuilder::new();
        let mut roots = vec![expression.symbol(SymbolRef::Observable(x)).unwrap()];
        edges.push((id.erase(), x.erase(), EdgeKind::DependsOn));
        edges.push((id.erase(), distribution.erase(), EdgeKind::DependsOn));
        if moment == ProbabilityMoment::Covariance {
            roots.push(expression.symbol(SymbolRef::Observable(y)).unwrap());
            edges.push((id.erase(), y.erase(), EdgeKind::DependsOn));
        }
        nodes.push(
            ObservableDef::new(
                id,
                result_type,
                expression.finish(roots).unwrap(),
                ObservableReduction::ProbabilityMoment {
                    distribution,
                    moment,
                },
            )
            .unwrap()
            .into(),
        );
    }
    Fixture {
        nodes,
        edges,
        distribution,
        variable: x,
        moments,
    }
}

#[test]
fn finite_law_moments_bind_named_outcomes_without_sampling() {
    let fixture = fixture([0.25, 0.75]);
    let program = admit(fixture.nodes, fixture.edges).unwrap();
    for (id, expected) in fixture.moments.into_iter().zip([1.5, 0.75, -0.75]) {
        for _ in 0..2 {
            let value = program
                .evaluate_finite_observable(id, &mut |_| panic!("no ambient sampled inputs"))
                .unwrap();
            assert_eq!(value.real_scalar_value().unwrap().value(), expected);
        }
        program.typed_observable(id).unwrap();
    }
    let error = program
        .evaluate_finite_observable(fixture.variable, &mut |_| None)
        .unwrap_err();
    assert!(error.message().contains("bound outcome"), "{error:?}");
    assert!(
        program
            .numerical_residuals(fixture.distribution.erase())
            .is_err()
    );
}

#[test]
fn negative_and_zero_mass_cannot_define_a_probability_law() {
    for (weights, expected) in [
        ([-0.25, 1.25], "nonnegative"),
        ([0.0, 0.0], "zero normalization"),
    ] {
        let fixture = fixture(weights);
        let errors = admit(fixture.nodes, fixture.edges).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains(expected)),
            "{errors:?}"
        );
    }
}

#[test]
fn an_unbound_random_variable_is_not_an_ordinary_observation() {
    let mut fixture = fixture([0.25, 0.75]);
    let id = Id::new();
    let mut expression = ExprDagBuilder::new();
    let root = expression
        .symbol(SymbolRef::Observable(fixture.variable))
        .unwrap();
    fixture.nodes.push(
        ObservableDef::new(
            id,
            scalar([0, 1, 0, 0, 0, 0, 0]),
            expression.finish([root]).unwrap(),
            ObservableReduction::Value,
        )
        .unwrap()
        .into(),
    );
    fixture
        .edges
        .push((id.erase(), fixture.variable.erase(), EdgeKind::DependsOn));
    let errors = admit(fixture.nodes, fixture.edges).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("probability-law context")),
        "{errors:?}"
    );
}
