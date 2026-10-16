//! A named finite law supplies the weights, outcomes and loss for native AD.
use super::*;
use eqiora_core::ValueLiteral;
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent, ScalarOperatorIr};
use eqiora_schema::kernel::{
    ExprDagBuilder, KernelNode, RelationMeaning, SymbolRef, UnaryMathFunction,
};

fn document() -> ModelDocument {
    ModelDocument::compile_selected(
        "bernoulli.eqi",
        r#"enum Outcome {Low,High}
        model Loss() {
          parameter theta:1=1;
          distribution law:normalize_mass(Outcome,weights=[3,theta]);
          observable loss:1=random_variable(law,
            if outcome(law)==Outcome.Low then 2+theta else 6+2*theta);
          observable mean:1=expectation(loss);
        }"#,
        "Loss",
        &[],
    )
    .unwrap()
}

fn real(value: f64) -> ValueLiteral {
    ValueLiteral::try_from(DynQuantity::new(value, DimExponents::DIMENSIONLESS)).unwrap()
}

#[test]
fn finite_law_weights_and_losses_resolve_the_same_design_point() {
    let document = document();
    let parameter = document.parameter_ref("theta").unwrap().id();
    let mean = document.observable_ref("mean").unwrap().id();
    // E=(3(2+t)+t(6+2t))/(3+t). Both weights and losses must
    // use t=2 instead of the declaration's default t=1.
    let value = document
        .program()
        .evaluate_finite_observable(mean, &mut |symbol| {
            assert_eq!(symbol, SymbolRef::Parameter(parameter));
            Some(real(2.0))
        })
        .unwrap();
    assert_eq!(value, real(32.0 / 5.0));
    assert!(
        document
            .program()
            .evaluate_finite_observable(mean, &mut |_| Some(real(-1.0)))
            .is_err()
    );
    assert!(
        document
            .program()
            .evaluate_finite_observable(mean, &mut |_| None)
            .is_err()
    );
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    assert!(
        document
            .program()
            .evaluate_finite_observable(mean, &mut |_| {
                Some(ValueLiteral::try_from(DynQuantity::new(2.0, length)).unwrap())
            })
            .unwrap_err()
            .message()
            .contains("admitted type")
    );
}

#[test]
fn bernoulli_score_reference_differentiates_declared_normalized_weights() {
    let document = document();
    let kernel = document.program();
    let parameter = document.parameter_ref("theta").unwrap().id();
    let KernelNode::Observable(loss) = kernel.node(document.aliases()["loss"]).unwrap() else {
        panic!()
    };
    let distribution = loss.reduction().distribution().unwrap();
    let KernelNode::Relation(law) = kernel.node(distribution.erase()).unwrap() else {
        panic!()
    };
    let RelationMeaning::ProbabilityMass { outcomes, weights } = law.meaning() else {
        panic!()
    };
    let KernelNode::Enum(outcomes) = kernel.node(outcomes.erase()).unwrap() else {
        panic!()
    };
    let loss_ir = ScalarOperatorIr::lower(loss.expression()).unwrap();
    let mut scores = Vec::new();
    for tag in 0..outcomes.members().len() {
        let mut builder = ExprDagBuilder::from_dag(law.expression());
        let masses = (0..outcomes.members().len())
            .map(|index| builder.index(*weights, index as u32).unwrap())
            .collect::<Vec<_>>();
        let total = masses
            .iter()
            .copied()
            .reduce(|a, b| builder.add(a, b).unwrap())
            .unwrap();
        let probability = builder.div(masses[tag], total).unwrap();
        let log_probability = builder
            .unary_math(UnaryMathFunction::Log, probability)
            .unwrap();
        let ir = ScalarOperatorIr::lower(&builder.finish([probability, log_probability]).unwrap())
            .unwrap();
        scores.push(ir);
    }
    // Exact derivative of (6+9t+2t²)/(3+t) at t=1 is 35/16.
    // The score term alone gives 15/16; the direct loss term is 5/4.
    let mut gradient = 0.0;
    for (tag, score_ir) in scores.iter().enumerate() {
        let point = loss_ir
            .symbols()
            .iter()
            .map(|symbol| match *symbol {
                SymbolRef::Outcome(id) if id == distribution => outcomes.value(tag as u32).unwrap(),
                SymbolRef::Parameter(id) if id == parameter => real(1.0),
                _ => panic!("foreign law or unbound dependency"),
            })
            .collect::<Vec<_>>();
        let roles = loss_ir
            .symbols()
            .iter()
            .map(|symbol| match symbol {
                SymbolRef::Outcome(_) => DifferentiationRole::Frozen,
                _ => DifferentiationRole::Parameter,
            })
            .collect::<Vec<_>>();
        let loss = loss_ir.linearize_typed(&point, &roles).unwrap();
        assert_eq!(score_ir.symbols(), &[SymbolRef::Parameter(parameter)]);
        let score = score_ir
            .linearize_typed(&[real(1.0)], &[DifferentiationRole::Parameter])
            .unwrap();
        let mut loss_value = [0.0];
        let mut loss_derivative = [0.0];
        loss.primal(&mut loss_value).unwrap();
        loss.jvp(RelationTangent::Parameter(&[1.0]), &mut loss_derivative)
            .unwrap();
        let mut values = [0.0; 2];
        let mut derivatives = [0.0; 2];
        score.primal(&mut values).unwrap();
        score
            .jvp(RelationTangent::Parameter(&[1.0]), &mut derivatives)
            .unwrap();
        assert_eq!(loss_value, [if tag == 0 { 3.0 } else { 8.0 }]);
        assert_eq!(values[0], if tag == 0 { 0.75 } else { 0.25 });
        assert_eq!(derivatives[1], if tag == 0 { -0.25 } else { 0.75 });
        gradient += values[0] * (loss_derivative[0] + loss_value[0] * derivatives[1]);
    }
    assert_eq!(gradient, 35.0 / 16.0);
}
