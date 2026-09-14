use eqiora::artifact::{ModelEnvelope, ModelTransactionEnvelope};
use eqiora::compiler::{CompiledModel, ModelSymbols, compile};
use eqiora::entity::kinds;
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::ir::ComponentScalarization;
use eqiora::kernel::typing::{
    ExpressionType, RootContract, SpatialSupport, TypeViolation, TypedResidual, isotropic_lift,
    symmetric_part,
};
use eqiora::kernel::{ExprDagBuilder, ExprNode, KernelNode, SymbolRef};
use eqiora::sem::KernelProgram;
use eqiora::{DimExponents, Entity, Id, RawId, ValueShape};
use eqiora_core::ValueFrame;

const ELASTIC_RELATION: &str =
    include_str!("../../../verify/language/canonical-tensor-operators/models/elastic-relation.eqi");

fn compile_one(file: &str, source: &str) -> CompiledModel {
    let mut compiled = compile(file, source).expect("tensor source must compile");
    assert_eq!(compiled.len(), 1, "fixture must define exactly one Model");
    compiled.pop().unwrap()
}

fn admit(compiled: CompiledModel) -> (KernelProgram, ModelSymbols) {
    let (transaction, model, symbols) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store
        .commit(transaction)
        .expect("typed tensor transaction must commit atomically");
    let program = KernelProgram::from_snapshot(&store.snapshot(), model)
        .expect("committed tensor Model must pass whole-model validation");
    (program, symbols)
}

fn typed<I: Entity>(symbols: &ModelSymbols, name: &str) -> Id<I> {
    symbols
        .get(name)
        .and_then(RawId::downcast)
        .unwrap_or_else(|| panic!("missing typed symbol `{name}`"))
}

#[test]
fn source_meaning_crosses_the_closed_current_wire() {
    let compiled = compile_one("elastic-relation.eqi", ELASTIC_RELATION);
    let transaction = compiled.transaction();

    let transaction =
        ModelTransactionEnvelope::from_transaction(transaction).expect("current transaction");
    let transaction_bytes = transaction.canonical_json().unwrap();
    let replayed_transaction =
        ModelTransactionEnvelope::from_json(&transaction_bytes, Default::default())
            .expect("bounded current transaction decode")
            .to_transaction()
            .expect("typed current transaction reconstruction");
    assert_eq!(replayed_transaction.ops(), compiled.transaction().ops());
    assert_eq!(
        replayed_transaction.preconditions(),
        compiled.transaction().preconditions()
    );

    let balance = typed::<kinds::Relation>(compiled.symbols(), "balance");
    let (program, _) = admit(compiled);
    let KernelNode::Relation(relation) = program
        .node(balance.erase())
        .expect("compiled balance Relation belongs to the Model")
    else {
        panic!("balance symbol must retain Relation kind");
    };
    assert_eq!(
        relation
            .expression()
            .nodes()
            .iter()
            .filter(|node| matches!(node, ExprNode::SymmetricPart(_)))
            .count(),
        1
    );
    assert_eq!(
        relation
            .expression()
            .nodes()
            .iter()
            .filter(|node| matches!(node, ExprNode::IsotropicLift(_)))
            .count(),
        1
    );

    let model = ModelEnvelope::from_program(&program).expect("current Model");
    let model_bytes = model.canonical_json().unwrap();
    let model_text = String::from_utf8_lossy(&model_bytes);
    assert!(model_text.contains("eqiora.model-envelope/v27"));
    assert!(model_text.contains("symmetric-part"));
    assert!(model_text.contains("isotropic-lift"));
    let replayed_model = ModelEnvelope::from_json(&model_bytes, Default::default())
        .expect("bounded current Model decode");
    assert_eq!(replayed_model.canonical_json().unwrap(), model_bytes);
    assert_eq!(replayed_model.digest().unwrap(), model.digest().unwrap());
    assert_eq!(replayed_model.to_program().unwrap(), program);

    let document = eqiora::api::ModelDocument::compile("elastic-relation.eqi", ELASTIC_RELATION)
        .expect("public facade must preserve the current wire");
    let document_bytes = document.canonical_json().unwrap();
    assert!(String::from_utf8_lossy(&document_bytes).contains("eqiora.model-envelope/v27"));
    assert_eq!(
        eqiora::api::ModelDocument::replay(&document_bytes)
            .unwrap()
            .canonical_json()
            .unwrap(),
        document_bytes
    );
}

#[test]
fn component_scalarization_is_the_exact_pointwise_tensor_map() {
    let stress = Id::<kinds::Field>::new();
    let pressure = Id::<kinds::Field>::new();
    let mut expression = ExprDagBuilder::new();
    let stress_value = expression.symbol(SymbolRef::Field(stress)).unwrap();
    let pressure_value = expression.symbol(SymbolRef::Field(pressure)).unwrap();
    let symmetric = expression.symmetric_part(stress_value).unwrap();
    let isotropic = expression.isotropic_lift(pressure_value).unwrap();
    let residual = expression.add(symmetric, isotropic).unwrap();
    let expression = expression.finish([residual]).unwrap();
    let dimension =
        DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
    let support = SpatialSupport::Volume {
        domain: "body",
        dimensions: 2,
    };
    let typed_residual = TypedResidual::infer(
        expression,
        Some(support.clone()),
        RootContract::ComponentwiseResidual,
        |symbol| {
            Ok::<_, ()>(match symbol {
                SymbolRef::Field(field) if field == stress => ExpressionType::shaped(
                    dimension,
                    ValueShape::new([2, 2]).unwrap(),
                    ValueFrame::SpatialCartesian,
                    Some(support.clone()),
                )
                .unwrap(),
                SymbolRef::Field(field) if field == pressure => {
                    ExpressionType::scalar(dimension, Some(support.clone()))
                }
                _ => unreachable!("closed pointwise tensor fixture"),
            })
        },
    )
    .expect("pointwise tensor expression must carry one complete typing proof");
    let scalarized = ComponentScalarization::lower(&typed_residual)
        .expect("pointwise tensor structure must scalarize deterministically");
    let coordinates = scalarized
        .rows()
        .iter()
        .map(|row| row.component_index())
        .collect::<Vec<_>>();
    assert_eq!(
        coordinates,
        [vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
    );

    let values = scalarized
        .evaluate(|coordinate| match coordinate.symbol() {
            SymbolRef::Field(field) if field == stress => match coordinate.component_index() {
                [0, 0] => Some(2.0),
                [0, 1] => Some(6.0),
                [1, 0] => Some(10.0),
                [1, 1] => Some(4.0),
                _ => None,
            },
            SymbolRef::Field(field)
                if field == pressure && coordinate.component_index().is_empty() =>
            {
                Some(3.0)
            }
            _ => None,
        })
        .expect("all exact shaped coordinates are supplied");
    assert_eq!(values, [5.0, 8.0, 8.0, 7.0]);
}

#[test]
fn tensor_typing_fails_closed_without_exact_shape_frame_and_volume_support() {
    let dimension =
        DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
    let volume = SpatialSupport::Volume {
        domain: "body",
        dimensions: 2,
    };
    let boundary = SpatialSupport::Boundary {
        domain: "wall",
        parent: "body",
        dimensions: 2,
    };
    let tensor = ExpressionType::shaped(
        dimension,
        ValueShape::new([2, 2]).unwrap(),
        ValueFrame::SpatialCartesian,
        Some(volume.clone()),
    )
    .unwrap();
    assert_eq!(symmetric_part(&tensor).unwrap(), tensor);

    for invalid in [
        ExpressionType::shaped(
            dimension,
            ValueShape::new([2, 3]).unwrap(),
            ValueFrame::SpatialCartesian,
            Some(volume.clone()),
        )
        .unwrap(),
        ExpressionType::shaped(
            dimension,
            ValueShape::new([2, 2]).unwrap(),
            ValueFrame::Invariant,
            Some(volume.clone()),
        )
        .unwrap(),
    ] {
        assert!(matches!(
            symmetric_part(&invalid),
            Err(TypeViolation::SymmetricPartRequiresSquareSpatialTensor)
        ));
    }
    let boundary_tensor = ExpressionType::shaped(
        dimension,
        ValueShape::new([2, 2]).unwrap(),
        ValueFrame::SpatialCartesian,
        Some(boundary.clone()),
    )
    .unwrap();
    assert!(matches!(
        symmetric_part(&boundary_tensor),
        Err(TypeViolation::SymmetricPartRequiresVolume)
    ));

    let scalar = ExpressionType::scalar(dimension, Some(volume));
    let lifted = isotropic_lift(&scalar).unwrap();
    assert_eq!(lifted.dimension(), dimension);
    assert_eq!(lifted.shape(), &ValueShape::new([2, 2]).unwrap());
    assert_eq!(lifted.frame(), ValueFrame::SpatialCartesian);
    assert_eq!(lifted.support, scalar.support);
    assert!(matches!(
        isotropic_lift(&ExpressionType::<&str>::scalar(dimension, None)),
        Err(TypeViolation::IsotropicLiftRequiresVolume)
    ));
    assert!(matches!(
        isotropic_lift(&ExpressionType::scalar(dimension, Some(boundary))),
        Err(TypeViolation::IsotropicLiftRequiresVolume)
    ));
    assert!(matches!(
        isotropic_lift(&tensor),
        Err(TypeViolation::IsotropicLiftRequiresInvariantScalar)
    ));
}
