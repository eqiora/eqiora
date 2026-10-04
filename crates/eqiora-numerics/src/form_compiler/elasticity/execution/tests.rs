use super::*;

#[test]
fn displacement_identity_is_visible_through_generic_tensor_applications() {
    use eqiora_schema::kernel::ExprDagBuilder;
    let displacement = Id::<kinds::Field>::new();
    let other = Id::<kinds::Field>::new();
    let mut dag = ExprDagBuilder::new();
    let field = dag.symbol(SymbolRef::Field(displacement)).unwrap();
    let gradient = dag.gradient(field).unwrap();
    let stress = dag
        .pure_operator(
            &PureOperatorDefinition::symmetric_part().unwrap(),
            [gradient],
        )
        .unwrap();
    let dag = dag.finish([stress]).unwrap();
    assert!(contains_displacement(&dag, stress, displacement.erase()));
    assert!(!contains_displacement(&dag, stress, other.erase()));
}
