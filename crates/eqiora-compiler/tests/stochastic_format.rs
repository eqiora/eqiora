//! Formatting and source lowering share the same closed process syntax.
use eqiora_graph::Op;
use eqiora_schema::kernel::{KernelNode, RelationMeaning};

#[test]
fn process_syntax_round_trips_notation_calculus_and_all_coefficients() {
    let source = r#"model M() {
        noise driving @{W}: wiener(covariance=[[1,0],[0,1]], independent=true);
        state value: 1;
        initial {value=0;}
        law evolution @{L} for value {
            calculus stratonovich;
            drift 0[1/s];
            diffusion driving=[1[1/s^(1/2)],2[1/s^(1/2)]];
        }
    }"#;
    let parsed = eqiora_lang::parse("format.eqi", source)
        .into_document()
        .unwrap();
    let formatted = eqiora_lang::format(&parsed);
    assert!(formatted.contains("@{W}"));
    assert!(formatted.contains("@{L}"));
    let reparsed = eqiora_lang::parse("format.eqi", &formatted)
        .into_document()
        .unwrap();
    assert_eq!(eqiora_lang::format(&reparsed), formatted);
    for source in [source, &formatted] {
        let compiled = eqiora_compiler::compile("format.eqi", source).unwrap();
        let roots = compiled[0]
            .transaction()
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::DefineKernelNode {
                    node: KernelNode::Relation(relation),
                } if matches!(relation.meaning(), RelationMeaning::Stochastic { .. }) => {
                    Some(relation.expression().roots().len())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(roots, [2]);
    }
}
