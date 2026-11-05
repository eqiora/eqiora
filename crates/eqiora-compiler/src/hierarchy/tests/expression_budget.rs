//! Small independent DAG counts exercise the post-substitution aggregate limit.
use super::*;
fn compile(source: &str, limit: usize) -> Result<Vec<CompiledModel>, Vec<Diagnostic>> {
    let document = eqiora_lang::parse("budget.eqi", source)
        .into_document()
        .unwrap();
    compile_hierarchy_with_limits(
        "budget.eqi",
        source.len(),
        &document,
        HierarchyLimits {
            max_expanded_expression_nodes: limit,
            ..Default::default()
        },
    )
}
#[test]
fn nested_indexed_expression_substitution_is_charged_per_consuming_relation() {
    let source = "component Leaf(support body:volume(ambient_dimension=2),expression f:m on body){relation a on body{f+f=0[m];}relation b on body{f+f=0[m];}} component Outer(support body:volume(ambient_dimension=2),expression g:m on body){let forwarded=g;indexset I=range(4);instance leaf[i in I]:Leaf(body=body,f=forwarded);} model M(){domain body=box(0,1,0,1);instance outer:Outer(body=body,g=(coordinate(0)+coordinate(1))+(coordinate(0)+coordinate(1)));}";
    // Each relation owns Add(f,f), literal zero, and the shared seven-node f.
    // Two relations in each of four indexed children therefore cost exactly72.
    compile(source, 72).unwrap_or_else(|errors| panic!("{errors:?}"));
    let errors = compile(source, 71).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("expanded expression DAG nodes")),
        "{errors:?}"
    );
}
#[test]
fn configured_expression_budget_also_controls_static_definition_preflight() {
    let errors = compile("model M(){variable x:1;relation r{x+x=0;}}", 1).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("expanded expression nodes")),
        "{errors:?}"
    );
}

#[test]
fn indexed_argument_source_work_is_bounded_before_scope_materialization() {
    let source = "component Leaf(support body:volume(ambient_dimension=2),expression f:m on body){relation r on body{f=0[m];}} model M(){domain body=box(0,1,0,1);indexset I=range(4);instance leaf[i in I]:Leaf(body=body,f=(coordinate(0)+coordinate(1))+(coordinate(0)+coordinate(1)));}";
    compile(source, 100).unwrap();
    // Each binding list has a support name plus eleven authored argument nodes;
    // its four occurrences alone exceed47, before any relation is materialized.
    let errors = compile(source, 47).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("expanded expression nodes")),
        "{errors:?}"
    );
}
