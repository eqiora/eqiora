//! Native AST validation remains inside the one compiler lowering owner.

fn compile_model(
    file: &str,
    model: &eqiora_lang::ModelDecl,
) -> Result<crate::CompiledModel, Vec<eqiora_core::Diagnostic>> {
    let document = eqiora_lang::SourceAstFactory::document(
        Vec::new(),
        Vec::new(),
        Vec::new(),
        vec![model.clone()],
    )
    .unwrap();
    crate::hierarchy::selected::local_document(
        file,
        0,
        document,
        None,
        &[],
        crate::hierarchy::HierarchyLimits::default(),
    )
    .map(|mut models| models.remove(0))
}
use crate::compile;

#[test]
fn source_and_factory_admit_boundary_field_support() {
    use eqiora_lang::{Item, SourceAstFactory, VisibilitySyntax};
    let source = "model M() { domain body = box(0,1); domain wall = boundary(body, axis = 0, side = lower); state x: 1 on wall; initial { x = 0; } }";
    compile("boundary.eqi", source).unwrap();
    compile("boundary.eqi", &format!("component Marker() {{}} {source}")).unwrap();
    let document = eqiora_lang::parse("boundary.eqi", source)
        .into_document()
        .unwrap();
    let model = &document.models()[0];
    let items = model
        .items()
        .iter()
        .map(|item| match item {
            Item::Field(field) => Item::Field(
                SourceAstFactory::field(
                    field.name(),
                    field.domain().map(str::to_owned),
                    field.role(),
                    field.spatial_regularity(),
                    field.activation().clone(),
                    field.value_type().clone(),
                    field.range(),
                )
                .unwrap(),
            ),
            other => other.clone(),
        })
        .collect();
    let rebuilt = SourceAstFactory::model(
        VisibilitySyntax::Private,
        model.name(),
        model.signature().to_vec(),
        items,
        model.range(),
    )
    .unwrap();
    compile_model("factory.eqi", &rebuilt).unwrap();
    compile("required.eqi", "component C(support body: volume(ambient_dimension = 1), support wall: boundary(parent = body), state x: 1 on wall) {} model M() { variable y: 1; relation r { y = 0; } }").unwrap();
}

#[test]
fn invalid_unused_clock_definitions_keep_operand_ranges() {
    for container in ["model M()", "component Unused()"] {
        let source = format!(
            "{container} {{ clock tick = periodic(1[s] + 1[m]); }} model Root() {{ variable x:1; relation r {{ x=0; }} }}"
        );
        let errors = compile("clock.eqi", &source).unwrap_err();
        let error = errors
            .iter()
            .find(|error| error.message().contains("equal dimensions"))
            .expect("dimension diagnostic");
        let span = error.source_span().unwrap();
        assert_eq!(
            &source[span.start as usize..span.end as usize],
            "1[s] + 1[m]"
        );
    }
    let source = "model M() { clock tick = periodic(1[s] + 1[m]); }";
    let document = eqiora_lang::parse("clock.eqi", source)
        .into_document()
        .unwrap();
    let errors = compile_model("clock.eqi", &document.models()[0]).unwrap_err();
    let span = errors
        .iter()
        .find(|error| error.message().contains("equal dimensions"))
        .unwrap()
        .source_span()
        .unwrap();
    assert_eq!(
        &source[span.start as usize..span.end as usize],
        "1[s] + 1[m]"
    );
}
