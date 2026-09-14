//! Ordinary source lowering retains physical Law identity independently of names.

use eqiora_compiler::compile;
use eqiora_graph::Op;
use eqiora_schema::kernel::{KernelNode, RelationMeaning};

fn steady(field_type: &str, coefficient_type: &str, source_type: &str) -> String {
    format!(
        r#"
public component Balance(
  support body: volume(ambient_dimension = 1),
  variable value: {field_type} on body,
  parameter coefficient: {coefficient_type},
  parameter production: {source_type}
) {{
  law conservation on body {{
    flux -coefficient * grad(value);
    source production;
  }}
}}
model Main() {{
  domain body = box(0, 1);
  variable value: {field_type} on body;
  parameter coefficient: {coefficient_type} = 2;
  parameter production: {source_type} = 4;
  instance balance: Balance(body = body, value = value,
    coefficient = coefficient, production = production);
}}
"#
    )
}

#[test]
fn independent_heat_and_mass_consumers_retain_one_physical_vocabulary() {
    for source in [
        steady("K", "kg * m / s ^ 3 / K", "kg / m / s ^ 3"),
        steady("kg / m ^ 3", "m ^ 2 / s", "kg / m ^ 3 / s"),
    ] {
        let document = eqiora_lang::parse("law.eqi", &source)
            .into_document()
            .unwrap();
        let formatted = eqiora_lang::format(&document);
        assert!(formatted.contains("law conservation on body"));
        for authored in [&source, &formatted] {
            let models = compile("law.eqi", authored).unwrap();
            let laws = models[0]
                .transaction()
                .ops()
                .iter()
                .filter_map(|op| {
                    let Op::DefineKernelNode {
                        node: KernelNode::Relation(relation),
                    } = op
                    else {
                        return None;
                    };
                    let RelationMeaning::Conservation(terms) = relation.meaning() else {
                        return None;
                    };
                    Some((relation, terms))
                })
                .collect::<Vec<_>>();
            assert_eq!(laws.len(), 1);
            let (relation, terms) = laws[0];
            terms.validate_balance(relation.expression()).unwrap();
        }
    }
}

#[test]
fn wrong_physical_source_units_and_foreign_support_fail_source_admission() {
    let source = steady("K", "kg * m / s ^ 3 / K", "kg / m / s ^ 3");
    let wrong_units = source.replace("source production;", "source coefficient;");
    let error = compile("wrong-source.eqi", &wrong_units).unwrap_err();
    assert!(
        error
            .iter()
            .any(|error| error.code() == eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR),
        "{error:?}"
    );
    let foreign = source
        .replace(
            "domain body = box(0, 1);",
            "domain body = box(0, 1); domain other = box(0, 1);",
        )
        .replace(
            "instance balance: Balance(body = body",
            "instance balance: Balance(body = other",
        );
    assert!(compile("foreign-support.eqi", &foreign).is_err());
}

#[test]
fn law_requires_each_physical_term_once() {
    let source = steady("K", "kg * m / s ^ 3 / K", "kg / m / s ^ 3");
    for mutant in [
        source.replace("source production;", ""),
        source.replace(
            "source production;",
            "source production; source production;",
        ),
        source.replace("flux -coefficient * grad(value);", ""),
        source.replace(
            "flux -coefficient * grad(value);",
            "flux -coefficient * grad(value); flux -coefficient * grad(value);",
        ),
    ] {
        assert!(
            eqiora_lang::parse("bad-law.eqi", &mutant)
                .into_document()
                .is_err()
        );
    }
}
