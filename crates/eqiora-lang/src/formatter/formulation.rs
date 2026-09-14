//! Canonical formatting for authored mathematical formulations.

use core::fmt::Write;

use crate::ast::formulation::FormulationDecl;
use crate::ast::{ComponentDecl, VisibilitySyntax};

use super::{format_component_item, format_expression, write_indent};

pub(super) fn format_component(
    component: &ComponentDecl,
    output: &mut crate::formatter::comments::Output,
) {
    output.begin(&component.comments);
    if component.visibility == VisibilitySyntax::Public {
        output.push_str("public ");
    }
    write!(
        output,
        "component {}",
        component.comments.named(&component.name)
    )
    .expect("String write");
    super::signature::format_signature(&component.signature, output);
    output.push_str(" {\n");
    for item in &component.items {
        format_component_item(item, 2, output);
    }
    for formulation in &component.formulations {
        format_formulation(formulation, 2, output);
    }
    output.push_str("}\n");
    output.end();
}

pub(super) fn format_formulation(
    declaration: &FormulationDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    output.begin(&declaration.comments);
    write_indent(output, indent);
    output.push_str("form ");
    output.push_str(&declaration.name);
    writeln!(output, " for {} {{", declaration.relation).expect("String write");
    write_indent(output, indent + 2);
    writeln!(
        output,
        "test {}: 1 for {} zero_on {};",
        declaration.test,
        declaration.trial,
        declaration.zero_on.join(", ")
    )
    .expect("String write");
    write_indent(output, indent + 2);
    format_expression(&declaration.left, 0, output);
    output.push_str(" = ");
    format_expression(&declaration.right, 0, output);
    output.push_str(";\n");
    write_indent(output, indent);
    output.push_str("}\n");
    output.end();
}

#[cfg(test)]
mod tests {
    use crate::{format, parse};

    #[test]
    fn primal_form_has_one_canonical_roundtrip() {
        let source = "component D(support region:volume(ambient_dimension=2)) {variable u: 1 on region;relation balance on region{-div(grad(u))=f;}form weak for balance { test w: 1 for u zero_on surface;integrate(region,dot(grad(w),grad(u)))=integrate(region,w*f);}}";
        let first = parse("form.eqi", source).into_document().unwrap();
        let formatted = format(&first);
        let second = parse("form.eqi", &formatted).into_document().unwrap();

        assert_eq!(format(&second), formatted);
        assert!(
            formatted.contains("form weak for balance {\n    test w: 1 for u zero_on surface;\n")
        );
        assert!(
            formatted
                .contains("integrate(region, dot(grad(w), grad(u))) = integrate(region, w * f);")
        );
    }
}
