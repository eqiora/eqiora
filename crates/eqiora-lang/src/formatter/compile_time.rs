use core::fmt::Write;

use crate::ast::{NamedDefinitionDecl, ParameterDecl};

use super::{format_expression, write_indent};

pub(super) fn format_dimension(
    declaration: &NamedDefinitionDecl,
    output: &mut crate::formatter::comments::Output,
) {
    output.begin(&declaration.comments);
    if declaration.visibility() == crate::VisibilitySyntax::Public {
        output.push_str("public ");
    }
    write!(
        output,
        "dimension {} = ",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    format_expression(&declaration.value, 0, output);
    output.push_str(";\n");
    output.end();
}

pub(super) fn format_parameter(
    declaration: &ParameterDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    write!(
        output,
        "parameter {}: ",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    super::value_type::format_value_type(&declaration.value_type, output);
    output.push_str(" = ");
    format_expression(&declaration.value, 0, output);
    output.push_str(";\n");
}

pub(super) fn format_observable(
    declaration: &crate::ObservableDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    write!(
        output,
        "observable {}: ",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    super::value_type::format_value_type(&declaration.value_type, output);
    if let Some(domain) = declaration.domain() {
        write!(output, " on {domain}").expect("String write");
    }
    output.push_str(" = ");
    format_expression(&declaration.value, 0, output);
    output.push_str(";\n");
}

pub(super) fn format_let(
    declaration: &NamedDefinitionDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    write!(
        output,
        "let {}",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    if let Some(value_type) = &declaration.value_type {
        output.push_str(": ");
        super::value_type::format_value_type(value_type, output);
    }
    if let Some(domain) = &declaration.domain {
        write!(output, " on {domain}").expect("String write");
    }
    if let Some(activation) = &declaration.activation {
        write!(output, " at {activation}").expect("String write");
    }
    output.push_str(" = ");
    format_expression(&declaration.value, 0, output);
    output.push_str(";\n");
}

pub(super) fn format_coordinate(
    declaration: &NamedDefinitionDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    write!(
        output,
        "coordinate {}: ",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    super::value_type::format_value_type(
        declaration
            .value_type
            .as_ref()
            .expect("coordinate dimension"),
        output,
    );
    write!(
        output,
        " on {} from ",
        declaration.domain.as_ref().expect("coordinate support")
    )
    .expect("String write");
    format_expression(&declaration.value, 0, output);
    output.push_str(";\n");
}
