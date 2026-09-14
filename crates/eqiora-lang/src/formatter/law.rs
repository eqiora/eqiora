//! Formatting of retained physical Law terms.

use super::{format_expression, write_indent};
use crate::ast::{ConservationSyntax, RelationDecl};
use core::fmt::Write;

pub(super) fn format_law(
    declaration: &RelationDecl,
    terms: &ConservationSyntax,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    writeln!(
        output,
        "law {} on {} {{",
        declaration.comments.named(&declaration.name),
        declaration
            .domain
            .as_deref()
            .expect("Law owns source support")
    )
    .expect("String write");
    for (name, value) in [
        ("storage", terms.storage()),
        ("flux", Some(terms.flux())),
        ("source", Some(terms.source())),
    ] {
        if let Some(value) = value {
            write_indent(output, indent + 2);
            write!(output, "{name} ").expect("String write");
            format_expression(value, 0, output);
            output.push_str(";\n");
        }
    }
    write_indent(output, indent);
    output.push_str("}\n");
}
