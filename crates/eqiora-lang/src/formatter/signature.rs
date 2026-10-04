//! Canonical shared external signature rendering.
use super::*;
use crate::{ActivationSyntax, SignatureItem};

pub(super) fn format_signature(items: &[SignatureItem], output: &mut comments::Output) {
    output.push('(');
    if !items.is_empty() {
        output.push('\n');
    }
    for item in items {
        output.begin(item.source_comments());
        write_indent(output, 2);
        match item {
            SignatureItem::Parameter(value) => {
                write!(output, "parameter {}: ", value.comments.named(&value.name))
                    .expect("String write");
                value_type::format_value_type(&value.value_type, output);
                if let Some(default) = &value.default {
                    output.push_str(" = ");
                    format_expression(default, 0, output);
                }
            }
            SignatureItem::Support(value) => {
                write!(output, "support {}: ", value.comments.named(&value.name))
                    .expect("String write");
                match &value.syntax {
                    SupportSlotSyntax::Interval { dimension } => {
                        output.push_str("interval(");
                        format_expression(dimension, 0, output);
                        output.push(')');
                        Ok(())
                    }
                    SupportSlotSyntax::Volume { ambient_dimension } => {
                        write!(output, "volume(ambient_dimension = {ambient_dimension})")
                    }
                    SupportSlotSyntax::Boundary { parent } => {
                        write!(output, "boundary(parent = {parent})")
                    }
                    SupportSlotSyntax::CompleteExterior { parent } => {
                        write!(output, "complete_exterior(parent = {parent})")
                    }
                }
                .expect("String write");
            }
            SignatureItem::Field(value) => format_unknown_head(value, output),
            SignatureItem::Clock(value) => {
                write!(
                    output,
                    "clock {}: periodic",
                    value.comments.named(&value.name)
                )
                .expect("String write");
            }
            SignatureItem::Property(value) => {
                write!(
                    output,
                    "property {}: {}",
                    value.comments.named(&value.name),
                    value.contract
                )
                .expect("String write");
            }
            SignatureItem::Input(value) | SignatureItem::Output(value) => {
                let direction = if matches!(item, SignatureItem::Input(_)) {
                    "input"
                } else {
                    "output"
                };
                write!(
                    output,
                    "{direction} {}: ",
                    value.comments.named(&value.name)
                )
                .expect("String write");
                value_type::format_value_type(&value.value_type, output);
                if let Some(domain) = &value.domain {
                    write!(output, " on {domain}").expect("String write");
                }
                if let ActivationSyntax::Named(clock) = &value.activation {
                    write!(output, " at {clock}").expect("String write");
                }
            }
            SignatureItem::Port(value) => {
                write!(output, "port {}: ", value.comments.named(&value.name))
                    .expect("String write");
                format_port_syntax(&value.syntax, output);
            }
            SignatureItem::PortFamily(value) => {
                write!(
                    output,
                    "port {}[{} in {}]: ",
                    value.port.comments.named(&value.port.name),
                    value.binder.member,
                    value.binder.set
                )
                .expect("String write");
                format_port_syntax(&value.port.syntax, output);
            }
        }
        output.push_str(",\n");
        output.end();
    }
    output.push(')');
}
