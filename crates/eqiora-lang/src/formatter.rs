//! One canonical source style derived from the recovered syntax tree.
//! Declaration-specific modules keep the top-level traversal readable.
use core::fmt::Write;

mod cartesian;
mod comments;
mod compile_time;
mod document;
mod expression;
mod signature;
use expression::format_expression;
mod formulation;
mod helpers;
mod law;
mod property;
mod relation;
mod value_type;

use crate::ast::{
    BoundaryConnectionDecl, BoundaryPairingSyntax, BoundaryPortReferenceSyntax, BoundarySideSyntax,
    ClockDecl, ComponentItem, ComponentPortFamilyDecl, ConnectionDecl, ConnectionSyntax,
    ConnectorSyntax, Document, DomainSyntax, FamilyBinderSyntax, FieldDecl, FrameSyntax,
    InstanceDecl, Item, PortSyntax, PureOperatorDecl, PureValueClassSyntax, SignalDirectionSyntax,
    SupportSlotSyntax, ValueShapeSyntax, VisibilitySyntax,
};
use cartesian::format_cartesian_coordinate;
use compile_time::{format_let, format_parameter};
use formulation::format_component;
use helpers::{format_boundary_port_selector, format_scalar_physical, write_indent};
use property::format_properties;
use relation::{format_relation, format_relation_family};

pub(crate) fn expression_source(expression: &crate::Expr) -> String {
    let mut output = comments::Output::default();
    format_expression(expression, 0, &mut output);
    output.finish()
}

/// Canonically format syntax and the comment trivia owned by each declaration.
#[must_use]
pub fn format(document: &Document) -> String {
    let mut output = comments::Output::default();
    output.begin(&document.comments);
    let mut declaration_count = document::format_header(document, &mut output);
    format_properties(document, &mut output, &mut declaration_count);
    for declaration in &document.enumerations {
        separate_declaration(&mut output, &mut declaration_count);
        output.begin(&declaration.comments);
        if declaration.visibility() == VisibilitySyntax::Public {
            output.push_str("public ");
        }
        write!(
            output,
            "enum {} {{ ",
            declaration.comments.named(&declaration.name)
        )
        .expect("String write");
        for (index, tag) in declaration.tags().iter().enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            write!(output, "{tag}").expect("String write");
        }
        output.push_str(" }\n");
        output.end();
    }
    for declaration in &document.records {
        separate_declaration(&mut output, &mut declaration_count);
        output.begin(&declaration.comments);
        if declaration.visibility() == VisibilitySyntax::Public {
            output.push_str("public ");
        }
        writeln!(
            output,
            "record {} {{",
            declaration.comments.named(&declaration.name)
        )
        .expect("String write");
        for member in declaration.members() {
            output.begin(&member.comments);
            write!(output, "    {}: ", member.comments.named(&member.name)).expect("String write");
            value_type::format_value_type(member.value_type(), &mut output);
            output.push_str(",\n");
            output.end();
        }
        output.push_str("}\n");
        output.end();
    }
    for space in &document.finite_spaces {
        separate_declaration(&mut output, &mut declaration_count);
        output.begin(&space.comments);
        if space.visibility == VisibilitySyntax::Public {
            output.push_str("public ");
        }
        write!(output, "space {} = ", space.comments.named(&space.name)).expect("String write");
        format_expression(space.value(), 0, &mut output);
        output.push_str(";\n");
        output.end();
    }
    for connector in &document.connectors {
        separate_declaration(&mut output, &mut declaration_count);
        output.begin(&connector.comments);
        if connector.visibility == VisibilitySyntax::Public {
            output.push_str("public ");
        }
        writeln!(
            output,
            "connector {} {{",
            connector.comments.named(&connector.name)
        )
        .expect("String write");
        match &connector.syntax {
            ConnectorSyntax::ScalarPhysical {
                across_name,
                across_type,
                through_name,
                through_type,
            } => {
                write!(output, "  across {across_name}: ").expect("String write");
                value_type::format_value_type(across_type, &mut output);
                write!(output, ";\n  through {through_name}: ").expect("String write");
                value_type::format_value_type(through_type, &mut output);
                output.push_str(";\n");
            }
            ConnectorSyntax::FieldPhysical {
                trace,
                flux,
                shape,
                frame,
                pairing,
            } => {
                output.push_str("  trace ");
                write!(output, "{}: ", trace.name).expect("String write");
                format_expression(&trace.dimension, 0, &mut output);
                output.push_str(";\n  flux ");
                write!(output, "{}: ", flux.name).expect("String write");
                format_expression(&flux.dimension, 0, &mut output);
                output.push_str(";\n  shape ");
                format_value_shape(shape, &mut output);
                output.push_str(";\n  frame ");
                output.push_str(match frame {
                    FrameSyntax::Invariant => "invariant",
                    FrameSyntax::Spatial => "spatial",
                });
                output.push_str(";\n  pairing ");
                output.push_str(match pairing {
                    BoundaryPairingSyntax::EuclideanBoundaryDuality => "euclidean_boundary_duality",
                });
                output.push_str(";\n  orientation parent_outward;\n");
            }
        }
        output.push_str("}\n");
        output.end();
    }
    for operator in &document.pure_operators {
        separate_declaration(&mut output, &mut declaration_count);
        format_pure_operator(operator, &mut output);
    }
    for component in &document.components {
        separate_declaration(&mut output, &mut declaration_count);
        format_component(component, &mut output);
    }
    for model in &document.models {
        separate_declaration(&mut output, &mut declaration_count);
        output.begin(&model.comments);
        if model.visibility == VisibilitySyntax::Public {
            output.push_str("public ");
        }
        write!(output, "model {}", model.comments.named(&model.name))
            .expect("String writes cannot fail");
        signature::format_signature(&model.signature, &mut output);
        output.push_str(" {\n");
        for item in &model.items {
            format_item(item, 2, &mut output);
        }
        for form in &model.formulations {
            formulation::format_formulation(form, 2, &mut output);
        }
        output.push_str("}\n");
        output.end();
    }
    output.end();
    output.finish()
}
fn format_pure_operator(
    declaration: &PureOperatorDecl,
    output: &mut crate::formatter::comments::Output,
) {
    output.begin(&declaration.comments);
    if declaration.visibility == VisibilitySyntax::Public {
        output.push_str("public ");
    }
    write!(
        output,
        "operator {}(",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    for (index, formal) in declaration.formals.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        output.begin(&formal.comments);
        write!(output, "input {}: ", formal.comments.named(&formal.name)).expect("String write");
        format_pure_value_class(&formal.value_class, output);
        output.end();
    }
    output.push_str("): ");
    format_pure_value_class(&declaration.result, output);
    output.push_str(" = ");
    format_expression(&declaration.body, 0, output);
    output.push_str(";\n");
    output.end();
}
fn format_pure_value_class(
    value_class: &PureValueClassSyntax,
    output: &mut crate::formatter::comments::Output,
) {
    match value_class {
        PureValueClassSyntax::Typed(value) => value_type::format_value_type(value, output),
        PureValueClassSyntax::Scalar => output.push_str("scalar"),
        PureValueClassSyntax::Spatial { rank } => {
            write!(output, "spatial[{}]", rank.value).expect("String write");
        }
    }
}

fn separate_declaration(
    output: &mut crate::formatter::comments::Output,
    declaration_count: &mut usize,
) {
    output.push_str(if *declaration_count == 0 { "" } else { "\n" });
    *declaration_count += 1;
}

fn format_component_item(
    item: &ComponentItem,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    output.begin(item.source_comments());
    match item {
        ComponentItem::Domain(declaration) => format_domain(declaration, indent, output),
        ComponentItem::Let(declaration) => format_let(declaration, indent, output),
        ComponentItem::Coordinate(declaration) => {
            compile_time::format_coordinate(declaration, indent, output)
        }
        ComponentItem::Parameter(declaration) => {
            write_indent(output, indent);
            if declaration.visibility == VisibilitySyntax::Public {
                output.push_str("public ");
            }
            write!(
                output,
                "parameter {}: ",
                declaration.comments.named(&declaration.name)
            )
            .expect("String write");
            value_type::format_value_type(&declaration.value_type, output);
            if let Some(default) = &declaration.default {
                output.push_str(" = ");
                format_expression(default, 0, output);
            }
            output.push_str(";\n");
        }
        ComponentItem::Port(declaration) => {
            write_indent(output, indent);
            if declaration.visibility == VisibilitySyntax::Public {
                output.push_str("public ");
            }
            write!(
                output,
                "port {}: ",
                declaration.comments.named(&declaration.name)
            )
            .expect("String write");
            format_port_syntax(&declaration.syntax, output);
            output.push_str(";\n");
        }
        ComponentItem::PortFamily(declaration) => {
            format_component_port_family(declaration, indent, output);
        }
        ComponentItem::Observable(declaration) => {
            compile_time::format_observable(declaration, indent, output)
        }
        ComponentItem::Field(declaration) => format_field(declaration, indent, output),
        ComponentItem::Initial(declaration) => format_initial(declaration, indent, output),
        ComponentItem::Event(declaration) => format_event(declaration, indent, output),
        ComponentItem::Clock(declaration) => format_clock(declaration, indent, output),
        ComponentItem::Relation(declaration) => format_relation(declaration, indent, output),
        ComponentItem::RelationFamily(declaration) => {
            format_relation_family(declaration, indent, output);
        }
        ComponentItem::Connection(declaration) => format_connection(declaration, indent, output),
        ComponentItem::BoundaryConnection(declaration) => {
            format_boundary_connection(declaration, indent, output);
        }
        ComponentItem::IndexSet(declaration) => format_index_set(declaration, indent, output),
        ComponentItem::Instance(declaration) => format_instance(declaration, indent, output),
    }
    output.end();
}

fn format_domain(declaration: &crate::DomainDecl, indent: usize, output: &mut comments::Output) {
    write_indent(output, indent);
    if matches!(declaration.syntax, DomainSyntax::Product { .. }) {
        write!(
            output,
            "support {}: ",
            declaration.comments.named(&declaration.name)
        )
        .expect("String write");
    } else {
        write!(
            output,
            "domain {} = ",
            declaration.comments.named(&declaration.name)
        )
        .expect("String write");
    }
    match &declaration.syntax {
        DomainSyntax::Product { factors } => {
            write!(output, "product({})", factors.join(", ")).expect("String write");
        }
        DomainSyntax::CartesianBox(bounds) => {
            output.push_str("box(");
            for (index, (lower, upper)) in bounds.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                format_cartesian_coordinate(lower, output);
                output.push_str(", ");
                format_cartesian_coordinate(upper, output);
            }
            output.push(')');
        }
        DomainSyntax::Boundary { parent, axis, side } => {
            write!(output, "boundary({parent}, axis = {axis}, side = ").expect("String write");
            output.push_str(match side {
                BoundarySideSyntax::Lower => "lower",
                BoundarySideSyntax::Upper => "upper",
            });
            output.push(')');
        }
        DomainSyntax::ScalarPhysical {
            across_name,
            across_type,
            through_name,
            through_type,
        } => format_scalar_physical(across_name, across_type, through_name, through_type, output),
    }
    output.push_str(";\n");
}

fn format_item(item: &Item, indent: usize, output: &mut crate::formatter::comments::Output) {
    output.begin(item.source_comments());
    match item {
        Item::Domain(declaration) => format_domain(declaration, indent, output),
        Item::Field(declaration) => format_field(declaration, indent, output),
        Item::Initial(declaration) => format_initial(declaration, indent, output),
        Item::Observable(declaration) => {
            compile_time::format_observable(declaration, indent, output)
        }
        Item::Parameter(declaration) => format_parameter(declaration, indent, output),
        Item::Let(declaration) => format_let(declaration, indent, output),
        Item::Coordinate(declaration) => {
            compile_time::format_coordinate(declaration, indent, output)
        }
        Item::Port(declaration) => {
            write_indent(output, indent);
            write!(
                output,
                "port {}: ",
                declaration.comments.named(&declaration.name)
            )
            .expect("String write");
            format_port_syntax(&declaration.syntax, output);
            output.push_str(";\n");
        }
        Item::Event(declaration) => format_event(declaration, indent, output),
        Item::Clock(declaration) => format_clock(declaration, indent, output),
        Item::Relation(declaration) => format_relation(declaration, indent, output),
        Item::RelationFamily(declaration) => format_relation_family(declaration, indent, output),
        Item::Connection(declaration) => format_connection(declaration, indent, output),
        Item::BoundaryConnection(declaration) => {
            format_boundary_connection(declaration, indent, output);
        }
        Item::IndexSet(declaration) => format_index_set(declaration, indent, output),
        Item::Instance(declaration) => format_instance(declaration, indent, output),
    }
    output.end();
}

fn format_field(
    declaration: &FieldDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    format_unknown_head(declaration, output);
    output.push_str(";\n");
}

fn format_unknown_head(declaration: &FieldDecl, output: &mut crate::formatter::comments::Output) {
    let role = match declaration.role {
        crate::ast::FieldRoleSyntax::Variable => "variable",
        crate::ast::FieldRoleSyntax::State => "state",
    };
    write!(
        output,
        "{role} {}: ",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    value_type::format_value_type(&declaration.value_type, output);
    if let Some(domain) = &declaration.domain {
        write!(output, " on {domain}").expect("String write");
    }
    if let crate::ast::ActivationSyntax::Named(clock) = &declaration.activation {
        write!(output, " at {clock}").expect("String write");
    }
}

fn format_initial(
    declaration: &crate::ast::InitialDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    output.push_str("initial {\n");
    for equation in declaration.equations() {
        write_indent(output, indent + 2);
        format_expression(equation.left(), 0, output);
        output.push_str(" = ");
        format_expression(equation.right(), 0, output);
        output.push_str(";\n");
    }
    write_indent(output, indent);
    output.push_str("}\n");
}

fn format_value_shape(shape: &ValueShapeSyntax, output: &mut crate::formatter::comments::Output) {
    match shape {
        ValueShapeSyntax::Scalar => output.push_str("[]"),
        ValueShapeSyntax::Exact(extents) => {
            output.push('[');
            for (index, extent) in extents.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                write!(output, "{extent}").expect("String write");
            }
            output.push(']');
        }
        ValueShapeSyntax::SpatialVector => output.push_str("spatial_vector"),
    }
}

fn format_port_syntax(syntax: &PortSyntax, output: &mut crate::formatter::comments::Output) {
    match syntax {
        PortSyntax::Signal {
            direction,
            value_type,
            domain,
            activation,
        } => {
            output.push_str(match direction {
                SignalDirectionSyntax::Input => "signal input ",
                SignalDirectionSyntax::Output => "signal output ",
            });
            value_type::format_value_type(value_type, output);
            if let Some(domain) = domain {
                write!(output, " on {domain}").expect("String write");
            }
            if let crate::ActivationSyntax::Named(clock) = activation {
                write!(output, " at {clock}").expect("String write");
            }
        }
        PortSyntax::ScalarPhysical { domain } => {
            write!(output, "{domain}").expect("String write");
        }
        PortSyntax::ScalarPhysicalConnector { connector } => {
            write!(output, "{connector}").expect("String write");
        }
        PortSyntax::FieldPhysical { connector, support } => {
            write!(output, "{connector} over {support}").expect("String write");
        }
    }
}

fn format_component_port_family(
    declaration: &ComponentPortFamilyDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    let port = &declaration.port;
    write_indent(output, indent);
    if port.visibility == VisibilitySyntax::Public {
        output.push_str("public ");
    }
    write!(output, "port {}", port.comments.named(&port.name)).expect("String write");
    format_boundary_family_binder(&declaration.binder, output);
    output.push_str(": ");
    format_port_syntax(&port.syntax, output);
    output.push_str(";\n");
}

fn format_boundary_family_binder(
    binder: &FamilyBinderSyntax,
    output: &mut crate::formatter::comments::Output,
) {
    write!(output, "[{} in {}]", binder.member, binder.set).expect("String write");
}

fn format_event(
    declaration: &crate::EventDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    write!(
        output,
        "event {} = crossing(",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    format_expression(declaration.guard(), 0, output);
    let direction = match declaration.direction() {
        eqiora_schema::kernel::EventDirection::Any => "any",
        eqiora_schema::kernel::EventDirection::Rising => "rising",
        eqiora_schema::kernel::EventDirection::Falling => "falling",
    };
    write!(output, ", direction = {direction}").expect("String write");
    if declaration.priority() != 0 {
        write!(output, ", priority = {}", declaration.priority()).expect("String write");
    }
    writeln!(output, ");").expect("String write");
}

fn format_clock(
    declaration: &ClockDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    write!(
        output,
        "clock {} = periodic(",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    format_expression(&declaration.period, 0, output);
    output.push_str(", phase = ");
    format_expression(&declaration.phase, 0, output);
    output.push_str(");\n");
}

fn format_connection(
    declaration: &ConnectionDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    match declaration.syntax {
        ConnectionSyntax::Signal => {
            output.push_str("connect ");
            if let Some(binder) = &declaration.binder {
                format_boundary_family_binder(binder, output);
                output.push(' ');
            }
            if let Some((source, targets)) = declaration.ports.split_first() {
                format_expression(source, 0, output);
                output.push_str(" -> ");
                format_endpoint_list(targets, output);
            }
        }
        ConnectionSyntax::Conserving => {
            output.push_str("connect ");
            if let Some(binder) = &declaration.binder {
                format_boundary_family_binder(binder, output);
                output.push(' ');
            }
            format_endpoint_list(&declaration.ports, output);
        }
        ConnectionSyntax::SpatialPeriodic => {
            output.push_str("connect periodic ");
            format_endpoint_list(&declaration.ports, output);
        }
    }
    output.push_str(";\n");
}

fn format_boundary_connection(
    declaration: &BoundaryConnectionDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    output.push_str(match declaration.syntax {
        ConnectionSyntax::Conserving => "connect",
        ConnectionSyntax::SpatialPeriodic => "connect periodic",
        ConnectionSyntax::Signal => "connect",
    });
    if let Some(binder) = &declaration.binder {
        output.push(' ');
        format_boundary_family_binder(binder, output);
    }
    output.push(' ');
    for (index, port) in declaration.ports.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        format_boundary_port_reference(port, output);
    }
    output.push_str(";\n");
}

fn format_boundary_port_reference(
    reference: &BoundaryPortReferenceSyntax,
    output: &mut crate::formatter::comments::Output,
) {
    write!(output, "{}", reference.port).expect("String write");
    if let Some(selector) = &reference.selector {
        format_boundary_port_selector(selector, output);
    }
}

fn format_instance(
    declaration: &InstanceDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    write!(
        output,
        "instance {}",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    if let Some(family) = &declaration.family {
        write!(output, "[{} in {}]", family.member(), family.set()).expect("String write");
    }
    write!(output, ": {}", declaration.definition).expect("String write");
    output.push('(');
    for (index, binding) in declaration.bindings.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        output.begin(&binding.comments);
        write!(output, "{} = ", binding.name).expect("String write");
        format_expression(&binding.value, 0, output);
        output.end();
    }
    output.push(')');
    output.push_str(";\n");
}

fn format_number(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests;

fn format_index_set(
    declaration: &crate::NamedDefinitionDecl,
    indent: usize,
    output: &mut comments::Output,
) {
    write_indent(output, indent);
    write!(
        output,
        "indexset {} = ",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    format_expression(declaration.value(), 0, output);
    output.push_str(";\n");
}

fn format_endpoint_list(endpoints: &[crate::Expr], output: &mut comments::Output) {
    for (index, endpoint) in endpoints.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        format_expression(endpoint, 0, output);
    }
}
