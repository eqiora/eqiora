use crate::ast::{ComponentItem, ConnectorSyntax, Document, DomainSyntax, Expr, Item, PortSyntax};

use super::SourceAstFactory;

impl SourceAstFactory {
    /// Rewrite every physical-dimension expression in one owned source unit.
    ///
    /// This compiler-facing transform deliberately excludes ordinary value
    /// expressions. It lets semantic dimension aliases erase once before the
    /// existing lowerers without exposing mutable declaration fields.
    #[doc(hidden)]
    pub fn rewrite_dimension_expressions(
        document: &mut Document,
        mut rewrite: impl FnMut(&Expr) -> Expr,
    ) {
        for declaration in &mut document.dimensions {
            declaration.value = rewrite(&declaration.value);
        }
        for record in &mut document.records {
            for member in &mut record.members {
                member.value_type.rewrite_dimension(&mut rewrite);
            }
        }
        for operator in &mut document.pure_operators {
            for formal in &mut operator.formals {
                if let crate::PureValueClassSyntax::Typed(value) = &mut formal.value_class {
                    value.rewrite_dimension(&mut rewrite);
                }
            }
            if let crate::PureValueClassSyntax::Typed(value) = &mut operator.result {
                value.rewrite_dimension(&mut rewrite);
            }
        }
        for declaration in &mut document.property_contracts {
            declaration.value_type.rewrite_dimension(&mut rewrite);
            for (_, input) in &mut declaration.inputs {
                input.rewrite_dimension(&mut rewrite);
            }
        }
        for declaration in &mut document.property_releases {
            if let Some(dimension) = &mut declaration.source_dimension {
                *dimension = rewrite(dimension);
            }
            if let crate::PropertySourceSyntax::Table(table) = &mut declaration.source_value {
                table.axis_dimension = rewrite(&table.axis_dimension);
                table.value_dimension = rewrite(&table.value_dimension);
            }
        }
        for connector in &mut document.connectors {
            rewrite_connector(&mut connector.syntax, &mut rewrite);
        }
        for component in &mut document.components {
            for formulation in &mut component.formulations {
                if let crate::FormulationBinding::WeakTests { tests } = &mut formulation.binding {
                    for (_, _, _, dimension) in tests {
                        *dimension = rewrite(dimension);
                    }
                }
            }
            for item in &mut component.signature {
                rewrite_signature(item, &mut rewrite);
            }
            for item in &mut component.items {
                rewrite_component_item(item, &mut rewrite);
            }
        }
        for model in &mut document.models {
            for item in &mut model.signature {
                rewrite_signature(item, &mut rewrite);
            }
            for item in &mut model.items {
                rewrite_item(item, &mut rewrite);
            }
        }
    }
}

fn rewrite_connector(syntax: &mut ConnectorSyntax, rewrite: &mut impl FnMut(&Expr) -> Expr) {
    match syntax {
        ConnectorSyntax::ScalarPhysical {
            across_type,
            through_type,
            ..
        } => {
            across_type.rewrite_dimension(rewrite);
            through_type.rewrite_dimension(rewrite);
        }
        ConnectorSyntax::FieldPhysical { trace, flux, .. } => {
            *trace.dimension = rewrite(&trace.dimension);
            *flux.dimension = rewrite(&flux.dimension);
        }
    }
}

fn rewrite_component_item(item: &mut ComponentItem, rewrite: &mut impl FnMut(&Expr) -> Expr) {
    match item {
        ComponentItem::Let(declaration) => {
            if let Some(value_type) = &mut declaration.value_type {
                value_type.rewrite_dimension(rewrite);
            }
        }
        ComponentItem::Parameter(declaration) => {
            declaration.value_type.rewrite_dimension(rewrite);
        }
        ComponentItem::Port(declaration) => rewrite_port(&mut declaration.syntax, rewrite),
        ComponentItem::PortFamily(declaration) => {
            rewrite_port(&mut declaration.port.syntax, rewrite);
        }
        ComponentItem::Observable(declaration) => {
            declaration.value_type.rewrite_dimension(rewrite);
        }
        ComponentItem::Field(declaration) => {
            declaration.value_type.rewrite_dimension(rewrite);
        }
        ComponentItem::Domain(_)
        | ComponentItem::IndexSet(_)
        | ComponentItem::Instance(_)
        | ComponentItem::Initial(_)
        | ComponentItem::Event(_)
        | ComponentItem::Clock(_)
        | ComponentItem::Relation(_)
        | ComponentItem::RelationFamily(_)
        | ComponentItem::Connection(_)
        | ComponentItem::BoundaryConnection(_) => {}
    }
}

fn rewrite_item(item: &mut Item, rewrite: &mut impl FnMut(&Expr) -> Expr) {
    match item {
        Item::Domain(declaration) => {
            if let DomainSyntax::ScalarPhysical {
                across_type,
                through_type,
                ..
            } = &mut declaration.syntax
            {
                across_type.rewrite_dimension(rewrite);
                through_type.rewrite_dimension(rewrite);
            }
        }
        Item::Observable(declaration) => {
            declaration.value_type.rewrite_dimension(rewrite);
        }
        Item::Field(declaration) => {
            declaration.value_type.rewrite_dimension(rewrite);
        }
        Item::Parameter(declaration) => {
            declaration.value_type.rewrite_dimension(rewrite);
        }
        Item::Let(declaration) => {
            if let Some(value_type) = &mut declaration.value_type {
                value_type.rewrite_dimension(rewrite);
            }
        }
        Item::Port(declaration) => rewrite_port(&mut declaration.syntax, rewrite),
        Item::IndexSet(_)
        | Item::Initial(_)
        | Item::Event(_)
        | Item::Clock(_)
        | Item::Relation(_)
        | Item::RelationFamily(_)
        | Item::Connection(_)
        | Item::BoundaryConnection(_)
        | Item::Instance(_) => {}
    }
}

fn rewrite_port(syntax: &mut PortSyntax, rewrite: &mut impl FnMut(&Expr) -> Expr) {
    if let PortSyntax::Signal { value_type, .. } = syntax {
        value_type.rewrite_dimension(rewrite);
    }
}

fn rewrite_signature(item: &mut crate::SignatureItem, rewrite: &mut impl FnMut(&Expr) -> Expr) {
    match item {
        crate::SignatureItem::Parameter(value) => {
            value.value_type.rewrite_dimension(rewrite);
        }
        crate::SignatureItem::Input(value)
        | crate::SignatureItem::Output(value)
        | crate::SignatureItem::Field(value) => {
            value.value_type.rewrite_dimension(rewrite);
        }
        crate::SignatureItem::Port(value) => rewrite_port(&mut value.syntax, rewrite),
        crate::SignatureItem::PortFamily(value) => rewrite_port(&mut value.port.syntax, rewrite),
        crate::SignatureItem::Support(value) => {
            if let crate::SupportSlotSyntax::Interval { dimension } = &mut value.syntax {
                *dimension = rewrite(dimension);
            }
        }
        crate::SignatureItem::Clock(_) | crate::SignatureItem::Property(_) => {}
    }
}
