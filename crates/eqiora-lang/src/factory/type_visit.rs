//! One traversal of authored complete type owners for lexical nominal resolution.
use crate::{
    ComponentItem, ConnectorSyntax, Document, Item, PortSyntax, SignatureItem, ValueTypeSyntax,
};

impl super::SourceAstFactory {
    /// Visit every complete type with its enclosing container, preserving syntax and ranges.
    #[doc(hidden)]
    pub fn visit_value_types(
        document: &mut Document,
        mut visit: impl FnMut(Option<&str>, &mut ValueTypeSyntax),
    ) {
        for record in &mut document.records {
            for member in &mut record.members {
                visit(None, &mut member.value_type);
            }
        }
        for contract in &mut document.property_contracts {
            visit(None, &mut contract.value_type);
        }
        for connector in &mut document.connectors {
            if let ConnectorSyntax::ScalarPhysical {
                across_type,
                through_type,
                ..
            } = &mut connector.syntax
            {
                visit(None, across_type);
                visit(None, through_type);
            }
        }
        for component in &mut document.components {
            let scope = Some(component.name.as_str());
            for item in &mut component.signature {
                signature(item, scope, &mut visit);
            }
            for item in &mut component.items {
                match item {
                    ComponentItem::Parameter(value) => visit(scope, &mut value.value_type),
                    ComponentItem::Let(value) => {
                        if let Some(value_type) = &mut value.value_type {
                            visit(scope, value_type);
                        }
                    }
                    ComponentItem::Field(value) => visit(scope, &mut value.value_type),
                    ComponentItem::Observable(value) => visit(scope, &mut value.value_type),
                    ComponentItem::Port(value) => port(&mut value.syntax, scope, &mut visit),
                    ComponentItem::PortFamily(value) => {
                        port(&mut value.port.syntax, scope, &mut visit)
                    }
                    _ => {}
                }
            }
        }
        for model in &mut document.models {
            let scope = Some(model.name.as_str());
            for item in &mut model.signature {
                signature(item, scope, &mut visit);
            }
            for item in &mut model.items {
                match item {
                    Item::Parameter(value) => visit(scope, &mut value.value_type),
                    Item::Let(value) => {
                        if let Some(value_type) = &mut value.value_type {
                            visit(scope, value_type);
                        }
                    }
                    Item::Field(value) => visit(scope, &mut value.value_type),
                    Item::Observable(value) => visit(scope, &mut value.value_type),
                    Item::Port(value) => port(&mut value.syntax, scope, &mut visit),
                    Item::Domain(value) => {
                        if let crate::DomainSyntax::ScalarPhysical {
                            across_type,
                            through_type,
                            ..
                        } = &mut value.syntax
                        {
                            visit(scope, across_type);
                            visit(scope, through_type);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}
fn signature(
    item: &mut SignatureItem,
    scope: Option<&str>,
    visit: &mut impl FnMut(Option<&str>, &mut ValueTypeSyntax),
) {
    match item {
        SignatureItem::Parameter(value) => visit(scope, &mut value.value_type),
        SignatureItem::Field(value)
        | SignatureItem::Input(value)
        | SignatureItem::Output(value) => visit(scope, &mut value.value_type),
        SignatureItem::Port(value) => port(&mut value.syntax, scope, visit),
        SignatureItem::PortFamily(value) => port(&mut value.port.syntax, scope, visit),
        _ => {}
    }
}
fn port(
    port: &mut PortSyntax,
    scope: Option<&str>,
    visit: &mut impl FnMut(Option<&str>, &mut ValueTypeSyntax),
) {
    if let PortSyntax::Signal { value_type, .. } = port {
        visit(scope, value_type);
    }
}

impl super::SourceAstFactory {
    /// Visit values with their authored complete initializer type, without inferring context.
    #[doc(hidden)]
    pub fn visit_typed_initializers(
        document: &mut Document,
        mut visit: impl FnMut(&ValueTypeSyntax, &mut crate::Expr),
    ) {
        let signature =
            |items: &mut [SignatureItem],
             visit: &mut dyn FnMut(&ValueTypeSyntax, &mut crate::Expr)| {
                for item in items {
                    if let SignatureItem::Parameter(parameter) = item
                        && let Some(value) = &mut parameter.default
                    {
                        visit(&parameter.value_type, value);
                    }
                }
            };
        for model in &mut document.models {
            signature(&mut model.signature, &mut visit);
            for item in &mut model.items {
                match item {
                    Item::Parameter(value) => visit(&value.value_type, &mut value.value),
                    Item::Let(value) => {
                        if let Some(ty) = &value.value_type {
                            visit(ty, &mut value.value);
                        }
                    }
                    Item::Observable(value) => visit(&value.value_type, &mut value.value),
                    _ => {}
                }
            }
        }
        for component in &mut document.components {
            signature(&mut component.signature, &mut visit);
            for item in &mut component.items {
                match item {
                    ComponentItem::Parameter(value) => {
                        if let Some(default) = &mut value.default {
                            visit(&value.value_type, default);
                        }
                    }
                    ComponentItem::Let(value) => {
                        if let Some(ty) = &value.value_type {
                            visit(ty, &mut value.value);
                        }
                    }
                    ComponentItem::Observable(value) => visit(&value.value_type, &mut value.value),
                    _ => {}
                }
            }
        }
    }
}
