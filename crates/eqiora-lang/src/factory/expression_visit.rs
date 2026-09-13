//! Scoped traversal of authored value expressions for lexical resolution.
use crate::{ComponentItem, Document, Expr, ExprKind, Item, SignatureItem};

impl super::SourceAstFactory {
    /// Visit value expressions in lexical container scope, after their children.
    #[doc(hidden)]
    pub fn visit_expressions(
        document: &mut Document,
        mut visit: impl FnMut(Option<&str>, &mut Expr),
    ) {
        for operator in &mut document.pure_operators {
            expression(Some(operator.name.as_str()), &mut operator.body, &mut visit);
        }
        for release in &mut document.property_releases {
            match &mut release.source_value {
                crate::PropertySourceSyntax::Expression(value) => {
                    expression(None, value, &mut visit)
                }
                crate::PropertySourceSyntax::Table(table) => {
                    for endpoint in &mut table.validity {
                        expression(None, endpoint, &mut visit);
                    }
                }
            }
            if let Some(validity) = &mut release.validity {
                expression(None, validity, &mut visit);
            }
        }
        for component in &mut document.components {
            let scope = Some(component.name.as_str());
            signature(&mut component.signature, scope, &mut visit);
            for item in &mut component.items {
                match item {
                    ComponentItem::Parameter(value) => {
                        if let Some(value) = &mut value.default {
                            expression(scope, value, &mut visit);
                        }
                    }
                    ComponentItem::Let(value) | ComponentItem::IndexSet(value) => {
                        expression(scope, &mut value.value, &mut visit)
                    }
                    ComponentItem::Event(value) => expression(scope, &mut value.guard, &mut visit),
                    ComponentItem::Observable(value) => {
                        expression(scope, &mut value.value, &mut visit)
                    }
                    ComponentItem::Relation(value) => relation(scope, value, &mut visit),
                    ComponentItem::Initial(value) => {
                        equations(scope, &mut value.equations, &mut visit)
                    }
                    ComponentItem::RelationFamily(value) => {
                        relation(scope, &mut value.relation, &mut visit)
                    }
                    ComponentItem::Instance(value) => {
                        for binding in &mut value.bindings {
                            expression(scope, &mut binding.value, &mut visit);
                        }
                    }
                    ComponentItem::Connection(value) => {
                        for endpoint in &mut value.ports {
                            expression(scope, endpoint, &mut visit);
                        }
                    }
                    _ => {}
                }
            }
        }
        for model in &mut document.models {
            let scope = Some(model.name.as_str());
            signature(&mut model.signature, scope, &mut visit);
            for item in &mut model.items {
                match item {
                    Item::Parameter(value) => expression(scope, &mut value.value, &mut visit),
                    Item::Let(value) | Item::IndexSet(value) => {
                        expression(scope, &mut value.value, &mut visit)
                    }
                    Item::Event(value) => expression(scope, &mut value.guard, &mut visit),
                    Item::Observable(value) => expression(scope, &mut value.value, &mut visit),
                    Item::Relation(value) => relation(scope, value, &mut visit),
                    Item::RelationFamily(value) => relation(scope, &mut value.relation, &mut visit),
                    Item::Initial(value) => equations(scope, &mut value.equations, &mut visit),
                    Item::Instance(value) => {
                        for binding in &mut value.bindings {
                            expression(scope, &mut binding.value, &mut visit);
                        }
                    }
                    Item::Connection(value) => {
                        for endpoint in &mut value.ports {
                            expression(scope, endpoint, &mut visit);
                        }
                    }
                    _ => {}
                }
            }
        }
        // Array extents are value expressions in the declaration's lexical scope,
        // not dimension syntax. Reuse the existing typed-declaration walk so enum
        // cases and nominal members receive the same resolution as initializers.
        Self::visit_value_types(document, |scope, syntax| {
            if let crate::ValueTypeSyntaxKind::Array { extent, .. } = syntax.kind.as_mut() {
                expression(scope, extent, &mut visit);
            }
        });
    }
}
fn signature(
    values: &mut [SignatureItem],
    scope: Option<&str>,
    visit: &mut impl FnMut(Option<&str>, &mut Expr),
) {
    for item in values {
        if let SignatureItem::Parameter(parameter) = item
            && let Some(value) = &mut parameter.default
        {
            expression(scope, value, visit);
        }
    }
}
fn relation(
    scope: Option<&str>,
    value: &mut crate::RelationDecl,
    visit: &mut impl FnMut(Option<&str>, &mut Expr),
) {
    match &mut value.body {
        crate::RelationBody::Conditions(values) => equations(scope, values, visit),
        crate::RelationBody::Conservation(law) => {
            expression(scope, &mut law.flux, visit);
            expression(scope, &mut law.source, visit);
        }
    }
}
fn equations(
    scope: Option<&str>,
    values: &mut [crate::RelationCondition],
    visit: &mut impl FnMut(Option<&str>, &mut Expr),
) {
    for equation in values {
        expression(scope, &mut equation.left, visit);
        expression(scope, &mut equation.right, visit);
    }
}
pub(crate) fn expression(
    scope: Option<&str>,
    value: &mut Expr,
    visit: &mut impl FnMut(Option<&str>, &mut Expr),
) {
    match &mut value.kind {
        ExprKind::Case { value, arms } => {
            expression(scope, value, visit);
            for arm in arms {
                expression(scope, &mut arm.value, visit);
            }
        }
        ExprKind::Select {
            condition,
            then_value,
            else_value,
        } => {
            expression(scope, condition, visit);
            expression(scope, then_value, visit);
            expression(scope, else_value, visit);
        }
        ExprKind::Partial { value, .. }
        | ExprKind::Unary { value, .. }
        | ExprKind::Member { value, .. }
        | ExprKind::Reduction { value, .. } => expression(scope, value, visit),
        ExprKind::Binary { left, right, .. } => {
            expression(scope, left, visit);
            expression(scope, right, visit);
        }
        ExprKind::Index { value, index } => {
            expression(scope, value, visit);
            expression(scope, index, visit);
        }
        ExprKind::Slice {
            value,
            lower,
            upper,
        } => {
            expression(scope, value, visit);
            expression(scope, lower, visit);
            expression(scope, upper, visit);
        }
        ExprKind::Call { arguments, .. } => match arguments {
            crate::CallArguments::Positional(values) => {
                for value in values {
                    expression(scope, value, visit);
                }
            }
            crate::CallArguments::Named(bindings) => {
                for binding in bindings {
                    expression(scope, &mut binding.value, visit);
                }
            }
        },
        ExprKind::Array(values) => {
            for value in values {
                expression(scope, value, visit);
            }
        }
        _ => {}
    }
    visit(scope, value);
}
