//! Read-only descriptors over declarations in the explicit imported Module.

use eqiora::language::{
    ComponentDecl, ConnectorSyntax, ExprKind, Module, PortSyntax, SignatureItem, SupportSlotSyntax,
    VisibilitySyntax,
};
use pyo3::prelude::*;

use super::expression::syntax_error;

pub(super) fn record(module: &Module, name: &str) -> PyResult<Vec<(String, String)>> {
    let mut found = module
        .document()
        .records()
        .iter()
        .filter(|value| value.name() == name);
    let value = found
        .next()
        .ok_or_else(|| syntax_error("import requires one public Record declaration"))?;
    if found.next().is_some() || value.visibility() != VisibilitySyntax::Public {
        return Err(syntax_error(
            "import requires one public Record declaration",
        ));
    }
    Ok(value
        .members()
        .iter()
        .map(|member| (member.name().to_owned(), member.value_type().to_source()))
        .collect())
}

pub(super) fn enumeration(module: &Module, name: &str) -> PyResult<Vec<String>> {
    let mut found = module
        .document()
        .enumerations()
        .iter()
        .filter(|value| value.name() == name);
    let value = found
        .next()
        .ok_or_else(|| syntax_error("import requires one public Enum declaration"))?;
    if found.next().is_some() || value.visibility() != VisibilitySyntax::Public {
        return Err(syntax_error("import requires one public Enum declaration"));
    }
    Ok(value.tags().iter().map(|tag| tag.to_string()).collect())
}

pub(super) fn finite_space(module: &Module, name: &str) -> PyResult<Vec<String>> {
    let mut found = module
        .document()
        .finite_spaces()
        .iter()
        .filter(|value| value.name() == name);
    let value = found
        .next()
        .ok_or_else(|| syntax_error("import requires one public finite Space declaration"))?;
    if found.next().is_some() || value.visibility() != VisibilitySyntax::Public {
        return Err(syntax_error(
            "import requires one public finite Space declaration",
        ));
    }
    let ExprKind::Call { callee, arguments } = value.value().kind() else {
        return Err(syntax_error("invalid finite Space declaration"));
    };
    if callee.as_str() != "orthonormal" {
        return Err(syntax_error("invalid finite Space declaration"));
    }
    arguments
        .expressions()
        .map(|label| match label.kind() {
            ExprKind::Name(name) => Ok(name.clone()),
            _ => Err(syntax_error("invalid finite Space basis label")),
        })
        .collect()
}

pub(super) fn connector(
    module: &Module,
    name: &str,
    public: bool,
) -> PyResult<(String, String, String)> {
    let mut found = module
        .document()
        .connectors()
        .iter()
        .filter(|value| value.name() == name);
    let value = found
        .next()
        .ok_or_else(|| syntax_error("import requires an exact Connector declaration"))?;
    if found.next().is_some() || (public && value.visibility() != VisibilitySyntax::Public) {
        return Err(syntax_error(
            "import requires one public Connector declaration",
        ));
    }
    Ok(match value.syntax() {
        ConnectorSyntax::ScalarPhysical {
            across_name,
            through_name,
            ..
        } => (
            "scalar".to_owned(),
            across_name.clone(),
            through_name.clone(),
        ),
        ConnectorSyntax::FieldPhysical { trace, flux, .. } => (
            "field".to_owned(),
            trace.name().to_owned(),
            flux.name().to_owned(),
        ),
        _ => return Err(syntax_error("unsupported Connector declaration")),
    })
}

fn component<'a>(module: &'a Module, name: &str) -> PyResult<&'a ComponentDecl> {
    let mut found = module
        .document()
        .components()
        .iter()
        .filter(|value| value.name() == name);
    let value = found
        .next()
        .ok_or_else(|| syntax_error("import requires one public Component"))?;
    if found.next().is_some() || value.visibility() != VisibilitySyntax::Public {
        return Err(syntax_error("import requires one public Component"));
    }
    Ok(value)
}

pub(super) type PortDescriptor = (String, String, Option<String>, Option<(String, String)>);

pub(super) fn ports(module: &Module, name: &str) -> PyResult<Vec<PortDescriptor>> {
    let mut ports = Vec::new();
    for item in component(module, name)?.signature() {
        let (port, binder) = match item {
            SignatureItem::Port(value) => (value, None),
            SignatureItem::PortFamily(value) => (
                value.port(),
                Some((
                    value.binder().member().to_owned(),
                    value.binder().set().to_string(),
                )),
            ),
            _ => continue,
        };
        let (connector, support) = match port.syntax() {
            PortSyntax::ScalarPhysicalConnector { connector } => (connector.to_string(), None),
            PortSyntax::FieldPhysical { connector, support } => {
                (connector.to_string(), Some(support.clone()))
            }
            _ => {
                return Err(syntax_error(
                    "imported physical port requires a nominal Connector",
                ));
            }
        };
        ports.push((port.name().to_owned(), connector, support, binder));
    }
    Ok(ports)
}

pub(super) type SupportDescriptor = (String, String, Option<String>, Option<usize>);

pub(super) fn supports(module: &Module, name: &str) -> PyResult<Vec<SupportDescriptor>> {
    component(module, name)?
        .signature()
        .iter()
        .filter_map(|item| {
            let SignatureItem::Support(value) = item else {
                return None;
            };
            Some(match value.syntax() {
                SupportSlotSyntax::Volume { ambient_dimension } => Ok((
                    value.name().to_owned(),
                    "volume".to_owned(),
                    None,
                    Some(*ambient_dimension),
                )),
                SupportSlotSyntax::Boundary { parent } => Ok((
                    value.name().to_owned(),
                    "boundary".to_owned(),
                    Some(parent.clone()),
                    None,
                )),
                SupportSlotSyntax::CompleteExterior { parent } => Ok((
                    value.name().to_owned(),
                    "complete_exterior".to_owned(),
                    Some(parent.clone()),
                    None,
                )),
                _ => Err(syntax_error("unsupported imported support declaration")),
            })
        })
        .collect()
}

/// Signature metadata only: bodies remain owned by the exact provider module.
pub(super) type OperatorDescriptor = (Vec<(String, String)>, String);

pub(super) fn operator(module: &Module, name: &str) -> PyResult<OperatorDescriptor> {
    use eqiora::language::PureValueClassSyntax;
    fn syntax(value: &PureValueClassSyntax) -> PyResult<String> {
        match value {
            PureValueClassSyntax::Typed(value) => Ok(value.to_source()),
            PureValueClassSyntax::Scalar => Ok("scalar".to_owned()),
            PureValueClassSyntax::Spatial { rank } => Ok(format!("spatial[{}]", rank.value())),
            _ => Err(syntax_error("unsupported imported operator value class")),
        }
    }
    let mut found = module
        .document()
        .pure_operators()
        .iter()
        .filter(|value| value.name() == name);
    let value = found
        .next()
        .ok_or_else(|| syntax_error("import requires one public operator"))?;
    if found.next().is_some() || value.visibility() != VisibilitySyntax::Public {
        return Err(syntax_error("import requires one public operator"));
    }
    Ok((
        value
            .formals()
            .iter()
            .map(|formal| Ok((formal.name().to_owned(), syntax(formal.value_class())?)))
            .collect::<PyResult<_>>()?,
        syntax(value.result())?,
    ))
}
