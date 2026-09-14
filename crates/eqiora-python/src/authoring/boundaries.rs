//! Exact field interfaces and finite boundary families reuse source AST owners.

use eqiora::language::{
    ActivationSyntax, BoundaryPairingSyntax, ComponentItem, ConnectorDecl, ConnectorSyntax,
    ExprKind, FrameSyntax, PortSyntax, RelationConditionKind, SignatureItem,
    SourceAstFactory as Ast, SupportSlotSyntax, TextRange, ValueShapeSyntax, ValueTypeSyntax,
    ValueTypeSyntaxKind, VisibilitySyntax,
};
use pyo3::prelude::*;

use super::declaration::{Declaration, PyAstDeclaration};
use super::expression::{PyAstExpression, path, syntax_error};
use crate::modeling::PyValueType;

pub(super) type FieldConnectorInput<'py> = (
    String,
    String,
    PyRef<'py, PyValueType>,
    String,
    PyRef<'py, PyValueType>,
    bool,
    u32,
);

fn range(ordinal: u32) -> TextRange {
    TextRange::new(ordinal, ordinal.saturating_add(1))
}

fn shape(kind: &PyValueType) -> PyResult<(ValueShapeSyntax, FrameSyntax)> {
    use eqiora::{ScalarDomain, ValueFrame};
    if kind.value.scalar_domain() != ScalarDomain::Real || kind.value.array_rank() != 0 {
        return Err(syntax_error(
            "field connector quantities require real types",
        ));
    }
    let shape = if kind.value.shape().is_scalar() {
        ValueShapeSyntax::Scalar
    } else {
        ValueShapeSyntax::Exact(
            kind.value
                .shape()
                .extents()
                .iter()
                .map(|extent| extent.get())
                .collect(),
        )
    };
    let frame = match kind.value.frame() {
        ValueFrame::Invariant => FrameSyntax::Invariant,
        ValueFrame::SpatialCartesian => FrameSyntax::Spatial,
    };
    Ok((shape, frame))
}

fn dimension(kind: &PyValueType) -> PyResult<eqiora::language::Expr> {
    let scalar = eqiora::ValueType::scalar(eqiora::ScalarDomain::Real, kind.value.dimension())
        .map_err(syntax_error)?;
    let syntax = ValueTypeSyntax::from_checked(&scalar, |_| None).map_err(syntax_error)?;
    if let ValueTypeSyntaxKind::Named(name) = syntax.kind() {
        let expression = if name.is_qualified() {
            ExprKind::Path(name.clone())
        } else {
            ExprKind::Name(name.to_string())
        };
        Ast::expression(expression, syntax.range()).map_err(syntax_error)
    } else {
        syntax
            .dimension()
            .cloned()
            .ok_or_else(|| syntax_error("field quantity needs a physical dimension"))
    }
}

pub(super) fn connectors(inputs: Vec<FieldConnectorInput<'_>>) -> PyResult<Vec<ConnectorDecl>> {
    inputs
        .into_iter()
        .map(
            |(name, trace_name, trace, flux_name, flux, spatial_vector, ordinal)| {
                let (mut shape, mut frame) = shape(&trace)?;
                if (shape.clone(), frame) != self::shape(&flux)? {
                    return Err(syntax_error(
                        "field connector quantities require identical shape and frame",
                    ));
                }
                if spatial_vector {
                    if shape != ValueShapeSyntax::Scalar {
                        return Err(syntax_error(
                            "generic spatial_vector requires scalar quantity types",
                        ));
                    }
                    shape = ValueShapeSyntax::SpatialVector;
                    frame = FrameSyntax::Spatial;
                }
                let quantity = |name, kind: &PyValueType| {
                    Ast::connector_quantity(name, dimension(kind)?).map_err(syntax_error)
                };
                Ast::connector(
                    VisibilitySyntax::Public,
                    name,
                    ConnectorSyntax::FieldPhysical {
                        trace: quantity(trace_name, &trace)?,
                        flux: quantity(flux_name, &flux)?,
                        shape,
                        frame,
                        pairing: BoundaryPairingSyntax::EuclideanBoundaryDuality,
                    },
                    range(ordinal),
                )
                .map_err(syntax_error)
            },
        )
        .collect()
}

pub(super) fn exterior(name: String, parent: String, ordinal: u32) -> PyResult<PyAstDeclaration> {
    Ok(PyAstDeclaration {
        value: Declaration::Signature(SignatureItem::Support(
            Ast::support_slot(
                VisibilitySyntax::Public,
                name,
                SupportSlotSyntax::CompleteExterior { parent },
                range(ordinal),
            )
            .map_err(syntax_error)?,
        )),
    })
}

pub(super) fn port(
    name: String,
    connector: &str,
    support: String,
    set: Option<String>,
    ordinal: u32,
) -> PyResult<PyAstDeclaration> {
    let range = range(ordinal);
    let binder = set
        .map(|set| Ast::boundary_family_binder(support.clone(), set, range).map_err(syntax_error))
        .transpose()?;
    let port = Ast::component_port(
        VisibilitySyntax::Public,
        name,
        PortSyntax::FieldPhysical {
            connector: path(connector)?,
            support,
        },
        range,
    )
    .map_err(syntax_error)?;
    Ok(PyAstDeclaration {
        value: Declaration::Signature(match binder {
            Some(binder) => SignatureItem::PortFamily(
                Ast::component_port_family(port, binder).map_err(syntax_error)?,
            ),
            None => SignatureItem::Port(port),
        }),
    })
}

pub(super) type Endpoint = (String, Option<(String, String)>);

pub(super) fn connection(
    ports: Vec<Endpoint>,
    binder: Option<(String, String)>,
    periodic: bool,
    ordinal: u32,
) -> PyResult<PyAstDeclaration> {
    if ports.len() > 256 {
        return Err(syntax_error("connection exceeds 256 endpoints"));
    }
    let range = range(ordinal);
    let ports = ports
        .into_iter()
        .map(|(port, selector)| {
            let selector = selector
                .map(|(member, target)| {
                    Ast::boundary_port_selector(member, target, range).map_err(syntax_error)
                })
                .transpose()?;
            Ast::boundary_port_reference(path(&port)?, selector).map_err(syntax_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let value = if periodic {
        if binder.is_some() {
            return Err(syntax_error(
                "a spatial-periodic connection cannot carry a family binder",
            ));
        }
        Ast::spatial_periodic_boundary_connection(ports, range)
    } else {
        let binder = binder
            .map(|(member, set)| {
                Ast::boundary_family_binder(member, set, range).map_err(syntax_error)
            })
            .transpose()?;
        Ast::boundary_connection(binder, ports, range)
    }
    .map_err(syntax_error)?;
    Ok(PyAstDeclaration {
        value: Declaration::Item(ComponentItem::BoundaryConnection(value)),
    })
}

pub(super) fn relation(
    name: String,
    member: String,
    set: String,
    conditions: Vec<(
        String,
        PyRef<'_, PyAstExpression>,
        PyRef<'_, PyAstExpression>,
    )>,
    ordinal: u32,
) -> PyResult<PyAstDeclaration> {
    if conditions.len() > 256 {
        return Err(syntax_error("relation exceeds the 256-condition limit"));
    }
    let range = range(ordinal);
    let binder = Ast::boundary_family_binder(member.clone(), set, range).map_err(syntax_error)?;
    let conditions = conditions
        .into_iter()
        .map(|(kind, left, right)| {
            let kind = match kind.as_str() {
                "equality" => RelationConditionKind::Equality,
                "inequality" => RelationConditionKind::Inequality,
                "complementarity" => RelationConditionKind::Complementarity,
                _ => return Err(syntax_error("unknown mathematical condition kind")),
            };
            Ast::condition(kind, left.value.clone(), right.value.clone(), range)
                .map_err(syntax_error)
        })
        .collect::<PyResult<_>>()?;
    let relation = Ast::relation(
        name,
        ActivationSyntax::Continuous,
        Some(member),
        conditions,
        range,
    )
    .map_err(syntax_error)?;
    Ok(PyAstDeclaration {
        value: Declaration::Item(ComponentItem::RelationFamily(
            Ast::relation_family(relation, binder).map_err(syntax_error)?,
        )),
    })
}
