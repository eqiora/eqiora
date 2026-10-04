//! Exact symbol dependencies and mathematical types in one admitted Model.
use super::*;

pub(super) fn symbol_id(symbol: SymbolRef) -> Option<RawId> {
    match symbol {
        SymbolRef::Field(id)
        | SymbolRef::Derivative(id)
        | SymbolRef::Pre(id)
        | SymbolRef::Next(id) => Some(id.erase()),
        SymbolRef::Parameter(id) => Some(id.erase()),
        SymbolRef::Observable(id) => Some(id.erase()),
        SymbolRef::Port(id)
        | SymbolRef::Across(id)
        | SymbolRef::Through(id)
        | SymbolRef::PortTrace(id)
        | SymbolRef::PortFlux(id) => Some(id.erase()),
        SymbolRef::Coordinate { support, .. } => Some(support.erase()),
        SymbolRef::Time => None,
        _ => None,
    }
}

pub(super) fn symbol_type(
    symbol: SymbolRef,
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    spatial_supports: &BTreeMap<RawId, SpatialSupport<RawId>>,
) -> Result<ExpressionType<RawId>, SymbolTypeError> {
    match symbol {
        SymbolRef::Field(id) | SymbolRef::Pre(id) | SymbolRef::Next(id) => {
            match nodes.get(&id.erase()) {
                Some(KernelNode::Field(field)) => {
                    if matches!(symbol, SymbolRef::Pre(_) | SymbolRef::Next(_))
                        && (field.role() != eqiora_schema::kernel::FieldRole::State)
                    {
                        return Err(SymbolTypeError::WrongFieldRole);
                    }
                    Ok(ExpressionType::new(
                        field.value_type().clone(),
                        field_support(id.erase(), edges, spatial_supports),
                    ))
                }
                _ => Err(SymbolTypeError::Missing),
            }
        }
        SymbolRef::Derivative(id) => match nodes.get(&id.erase()) {
            Some(KernelNode::Field(field))
                if field.role() != eqiora_schema::kernel::FieldRole::State
                    || !edge_targets(edges, id.erase(), EdgeKind::ClockedBy).is_empty() =>
            {
                Err(SymbolTypeError::WrongFieldRole)
            }
            Some(KernelNode::Field(field)) => typing::time_derivative(&ExpressionType::new(
                field.value_type().clone(),
                field_support(id.erase(), edges, spatial_supports),
            ))
            .map_err(SymbolTypeError::Typing),
            _ => Err(SymbolTypeError::Missing),
        },
        SymbolRef::Observable(id) => match nodes.get(&id.erase()) {
            Some(KernelNode::Observable(value)) => Ok(ExpressionType::new(
                value.value_type().clone(),
                field_support(id.erase(), edges, spatial_supports),
            )),
            _ => Err(SymbolTypeError::Missing),
        },
        SymbolRef::Parameter(id) => match nodes.get(&id.erase()) {
            Some(KernelNode::Parameter(parameter)) => {
                Ok(ExpressionType::new(parameter.value_type().clone(), None))
            }
            _ => Err(SymbolTypeError::Missing),
        },
        SymbolRef::Port(id) => match nodes.get(&id.erase()) {
            Some(KernelNode::Port(port)) => port
                .signal_contract()
                .map(|(_, value_type)| {
                    ExpressionType::new(
                        value_type.clone(),
                        field_support(id.erase(), edges, spatial_supports),
                    )
                })
                .ok_or(SymbolTypeError::WrongPortContract),
            _ => Err(SymbolTypeError::Missing),
        },
        SymbolRef::Across(id) | SymbolRef::Through(id) => {
            let Some(KernelNode::Port(port)) = nodes.get(&id.erase()) else {
                return Err(SymbolTypeError::Missing);
            };
            let Some(domain) = port.physical_domain() else {
                return Err(SymbolTypeError::WrongPortContract);
            };
            let Some(KernelNode::Domain(domain)) = nodes.get(&domain.erase()) else {
                return Err(SymbolTypeError::Missing);
            };
            let DomainKind::ScalarPhysical {
                across_type,
                through_type,
            } = domain.kind()
            else {
                return Err(SymbolTypeError::WrongPortContract);
            };
            let value_type = if matches!(symbol, SymbolRef::Across(_)) {
                across_type
            } else {
                through_type
            };
            Ok(ExpressionType::new(value_type.clone(), None))
        }
        SymbolRef::PortTrace(id) | SymbolRef::PortFlux(id) => {
            let Some(KernelNode::Port(port)) = nodes.get(&id.erase()) else {
                return Err(SymbolTypeError::Missing);
            };
            let Some((connector, boundary)) = port.boundary_physical_contract() else {
                return Err(SymbolTypeError::WrongPortContract);
            };
            let Some(KernelNode::Domain(connector)) = nodes.get(&connector.erase()) else {
                return Err(SymbolTypeError::Missing);
            };
            let DomainKind::BoundaryPhysical { connector } = connector.kind() else {
                return Err(SymbolTypeError::WrongPortContract);
            };
            let Some(support) = spatial_supports.get(&boundary.erase()).cloned() else {
                return Err(SymbolTypeError::WrongPortContract);
            };
            let value_type = if matches!(symbol, SymbolRef::PortTrace(_)) {
                connector.trace_type()
            } else {
                connector.flux_type()
            };
            Ok(ExpressionType::new(value_type.clone(), Some(support)))
        }
        SymbolRef::Coordinate {
            support,
            factor,
            axis,
        } => {
            let support = spatial_supports
                .get(&support.erase())
                .ok_or(SymbolTypeError::Missing)?;
            eqiora_schema::kernel::typing::coordinate(&factor.erase(), axis, Some(support))
                .map_err(SymbolTypeError::Typing)
        }
        SymbolRef::Time => Ok(ExpressionType::scalar(
            DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension"),
            None,
        )),
        _ => Err(SymbolTypeError::Missing),
    }
}

pub(super) enum SymbolTypeError {
    Missing,
    Typing(TypeViolation<RawId>),
    WrongPortContract,
    WrongFieldRole,
}
