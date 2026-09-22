//! Exact endpoint contracts and directed connection validation.
use super::*;

pub(in crate::hierarchy::body_check) fn validate_connection(
    scope: &DefinitionScope<'_, '_>,
    declaration: &ConnectionDecl,
    connected_ports: &mut BTreeSet<Vec<String>>,
    connection_limits: ConnectionSetLimits,
) -> Result<Option<PhysicalConnectionFragment>, Diagnostic> {
    let paths = declaration
        .port_expressions()
        .iter()
        .map(|expression| {
            if matches!(expression.kind(), eqiora_lang::ExprKind::Member { .. }) {
                scope.indexed_member(expression).map(|(path, _)| path)
            } else {
                crate::source_endpoints::path(scope.file, expression)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut keys = Vec::with_capacity(paths.len());
    let mut contracts = Vec::with_capacity(paths.len());
    for (path, expression) in paths.iter().zip(declaration.port_expressions()) {
        keys.push(
            if matches!(expression.kind(), eqiora_lang::ExprKind::Member { .. }) {
                scope.indexed_member(expression)?.1
            } else {
                path.segments().map(str::to_owned).collect::<Vec<_>>()
            },
        );
        let contract = if matches!(expression.kind(), eqiora_lang::ExprKind::Member { .. }) {
            let key = &keys[keys.len() - 1];
            match scope.resolve_symbol_at(path, key[1].parse().ok())? {
                SymbolContract::Port(contract) => contract,
                _ => return Err(scope.wrong_local_kind(expression.range(), path.as_str(), "Port")),
            }
        } else {
            scope.resolve_port(path)?
        };
        contracts.push(contract.for_connection(
            declaration.syntax(),
            scope.exposed_signals.contains(path.as_str()),
        ));
    }
    validate_resolved_connection(
        declaration,
        &keys,
        &contracts,
        connected_ports,
        connection_limits,
        scope.file,
    )
}

pub(in crate::hierarchy::body_check) fn validate_resolved_connection(
    declaration: &ConnectionDecl,
    keys: &[Vec<String>],
    contracts: &[PortContract],
    connected_ports: &mut BTreeSet<Vec<String>>,
    connection_limits: ConnectionSetLimits,
    file: &str,
) -> Result<Option<PhysicalConnectionFragment>, Diagnostic> {
    if keys.iter().collect::<BTreeSet<_>>().len() != keys.len() {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            declaration.range(),
            "Connection repeats the same Port",
        ));
    }
    let scalar_physical = matches!(contracts.first(), Some(PortContract::Physical { .. }))
        && contracts
            .iter()
            .all(|contract| matches!(contract, PortContract::Physical { .. }));
    if scalar_physical {
        validate_connection_contract(declaration, contracts, file)?;
        let endpoints = keys
            .iter()
            .map(|key| {
                ResolvedPhysicalEndpoint::from_key(key).ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        declaration.range(),
                        "physical connection requires an exact static indexed occurrence",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        return ConnectionFragment::try_new(endpoints, connection_limits)
            .map(Some)
            .map_err(|error| connection_fragment_error(file, declaration.range(), error));
    }
    let boundary_physical = matches!(
        contracts.first(),
        Some(PortContract::BoundaryPhysical { .. })
    ) && contracts
        .iter()
        .all(|contract| matches!(contract, PortContract::BoundaryPhysical { .. }));
    if boundary_physical {
        if declaration.syntax() != ConnectionSyntax::Conserving {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                "field-physical Ports require a conserving Connection",
            ));
        }
        let Some(PortContract::BoundaryPhysical { nominal, .. }) = contracts.first() else {
            unreachable!("boundary-physical family was established");
        };
        if contracts.iter().skip(1).any(|contract| {
            !matches!(contract, PortContract::BoundaryPhysical { nominal: candidate, .. } if candidate == nominal)
        }) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                "field-physical Connection requires the exact same specialized Connector",
            ));
        }
        let endpoints = keys
            .iter()
            .map(|key| {
                ResolvedPhysicalEndpoint::from_key(key).ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        declaration.range(),
                        "physical connection requires an exact static indexed occurrence",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        return ConnectionFragment::try_new(endpoints, connection_limits)
            .map(Some)
            .map_err(|error| connection_fragment_error(file, declaration.range(), error));
    }
    let members = if declaration.syntax() == ConnectionSyntax::Signal {
        &keys[1..]
    } else {
        keys
    };
    if let Some(key) = members
        .iter()
        .find(|key| connected_ports.contains(key.as_slice()))
    {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            declaration.range(),
            format!(
                "Port `{}` already belongs to another Connection",
                key.join(".")
            ),
        ));
    }
    validate_connection_contract(declaration, contracts, file)?;
    connected_ports.extend(members.iter().cloned());
    Ok(None)
}
