//! Resolve owned scalar and causal Port declarations into checked contracts.
use super::*;

pub(in crate::hierarchy) fn field_expression_type<I>(
    file: &str,
    declaration: &FieldDecl,
    support: Option<SpatialSupport<I>>,
    values: &crate::hierarchy::parameters::SymbolicParameterMap,
) -> Result<ExpressionType<I>, Diagnostic> {
    if support.as_ref().is_some_and(|support| {
        !matches!(
            support,
            SpatialSupport::Volume { .. }
                | SpatialSupport::Boundary { .. }
                | SpatialSupport::Coordinates { .. }
        )
    }) {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            declaration.range(),
            "source Field requires a volume, boundary, or coordinate-factor support",
        ));
    }
    let syntax =
        crate::hierarchy::parameters::specialize_type(file, declaration.value_type(), values)?;
    let value_type = crate::value_types::lower_value_type(file, &syntax, support.as_ref())?;
    Ok(ExpressionType::new(value_type, support))
}

pub(in crate::hierarchy::body_check) fn component_port_contract(
    elaborator: &Elaborator<'_>,
    owner: &ComponentDefinition<'_>,
    declaration: &ComponentPortDecl,
    values: &crate::hierarchy::parameters::SymbolicParameterMap,
) -> Result<PortContract, Vec<Diagnostic>> {
    let file = owner.file;
    match declaration.syntax() {
        PortSyntax::Signal {
            direction,
            value_type,
            domain,
            activation,
        } => {
            if let eqiora_lang::ActivationSyntax::Named(clock) = activation
                && crate::hierarchy::clocks::component(file, owner.declaration, clock).is_none()
            {
                return Err(vec![unresolved(
                    file,
                    declaration.range(),
                    clock,
                    "signal clock activation",
                )]);
            }
            let interface =
                crate::hierarchy::supports::component_support_interface(file, owner.declaration)?;
            let support = domain
                .as_deref()
                .map(|name| {
                    interface
                        .get(name)
                        .map(|contract| contract.support().clone())
                        .ok_or_else(|| {
                            vec![unresolved(
                                file,
                                declaration.range(),
                                name,
                                "signal support",
                            )]
                        })
                })
                .transpose()?;
            let syntax = crate::hierarchy::parameters::specialize_type(file, value_type, values)
                .map_err(|error| vec![error])?;
            let value_type = crate::value_types::lower_value_type(file, &syntax, support.as_ref())
                .map_err(|error| vec![error])?;
            Ok(PortContract::Signal {
                direction: *direction,
                value_type,
                support,
                activation: activation.clone(),
            })
        }
        PortSyntax::ScalarPhysicalConnector { connector } => {
            let connector = elaborator
                .resolve_connector(&owner.namespace, connector, file, declaration.range())
                .map_err(|error| vec![error])?;
            let eqiora_lang::ConnectorSyntax::ScalarPhysical {
                across_name,
                across_type,
                through_name,
                through_type,
            } = connector.declaration.syntax()
            else {
                return Err(vec![source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    connector.file,
                    connector.declaration.range(),
                    "Connector syntax is newer than definition-body validation",
                )]);
            };
            let mut diagnostics = Vec::new();
            let across_type = crate::value_types::lower_scalar_type(connector.file, across_type)
                .map_err(|error| diagnostics.push(error))
                .ok();
            let through_type = crate::value_types::lower_scalar_type(connector.file, through_type)
                .map_err(|error| diagnostics.push(error))
                .ok();
            match (across_type, through_type) {
                (Some(across_type), Some(through_type)) => Ok(PortContract::Physical {
                    nominal: PhysicalNominal::Connector(DefinitionKey {
                        namespace: connector.namespace,
                        name: connector.declaration.name().to_owned(),
                    }),
                    quantities: PhysicalMemberNames::Scalar {
                        across: across_name.clone(),
                        through: through_name.clone(),
                    },
                    across_type,
                    through_type,
                }),
                _ => Err(diagnostics),
            }
        }
        PortSyntax::FieldPhysical { connector, support } => {
            let connector = elaborator
                .resolve_connector(&owner.namespace, connector, file, declaration.range())
                .map_err(|error| vec![error])?;
            let ConnectorSyntax::FieldPhysical {
                trace,
                flux,
                shape,
                frame,
                pairing,
            } = connector.declaration.syntax()
            else {
                return Err(vec![source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    declaration.range(),
                    "field-physical Port requires a field-physical Connector",
                )]);
            };
            let interface =
                crate::hierarchy::supports::component_support_interface(file, owner.declaration)?;
            let support_contract =
                interface.visible_support(support).cloned().ok_or_else(|| {
                    vec![source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        declaration.range(),
                        format!(
                            "field-physical Port support `{support}` is not a public support slot"
                        ),
                    )]
                })?;
            boundary_port_contract(
                connector,
                support_contract,
                file,
                declaration.range(),
                trace,
                flux,
                shape,
                *frame,
                *pairing,
            )
        }
        _ => Err(vec![source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            declaration.range(),
            "component Port must be an explicit signal or nominal Connector interface",
        )]),
    }
}

pub(in crate::hierarchy::body_check) fn model_port_contract(
    scope: &DefinitionScope<'_, '_>,
    declaration: &PortDecl,
) -> Result<PortContract, Diagnostic> {
    match declaration.syntax() {
        PortSyntax::Signal {
            direction,
            value_type,
            domain,
            activation,
        } => {
            let support = domain
                .as_deref()
                .map(|name| {
                    scope.spatial_support(name).ok_or_else(|| {
                        scope.wrong_local_kind(declaration.range(), name, "signal support")
                    })
                })
                .transpose()?;
            let syntax = crate::hierarchy::parameters::specialize_type(
                scope.file,
                value_type,
                &scope.static_values,
            )?;
            let value_type =
                crate::value_types::lower_value_type(scope.file, &syntax, support.as_ref())?;
            Ok(PortContract::Signal {
                direction: *direction,
                value_type,
                support,
                activation: activation.clone(),
            })
        }
        PortSyntax::ScalarPhysical { domain } => match scope.symbols.get(domain) {
            Some(SymbolContract::Domain(DomainContract::Physical { quantities, types })) => {
                Ok(PortContract::Physical {
                    nominal: PhysicalNominal::ModelDomain(domain.clone()),
                    quantities: quantities.clone(),
                    across_type: types.0.clone(),
                    through_type: types.1.clone(),
                })
            }
            Some(_) => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                scope.file,
                declaration.range(),
                format!("physical Port Domain `{domain}` is not scalar physical"),
            )),
            None => Err(unresolved(
                scope.file,
                declaration.range(),
                domain,
                "scalar physical Domain",
            )),
        },
        PortSyntax::ScalarPhysicalConnector { .. } => Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "model-level Port cannot use a component Connector declaration directly",
        )),
        PortSyntax::FieldPhysical { .. } => Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "model-level field-physical Port requires hierarchy specialization",
        )),
        _ => Err(source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            scope.file,
            declaration.range(),
            "Port syntax is newer than definition-body validation",
        )),
    }
}
