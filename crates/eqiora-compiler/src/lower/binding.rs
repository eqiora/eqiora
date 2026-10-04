use super::*;

#[derive(Debug, Clone)]
pub(super) enum Binding {
    Domain(Id<kinds::Domain>, DomainContract),
    Representation(Id<kinds::Representation>),
    Field(Id<kinds::Field>, FieldContract),
    Parameter(Id<kinds::Parameter>, eqiora_core::ValueType),
    Port(Id<kinds::Port>, PortContract),
    Clock(Id<kinds::ClockDomain>, eqiora_schema::kernel::RationalTime),
    Event(Id<kinds::Activation>),
    Observable(
        Id<kinds::Observable>,
        eqiora_lang::ValueTypeSyntax,
        Option<String>,
    ),
    Relation {
        relation: Id<kinds::Relation>,
        activation: Id<kinds::Activation>,
    },
}

impl Binding {
    pub(super) fn primary_id(&self) -> RawId {
        match self {
            Self::Domain(id, _) => id.erase(),
            Self::Representation(id) => id.erase(),
            Self::Field(id, _) => id.erase(),
            Self::Parameter(id, _) => id.erase(),
            Self::Port(id, _) => id.erase(),
            Self::Clock(id, _) => id.erase(),
            Self::Observable(id, ..) => id.erase(),
            Self::Event(id) => id.erase(),
            Self::Relation { relation, .. } => relation.erase(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum DomainContract {
    Spatial {
        dimensions: Option<usize>,
        parent: Option<String>,
    },
    ScalarPhysical {
        across_type: eqiora_core::ValueType,
        through_type: eqiora_core::ValueType,
    },
    BoundaryPhysical(BoundaryPhysicalConnector),
}

#[derive(Debug, Clone)]
pub(super) struct FieldContract {
    pub(super) dimension: DimExponents,
    pub(super) value_type: eqiora_lang::ValueTypeSyntax,
    pub(super) domain: Option<String>,
    pub(super) role: eqiora_lang::FieldRoleSyntax,
    pub(super) activation: ActivationSyntax,
}

#[derive(Debug, Clone)]
pub(super) enum PortContract {
    Signal {
        direction: SignalDirectionSyntax,
        value_type: eqiora_lang::ValueTypeSyntax,
        domain: Option<String>,
        activation: ActivationSyntax,
    },
    ScalarPhysical {
        domain: String,
    },
    BoundaryPhysical {
        connector: String,
        boundary: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ResolvedPortContract {
    Signal {
        direction: SignalDirectionSyntax,
        value_type: eqiora_core::ValueType,
        support: Option<eqiora_schema::kernel::typing::SpatialSupport<RawId>>,
        clock: Option<Id<kinds::ClockDomain>>,
    },
    ScalarPhysical {
        domain: Id<kinds::Domain>,
        across_type: eqiora_core::ValueType,
        through_type: eqiora_core::ValueType,
    },
    BoundaryPhysical {
        connector: Id<kinds::Domain>,
        boundary: Id<kinds::Domain>,
        trace_type: eqiora_core::ValueType,
        flux_type: eqiora_core::ValueType,
    },
}

pub(super) fn bind_domain(
    file: &str,
    range: TextRange,
    syntax: &DomainSyntax,
) -> Result<DomainContract, Diagnostic> {
    match syntax {
        DomainSyntax::ScalarPhysical {
            across_type,
            through_type,
            ..
        } => Ok(DomainContract::ScalarPhysical {
            across_type: crate::value_types::lower_scalar_type(file, across_type)?,
            through_type: crate::value_types::lower_scalar_type(file, through_type)?,
        }),
        DomainSyntax::CartesianBox(bounds) => Ok(DomainContract::Spatial {
            dimensions: Some(bounds.len()),
            parent: None,
        }),
        DomainSyntax::Boundary { parent, .. } => Ok(DomainContract::Spatial {
            dimensions: None,
            parent: Some(parent.clone()),
        }),
        _ => Err(source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            range,
            "Domain syntax is newer than this compiler",
        )),
    }
}

pub(super) fn resolve_field_contract(
    file: &str,
    range: TextRange,
    contract: &FieldContract,
    bindings: &BTreeMap<String, Binding>,
) -> Result<eqiora_core::ValueType, Diagnostic> {
    if contract.domain.as_ref().is_some_and(|name| {
        matches!(
            bindings.get(name),
            Some(Binding::Domain(
                _,
                DomainContract::Spatial {
                    parent: Some(_),
                    ..
                }
            ))
        )
    }) {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            "source Field requires a volume support",
        ));
    }
    let support = contract.domain.as_ref().and_then(|name| {
        let Binding::Domain(
            id,
            DomainContract::Spatial {
                dimensions: Some(dimensions),
                ..
            },
        ) = bindings.get(name)?
        else {
            return None;
        };
        Some(eqiora_schema::kernel::typing::SpatialSupport::Volume {
            domain: id.erase(),
            dimensions: *dimensions,
        })
    });
    crate::value_types::lower_value_type(file, &contract.value_type, support.as_ref())
}

pub(super) fn bind_port(
    file: &str,
    range: TextRange,
    syntax: &PortSyntax,
) -> Result<PortContract, Diagnostic> {
    match syntax {
        PortSyntax::Signal {
            direction,
            value_type,
            domain,
            activation,
        } => Ok(PortContract::Signal {
            direction: *direction,
            value_type: value_type.clone(),
            domain: domain.clone(),
            activation: activation.clone(),
        }),
        PortSyntax::ScalarPhysical { domain } => Ok(PortContract::ScalarPhysical {
            domain: domain.clone(),
        }),
        _ => Err(source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            range,
            "Port syntax is newer than this compiler",
        )),
    }
}

pub(super) fn resolve_port_contract(
    file: &str,
    range: TextRange,
    contract: &PortContract,
    bindings: &BTreeMap<String, Binding>,
) -> Result<ResolvedPortContract, Diagnostic> {
    match contract {
        PortContract::Signal {
            direction,
            value_type,
            domain,
            activation,
        } => {
            let support = domain
                .as_deref()
                .map(|name| super::expression::relation_support(file, range, name, bindings))
                .transpose()?;
            let value_type =
                crate::value_types::lower_value_type(file, value_type, support.as_ref())?;
            let clock = match activation {
                ActivationSyntax::Continuous => None,
                ActivationSyntax::Named(name) => match bindings.get(name) {
                    Some(Binding::Clock(id, _)) => Some(*id),
                    _ => return Err(unresolved(file, range, name, "signal clock")),
                },
                _ => {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        range,
                        "unsupported signal activation",
                    ));
                }
            };
            Ok(ResolvedPortContract::Signal {
                direction: *direction,
                value_type,
                support,
                clock,
            })
        }
        PortContract::ScalarPhysical { domain } => {
            let Some(Binding::Domain(
                domain_id,
                DomainContract::ScalarPhysical {
                    across_type,
                    through_type,
                },
            )) = bindings.get(domain)
            else {
                return match bindings.get(domain) {
                    Some(_) => Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        range,
                        format!("physical Port Domain `{domain}` is not scalar physical"),
                    )),
                    None => Err(unresolved(file, range, domain, "scalar physical Domain")),
                };
            };
            Ok(ResolvedPortContract::ScalarPhysical {
                domain: *domain_id,
                across_type: across_type.clone(),
                through_type: through_type.clone(),
            })
        }
        PortContract::BoundaryPhysical {
            connector,
            boundary,
        } => {
            let Some(Binding::Domain(connector_id, DomainContract::BoundaryPhysical(contract))) =
                bindings.get(connector)
            else {
                return Err(unresolved(
                    file,
                    range,
                    connector,
                    "field-physical Connector Domain",
                ));
            };
            let Some(Binding::Domain(boundary_id, DomainContract::Spatial { .. })) =
                bindings.get(boundary)
            else {
                return Err(unresolved(
                    file,
                    range,
                    boundary,
                    "field-physical boundary Domain",
                ));
            };
            Ok(ResolvedPortContract::BoundaryPhysical {
                connector: *connector_id,
                boundary: *boundary_id,
                trace_type: contract.trace_type().clone(),
                flux_type: contract.flux_type().clone(),
            })
        }
    }
}

pub(super) fn insert_binding(
    file: &str,
    bindings: &mut BTreeMap<String, Binding>,
    name: &str,
    binding: Binding,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if name == crate::math::ROOT {
        diagnostics.push(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            "identifier `math` is reserved for compiler-owned scalar mathematics",
        ));
    } else if is_reserved(name) {
        diagnostics.push(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            format!("`{name}` is reserved by Eqiora Language v0"),
        ));
    } else if bindings.insert(name.to_owned(), binding).is_some() {
        diagnostics.push(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            format!("duplicate declaration name `{name}`"),
        ));
    }
}

fn is_reserved(name: &str) -> bool {
    matches!(
        name,
        "model"
            | "domain"
            | "representation"
            | "field"
            | "parameter"
            | "port"
            | "clock"
            | "relation"
            | "connect"
            | "boundary"
            | "box"
            | "axis"
            | "side"
            | "lower"
            | "upper"
            | "continuum"
            | "on"
            | "as"
            | "continuous"
            | "periodic"
            | "signal"
            | "conserving"
            | "scalar_physical"
            | "input"
            | "output"
            | "period"
            | "phase"
            | "time"
            | "derivative"
            | "pre"
            | "next"
            | "grad"
            | "div"
            | "symmetric_part"
            | "isotropic_lift"
            | "trace"
            | "normal"
            | "across"
            | "through"
            | "kg"
            | "m"
            | "s"
            | "A"
            | "K"
            | "mol"
            | "cd"
    )
}
