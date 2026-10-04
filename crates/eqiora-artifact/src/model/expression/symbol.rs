//! Exact nominal references carried by expression symbols.
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireSymbol {
    Field {
        id: WireId,
    },
    Derivative {
        id: WireId,
    },
    Pre {
        id: WireId,
    },
    Next {
        id: WireId,
    },
    Parameter {
        id: WireId,
    },
    Observable {
        id: WireId,
    },
    Port {
        id: WireId,
    },
    Across {
        id: WireId,
    },
    Through {
        id: WireId,
    },
    PortTrace {
        id: WireId,
    },
    PortFlux {
        id: WireId,
    },
    Coordinate {
        support: WireId,
        factor: WireId,
        axis: usize,
    },
    Time,
}

impl WireSymbol {
    pub(crate) fn encode(value: SymbolRef) -> Result<Self, Diagnostic> {
        match value {
            SymbolRef::Field(id) => Ok(Self::Field {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Derivative(id) => Ok(Self::Derivative {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Pre(id) => Ok(Self::Pre {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Next(id) => Ok(Self::Next {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Observable(id) => Ok(Self::Observable {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Parameter(id) => Ok(Self::Parameter {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Port(id) => Ok(Self::Port {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Across(id) => Ok(Self::Across {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Through(id) => Ok(Self::Through {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::PortTrace(id) => Ok(Self::PortTrace {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::PortFlux(id) => Ok(Self::PortFlux {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Coordinate {
                support,
                factor,
                axis,
            } => Ok(Self::Coordinate {
                support: WireId::from_raw(support.erase()),
                factor: WireId::from_raw(factor.erase()),
                axis,
            }),
            SymbolRef::Time => Ok(Self::Time),
            _ => Err(invalid_artifact(
                "symbol kind is newer than the supported model wire vocabulary",
            )),
        }
    }

    pub(crate) fn decode(&self) -> Result<SymbolRef, Diagnostic> {
        Ok(match self {
            Self::Field { id } => SymbolRef::Field(id.typed::<kinds::Field>()?),
            Self::Derivative { id } => SymbolRef::Derivative(id.typed::<kinds::Field>()?),
            Self::Pre { id } => SymbolRef::Pre(id.typed::<kinds::Field>()?),
            Self::Next { id } => SymbolRef::Next(id.typed::<kinds::Field>()?),
            Self::Parameter { id } => SymbolRef::Parameter(id.typed::<kinds::Parameter>()?),
            Self::Observable { id } => SymbolRef::Observable(id.typed::<kinds::Observable>()?),
            Self::Port { id } => SymbolRef::Port(id.typed::<kinds::Port>()?),
            Self::Across { id } => SymbolRef::Across(id.typed::<kinds::Port>()?),
            Self::Through { id } => SymbolRef::Through(id.typed::<kinds::Port>()?),
            Self::PortTrace { id } => SymbolRef::PortTrace(id.typed::<kinds::Port>()?),
            Self::PortFlux { id } => SymbolRef::PortFlux(id.typed::<kinds::Port>()?),
            Self::Coordinate {
                support,
                factor,
                axis,
            } => SymbolRef::Coordinate {
                support: support.typed::<kinds::Domain>()?,
                factor: factor.typed::<kinds::Domain>()?,
                axis: *axis,
            },
            Self::Time => SymbolRef::Time,
        })
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = &WireId> {
        let ids = match self {
            Self::Field { id }
            | Self::Derivative { id }
            | Self::Pre { id }
            | Self::Next { id }
            | Self::Parameter { id }
            | Self::Observable { id }
            | Self::Port { id }
            | Self::Across { id }
            | Self::Through { id }
            | Self::PortTrace { id }
            | Self::PortFlux { id } => [Some(id), None],
            Self::Coordinate {
                support, factor, ..
            } => [Some(support), Some(factor)],
            Self::Time => [None, None],
        };
        ids.into_iter().flatten()
    }
}
