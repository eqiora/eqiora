//! Scalar connection contract validation and diagnostics.

use super::*;

pub(super) fn validate_connection_contract(
    declaration: &ConnectionDecl,
    contracts: &[PortContract],
    file: &str,
) -> Result<(), Diagnostic> {
    let kind = match declaration.syntax() {
        ConnectionSyntax::Signal => ScalarConnectionKind::Signal,
        ConnectionSyntax::Conserving => ScalarConnectionKind::Conserving,
        ConnectionSyntax::SpatialPeriodic => {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                "spatial-periodic Connection requires exact boundary Port references",
            ));
        }
    };
    let ports = contracts
        .iter()
        .map(connection_port_contract)
        .collect::<Vec<_>>();
    ScalarPortContract::validate_connection(kind, &ports).map_err(|violation| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            declaration.range(),
            connection_violation_message(violation),
        )
    })?;
    if kind == ScalarConnectionKind::Signal
        && !matches!(
            contracts.first(),
            Some(PortContract::Signal {
                direction: SignalDirectionSyntax::Output,
                ..
            })
        )
    {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            declaration.range(),
            "signal Connection source before `->` must be its output Port",
        ));
    }
    Ok(())
}

fn connection_port_contract(contract: &PortContract) -> ScalarPortContract<&PhysicalNominal> {
    match contract {
        PortContract::Signal {
            direction,
            value_type,
            ..
        } => ScalarPortContract::Signal {
            direction: match direction {
                SignalDirectionSyntax::Input => SignalDirection::Input,
                SignalDirectionSyntax::Output => SignalDirection::Output,
            },
            value_type: value_type.clone(),
        },
        PortContract::Physical { nominal, .. } => ScalarPortContract::ScalarPhysical { nominal },
        PortContract::BoundaryPhysical { nominal, .. } => {
            ScalarPortContract::ScalarPhysical { nominal }
        }
    }
}

fn connection_violation_message(violation: ScalarConnectionViolation) -> &'static str {
    match violation {
        ScalarConnectionViolation::TooFewPorts { .. } => {
            "Connection requires at least two visible Ports"
        }
        ScalarConnectionViolation::SignalDirections { .. } => {
            "signal Connection requires exactly one output and one or more inputs"
        }
        ScalarConnectionViolation::SignalTypeMismatch => {
            "signal Connection requires dimension-matched inputs with compatible scalar domains, shapes and frames"
        }
        ScalarConnectionViolation::MixedConservingFamilies => {
            "conserving Connection cannot mix signal and scalar physical Ports"
        }
        ScalarConnectionViolation::PhysicalNominalMismatch => {
            "conserving Connection requires scalar physical Ports on the exact same nominal Connector or Domain"
        }
    }
}

pub(super) fn connection_fragment_error(
    file: &str,
    range: TextRange,
    error: ConnectionSetError,
) -> Diagnostic {
    let code = match error {
        ConnectionSetError::TooFewMembers { .. } | ConnectionSetError::DuplicateMember => {
            codes::LANGUAGE_TYPE_ERROR
        }
        ConnectionSetError::LimitExceeded { .. }
        | ConnectionSetError::CountOverflow { .. }
        | ConnectionSetError::Allocation { .. } => codes::LANGUAGE_LOWERING_ERROR,
    };
    source_error(code, file, range, error.to_string())
}
