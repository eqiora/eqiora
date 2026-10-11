//! Pure compatibility rules for the current scalar Port families.
//!
//! This module deliberately describes only the scalar kernel contract. Future
//! field-valued physical interfaces can share the same connection algebra
//! without being forced into this closed payload. Source spans, graph paths,
//! and diagnostic prose remain responsibilities of each consuming layer.

#[cfg(test)]
use eqiora_core::DimExponents;
use eqiora_core::ValueType;

use super::SignalDirection;

/// Connection semantics understood by the scalar compatibility contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarConnectionKind {
    /// One causal output drives one or more inputs.
    Signal,
    /// Acausal members form one conserving connection set.
    Conserving,
}

/// Compatibility-relevant type of one scalar Port.
///
/// `I` is the exact nominal identity of a scalar physical connector or Domain.
/// Matching dimensions never substitute for matching that identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScalarPortContract<I> {
    /// Causal scalar signal.
    Signal {
        /// Direction relative to the owning relation network.
        direction: SignalDirection,
        /// Complete mathematical type of the carried value.
        value_type: ValueType,
    },
    /// Scalar physical across/through pair with exact nominal identity.
    ScalarPhysical {
        /// Connector or Domain identity defining the physical type.
        nominal: I,
    },
}

/// One structured scalar-connection compatibility failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarConnectionViolation {
    /// A connection has fewer than two members.
    TooFewPorts {
        /// Number of supplied Ports.
        found: usize,
    },
    /// A signal connection does not contain exactly one output and otherwise
    /// only inputs.
    SignalDirections {
        /// Number of output Ports.
        outputs: usize,
        /// Number of input Ports.
        inputs: usize,
        /// Total number of Ports, including incompatible kinds.
        total: usize,
    },
    /// An input cannot receive the output's complete mathematical type.
    SignalTypeMismatch,
    /// A conserving connection mixes scalar Port families.
    MixedConservingFamilies,
    /// Scalar physical Ports do not have one exact nominal identity.
    PhysicalNominalMismatch,
}

/// Validate the current scalar Port compatibility algebra.
///
/// The check is independent of syntax ordering and graph storage. Signal
/// causality follows Port direction: exactly one output and one or more inputs
/// must accept the output's complete type, with real-to-complex embedding only.
/// Scalar physical Ports share one exact nominal identity.
///
/// # Errors
/// Returns the first structural incompatibility in a stable rule order.
pub fn validate_scalar_connection<I: Eq>(
    kind: ScalarConnectionKind,
    ports: &[ScalarPortContract<I>],
) -> Result<(), ScalarConnectionViolation> {
    if ports.len() < 2 {
        return Err(ScalarConnectionViolation::TooFewPorts { found: ports.len() });
    }
    match kind {
        ScalarConnectionKind::Signal => validate_signal(ports),
        ScalarConnectionKind::Conserving => validate_conserving(ports),
    }
}

fn validate_signal<I>(ports: &[ScalarPortContract<I>]) -> Result<(), ScalarConnectionViolation> {
    let mut outputs = 0;
    let mut inputs = 0;
    let mut output_type = None;
    for port in ports {
        let ScalarPortContract::Signal {
            direction,
            value_type,
        } = port
        else {
            continue;
        };
        match direction {
            SignalDirection::Input => inputs += 1,
            SignalDirection::Output => {
                outputs += 1;
                output_type = Some(value_type);
            }
        }
    }
    if outputs != 1 || inputs + outputs != ports.len() {
        return Err(ScalarConnectionViolation::SignalDirections {
            outputs,
            inputs,
            total: ports.len(),
        });
    }
    let output_type = output_type.expect("exactly one output was checked");
    for port in ports {
        if let ScalarPortContract::Signal {
            direction: SignalDirection::Input,
            value_type,
        } = port
            && output_type
                .clone()
                .with_common_scalar_domain(value_type)
                .as_ref()
                != Some(value_type)
        {
            return Err(ScalarConnectionViolation::SignalTypeMismatch);
        }
    }
    Ok(())
}

fn validate_conserving<I: Eq>(
    ports: &[ScalarPortContract<I>],
) -> Result<(), ScalarConnectionViolation> {
    match &ports[0] {
        ScalarPortContract::ScalarPhysical { nominal } => {
            let mut nominal_matches = true;
            for port in ports {
                let ScalarPortContract::ScalarPhysical { nominal: candidate } = port else {
                    return Err(ScalarConnectionViolation::MixedConservingFamilies);
                };
                nominal_matches &= candidate == nominal;
            }
            if nominal_matches {
                Ok(())
            } else {
                Err(ScalarConnectionViolation::PhysicalNominalMismatch)
            }
        }
        ScalarPortContract::Signal { .. } => {
            Err(ScalarConnectionViolation::MixedConservingFamilies)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT: DimExponents = DimExponents::DIMENSIONLESS;
    const LENGTH: DimExponents =
        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");

    fn signal(direction: SignalDirection, dimension: DimExponents) -> ScalarPortContract<u8> {
        ScalarPortContract::Signal {
            direction,
            value_type: ValueType::scalar(eqiora_core::ScalarDomain::Real, dimension)
                .expect("checked scalar type"),
        }
    }

    #[test]
    fn signal_types_preserve_roles_and_embed_only_toward_complex_inputs() {
        use eqiora_core::{ScalarDomain, ValueFrame, ValueShape};
        let real = ValueType::scalar(ScalarDomain::Real, LENGTH).expect("checked scalar type");
        let complex =
            ValueType::scalar(ScalarDomain::Complex, LENGTH).expect("checked scalar type");
        let array = real.clone().array(3).unwrap();
        let vector = ValueType::shaped(
            ScalarDomain::Real,
            LENGTH,
            ValueShape::new([3]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        for (output, input, accepted) in [
            (real.clone(), complex.clone(), true),
            (complex.clone(), real.clone(), false),
            (array.clone(), complex.array(3).unwrap(), true),
            (array.clone(), real.clone().array(2).unwrap(), false),
            (array.clone(), vector.clone(), false),
            (vector, array.clone(), false),
            (array, real.clone(), false),
            (
                real.clone(),
                real.with_dimension(UNIT)
                    .expect("checked numeric dimension"),
                false,
            ),
        ] {
            let ports = [
                ScalarPortContract::<u8>::Signal {
                    direction: SignalDirection::Input,
                    value_type: input,
                },
                ScalarPortContract::Signal {
                    direction: SignalDirection::Output,
                    value_type: output,
                },
            ];
            let expected = if accepted {
                Ok(())
            } else {
                Err(ScalarConnectionViolation::SignalTypeMismatch)
            };
            assert_eq!(
                validate_scalar_connection(ScalarConnectionKind::Signal, &ports),
                expected
            );
        }
    }

    #[test]
    fn signal_is_one_output_plus_dimension_matched_inputs() {
        let valid = [
            signal(SignalDirection::Input, LENGTH),
            signal(SignalDirection::Output, LENGTH),
            signal(SignalDirection::Input, LENGTH),
        ];
        assert_eq!(
            validate_scalar_connection(ScalarConnectionKind::Signal, &valid),
            Ok(())
        );

        let two_outputs = [
            signal(SignalDirection::Output, UNIT),
            signal(SignalDirection::Output, UNIT),
        ];
        assert!(matches!(
            validate_scalar_connection(ScalarConnectionKind::Signal, &two_outputs),
            Err(ScalarConnectionViolation::SignalDirections { outputs: 2, .. })
        ));

        let mismatched = [
            signal(SignalDirection::Output, UNIT),
            signal(SignalDirection::Input, LENGTH),
        ];
        assert_eq!(
            validate_scalar_connection(ScalarConnectionKind::Signal, &mismatched),
            Err(ScalarConnectionViolation::SignalTypeMismatch)
        );
    }

    #[test]
    fn physical_ports_match_nominal_identity() {
        let nominal_mismatch = [
            ScalarPortContract::ScalarPhysical { nominal: 1_u8 },
            ScalarPortContract::ScalarPhysical { nominal: 2_u8 },
        ];
        assert_eq!(
            validate_scalar_connection(ScalarConnectionKind::Conserving, &nominal_mismatch),
            Err(ScalarConnectionViolation::PhysicalNominalMismatch)
        );

        let exact_nominal = [
            ScalarPortContract::ScalarPhysical { nominal: 7_u8 },
            ScalarPortContract::ScalarPhysical { nominal: 7_u8 },
        ];
        assert_eq!(
            validate_scalar_connection(ScalarConnectionKind::Conserving, &exact_nominal),
            Ok(())
        );
    }

    #[test]
    fn connection_families_never_coerce() {
        let mixed = [
            signal(SignalDirection::Output, UNIT),
            ScalarPortContract::ScalarPhysical { nominal: 1_u8 },
        ];
        assert_eq!(
            validate_scalar_connection(ScalarConnectionKind::Conserving, &mixed),
            Err(ScalarConnectionViolation::MixedConservingFamilies)
        );
        assert_eq!(
            validate_scalar_connection(
                ScalarConnectionKind::Signal,
                &[signal(SignalDirection::Output, UNIT)]
            ),
            Err(ScalarConnectionViolation::TooFewPorts { found: 1 })
        );
    }
}
