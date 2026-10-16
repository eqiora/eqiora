//! Complete physical time States, shared by deterministic and stochastic Runs.
use eqiora_artifact::{CanonicalModelArtifact, ModelEnvelope};
use eqiora_core::{Diagnostic, DimExponents, Id, TimeStateCoordinate, entity::kinds};
use eqiora_schema::kernel::KernelNode;
use sha2::{Digest, Sha256};
use std::sync::Arc;
mod artifact;
pub(crate) mod checkpoint;
mod continuation;
pub use continuation::CommonTimeCheckpoint;
mod plan;
mod request;
pub(crate) mod request_artifact;
pub use request::CommonTimeRunRequest;
mod space;
pub use plan::CommonTimePlan;
pub(crate) use space::TimeStateSpace;

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
}

/// Complete accepted numeric State for one exact finite time state space.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonTimeState {
    space: Arc<TimeStateSpace>,
    identity: String,
    time_s: f64,
    values: Vec<f64>,
    source_kind: &'static str,
}

impl CommonTimeState {
    pub(crate) fn new(
        space: Arc<TimeStateSpace>,
        time_s: f64,
        values: Vec<f64>,
        source_kind: &'static str,
    ) -> Result<Self, Diagnostic> {
        if !time_s.is_finite()
            || time_s < 0.0
            || time_s.to_bits() == (-0.0_f64).to_bits()
            || values.len() != space.coordinates.len()
            || values.iter().any(|value| !value.is_finite())
            || !matches!(source_kind, "initial" | "result" | "checkpoint")
        {
            return Err(invalid(
                "finite time State requires finite time and one finite value per exact state Field",
            ));
        }
        let mut bytes = Vec::new();
        push(&mut bytes, space.identity.as_bytes());
        bytes.extend_from_slice(&time_s.to_bits().to_be_bytes());
        for value in &values {
            bytes.extend_from_slice(&value.to_bits().to_be_bytes());
        }
        let identity = digest(b"eqiora.common-time-state/v1\0", &bytes);
        Ok(Self {
            space,
            identity,
            time_s,
            values,
            source_kind,
        })
    }

    /// Recover one complete typed Field or stored derivative from this exact State.
    ///
    /// # Errors
    /// Rejects a foreign Model, absent Field/order, or incomplete coordinate layout.
    pub fn field_value(
        &self,
        model: &ModelEnvelope,
        field: Id<kinds::Field>,
        derivative_order: u32,
    ) -> Result<eqiora_core::ValueLiteral, Diagnostic> {
        if model.artifact_reference()?.artifact().to_string() != self.space.model_digest {
            return Err(invalid(
                "time State value belongs to a different exact Model",
            ));
        }
        let kernel = model
            .to_program()
            .map_err(|errors| errors.into_iter().next().expect("failed Model replay"))?;
        let Some(KernelNode::Field(definition)) = kernel.node(field.erase()) else {
            return Err(invalid("time State value requires an exact Field"));
        };
        let coordinate = |component, imaginary| {
            self.space
                .coordinates
                .iter()
                .position(|coordinate| {
                    *coordinate
                        == eqiora_core::TimeStateCoordinate::new(
                            field,
                            derivative_order,
                            component,
                            imaginary,
                        )
                })
                .ok_or_else(|| {
                    invalid("time State omits the requested Field derivative coordinate")
                })
        };
        let first = coordinate(0, false)?;
        let value_type = definition
            .value_type()
            .clone()
            .with_dimension(self.space.dimensions[first])
            .map_err(|error| invalid(error.to_string()))?;
        let complex = value_type.scalar_domain() == eqiora_core::ScalarDomain::Complex;
        let count = value_type
            .shape()
            .component_count()
            .ok_or_else(|| invalid("time State Field shape is not representable"))?;
        let components = (0..count)
            .map(|component| {
                Ok((
                    self.values[coordinate(component, false)?],
                    if complex {
                        self.values[coordinate(component, true)?]
                    } else {
                        0.
                    },
                ))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        eqiora_core::ValueLiteral::new(value_type, components)
            .map_err(|error| invalid(error.to_string()))
    }

    #[must_use]
    pub fn state_space_identity(&self) -> &str {
        &self.space.identity
    }

    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    #[must_use]
    pub fn model_digest(&self) -> &str {
        &self.space.model_digest
    }

    #[must_use]
    pub const fn time_s(&self) -> f64 {
        self.time_s
    }

    #[must_use]
    pub fn state_coordinates(&self) -> &[eqiora_core::TimeStateCoordinate] {
        &self.space.coordinates
    }

    #[must_use]
    pub fn dimensions(&self) -> &[DimExponents] {
        &self.space.dimensions
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[must_use]
    pub const fn source_kind(&self) -> &'static str {
        self.source_kind
    }
}

pub(crate) fn push(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value);
}

pub(crate) fn digest(domain: &[u8], bytes: &[u8]) -> String {
    let value = Sha256::digest([domain, bytes].concat());
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn dimension_bytes(value: DimExponents) -> [u8; 56] {
    let mut bytes = [0; 56];
    for (chunk, (numerator, denominator)) in bytes
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(value.exponents())
    {
        chunk[..4].copy_from_slice(&numerator.to_be_bytes());
        chunk[4..].copy_from_slice(&denominator.to_be_bytes());
    }
    bytes
}
