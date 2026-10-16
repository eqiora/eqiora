//! One Model-authenticated layout for complete numeric time States.
use super::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TimeStateSpace {
    pub(super) identity: String,
    pub(super) model_digest: String,
    pub(super) coordinates: Vec<TimeStateCoordinate>,
    pub(super) dimensions: Vec<DimExponents>,
}

impl TimeStateSpace {
    pub(crate) fn new(
        model: &ModelEnvelope,
        coordinates: Vec<TimeStateCoordinate>,
        policy_identity: &[u8],
    ) -> Result<Self, Diagnostic> {
        let kernel = model
            .to_program()
            .map_err(|errors| errors.into_iter().next().expect("Model diagnostic"))?;
        if coordinates.is_empty() {
            return Err(invalid("time state space requires at least one coordinate"));
        }
        let mut seen = HashSet::new();
        let mut groups = HashMap::new();
        let dimensions = coordinates
            .iter()
            .map(|coordinate| {
                let Some(KernelNode::Field(field)) = kernel.node(coordinate.field().erase()) else {
                    return Err(invalid(
                        "time state coordinate requires an exact Model Field",
                    ));
                };
                let ty = field.value_type();
                let parts = match ty.scalar_domain() {
                    eqiora_core::ScalarDomain::Real => 1,
                    eqiora_core::ScalarDomain::Complex => 2,
                    _ => {
                        return Err(invalid(
                            "time state space requires real or complex coordinates",
                        ));
                    }
                };
                let count = ty
                    .shape()
                    .component_count()
                    .ok_or_else(|| invalid("time state shape exceeds addressable storage"))?;
                if coordinate.component() >= count
                    || (coordinate.is_imaginary() && parts != 2)
                    || !seen.insert(*coordinate)
                {
                    return Err(invalid(
                        "time state coordinates repeat or differ from the exact Field type",
                    ));
                }
                let group = groups
                    .entry((coordinate.field(), coordinate.derivative_order()))
                    .or_insert((
                        0usize,
                        count
                            .checked_mul(parts)
                            .ok_or_else(|| invalid("time state coordinate count overflows"))?,
                    ));
                group.0 += 1;
                if let Some(order) = std::num::NonZeroU32::new(coordinate.derivative_order()) {
                    eqiora_schema::kernel::typing::time_derivative(
                        &eqiora_schema::kernel::typing::ExpressionType::<()>::scalar(
                            field.dimension(),
                            None,
                        ),
                        order,
                    )
                    .map(|typed| typed.dimension())
                    .map_err(|_| invalid("time state derivative dimension is not representable"))
                } else {
                    Ok(field.dimension())
                }
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        if groups.values().any(|(actual, expected)| actual != expected) {
            return Err(invalid(
                "time state space omits part of a complete Field or derivative value",
            ));
        }
        let model_digest = model.artifact_reference()?.artifact().to_string();
        let mut bytes = Vec::new();
        push(&mut bytes, model_digest.as_bytes());
        bytes.extend_from_slice(&(coordinates.len() as u64).to_be_bytes());
        for (coordinate, dimension) in coordinates.iter().zip(&dimensions) {
            bytes.extend_from_slice(&coordinate.field().ulid().to_bytes());
            bytes.extend_from_slice(&coordinate.derivative_order().to_be_bytes());
            bytes.extend_from_slice(&(coordinate.component() as u64).to_be_bytes());
            bytes.push(u8::from(coordinate.is_imaginary()));
            bytes.extend_from_slice(&dimension_bytes(*dimension));
        }
        push(&mut bytes, b"real-coordinate-f64/no-method-history/v1");
        push(&mut bytes, policy_identity);
        Ok(Self {
            identity: digest(b"eqiora.common-time-state-space/v1\0", &bytes),
            model_digest,
            coordinates,
            dimensions,
        })
    }

    pub(crate) fn identity(&self) -> &str {
        &self.identity
    }
    pub(crate) fn model_digest(&self) -> &str {
        &self.model_digest
    }
    pub(crate) fn coordinates(&self) -> &[TimeStateCoordinate] {
        &self.coordinates
    }
    pub(crate) fn dimensions(&self) -> &[DimExponents] {
        &self.dimensions
    }
}
