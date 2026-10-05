//! Exact physical normalization of original finite equality conditions.
use super::*;
use crate::finite_constraints::ConstraintRef;
use eqiora_realization::PositivePhysicalScale;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireResidualScale {
    relation_ulid: String,
    ordinal: u32,
    value: f64,
    dimension: [(i32, i32); 7],
}

impl WireResidualScale {
    pub(super) fn from_native(entry: &(ConstraintRef, PositivePhysicalScale)) -> Self {
        Self {
            relation_ulid: entry.0.relation().ulid().to_string(),
            ordinal: entry.0.ordinal(),
            value: entry.1.quantity().value(),
            dimension: entry.1.quantity().dim().exponents(),
        }
    }

    pub(super) fn to_native(&self) -> Result<(ConstraintRef, PositivePhysicalScale), Diagnostic> {
        let dimension = DimExponents::from_rationals(self.dimension)
            .ok_or_else(|| invalid("residual scale has invalid physical dimension exponents"))?;
        Ok((
            ConstraintRef::new(
                parse_id::<kinds::Relation>(&self.relation_ulid, "Relation")?,
                self.ordinal,
            ),
            PositivePhysicalScale::new(DynQuantity::new(self.value, dimension))?,
        ))
    }
}

pub(in crate::numerical_admission) fn bytes(
    scales: &[(ConstraintRef, PositivePhysicalScale)],
) -> Result<Vec<u8>, Diagnostic> {
    serde_json::to_vec(
        &scales
            .iter()
            .map(WireResidualScale::from_native)
            .collect::<Vec<_>>(),
    )
    .map_err(|error| invalid(format!("cannot encode finite residual scales: {error}")))
}
