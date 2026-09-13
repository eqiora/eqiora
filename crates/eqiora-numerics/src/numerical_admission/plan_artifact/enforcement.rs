//! Canonical numerical enforcement policy embedded in the exact finite Plan.
use super::*;
use crate::finite_constraints::{ConstraintRef, ConstraintTolerance, FiniteConstraintEnforcement};
use eqiora_core::DynQuantity;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "algorithm", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireEnforcement {
    ActiveSet {
        max_active_sets: u32,
        tolerances: Vec<WireTolerance>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireTolerance {
    relation_ulid: String,
    ordinal: u32,
    operands: WireOperands,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireOperands {
    Inequality {
        value: f64,
        dimension: [(i32, i32); 7],
    },
    Complementarity {
        left_value: f64,
        left_dimension: [(i32, i32); 7],
        right_value: f64,
        right_dimension: [(i32, i32); 7],
    },
}

impl WireEnforcement {
    pub(super) fn from_native(value: &FiniteConstraintEnforcement) -> Self {
        Self::ActiveSet {
            max_active_sets: value.max_active_sets(),
            tolerances: value
                .tolerances()
                .iter()
                .map(|entry| WireTolerance {
                    relation_ulid: entry.reference().relation().ulid().to_string(),
                    ordinal: entry.reference().ordinal(),
                    operands: match entry.right() {
                        None => WireOperands::Inequality {
                            value: entry.left().value(),
                            dimension: entry.left().dim().exponents(),
                        },
                        Some(right) => WireOperands::Complementarity {
                            left_value: entry.left().value(),
                            left_dimension: entry.left().dim().exponents(),
                            right_value: right.value(),
                            right_dimension: right.dim().exponents(),
                        },
                    },
                })
                .collect(),
        }
    }

    pub(super) fn to_native(&self) -> Result<FiniteConstraintEnforcement, Diagnostic> {
        let Self::ActiveSet {
            max_active_sets,
            tolerances,
        } = self;
        let tolerances = tolerances
            .iter()
            .map(|entry| {
                let reference = ConstraintRef::new(
                    parse_id::<kinds::Relation>(&entry.relation_ulid, "Relation")?,
                    entry.ordinal,
                );
                match &entry.operands {
                    WireOperands::Inequality { value, dimension } => {
                        ConstraintTolerance::inequality(reference, quantity(*value, *dimension)?)
                    }
                    WireOperands::Complementarity {
                        left_value,
                        left_dimension,
                        right_value,
                        right_dimension,
                    } => ConstraintTolerance::complementarity(
                        reference,
                        quantity(*left_value, *left_dimension)?,
                        quantity(*right_value, *right_dimension)?,
                    ),
                }
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        FiniteConstraintEnforcement::active_set(tolerances, *max_active_sets)
    }
}

fn quantity(value: f64, dimension: [(i32, i32); 7]) -> Result<DynQuantity, Diagnostic> {
    let dimension = DimExponents::from_rationals(dimension)
        .ok_or_else(|| invalid("finite enforcement has invalid physical dimension exponents"))?;
    Ok(DynQuantity::new(value, dimension))
}
