//! State variations belong to the exact accepted coefficient inventory.

use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DynQuantity, Id, RawId, ValueLiteral};
use eqiora_meshing::QuadratureRule;
use std::collections::{BTreeMap, HashMap};

use super::super::CommonResultPayload;
use super::{CommonResult, invalid};

/// A finite real Field direction bound to one exact accepted Result.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonObservableStateTangent {
    result_identity: String,
    fields: BTreeMap<RawId, Vec<f64>>,
}

pub(super) enum StateDerivative<'a> {
    First(&'a BTreeMap<RawId, Vec<f64>>),
    Second {
        wrt: Id<kinds::Field>,
        directions: [&'a BTreeMap<RawId, Vec<f64>>; 2],
    },
}

impl StateDerivative<'_> {
    pub(super) fn directions(&self) -> [Option<&BTreeMap<RawId, Vec<f64>>>; 2] {
        match self {
            Self::First(direction) => [Some(direction), None],
            Self::Second { directions, .. } => [Some(directions[0]), Some(directions[1])],
        }
    }
}

impl CommonResult {
    /// Bind a dimensioned coefficient direction to this Result's real Fields.
    ///
    /// Omitted Fields have zero variation. Supplied coefficients follow the
    /// accepted Field's canonical vertex order; duplicate or foreign Fields reject.
    /// # Errors
    /// Rejects wrong units, support, shape, nonfinite values or unavailable Fields.
    pub fn observable_state_tangent(
        &self,
        fields: impl IntoIterator<Item = (Id<kinds::Field>, Vec<DynQuantity>)>,
    ) -> Result<CommonObservableStateTangent, Diagnostic> {
        let CommonResultPayload::Static(payload) = &self.payload else {
            return Err(invalid(
                "Observable State tangent requires an instantaneous spatial Result",
            ));
        };
        let mut directions = BTreeMap::new();
        for (id, values) in fields {
            let field = payload
                .fields
                .iter()
                .find(|field| field.field_id == id.ulid().to_string())
                .ok_or_else(|| invalid("Observable State tangent Field is outside this Result"))?;
            let [block] = field.blocks.as_slice() else {
                return Err(invalid(
                    "Observable State tangent requires one coefficient block",
                ));
            };
            if block.values.len() != values.len()
                || values
                    .iter()
                    .any(|value| !value.value().is_finite() || value.dim() != field.dimension)
            {
                return Err(invalid(
                    "Observable State tangent coefficients differ from the accepted Field type or shape",
                ));
            }
            if directions
                .insert(
                    id.erase(),
                    values.iter().map(|value| value.value()).collect(),
                )
                .is_some()
            {
                return Err(invalid("Observable State tangent repeats a Field"));
            }
        }
        Ok(CommonObservableStateTangent {
            result_identity: self.identity().to_owned(),
            fields: directions,
        })
    }

    /// Apply the spatial functional's first variation to an exact State direction.
    ///
    /// The accepted Model Parameters, Geometry and quadrature stay fixed. Q1 value
    /// and normal-gradient variations use the same basis and measure as the primal;
    /// Operator IR owns the scalar chain rule. This is not a reduced-solve or
    /// geometry-shape sensitivity.
    /// # Errors
    /// Rejects foreign/stale lineage, wrong intervals of support, unsupported
    /// derivative operations, and missing spatial integration meaning.
    pub fn observe_state_jvp(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
        tangent: &CommonObservableStateTangent,
    ) -> Result<ValueLiteral, Diagnostic> {
        if tangent.result_identity != self.identity() || model != self.plan().model_artifact() {
            return Err(invalid(
                "Observable State tangent belongs to a foreign or stale Result/Model",
            ));
        }
        let program = super::observation_program(self, model)?;
        let (_, derivative) = super::composite::evaluate(
            self,
            &program,
            observable,
            quadratures,
            Some(&StateDerivative::First(&tangent.fields)),
        )?;
        derivative.ok_or_else(|| invalid("Observable State JVP has no derivative"))
    }
}

impl CommonResult {
    /// Evaluate the ordered second variation of a fixed polynomial functional.
    ///
    /// Both coefficient directions belong to this exact Result and vary only
    /// `wrt`; all other Fields, Parameters and Geometry stay fixed. The source
    /// compiler owns the local first/second derivation. This explicit State
    /// product is not a Hessian through the solve or a stability certificate.
    /// Directions need not satisfy essential restrictions; callers distinguish
    /// arbitrary State products from admissible stationarity perturbations.
    /// # Errors
    /// Rejects foreign lineage, other varied Fields, unsupported local calculus,
    /// nonlinear reduced compositions, and missing or incompatible quadrature.
    pub fn observe_state_second_variation(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
        wrt: Id<kinds::Field>,
        directions: [&CommonObservableStateTangent; 2],
    ) -> Result<ValueLiteral, Diagnostic> {
        if model != self.plan().model_artifact()
            || directions
                .iter()
                .any(|direction| direction.result_identity != self.identity())
        {
            return Err(invalid(
                "second variation belongs to a foreign or stale Result/Model",
            ));
        }
        if directions.iter().any(|direction| {
            direction.fields.iter().any(|(field, values)| {
                *field != wrt.erase() && values.iter().any(|value| *value != 0.0)
            })
        }) {
            return Err(invalid(
                "second variation directions must hold all other Fields fixed",
            ));
        }
        let program = super::observation_program(self, model)?;
        let derivative = StateDerivative::Second {
            wrt,
            directions: [&directions[0].fields, &directions[1].fields],
        };
        let (_, value) =
            super::composite::evaluate(self, &program, observable, quadratures, Some(&derivative))?;
        value.ok_or_else(|| invalid("second variation has no derived value"))
    }
}
