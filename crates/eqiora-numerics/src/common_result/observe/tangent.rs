//! State variations belong to the exact accepted coefficient inventory.

use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DynQuantity, Id, RawId, ValueLiteral};
use eqiora_meshing::QuadratureRule;
use eqiora_schema::kernel::{KernelNode, ObservableReduction};
use std::collections::BTreeMap;

use super::super::CommonResultPayload;
use super::{CommonResult, invalid, spatial};

/// A finite real Field direction bound to one exact accepted Result.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonObservableStateTangent {
    result_identity: String,
    fields: BTreeMap<RawId, Vec<f64>>,
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
        quadrature: &QuadratureRule,
        tangent: &CommonObservableStateTangent,
    ) -> Result<ValueLiteral, Diagnostic> {
        if tangent.result_identity != self.identity() || model != self.plan().model_artifact() {
            return Err(invalid(
                "Observable State tangent belongs to a foreign or stale Result/Model",
            ));
        }
        let program = super::observation_program(self, model)?;
        let typed = program.typed_observable(observable).map_err(|errors| {
            errors
                .into_iter()
                .next()
                .expect("failed typing has diagnostic")
        })?;
        let Some(KernelNode::Observable(definition)) = program.node(observable.erase()) else {
            return Err(invalid("Observable is missing from this Model"));
        };
        let ObservableReduction::SpatialIntegral { domain, .. } = definition.reduction() else {
            return Err(invalid(
                "Observable State JVP currently requires a spatial functional",
            ));
        };
        spatial::integrate(
            self,
            &program,
            definition,
            &typed,
            domain,
            quadrature,
            Some(&tangent.fields),
        )
    }
}
