//! Persist static selections; derive their types from the exact admitted Plan.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::*;

mod quadrature;

const SCHEMA: &str = "eqiora.differentiable-program-identity/v2";
const MAX_BYTES: usize = 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireIdentity {
    schema: String,
    model: String,
    plan: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    initial_state: Option<String>,
    quadratures: BTreeMap<String, quadrature::Rule>,
}

impl DifferentiableProgramIdentity {
    /// Encode ordered selections and exact Model, Plan and initial-State references.
    ///
    /// This is static meaning, not an accepted evaluation. The owning collection
    /// must also retain the referenced Plan and any initial State through their
    /// existing artifact codecs. No provider or numerical state is encoded here.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        let bytes = serde_json::to_vec(&WireIdentity {
            schema: SCHEMA.to_owned(),
            model: self.model.artifact().as_str().to_owned(),
            plan: self.plan_identity.clone(),
            inputs: self
                .inputs
                .iter()
                .map(|value| value.entity().to_string())
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(|value| value.entity().to_string())
                .collect(),
            initial_state: self.initial_state_identity.clone(),
            quadratures: self
                .quadratures
                .iter()
                .map(|(domain, rule)| (domain.erase().to_string(), quadrature::Rule::encode(rule)))
                .collect(),
        })
        .map_err(|error| invalid(format!("cannot encode Program identity: {error}")))?;
        check_size(&bytes)?;
        Ok(bytes)
    }

    /// Reconstruct typed ordered signatures from the exact referenced artifacts.
    ///
    /// Does not solve, draw samples or construct a live differentiable Program.
    /// Types, units, meshes and coefficient order come from the Plan's Model and
    /// realization; they are never inferred from array positions or source names.
    ///
    /// # Errors
    /// Rejects foreign owners, missing or repeated selections, invalid roles,
    /// mismatched initial States, noncanonical bytes and content over 1 MiB.
    pub fn from_bytes(
        bytes: &[u8],
        plan: &ResolvedCommonPlan,
        initial: Option<&CommonAlgebraicState>,
    ) -> Result<Self, Vec<Diagnostic>> {
        check_size(bytes).map_err(single)?;
        let wire: WireIdentity = serde_json::from_slice(bytes)
            .map_err(|error| single(invalid(format!("invalid Program identity: {error}"))))?;
        let model = plan.model_artifact().artifact_reference().map_err(single)?;
        if wire.schema != SCHEMA
            || wire.model != model.artifact().as_str()
            || wire.plan != plan.identity()
            || wire.initial_state.as_deref() != initial.map(CommonAlgebraicState::identity)
        {
            return Err(single(invalid(
                "Program identity differs from its exact artifacts",
            )));
        }
        let kernel = signature::kernel(plan).map_err(single)?;
        let entities = kernel
            .nodes()
            .map(|node| (node.id().to_string(), node.id()))
            .collect::<BTreeMap<_, _>>();
        let resolve = |id: &String| {
            entities
                .get(id)
                .copied()
                .ok_or_else(|| single(invalid("Program selection is absent from its exact Model")))
        };
        let inputs = wire
            .inputs
            .iter()
            .map(|id| {
                Ok(ModelEntityRef {
                    model: model.clone(),
                    id: resolve(id)?.downcast().ok_or_else(|| {
                        single(invalid("Program input requires a Parameter identity"))
                    })?,
                })
            })
            .collect::<Result<Vec<_>, Vec<Diagnostic>>>()?;
        let outputs = wire
            .outputs
            .iter()
            .map(|id| {
                Ok(ModelEntityRef {
                    model: model.clone(),
                    id: resolve(id)?,
                })
            })
            .collect::<Result<Vec<_>, Vec<Diagnostic>>>()?;
        let rules = wire
            .quadratures
            .into_iter()
            .map(|(domain, rule)| {
                let domain = resolve(&domain)?.downcast().ok_or_else(|| {
                    single(invalid("Program quadrature requires a Domain identity"))
                })?;
                Ok((domain, rule.decode().map_err(single)?))
            })
            .collect::<Result<_, Vec<Diagnostic>>>()?;
        let identity = Self::from_plan(plan, &inputs, &outputs, &rules, initial)?;
        if identity.to_bytes().map_err(single)? != bytes {
            return Err(single(invalid("Program identity bytes are not canonical")));
        }
        Ok(identity)
    }
}

fn check_size(bytes: &[u8]) -> Result<(), Diagnostic> {
    if bytes.len() > MAX_BYTES {
        return Err(invalid("Program identity exceeds its 1 MiB byte limit"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
