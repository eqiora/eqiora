//! Borrow one exact accepted occurrence without manufacturing another Result.
use std::collections::HashMap;

use eqiora_core::{Diagnostic, Id, entity::kinds};

use super::super::{CommonFieldAssociation, CommonResult, CommonResultPayload, invalid};

#[derive(Clone, Copy)]
pub(super) struct FieldCoefficients<'a> {
    pub(super) association: CommonFieldAssociation,
    pub(super) values: &'a [f64],
}

pub(super) struct Occurrence<'a> {
    pub(super) result: &'a CommonResult,
    output: Option<(usize, &'a crate::CommonState)>,
    fields: HashMap<Id<kinds::Field>, &'a [f64]>,
}

impl<'a> Occurrence<'a> {
    pub(super) fn new(
        result: &'a CommonResult,
        output_index: Option<usize>,
    ) -> Result<Self, Diagnostic> {
        let output = match (&result.payload, output_index) {
            (CommonResultPayload::Trajectory { trajectory, .. }, Some(index)) => {
                let steps = trajectory.spatial_steps().ok_or_else(|| {
                    invalid("output-index observations require an admitted spatial Trajectory")
                })?;
                let (_, step) = steps.get(index).ok_or_else(|| {
                    invalid("observation output index is outside the retained schedule")
                })?;
                Some((index, step.state()))
            }
            (CommonResultPayload::Trajectory { .. }, None) => {
                return Err(invalid(
                    "a Trajectory observation requires an explicit output index",
                ));
            }
            (_, Some(_)) => {
                return Err(invalid(
                    "an instantaneous Result does not have output indices",
                ));
            }
            (_, None) => None,
        };
        let fields = if let Some((_, state)) = output {
            let plan = result.plan().as_spatial().ok_or_else(|| {
                invalid("output observation requires the common spatial Field realization")
            })?;
            plan.require_fixed_observation_geometry()?;
            plan.state_fields(state)?
                .map(|(field, _, _, values)| (field, values))
                .collect()
        } else {
            HashMap::new()
        };
        Ok(Self {
            result,
            output,
            fields,
        })
    }

    pub(super) fn plan(&self) -> &crate::ResolvedCommonPlan {
        self.result.plan()
    }

    pub(super) fn time_s(&self) -> Option<f64> {
        self.output.map(|(_, state)| state.time_s())
    }

    pub(super) fn output(&self) -> Option<(usize, f64, String)> {
        self.output
            .map(|(index, state)| (index, state.time_s(), state.identity().to_owned()))
    }

    pub(super) fn field(&self, id: Id<kinds::Field>) -> Result<FieldCoefficients<'_>, Diagnostic> {
        if self.output.is_some() {
            let values = self.fields.get(&id).ok_or_else(|| {
                invalid("Observable required Field is absent from the accepted State")
            })?;
            let (_, layouts) = super::super::linear::field_layout(
                self.plan()
                    .as_spatial()
                    .expect("admitted spatial occurrence"),
                id,
            )?;
            let [(association, _)] = layouts.as_slice() else {
                return Err(invalid("Observable requires exactly one coefficient block"));
            };
            return Ok(FieldCoefficients {
                association: *association,
                values,
            });
        }
        let CommonResultPayload::Static(payload) = &self.result.payload else {
            return Err(invalid(
                "spatial Observable requires accepted spatial coefficients",
            ));
        };
        let field = payload
            .fields
            .iter()
            .find(|field| field.field_id == id.ulid().to_string())
            .ok_or_else(|| invalid("Observable required Field is absent from accepted Result"))?;
        let [block] = field.blocks.as_slice() else {
            return Err(invalid("Observable requires exactly one coefficient block"));
        };
        Ok(FieldCoefficients {
            association: block.association,
            values: &block.values,
        })
    }
}
