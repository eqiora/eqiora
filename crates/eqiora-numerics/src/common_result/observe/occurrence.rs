//! Borrow one exact accepted occurrence without manufacturing another Result.
use std::collections::HashMap;

use eqiora_core::{Diagnostic, Id, entity::kinds};

use super::super::{CommonFieldAssociation, CommonResult, CommonResultPayload, invalid};
use crate::{CommonState, ResolvedCommonPlan};

#[derive(Clone, Copy)]
pub(super) struct FieldCoefficients<'a> {
    pub(super) association: CommonFieldAssociation,
    pub(super) values: &'a [f64],
}

enum Source<'a> {
    Result(&'a CommonResult),
    State(&'a ResolvedCommonPlan, &'a CommonState),
}

pub(super) struct Occurrence<'a> {
    source: Source<'a>,
    output_index: Option<usize>,
    fields: HashMap<Id<kinds::Field>, &'a [f64]>,
}

impl<'a> Occurrence<'a> {
    pub(super) fn new(
        result: &'a CommonResult,
        output_index: Option<usize>,
    ) -> Result<Self, Diagnostic> {
        match (&result.payload, output_index) {
            (CommonResultPayload::Trajectory { trajectory, .. }, Some(index)) => {
                let steps = trajectory.spatial_steps().ok_or_else(|| {
                    invalid("output-index observations require an admitted spatial Trajectory")
                })?;
                let (_, step) = steps.get(index).ok_or_else(|| {
                    invalid("observation output index is outside the retained schedule")
                })?;
                let mut occurrence = Self::from_state(result.plan(), step.state())?;
                occurrence.output_index = Some(index);
                Ok(occurrence)
            }
            (CommonResultPayload::Trajectory { .. }, None) => Err(invalid(
                "a Trajectory observation requires an explicit output index",
            )),
            (_, Some(_)) => Err(invalid(
                "an instantaneous Result does not have output indices",
            )),
            (_, None) => Ok(Self {
                source: Source::Result(result),
                output_index: None,
                fields: HashMap::new(),
            }),
        }
    }

    pub(super) fn from_state(
        plan: &'a ResolvedCommonPlan,
        state: &'a CommonState,
    ) -> Result<Self, Diagnostic> {
        let spatial = plan.as_spatial().ok_or_else(|| {
            invalid("output observation requires the common spatial Field realization")
        })?;
        spatial.require_fixed_observation_geometry()?;
        let fields = spatial
            .state_fields(state)?
            .map(|(field, _, _, values)| (field, values))
            .collect();
        Ok(Self {
            source: Source::State(plan, state),
            output_index: None,
            fields,
        })
    }

    pub(super) fn plan(&self) -> &ResolvedCommonPlan {
        match self.source {
            Source::Result(result) => result.plan(),
            Source::State(plan, _) => plan,
        }
    }

    pub(super) fn finite_values(&self) -> Option<&[f64]> {
        match self.source {
            Source::Result(result) => result.finite_values(),
            Source::State(..) => None,
        }
    }

    pub(super) fn time_s(&self) -> Option<f64> {
        match self.source {
            Source::State(_, state) => Some(state.time_s()),
            Source::Result(_) => None,
        }
    }

    pub(super) fn output(&self) -> Option<(usize, f64, String)> {
        match self.source {
            Source::State(_, state) => self
                .output_index
                .map(|index| (index, state.time_s(), state.identity().to_owned())),
            Source::Result(_) => None,
        }
    }

    pub(super) fn field(&self, id: Id<kinds::Field>) -> Result<FieldCoefficients<'_>, Diagnostic> {
        if let Source::State(..) = self.source {
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
        let Source::Result(result) = self.source else {
            unreachable!()
        };
        let CommonResultPayload::Static(payload) = &result.payload else {
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
