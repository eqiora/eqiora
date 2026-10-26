//! Borrow one exact accepted occurrence without manufacturing another Result.
use std::collections::HashMap;

use eqiora_core::{Diagnostic, Id, ValueLiteral, entity::kinds};

use super::super::{CommonFieldAssociation, CommonResult, CommonResultPayload, invalid};
use crate::{CommonAlgebraicPlan, CommonSpatialPlan, CommonState, ResolvedCommonPlan};

#[derive(Clone, Copy)]
pub(super) struct FieldCoefficients<'a> {
    pub(super) association: CommonFieldAssociation,
    pub(super) values: &'a [f64],
}

enum Source<'a> {
    Result(&'a CommonResult),
    Spatial {
        plan: &'a CommonSpatialPlan,
        state_identity: Option<&'a str>,
        time_s: Option<f64>,
        parameters: Option<&'a crate::parameter_point::ParameterPoint>,
    },
}

pub(super) struct Occurrence<'a> {
    source: Source<'a>,
    output_index: Option<usize>,
    fields: HashMap<Id<kinds::Field>, &'a [f64]>,
}

impl<'a> Occurrence<'a> {
    pub(super) fn from_point(
        plan: &'a CommonSpatialPlan,
        parameters: Option<&'a crate::parameter_point::ParameterPoint>,
        time_s: Option<f64>,
        fields: impl IntoIterator<Item = (Id<kinds::Field>, &'a [f64])>,
    ) -> Result<Self, Diagnostic> {
        plan.require_fixed_observation_geometry()?;
        if time_s.is_some_and(|time| !time.is_finite()) {
            return Err(invalid("observation point time must be finite"));
        }
        let mut coefficients = HashMap::new();
        for (field, values) in fields {
            let value_type = plan
                .fields()
                .find(|(candidate, _)| *candidate == field)
                .map(|(_, ty)| ty)
                .ok_or_else(|| invalid("observation point Field is outside the exact Plan"))?;
            let (_, layouts) = super::super::spatial::field_layout(plan, field)?;
            let width = if value_type.scalar_domain() == eqiora_core::ScalarDomain::Complex {
                2usize
            } else {
                1usize
            };
            let count = layouts.iter().try_fold(0usize, |total, (_, shape)| {
                shape
                    .iter()
                    .try_fold(width, |count, extent| count.checked_mul(*extent))
                    .and_then(|count| total.checked_add(count))
            });
            if count != Some(values.len())
                || values.iter().any(|value| !value.is_finite())
                || coefficients.insert(field, values).is_some()
            {
                return Err(invalid(
                    "observation point requires complete distinct finite Field coefficients",
                ));
            }
        }
        if coefficients.len() != plan.fields().len() {
            return Err(invalid("observation point omits an exact Plan Field"));
        }
        Ok(Self {
            source: Source::Spatial {
                plan,
                state_identity: None,
                time_s,
                parameters,
            },
            output_index: None,
            fields: coefficients,
        })
    }

    pub(super) fn new(
        result: &'a CommonResult,
        output_index: Option<usize>,
    ) -> Result<Self, Diagnostic> {
        match (&result.payload, output_index) {
            (CommonResultPayload::Trajectory { trajectory, .. }, Some(index)) => {
                let state = trajectory.spatial_state(index).ok_or_else(|| {
                    invalid("observation output index is outside the retained schedule")
                })?;
                let mut occurrence = Self::from_state(result.plan(), state)?;
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
        let fields = spatial
            .state_fields(state)?
            .map(|(field, _, _, values)| (field, values));
        let mut occurrence = Self::from_point(spatial, None, state.time_s(), fields)?;
        if let Source::Spatial {
            state_identity: accepted,
            ..
        } = &mut occurrence.source
        {
            *accepted = Some(state.identity());
        }
        Ok(occurrence)
    }

    pub(super) fn spatial_plan(&self) -> Option<&CommonSpatialPlan> {
        match self.source {
            Source::Result(result) => result.plan().as_spatial(),
            Source::Spatial { plan, .. } => Some(plan),
        }
    }

    pub(super) fn algebraic_plan(&self) -> Option<&CommonAlgebraicPlan> {
        match self.source {
            Source::Result(result) => result.plan().as_algebraic(),
            Source::Spatial { .. } => None,
        }
    }

    pub(super) fn authenticated_mesh(&self) -> Option<crate::AuthenticatedCommonMesh> {
        match self.source {
            Source::Result(result) => result.plan().authenticated_mesh(),
            Source::Spatial { plan, .. } => Some(plan.authenticated_mesh()),
        }
    }

    pub(super) fn finite_values(&self) -> Option<&[f64]> {
        match self.source {
            Source::Result(result) => result.finite_values(),
            Source::Spatial { .. } => None,
        }
    }

    pub(super) fn time_s(&self) -> Option<f64> {
        match self.source {
            Source::Spatial { time_s, .. } => time_s,
            Source::Result(_) => None,
        }
    }

    pub(super) fn output(&self) -> Option<(usize, f64, String)> {
        match self.source {
            Source::Spatial {
                state_identity: Some(identity),
                time_s,
                ..
            } => self
                .output_index
                .zip(time_s)
                .map(|(index, time)| (index, time, identity.to_owned())),
            Source::Result(_)
            | Source::Spatial {
                state_identity: None,
                ..
            } => None,
        }
    }

    pub(super) fn field(&self, id: Id<kinds::Field>) -> Result<FieldCoefficients<'_>, Diagnostic> {
        if matches!(self.source, Source::Spatial { .. }) {
            let values = self.fields.get(&id).ok_or_else(|| {
                invalid("Observable required Field is absent from the accepted spatial point")
            })?;
            let (_, layouts) = super::super::spatial::field_layout(
                self.spatial_plan().expect("admitted spatial occurrence"),
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

    pub(super) fn parameter_point(&self) -> Option<&crate::parameter_point::ParameterPoint> {
        match self.source {
            Source::Spatial { parameters, .. } => parameters,
            Source::Result(_) => None,
        }
    }

    pub(super) fn parameter<'p>(
        &'p self,
        program: &'p eqiora_sem::KernelProgram,
        id: Id<kinds::Parameter>,
    ) -> Option<&'p ValueLiteral> {
        self.parameter_point()
            .map_or_else(|| program.typed_value(id.erase()), |point| point.get(id))
    }
}
