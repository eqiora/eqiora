use std::collections::BTreeMap;
use std::ops::Range;

use eqiora_core::{Diagnostic, DimExponents, DynQuantity, RawId, ValueType};
use eqiora_meshing::{ReferenceCell, ReferenceCellFamily};
use eqiora_realization::{BackwardEulerStateBinding, Space, SpaceFamily};

use crate::discrete_space::{
    DiscreteSpace, HypercubeQ1Space, SimplexP1BubbleSpace, SimplexP1Space,
};

use super::{CompiledRegionForm, Role, components, invalid};

#[derive(Debug, Clone, Copy)]
pub(crate) struct RegionFieldBinding {
    pub(crate) field: RawId,
    pub(crate) space: Space,
    pub(crate) scale: DynQuantity,
}

#[derive(Debug, Clone)]
pub(crate) struct RegionTimeBinding {
    pub(crate) step: DynQuantity,
    pub(crate) states: Vec<BackwardEulerStateBinding>,
}

/// Field-major, basis-major, Cartesian-component-minor local ordering.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegionFieldLayout {
    pub(crate) field: RawId,
    pub(crate) value_type: ValueType,
    pub(crate) space: Space,
    pub(crate) range: Range<usize>,
    pub(crate) components: usize,
    pub(crate) scale: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundRegionForm {
    pub(super) form: CompiledRegionForm,
    pub(super) reference: ReferenceCell,
    pub(super) fields: Vec<RegionFieldLayout>,
    pub(super) previous: BTreeMap<RawId, RegionFieldLayout>,
    pub(super) eliminations: BTreeMap<RawId, RawId>,
    pub(super) step: Option<f64>,
    pub(super) row_multipliers: Vec<f64>,
}

impl CompiledRegionForm {
    pub(crate) fn bind(
        &self,
        reference: ReferenceCell,
        fields: &[RegionFieldBinding],
        row_multipliers: &BTreeMap<RawId, DynQuantity>,
        time: Option<&RegionTimeBinding>,
    ) -> Result<BoundRegionForm, Diagnostic> {
        if reference.dimension() != self.dimension
            || fields.len() != self.rows.len()
            || row_multipliers.len() != self.rows.len()
        {
            return Err(invalid(
                "region binding requires exact Field, row and reference dimension coverage",
            ));
        }
        let by_field = fields
            .iter()
            .map(|binding| (binding.field, binding))
            .collect::<BTreeMap<_, _>>();
        if by_field.len() != fields.len() {
            return Err(invalid("duplicate region Field binding"));
        }
        let mut offset = 0usize;
        let mut layouts = Vec::new();
        let mut multipliers = Vec::new();
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("length");
        let measure = length
            .pow(self.dimension as i32, 1)
            .expect("dimension at most three");
        for row in &self.rows {
            let binding = by_field
                .get(&row.tested)
                .ok_or_else(|| invalid("missing exact region Field binding"))?;
            let value_type = &self.roles.fields[&row.tested].1;
            positive_scale(binding.scale, value_type)?;
            let count = components(value_type, self.dimension)?;
            let local = basis(binding.space, reference)?
                .local_dofs()
                .len()
                .checked_mul(count)
                .ok_or_else(|| invalid("region local DOF count overflow"))?;
            let end = offset
                .checked_add(local)
                .ok_or_else(|| invalid("region local DOF count overflow"))?;
            layouts.push(RegionFieldLayout {
                field: row.tested,
                value_type: value_type.clone(),
                space: binding.space,
                range: offset..end,
                components: count,
                scale: binding.scale.value(),
            });
            offset = end;
            let multiplier = row_multipliers
                .get(&row.relation)
                .ok_or_else(|| invalid("missing exact residual-row normalization"))?;
            if !multiplier.value().is_finite()
                || multiplier.value() == 0.0
                || row
                    .value_type
                    .dimension()
                    .mul(measure)
                    .and_then(|dim| dim.mul(multiplier.dim()))
                    != Some(DimExponents::DIMENSIONLESS)
            {
                return Err(invalid(
                    "row normalization must cancel exact residual units times geometric measure",
                ));
            }
            multipliers.push(multiplier.value());
        }
        let expected = self
            .roles
            .relations
            .values()
            .filter_map(|role| match role.kind {
                Role::Kinematic { state, rate } => Some((state, rate)),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let mut eliminations = BTreeMap::new();
        let mut states = BTreeMap::new();
        let second = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("time");
        if let Some(time) = time {
            if time.step.dim() != second
                || !time.step.value().is_finite()
                || time.step.value() <= 0.0
            {
                return Err(invalid(
                    "region Backward Euler step must be positive finite time",
                ));
            }
            for state in &time.states {
                let pair = state.pair();
                let id = pair.state().erase();
                let rate = pair.rate().erase();
                if eliminations.insert(id, rate).is_some()
                    || expected.get(&id) != Some(&rate)
                    || self
                        .roles
                        .relations
                        .get(&pair.relation().erase())
                        .is_none_or(|relation| {
                            relation.kind != (Role::Kinematic { state: id, rate })
                        })
                {
                    return Err(invalid(
                        "Plan state elimination differs from exact Model kinematics",
                    ));
                }
                let layout = layouts
                    .iter()
                    .find(|layout| layout.field == rate)
                    .ok_or_else(|| invalid("eliminated state rate is not an algebraic Field"))?;
                let value_type = &self.roles.fields[&id].1;
                positive_scale(state.state_scale().quantity(), value_type)?;
                if state.state_space() != layout.space
                    || value_type.dimension().div(second) != Some(layout.value_type.dimension())
                    || value_type.shape() != layout.value_type.shape()
                    || value_type.frame() != layout.value_type.frame()
                {
                    return Err(invalid(
                        "state elimination requires matching typed rate and discrete space",
                    ));
                }
                states.insert(
                    id,
                    RegionFieldLayout {
                        field: id,
                        value_type: value_type.clone(),
                        space: layout.space,
                        range: 0..layout.range.len(),
                        components: layout.components,
                        scale: state.state_scale().quantity().value(),
                    },
                );
            }
        }
        if expected != eliminations {
            return Err(invalid("Plan must bind every exact Model kinematic pair"));
        }
        let mut previous = BTreeMap::new();
        for term in self.rows.iter().flat_map(|row| &row.terms) {
            if let Some(state) = states.get(&term.trial) {
                if !term.derivative {
                    previous.insert(term.trial, state.clone());
                }
            } else {
                let layout = layouts
                    .iter()
                    .find(|layout| layout.field == term.trial)
                    .ok_or_else(|| invalid("trial Field lacks an algebraic or state binding"))?;
                if term.derivative {
                    if time.is_none() {
                        return Err(invalid(
                            "time derivative requires explicit Backward Euler Plan",
                        ));
                    }
                    let mut layout = layout.clone();
                    layout.range = 0..layout.range.len();
                    previous.insert(term.trial, layout);
                }
            }
        }
        Ok(BoundRegionForm {
            form: self.clone(),
            reference,
            fields: layouts,
            previous,
            eliminations,
            step: time.map(|time| time.step.value()),
            row_multipliers: multipliers,
        })
    }
}

impl BoundRegionForm {
    pub(crate) const fn domain(&self) -> RawId {
        self.form.domain()
    }

    pub(crate) const fn reference_cell(&self) -> ReferenceCell {
        self.reference
    }

    pub(crate) fn fields(&self) -> &[RegionFieldLayout] {
        &self.fields
    }
    /// Required physical previous coefficients, independently indexed within each Field.
    pub(crate) fn previous_fields(&self) -> &BTreeMap<RawId, RegionFieldLayout> {
        &self.previous
    }
}

fn positive_scale(scale: DynQuantity, value_type: &ValueType) -> Result<(), Diagnostic> {
    if scale.dim() != value_type.dimension() || !scale.value().is_finite() || scale.value() <= 0.0 {
        return Err(invalid(
            "Field scale must be positive finite and match its exact ValueType units",
        ));
    }
    Ok(())
}

pub(crate) fn basis(
    space: Space,
    reference: ReferenceCell,
) -> Result<Box<dyn DiscreteSpace>, Diagnostic> {
    match (space.family(), reference.family()) {
        (SpaceFamily::ContinuousLagrange { order }, ReferenceCellFamily::Simplex)
            if order.get() == 1 =>
        {
            Ok(Box::new(SimplexP1Space::new(reference.dimension())?))
        }
        (SpaceFamily::ContinuousLagrange { order }, ReferenceCellFamily::Hypercube)
            if order.get() == 1 =>
        {
            Ok(Box::new(HypercubeQ1Space::new(reference.dimension())?))
        }
        (SpaceFamily::SimplexP1Bubble, ReferenceCellFamily::Simplex) => {
            Ok(Box::new(SimplexP1BubbleSpace::new(reference.dimension())?))
        }
        _ => Err(invalid(
            "region form requires P1, Q1 or simplex P1-bubble bases",
        )),
    }
}
