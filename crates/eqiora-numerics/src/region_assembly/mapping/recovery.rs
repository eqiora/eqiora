//! Complete typed Field recovery from the single exact global coordinate map.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecoveredRegionField<S: Coefficient> {
    pub(crate) domain: RawId,
    pub(crate) value_type: eqiora_core::ValueType,
    /// The exact coefficient functionals, including moment versus nodal meaning.
    pub(crate) space: eqiora_realization::Space,
    pub(crate) coefficients: BTreeMap<FieldDof, S>,
}

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    /// Recover every requested exact Field, including essential and quotient values.
    /// The request must equal the complete admitted inventory; order is immaterial.
    pub(crate) fn recover(
        &self,
        reduced: &[S],
        requested: &[RawId],
    ) -> Result<BTreeMap<RawId, RecoveredRegionField<S>>, Diagnostic> {
        let inventory = requested.iter().copied().collect::<BTreeSet<_>>();
        if inventory.len() != requested.len() || inventory != self.fields.keys().copied().collect()
        {
            return Err(invalid(
                "Field recovery requires the complete exact mapped inventory",
            ));
        }
        let full = self.constraints.lift(reduced)?;
        let mut recovered = self
            .fields
            .iter()
            .map(|(&field, (domain, layout))| {
                (
                    field,
                    RecoveredRegionField {
                        domain: *domain,
                        value_type: layout.value_type.clone(),
                        space: layout.space,
                        coefficients: BTreeMap::new(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        for (key, index) in &self.globals {
            let physical = full[*index] * self.fields[&key.field].1.scale;
            if !physical.is_finite() {
                return Err(invalid(
                    "Field recovery produced a nonfinite physical value",
                ));
            }
            recovered
                .get_mut(&key.field)
                .expect("authenticated Field coordinate")
                .coefficients
                .insert(*key, physical);
        }
        if recovered
            .values()
            .any(|field| field.coefficients.is_empty())
        {
            return Err(invalid(
                "Field recovery omitted exact supporting coordinates",
            ));
        }
        Ok(recovered)
    }
}

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    /// Recover all physical Fields and apply every admitted exact state-rate relation.
    /// All validation and arithmetic complete before the new inventory is returned.
    pub(crate) fn recover_step(
        &self,
        reduced: &[S],
        previous: &BTreeMap<RawId, RecoveredRegionField<S>>,
        step: &eqiora_realization::BackwardEulerStep,
        prescribed_states: &BTreeMap<FieldDof, S>,
    ) -> Result<BTreeMap<RawId, RecoveredRegionField<S>>, Diagnostic> {
        self.validate_step_history(previous, step)?;
        self.validate_state_constraints(previous, step, prescribed_states)?;
        let requested = self.fields.keys().copied().collect::<Vec<_>>();
        let mut recovered = self.recover(reduced, &requested)?;
        for state in step.eliminated_states() {
            let pair = state.pair();
            let old = &previous[&pair.state().erase()];
            let rate = recovered
                .get(&pair.rate().erase())
                .expect("validated exact rate");
            let mut next = old.clone();
            for (&key, value) in &mut next.coefficients {
                let rate_key = FieldDof {
                    field: pair.rate().erase(),
                    ..key
                };
                *value = match prescribed_states.get(&key) {
                    // Strong physical boundary data is lifted exactly. The matching
                    // rate constraint has already been checked against old history.
                    Some(prescribed) => *prescribed,
                    None => *value + rate.coefficients[&rate_key] * step.duration().value(),
                };
                if !value.is_finite() {
                    return Err(invalid("step state recovery produced a nonfinite value"));
                }
            }
            if recovered.insert(pair.state().erase(), next).is_some() {
                return Err(invalid(
                    "step recovery duplicates an exact represented Field",
                ));
            }
        }
        Ok(recovered)
    }
}

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    /// Validate exact physical Field inventory and equality of shared quotient coordinates.
    pub(crate) fn validate_physical(
        &self,
        fields: &BTreeMap<RawId, RecoveredRegionField<S>>,
    ) -> Result<(), Diagnostic> {
        let mut globals = vec![None; self.full_count()];
        for (&id, (domain, layout)) in &self.fields {
            let field = fields
                .get(&id)
                .ok_or_else(|| invalid("history omits an exact algebraic Field"))?;
            if field.domain != *domain
                || field.value_type != layout.value_type
                || field.space != layout.space
                || field.coefficients.keys().copied().collect::<BTreeSet<_>>()
                    != self.keys().filter(|key| key.field == id).collect()
            {
                return Err(invalid(
                    "history has stale Field/Domain/type/Space or coordinate ownership",
                ));
            }
            for (&key, &physical) in &field.coefficients {
                let value = physical / layout.scale;
                let global = self.global_dof(key).expect("exact coordinate inventory");
                if !value.is_finite() || globals[global].is_some_and(|old| old != value) {
                    return Err(invalid(
                        "history is nonfinite or disagrees on an exact trace quotient",
                    ));
                }
                globals[global] = Some(value);
            }
        }
        Ok(())
    }
}

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    pub(crate) fn validate_step_history(
        &self,
        previous: &BTreeMap<RawId, RecoveredRegionField<S>>,
        step: &eqiora_realization::BackwardEulerStep,
    ) -> Result<(), Diagnostic> {
        let expected = self
            .fields
            .keys()
            .copied()
            .chain(
                step.eliminated_states()
                    .iter()
                    .map(|state| state.pair().state().erase()),
            )
            .collect::<BTreeSet<_>>();
        if previous.keys().copied().collect::<BTreeSet<_>>() != expected {
            return Err(invalid(
                "step history differs from complete physical Field inventory",
            ));
        }
        self.validate_physical(previous)?;
        for state in step.eliminated_states() {
            let pair = state.pair();
            let old = &previous[&pair.state().erase()];
            let (domain, rate) = self
                .field_layout(pair.rate().erase())
                .ok_or_else(|| invalid("step state has no exact algebraic rate"))?;
            crate::form_compiler::region::state_layout(*state, &old.value_type, rate)?;
            let keys = self
                .keys()
                .filter(|key| key.field == pair.rate().erase())
                .map(|key| FieldDof {
                    field: pair.state().erase(),
                    ..key
                })
                .collect::<BTreeSet<_>>();
            if old.domain != domain
                || old.space != state.state_space()
                || old.coefficients.keys().copied().collect::<BTreeSet<_>>() != keys
                || old.coefficients.values().any(|value| !value.is_finite())
            {
                return Err(invalid(
                    "step state differs from exact rate Domain/type or coordinates",
                ));
            }
        }
        Ok(())
    }
}

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    /// A state boundary must agree with the rate constraint used by the solve.
    pub(crate) fn validate_state_constraints(
        &self,
        previous: &BTreeMap<RawId, RecoveredRegionField<S>>,
        step: &eqiora_realization::BackwardEulerStep,
        prescribed: &BTreeMap<FieldDof, S>,
    ) -> Result<(), Diagnostic> {
        if prescribed.is_empty() {
            return Ok(());
        }
        let lifted = self.lift(&vec![<S as From<f64>>::from(0.0); self.free_count()], false)?;
        for (key, value) in prescribed {
            let state = step
                .eliminated_states()
                .iter()
                .find(|state| state.pair().state().erase() == key.field)
                .ok_or_else(|| invalid("boundary data is not an eliminated state coordinate"))?;
            let old = previous
                .get(&key.field)
                .and_then(|field| field.coefficients.get(key))
                .ok_or_else(|| invalid("state boundary omits its exact previous coordinate"))?;
            let rate_key = FieldDof {
                field: state.pair().rate().erase(),
                ..*key
            };
            let global = self
                .global_dof(rate_key)
                .ok_or_else(|| invalid("state boundary lacks its exact mapped rate coordinate"))?;
            let (_, layout) = self.field_layout(rate_key.field).expect("mapped rate");
            let expected = (*value - *old) / step.duration().value() / layout.scale;
            if !value.is_finite()
                || !expected.is_finite()
                || self.free_dof(rate_key).is_some()
                || lifted[global] != expected
            {
                return Err(invalid(
                    "state boundary differs from its history-dependent rate constraint",
                ));
            }
        }
        Ok(())
    }
}
