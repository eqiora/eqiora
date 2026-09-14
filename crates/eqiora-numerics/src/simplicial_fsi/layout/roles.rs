//! Exact mathematical Field roles derived once from equations and the complete Plan.
use super::*;
use crate::form_compiler::equation_roles::{EquationRoles, Role};
use eqiora_realization::{ConformingTraceQuotient, CoupledFieldwiseRealizationPlan, Space};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct FsiRoles {
    pub(super) velocities: BTreeMap<RawId, RawId>,
    pub(super) constraints: BTreeMap<RawId, RawId>,
    pub(super) quotients: Vec<ConformingTraceQuotient>,
    pub(super) bindings: BTreeMap<RawId, (RawId, Space)>,
}

impl FsiRoles {
    pub(super) fn derive(
        equations: &EquationRoles,
        plan: &CoupledFieldwiseRealizationPlan,
    ) -> Result<Self, Diagnostic> {
        let states = plan.time_step().eliminated_states().to_vec();
        let expected_states = states
            .iter()
            .map(|state| {
                let pair = state.pair();
                (
                    pair.relation().erase(),
                    (pair.state().erase(), pair.rate().erase()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let actual_states = equations
            .relations
            .iter()
            .filter_map(|(&id, relation)| match relation.kind {
                Role::Kinematic { state, rate } => Some((id, (state, rate))),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        if expected_states != actual_states {
            return Err(invalid(
                "Plan state/rate Relations differ from the complete equation inventory",
            ));
        }
        let bindings = plan
            .spatial()
            .domains()
            .iter()
            .flat_map(|domain| {
                domain.field_spaces().iter().map(move |field| {
                    (
                        field.field().erase(),
                        (domain.domain().erase(), field.space()),
                    )
                })
            })
            .collect::<BTreeMap<_, _>>();
        let mut constraints = BTreeMap::new();
        for &(constrained, tested) in equations.constraints.values() {
            if constraints.insert(tested, constrained).is_some() {
                return Err(invalid(
                    "constraint multiplier has repeated equation ownership",
                ));
            }
        }
        // The current independent pressure-closure check certifies one constant
        // mode. Plural pressure spaces require a whole-subspace rank witness.
        if constraints.len() != 1 {
            return Err(invalid(
                "pressure closure currently requires one exact constant-mode inventory; plural pressure subspace closure is unsupported",
            ));
        }
        let p1 = Space::continuous_lagrange(std::num::NonZeroU16::MIN);
        let mut velocities = BTreeMap::new();
        let mut expected = BTreeSet::new();
        let mut admit = |field: RawId, space: Space| -> Result<RawId, Diagnostic> {
            let &(domain, admitted) = bindings
                .get(&field)
                .ok_or_else(|| invalid("equation Field is absent from the complete Plan"))?;
            if admitted != space
                || equations.fields.get(&field).map(|(owner, _)| *owner) != Some(domain)
            {
                return Err(invalid(
                    "equation Field differs from exact Domain/space binding",
                ));
            }
            expected.insert(field);
            Ok(domain)
        };
        for (&multiplier, &velocity) in &constraints {
            let domain = admit(velocity, Space::simplex_p1_bubble())?;
            if admit(multiplier, p1)? != domain {
                return Err(invalid(
                    "constraint and tested multiplier have different exact Domains",
                ));
            }
            if velocities
                .insert(domain, velocity)
                .is_some_and(|old| old != velocity)
            {
                return Err(invalid(
                    "Region has multiple incompatible velocity execution witnesses",
                ));
            }
        }
        for state in &states {
            let rate = state.pair().rate().erase();
            let domain = admit(rate, p1)?;
            if velocities
                .insert(domain, rate)
                .is_some_and(|old| old != rate)
            {
                return Err(invalid(
                    "Region has multiple incompatible state-rate execution witnesses",
                ));
            }
        }
        let tested = equations
            .relations
            .values()
            .filter_map(|relation| match relation.kind {
                Role::Residual { tested } => Some(tested),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        if expected != tested || expected != bindings.keys().copied().collect() {
            return Err(invalid(
                "projection requires complete exact equation and algebraic Field inventories",
            ));
        }
        let quotients = plan.spatial().trace_quotients().to_vec();
        for quotient in &quotients {
            for endpoint in quotient.endpoints() {
                if velocities.get(&endpoint.domain().erase()) != Some(&endpoint.field().erase()) {
                    return Err(invalid(
                        "quotient endpoint differs from its exact Region velocity witness",
                    ));
                }
            }
        }
        Ok(Self {
            velocities,
            constraints,
            quotients,
            bindings,
        })
    }
}

#[cfg(test)]
mod tests;
