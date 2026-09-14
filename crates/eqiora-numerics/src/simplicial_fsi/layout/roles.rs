//! Exact current FSI projection roles, derived from equations and the admitted Plan.
use super::*;
use crate::form_compiler::equation_roles::{EquationRoles, Role};
use eqiora_realization::{CoupledFieldwiseRealizationPlan, Space};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct FsiRoles {
    pub(super) connection: RawId,
    pub(super) fluid_velocity: RawId,
    pub(super) pressure: RawId,
    pub(super) solid_velocity: RawId,
    pub(super) bindings: BTreeMap<RawId, (RawId, Space)>,
}

impl FsiRoles {
    pub(super) fn derive(
        equations: &EquationRoles,
        plan: &CoupledFieldwiseRealizationPlan,
    ) -> Result<Self, Diagnostic> {
        let domains = plan.spatial().domains();
        let state = plan.time_step().eliminated_state().pair();
        let state_rates = equations
            .relations
            .values()
            .filter_map(|relation| match relation.kind {
                Role::Kinematic { state, rate } => Some((state, rate)),
                _ => None,
            })
            .collect::<Vec<_>>();
        if state_rates != [(state.state().erase(), state.rate().erase())] {
            return Err(invalid(
                "FSI Plan state/rate differs from its complete supported kinematic inventory",
            ));
        }
        let [quotient] = plan.spatial().trace_quotients() else {
            return Err(invalid(
                "FSI reaction/state projection does not yet support plural trace quotients",
            ));
        };
        let endpoints = quotient.endpoints();
        let Some(solid) = endpoints
            .iter()
            .find(|endpoint| endpoint.field() == state.rate())
        else {
            return Err(invalid(
                "FSI trace quotient omits the exact state-rate Field",
            ));
        };
        let fluid = endpoints
            .iter()
            .find(|endpoint| endpoint.field() != state.rate())
            .ok_or_else(|| invalid("FSI quotient has no distinct coupled velocity Field"))?;
        let pressures = equations
            .constraints
            .values()
            .filter_map(|&(constrained, tested)| {
                (constrained == fluid.field().erase()).then_some(tested)
            })
            .collect::<BTreeSet<_>>();
        let pressure = match pressures.iter().copied().collect::<Vec<_>>().as_slice() {
            [pressure] => *pressure,
            _ => {
                return Err(invalid(
                    "FSI projection requires one exact scalar constraint paired with the quotient velocity",
                ));
            }
        };
        let bindings = domains
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
        let roles = Self {
            connection: quotient.connection().erase(),
            fluid_velocity: fluid.field().erase(),
            pressure,
            solid_velocity: solid.field().erase(),
            bindings,
        };
        let expected = BTreeSet::from([roles.fluid_velocity, roles.pressure, roles.solid_velocity]);
        let tested = equations
            .relations
            .values()
            .filter_map(|relation| match relation.kind {
                Role::Residual { tested } => Some(tested),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        if roles.bindings.keys().copied().collect::<BTreeSet<_>>() != expected || tested != expected
        {
            return Err(invalid(
                "FSI reaction/state projection does not yet support additional algebraic Field roles",
            ));
        }
        let p1 = Space::continuous_lagrange(std::num::NonZeroU16::MIN);
        for (field, domain, space) in [
            (
                roles.fluid_velocity,
                fluid.domain().erase(),
                Space::simplex_p1_bubble(),
            ),
            (roles.pressure, fluid.domain().erase(), p1),
            (roles.solid_velocity, solid.domain().erase(), p1),
        ] {
            if roles.bindings.get(&field) != Some(&(domain, space))
                || equations.fields.get(&field).map(|(owner, _)| *owner) != Some(domain)
            {
                return Err(invalid(
                    "FSI role differs from its exact equation Domain or admitted discrete space",
                ));
            }
        }
        Ok(roles)
    }
}

#[cfg(test)]
mod tests;
