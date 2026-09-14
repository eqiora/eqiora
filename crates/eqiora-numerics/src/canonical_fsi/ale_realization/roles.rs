//! Exact semantic identities for the admitted ALE action.

use super::*;

pub(super) fn field_identities<const D: usize>(
    model: &AleFsiCartesianModel<D>,
) -> AleFsiFieldIdentities<D> {
    AleFsiFieldIdentities::<D> {
        fluid_velocity: fluid_velocity(model),
        fluid_pressure: fluid_pressure(model),
        solid_velocity: solid_velocity(model),
        solid_displacement: solid_displacement(model),
    }
}

pub(super) fn trace_quotient<const D: usize>(
    model: &AleFsiCartesianModel<D>,
) -> ConformingTraceQuotient {
    ConformingTraceQuotient::new(
        connection(model),
        TraceFieldEndpoint::new(fluid_domain(model), fluid_velocity(model)),
        TraceFieldEndpoint::new(solid_domain(model), solid_velocity(model)),
    )
    .expect("lowered ALE FSI interface joins distinct Domains")
}

pub(super) fn state_pair<const D: usize>(
    model: &AleFsiCartesianModel<D>,
) -> BackwardEulerStatePair {
    BackwardEulerStatePair::new(
        solid_kinematic_relation(model),
        solid_displacement(model),
        solid_velocity(model),
    )
    .expect("lowered ALE FSI solid state and rate are distinct")
}

pub(super) fn fluid_domain<const D: usize>(model: &AleFsiCartesianModel<D>) -> Id<kinds::Domain> {
    model
        .fluid()
        .domain()
        .downcast()
        .expect("lowered ALE fluid Domain retains its kind")
}

pub(super) fn solid_domain<const D: usize>(model: &AleFsiCartesianModel<D>) -> Id<kinds::Domain> {
    let domain = model.solid().continuum().domain();
    domain
        .downcast()
        .expect("lowered ALE solid Domain retains its kind")
}

pub(super) fn fluid_velocity<const D: usize>(model: &AleFsiCartesianModel<D>) -> Id<kinds::Field> {
    model
        .fluid()
        .velocity()
        .downcast()
        .expect("lowered ALE fluid velocity retains its kind")
}

pub(super) fn fluid_pressure<const D: usize>(model: &AleFsiCartesianModel<D>) -> Id<kinds::Field> {
    model
        .fluid()
        .pressure()
        .downcast()
        .expect("lowered ALE pressure retains its kind")
}

pub(super) fn solid_velocity<const D: usize>(model: &AleFsiCartesianModel<D>) -> Id<kinds::Field> {
    model
        .solid()
        .velocity()
        .downcast()
        .expect("lowered ALE solid velocity retains its kind")
}

pub(super) fn solid_displacement<const D: usize>(
    model: &AleFsiCartesianModel<D>,
) -> Id<kinds::Field> {
    let displacement = model.solid().continuum().displacement();
    displacement
        .downcast()
        .expect("lowered ALE solid displacement retains its kind")
}

pub(super) fn fluid_relation<const D: usize>(
    model: &AleFsiCartesianModel<D>,
) -> Id<kinds::Relation> {
    model
        .fluid()
        .momentum_relation()
        .downcast()
        .expect("lowered ALE fluid momentum retains its kind")
}

pub(super) fn solid_kinematic_relation<const D: usize>(
    model: &AleFsiCartesianModel<D>,
) -> Id<kinds::Relation> {
    model
        .solid()
        .kinematic_relation()
        .downcast()
        .expect("lowered ALE solid kinematics retains its kind")
}

pub(super) fn connection<const D: usize>(model: &AleFsiCartesianModel<D>) -> Id<kinds::Connection> {
    model
        .interface()
        .connection()
        .downcast()
        .expect("lowered ALE FSI Connection retains its kind")
}
