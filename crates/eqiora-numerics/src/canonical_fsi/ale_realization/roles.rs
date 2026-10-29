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
        eqiora_realization::ConformingTraceSource::ConservingConnection(connection(model)),
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

pub(super) fn require_zero_load<const D: usize>(
    model: &AleFsiCartesianModel<D>,
) -> Result<(), Diagnostic> {
    let fluid_load = model
        .fluid()
        .force_potential_expression()
        .map(|source| source.constant_value());
    let solid = model.solid().continuum();
    let solid_load = solid.load_potential_expression().constant_value();
    if fluid_load.is_some_and(|value| value != Some(0.0)) || solid_load != Some(0.0) {
        return Err(invalid_realization(
            "fixed-topology ALE FSI v1 requires exact zero canonical fluid and solid load potentials",
        ));
    }
    Ok(())
}

pub(super) fn require_boundary_meaning<const D: usize>(
    model: &AleFsiCartesianModel<D>,
) -> Result<(), Diagnostic> {
    let interface = model.interface();
    require_physics_boundary::<D>(
        |axis, side| model.fluid().cartesian_boundary(axis, side),
        interface.axis(),
        interface
            .endpoint(model.fluid().domain())
            .expect("lowered ALE interface owns the fluid Domain"),
        interface.connection(),
        "fluid",
    )?;
    require_physics_boundary::<D>(
        |axis, side| {
            model
                .solid()
                .continuum()
                .boundary_inventory()
                .boundary(axis, side)
        },
        interface.axis(),
        interface
            .endpoint(model.solid().continuum().domain())
            .expect("lowered ALE interface owns the solid Domain"),
        interface.connection(),
        "solid",
    )
}

fn require_physics_boundary<'a, const D: usize>(
    boundary: impl Fn(
        usize,
        BoundarySide,
    ) -> Option<&'a crate::canonical_boundary::PhysicalBoundaryEntry>,
    interface_axis: usize,
    interface_side: FsiInterfaceSide,
    connection: eqiora_core::RawId,
    physics: &str,
) -> Result<(), Diagnostic> {
    for axis in 0..D {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            let entry = boundary(axis, side).ok_or_else(|| {
                invalid_realization(format!(
                    "fixed-topology ALE FSI {physics} boundary inventory omits axis {axis} {side:?}"
                ))
            })?;
            if axis == interface_axis && side == interface_side.side() {
                if entry.boundary() != interface_side.boundary()
                    || entry.disposition()
                        != (PhysicalBoundaryDisposition::PortBinding {
                            connection,
                            port: interface_side.port(),
                        })
                {
                    return Err(invalid_realization(format!(
                        "fixed-topology ALE FSI {physics} interface boundary identity or live Port binding drifted"
                    )));
                }
            } else if entry.disposition() != PhysicalBoundaryDisposition::TraceZero {
                return Err(invalid_realization(format!(
                    "fixed-topology ALE FSI v1 requires TraceZero on every exterior {physics} side"
                )));
            }
        }
    }
    Ok(())
}
