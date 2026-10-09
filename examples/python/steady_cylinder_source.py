"""Author the steady-cylinder equations as one Eqiora Language Source."""

from eqiora import lang as q
from eqiora import Dimension, FieldRole, Module, ValueType

def build_source() -> Module:
    """Return the complete equations-only steady-cylinder Component."""

    source = Module("main")
    stokes = source.component(
        "SteadyFlowPastCylinder",
        doc="Equations-only steady incompressible flow around a cylinder.",
    )
    fluid = stokes.volume("fluid", dimensions=2)
    inlet = stokes.boundary("inlet", parent=fluid)
    outlet = stokes.boundary("outlet", parent=fluid)
    walls = stokes.boundary("walls", parent=fluid)
    cylinder = stokes.boundary("cylinder", parent=fluid)

    dynamic_viscosity = stokes.parameter(
        "dynamic_viscosity", value_type=ValueType.real(Dimension(mass=1, length=-1, time=-1))
    )
    zero_pressure = stokes.parameter(
        "zero_pressure", value_type=ValueType.real(Dimension(mass=1, length=-1, time=-2))
    )
    inlet_speed = stokes.parameter(
        "inlet_speed", value_type=ValueType.real(Dimension(length=1, time=-1))
    )
    channel_height = stokes.parameter("channel_height", value_type=ValueType.real(Dimension(length=1)))

    velocity = stokes.field(
        "velocity", role=FieldRole.Variable,
        on=fluid,
        value_type=ValueType.vector(ValueType.real(Dimension(length=1, time=-1)), 2),
    )
    pressure = stokes.field(
        "pressure", role=FieldRole.Variable,
        on=fluid,
        value_type=ValueType.real(Dimension(mass=1, length=-1, time=-2)),
    )
    force_potential = stokes.field(
        "force_potential", role=FieldRole.Variable,
        on=fluid,
        value_type=ValueType.real(Dimension(mass=1, length=-1, time=-2)),
    )
    inlet_profile = stokes.field("inlet_profile", role=FieldRole.Variable, on=fluid, value_type=ValueType.real(Dimension(length=1, time=-1)))

    stokes.relation('force_definition', q.equation(force_potential - zero_pressure, 0), on=fluid)
    stokes.relation('inlet_profile_definition', q.equation(inlet_profile - 4 * inlet_speed * q.coordinate(1) * (channel_height - q.coordinate(1)) / channel_height ** 2, 0), on=fluid)
    stress = 2 * dynamic_viscosity * q.symmetric_part(
        q.grad(velocity)
    ) - q.isotropic_lift(pressure)
    stokes.relation('momentum', q.equation(-q.div(stress) - q.grad(force_potential), 0), on=fluid)
    stokes.relation('incompressibility', q.equation(q.div(velocity), 0), on=fluid)
    stokes.relation('inlet_velocity', q.equation(q.trace(velocity) + q.normal(q.isotropic_lift(inlet_profile)), 0), on=inlet)
    stokes.relation('outlet_traction', q.equation(q.normal(stress), 0), on=outlet)
    stokes.relation('wall_velocity', q.equation(q.trace(velocity), 0), on=walls)
    stokes.relation('cylinder_velocity', q.equation(q.trace(velocity), 0), on=cylinder)
    return source


if __name__ == "__main__":
    print(build_source().to_eqi(), end="")
