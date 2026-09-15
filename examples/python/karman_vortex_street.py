"""Compute and optionally plot a Reynolds-100 Kármán vortex street."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import numpy.typing as npt

import eqiora

DENSITY_KG_PER_M3 = 1.0
DYNAMIC_VISCOSITY_PA_S = 1.0e-3
CYLINDER_DIAMETER_M = 0.1
INLET_MAXIMUM_M_PER_S = 1.5
INLET_MEAN_M_PER_S = 1.0
TIME_STEP_S = 0.01
DEFAULT_SPIN_UP_STEPS = 700
DEFAULT_SAMPLE_STEPS = 200
DEFAULT_SAMPLE_STRIDE = 2
SPIN_UP_CHUNK_STEPS = 100


@dataclass(frozen=True)
class WakeSimulation:
    """One developed-window result and its cylinder observables."""

    geometry: eqiora.geometry.Geometry
    mesh: eqiora.meshing.Mesh
    plan: eqiora.Plan
    result: eqiora.Result
    times_s: npt.NDArray[np.float64]
    drag_coefficients: npt.NDArray[np.float64]
    lift_coefficients: npt.NDArray[np.float64]
    pressure_differences_pa: npt.NDArray[np.float64]
    strouhal_number: float | None

    @property
    def final_state(self) -> eqiora.State:
        return self.result.trajectory.states[-1]

    @property
    def final_vorticity(self) -> eqiora.trajectory.DerivedFieldSnapshot:
        return self.final_state.curl(self.plan.capability.velocity)


def _prepare() -> tuple[
    eqiora.geometry.Geometry,
    eqiora.meshing.Mesh,
    eqiora.Plan,
    eqiora.State,
]:
    graph = eqiora.geometry.GeometryGraph()
    channel = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
    cylinder = graph.circle(center=(0.2, 0.2), radius=CYLINDER_DIAMETER_M / 2.0)
    fluid = graph.subtract(channel, cylinder)
    geometry = graph.build(
        fluid,
        named_topology={
            "fluid": fluid.region,
            "inlet": channel.boundaries[0],
            "outlet": channel.boundaries[1],
            "walls": channel.boundaries[2:4],
            "cylinder": cylinder.boundaries[0],
        },
    )
    mesh = eqiora.meshing.generate(
        eqiora.meshing.resolve(
            geometry,
            eqiora.meshing.GmshMesher(
                maximum_boundary_error=1.0e-4,
                maximum_target_size=0.02,
                minimum_mean_ratio=1.0e-5,
                maximum_boundary_facets=50,
            ),
        )
    )
    parameters = {
        "dynamic_viscosity": DYNAMIC_VISCOSITY_PA_S,
        "zero_pressure": 0.0,
        "inlet_speed": INLET_MAXIMUM_M_PER_S,
        "channel_height": 0.41,
    }
    support_bindings = {
        "fluid": geometry.selection("fluid"),
        **{
            name: (geometry.selection(name), geometry.selection("fluid"))
            for name in ("inlet", "outlet", "walls", "cylinder")
        },
    }
    linear = eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-12,
        maximum_iterations=20_000,
    )
    source_root = files(eqiora).joinpath("examples")
    steady_model = eqiora.compile(
        path=source_root.joinpath("steady-flow-past-cylinder.eqi"),
        geometry=geometry,
        entry="SteadyFlowPastCylinder",
        bindings={**support_bindings, **parameters},
    )
    steady_plan = eqiora.resolve(
        steady_model,
        mesh=mesh,
        spatial=eqiora.fem.MiniP1(),
        solve=linear,
        scaling=None,
    )
    steady_result = eqiora.run(steady_plan)

    transient_model = eqiora.compile(
        path=source_root.joinpath("transient-flow-past-cylinder.eqi"),
        geometry=geometry,
        entry="TransientFlowPastCylinder",
        bindings={**support_bindings, "density": DENSITY_KG_PER_M3, **parameters},
    )
    plan = eqiora.resolve(
        transient_model,
        mesh=mesh,
        spatial=eqiora.fem.MiniP1(),
        temporal=eqiora.time.BackwardEuler(TIME_STEP_S),
        solve=eqiora.solve.Newton(linear=linear),
        scaling=eqiora.fluid.IncompressibleScaling(
            length_m=CYLINDER_DIAMETER_M,
            velocity_m_per_s=INLET_MEAN_M_PER_S,
            pressure_pa=DENSITY_KG_PER_M3 * INLET_MEAN_M_PER_S**2,
        ),
    )
    steady_velocity = steady_result.output(steady_plan.capability.velocity)
    steady_pressure = steady_result.output(steady_plan.capability.pressure)
    initial = eqiora.State.initial(
        plan,
        time_s=0.0,
        fields=(
            eqiora.InitialField(
                plan.capability.velocity,
                vertex_values=np.asarray(steady_velocity.values("vertex")).reshape(
                    mesh.vertex_count, 2
                ),
                cell_values=np.asarray(steady_velocity.values("cell-bubble")).reshape(
                    mesh.cell_count, 2
                ),
            ),
            eqiora.InitialField(
                plan.capability.pressure,
                vertex_values=np.asarray(steady_pressure.values("vertex")),
            ),
        ),
    )
    return geometry, mesh, plan, initial


def _spin_up(plan: eqiora.Plan, state: eqiora.State, steps: int) -> eqiora.State:
    remaining = steps
    while remaining:
        chunk = min(remaining, SPIN_UP_CHUNK_STEPS)
        run = eqiora.run(plan, state=state, steps=chunk, output_steps=(chunk,))
        state = run.trajectory.state(chunk)
        remaining -= chunk
    return state


def _strouhal(times_s: npt.NDArray[np.float64], lift: npt.NDArray[np.float64]) -> float | None:
    centered = lift - float(np.mean(lift))
    rising = np.flatnonzero((centered[:-1] <= 0.0) & (centered[1:] > 0.0))
    if rising.size < 3:
        return None
    crossings = []
    for index in rising:
        fraction = -centered[index] / (centered[index + 1] - centered[index])
        crossings.append(times_s[index] + fraction * (times_s[index + 1] - times_s[index]))
    period_s = float(np.median(np.diff(crossings)))
    return CYLINDER_DIAMETER_M / (INLET_MEAN_M_PER_S * period_s)


def solve(
    *,
    spin_up_steps: int = DEFAULT_SPIN_UP_STEPS,
    sample_steps: int = DEFAULT_SAMPLE_STEPS,
    sample_stride: int = DEFAULT_SAMPLE_STRIDE,
    profile: bool = True,
) -> WakeSimulation:
    """Advance through spin-up, then retain one sampled wake window."""
    if spin_up_steps < 0 or sample_steps < 1 or sample_stride < 1:
        raise ValueError("spin-up must be non-negative and sampling controls must be positive")
    if sample_steps % sample_stride:
        raise ValueError("sample_steps must be divisible by sample_stride")

    geometry, mesh, plan, state = _prepare()
    state = _spin_up(plan, state, spin_up_steps)
    output_steps = tuple(range(sample_stride, sample_steps + 1, sample_stride))
    result = eqiora.run(
        plan,
        state=state,
        steps=sample_steps,
        output_steps=output_steps,
        profile=profile,
    )
    states = result.trajectory.states
    cylinder = geometry.selection("cylinder")
    coefficient_scale = 2.0 / (
        DENSITY_KG_PER_M3 * INLET_MEAN_M_PER_S**2 * CYLINDER_DIAMETER_M
    )
    times_s = np.asarray([accepted.time_s for accepted in states])
    forces = np.asarray(
        [accepted.boundary_force(cylinder).on_selection for accepted in states]
    )
    pressure_differences_pa = np.asarray(
        [
            accepted.sample(plan.capability.pressure, at=(0.15, 0.2)).value
            - accepted.sample(plan.capability.pressure, at=(0.25, 0.2)).value
            for accepted in states
        ]
    )
    drag = coefficient_scale * forces[:, 0]
    lift = coefficient_scale * forces[:, 1]
    return WakeSimulation(
        geometry=geometry,
        mesh=mesh,
        plan=plan,
        result=result,
        times_s=times_s,
        drag_coefficients=drag,
        lift_coefficients=lift,
        pressure_differences_pa=pressure_differences_pa,
        strouhal_number=_strouhal(times_s, lift),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spin-up-steps", type=int, default=DEFAULT_SPIN_UP_STEPS)
    parser.add_argument("--sample-steps", type=int, default=DEFAULT_SAMPLE_STEPS)
    parser.add_argument("--sample-stride", type=int, default=DEFAULT_SAMPLE_STRIDE)
    parser.add_argument(
        "--vorticity-png",
        type=Path,
        help="save the final vorticity field (requires eqiora[matplotlib])",
    )
    arguments = parser.parse_args()
    wake = solve(
        spin_up_steps=arguments.spin_up_steps,
        sample_steps=arguments.sample_steps,
        sample_stride=arguments.sample_stride,
    )
    state = wake.final_state
    omega = wake.final_vorticity.values("cell")
    print("Reynolds number", DENSITY_KG_PER_M3 * INLET_MEAN_M_PER_S * CYLINDER_DIAMETER_M / DYNAMIC_VISCOSITY_PA_S)
    print("method MINI/P1, backward Euler", TIME_STEP_S, "s")
    print("plan", wake.plan.identity)
    print("trajectory", wake.result.trajectory.digest)
    print("state", state.step, state.time_s, state.digest)
    print("vorticity", float(omega.min()), float(omega.max()), "s^-1")
    print("drag coefficient range", float(wake.drag_coefficients.min()), float(wake.drag_coefficients.max()))
    print("lift coefficient range", float(wake.lift_coefficients.min()), float(wake.lift_coefficients.max()))
    print("pressure-difference range", float(wake.pressure_differences_pa.min()), float(wake.pressure_differences_pa.max()), "Pa")
    print("sampled Strouhal number", wake.strouhal_number)
    if wake.result.profile is not None:
        print(wake.result.profile.summary())
    if arguments.vorticity_png is not None:
        import eqiora.matplotlib as eqplot

        figure = eqplot.plot_scalar_field(
            wake.result.trajectory,
            step=state.step,
            field=wake.final_vorticity,
        )
        figure.savefig(arguments.vorticity_png, dpi=180)
        print("vorticity still", arguments.vorticity_png)


if __name__ == "__main__":
    main()
