"""Manufactured translation through bundled components on an authentic Gmsh mesh."""

import numpy as np
import pytest

import eqiora
from test_exact_cylinder_stokes_result import geometry_and_mesh


SOURCE = """
import Eqiora.Fluid.fluid as fluid;
public component Translation(
  support body: volume(ambient_dimension=2),
  support inlet: boundary(parent=body),
  support outlet: boundary(parent=body),
  support walls: boundary(parent=body),
  support cylinder: boundary(parent=body)
) {
  variable velocity: vector<m/s,2> on body in smooth;
  variable pressure: Pa on body in h1;
  parameter speed: m/s = SPEED;
  instance flow: fluid.SteadyStokes2d(
    body=body, exterior=boundaries(inlet,outlet,walls,cylinder),
    velocity=velocity, pressure=pressure, dynamic_viscosity=1[Pa*s]
  );
  instance a: fluid.PrescribedVelocity2d(body=body,face=inlet,
    velocity=speed*tensor_value(frame=body,components=[1,0]));
  instance b: fluid.PrescribedVelocity2d(body=body,face=outlet,
    velocity=speed*tensor_value(frame=body,components=[1,0]));
  instance c: fluid.PrescribedVelocity2d(body=body,face=walls,
    velocity=speed*tensor_value(frame=body,components=[1,0]));
  instance d: fluid.PrescribedVelocity2d(body=body,face=cylinder,
    velocity=speed*tensor_value(frame=body,components=[1,0]));
  connect flow.mechanical[boundary=inlet], a.mechanical;
  connect flow.mechanical[boundary=outlet], b.mechanical;
  connect flow.mechanical[boundary=walls], c.mechanical;
  connect flow.mechanical[boundary=cylinder], d.mechanical;
}
"""


@pytest.mark.parametrize("speed", [0.7, 1.3])
def test_imported_translation_on_curved_and_grouped_boundaries(speed):
    # Uniform u=(U,0) has zero strain and divergence, with p=0 under the
    # all-Dirichlet zero-mean gauge. It is represented exactly by MINI/P1 on
    # any triangular mesh. The circular boundary moves with the flow here;
    # this is not a stationary no-slip obstacle or a reaction-force oracle.
    # Reuse the physical reconstruction tolerance of
    # verify/fluid/port-closed-si-mini-stokes-2d, not observed Eqiora output.
    geometry, mesh = geometry_and_mesh()
    model = eqiora.compile(
        source=SOURCE.replace("SPEED", str(speed)), entry="Translation",
        geometry=geometry,
        bindings={"body": geometry.selection("fluid"), **{
            name: geometry.selection(name)
            for name in ("inlet", "outlet", "walls", "cylinder")
        }},
    )
    assert len(model.field_ids) == 2
    assert model.parameter_ids == [model.parameter("definition.speed").id]
    plan = eqiora.resolve(
        model, mesh=mesh, spatial=eqiora.fem.MiniP1(), scaling=None,
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.SparseLu,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Fast,
            provider=eqiora.solve.SolverProvider.faer(),
            relative_tolerance=1e-10, absolute_tolerance=1e-13,
            maximum_iterations=10_000,
        ),
    )
    result = eqiora.run(plan)
    for candidate in (result, eqiora.Result.from_bytes(plan, result.to_bytes())):
        velocity = candidate.output(plan.capability.velocity)
        vertices = np.asarray(velocity.values("vertex")).reshape(-1, 2)
        np.testing.assert_allclose(vertices[:, 0], speed, atol=2e-10, rtol=0)
        np.testing.assert_allclose(vertices[:, 1], 0, atol=2e-10, rtol=0)
        np.testing.assert_allclose(velocity.values("cell-bubble"), 0, atol=2e-10, rtol=0)
        pressure = candidate.output(plan.capability.pressure)
        np.testing.assert_allclose(pressure.values("vertex"), 0, atol=2e-10, rtol=0)
