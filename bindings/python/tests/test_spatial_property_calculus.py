"""Prescribed material expressions use the ordinary spatial coefficient calculus."""
import numpy as np
import pytest
import eqiora as q


@pytest.mark.parametrize("method", [q.fem.Q1, q.fvm.CellCenteredTpfa])
@pytest.mark.parametrize("consumer", ["source", "boundary"])
@pytest.mark.parametrize("profile,expression", [
    ("first_partials", "1+value^2"),
    ("value_only", "math.sin(value)"),
])
def test_guarded_property_coefficients_preserve_declared_profiles(method, consumer, profile, expression):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1],
    })
    body = geometry.selection("body")
    operating = "(gain*coordinate(0)/1[m])"
    reference = expression.replace("value", operating)
    balance = (f"-div(grad(u))+response(value={operating})/1[m^2]=({reference})/1[m^2]"
               if consumer == "source" else "-div(grad(u))=0")
    right_value = "gain" if consumer == "source" else "response(value=gain)"
    source = f"""
    property contract Response(input value:1):1 {{ derivatives {profile}; }}
    property release Material:Response {{
        analytic {{ value={expression}; source_unit:1=1; }}
        validity value>=-1 and value<=3; outside reject; branch single;
        citation org.example.manufactured; license spdx.CC0_1_0;
    }}
    component Equation(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body),
        parameter gain:1,property response:Response) {{
        variable u:1 on body in smooth;
        relation balance on body {{{balance};}}
        relation left_value on left {{trace(u)=0;}}
        relation right_value on right {{trace(u)={right_value};}}
    }}
    public model Main(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {{
        parameter gain:1=2;
        instance physics:Equation(body=body,left=left,right=right,gain=gain,response=Material);
    }}
    """
    bindings = {
        "body": body, "left": (geometry.selection("left"), body),
        "right": (geometry.selection("right"), body),
    }
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(4,))))
    solve = q.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=64,
        algorithm=q.solve.LinearSolver.SparseLu, preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())

    def plan_at(gain):
        model = q.compile(source=source.replace("parameter gain:1=2;", f"parameter gain:1={gain};"),
                          entry="Main", geometry=geometry, bindings=bindings)
        return model, q.resolve(model, mesh=mesh, spatial=method(), solve=solve)

    model, plan = plan_at(2.)
    q.run(plan)
    positions = np.linspace(0., 1., 5) if method is q.fem.Q1 else (np.arange(4)+.5)/4
    weights = np.arange(1, len(positions)+1)/4
    inputs = (plan.model.parameter(model.parameter("gain").id),)
    if profile == "value_only":
        # Selecting derivative inputs must honor the release contract, while
        # ordinary execution still evaluates its values and domain guards.
        with pytest.raises(q.ValidationError, match="value-only contract"):
            q.diff.compile(plan, inputs=inputs, outputs=(plan.fields[0],))
        for gain in (1.5, 2.):
            _, current = plan_at(gain)
            result = q.run(current)
            values = result.output(current.fields[0]).values("vertex" if method is q.fem.Q1 else "cell")
            value = gain if consumer == "source" else np.sin(gain)
            np.testing.assert_allclose(values, value*positions, rtol=0, atol=2e-10)
        with pytest.raises(q.EqioraError, match="required expression domain condition is false"):
            _, outside = plan_at(4.)
            q.run(outside)
        return

    program = q.diff.compile(plan, inputs=inputs, outputs=(plan.fields[0],))
    # Source and boundary consumers both have an independently known harmonic
    # solution. The boundary law is the sole nonlinear gain dependence, so
    # freezing material derivatives cannot cancel against a frozen source.
    for gain in (1.5, 2.):
        value = gain if consumer == "source" else 1+gain**2
        slope = 1. if consumer == "source" else 2*gain
        point = program.evaluate((gain,))
        np.testing.assert_allclose(point.primal().output[0], value*positions, rtol=0, atol=2e-10)
        np.testing.assert_allclose(point.jvp((1.,)).tangent[0], slope*positions, rtol=0, atol=2e-10)
        np.testing.assert_allclose(point.vjp((weights,)).input_cotangent[0],
                                   slope*np.dot(weights, positions), rtol=0, atol=2e-10)
    with pytest.raises(q.ExecutionError, match="required expression domain condition is false"):
        program.evaluate((4.,)).primal()
