"""Timeless spatial Newton uses ordinary State, Run and Result ownership."""
import hashlib
import json

import eqiora as q
import numpy as np
import pytest


def reseal(encoded, old, new):
    wire = json.loads(encoded)
    assert old in encoded
    changed = encoded.replace(old, new, 1)
    content = changed.split(b'"content":', 1)[1][:-1]
    identity = hashlib.sha256(wire["schema"].encode() + b"\0" + content).hexdigest()
    return changed.replace(wire["identity"].encode(), identity.encode(), 1)


@pytest.mark.parametrize("exact_seed", [False, True])
def test_stationary_newton_replays_seed_residual_and_actual_solve_history(exact_seed):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1],
    })
    body = geometry.selection("body")
    model = q.compile(source="""public model Main(
        support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {
        variable u:1 on body in smooth;
        relation balance on body {-div(grad(u))=0;}
        relation left_value on left {trace(u)=0;}
        relation right_value on right {trace(u)=2;}
    }""", entry="Main", geometry=geometry, bindings={
        "body": body, "left": (geometry.selection("left"), body),
        "right": (geometry.selection("right"), body),
    })
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(4,))))
    linear = q.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=64,
        algorithm=q.solve.LinearSolver.SparseLu, preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=q.solve.Newton(
        linear=linear, relative_tolerance=0., absolute_tolerance=1e-12,
        maximum_iterations=16, maximum_line_search_steps=8))
    plan = q.Plan.from_bytes(plan.to_bytes())
    field = model.field(model.field_ids[0])
    # Independently: u''=0 with endpoint values 0 and 2 gives u=2x.
    exact = 2*np.asarray(mesh.coordinates)[:, 0]
    values = exact if exact_seed else np.array([0., -1., 4., 0., 2.])
    state = q.State.initial(plan, fields=(q.InitialField(field, vertex_values=values),))
    before = state.to_bytes()
    assert state.time_s is None
    state = q.State.from_bytes(plan, before)
    with pytest.raises(q.ExecutionError, match="initial State"):
        q.run(plan)
    result = q.run(plan, state=state)
    assert state.to_bytes() == before
    assert result.solve.completed_iterations == (0 if exact_seed else 1)
    assert len(result.solve.linear_solves) == result.solve.completed_iterations
    assert result.solve.true_residual_norm <= result.solve.residual_target
    assert result.solve.residual_target == 1e-12
    encoded = result.to_bytes()
    for accepted in (result, q.Result.from_bytes(plan, encoded)):
        assert accepted.initial_state.to_bytes() == before
        assert accepted.initial_state.time_s is None
        np.testing.assert_allclose(accepted.output(field).values("vertex"), exact, rtol=0., atol=1e-11)
        assert accepted.to_bytes() == encoded
    assert reseal(encoded, b'"schema"', b'"schema"') == encoded
    iterations = result.solve.completed_iterations
    forged = reseal(encoded, f'"iterations":{iterations}'.encode(), f'"iterations":{iterations+1}'.encode())
    with pytest.raises(q.ValidationError, match="iteration"):
        q.Result.from_bytes(plan, forged)
    wire = json.loads(encoded)
    seed = wire["content"]["payload"]["acceptance"]["initial_state_base64"]
    forged = reseal(encoded, f'"initial_state_base64":"{seed}"'.encode(), b'"initial_state_base64":null')
    with pytest.raises(q.ValidationError, match="initial State"):
        q.Result.from_bytes(plan, forged)
    if not exact_seed:
        marker = b'"initial_residual_norm":'
        original = encoded.split(marker, 1)[1].split(b",", 1)[0]
        forged = reseal(encoded, marker+original, marker+b"0.0")
        with pytest.raises(q.ValidationError, match="predictor"):
            q.Result.from_bytes(plan, forged)


@pytest.mark.parametrize("law", ["polynomial_flux", "property_reaction"])
def test_stationary_material_runs_and_replays(law):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1],
    })
    body = geometry.selection("body")
    supports = """support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)"""
    if law == "property_reaction":
        source = f"""
        property contract Response(input value:1):1 {{derivatives first_partials;}}
        property release Material:Response {{
            analytic {{value=1+value^2;source_unit:1=1;}}
            validity value>=-1 and value<=3;outside reject;branch single;
            citation org.example.manufactured;license spdx.CC0_1_0;
        }}
        component Equation({supports},property response:Response) {{
            variable u:1 on body in smooth;
            relation balance on body {{
                -div(response(value=u)*grad(u))+(response(value=u)-5)/1[m^2]=0;
            }}
            relation left_value on left {{trace(u)=2;}}
            relation right_value on right {{trace(u)=2;}}
        }}
        public model Main({supports}) {{
            instance physics:Equation(body=body,left=left,right=right,response=Material);
        }}
        """
    else:
        source = f"""public model Main({supports}) {{
            variable u:1 on body in smooth;
            relation balance on body {{
                -div((1+u^2)*grad(u))+2*(1+coordinate(0)/1[m])/1[m^2]=0;
            }}
            relation left_value on left {{trace(u)=1;}}
            relation right_flux on right {{normal((1+u^2)*grad(u))=5[1/m];}}
        }}"""
    model = q.compile(source=source, entry="Main", geometry=geometry, bindings={
        "body": body, "left": (geometry.selection("left"), body),
        "right": (geometry.selection("right"), body),
    })
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(4,))))
    linear = q.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=64,
        algorithm=q.solve.LinearSolver.SparseLu, preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=q.solve.Newton(
        linear=linear, relative_tolerance=0., absolute_tolerance=1e-11,
        maximum_iterations=16, maximum_line_search_steps=16))
    plan = q.Plan.from_bytes(plan.to_bytes())
    field = model.field(model.field_ids[0])
    coordinates = np.asarray(mesh.coordinates)[:, 0]
    # Independent solutions: constant u=2 gives response=5 and zero gradient;
    # u=1+x gives q'=2(1+x), with the prescribed outlet flux q(1)=5.
    exact = np.full_like(coordinates, 2.) if law == "property_reaction" else 1+coordinates
    # Exact Q1 integration gives initial interior residual (-235/12,-1,-235/12)
    # and Newton direction (241/49,1977/392,241/49). The full trial violates
    # the upper bound 3; its half step is admissible and decreases the residual.
    values = np.array([2., 0., 0., 0., 2.]) if law == "property_reaction" else np.ones(5)
    state = q.State.initial(plan, fields=(q.InitialField(field, vertex_values=values),))
    before = state.to_bytes()
    result = q.run(plan, state=state)
    assert state.to_bytes() == before
    assert result.solve.completed_iterations > 1
    assert result.solve.true_residual_norm <= result.solve.residual_target
    encoded = result.to_bytes()
    for accepted in (result, q.Result.from_bytes(plan, encoded)):
        np.testing.assert_allclose(accepted.output(field).values("vertex"), exact, rtol=0., atol=1e-10)
        assert accepted.initial_state.to_bytes() == before
        assert accepted.initial_state.time_s is None
    wire = json.loads(encoded)
    block = wire["content"]["payload"]["fields"][0]["blocks"][0]
    changed = list(block["values"])
    changed[2] += 0.25
    old = b'"values":'+json.dumps(block["values"], separators=(",", ":")).encode()
    new = b'"values":'+json.dumps(changed, separators=(",", ":")).encode()
    with pytest.raises(q.ValidationError, match="residual"):
        q.Result.from_bytes(plan, reseal(encoded, old, new))
    if law == "property_reaction":
        undamped = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=q.solve.Newton(
            linear=linear, relative_tolerance=0., absolute_tolerance=1e-11,
            maximum_iterations=16, maximum_line_search_steps=0))
        rejected_seed = q.State.initial(undamped, fields=(q.InitialField(field, vertex_values=values),))
        rejected_bytes = rejected_seed.to_bytes()
        with pytest.raises(q.ExecutionError, match="line search found no residual-decreasing admissible point"):
            q.run(undamped, state=rejected_seed)
        assert rejected_seed.to_bytes() == rejected_bytes
        invalid = values.copy()
        invalid[2] = 4.
        with pytest.raises(q.EqioraError, match="required expression domain condition is false"):
            bad = q.State.initial(plan, fields=(q.InitialField(field, vertex_values=invalid),))
            q.run(plan, state=bad)
        assert state.to_bytes() == before
