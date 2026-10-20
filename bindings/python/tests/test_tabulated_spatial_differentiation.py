"""Exact table releases share the ordinary spatial material and derivative owners."""

import numpy as np
import pytest
import eqiora as q

from test_nonlinear_spatial_differentiation import exact, newton_policy


def table_plan(tmp_path, right=2):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0.0, 1.0))
    geometry = graph.build(
        interval,
        named_topology={
            "body": interval.region,
            "left": interval.boundaries[0],
            "right": interval.boundaries[1],
        },
    )
    body = geometry.selection("body")
    supports = """support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)"""
    source = f"""
    property contract Law(input value:1):1 {{derivatives first_open_intervals;}}
    property release Release:Law {{
        table {{data samples;axis value:1;value response:1;
            interpolation piecewise_affine;preprocessing identity;missing reject;
            knot_derivative reject;endpoint_derivative reject;}}
        validity value in [-0.5,5];outside reject;branch single;
        citation manufactured;license license;
    }}
    component Equation({supports},parameter p:1,parameter r:1,property response:Law) {{
        variable u:1 on body in smooth;
        relation balance on body {{-div(response(value=p*u)*grad(u))=0;}}
        relation lower on left {{trace(u)=1;}}
        relation upper on right {{trace(u)=r;}}
    }}
    public model Main({supports}) {{parameter p:1=1;parameter r:1={right};
        instance physics:Equation(body=body,left=left,right=right,p=p,r=r,response=Release);
    }}"""
    project = tmp_path / "project"
    for directory in ("src", "data", "docs"):
        (project / directory).mkdir(parents=True)
    (project / "src/main.eqi").write_text(source)
    (project / "eqiora.toml").write_text(
        '[package]\nname="org.example.TableMaterialProbe"\nversion="1.0.0"\nentry="main"\n'
    )
    # k(s)=1+s for -0.5<s<3.5, then k(s)=2s-2.5 for 3.5<s<5.
    samples = q.data.ResolvedArray.from_f64((3, 2), [-0.5, 0.5, 3.5, 4.5, 5.0, 7.5])
    (project / "data/samples.json").write_bytes(samples.to_bytes())
    (project / "docs/manufactured.md").write_text(
        "Synthetic piecewise law k(s)=1+s below3.5 and 2s-2.5 above3.5; endpoint values .5,4.5,7.5."
    )
    (project / "docs/license.md").write_text("CC0-1.0 synthetic definition.")
    store = tmp_path / "store"
    store.mkdir()
    lock = q.resolve_local_project(project, store)
    model = q.compile_package(
        store,
        lock,
        entry="Main",
        geometry=geometry,
        bindings={
            "body": body,
            "left": (geometry.selection("left"), body),
            "right": (geometry.selection("right"), body),
        },
    )
    mesh = q.meshing.generate(
        q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(2,)))
    )
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=newton_policy())
    plan = q.Plan.from_bytes(plan.to_bytes())
    seed = q.State.initial(
        plan,
        fields=(
            q.InitialField(plan.fields[0], vertex_values=[1.0, (1 + right) / 2, right]),
        ),
    )
    program = q.diff.compile(
        plan,
        inputs=tuple(
            plan.model.parameter(model.parameter(name).id) for name in ("p", "r")
        ),
        outputs=(plan.fields[0],),
        state=seed,
    )
    return plan, seed, program


def test_tabulated_material_primal_products_and_replay(tmp_path):
    plan, seed, program = table_plan(tmp_path)
    before = seed.to_bytes()
    result = q.run(plan, state=seed)
    assert q.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
    for p, r in [(0.0, 2.0), (1.0, 2.0), (1.0, 3.0)]:
        u, dp, dr = exact(p, r)
        point = program.evaluate((p, r))
        np.testing.assert_allclose(
            point.primal().output[0], [1.0, u, r], rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            point.jvp((1.0, 0.0)).tangent[0], [0.0, dp, 0.0], rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            point.jvp((0.0, 1.0)).tangent[0], [0.0, dr, 1.0], rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            point.vjp((np.array([2.0, 3.0, 5.0]),)).input_cotangent,
            [3 * dp, 3 * dr + 5],
            rtol=0,
            atol=1e-10,
        )
    collection = program.map(
        (np.array([0.0, 1.0]), np.array([2.0, 3.0])), point_shape=(2,)
    ).execute_recorded()
    assert collection.complete
    encoded = collection.to_bytes()
    assert q.EvaluationCollection.from_bytes(encoded).to_bytes() == encoded
    assert seed.to_bytes() == before


def test_tabulated_material_preserves_knot_endpoint_and_outside_rejection(tmp_path):
    _, seed, program = table_plan(tmp_path, right=1)
    before = seed.to_bytes()
    np.testing.assert_allclose(
        program.primal().output[0], np.ones(3), rtol=0, atol=1e-10
    )
    for p in [-1.0, -0.5, 3.5, 5.0, 6.0]:
        with pytest.raises(q.EqioraError):
            program.evaluate((p, 1.0))
    assert seed.to_bytes() == before
    np.testing.assert_allclose(
        program.primal().output[0], np.ones(3), rtol=0, atol=1e-10
    )


def test_tabulated_material_uses_each_quadrature_segment_slope(tmp_path):
    _, _, program = table_plan(tmp_path)
    # This is a discrete two-point Gauss Q1 oracle, not an exact continuum
    # integral across a table knot. For p=r=2, the last quadrature point uses
    # k(s)=2s-2.5 and the other three use k(s)=1+s.
    a = (1 - 1 / np.sqrt(3)) / 2
    b = 1 - a
    A = 2 + a
    B = 0.25 + 2 / np.sqrt(3)
    C = -(4.5 + 4 * b)
    u = (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A)
    p = r = 2.0
    samples = p * np.array([b + a * u, a + b * u, b * u + a * r, a * u + b * r])
    assert np.all(samples[:3] < 3.5) and 3.5 < samples[3] < 5.0
    # Half of the free residual is KL*(u-1)+KR*(u-r), where each K is
    # the mean of its two independently interpolated Gauss values.
    c = b / 2 + a
    d = a / 2 + b
    kl = 1 + p * (1 + u) / 2
    kr = -0.75 + p * (c * u + d * r)
    fu = p * (u - 1) / 2 + kl + p * c * (u - r) + kr
    fp = (1 + u) * (u - 1) / 2 + (c * u + d * r) * (u - r)
    fr = p * d * (u - r) - kr
    dp = -fp / fu
    dr = -fr / fu
    point = program.evaluate((p, r))
    np.testing.assert_allclose(
        point.primal().output[0], [1.0, u, r], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        point.jvp((1.0, 0.0)).tangent[0], [0.0, dp, 0.0], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        point.jvp((0.0, 1.0)).tangent[0], [0.0, dr, 1.0], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        point.vjp((np.array([2.0, 3.0, 5.0]),)).input_cotangent,
        [3 * dp, 3 * dr + 5],
        rtol=0,
        atol=1e-10,
    )
