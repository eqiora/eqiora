"""Python-authored weak forms execute through the same emitted-source path."""

import eqiora
import pytest


def test_finite_weak_authoring_preserves_spectral_phase():
    q = eqiora.lang
    module = eqiora.Module("main")
    spin = module.space("Spin", labels=("up", "down"))
    component = module.component("Wave")
    scalar = eqiora.ValueType.complex()
    h = component.parameter("h", value_type=eqiora.ValueType.linear_map(scalar, spin, spin))
    component.set_default(h, component.linear_map(spin, spin, ((2, -1j), (1j, 2))))
    u = component.field("u", value_type=eqiora.ValueType.coordinates(scalar, spin), role=eqiora.FieldRole.Variable)
    eigenvalue = component.field("lambda", value_type=eqiora.ValueType.real(), role=eqiora.FieldRole.Variable)
    states = component.relation("states", q.equation(q.apply(h, u), eigenvalue*u))
    eta = component.test("eta", for_=u)
    component.weak_form("weak", [states], equations=[(q.inner(eta, q.apply(h, u)), q.inner(eta, eigenvalue*u))])
    emitted = module.to_eqi()
    models = [eqiora.compile(source=source, entry="Wave") for source in (module, emitted)]
    assert models[0].digest == models[1].digest
    solve = eqiora.solve.HermitianEigen(count=2, provider=eqiora.solve.SolverProvider.faer(), residual_tolerance=1e-12, normalization_tolerance=1e-12)
    identities = []
    for model in models:
        plan = eqiora.resolve(model, solve=solve)
        identities.append(plan.identity)
        restored = eqiora.Plan.from_bytes(plan.to_bytes())
        assert restored.to_bytes() == plan.to_bytes()
        result = eqiora.run(restored)
        # H=2I+sigma_y: eigenvalues 1,3 and projectors (I-/+sigma_y)/2.
        # Off-diagonal phases distinguish H from its transpose with the same spectrum.
        for index, (value, off_diagonal) in enumerate(((1, .5j), (3, -.5j))):
            assert abs(result.eigenpair(index).eigenvalue-value) < 1e-12
            projector, _ = result.eigenprojector([index])
            expected = ((.5, off_diagonal), (off_diagonal.conjugate(), .5))
            for row in range(2):
                for col in range(2):
                    assert abs(projector[row][col]-expected[row][col]) < 1e-12
        assert eqiora.Result.from_bytes(restored, result.to_bytes()).to_bytes() == result.to_bytes()
    assert identities[0] == identities[1]


def test_spatial_weak_authoring_includes_complex_boundary_load():
    q = eqiora.lang
    module = eqiora.Module("main")
    component = module.component("Wave")
    body = component.volume("body", dimensions=1)
    left = component.boundary("left", parent=body)
    right = component.boundary("right", parent=body)
    scalar = eqiora.ValueType.complex()
    a = component.parameter("a", value_type=eqiora.ValueType.complex(eqiora.Dimension(length=2)))
    reaction = component.parameter("q", value_type=scalar)
    f = component.parameter("f", value_type=scalar)
    slope = component.parameter("s", value_type=eqiora.ValueType.complex(eqiora.Dimension(length=-1)))
    flux = component.parameter("g", value_type=eqiora.ValueType.complex(eqiora.Dimension(length=1)))
    u = component.field("u", value_type=scalar, role=eqiora.FieldRole.Variable, on=body)
    balance = component.relation("balance", q.equation(-q.div(a*q.grad(u))+reaction*u, f+slope*q.coordinate(0)), on=body)
    component.relation("fixed", q.equation(q.trace(u), 1+3j), on=left)
    component.relation("natural", q.equation(q.normal(a*q.grad(u)), flux), on=right)
    eta = component.test("eta", for_=u, zero_on=left)
    component.weak_form("weak", [balance], equations=[(
        q.integrate(body, q.inner(q.grad(eta), a*q.grad(u))+q.inner(eta, reaction*u)),
        q.integrate(body, q.inner(eta, f+slope*q.coordinate(0)))+q.integrate(right, q.inner(q.trace(eta), flux)),
    )])
    emitted = module.to_eqi()
    graph = eqiora.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 6.))
    geometry = graph.build(interval, named_topology={"body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1]})
    bindings = {"body": geometry.selection("body"), "left": (geometry.selection("left"), geometry.selection("body")), "right": (geometry.selection("right"), geometry.selection("body")), "a": 6+6j, "q": 1+1j, "f": -2+4j, "s": 3+1j, "g": 18+6j}
    models = [eqiora.compile(source=source, entry="Wave", geometry=geometry, bindings=bindings) for source in (module, emitted)]
    assert models[0].digest == models[1].digest
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry, eqiora.meshing.CartesianMesher(cells=(2,))))
    solve = eqiora.solve.Linear(algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized, preconditioner=eqiora.solve.Preconditioner.Identity, reduction=eqiora.solve.Reduction.Reproducible, provider=eqiora.solve.SolverProvider.reference(), relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=128)
    identities = []
    for model in models:
        plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=solve)
        identities.append(plan.identity)
        restored = eqiora.Plan.from_bytes(plan.to_bytes())
        assert restored.to_bytes() == plan.to_bytes()
        result = eqiora.run(restored)
        field = model.field(model.authored_formulations[0].trial_field_ids[0])
        values = result.output(field).values("vertex")
        # u=1+3i+(2-i)x: q*u=f+s*x and outward a*u'=18+6i at x=6.
        assert len(values) == 3
        for index, expected in enumerate((1+3j, 7+0j, 13-3j)):
            assert abs(values[index]-expected) < 1e-10
        replayed = eqiora.Result.from_bytes(restored, result.to_bytes())
        assert replayed.to_bytes() == result.to_bytes()
    assert identities[0] == identities[1]


def test_inner_and_integrate_preserve_component_ownership():
    q = eqiora.lang
    module = eqiora.Module("main")
    first = module.component("First")
    second = module.component("Second")
    x = first.parameter("x", value_type=eqiora.ValueType.complex())
    y = second.parameter("y", value_type=eqiora.ValueType.complex())
    body = first.volume("body", dimensions=1)
    boundary = first.boundary("side", parent=body)
    exterior = first.complete_exterior("exterior", parent=body)
    with pytest.raises(q.ModuleError, match="different Module or Component"):
        q.inner(x, y)
    with pytest.raises(q.ModuleError, match="same Component"):
        q.integrate(boundary, y)
    for unsupported in (exterior, exterior.member("member")):
        with pytest.raises(q.ModuleError, match="volume or boundary"):
            q.integrate(unsupported, x)
