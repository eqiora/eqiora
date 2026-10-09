"""Python-authored weak forms execute through the same emitted-source path."""

import shutil

import eqiora
import pytest


@pytest.mark.parametrize("owner", ["component", "model"])
def test_finite_weak_authoring_preserves_spectral_phase(tmp_path, owner):
    q = eqiora.lang
    module = eqiora.Module("main")
    spin = module.space("Spin", labels=("up", "down"))
    component = getattr(module, owner)("Wave")
    scalar = eqiora.ValueType.complex()
    h = component.parameter("h", value_type=eqiora.ValueType.linear_map(scalar, spin, spin))
    component.set_default(h, component.linear_map(spin, spin, ((2, -1j), (1j, 2))))
    u = component.field("u", value_type=eqiora.ValueType.coordinates(scalar, spin), role=eqiora.FieldRole.Variable)
    eigenvalue = component.field("lambda", value_type=eqiora.ValueType.real(), role=eqiora.FieldRole.Variable)
    states = component.relation("states", q.equation(q.apply(h, u), eigenvalue*u))
    eta = component.test("eta", for_=u)
    component.weak_form("weak", [states], equations=[(q.inner(eta, q.apply(h, u)), q.inner(eta, eigenvalue*u))])
    models = authoring_models(module, tmp_path)
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
    assert identities[2] == identities[3]
    assert identities[0] != identities[2]


@pytest.mark.parametrize("owner", ["component", "model"])
def test_spatial_weak_authoring_includes_complex_boundary_load(tmp_path, owner):
    q = eqiora.lang
    module = eqiora.Module("main")
    component = getattr(module, owner)("Wave")
    body = component.volume("body", dimensions=1)
    left = component.boundary("left", parent=body)
    right = component.boundary("right", parent=body)
    scalar = eqiora.ValueType.complex()
    a = component.parameter("a", value_type=eqiora.ValueType.complex(eqiora.Dimension(length=2)))
    reaction = component.parameter("q", value_type=scalar)
    f = component.parameter("f", value_type=scalar)
    slope = component.parameter("s", value_type=eqiora.ValueType.complex(eqiora.Dimension(length=-1)))
    flux = component.parameter("g", value_type=eqiora.ValueType.complex(eqiora.Dimension(length=1)))
    u = component.field("u", spatial_regularity=eqiora.SpatialRegularity.Smooth, value_type=scalar, role=eqiora.FieldRole.Variable, on=body)
    balance = component.relation("balance", q.equation(-q.div(a*q.grad(u))+reaction*u, f+slope*q.coordinate(0)), on=body)
    component.relation("fixed", q.equation(q.trace(u), 1+3j), on=left)
    component.relation("natural", q.equation(q.normal(a*q.grad(u)), flux), on=right)
    eta = component.test("eta", for_=u, zero_on=left)
    component.weak_form("weak", [balance], equations=[(
        q.integrate(body, q.inner(q.grad(eta), a*q.grad(u))+q.inner(eta, reaction*u)),
        q.integrate(body, q.inner(eta, f+slope*q.coordinate(0)))+q.integrate(right, q.inner(q.trace(eta), flux)),
    )])
    graph = eqiora.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 6.))
    geometry = graph.build(interval, named_topology={"body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1]})
    bindings = {"body": geometry.selection("body"), "left": (geometry.selection("left"), geometry.selection("body")), "right": (geometry.selection("right"), geometry.selection("body")), "a": 6+6j, "q": 1+1j, "f": -2+4j, "s": 3+1j, "g": 18+6j}
    models = authoring_models(module, tmp_path, geometry=geometry, bindings=bindings)
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
    assert identities[2] == identities[3]
    assert identities[0] != identities[2]


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


def authoring_models(module, tmp_path, **bindings):
    emitted = module.to_eqi()
    models = [eqiora.compile(source=source, entry="Wave", **bindings) for source in (module, emitted)]
    assert models[0].digest == models[1].digest
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "eqiora.toml").write_text('[package]\nname = "eqiora.local_project"\nversion = "0.1.0"\nentry = "main"\n')
    (project / "src/main.eqi").write_text(emitted)
    store = tmp_path / "store"
    store.mkdir()
    lock = eqiora.resolve_local_project(project, store)
    packaged = eqiora.compile_package(store, lock, entry="Wave", **bindings)
    assert packaged.structural_fingerprint == models[0].structural_fingerprint
    assert packaged.package_compilation_digest is not None
    assert models[0].package_compilation_digest is None
    assert packaged.digest != models[0].digest
    vendor = project / "vendor"
    vendor.mkdir()
    assert eqiora.vendor_project(project, store, vendor) == lock
    shutil.rmtree(store)
    moved = tmp_path / "moved"
    project.rename(moved)
    assert eqiora.open_project(moved, moved / "vendor") == lock
    offline = eqiora.compile_package(moved / "vendor", lock, entry="Wave", **bindings)
    assert offline.digest == packaged.digest
    assert offline.package_compilation_digest == packaged.package_compilation_digest
    return [*models, packaged, offline]
