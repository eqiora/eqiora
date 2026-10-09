"""Harmonic Module requests share the source, package and ordinary execution owners."""
import shutil

import eqiora
import pytest


def wave_module(owner_kind):
    q, units = eqiora.lang, eqiora.units
    module = eqiora.Module("main")
    owner = getattr(module, owner_kind)("Wave")
    body = owner.volume("body", dimensions=1)
    left, right = (owner.boundary(name, parent=body) for name in ("left", "right"))
    force = owner.input("force", value_type=eqiora.ValueType.real(eqiora.Dimension(time=-2)))
    boundary = owner.input("boundary_value", value_type=eqiora.ValueType.real())
    u = owner.field("u", value_type=eqiora.ValueType.real(), role=eqiora.FieldRole.State, on=body)
    owner.initial((u, 0), (q.derivative(u), q.quantity(0, units.s**-1)))
    flux = q.quantity(1, units.m**2 / units.s**2) * q.grad(u)
    balance = owner.relation("balance", q.equation(
        q.derivative(q.derivative(u)) + q.quantity(1, units.s**-1)*q.derivative(u) - q.div(flux), force), on=body)
    fixed = owner.relation("fixed", q.equation(q.trace(u), boundary), on=left)
    natural = owner.relation("flux", q.equation(q.normal(flux), q.quantity(0, units.m/units.s**2)), on=right)
    arguments = dict(angular_frequency=q.quantity(1, units.s**-1),
        convention="negative_exponential", normalization="peak",
        excitations=[(force, q.math.complex(q.quantity(-1, units.s**-2), q.quantity(-3, units.s**-2))),
                     (boundary, 2+1j)], amplitudes=[("u_hat", u, eqiora.ValueType.complex())])
    return module, owner, (balance, fixed, natural), arguments


@pytest.mark.parametrize("owner_kind", ["model", "component"])
def test_harmonic_wave_module_source_package_and_moved_offline_execution(tmp_path, owner_kind):
    module, owner, relations, arguments = wave_module(owner_kind)
    owner.harmonic_form("response", relations, **arguments)
    graph = eqiora.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={"body": interval.region,
        "left": interval.boundaries[0], "right": interval.boundaries[1]})
    bindings = {"body": geometry.selection("body"),
        "left": (geometry.selection("left"), geometry.selection("body")),
        "right": (geometry.selection("right"), geometry.selection("body"))}
    options = dict(entry="Wave", geometry=geometry, bindings=bindings)
    source = module.to_eqi()
    models = [eqiora.compile(source=candidate, **options) for candidate in (module, source)]
    assert models[0].digest == models[1].digest
    project, store = tmp_path / "project", tmp_path / "store"
    (project / "src").mkdir(parents=True)
    store.mkdir()
    (project / "eqiora.toml").write_text('[package]\nname = "eqiora.local_project"\nversion = "0.1.0"\nentry = "main"\n')
    (project / "src/main.eqi").write_text(source)
    lock = eqiora.resolve_local_project(project, store)
    packaged = eqiora.compile_package(store, lock, **options)
    assert packaged.structural_fingerprint == models[0].structural_fingerprint
    vendor = project / "vendor"
    vendor.mkdir()
    assert eqiora.vendor_project(project, store, vendor) == lock
    shutil.rmtree(store)
    moved = tmp_path / "moved"
    project.rename(moved)
    assert eqiora.open_project(moved, moved / "vendor") == lock
    offline = eqiora.compile_package(moved / "vendor", lock, **options)
    assert offline.digest == packaged.digest
    models.extend((packaged, offline))
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry, eqiora.meshing.CartesianMesher(cells=(2,))))
    linear = eqiora.solve.Linear(algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=eqiora.solve.Preconditioner.Identity, reduction=eqiora.solve.Reduction.Reproducible,
        provider=eqiora.solve.SolverProvider.reference(), relative_tolerance=1e-13,
        absolute_tolerance=1e-15, maximum_iterations=64)
    for model in models:
        plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
        plan = eqiora.Plan.from_bytes(plan.to_bytes())
        assert plan.harmonic_original_model.digest == model.digest
        ((name, original, amplitude),) = plan.harmonic_amplitudes
        assert name == "u_hat" and original.model_digest == model.digest
        result = eqiora.run(plan)
        result = eqiora.Result.from_bytes(plan, result.to_bytes())
        assert list(result.output(amplitude).values("vertex")) == pytest.approx([2+1j]*3, abs=1e-10)
        assert list(result.reconstruct_harmonic_field_block(original, time_seconds=0)) == pytest.approx([2.]*3, abs=1e-10)


def test_harmonic_form_rejects_foreign_ownership_without_consuming_the_form_slot():
    _, owner, relations, arguments = wave_module("model")
    _, foreign, _, foreign_arguments = wave_module("model")
    for changes, message in [
        ({"excitations": foreign_arguments["excitations"]}, "input from this Component"),
        ({"amplitudes": foreign_arguments["amplitudes"]}, "body Field from this Component"),
        ({"angular_frequency": foreign.parameter("omega", value_type=eqiora.ValueType.real(eqiora.Dimension(time=-1)))}, "expressions must belong"),
        ({"normalization": "rms"}, "peak normalization"),
    ]:
        with pytest.raises(eqiora.lang.ModuleError, match=message):
            owner.harmonic_form("response", relations, **(arguments | changes))
    owner.harmonic_form("response", relations, **arguments)
    with pytest.raises(eqiora.lang.ModuleError, match="one named form"):
        owner.harmonic_form("second", relations, **arguments)
