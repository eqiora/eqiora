"""Installed authoring preserves retained energies and ordered variation directions."""
import pytest

import eqiora as q


def test_installed_first_and_second_variation_authoring():
    graph = q.geometry.GeometryGraph()
    square = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    geometry = graph.build(square, named_topology={"body": square.region, **{f"face_{i}": face for i, face in enumerate(square.boundaries)}})
    for profile in ("first", "second", "ordinary_duplicate"):
        order = 1 if profile == "first" else 2
        module = q.Module("energy")
        component = module.component("Energy")
        body = component.volume("body", dimensions=2)
        u = component.field("u", on=body, role=q.FieldRole.Variable,
                            value_type=q.ValueType.real(q.Dimension(length=1)))
        a = component.parameter("a", value_type=q.ValueType.real(q.Dimension(mass=1, length=-2, time=-2)))
        balance = component.relation("balance", q.lang.equation(a*u, 0), on=body)
        energy = component.observable("energy", a*u*u/2, integrate_over=body,
                                      value_type=q.ValueType.real(q.Dimension(mass=1, length=2, time=-2)))
        combined = component.observable("combined", energy+energy-energy,
                                        value_type=q.ValueType.real(q.Dimension(mass=1, length=2, time=-2)))
        eta = component.test("eta", for_=u, dimension=q.Dimension(length=1))
        value = q.lang.variation(combined, wrt=u, direction=eta, holding=(a,))
        if order == 2:
            zeta = component.test("zeta", for_=u, dimension=q.Dimension(length=1))
            value = (q.lang.integrate(body, a*eta*zeta) if profile == "ordinary_duplicate"
                     else q.lang.variation(value, wrt=u, direction=zeta, holding=(a,)))
        component.weak_form("variation", [balance], equations=[(value, 0)])
        bindings = {"body": geometry.selection("body"), "a": 2.0}
        for source in (module, module.to_eqi()):
            if profile == "ordinary_duplicate":
                with pytest.raises(q.ValidationError, match="equations and test/trial inventories differ"):
                    q.compile(source=source, entry="Energy", geometry=geometry, bindings=bindings)
                continue
            model = q.compile(source=source, entry="Energy", geometry=geometry, bindings=bindings)
            form = model.authored_formulations[0]
            assert len(form.trial_field_ids) == 1
            assert [item[0] for item in form.test_restrictions] == ["eta", "zeta"][:order]
            assert all(not item[2] for item in form.test_restrictions)
        with pytest.raises(AttributeError):
            energy._name = "changed"

    coupled = q.Module("coupled")
    component = coupled.model("Coupled")
    x = component.field("x", role=q.FieldRole.Variable, value_type=q.ValueType.real())
    derived = component.observable("out", x+x, value_type=q.ValueType.real())
    component.relation("balance", q.lang.equation(x, derived))
    model = q.compile(source=coupled, entry="Coupled")
    output = model.observable("out")
    model = q.Model.from_bytes(model.to_bytes())
    solve = q.solve.Linear(relative_tolerance=1e-12, absolute_tolerance=1e-14,
                           maximum_iterations=8, algorithm=q.solve.LinearSolver.SparseLu,
                           preconditioner=q.solve.Preconditioner.Identity,
                           reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())
    plan = q.Plan.from_bytes(q.resolve(model, solve=solve).to_bytes())
    result = q.run(plan, state=q.State.initial(plan))
    result = q.Result.from_bytes(plan, result.to_bytes())
    # The authored equation x=2x fixes x=0, hence the derived output is zero.
    assert result.observe(output).value == pytest.approx(0.0, rel=0, abs=1e-12)
