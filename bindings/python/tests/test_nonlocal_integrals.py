"""Bounded nonlocal actions through the installed ordinary lifecycle and finite AD."""
from pathlib import Path

import numpy as np
import pytest
import eqiora as q

LENGTH = q.Dimension(length=1)
SOURCE = (Path(__file__).resolve().parents[3] /
          "verify/language/factor-integrals/models/nonlocal-interaction.eqi").read_text()


def compile_model(source=SOURCE, source_upper=2):
    return q.compile(source=source, entry="Interaction", bindings={
        "target": q.CoordinateInterval(0, 1, dimension=LENGTH),
        "source": q.CoordinateInterval(0, source_upper, dimension=LENGTH),
    })


def linear():
    return q.solve.Linear(relative_tolerance=1e-12, absolute_tolerance=1e-14,
        maximum_iterations=8, algorithm=q.solve.LinearSolver.SparseLu,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())


@pytest.mark.parametrize("source_upper,coupled,slope", [(1, False, 1/3), (2, False, 8/3), (2, True, 4)])
def test_nonlocal_action_and_truncation_survive_model_plan_result_replay(source_upper, coupled, slope):
    source = SOURCE if coupled else SOURCE.replace("inventory=2[m]", "amplitude=1")
    model = compile_model(source, source_upper)
    output = model.observable("action")
    model = q.Model.from_bytes(model.to_bytes())
    plan = q.Plan.from_bytes(q.resolve(model, solve=linear()).to_bytes())
    result = q.run(plan, state=q.State.initial(plan))
    result = q.Result.from_bytes(plan, result.to_bytes())
    # integral_0^b x*y^2 dy = x*b^3/3; the coupled case fixes A=3/2.
    for x in (0, 0.25, 1):
        observed = result.observe_at(output, [(x, LENGTH)], quadrature_points=2)
        assert observed.value == pytest.approx(slope*x, rel=0, abs=1e-12)
        assert observed.value_type == q.ValueType.real()
    with pytest.raises(q.ValidationError, match="support"):
        result.observe_at(output, [(2, LENGTH)], quadrature_points=2)


def test_nonlocal_residual_output_and_total_actions_keep_integrated_dependence():
    source = SOURCE.replace("inventory=2[m]", "inventory=amount; inequality(amount>=0[m])").replace(
        "variable amplitude:1;", """variable amplitude:1; parameter amount:m=2[m];
        coordinate target_x:m on target from target;
        observable weighted:m=integral(target_x/1[m]*action,measure(target));
        observable readout:m=weighted;""")
    model = compile_model(source)
    enforcement = q.solve.StrictInterior(margins=(q.solve.ConstraintTolerance.inequality(
        model.constraint("prescribed_inventory", 1), 1e-10, LENGTH),))
    plan = q.resolve(model, solve=q.solve.Newton(linear=linear(), relative_tolerance=0.0,
        absolute_tolerance=1e-12, maximum_iterations=8, maximum_line_search_steps=8),
        enforcement=enforcement)
    seed = q.State.initial(plan, fields=(q.InitialField(model.field("amplitude"), value=1.0),))
    program = q.diff.compile(plan, inputs=(model.parameter("amount"),),
                            output=model.observable("readout"), state=seed)
    def vector(x):
        return np.array([x], dtype=np.float64)
    def close(value, expected):
        np.testing.assert_allclose(value.numpy(), [expected], rtol=0, atol=1e-12)
    # R(A,p)=4A/3-p, O(A)=8A/9, O(p)=2p/3: independently integrated polynomials.
    for amount in (2, 4):
        point = program.evaluate(vector(amount))
        close(point.accepted_unknowns, 3*amount/4)
        close(point.primal().output, 2*amount/3)
        close(point.residual_jvp(vector(1), vector(0)), 4/3)
        da, dp = point.residual_vjp(vector(3))
        close(da, 4)
        close(dp, -3)
        close(point.output_partial_jvp(vector(1), vector(0)), 8/9)
        da, dp = point.output_partial_vjp(vector(9))
        close(da, 8)
        close(dp, 0)
        close(point.jvp(vector(3)).tangent, 2)
        close(point.vjp(vector(3)).input_cotangent, 2)


def test_nonlocal_atomic_sum_retains_typed_indices_and_dimensioned_masses():
    source = """operator kernel(input x:m,input y:m,input scale:1/m^3):1/m=x*y*scale;
    model Interaction(support target:interval(m), support source:interval(m)) {
      indexset Atoms=range(2);
      coordinate x:m on target from target;
      variable anchor:1; relation fixed {anchor=1;}
      observable action:1 on target=sum(
        kernel(x=x,y=(to_real(ordinal(i))+1)*1[m],scale=1[1/m^3])
        *(to_real(ordinal(i))+1)*(to_real(ordinal(i))+1)*1[m],over=(i in Atoms));
    }"""
    model = compile_model(source)
    output = model.observable("action")
    model = q.Model.from_bytes(model.to_bytes())
    plan = q.Plan.from_bytes(q.resolve(model, solve=linear()).to_bytes())
    result = q.run(plan, state=q.State.initial(plan))
    result = q.Result.from_bytes(plan, result.to_bytes())
    # Discrete masses 1,2 m at y=1,2 m give x*(1^3+2^3)=9x.
    # This is an atomic measure, not continuous quadrature on the unused source interval.
    for x in (0, 0.25, 1):
        observed = result.observe_at(output, [(x, LENGTH)])
        assert observed.value == pytest.approx(9*x, rel=0, abs=1e-12)
        assert len(observed.quadratures) == 0
