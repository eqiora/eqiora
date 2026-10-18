"""Representable quotient actions through the installed finite Program API."""
import math

import numpy as np
import pytest

import eqiora


@pytest.mark.parametrize("exponent", [-1020, -700, 0, 700, 1020])
def test_quotient_parameter_actions_do_not_square_the_denominator(exponent):
    scale = math.ldexp(1.0, exponent)
    model = eqiora.compile(source=f"""model M() {{
        parameter a: 1 = {scale!r};
        parameter b: 1 = {scale!r};
        variable z: 1;
        relation pin {{ z = 0; }}
        observable ratio: 1 = a / b;
    }}""")
    plan = eqiora.resolve(model, solve=eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
    ))
    program = eqiora.diff.compile(
        plan,
        inputs=(model.parameter("a"), model.parameter("b")),
        outputs=(model.observable("ratio"),),
    )
    # At (a,b)=(k*s,s), the ratio is k and its gradient is (1/s,-k/s).
    # Powers of two make every reference below exactly representable.
    for coefficient in (1.0, 2.0):
        point = program.evaluate((coefficient * scale, scale))
        np.testing.assert_array_equal(point.primal().output[0], coefficient)
        np.testing.assert_array_equal(point.jvp((scale, 0.0)).tangent[0], 1.0)
        np.testing.assert_array_equal(point.jvp((0.0, scale)).tangent[0], -coefficient)
        np.testing.assert_array_equal(
            point.vjp((scale,)).input_cotangent, (1.0, -coefficient)
        )
        np.testing.assert_array_equal(
            point.vjp((1.0,)).input_cotangent,
            (1.0 / scale, -coefficient / scale),
        )
