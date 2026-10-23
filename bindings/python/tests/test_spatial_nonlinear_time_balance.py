"""Original weak/cell balances for nonconstant storage, independent of assembly."""

import eqiora as q
import numpy as np
import pytest
from numpy.polynomial import polynomial as poly


def _problem(spatial, complex_field, quadratic):
    c = 1 + 2j if complex_field else 1.0
    coefficient = "math.complex(1,2)" if complex_field else "1"
    scalar = "complex<1>" if complex_field else "1"
    t = "(time()/1[s])"
    amplitude = f"(1+{t}+{t}*{t})" if quadratic else f"(1+{t})"
    rate = f"(1+2*{t})" if quadratic else "1"
    phi = "(1+coordinate(0)/1[m])"
    # u*=c*a(t)*(1+x), so its Laplacian vanishes. The forcing below is
    # (1+|u*|²)*u*_t, with dimensions written before numerical SI evaluation.
    source = f"""public component Profile(
        support body:volume(ambient_dimension=1),
        support left:boundary(parent=body), support right:boundary(parent=body)) {{
        state u:{scalar} on body in smooth;
        relation flow on body {{
            1[s/m^2]*(1+math.abs2(u))*derivative(u)-div(grad(u))=
            (1+{int(abs(c) ** 2)}*{amplitude}*{amplitude}*{phi}*{phi})
                *{coefficient}*{rate}*{phi}/1[m^2];
        }}
        relation left_value on left {{trace(u)={coefficient}*{amplitude};}}
        relation right_value on right {{trace(u)=2*{coefficient}*{amplitude};}}
    }}"""
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
    model = q.compile(
        source=source,
        entry="Profile",
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
    field = model.field("definition.u")
    linear = q.solve.Linear(
        algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.reference(),
        relative_tolerance=1e-12,
        absolute_tolerance=1e-14,
        maximum_iterations=100,
    )
    plan = q.resolve(
        model,
        mesh=mesh,
        spatial=spatial(),
        solve=q.solve.Newton(
            linear=linear,
            relative_tolerance=0.0,
            absolute_tolerance=1e-12,
            maximum_iterations=20,
            maximum_line_search_steps=16,
        ),
        temporal=q.time.ImplicitMidpoint(
            step_s=1 / 16,
            relative_tolerance=1e-12,
            absolute_tolerances={
                (field, 0, 0, imaginary): 1e-13
                for imaginary in ([False, True] if complex_field else [False])
            },
        ),
    )
    return plan, field, c


def _weak_balance(state, rate, c, a, da):
    """Integrate polynomial coefficients against hats; no numerical quadrature."""
    dx = 0.5
    residual = np.zeros(3, dtype=state.dtype)
    storage = np.zeros_like(residual)
    for cell in range(2):
        u = [state[cell], state[cell + 1] - state[cell]]
        velocity = [rate[cell], rate[cell + 1] - rate[cell]]
        phi = [1 + cell * dx, dx]
        nonlinear_storage = poly.polymul(poly.polymul(u, np.conj(u)), velocity)
        forcing = c * da * poly.polyadd(phi, abs(c) ** 2 * a * a * poly.polypow(phi, 3))
        balance = poly.polysub(poly.polyadd(velocity, nonlinear_storage), forcing)
        for local, hat in enumerate(([1, -1], [0, 1])):
            # x=x_left+dx*xi; integral_0^1 xi^k dxi=1/(k+1).
            def integrate(coefficients):
                return dx * sum(v / (k + 1) for k, v in enumerate(coefficients))

            residual[cell + local] += integrate(poly.polymul(hat, balance))
            residual[cell + local] += (-1 if local == 0 else 1) * u[1] / dx
            storage[cell + local] += integrate(poly.polymul(hat, nonlinear_storage))
    return residual[1:-1], storage[1:-1]


def _cell_balance(state, rate, c, a, da):
    dx = 0.5
    phi = np.array([1.25, 1.75])
    # Exact cubic cell moment: mean(phi³)=phi(center)³+phi(center)*dx²/4.
    # Point-sampling this source would silently change the discrete equation.
    mean_cube = phi**3 + phi * dx**2 / 4
    forcing = c * da * dx * (phi + abs(c) ** 2 * a * a * mean_cube)
    internal_flux = (state[0] - state[1]) / dx
    flux = np.array([internal_flux, -internal_flux])
    flux[0] += 2 * (state[0] - c * a) / dx
    flux[1] += 2 * (state[1] - 2 * c * a) / dx
    storage = dx * np.abs(state) ** 2 * rate
    return dx * rate + storage + flux - forcing, storage


@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize(
    ("spatial", "quadratic"),
    [(q.fem.Q1, False), (q.fem.Q1, True), (q.fvm.CellCenteredTpfa, True)],
)
def test_midpoint_original_nonconstant_storage_balance(
    spatial, quadratic, complex_field
):
    plan, field, c = _problem(spatial, complex_field, quadratic)
    nodal = spatial is q.fem.Q1
    association = "vertex" if nodal else "cell"
    profile = np.array([1.0, 1.5, 2.0] if nodal else [1.25, 1.75])
    initial_values = c * profile
    initial = q.State.initial(
        plan,
        fields=(
            q.InitialField(
                field,
                **{("vertex_values" if nodal else "cell_values"): initial_values},
            ),
        ),
    )
    initial_bytes = initial.to_bytes()
    h = 1 / 16
    times = tuple(j * h for j in range(1, 5))
    result = q.run(plan, state=initial, until_s=times[-1], output_times_s=times)
    assert len(result.trajectory.states) == len(times)
    previous = initial_values
    for index, accepted in enumerate(result.trajectory.states):
        current = np.asarray(accepted.field(field).values(association))
        midpoint = (previous + current) / 2
        velocity = (current - previous) / h
        t = (index + 0.5) * h
        a, da = 1 + t + quadratic * t * t, 1 + 2 * quadratic * t
        if nodal:
            # Essential traces use g(t_mid), not averaged endpoint traces.
            # Their analytic rates equal the secant only for these degree<=2 data.
            midpoint[0], midpoint[-1] = c * a, 2 * c * a
            velocity[0], velocity[-1] = c * da, 2 * c * da
        balance = _weak_balance if nodal else _cell_balance
        residual, nonlinear_storage = balance(midpoint, velocity, c, a, da)
        max_u, max_rate = np.max(np.abs(midpoint)), np.max(np.abs(velocity))
        # Bound the stage Jacobian row sum independently: the test volume is
        # dx=1/2, and half the diffusion absolute row sum is at most 3/dx.
        # The nonlinear storage derivative contributes |u|*|u_dot|.
        jacobian_bound = 0.5 * ((1 + max_u**2) / h + max_u * max_rate) + 6
        coordinate_error = (
            2 * np.sqrt(len(current)) * (1e-13 + 1e-12 * np.max(np.abs(current)))
        )
        arithmetic = (
            256
            * np.finfo(float).eps
            * (
                1
                + (1 + max_u**2) * max_rate
                + 2 * max_u
                + abs(c * da) * (2 + 8 * abs(c) ** 2 * a * a)
            )
        )
        allowance = jacobian_bound * coordinate_error + arithmetic
        assert np.all(np.abs(residual) <= allowance)
        # A frozen unit capacity must fail this same oracle by a resolved margin.
        assert np.all(np.abs(residual - nonlinear_storage) > 100 * allowance)
        if nodal and not quadratic:
            # Affine time and space are exactly representable by Q1 midpoint.
            np.testing.assert_allclose(
                current, c * (1 + times[index]) * profile, rtol=0, atol=1e-10
            )
        previous = current
    assert initial.to_bytes() == initial_bytes
    assert q.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
