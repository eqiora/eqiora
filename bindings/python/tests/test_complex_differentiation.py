"""Real-paired complex sensitivities through the common Python Program."""
import eqiora


def test_common_complex_sensitivity_uses_real_pairing_at_each_parameter_point():
    import numpy as np

    # Both equations imply z=p-ip, J=2p²+p, and dJ/dp=4p+1.
    for equation, controls in (
        ("a*z=math.complex(3,1)*p", eqiora.solve.Linear(
            relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
            algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible, provider=eqiora.solve.SolverProvider.reference(),
        )),
        ("a*z+math.conj(z)=math.complex(4,2)*p", eqiora.solve.Linear(
            relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
            algorithm=eqiora.solve.LinearSolver.SparseLu,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Fast, provider=eqiora.solve.SolverProvider.faer(),
        )),
    ):
        model = eqiora.compile(source=f"""model M(){{parameter p:1=3;
            parameter a:complex<1>=math.complex(1,2);variable z:complex<1>;
            relation r{{{equation};}}observable output:1=math.abs2(z)+p;}}""")
        plan = eqiora.resolve(model, solve=controls)
        program = eqiora.diff.compile(plan, inputs=(model.parameter("p"),),
                                     output=model.observable("output"))
        for p in (3., 5.):
            point = program.evaluate(np.array([p], dtype=np.float64))
            np.testing.assert_allclose(point.accepted_unknowns.numpy(), [p, -p], rtol=0, atol=1e-10)
            np.testing.assert_allclose(point.primal().output.numpy(), [2*p*p+p], rtol=0, atol=1e-9)
            np.testing.assert_allclose(point.jvp(np.array([2.])).tangent.numpy(),
                                       [2*(4*p+1)], rtol=0, atol=1e-9)
            reverse = point.vjp(np.array([1.]))
            np.testing.assert_allclose(reverse.input_cotangent.numpy(), [4*p+1], rtol=0, atol=1e-9)
            assert reverse.evidence.primal_solve is not None
            assert reverse.evidence.initial_state_identity is None
        np.testing.assert_allclose(program.vjp(np.array([1.])).input_cotangent.numpy(),
                                   [13.], rtol=0, atol=1e-9)
