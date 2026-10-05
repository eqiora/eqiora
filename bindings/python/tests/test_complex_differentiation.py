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


def test_common_typed_parameter_coordinates_preserve_selected_order_and_complex_parts():
    import numpy as np

    model = eqiora.compile(source="""model M(){parameter p:1=2;
        parameter c:array<complex<1>,2>=[math.complex(1,2),math.complex(3,4)];
        variable z:array<complex<1>,2>;relation r{z=p*c;}
        observable output:1=math.abs2(z[0])+math.abs2(z[1])+p;}""")
    plan = eqiora.resolve(model, solve=eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Reproducible, provider=eqiora.solve.SolverProvider.reference(),
    ))
    # J=p² sum|c|²+p; grad=(2p sum|c|²+1, 2p² Re(c0), 2p² Im(c0), ...).
    for names in (("p", "c"), ("c", "p")):
        program = eqiora.diff.compile(plan, inputs=tuple(model.parameter(name) for name in names),
                                     output=model.observable("output"))
        assert program.input_shape == (5,)
        assert program.input_ids == [model.parameter(name).id for name in names]
        for values, expected, gradient in (
            ([2.,1.,2.,3.,4.], 122., [121.,8.,16.,24.,32.]),
            ([3.,2.,-1.,-2.,1.], 93., [61.,36.,-18.,-36.,18.]),
        ):
            if names[0] == "c":
                values = values[1:] + values[:1]
                gradient = gradient[1:] + gradient[:1]
            point = program.evaluate(np.array(values))
            np.testing.assert_allclose(point.primal().output.numpy(), [expected], rtol=0, atol=1e-9)
            np.testing.assert_allclose(point.vjp(np.array([1.])).input_cotangent.numpy(), gradient, rtol=0, atol=1e-9)
            np.testing.assert_allclose(point.jvp(np.ones(5)).tangent.numpy(), [sum(gradient)], rtol=0, atol=1e-9)
        np.testing.assert_allclose(program.primal().output.numpy(), [122.], rtol=0, atol=1e-9)
