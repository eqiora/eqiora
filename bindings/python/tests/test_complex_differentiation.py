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
                                     outputs=(model.observable("output"),))
        for p in (3., 5.):
            point = program.evaluate((p,))
            np.testing.assert_allclose(point.accepted_unknowns.numpy(), [p, -p], rtol=0, atol=1e-10)
            np.testing.assert_allclose(point.primal().output[0], [2*p*p+p], rtol=0, atol=1e-9)
            np.testing.assert_allclose(point.jvp((2.,)).tangent[0],
                                       [2*(4*p+1)], rtol=0, atol=1e-9)
            reverse = point.vjp((1.,))
            np.testing.assert_allclose(reverse.input_cotangent, [4*p+1], rtol=0, atol=1e-9)
            assert reverse.evidence.primal_solve is not None
            assert reverse.evidence.initial_state_identity is None
        np.testing.assert_allclose(program.vjp((1.,)).input_cotangent,
                                   [13.], rtol=0, atol=1e-9)


def test_common_typed_parameter_coordinates_preserve_selected_order_and_complex_parts():
    import numpy as np

    class Producer:
        def __init__(self, values, device=(1, 0)):
            self.values, self.device = values, device
            self.queries, self.exports = 0, []

        def __dlpack_device__(self):
            self.queries += 1
            return self.device

        def __dlpack__(self, **kwargs):
            self.exports.append(kwargs)
            return self.values.__dlpack__(**kwargs)

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
                                     outputs=(model.observable("output"),))
        assert program.input_shape == (5,)
        assert program.input_ids == [model.parameter(name).id for name in names]
        for p, c, expected, gradient, directional in (
            (2., [1+2j, 3+4j], 122., (121., np.array([8+16j, 24+32j])), 201.),
            (3., [2-1j, -2+1j], 93., (61., np.array([36-18j, -36+18j])), 61.),
        ):
            values = (p, np.array(c, dtype=np.complex128))
            tangent = (1., np.full(2, 1+1j, dtype=np.complex128))
            if names[0] == "c":
                values = values[::-1]
                gradient = gradient[::-1]
                tangent = tangent[::-1]
            point = program.evaluate(values)
            np.testing.assert_allclose(point.primal().output[0], [expected], rtol=0, atol=1e-9)
            for actual, expected_leaf in zip(point.vjp((1.,)).input_cotangent, gradient, strict=True):
                np.testing.assert_allclose(actual, expected_leaf, rtol=0, atol=1e-9)
            np.testing.assert_allclose(point.jvp(tangent).tangent[0], directional, rtol=0, atol=1e-9)
        np.testing.assert_allclose(program.primal().output[0], [122.], rtol=0, atol=1e-9)

        def parameters(c):
            return (3., c) if names[0] == "p" else (c, 3.)

        array = np.array([2-1j, -2+1j], dtype=np.complex128)
        producer = Producer(array)
        accepted = program.evaluate(parameters(producer))
        array[:] = 99+101j
        assert producer.queries == 1 and len(producer.exports) == 1
        assert producer.exports[0]["copy"] is False
        assert producer.exports[0]["dl_device"] == (1, 0)
        np.testing.assert_allclose(accepted.primal().output[0], [93.], rtol=0, atol=1e-9)
        for rejected in (
            np.ones((1, 2), dtype=np.complex128),
            np.ones(2, dtype=np.float64),
            np.ones(2, dtype=np.complex64),
            np.array([1j, complex(0., np.inf)]),
            np.ones(4, dtype=np.complex128)[::2],
        ):
            try:
                program.evaluate(parameters(rejected))
            except (BufferError, eqiora.ValidationError):
                pass
            else:
                raise AssertionError("invalid complex leaf was admitted")
        foreign = Producer(np.ones(2, dtype=np.complex128), device=(2, 0))
        try:
            program.evaluate(parameters(foreign))
        except BufferError as error:
            assert "CPU device 0" in str(error)
        else:
            raise AssertionError("foreign device was admitted")
        assert foreign.queries == 1 and foreign.exports == []


def _direct_plan(model):
    return eqiora.resolve(model, solve=eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast, provider=eqiora.solve.SolverProvider.faer(),
    ))


def test_structured_outputs_keep_complex_fields_loss_and_margin_associated():
    import numpy as np

    model = eqiora.compile(source="""model M(){parameter p:1=3;
        parameter q:complex<1>=math.complex(2,-4);variable z:complex<1>;
        relation r{math.complex(1,2)*z+math.conj(z)=math.complex(4,2)*p+q;}
        observable loss:1=math.abs2(z)+p;observable margin:1=p-1;}""")
    plan = _direct_plan(model)
    outputs = (model.field("z"), model.observable("loss"), model.observable("margin"))
    program = eqiora.diff.compile(plan, inputs=(model.parameter("p"), model.parameter("q")), outputs=outputs)
    assert program.output_ids == [value.id for value in outputs]
    assert tuple(leaf.id for leaf in program.outputs) == tuple(program.output_ids)
    assert tuple(leaf.shape for leaf in program.outputs) == ((), (), ())
    assert tuple(leaf.value_type.scalar_domain for leaf in program.outputs) == ("complex", "real", "real")
    assert all(leaf.model_digest == program.model_digest and leaf.space is None for leaf in program.outputs)
    point = program.evaluate((3., 2-4j))
    assert tuple(leaf.shape for leaf in point.point) == ((), ())
    assert point.point[1].dtype == np.complex128
    # z=p+Im(q)/2 + i*(-p+(Im(q)-Re(q))/2), loss=|z|²+p.
    primal = point.primal()
    assert primal.evidence.output_ids == program.output_ids
    np.testing.assert_allclose(primal.output, (1-6j, 40., 2.), rtol=0, atol=1e-9)
    assert tuple(leaf.dtype for leaf in primal.output) == (np.complex128, np.float64, np.float64)
    for leaf in primal.output:
        assert not leaf.flags.writeable
        try:
            leaf.setflags(write=True)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted output became writable")
    forward = point.jvp((1., 2-1j))
    reverse = point.vjp((2-3j, .5, 2.))
    np.testing.assert_allclose(forward.tangent, (.5-2.5j, 32., 1.), rtol=0, atol=1e-9)
    np.testing.assert_allclose(reverse.input_cotangent, (14.5, 4.5-3j), rtol=0, atol=1e-9)
    for leaves in (forward.tangent, reverse.input_cotangent):
        for leaf in leaves:
            assert leaf.shape == ()
            assert not leaf.flags.writeable
            try:
                leaf.setflags(write=True)
            except ValueError:
                pass
            else:
                raise AssertionError("derivative leaf became writable")
    # The same real pairing holds with complex leaves, without unpacking coordinates.
    left = sum(np.vdot(c, t).real for c, t in zip((2-3j, .5, 2.), forward.tangent, strict=True))
    right = sum(np.vdot(c, t).real for c, t in zip(reverse.input_cotangent, (1., 2-1j), strict=True))
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-9)
    for occurrence in (forward, reverse):
        assert occurrence.evidence.output_ids == program.output_ids
        for actual, expected in zip(occurrence.output, primal.output):
            np.testing.assert_array_equal(actual, expected)
    reordered = eqiora.diff.compile(plan, inputs=(model.parameter("p"), model.parameter("q")), outputs=outputs[::-1])
    np.testing.assert_allclose(reordered.primal().output, primal.output[::-1], rtol=0, atol=1e-9)
    assert reordered.outputs[2] == program.outputs[0]
    by_signature = dict(zip(program.outputs, ("field", "loss", "margin")))
    assert tuple(by_signature[leaf] for leaf in reordered.outputs) == ("margin", "loss", "field")
    for rejected in ((), (outputs[0], outputs[0]), (model.parameter("p"),)):
        try:
            eqiora.diff.compile(plan, inputs=(model.parameter("p"),), outputs=rejected)
        except eqiora.ValidationError:
            pass
        else:
            raise AssertionError("invalid output selection was admitted")


def test_structured_outputs_preserve_intrinsic_shapes_and_distinct_units():
    import numpy as np

    model = eqiora.compile(source="""model M(){parameter p:m=3[m];parameter q:m=2[m];
        variable u:m;relation r{u=p+2*q;}observable loss:m^2=u*u+p*q;}""")
    program = eqiora.diff.compile(_direct_plan(model), inputs=(model.parameter("p"), model.parameter("q")),
                                 outputs=(model.field("u"), model.observable("loss")))
    assert tuple(leaf.shape for leaf in program.outputs) == ((), ())
    assert program.outputs[0].value_type.dimension != program.outputs[1].value_type.dimension
    np.testing.assert_allclose(program.primal().output, (7., 55.), rtol=0, atol=1e-9)
    shaped = eqiora.compile(source="""model M(){parameter c:array<complex<1>,2>=[math.complex(1,2),math.complex(3,-4)];
        variable z:array<complex<1>,2>;relation r{z=3*c;}}
    """)
    shaped_plan = eqiora.resolve(shaped, solve=eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Reproducible, provider=eqiora.solve.SolverProvider.reference(),
    ))
    shaped_program = eqiora.diff.compile(shaped_plan, inputs=(shaped.parameter("c"),), outputs=(shaped.field("z"),))
    output, = shaped_program.primal().output
    assert output.shape == shaped_program.outputs[0].shape == (2,)
    assert output.dtype == np.complex128
    np.testing.assert_allclose(output, [3+6j, 9-12j], rtol=0, atol=1e-9)

    direction = np.array([1-2j, -3+4j], dtype=np.complex128)
    covector = np.array([2+3j, 5-7j], dtype=np.complex128)
    for owner in (shaped_program, shaped_program.evaluate((np.array([1+2j, 3-4j]),))):
        forward, = owner.jvp((direction,)).tangent
        reverse, = owner.vjp((covector,)).input_cotangent
        assert forward.shape == reverse.shape == (2,)
        assert forward.dtype == reverse.dtype == np.complex128
        np.testing.assert_allclose(forward, [3-6j, -9+12j], rtol=0, atol=1e-9)
        np.testing.assert_allclose(reverse, [6+9j, 15-21j], rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.vdot(covector, forward).real, np.vdot(reverse, direction).real, rtol=0, atol=1e-9)
        for action in (owner.jvp, owner.vjp):
            for bad in ((), (np.ones((1, 2), dtype=np.complex128),), (np.ones(2),),
                        (np.array([1j, complex(0., np.inf)]),), (direction, direction)):
                try:
                    action(bad)
                except (BufferError, eqiora.ValidationError):
                    pass
                else:
                    raise AssertionError("invalid derivative leaf was admitted")
