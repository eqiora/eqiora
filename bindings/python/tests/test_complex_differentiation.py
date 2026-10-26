"""Real-paired complex sensitivities through the common Python Program."""
import eqiora


def _reference_linear():
    return eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Reproducible, provider=eqiora.solve.SolverProvider.reference(),
    )


def test_common_complex_sensitivity_uses_real_pairing_at_each_parameter_point():
    import numpy as np

    # Both equations imply z=p-ip, J=2p²+p, and dJ/dp=4p+1.
    for equation, controls in (
        ("a*z=math.complex(3,1)*p", _reference_linear()),
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
            assert reverse.evidence.steps[0].primal_solves[0] is not None
            assert reverse.evidence.steps[0].initial_state_identity is None
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
    plan = eqiora.resolve(model, solve=_reference_linear())
    # J=p² sum|c|²+p; grad=(2p sum|c|²+1, 2p² Re(c0), 2p² Im(c0), ...).
    for names in (("p", "c"), ("c", "p")):
        program = eqiora.diff.compile(plan, inputs=tuple(model.parameter(name) for name in names),
                                     outputs=(model.observable("output"),))
        assert [leaf.id for leaf in program.inputs] == [model.parameter(name).id for name in names]
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
    assert [leaf.id for leaf in program.outputs] == [value.id for value in outputs]
    assert tuple(leaf.shape for leaf in program.outputs) == ((), (), ())
    assert tuple(leaf.value_type.scalar_domain for leaf in program.outputs) == ("complex", "real", "real")
    assert all(leaf.model_digest == program.model_digest and leaf.space is None for leaf in program.outputs)
    point = program.evaluate((3., 2-4j))
    assert tuple(leaf.shape for leaf in point.point) == ((), ())
    assert point.point[1].dtype == np.complex128
    # z=p+Im(q)/2 + i*(-p+(Im(q)-Re(q))/2), loss=|z|²+p.
    primal = point.primal()
    assert primal.evidence.output_ids == [leaf.id for leaf in program.outputs]
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
        assert occurrence.evidence.output_ids == [leaf.id for leaf in program.outputs]
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
    shaped_plan = eqiora.resolve(shaped, solve=_reference_linear())
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


def structured_map_program():
    model = eqiora.compile(source="""model M(){parameter p:1=2;
        parameter c:array<complex<1>,2>=[math.complex(1,2),math.complex(3,-4)];
        variable z:array<complex<1>,2>;relation r{
            z[0]=p*c[0]+math.conj(c[0]);z[1]=p*c[1]+math.conj(c[1]);}
        observable loss:1=math.abs2(z[0])+math.abs2(z[1])+p;
        observable margin:1=p-1;}""")
    program = eqiora.diff.compile(eqiora.resolve(model, solve=_reference_linear()),
        inputs=(model.parameter("p"), model.parameter("c")),
        outputs=(model.field("z"), model.observable("loss"), model.observable("margin")))
    return model, program


def test_structured_maps_keep_shared_complex_leaves_outputs_and_real_pairing():
    import numpy as np
    import pytest

    model, program = structured_map_program()
    c = np.array([1+2j, 3-4j])
    p = np.array([[2., 3.], [1., 2.]])
    plan = program.map((p,), point_shape=(2, 2), shared_inputs=(model.parameter("c"),), shared=(c,))
    frozen = c.copy()
    c[:] = 99+101j
    complete = plan.execute()
    assert isinstance(complete, eqiora.CompleteEvaluationMap)
    assert tuple(leaf.shape for leaf in plan.points) == ((2, 2), (2, 2, 2))
    assert tuple(leaf.shape for leaf in plan[0]) == ((), (2,))
    assert tuple(leaf.shape for leaf in complete.primal()) == ((2, 2, 2), (2, 2), (2, 2))
    z = p[..., None]*frozen + frozen.conj()
    loss = np.sum(np.abs(z)**2, axis=-1) + p
    np.testing.assert_allclose(complete.primal()[0][0, 0], [3+2j, 9-4j], rtol=0, atol=1e-9)
    np.testing.assert_allclose(complete.primal()[1][0], [112., 243.], rtol=0, atol=1e-9)
    for actual, expected in zip(complete.primal(), (z, loss, p-1), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)

    # z=p*c+conj(c); dz=c*dp+p*dc+conj(dc). This is real-linear, not holomorphic.
    dp = np.arange(12, dtype=np.float64).reshape(2, 3, 2)/4 - 1
    dc = np.array([[1-2j, -3+4j], [2+1j, 1-1j], [-1+3j, 2+4j]])
    forward = complete.jvp((dp,), shared=(dc,), seed_shape=(3,), point_axes=(0, 2))
    assert forward.shape == (2, 3, 2)
    expected_dz = frozen*dp[..., None] + p[:, None, :, None]*dc[None, :, None, :] + dc.conj()[None, :, None, :]
    expected_loss = 2*np.sum((z[:, None, :, :].conj()*expected_dz).real, axis=-1)+dp
    for actual, expected in zip(forward.tangent, (expected_dz, expected_loss, dp), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)

    w = np.full((2, 3, 2, 2), 2-3j)
    lam = np.full((2, 3, 2), .5)
    mu = np.full((2, 3, 2), -2.)
    reverse = complete.vjp((w, lam, mu), seed_shape=(3,), point_axes=(0, 2))
    # Re(conj(w)*dz) gives q=w+2*lambda*z, pbar=Re(conj(q)*c)+lambda+mu,
    # cbar=p*q+conj(q). Shared cbar sums in requested point order for each seed.
    q = w + 2*lam[..., None]*z[:, None, :, :]
    expected_p = np.sum((q.conj()*frozen).real, axis=-1) + lam + mu
    expected_c = np.sum(p[:, None, :, None]*q + q.conj(), axis=(0, 2))
    np.testing.assert_allclose(reverse.mapped_cotangents[0], expected_p, rtol=0, atol=1e-9)
    np.testing.assert_allclose(reverse.shared_cotangents[0], expected_c, rtol=0, atol=1e-9)
    for i, seed, j in np.ndindex(2, 3, 2):
        point = complete.member(i*2+j)
        for actual, expected in zip(forward.member((i*3+seed)*2+j).tangent,
                                   point.jvp((dp[i, seed, j], dc[seed])).tangent, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
    left = sum(np.vdot(covector, tangent).real for covector, tangent in zip((w, lam, mu), forward.tangent, strict=True))
    right = np.vdot(reverse.mapped_cotangents[0], dp).real + np.vdot(reverse.shared_cotangents[0], dc).real
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-9)
    for leaf in (*plan.points, *complete.primal(), *forward.output, *forward.tangent,
                 *reverse.shared_cotangents, *reverse.mapped_cotangents):
        assert not leaf.flags.writeable
        with pytest.raises(ValueError):
            leaf.setflags(write=True)

    empty = program.map((np.empty((2, 0)),), point_shape=(2, 0),
                        shared_inputs=(model.parameter("c"),), shared=(frozen,)).execute()
    assert tuple(leaf.shape for leaf in empty.primal()) == ((2, 0, 2), (2, 0), (2, 0))
    assert tuple(leaf.dtype for leaf in empty.primal()) == (np.complex128, np.float64, np.float64)
    zero_seed = complete.jvp((np.empty((2, 0, 2)),), shared=(np.empty((0, 2), dtype=np.complex128),),
                             seed_shape=(0,), point_axes=(0, 2))
    assert tuple(leaf.shape for leaf in zero_seed.tangent) == ((2, 0, 2, 2), (2, 0, 2), (2, 0, 2))
    for bad in (np.ones(2), np.ones((1, 2), dtype=np.complex128), np.ones(4, dtype=np.complex128)[::2]):
        with pytest.raises(BufferError):
            program.map((p,), point_shape=(2, 2), shared_inputs=(model.parameter("c"),), shared=(bad,))
    # Intrinsic dimensions never stand in for explicitly declared point axes.
    with pytest.raises(BufferError, match="shape"):
        program.map((p,), point_shape=(4,), shared_inputs=(model.parameter("c"),), shared=(frozen,))
